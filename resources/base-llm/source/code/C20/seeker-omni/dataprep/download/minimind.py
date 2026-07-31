import json
import random
import sys
import time
from pathlib import Path

from tqdm import tqdm

from ..data_paths import MINIMIND_DIR, MINIMIND_PRETRAIN_JSONL, MINIMIND_SFT_JSONL, MINIMIND_SFT_SEEKER, MINIMIND_TEXT_CORPUS
from .cleaning import keep_text, normalize_text
from .fetch import download


# 固定的清洗与构造规则（教学默认值，避免引入过多可调参数）。
_CLEAN_MIN_CHARS = 20
_CLEAN_MAX_CHARS = 512
_CLEAN_MIN_ZH_RATIO = 0.55
_CLEAN_MAX_LATIN_RATIO = 0.35
_CLEAN_MAX_SYMBOL_RATIO = 0.40

_SFT_SYSTEM_PROMPT_RATIO = 0.2
_SFT_EMPTY_THINK_RATIO = 0.05


def _make_text_corpus_from_minimind_pretrain(src_jsonl: Path, dst_txt: Path) -> None:
    dst_txt.parent.mkdir(parents=True, exist_ok=True)
    out_n = 0
    t0 = time.time()
    tmp = dst_txt.with_suffix(dst_txt.suffix + ".part")
    done = dst_txt.with_suffix(dst_txt.suffix + ".done")
    if tmp.exists():
        tmp.unlink()

    with src_jsonl.open("r", encoding="utf-8-sig") as r, tmp.open("w", encoding="utf-8", newline="\n") as w:
        for line in tqdm(r, desc="minimind: build tokenizer corpus", unit="lines", mininterval=2.0, file=sys.stdout):
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            text = normalize_text(str(obj.get("text", "")))
            if not keep_text(
                text,
                min_chars=_CLEAN_MIN_CHARS,
                max_chars=_CLEAN_MAX_CHARS,
                min_zh_ratio=_CLEAN_MIN_ZH_RATIO,
                max_latin_ratio=_CLEAN_MAX_LATIN_RATIO,
                max_symbol_ratio=_CLEAN_MAX_SYMBOL_RATIO,
            ):
                continue
            w.write(text + "\n")
            out_n += 1
    if out_n <= 0:
        raise RuntimeError(f"no lines written: {dst_txt}")
    tmp.replace(dst_txt)
    done.write_text("ok\n", encoding="utf-8")
    print(f"ok: minimind corpus -> {dst_txt} (lines={out_n}, sec={int(time.time()-t0)})")


def _convert_minimind_sft_to_seeker(src_jsonl: Path, dst_jsonl: Path, *, seed: int) -> None:
    dst_jsonl.parent.mkdir(parents=True, exist_ok=True)
    rng = random.Random(int(seed))
    t0 = time.time()
    tmp = dst_jsonl.with_suffix(dst_jsonl.suffix + ".part")
    done = dst_jsonl.with_suffix(dst_jsonl.suffix + ".done")
    if tmp.exists():
        tmp.unlink()

    _SYSTEM_PROMPTS = [
        "你是一个知识丰富的AI，尽力为用户提供准确的信息。",
        "你是minimind，一个小巧但有用的语言模型。",
        "你是一个专业的AI助手，请提供有价值的回答。",
        "你是minimind，请尽力帮助用户解决问题。",
        "你是一个可靠的AI，请给出准确的回答。",
        "You are a helpful AI assistant.",
        "You are minimind, a lightweight intelligent assistant.",
        "You are a friendly chatbot. Please answer the user's questions carefully.",
        "You are a knowledgeable AI. Try your best to provide accurate information.",
        "You are minimind, a small but useful language model.",
    ]

    def _maybe_strip_empty_think(s: str) -> str:
        s = str(s or "")
        if "<think>\n\n</think>" not in s:
            return s
        if rng.random() <= float(_SFT_EMPTY_THINK_RATIO):
            return s
        s = s.replace("<think>\n\n</think>\n\n", "")
        s = s.replace("<think>\n\n</think>\n", "")
        s = s.replace("<think>\n\n</think>", "")
        return s

    def _norm(s: str) -> str:
        return normalize_text(_maybe_strip_empty_think(str(s or "")))

    out_n = 0
    skip_n = 0
    with src_jsonl.open("r", encoding="utf-8-sig") as r, tmp.open("w", encoding="utf-8", newline="\n") as w:
        for line_no, line in enumerate(
            tqdm(r, desc="minimind: convert sft", unit="lines", mininterval=2.0, file=sys.stdout),
            start=1,
        ):
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            conv = obj.get("conversations")
            if not isinstance(conv, list) or not conv:
                skip_n += 1
                continue

            messages: list[dict[str, str]] = []

            first = conv[0] if isinstance(conv[0], dict) else None
            if first is not None and str(first.get("role") or "").strip().lower() == "system":
                content = _norm(first.get("content", ""))
                if content:
                    messages.append({"role": "system", "content": content})
                body = conv[1:]
            else:
                body = conv
                if rng.random() < float(_SFT_SYSTEM_PROMPT_RATIO):
                    messages.append({"role": "system", "content": str(rng.choice(_SYSTEM_PROMPTS))})
                else:
                    messages.append({"role": "system", "content": "You are a helpful assistant"})

            for m in body:
                if not isinstance(m, dict):
                    continue
                role = str(m.get("role") or "").strip().lower()
                if role not in ("user", "assistant"):
                    continue
                content = _norm(m.get("content", ""))
                if not content:
                    continue
                messages.append({"role": role, "content": content})

            if len(messages) < 3:
                skip_n += 1
                continue

            joined = " ".join([m["content"] for m in messages if m["role"] != "system"]).strip()
            if not keep_text(
                normalize_text(joined),
                min_chars=_CLEAN_MIN_CHARS,
                max_chars=_CLEAN_MAX_CHARS,
                min_zh_ratio=_CLEAN_MIN_ZH_RATIO,
                max_latin_ratio=_CLEAN_MAX_LATIN_RATIO,
                max_symbol_ratio=_CLEAN_MAX_SYMBOL_RATIO,
            ):
                skip_n += 1
                continue

            ex_id = str(obj.get("id", f"line-{line_no}"))
            row = {"id": ex_id, "conversations": messages}
            w.write(json.dumps(row, ensure_ascii=False) + "\n")
            out_n += 1

    if out_n <= 0:
        raise RuntimeError(f"no sft samples written: {dst_jsonl} (skipped={skip_n})")
    tmp.replace(dst_jsonl)
    done.write_text("ok\n", encoding="utf-8")
    print(f"ok: minimind sft -> {dst_jsonl} (rows={out_n}, skipped={skip_n}, sec={int(time.time()-t0)})")


def ensure_minimind(cfg: dict, *, seed: int) -> None:
    mm = cfg["minimind"]
    pretrain_url = str(mm["pretrain_url"])
    sft_url = str(mm["sft_url"])
    do_download = bool(mm["download"])
    overwrite_download = bool(mm["overwrite_download"])

    MINIMIND_DIR.mkdir(parents=True, exist_ok=True)
    pretrain_jsonl = MINIMIND_PRETRAIN_JSONL
    sft_jsonl = MINIMIND_SFT_JSONL
    text_corpus = MINIMIND_TEXT_CORPUS
    sft_seeker = MINIMIND_SFT_SEEKER

    if do_download:
        download(pretrain_url, pretrain_jsonl, overwrite=overwrite_download)
        download(sft_url, sft_jsonl, overwrite=overwrite_download)

    if not pretrain_jsonl.exists():
        raise FileNotFoundError(pretrain_jsonl)
    if not sft_jsonl.exists():
        raise FileNotFoundError(sft_jsonl)

    text_corpus_done = text_corpus.with_suffix(text_corpus.suffix + ".done")
    if text_corpus.exists() and text_corpus_done.exists() and text_corpus.stat().st_size > 0:
        print(f"skip: minimind corpus exists -> {text_corpus}")
    else:
        _make_text_corpus_from_minimind_pretrain(pretrain_jsonl, text_corpus)

    sft_seeker_done = sft_seeker.with_suffix(sft_seeker.suffix + ".done")
    if sft_seeker.exists() and sft_seeker_done.exists() and sft_seeker.stat().st_size > 0:
        print(f"skip: minimind sft exists -> {sft_seeker}")
    else:
        _convert_minimind_sft_to_seeker(sft_jsonl, sft_seeker, seed=seed)
