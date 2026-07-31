import json
import random
import shutil
import time
from pathlib import Path

from tokenizers import Tokenizer
from .text_bpe import train_text_bpe
from ..data_paths import DATA_INTERIM, MINIMIND_TEXT_CORPUS, TOKENIZER_DIR, TOKENIZER_VOCAB_SIZE


MINIMIND2_CHATML_TOKENS = [
    "<|endoftext|>",
    "<|im_start|>",
    "<|im_end|>",
    "<img_bos>",
    "<img>",
    "<img_eos>",
]


def _count_nonempty_lines(path: Path) -> int:
    n = 0
    with path.open("r", encoding="utf-8-sig") as f:
        for line in f:
            if line.strip():
                n += 1
    return n


def _sample_text_corpus(*, src: Path, dst: Path, ratio: float, seed: int, overwrite: bool) -> Path:
    ratio = float(ratio)
    if ratio >= 1.0:
        return src
    if ratio <= 0.0:
        raise ValueError(f"tokenizer.sample_ratio must be in (0,1], got {ratio}")

    if dst.exists() and not bool(overwrite) and int(dst.stat().st_size) > 0:
        return dst

    dst.parent.mkdir(parents=True, exist_ok=True)
    rng = random.Random(int(seed))

    kept_lines = 0
    total_lines = 0
    written = 0

    with src.open("rb") as r, dst.open("wb") as w:
        for line in r:
            total_lines += 1
            if rng.random() < ratio:
                w.write(line)
                written += len(line)
                kept_lines += 1

    if kept_lines <= 0 or written <= 0:
        raise RuntimeError(f"sampled text_corpus is empty: src={src} ratio={ratio} seed={seed}")

    print(
        f"ok: sample text_corpus -> {dst} "
        f"(ratio={ratio}, seed={seed}, kept_lines={kept_lines}/{total_lines}, mb={written/1024/1024:.1f})"
    )
    return dst


def _ensure_tokenizer(*, text_corpus: Path, out_dir: Path, vocab_size: int, overwrite: bool) -> None:
    meta_path = out_dir / "meta.json"
    if overwrite and out_dir.exists():
        shutil.rmtree(out_dir)

    if meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if meta.get("special_tokens_scheme") == "minimind2_chatml" and int(meta.get("vocab_size", -1)) == int(vocab_size):
            return

    out_dir.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    train_text_bpe(
        input_path=text_corpus,
        out_dir=out_dir,
        vocab_size=int(vocab_size),
        special_tokens=list(MINIMIND2_CHATML_TOKENS),
    )
    dt = int(time.time() - t0)
    print(f"ok: tokenizer -> {out_dir} (sec={dt})")


def ensure_tokenizer_and_load(cfg: dict) -> Tokenizer:
    seed = int(cfg.get("seed", 42))
    overwrite = bool(cfg.get("overwrite", False))

    tok_cfg = cfg.get("tokenizer", {})

    vocab_size = int(TOKENIZER_VOCAB_SIZE)
    tokenizer_out = Path(TOKENIZER_DIR)

    text_corpus = Path(MINIMIND_TEXT_CORPUS)
    if not text_corpus.exists():
        raise FileNotFoundError(text_corpus)

    tmp_root = Path(DATA_INTERIM) / "tmp"

    sample_ratio = float(tok_cfg.get("sample_ratio", 1.0))
    sample_seed = int(tok_cfg.get("sample_seed", seed))

    tokenizer_corpus_for_train = text_corpus
    if sample_ratio < 1.0:
        safe_name = text_corpus.name.replace(".", "_")
        ratio_str = f"{sample_ratio:.6f}".rstrip("0").rstrip(".")
        sampled = tmp_root / f"{safe_name}.sample_p{ratio_str}_seed{sample_seed}.txt"
        tokenizer_corpus_for_train = _sample_text_corpus(
            src=text_corpus,
            dst=sampled,
            ratio=sample_ratio,
            seed=sample_seed,
            overwrite=overwrite,
        )

    print(f"using tokenizer corpus: {tokenizer_corpus_for_train} (lines={_count_nonempty_lines(tokenizer_corpus_for_train)})")

    print("== prepare: tokenizer ==")
    _ensure_tokenizer(
        text_corpus=tokenizer_corpus_for_train,
        out_dir=tokenizer_out,
        vocab_size=vocab_size,
        overwrite=overwrite,
    )

    # 节省磁盘：采样得到的语料是临时文件。
    if tokenizer_corpus_for_train != text_corpus and tokenizer_corpus_for_train.exists():
        tokenizer_corpus_for_train.unlink(missing_ok=True)

    tok = Tokenizer.from_file(str(tokenizer_out / "tokenizer.json"))
    if int(tok.get_vocab_size()) != int(vocab_size):
        raise RuntimeError(f"tokenizer vocab mismatch: tok={tok.get_vocab_size()} expected={vocab_size}")
    return tok
