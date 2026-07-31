from pathlib import Path

import json
import numpy as np
from tokenizers import Tokenizer
from tqdm import tqdm

from .memmap_io import MemmapWriter


def _iter_json_objects(path: str | Path):
    with Path(path).open("r", encoding="utf-8-sig") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"invalid json at line {line_no}: {path}") from e


def _count_jsonl_lines(path: str | Path) -> int:
    n = 0
    with Path(path).open('r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                n += 1
    return n


def _detect_resume_start(out_dir: Path, *, num_samples: int, seq_len: int, rewind: int) -> int:
    meta_path = out_dir / 'meta.json'
    att_path = out_dir / 'attention_mask.bin'
    if not meta_path.exists() or not att_path.exists():
        return 0

    att = np.memmap(att_path, dtype=np.uint8, mode='r', shape=(int(num_samples), int(seq_len)))
    first_col = np.array(att[:, 0])
    zeros = first_col == 0
    if not bool(zeros.any()):
        return int(num_samples)

    first_zero = int(np.argmax(zeros))
    return max(0, int(first_zero) - int(rewind))


def build_sft_text_memmap_dataset(
    *,
    jsonl_path: str | Path,
    out_dir: str | Path,
    text_tokenizer: Tokenizer,
    max_seq_len: int,
    vocab_size: int,
    resume: bool = False,
    resume_rewind: int = 512,
    flush_every: int = 200,
) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    pad_id = text_tokenizer.token_to_id("<pad>")
    if pad_id is None:
        pad_id = text_tokenizer.token_to_id("<|endoftext|>")

    required = {"pad": pad_id}

    bos_id = text_tokenizer.token_to_id("<bos>")
    if bos_id is None:
        bos_id = text_tokenizer.token_to_id("<|im_start|>")

    eos_id = text_tokenizer.token_to_id("<eos>")
    if eos_id is None:
        eos_id = text_tokenizer.token_to_id("<|im_end|>")

    sys_id = text_tokenizer.token_to_id("<sys>")
    usr_id = text_tokenizer.token_to_id("<usr>")
    asst_id = text_tokenizer.token_to_id("<asst>")
    sep_id = text_tokenizer.token_to_id("<sep>")

    use_chatml = sys_id is None or usr_id is None or asst_id is None or sep_id is None

    required.update({"bos": bos_id, "eos": eos_id})
    if not bool(use_chatml):
        required.update({"<sys>": sys_id, "<usr>": usr_id, "<asst>": asst_id, "<sep>": sep_id})

    missing = [k for k, v in required.items() if v is None]
    if missing:
        raise ValueError(f'missing required tokens in tokenizer: {missing}')

    pad_id = int(pad_id)
    bos_id = int(bos_id)
    eos_id = int(eos_id)
    if not bool(use_chatml):
        sys_id = int(sys_id)  # type: ignore[arg-type]
        usr_id = int(usr_id)  # type: ignore[arg-type]
        asst_id = int(asst_id)  # type: ignore[arg-type]
        sep_id = int(sep_id)  # type: ignore[arg-type]

    if text_tokenizer.get_vocab_size() != int(vocab_size):
        raise ValueError(f'tokenizer vocab mismatch: tok={text_tokenizer.get_vocab_size()} cfg={vocab_size}')

    n = _count_jsonl_lines(jsonl_path)

    start_idx = 0
    mode = 'w+'
    if bool(resume) and (out / 'meta.json').exists():
        start_idx = _detect_resume_start(out, num_samples=int(n), seq_len=int(max_seq_len), rewind=int(resume_rewind))
        mode = 'r+'
        if int(start_idx) >= int(n):
            print(f'memmap already complete: {out} ({n} samples)')
            return

    writer = MemmapWriter(
        out,
        num_samples=int(n),
        seq_len=int(max_seq_len),
        vocab_size=int(vocab_size),
        pad_id=pad_id,
        mode=mode,
    )

    progress_path = out / 'progress.txt'
    if int(start_idx) > 0:
        progress_path.write_text(str(int(start_idx)), encoding='utf-8')

    pbar = tqdm(total=int(n), initial=int(start_idx), desc=f'sft-memmap:{out.name}', unit='sample')

    try:
        for idx, obj in enumerate(_iter_json_objects(jsonl_path)):
            if idx < int(start_idx):
                continue

            conversations = obj.get("conversations")
            if conversations is not None:
                if not bool(use_chatml):
                    raise ValueError("conversation-style SFT requires minimind2_chatml token scheme")
                if not isinstance(conversations, list) or not conversations:
                    raise ValueError("invalid conversations in SFT jsonl (expected non-empty list)")

                def _enc(x: str) -> list[int]:
                    return text_tokenizer.encode(str(x)).ids

                nl = _enc("\n")

                msgs: list[dict[str, str]] = []
                for m in conversations:
                    if not isinstance(m, dict):
                        continue
                    role = str(m.get("role") or "").strip().lower()
                    content = str(m.get("content") or "")
                    if role not in ("system", "user", "assistant"):
                        continue
                    if role != "system" and not content.strip():
                        continue
                    msgs.append({"role": role, "content": content})

                if not msgs:
                    raise ValueError("no usable messages in conversations")

                if msgs[0]["role"] != "system":
                    msgs.insert(0, {"role": "system", "content": "You are a helpful assistant"})

                tokens: list[int] = []
                labels_list: list[int] = []
                for m in msgs:
                    role = m["role"]
                    content = m["content"]

                    header = [bos_id] + _enc(f"{role}\n")
                    body = _enc(content)
                    footer = [eos_id] + nl

                    msg_tokens = header + body + footer
                    msg_labels = [-100] * len(msg_tokens)
                    if role == "assistant":
                        for j in range(len(header), len(msg_tokens)):
                            msg_labels[j] = msg_tokens[j]

                    tokens.extend(msg_tokens)
                    labels_list.extend(msg_labels)

                if len(tokens) > int(max_seq_len):
                    tokens = tokens[: int(max_seq_len)]
                    labels_list = labels_list[: int(max_seq_len)]

                input_ids = np.full((int(max_seq_len),), pad_id, dtype=np.int32)
                input_ids[: len(tokens)] = np.asarray(tokens, dtype=np.int32)

                attention_mask = np.zeros((int(max_seq_len),), dtype=np.uint8)
                attention_mask[: len(tokens)] = 1

                labels = np.full((int(max_seq_len),), -100, dtype=np.int32)
                labels[: len(labels_list)] = np.asarray(labels_list, dtype=np.int32)
                labels[input_ids == pad_id] = -100
            else:
                system = obj.get("system")
                prompt = str(obj.get("prompt", ""))
                answer = str(obj.get("answer", ""))
                if not prompt or not answer:
                    raise ValueError("invalid legacy SFT row (expected prompt/answer)")

                if not bool(use_chatml):
                    tokens = [bos_id]

                    if system:
                        tokens += [sys_id]  # type: ignore[list-item]
                        tokens += text_tokenizer.encode(str(system)).ids
                        tokens += [sep_id]  # type: ignore[list-item]

                    usr_content = text_tokenizer.encode(prompt).ids
                    ans_content = text_tokenizer.encode(answer).ids

                    # 为了适配 max_seq_len，尽量截断 prompt/system，同时保留 answer。
                    sys_tokens_len = len(tokens)
                    fixed_extra = 4  # <usr>, <sep>, <asst>, <eos>
                    avail_usr = int(max_seq_len) - (int(sys_tokens_len) + int(len(ans_content)) + int(fixed_extra))

                    if avail_usr < 0:
                        max_ans = max(1, int(max_seq_len) - (int(sys_tokens_len) + int(fixed_extra)))
                        ans_content = ans_content[: int(max_ans)]
                        avail_usr = int(max_seq_len) - (int(sys_tokens_len) + int(len(ans_content)) + int(fixed_extra))

                    if avail_usr < 0:
                        # system 段过长：退化为丢弃 system。
                        tokens = [bos_id]
                        sys_tokens_len = 1
                        max_ans = max(1, int(max_seq_len) - (int(sys_tokens_len) + int(fixed_extra)))
                        ans_content = ans_content[: int(max_ans)]
                        avail_usr = int(max_seq_len) - (int(sys_tokens_len) + int(len(ans_content)) + int(fixed_extra))

                    if int(avail_usr) <= 0:
                        usr_content = []
                    elif len(usr_content) > int(avail_usr):
                        usr_content = usr_content[-int(avail_usr) :]

                    tokens += [usr_id]  # type: ignore[list-item]
                    tokens += usr_content
                    tokens += [sep_id]  # type: ignore[list-item]

                    tokens += [asst_id]  # type: ignore[list-item]
                    answer_start = len(tokens)
                    tokens += ans_content
                    tokens += [eos_id]
                else:
                    def _enc(x: str) -> list[int]:
                        return text_tokenizer.encode(x).ids

                    nl = _enc("\n")
                    system_text = str(system) if system else "You are a helpful assistant"

                    sys_tokens = [bos_id] + _enc("system\n") + _enc(system_text) + [eos_id] + nl
                    user_header = [bos_id] + _enc("user\n")
                    user_footer = [eos_id] + nl
                    asst_header = [bos_id] + _enc("assistant\n")
                    asst_footer = [eos_id] + nl

                    usr_content = _enc(prompt)
                    ans_content = _enc(answer)

                    fixed = len(sys_tokens) + len(user_header) + len(user_footer) + len(asst_header) + len(asst_footer)
                    avail_usr = int(max_seq_len) - (int(fixed) + int(len(ans_content)))

                    if avail_usr < 0:
                        max_ans = max(1, int(max_seq_len) - int(fixed))
                        ans_content = ans_content[: int(max_ans)]
                        avail_usr = int(max_seq_len) - (int(fixed) + int(len(ans_content)))

                    if avail_usr < 0:
                        # system 段过长：退化为丢弃 system。
                        sys_tokens = []
                        fixed = len(sys_tokens) + len(user_header) + len(user_footer) + len(asst_header) + len(asst_footer)
                        max_ans = max(1, int(max_seq_len) - int(fixed))
                        ans_content = ans_content[: int(max_ans)]
                        avail_usr = int(max_seq_len) - (int(fixed) + int(len(ans_content)))

                    if int(avail_usr) <= 0:
                        usr_content = []
                    elif len(usr_content) > int(avail_usr):
                        usr_content = usr_content[-int(avail_usr) :]

                    tokens = sys_tokens + user_header + usr_content + user_footer + asst_header
                    answer_start = len(tokens)
                    tokens += ans_content + asst_footer

                if len(tokens) > int(max_seq_len):
                    tokens = tokens[: int(max_seq_len)]

                input_ids = np.full((int(max_seq_len),), pad_id, dtype=np.int32)
                input_ids[: len(tokens)] = np.asarray(tokens, dtype=np.int32)

                attention_mask = np.zeros((int(max_seq_len),), dtype=np.uint8)
                attention_mask[: len(tokens)] = 1

                labels = np.full((int(max_seq_len),), -100, dtype=np.int32)
                ans_start = min(int(answer_start), int(len(tokens)))
                labels[ans_start : len(tokens)] = input_ids[ans_start : len(tokens)]
                labels[input_ids == pad_id] = -100

            writer.write(idx, input_ids=input_ids, labels=labels, attention_mask=attention_mask)

            pbar.update(1)

            if int(flush_every) > 0 and ((idx + 1) % int(flush_every) == 0):
                writer.flush()
                progress_path.write_text(str(int(idx) + 1), encoding='utf-8')

        writer.flush()
        progress_path.write_text(str(int(n)), encoding='utf-8')
    finally:
        pbar.close()
