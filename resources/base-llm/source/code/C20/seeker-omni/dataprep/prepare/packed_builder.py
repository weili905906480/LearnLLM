from __future__ import annotations

import json
import random
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from tokenizers import Tokenizer
from tqdm import tqdm

from .memmap_io import MemmapMeta


@dataclass(frozen=True)
class PackedTextBuildStats:
    num_in_lines: int
    num_out_samples: int
    num_out_tokens: int


def build_packed_text_memmap_dataset(
    *,
    text_path: str | Path,
    out_dir: str | Path,
    text_tokenizer: Tokenizer,
    max_seq_len: int,
    vocab_size: int,
    max_samples: int | None = None,
    overwrite: bool = False,
    compact: bool = False,
    resume: bool = False,
    repeat: bool = False,
    shuffle_buffer: int = 0,
    seed: int = 42,
    flush_every: int = 4096,
) -> PackedTextBuildStats:
    """从纯文本构建 packed 的 LM 预训练数据集。

    输入格式：每行一个文本样本。
    输出格式：memmap 目录，包含 input_ids/labels/attention_mask。

    打包策略：
    - 每个 block 以 <|im_start|> 开头
    - 每行会变成 token_ids + <|im_end|>
    - 将多行拼接为定长 block，必要时会切分超长行
    """

    src = Path(text_path)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    if out.exists() and any(out.iterdir()) and not bool(resume):
        if not overwrite:
            raise FileExistsError(f"refusing to overwrite non-empty dir: {out}")
        for p in out.glob("*"):
            if p.is_file():
                p.unlink()

    # 只支持 minimind2_chatml：<|endoftext|>/<|im_start|>/<|im_end|>。
    pad_id = text_tokenizer.token_to_id("<|endoftext|>")
    bos_id = text_tokenizer.token_to_id("<|im_start|>")
    eos_id = text_tokenizer.token_to_id("<|im_end|>")

    missing = [name for name, tid in (("pad", pad_id), ("bos", bos_id), ("eos", eos_id)) if tid is None]
    if missing:
        raise ValueError(f"tokenizer missing required chatml tokens: {missing}")

    pad_id = int(pad_id)
    bos_id = int(bos_id)
    eos_id = int(eos_id)

    if int(text_tokenizer.get_vocab_size()) != int(vocab_size):
        raise ValueError(f"tokenizer vocab mismatch: tok={text_tokenizer.get_vocab_size()} cfg={vocab_size}")

    max_seq_len = int(max_seq_len)
    if max_seq_len <= 2:
        raise ValueError("max_seq_len must be > 2")

    compact = bool(compact)
    input_dtype = np.int32
    label_dtype = np.int32
    attention_dtype = np.uint8
    if compact:
        if int(vocab_size) <= 65535:
            input_dtype = np.uint16
        if int(vocab_size) <= 32767:
            label_dtype = np.int16

    input_path = out / "input_ids.bin"
    labels_path = out / "labels.bin"
    attn_path = out / "attention_mask.bin"
    meta_path = out / "meta.json"

    num_samples = 0
    num_in_lines = 0
    num_tokens = 0

    def write_block(block_tokens: list[int], *, f_in, f_lab, f_att, pbar: tqdm | None) -> None:
        nonlocal num_samples, num_tokens
        arr = np.full((max_seq_len,), pad_id, dtype=input_dtype)
        take = min(len(block_tokens), max_seq_len)
        if take > 0:
            arr[:take] = np.asarray(block_tokens[:take], dtype=input_dtype)

        att = np.zeros((max_seq_len,), dtype=attention_dtype)
        if take > 0:
            att[:take] = 1

        if label_dtype == np.int32:
            lab = arr.astype(np.int32, copy=True)
        else:
            lab = arr.astype(label_dtype, copy=True)
        lab[arr == pad_id] = -100

        arr.tofile(f_in)
        lab.tofile(f_lab)
        att.tofile(f_att)

        num_samples += 1
        num_tokens += int(take)
        if pbar is not None:
            pbar.update(1)

    buf: list[int] = [bos_id]

    def save_meta(samples: int) -> None:
        MemmapMeta(
            num_samples=int(samples),
            seq_len=int(max_seq_len),
            vocab_size=int(vocab_size),
            pad_id=int(pad_id),
            input_dtype=str(np.dtype(input_dtype).name),
            label_dtype=str(np.dtype(label_dtype).name),
            attention_dtype=str(np.dtype(attention_dtype).name),
        ).save(meta_path)

    kept0 = 0
    if bool(resume):
        if not meta_path.exists():
            save_meta(0)

        meta0 = MemmapMeta.load(meta_path)
        if int(meta0.seq_len) != int(max_seq_len) or int(meta0.vocab_size) != int(vocab_size):
            raise ValueError(f"resume meta mismatch: {meta0} (expected seq_len={max_seq_len}, vocab={vocab_size})")
        if int(meta0.pad_id) != int(pad_id):
            raise ValueError(f"resume pad_id mismatch: meta={meta0.pad_id} expected={pad_id}")
        if str(meta0.input_dtype) != str(np.dtype(input_dtype).name) or str(meta0.label_dtype) != str(np.dtype(label_dtype).name):
            raise ValueError(
                f"resume dtype mismatch: meta(input={meta0.input_dtype}, label={meta0.label_dtype}) "
                f"expected(input={np.dtype(input_dtype).name}, label={np.dtype(label_dtype).name})"
            )
        if str(meta0.attention_dtype) != str(np.dtype(attention_dtype).name):
            raise ValueError(f"resume attn dtype mismatch: meta={meta0.attention_dtype} expected={np.dtype(attention_dtype).name}")

        in_bytes = input_path.stat().st_size if input_path.exists() else 0
        lab_bytes = labels_path.stat().st_size if labels_path.exists() else 0
        att_bytes = attn_path.stat().st_size if attn_path.exists() else 0

        in_per = int(max_seq_len) * int(np.dtype(input_dtype).itemsize)
        lab_per = int(max_seq_len) * int(np.dtype(label_dtype).itemsize)
        att_per = int(max_seq_len) * int(np.dtype(attention_dtype).itemsize)

        if in_bytes % in_per != 0 or lab_bytes % lab_per != 0 or att_bytes % att_per != 0:
            raise ValueError("resume: existing .bin file sizes are not aligned to seq_len")

        kept0 = int(in_bytes // in_per)
        if int(lab_bytes // lab_per) != int(kept0) or int(att_bytes // att_per) != int(kept0):
            raise ValueError("resume: input/labels/attn sample counts mismatch")

        num_samples = int(kept0)
        num_tokens = int(kept0) * int(max_seq_len)
        save_meta(num_samples)

    file_mode = "ab" if bool(resume) and kept0 > 0 else "wb"

    rng = random.Random(int(seed))
    shuffle_buffer = int(shuffle_buffer)

    def iter_lines_once():
        with src.open("r", encoding="utf-8") as f_txt:
            if shuffle_buffer <= 1:
                yield from f_txt
                return

            buf_lines: list[str] = []
            for line in f_txt:
                buf_lines.append(line)
                if len(buf_lines) >= shuffle_buffer:
                    j = rng.randrange(len(buf_lines))
                    yield buf_lines.pop(j)
            while buf_lines:
                j = rng.randrange(len(buf_lines))
                yield buf_lines.pop(j)

    with (
        input_path.open(file_mode) as f_in,
        labels_path.open(file_mode) as f_lab,
        attn_path.open(file_mode) as f_att,
    ):
        pbar = tqdm(
            total=(int(max_samples) if max_samples is not None else None),
            initial=int(num_samples),
            desc=f"pack:text->{out.name}",
            unit="block",
        )

        while True:
            for line in iter_lines_once():
                if max_samples is not None and int(num_samples) >= int(max_samples):
                    break

                line = str(line).strip()
                if not line:
                    continue

                num_in_lines += 1

                ids = text_tokenizer.encode(line).ids
                if not ids:
                    continue

                ids.append(eos_id)

                pos = 0
                while pos < len(ids):
                    if max_samples is not None and int(num_samples) >= int(max_samples):
                        break

                    if len(buf) >= max_seq_len:
                        write_block(buf[:max_seq_len], f_in=f_in, f_lab=f_lab, f_att=f_att, pbar=pbar)
                        buf = [bos_id]
                        continue

                    remain = max_seq_len - len(buf)
                    take = min(remain, len(ids) - pos)
                    if take <= 0:
                        write_block(buf[:max_seq_len], f_in=f_in, f_lab=f_lab, f_att=f_att, pbar=pbar)
                        buf = [bos_id]
                        continue

                    buf.extend(ids[pos : pos + take])
                    pos += int(take)

                    if len(buf) == max_seq_len:
                        write_block(buf, f_in=f_in, f_lab=f_lab, f_att=f_att, pbar=pbar)
                        buf = [bos_id]

                    if int(flush_every) > 0 and (int(num_samples) % int(flush_every) == 0):
                        f_in.flush()
                        f_lab.flush()
                        f_att.flush()
                        save_meta(num_samples)

            if max_samples is not None and int(num_samples) >= int(max_samples):
                break
            if not bool(repeat) or max_samples is None:
                break

        pbar.close()

        if max_samples is None or int(num_samples) < int(max_samples):
            if len(buf) > 1:
                write_block(buf, f_in=f_in, f_lab=f_lab, f_att=f_att, pbar=None)

    save_meta(num_samples)

    stats = {
        "text_path": str(src).replace("\\", "/"),
        "out_dir": str(out).replace("\\", "/"),
        "num_in_lines": int(num_in_lines),
        "num_out_samples": int(num_samples),
        "num_out_tokens": int(num_tokens),
        "max_seq_len": int(max_seq_len),
        "created_at_unix": int(time.time()),
        "input_dtype": str(np.dtype(input_dtype).name),
        "label_dtype": str(np.dtype(label_dtype).name),
        "attention_dtype": str(np.dtype(attention_dtype).name),
    }
    (out / "stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")

    return PackedTextBuildStats(
        num_in_lines=int(num_in_lines),
        num_out_samples=int(num_samples),
        num_out_tokens=int(num_tokens),
    )
