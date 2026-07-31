import shutil
import time
from pathlib import Path

from tokenizers import Tokenizer

from .memmap_io import MemmapMeta
from .packed_builder import build_packed_text_memmap_dataset
from .sft_builder import build_sft_text_memmap_dataset

from ..data_paths import MINIMIND_SFT_SEEKER, MINIMIND_TEXT_CORPUS, TEXT_PRETRAIN_340, TEXT_SFT_340, TOKENIZER_VOCAB_SIZE


def _ensure_packed_text(
    *,
    text_path: Path,
    out_dir: Path,
    tokenizer: Tokenizer,
    vocab_size: int,
    max_seq_len: int,
    max_samples: int,
    seed: int,
    overwrite: bool,
) -> None:
    meta_path = out_dir / "meta.json"
    if meta_path.exists() and not overwrite:
        meta = MemmapMeta.load(meta_path)
        if int(meta.seq_len) == int(max_seq_len) and int(meta.vocab_size) == int(vocab_size):
            print(f"skip: packed text exists -> {out_dir}")
            return

    t0 = time.time()
    stats = build_packed_text_memmap_dataset(
        text_path=text_path,
        out_dir=out_dir,
        text_tokenizer=tokenizer,
        max_seq_len=int(max_seq_len),
        vocab_size=int(vocab_size),
        max_samples=int(max_samples),
        overwrite=bool(overwrite),
        compact=True,
        resume=False,
        repeat=False,
        shuffle_buffer=0,
        seed=int(seed),
        flush_every=4096,
    )
    print(f"ok: packed text -> {out_dir} (samples={stats.num_out_samples}, sec={int(time.time()-t0)})")


def _ensure_sft_text(
    *,
    in_jsonl: Path,
    out_dir: Path,
    tokenizer: Tokenizer,
    vocab_size: int,
    max_seq_len: int,
    overwrite: bool,
) -> None:
    meta_path = out_dir / "meta.json"
    if meta_path.exists() and not overwrite:
        meta = MemmapMeta.load(meta_path)
        if int(meta.seq_len) == int(max_seq_len) and int(meta.vocab_size) == int(vocab_size):
            print(f"skip: sft text exists -> {out_dir}")
            return

    if overwrite and out_dir.exists():
        shutil.rmtree(out_dir)

    t0 = time.time()
    build_sft_text_memmap_dataset(
        jsonl_path=str(in_jsonl),
        out_dir=str(out_dir),
        text_tokenizer=tokenizer,
        vocab_size=int(vocab_size),
        max_seq_len=int(max_seq_len),
        resume=False,
        flush_every=200,
    )
    print(f"ok: sft text -> {out_dir} (sec={int(time.time()-t0)})")


def ensure_memmaps(cfg: dict, tokenizer: Tokenizer) -> None:
    seed = int(cfg.get("seed", 42))
    overwrite = bool(cfg.get("overwrite", False))
    limits = cfg.get("limits", {})

    vocab_size = int(TOKENIZER_VOCAB_SIZE)

    text_corpus = Path(MINIMIND_TEXT_CORPUS)
    if not text_corpus.exists():
        raise FileNotFoundError(text_corpus)

    sft_jsonl = Path(MINIMIND_SFT_SEEKER)
    if not sft_jsonl.exists():
        raise FileNotFoundError(sft_jsonl)

    text340_dir = Path(TEXT_PRETRAIN_340)
    sft_dir = Path(TEXT_SFT_340)

    if overwrite:
        for d in (text340_dir, sft_dir):
            if d.exists():
                shutil.rmtree(d)

    max_samples_512 = int(limits.get("max_samples_512", 1_000_000))

    print("== prepare: packed text (340) ==")
    _ensure_packed_text(
        text_path=text_corpus,
        out_dir=text340_dir,
        tokenizer=tokenizer,
        vocab_size=vocab_size,
        max_seq_len=340,
        max_samples=max_samples_512,
        seed=seed,
        overwrite=overwrite,
    )

    print("== prepare: sft text (340) ==")
    _ensure_sft_text(
        in_jsonl=sft_jsonl,
        out_dir=sft_dir,
        tokenizer=tokenizer,
        vocab_size=vocab_size,
        max_seq_len=340,
        overwrite=overwrite,
    )

    print("ok: processed ready -> data/processed")
