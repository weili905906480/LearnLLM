import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class MemmapMeta:
    """memmap 数据集的元信息（用于数据准备阶段写入/校验）。"""

    num_samples: int
    seq_len: int
    vocab_size: int
    pad_id: int

    input_dtype: str = "int32"
    label_dtype: str = "int32"
    attention_dtype: str = "uint8"

    image_tokens: int | None = None
    image_feat_dim: int | None = None

    @staticmethod
    def load(path: str | Path) -> "MemmapMeta":
        p = Path(path)
        obj = json.loads(p.read_text(encoding="utf-8"))
        return MemmapMeta(
            num_samples=int(obj["num_samples"]),
            seq_len=int(obj["seq_len"]),
            vocab_size=int(obj["vocab_size"]),
            pad_id=int(obj["pad_id"]),
            input_dtype=str(obj.get("input_dtype", "int32")),
            label_dtype=str(obj.get("label_dtype", "int32")),
            attention_dtype=str(obj.get("attention_dtype", "uint8")),
            image_tokens=(int(obj["image_tokens"]) if obj.get("image_tokens") is not None else None),
            image_feat_dim=(int(obj["image_feat_dim"]) if obj.get("image_feat_dim") is not None else None),
        )

    def save(self, path: str | Path) -> None:
        p = Path(path)
        obj = {
            "num_samples": int(self.num_samples),
            "seq_len": int(self.seq_len),
            "vocab_size": int(self.vocab_size),
            "pad_id": int(self.pad_id),
            "input_dtype": str(self.input_dtype),
            "label_dtype": str(self.label_dtype),
            "attention_dtype": str(self.attention_dtype),
        }
        if self.image_tokens is not None:
            obj["image_tokens"] = int(self.image_tokens)
            obj["image_feat_dim"] = int(self.image_feat_dim or 0)

        p.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


class MemmapWriter:
    """将 token/labels/attention 写入到 memmap 目录。"""

    def __init__(
        self,
        out_dir: str | Path,
        *,
        num_samples: int,
        seq_len: int,
        vocab_size: int,
        pad_id: int,
        input_dtype: str = "int32",
        label_dtype: str = "int32",
        attention_dtype: str = "uint8",
        image_tokens: int | None = None,
        image_feat_dim: int | None = None,
        mode: str = "w+",
    ) -> None:
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)

        mode = str(mode)
        if mode not in {"w+", "r+"}:
            raise ValueError(f"unsupported memmap mode: {mode!r} (expected 'w+' or 'r+')")

        expected = MemmapMeta(
            num_samples=int(num_samples),
            seq_len=int(seq_len),
            vocab_size=int(vocab_size),
            pad_id=int(pad_id),
            input_dtype=str(input_dtype),
            label_dtype=str(label_dtype),
            attention_dtype=str(attention_dtype),
            image_tokens=(int(image_tokens) if image_tokens is not None else None),
            image_feat_dim=(int(image_feat_dim) if image_feat_dim is not None else None),
        )

        meta_path = self.out_dir / "meta.json"
        if mode == "w+":
            self.meta = expected
            self.meta.save(meta_path)
        else:
            if not meta_path.exists():
                raise FileNotFoundError(f"missing meta.json for resume: {meta_path}")
            self.meta = MemmapMeta.load(meta_path)
            if self.meta != expected:
                raise ValueError(f"memmap meta mismatch for resume: actual={self.meta} expected={expected}")

        n = int(self.meta.num_samples)
        s = int(self.meta.seq_len)

        self.input_ids = np.memmap(
            self.out_dir / "input_ids.bin",
            dtype=np.dtype(self.meta.input_dtype),
            mode=mode,
            shape=(n, s),
        )
        self.labels = np.memmap(
            self.out_dir / "labels.bin",
            dtype=np.dtype(self.meta.label_dtype),
            mode=mode,
            shape=(n, s),
        )
        self.attention_mask = np.memmap(
            self.out_dir / "attention_mask.bin",
            dtype=np.dtype(self.meta.attention_dtype),
            mode=mode,
            shape=(n, s),
        )

        self.image_feats = None
        if self.meta.image_tokens is not None and self.meta.image_feat_dim is not None:
            self.image_feats = np.memmap(
                self.out_dir / "image_feats.bin",
                dtype=np.float16,
                mode=mode,
                shape=(n, int(self.meta.image_tokens), int(self.meta.image_feat_dim)),
            )

    def write(
        self,
        idx: int,
        *,
        input_ids: np.ndarray,
        labels: np.ndarray,
        attention_mask: np.ndarray,
        image_feats: np.ndarray | None = None,
    ) -> None:
        self.input_ids[idx] = input_ids.astype(self.input_ids.dtype, copy=False)
        self.labels[idx] = labels.astype(self.labels.dtype, copy=False)
        self.attention_mask[idx] = attention_mask.astype(self.attention_mask.dtype, copy=False)

        if self.image_feats is not None:
            if image_feats is None:
                raise ValueError("image_feats must be provided when meta includes image_feats")
            self.image_feats[idx] = image_feats.astype(np.float16, copy=False)

    def flush(self) -> None:
        self.input_ids.flush()
        self.labels.flush()
        self.attention_mask.flush()
        if self.image_feats is not None:
            self.image_feats.flush()

