import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np


class MemmapDataset:
    """读取 dataprep 生成的 memmap 数据集（训练侧只读）。"""

    def __init__(self, data_dir: str | Path):
        self.data_dir = Path(data_dir)
        meta_path = self.data_dir / "meta.json"
        obj = json.loads(meta_path.read_text(encoding="utf-8"))

        # 训练代码会使用 ds.meta.seq_len / ds.meta.vocab_size 等字段。
        self.meta = SimpleNamespace(
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

        n = int(self.meta.num_samples)
        s = int(self.meta.seq_len)

        self._input_ids = np.memmap(
            self.data_dir / "input_ids.bin",
            dtype=np.dtype(self.meta.input_dtype),
            mode="c",
            shape=(n, s),
        )
        self._labels = np.memmap(
            self.data_dir / "labels.bin",
            dtype=np.dtype(self.meta.label_dtype),
            mode="c",
            shape=(n, s),
        )
        self._attn = np.memmap(
            self.data_dir / "attention_mask.bin",
            dtype=np.dtype(self.meta.attention_dtype),
            mode="c",
            shape=(n, s),
        )

        self._image_feats = None
        if self.meta.image_tokens is not None:
            self._image_feats = np.memmap(
                self.data_dir / "image_feats.bin",
                dtype=np.float16,
                mode="c",
                shape=(n, int(self.meta.image_tokens), int(self.meta.image_feat_dim or 0)),
            )

    def __len__(self) -> int:
        return int(self.meta.num_samples)

    def __getitem__(self, idx: int):
        import torch

        out = {
            "input_ids": torch.from_numpy(self._input_ids[int(idx)]),
            "labels": torch.from_numpy(self._labels[int(idx)]),
            "attention_mask": torch.from_numpy(self._attn[int(idx)]),
        }
        if self._image_feats is not None:
            out["image_feats"] = torch.from_numpy(self._image_feats[int(idx)])
        return out

