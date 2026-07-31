"""Seeker-Omni 的数据流水线。

原始 JSONL -> 分词后的 memmap -> PyTorch Dataset。

本包不应依赖模型代码。
"""

from .memmap import MemmapDataset

__all__ = ["MemmapDataset"]
