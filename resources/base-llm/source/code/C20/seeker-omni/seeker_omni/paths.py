from __future__ import annotations

"""
训练侧路径常量。

数据准备由 `dataprep` 模块完成；本文件只声明训练/推理需要读取的交接路径。
"""

from pathlib import Path

DATA_PROCESSED = Path("data/processed")

# E2E 训练读取的图文 JSONL，由 dataprep 生成。
MM_TRAIN_JSONL = Path("data/interim/packs/mm/train_imgonly.jsonl")

# 默认分词器路径与词表大小。
TOKENIZER_VOCAB_SIZE = 6400
TOKENIZER_DIR = Path("artifacts/tokenizers/bpe_m2chatml_6400")

# 处理后的 memmap 数据集目录。
TEXT_PRETRAIN_340 = DATA_PROCESSED / "text_pretrain_packed_340_u16_offline"
TEXT_SFT_340 = DATA_PROCESSED / "text_sft_340"
