from pathlib import Path

DATA_RAW = Path("data/raw")
DATA_INTERIM = Path("data/interim")
DATA_PROCESSED = Path("data/processed")

ARTIFACTS = Path("artifacts")

# -----------------------------
# 默认配置
# -----------------------------

SEED = 42

# 控制数据准备阶段是否覆盖（分词器 + 处理后的 memmap）。
OVERWRITE = False

# MiniMind（文本）
MINIMIND_DOWNLOAD = True
MINIMIND_OVERWRITE_DOWNLOAD = False
MINIMIND_PRETRAIN_URL = (
    "https://hf-mirror.com/datasets/jingyaogong/minimind_dataset/resolve/main/pretrain_hq.jsonl?download=true"
)
MINIMIND_SFT_URL = "https://hf-mirror.com/datasets/jingyaogong/minimind_dataset/resolve/main/sft_mini_512.jsonl?download=true"

# Flickr8k（图文）
MM_DOWNLOAD = True
MM_OVERWRITE_DOWNLOAD = False
MM_IMAGES_URL = "https://github.com/jbrownlee/Datasets/releases/download/Flickr8k/Flickr8k_Dataset.zip"
MM_TEXT_URL = "https://github.com/jbrownlee/Datasets/releases/download/Flickr8k/Flickr8k_text.zip"
MM_ZHC_CAPTIONS_URL = "https://raw.githubusercontent.com/li-xirong/flickr8kcn/master/data/flickr8kzhc.caption.txt"

# 分词器训练
TOKENIZER_SAMPLE_RATIO = 0.6
TOKENIZER_SAMPLE_SEED = 42

# memmap 构建限制
MAX_SAMPLES_512 = 1_000_000

# MiniMind（文本）
MINIMIND_DIR = DATA_RAW / "minimind"
MINIMIND_PRETRAIN_JSONL = MINIMIND_DIR / "pretrain_hq.jsonl"
MINIMIND_SFT_JSONL = MINIMIND_DIR / "sft_mini_512.jsonl"
MINIMIND_TEXT_CORPUS = DATA_INTERIM / "tokenizer_corpus" / "minimind_pretrain_text.txt"
MINIMIND_SFT_SEEKER = DATA_INTERIM / "sft_converted" / "minimind_sft_chatml.jsonl"

# Flickr8k（图文）
FLICKR8K_DIR = DATA_RAW / "flickr8k"
FLICKR8K_IMAGES_ZIP = FLICKR8K_DIR / "Flickr8k_Dataset.zip"
FLICKR8K_TEXT_ZIP = FLICKR8K_DIR / "Flickr8k_text.zip"
FLICKR8K_IMAGES_DIR = FLICKR8K_DIR / "Flickr8k_Dataset"
FLICKR8K_TEXT_DIR = FLICKR8K_DIR / "text"
FLICKR8K_ZHC_CAPTIONS = FLICKR8K_DIR / "flickr8kzhc.caption.txt"
FLICKR8K_TRAIN_LIST = FLICKR8K_TEXT_DIR / "Flickr_8k.trainImages.txt"

# 图文 JSONL 交接文件（E2E 训练读取）
MM_TRAIN_JSONL = DATA_INTERIM / "packs" / "mm" / "train_imgonly.jsonl"

# 分词器（默认路线固定）
TOKENIZER_VOCAB_SIZE = 6400
TOKENIZER_DIR = ARTIFACTS / "tokenizers" / "bpe_m2chatml_6400"

# 处理后的数据集（固定输出结构）
TEXT_PRETRAIN_340 = DATA_PROCESSED / "text_pretrain_packed_340_u16_offline"
TEXT_SFT_340 = DATA_PROCESSED / "text_sft_340"


def default_dataprep_cfg() -> dict:
    """返回 dataprep 默认配置 dict"""

    return {
        "seed": int(SEED),
        "overwrite": bool(OVERWRITE),
        "minimind": {
            "download": bool(MINIMIND_DOWNLOAD),
            "overwrite_download": bool(MINIMIND_OVERWRITE_DOWNLOAD),
            "pretrain_url": str(MINIMIND_PRETRAIN_URL),
            "sft_url": str(MINIMIND_SFT_URL),
        },
        "mm": {
            "download": bool(MM_DOWNLOAD),
            "overwrite_download": bool(MM_OVERWRITE_DOWNLOAD),
            "images_url": str(MM_IMAGES_URL),
            "text_url": str(MM_TEXT_URL),
            "zhc_captions_url": str(MM_ZHC_CAPTIONS_URL),
        },
        "tokenizer": {
            "sample_ratio": float(TOKENIZER_SAMPLE_RATIO),
            "sample_seed": int(TOKENIZER_SAMPLE_SEED),
        },
        "limits": {
            "max_samples_512": int(MAX_SAMPLES_512),
        },
    }
