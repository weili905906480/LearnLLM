from .memmap import ensure_memmaps
from .tokenizer import ensure_tokenizer_and_load


def run(cfg: dict) -> None:
    tok = ensure_tokenizer_and_load(cfg)
    ensure_memmaps(cfg, tok)

