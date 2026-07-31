from .flickr8k import ensure_flickr8k_mm_jsonl
from .minimind import ensure_minimind


def run(cfg: dict) -> None:
    seed = int(cfg.get("seed", 42))

    print("== download: minimind ==")
    ensure_minimind(cfg, seed=seed)

    print("== download: flickr8k ==")
    ensure_flickr8k_mm_jsonl(cfg)

