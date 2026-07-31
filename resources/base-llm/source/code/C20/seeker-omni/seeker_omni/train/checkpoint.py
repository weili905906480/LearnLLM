from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def _jsonable(obj: Any) -> Any:
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, dict):
        return {k: _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, tuple):
        return [_jsonable(v) for v in obj]
    return obj


def save_checkpoint(
    out_dir: str | Path,
    *,
    model,
    optimizer,
    step: int,
    cfg: dict[str, Any],
    keep_last: int | None = None,
) -> Path:
    import torch

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    (out / "config.json").write_text(json.dumps(_jsonable(cfg), ensure_ascii=False, indent=2), encoding="utf-8")

    ckpt_path = out / f"step_{step:09d}.pt"
    tmp_path = ckpt_path.with_suffix(ckpt_path.suffix + ".tmp")
    try:
        torch.save(
            {
                "step": int(step),
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
            },
            tmp_path,
        )
        tmp_path.replace(ckpt_path)
    finally:
        tmp_path.unlink(missing_ok=True)

    if keep_last is not None:
        keep_last = int(keep_last)
        if keep_last > 0:
            pts = sorted(out.glob("step_*.pt"))
            if len(pts) > keep_last:
                for p in pts[: -keep_last]:
                    p.unlink(missing_ok=True)

    return ckpt_path


def latest_checkpoint(out_dir: str | Path) -> Path | None:
    out = Path(out_dir)
    if not out.exists():
        return None
    pts = sorted(out.glob("step_*.pt"))
    return pts[-1] if pts else None


def load_checkpoint(path: str | Path, *, model, optimizer: Any | None = None, strict: bool = True) -> int:
    import torch

    obj = torch.load(Path(path), map_location="cpu")
    model.load_state_dict(obj["model"], strict=bool(strict))
    if optimizer is not None:
        optimizer.load_state_dict(obj["optimizer"])
    return int(obj.get("step", 0))
