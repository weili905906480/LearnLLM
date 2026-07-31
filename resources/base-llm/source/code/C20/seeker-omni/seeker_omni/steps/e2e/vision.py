from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image


def load_rgb(path: str) -> Image.Image:
    img = Image.open(path)
    if img.mode in ("RGBA", "LA"):
        img = img.convert("RGB")
    return img


def pool_tokens_torch(x: torch.Tensor, *, target_tokens: int) -> torch.Tensor:
    # x: [B,T,D] -> [B,target,D]
    if x.ndim != 3:
        raise ValueError(f"expected [B,T,D], got {tuple(x.shape)}")
    if int(x.shape[1]) == int(target_tokens):
        return x
    t = x.transpose(1, 2)  # [B,D,T]
    pooled = F.adaptive_avg_pool1d(t, int(target_tokens)).transpose(1, 2)
    return pooled


def freeze_vision_all_but_last_n(vision: torch.nn.Module, *, last_n: int) -> None:
    for p in vision.parameters():
        p.requires_grad = False

    n = int(last_n)
    if n <= 0:
        return

    vm = getattr(vision, "vision_model", vision)
    layers = getattr(getattr(vm, "encoder", None), "layers", None)
    if layers is None:
        raise SystemExit("could not locate vision encoder layers to unfreeze (expected vision_model.encoder.layers)")

    layers = list(layers)
    for layer in layers[-n:]:
        for p in layer.parameters():
            p.requires_grad = True

    pln = getattr(vm, "post_layernorm", None)
    if pln is not None:
        for p in pln.parameters():
            p.requires_grad = True


def default_tb_dir(out_dir: Path) -> Path:
    parts = out_dir.parts
    rel = Path(*parts[1:]) if (len(parts) >= 2 and parts[0].lower() == "checkpoints") else out_dir
    return Path("outputs") / "tb" / rel
