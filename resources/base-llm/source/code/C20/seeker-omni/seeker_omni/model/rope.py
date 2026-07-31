import torch
from torch import nn


def _build_rope_cache(seq_len: int, dim: int, theta: float) -> tuple[torch.Tensor, torch.Tensor]:
    if dim % 2 != 0:
        raise ValueError(f"rope dim must be even, got {dim}")

    inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
    positions = torch.arange(seq_len, dtype=torch.float32)
    freqs = torch.einsum("i,j->ij", positions, inv_freq)  # [seq, dim/2]
    return torch.cos(freqs), torch.sin(freqs)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """应用 RoPE（旋转位置编码）。

    x:   [B, H, S, D]
    cos: [S, D/2]
    sin: [S, D/2]
    """
    x_even = x[..., 0::2]
    x_odd = x[..., 1::2]

    cos = cos[None, None, :, :]
    sin = sin[None, None, :, :]

    out_even = x_even * cos - x_odd * sin
    out_odd = x_even * sin + x_odd * cos
    return torch.stack((out_even, out_odd), dim=-1).flatten(-2)


class RotaryEmbedding(nn.Module):
    def __init__(self, dim: int, max_seq_len: int, theta: float = 1_000_000.0):
        super().__init__()
        cos, sin = _build_rope_cache(max_seq_len, dim, theta)
        self.register_buffer("cos_cache", cos, persistent=False)
        self.register_buffer("sin_cache", sin, persistent=False)

    def get_cos_sin(
        self,
        seq_len: int,
        *,
        device: torch.device,
        dtype: torch.dtype,
        offset: int = 0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        cos = self.cos_cache[offset : offset + seq_len].to(device=device, dtype=dtype)
        sin = self.sin_cache[offset : offset + seq_len].to(device=device, dtype=dtype)
        return cos, sin
