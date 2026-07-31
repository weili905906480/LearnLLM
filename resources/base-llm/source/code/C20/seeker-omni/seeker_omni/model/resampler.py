import torch
from torch import nn


class PerceiverResampler(nn.Module):
    def __init__(
        self,
        *,
        dim: int,
        num_latents: int,
        num_layers: int = 2,
        num_heads: int = 8,
        ff_mult: int = 4,
    ) -> None:
        super().__init__()
        self.dim = int(dim)
        self.num_latents = int(num_latents)
        self.num_layers = int(num_layers)
        self.num_heads = int(num_heads)
        self.ff_mult = int(ff_mult)

        if self.dim <= 0:
            raise ValueError("dim must be > 0")
        if self.num_latents <= 0:
            raise ValueError("num_latents must be > 0")
        if self.num_layers <= 0:
            raise ValueError("num_layers must be > 0")
        if self.num_heads <= 0:
            raise ValueError("num_heads must be > 0")
        if self.dim % self.num_heads != 0:
            raise ValueError(f"dim must be divisible by num_heads (dim={self.dim} heads={self.num_heads})")

        lat = torch.empty((self.num_latents, self.dim), dtype=torch.float32)
        nn.init.trunc_normal_(lat, std=0.02)
        self.latents = nn.Parameter(lat)

        layers: list[nn.Module] = []
        for _ in range(int(self.num_layers)):
            layers.append(
                nn.ModuleDict(
                    {
                        "ln_q": nn.LayerNorm(self.dim),
                        "ln_kv": nn.LayerNorm(self.dim),
                        "attn": nn.MultiheadAttention(self.dim, self.num_heads, batch_first=True),
                        "ln_ff": nn.LayerNorm(self.dim),
                        "ff": nn.Sequential(
                            nn.Linear(self.dim, self.dim * int(self.ff_mult)),
                            nn.GELU(),
                            nn.Linear(self.dim * int(self.ff_mult), self.dim),
                        ),
                    }
                )
            )
        self.layers = nn.ModuleList(layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f"expected x=[B,T,D], got {tuple(x.shape)}")
        if int(x.shape[-1]) != int(self.dim):
            raise ValueError(f"expected D={self.dim}, got {int(x.shape[-1])}")

        b = int(x.shape[0])
        latents = self.latents.unsqueeze(0).expand(b, -1, -1)

        for layer in self.layers:
            q = layer["ln_q"](latents)
            kv = layer["ln_kv"](x)
            attn_out, _ = layer["attn"](q, kv, kv, need_weights=False)
            latents = latents + attn_out
            latents = latents + layer["ff"](layer["ln_ff"](latents))

        return latents
