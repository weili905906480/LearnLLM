import torch
from torch import nn

from .attention import PastKeyValue, SeekerSelfAttention
from .mlp import SeekerMLP
from .norm import RMSNorm


class SeekerBlock(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        *,
        dropout: float = 0.0,
        intermediate_size: int | None = None,
    ):
        super().__init__()
        self.attn_norm = RMSNorm(hidden_size)
        self.attn = SeekerSelfAttention(hidden_size, num_heads, num_kv_heads, dropout=dropout)
        self.mlp_norm = RMSNorm(hidden_size)
        self.mlp = SeekerMLP(hidden_size, intermediate_size=intermediate_size, dropout=dropout)

    def forward(
        self,
        x: torch.Tensor,
        *,
        cos: torch.Tensor,
        sin: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        past_kv: PastKeyValue | None = None,
        use_cache: bool = False,
    ):
        if use_cache:
            attn_out, present_kv = self.attn(
                self.attn_norm(x),
                cos=cos,
                sin=sin,
                attention_mask=attention_mask,
                past_kv=past_kv,
                use_cache=True,
            )
            x = x + attn_out
            x = x + self.mlp(self.mlp_norm(x))
            return x, present_kv

        x = x + self.attn(self.attn_norm(x), cos=cos, sin=sin, attention_mask=attention_mask)
        x = x + self.mlp(self.mlp_norm(x))
        return x
