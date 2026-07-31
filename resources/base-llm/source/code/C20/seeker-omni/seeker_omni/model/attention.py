from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from .rope import apply_rope


PastKeyValue = tuple[torch.Tensor, torch.Tensor]


class SeekerSelfAttention(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        dropout: float = 0.0,
    ):
        super().__init__()
        if hidden_size % num_heads != 0:
            raise ValueError("hidden_size must be divisible by num_heads")
        if num_heads % num_kv_heads != 0:
            raise ValueError("num_heads must be divisible by num_kv_heads")

        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = hidden_size // num_heads
        self.dropout = dropout

        q_out = num_heads * self.head_dim
        kv_out = num_kv_heads * self.head_dim

        self.q_proj = nn.Linear(hidden_size, q_out, bias=False)
        self.k_proj = nn.Linear(hidden_size, kv_out, bias=False)
        self.v_proj = nn.Linear(hidden_size, kv_out, bias=False)
        self.o_proj = nn.Linear(q_out, hidden_size, bias=False)

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
        bsz, seq_len, _ = x.shape

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        q = q.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)  # [B,H,S,D]
        k = k.view(bsz, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)  # [B,K,S,D]
        v = v.view(bsz, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)

        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)

        past_len = 0
        if past_kv is not None:
            past_k, past_v = past_kv
            past_len = int(past_k.shape[2])
            k_all = torch.cat([past_k.to(device=k.device, dtype=k.dtype), k], dim=2)
            v_all = torch.cat([past_v.to(device=v.device, dtype=v.dtype), v], dim=2)
        else:
            k_all = k
            v_all = v

        if self.num_kv_heads != self.num_heads:
            repeat = self.num_heads // self.num_kv_heads
            k_rep = k_all.repeat_interleave(repeat, dim=1)
            v_rep = v_all.repeat_interleave(repeat, dim=1)
        else:
            k_rep = k_all
            v_rep = v_all

        key_len = int(k_rep.shape[2])
        query_len = int(q.shape[2])

        pad_bias = None
        if attention_mask is not None:
            if attention_mask.shape[0] != bsz or int(attention_mask.shape[1]) != key_len:
                raise ValueError(f"attention_mask must be [B,{key_len}], got {tuple(attention_mask.shape)}")

            mask = attention_mask
            if mask.dtype != torch.float32:
                mask = mask.to(dtype=torch.float32)
            neg = torch.finfo(q.dtype).min
            pad_bias = (1.0 - mask) * neg
            pad_bias = pad_bias[:, None, None, :]  # [B,1,1,K]

        if past_len == 0 and query_len == key_len:
            attn_bias = pad_bias
            out = F.scaled_dot_product_attention(
                q,
                k_rep,
                v_rep,
                attn_mask=attn_bias,
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=True,
            )
        else:
            neg = torch.finfo(q.dtype).min
            q_pos = torch.arange(query_len, device=q.device)
            k_pos = torch.arange(key_len, device=q.device)
            causal = k_pos[None, :] <= (past_len + q_pos[:, None])
            causal_bias = torch.where(causal, 0.0, neg).to(dtype=q.dtype)
            causal_bias = causal_bias[None, None, :, :]  # [1,1,Q,K]

            if pad_bias is not None:
                attn_bias = causal_bias + pad_bias
            else:
                attn_bias = causal_bias

            out = F.scaled_dot_product_attention(
                q,
                k_rep,
                v_rep,
                attn_mask=attn_bias,
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=False,
            )

        out = out.transpose(1, 2).contiguous().view(bsz, seq_len, self.hidden_size)
        out = self.o_proj(out)

        if use_cache:
            return out, (k_all, v_all)
        return out
