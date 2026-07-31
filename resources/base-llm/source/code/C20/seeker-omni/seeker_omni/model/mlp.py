import torch
import torch.nn.functional as F
from torch import nn


def _round_up(x: int, multiple: int) -> int:
    return ((x + multiple - 1) // multiple) * multiple


class SeekerMLP(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int | None = None, dropout: float = 0.0):
        super().__init__()
        if intermediate_size is None:
            intermediate_size = int(hidden_size * 8 / 3)
            intermediate_size = _round_up(intermediate_size, 64)

        self.w_gate = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.w_up = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.w_down = nn.Linear(intermediate_size, hidden_size, bias=False)
        self.dropout = dropout

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.silu(self.w_gate(x)) * self.w_up(x)
        x = self.w_down(x)
        return F.dropout(x, p=self.dropout, training=self.training)
