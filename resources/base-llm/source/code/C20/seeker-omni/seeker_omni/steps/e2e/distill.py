from __future__ import annotations

import torch
import torch.nn.functional as F


def mse_distill(student_hidden: torch.Tensor, teacher_hidden: torch.Tensor) -> torch.Tensor:
    return F.mse_loss(student_hidden.float(), teacher_hidden.float())

