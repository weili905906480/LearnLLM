from __future__ import annotations

import math


def cosine_lr(step: int, *, base_lr: float, total_steps: int, warmup_steps: int) -> float:
    if total_steps <= 0:
        return float(base_lr)
    if warmup_steps > 0 and step < warmup_steps:
        return float(base_lr) * float(step + 1) / float(warmup_steps)

    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    progress = min(max(progress, 0.0), 1.0)

    # 对齐 MiniMind：在余弦衰减末端保留 0.1 * base_lr 的下限。
    # 当 warmup_steps=0 时，对应：lr*(0.1 + 0.45*(1+cos(pi*t)))。区别仅在于
    # 这里用归一化后的进度 progress 表达。
    min_ratio = 0.1
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return float(base_lr) * (float(min_ratio) + (1.0 - float(min_ratio)) * float(cosine))
