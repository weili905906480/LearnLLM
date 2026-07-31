"""Seeker-Omni 模型包。

尽量保持模型代码不依赖数据集 / 特征提取器等数据准备逻辑。
"""

from .lm import SeekerOmniLM

__all__ = ["SeekerOmniLM"]

