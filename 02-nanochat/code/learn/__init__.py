"""
学习模块

包含 LLM/Transformer 学习所需的各种模块：
- math: 数学基础（线性代数、概率论、微积分）
- (未来可扩展: transformer, training, inference 等)
"""

from . import math as math_module

__all__ = ['math_module']

def run_all():
    """运行所有学习模块"""
    math_module.run_all_demos()
