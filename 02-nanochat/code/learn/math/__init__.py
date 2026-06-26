"""
数学基础学习模块

包含 LLM/Transformer 所需的数学基础知识：
- 线性代数
- 概率论与统计
- 微积分与梯度
- 激活函数
- 损失函数
"""

from .linear_algebra import *
from .probability import *
from .calculus import *
from .activations import *
from .losses import *

__all__ = [
    'linear_algebra',
    'probability',
    'calculus',
    'activations',
    'losses',
]

def run_all_demos():
    """运行所有数学模块的演示"""
    print("=" * 60)
    print("数学基础学习模块")
    print("=" * 60)
    
    from . import linear_algebra
    from . import probability
    from . import calculus
    from . import activations
    from . import losses
    
    print("\n1. 线性代数基础")
    print("-" * 40)
    linear_algebra.demo()
    
    print("\n2. 概率论与统计")
    print("-" * 40)
    probability.demo()
    
    print("\n3. 微积分与梯度")
    print("-" * 40)
    calculus.demo()
    
    print("\n4. 激活函数")
    print("-" * 40)
    activations.demo()
    
    print("\n5. 损失函数")
    print("-" * 40)
    losses.demo()
    
    print("\n" + "=" * 60)
    print("所有演示完成！")
    print("=" * 60)


if __name__ == "__main__":
    run_all_demos()
