"""
激活函数模块

包含神经网络中常用的激活函数：
- Sigmoid
- Tanh
- ReLU
- GELU
- Softmax
- Swish/SiLU

与 nanochat 的关联：
- nanochat/gpt.py:100 - MLP 中的 GELU 激活
- nanochat/gpt.py:472 - 输出层的 Softmax
"""

import numpy as np
from typing import Tuple


# =============================================================================
# Sigmoid 函数
# =============================================================================

def sigmoid(x: np.ndarray) -> np.ndarray:
    """Sigmoid 函数: σ(x) = 1 / (1 + e^(-x))
    
    特点:
    - 输出范围: (0, 1)
    - 常用于二分类输出层
    - 存在梯度消失问题
    """
    return 1 / (1 + np.exp(-x))


def sigmoid_derivative(x: np.ndarray) -> np.ndarray:
    """Sigmoid 导数: σ'(x) = σ(x) * (1 - σ(x))"""
    s = sigmoid(x)
    return s * (1 - s)


def sigmoid_demo():
    """Sigmoid 函数演示"""
    print("【Sigmoid 函数】")
    
    x = np.linspace(-5, 5, 11)
    y = sigmoid(x)
    dy = sigmoid_derivative(x)
    
    print(f"输入 x: {x}")
    print(f"Sigmoid: {y.round(4)}")
    print(f"导数:    {dy.round(4)}")
    
    print("\n特点:")
    print("  - 输出范围: (0, 1)")
    print("  - x=0 时，σ(0) = 0.5")
    print("  - 导数最大值: σ'(0) = 0.25")
    print("  - 问题: 深层网络会梯度消失")


# =============================================================================
# Tanh 函数
# =============================================================================

def tanh(x: np.ndarray) -> np.ndarray:
    """Tanh 函数: tanh(x) = (e^x - e^(-x)) / (e^x + e^(-x))
    
    特点:
    - 输出范围: (-1, 1)
    - 零中心化
    - 比 Sigmoid 更好
    """
    return np.tanh(x)


def tanh_derivative(x: np.ndarray) -> np.ndarray:
    """Tanh 导数: tanh'(x) = 1 - tanh²(x)"""
    return 1 - np.tanh(x) ** 2


def tanh_demo():
    """Tanh 函数演示"""
    print("\n【Tanh 函数】")
    
    x = np.linspace(-5, 5, 11)
    y = tanh(x)
    dy = tanh_derivative(x)
    
    print(f"输入 x: {x}")
    print(f"Tanh:    {y.round(4)}")
    print(f"导数:    {dy.round(4)}")
    
    print("\n特点:")
    print("  - 输出范围: (-1, 1)")
    print("  - 零中心化（均值为 0）")
    print("  - 导数最大值: tanh'(0) = 1")
    print("  - 比 Sigmoid 更常用")


# =============================================================================
# ReLU 函数
# =============================================================================

def relu(x: np.ndarray) -> np.ndarray:
    """ReLU 函数: ReLU(x) = max(0, x)
    
    特点:
    - 计算简单
    - 缓解梯度消失
    - 可能导致神经元死亡
    """
    return np.maximum(0, x)


def relu_derivative(x: np.ndarray) -> np.ndarray:
    """ReLU 导数: 1 if x > 0, else 0"""
    return (x > 0).astype(float)


def relu_demo():
    """ReLU 函数演示"""
    print("\n【ReLU 函数】")
    
    x = np.array([-2, -1, 0, 1, 2])
    y = relu(x)
    dy = relu_derivative(x)
    
    print(f"输入 x: {x}")
    print(f"ReLU:    {y}")
    print(f"导数:    {dy}")
    
    print("\n特点:")
    print("  - 计算简单: max(0, x)")
    print("  - 正区间梯度恒为 1")
    print("  - 负区间梯度恒为 0")
    print("  - 问题: 神经元死亡（Dead ReLU）")


# =============================================================================
# Leaky ReLU 函数
# =============================================================================

def leaky_relu(x: np.ndarray, alpha: float = 0.01) -> np.ndarray:
    """Leaky ReLU 函数: max(αx, x)
    
    解决 Dead ReLU 问题
    """
    return np.where(x > 0, x, alpha * x)


def leaky_relu_derivative(x: np.ndarray, alpha: float = 0.01) -> np.ndarray:
    """Leaky ReLU 导数"""
    return np.where(x > 0, 1.0, alpha)


def leaky_relu_demo():
    """Leaky ReLU 函数演示"""
    print("\n【Leaky ReLU 函数】")
    
    x = np.array([-2, -1, 0, 1, 2])
    y = leaky_relu(x)
    dy = leaky_relu_derivative(x)
    
    print(f"输入 x: {x}")
    print(f"Leaky ReLU: {y}")
    print(f"导数:       {dy}")
    
    print("\n特点:")
    print("  - 负区间有小的梯度 (α=0.01)")
    print("  - 解决 Dead ReLU 问题")


# =============================================================================
# GELU 函数 - Transformer 中最常用
# =============================================================================

def gelu(x: np.ndarray) -> np.ndarray:
    """GELU 函数: GELU(x) = x * Φ(x)
    
    其中 Φ(x) 是标准正态分布的 CDF
    
    近似公式: GELU(x) ≈ 0.5 * x * (1 + tanh(√(2/π) * (x + 0.044715 * x³)))
    
    与 nanochat 的关联：
    - nanochat/gpt.py:100 - MLP 中的激活函数
    - 比 ReLU 更平滑，效果更好
    """
    # 近似计算
    return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x ** 3)))


def gelu_derivative(x: np.ndarray) -> np.ndarray:
    """GELU 导数（近似）"""
    # 使用数值导数
    h = 1e-5
    return (gelu(x + h) - gelu(x - h)) / (2 * h)


def gelu_demo():
    """GELU 函数演示"""
    print("\n【GELU 函数 - Transformer 标准】")
    
    x = np.array([-2, -1, 0, 1, 2])
    y = gelu(x)
    dy = gelu_derivative(x)
    
    print(f"输入 x: {x}")
    print(f"GELU:    {y.round(4)}")
    print(f"导数:    {dy.round(4)}")
    
    print("\n特点:")
    print("  - 比 ReLU 更平滑")
    print("  - 负区间有小的非零输出")
    print("  - Transformer 的标准激活函数")
    print("  - nanochat 的 MLP 层使用 GELU")


# =============================================================================
# Swish/SiLU 函数
# =============================================================================

def swish(x: np.ndarray, beta: float = 1.0) -> np.ndarray:
    """Swish 函数: Swish(x) = x * σ(βx)
    
    当 β=1 时称为 SiLU
    """
    return x * sigmoid(beta * x)


def swish_demo():
    """Swish 函数演示"""
    print("\n【Swish/SiLU 函数】")
    
    x = np.array([-2, -1, 0, 1, 2])
    y = swish(x)
    
    print(f"输入 x: {x}")
    print(f"Swish:   {y.round(4)}")
    
    print("\n特点:")
    print("  - 自门控激活函数")
    print("  - β=1 时称为 SiLU")
    print("  - 在某些模型中优于 GELU")


# =============================================================================
# Softmax 函数
# =============================================================================

def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """Softmax 函数: softmax(x_i) = e^(x_i) / Σ e^(x_j)
    
    与 nanochat 的关联：
    - nanochat/gpt.py:472 - 输出层的概率分布
    - 注意力权重计算
    """
    # 减去最大值防止数值溢出
    exp_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)


def softmax_demo():
    """Softmax 函数演示"""
    print("\n【Softmax 函数】")
    
    x = np.array([2.0, 1.0, 0.1])
    y = softmax(x)
    
    print(f"输入 logits: {x}")
    print(f"Softmax:     {y.round(4)}")
    print(f"概率和:      {y.sum():.4f}")
    
    print("\n特点:")
    print("  - 输出是概率分布（和为 1）")
    print("  - 保持相对大小关系")
    print("  - 用于多分类输出层")


# =============================================================================
# 激活函数对比
# =============================================================================

def compare_activations():
    """激活函数对比"""
    print("\n【激活函数对比】")
    
    x = np.linspace(-3, 3, 7)
    
    print(f"输入 x: {x}")
    print(f"\n{'函数':<15} {'输出':<30} {'特点'}")
    print("-" * 60)
    
    activations = [
        ("Sigmoid", sigmoid),
        ("Tanh", tanh),
        ("ReLU", relu),
        ("Leaky ReLU", leaky_relu),
        ("GELU", gelu),
        ("Swish", swish),
    ]
    
    for name, func in activations:
        y = func(x)
        print(f"{name:<15} {str(y.round(2)):<30}")
    
    print("\n选择建议:")
    print("  - 隐藏层: GELU (Transformer) 或 ReLU (传统网络)")
    print("  - 二分类输出: Sigmoid")
    print("  - 多分类输出: Softmax")
    print("  - 回归输出: 无激活")


# =============================================================================
# 综合演示
# =============================================================================

def demo():
    """运行所有激活函数演示"""
    sigmoid_demo()
    tanh_demo()
    relu_demo()
    leaky_relu_demo()
    gelu_demo()
    swish_demo()
    softmax_demo()
    compare_activations()
    
    print("\n" + "=" * 60)
    print("激活函数演示完成！")
    print("=" * 60)
    print("\n关键概念:")
    print("  1. 激活函数引入非线性")
    print("  2. GELU 是 Transformer 的标准激活函数")
    print("  3. Softmax 用于输出概率分布")
    print("  4. 选择合适的激活函数很重要")


if __name__ == "__main__":
    demo()
