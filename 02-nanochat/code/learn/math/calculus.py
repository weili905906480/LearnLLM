"""
微积分与梯度基础模块

包含 LLM/Transformer 所需的微积分知识：
- 导数与偏导数
- 梯度
- 链式法则（反向传播的基础）
- 梯度下降

与 nanochat 的关联：
- 反向传播：loss.backward()
- 优化器：optimizer.step()
- 学习率调度：get_lr_multiplier()
"""

import numpy as np
from typing import Callable, Tuple


# =============================================================================
# 导数基础
# =============================================================================

def derivative_basics():
    """导数基础概念演示"""
    print("【导数基础】")
    
    # 导数的定义: f'(x) = lim(h→0) [f(x+h) - f(x)] / h
    print("导数定义: f'(x) = lim(h→0) [f(x+h) - f(x)] / h")
    
    # 数值导数
    def numerical_derivative(f: Callable, x: float, h: float = 1e-5) -> float:
        """数值导数（用于验证）"""
        return (f(x + h) - f(x - h)) / (2 * h)
    
    # 示例函数
    def f(x):
        return x ** 2
    
    def f_prime(x):
        return 2 * x
    
    x = 3.0
    numerical = numerical_derivative(f, x)
    analytical = f_prime(x)
    
    print(f"\n函数: f(x) = x²")
    print(f"在 x={x} 处:")
    print(f"  解析导数: f'({x}) = {analytical}")
    print(f"  数值导数: {numerical:.6f}")
    print(f"  误差: {abs(numerical - analytical):.10f}")
    
    # 常见函数的导数
    print("\n常见函数的导数:")
    print("  f(x) = x^n    →  f'(x) = n * x^(n-1)")
    print("  f(x) = e^x    →  f'(x) = e^x")
    print("  f(x) = ln(x)  →  f'(x) = 1/x")
    print("  f(x) = sin(x) →  f'(x) = cos(x)")
    print("  f(x) = cos(x) →  f'(x) = -sin(x)")


# =============================================================================
# 偏导数与梯度
# =============================================================================

def partial_derivatives():
    """偏导数与梯度演示"""
    print("\n【偏导数与梯度】")
    
    # 多元函数: f(x, y) = x² + y²
    def f(x, y):
        return x**2 + y**2
    
    # 偏导数
    def df_dx(x, y):
        return 2 * x
    
    def df_dy(x, y):
        return 2 * y
    
    x, y = 2.0, 3.0
    print(f"函数: f(x, y) = x² + y²")
    print(f"在点 ({x}, {y}) 处:")
    print(f"  f({x}, {y}) = {f(x, y)}")
    print(f"  ∂f/∂x = {df_dx(x, y)}")
    print(f"  ∂f/∂y = {df_dy(x, y)}")
    
    # 梯度向量
    gradient = np.array([df_dx(x, y), df_dy(x, y)])
    print(f"\n梯度 ∇f = {gradient}")
    print(f"梯度方向: 函数增长最快的方向")
    print(f"梯度大小: {np.linalg.norm(gradient):.4f}")
    
    # 梯度的几何意义
    print("\n梯度的几何意义:")
    print("  - 梯度指向函数增长最快的方向")
    print("  - 负梯度指向函数下降最快的方向")
    print("  - 梯度下降就是沿着负梯度方向更新参数")


# =============================================================================
# 链式法则
# =============================================================================

def chain_rule():
    """链式法则 - 反向传播的数学基础"""
    print("\n【链式法则】")
    
    # 链式法则: dz/dx = dz/dy * dy/dx
    print("链式法则: dz/dx = dz/dy * dy/dx")
    
    # 示例: z = f(y), y = g(x)
    # z = (2x + 1)²
    
    def g(x):
        return 2 * x + 1  # y = g(x) = 2x + 1
    
    def f(y):
        return y ** 2  # z = f(y) = y²
    
    def composite(x):
        return f(g(x))  # z = (2x + 1)²
    
    # 手动计算导数
    def g_prime(x):
        return 2  # dy/dx = 2
    
    def f_prime(y):
        return 2 * y  # dz/dy = 2y
    
    def composite_prime(x):
        # dz/dx = dz/dy * dy/dx = 2y * 2 = 2(2x+1) * 2 = 4(2x+1)
        return 4 * (2 * x + 1)
    
    x = 3.0
    y = g(x)
    z = f(y)
    
    print(f"\n计算 z = (2x + 1)² 在 x={x} 处的导数:")
    print(f"  x = {x}")
    print(f"  y = g(x) = 2x + 1 = {y}")
    print(f"  z = f(y) = y² = {z}")
    
    # 链式法则计算
    dz_dy = f_prime(y)
    dy_dx = g_prime(x)
    dz_dx_chain = dz_dy * dy_dx
    
    # 直接计算
    dz_dx_direct = composite_prime(x)
    
    print(f"\n链式法则:")
    print(f"  dz/dy = 2y = {dz_dy}")
    print(f"  dy/dx = 2")
    print(f"  dz/dx = dz/dy * dy/dx = {dz_dy} * {dy_dx} = {dz_dx_chain}")
    
    print(f"\n直接计算:")
    print(f"  dz/dx = 4(2x+1) = {dz_dx_direct}")
    
    print(f"\n结果一致: {np.isclose(dz_dx_chain, dz_dx_direct)}")


# =============================================================================
# 反向传播示例
# =============================================================================

def backpropagation_demo():
    """反向传播示例 - 神经网络的梯度计算"""
    print("\n【反向传播示例】")
    
    # 简单的两层网络
    # 输入 x → 线性层1 → ReLU → 线性层2 → 输出 y
    # 损失 L = (y - target)²
    
    np.random.seed(42)
    
    # 输入和目标
    x = np.array([1.0, 2.0])
    target = np.array([1.0])
    
    # 权重初始化
    W1 = np.array([[0.1, 0.2],
                   [0.3, 0.4]])  # 2x2
    b1 = np.array([0.1, 0.1])    # 2
    
    W2 = np.array([[0.5, 0.6]])  # 1x2
    b2 = np.array([0.1])         # 1
    
    print("前向传播:")
    
    # 第一层: h1 = W1 @ x + b1
    h1 = W1 @ x + b1
    print(f"  h1 = W1 @ x + b1 = {h1}")
    
    # ReLU 激活
    h1_relu = np.maximum(0, h1)
    print(f"  h1_relu = ReLU(h1) = {h1_relu}")
    
    # 第二层: y = W2 @ h1_relu + b2
    y = W2 @ h1_relu + b2
    print(f"  y = W2 @ h1_relu + b2 = {y}")
    
    # 损失: L = (y - target)²
    loss = (y - target) ** 2
    print(f"  L = (y - target)² = {loss}")
    
    print("\n反向传播:")
    
    # dL/dy = 2(y - target)
    dL_dy = 2 * (y - target)
    print(f"  dL/dy = 2(y - target) = {dL_dy}")
    
    # dL/dW2 = dL/dy * dy/dW2 = dL/dy * h1_relu
    dL_dW2 = dL_dy * h1_relu
    print(f"  dL/dW2 = dL/dy * h1_relu = {dL_dW2}")
    
    # dL/db2 = dL/dy
    dL_db2 = dL_dy
    print(f"  dL/db2 = dL/dy = {dL_db2}")
    
    # dL/dh1_relu = dL/dy * dy/dh1_relu = dL/dy * W2
    dL_dh1_relu = dL_dy * W2
    print(f"  dL/dh1_relu = dL/dy * W2 = {dL_dh1_relu}")
    
    # dL/dh1 = dL/dh1_relu * dh1_relu/dh1
    # ReLU 的导数: 1 if h1 > 0, else 0
    relu_grad = (h1 > 0).astype(float)
    dL_dh1 = dL_dh1_relu * relu_grad
    print(f"  dL/dh1 = dL/dh1_relu * ReLU_grad = {dL_dh1}")
    
    # dL/dW1 = dL/dh1 * dh1/dW1 = dL/dh1 * x
    dL_dW1 = dL_dh1.reshape(-1, 1) * x
    print(f"  dL/dW1 = {dL_dW1}")
    
    # dL/db1 = dL/dh1
    dL_db1 = dL_dh1
    print(f"  dL/db1 = {dL_db1}")
    
    print("\n梯度下降更新 (lr=0.1):")
    lr = 0.1
    W1_new = W1 - lr * dL_dW1
    b1_new = b1 - lr * dL_db1
    W2_new = W2 - lr * dL_dW2
    b2_new = b2 - lr * dL_db2
    
    print(f"  W1 更新:\n{W1} → \n{W1_new}")
    print(f"  b1 更新: {b1} → {b1_new}")


# =============================================================================
# 梯度下降
# =============================================================================

def gradient_descent():
    """梯度下降优化演示"""
    print("\n【梯度下降优化】")
    
    # 目标函数: f(x) = x² (最小值在 x=0)
    def f(x):
        return x ** 2
    
    def grad_f(x):
        return 2 * x
    
    # 梯度下降
    x = 5.0  # 初始点
    lr = 0.1  # 学习率
    iterations = 20
    
    print(f"目标函数: f(x) = x²")
    print(f"初始点: x = {x}")
    print(f"学习率: lr = {lr}")
    print(f"\n梯度下降过程:")
    
    for i in range(iterations):
        grad = grad_f(x)
        x_new = x - lr * grad
        
        if i % 5 == 0 or i == iterations - 1:
            print(f"  迭代 {i:2d}: x = {x:8.4f}, f(x) = {f(x):8.4f}, grad = {grad:8.4f}")
        
        x = x_new
    
    print(f"\n最终结果: x = {x:.6f}, f(x) = {f(x):.6f}")
    
    # 学习率的影响
    print("\n学习率的影响:")
    for lr in [0.01, 0.1, 0.5, 1.0]:
        x = 5.0
        for _ in range(20):
            x = x - lr * grad_f(x)
        print(f"  lr={lr:4.2f}: 最终 x = {x:8.4f}")


# =============================================================================
# 学习率调度
# =============================================================================

def learning_rate_schedule():
    """学习率调度策略"""
    print("\n【学习率调度】")
    
    # 模拟训练过程
    total_steps = 1000
    warmup_steps = 100
    base_lr = 0.001
    
    steps = np.arange(total_steps)
    
    # 1. 恒定学习率
    lr_constant = np.full_like(steps, base_lr, dtype=float)
    
    # 2. Warmup + 恒定
    lr_warmup = np.where(
        steps < warmup_steps,
        base_lr * steps / warmup_steps,
        base_lr
    )
    
    # 3. Warmup + Cosine Decay
    lr_cosine = np.where(
        steps < warmup_steps,
        base_lr * steps / warmup_steps,
        base_lr * 0.5 * (1 + np.cos(np.pi * (steps - warmup_steps) / (total_steps - warmup_steps)))
    )
    
    # 4. Warmup + Linear Decay
    lr_linear = np.where(
        steps < warmup_steps,
        base_lr * steps / warmup_steps,
        base_lr * (1 - (steps - warmup_steps) / (total_steps - warmup_steps))
    )
    
    print("学习率调度策略:")
    print("  1. 恒定学习率")
    print("  2. Warmup + 恒定")
    print("  3. Warmup + Cosine Decay")
    print("  4. Warmup + Linear Decay")
    
    # 与 nanochat 的关联
    print("\nnanochat 使用的学习率调度:")
    print("  - 预训练: Linear Warmup + Linear Decay")
    print("  - SFT: Warmup + Linear Decay + Final LR Fraction")


# =============================================================================
# 梯度问题与解决方案
# =============================================================================

def gradient_problems():
    """梯度消失与梯度爆炸问题"""
    print("\n【梯度问题】")
    
    # 梯度消失
    print("梯度消失:")
    print("  - 问题: 深层网络的梯度趋近于 0")
    print("  - 原因: 连续的小数相乘")
    print("  - 示例: 0.5^10 = {:.6f}".format(0.5 ** 10))
    print("  - 解决: ReLU 激活、残差连接、归一化")
    
    # 梯度爆炸
    print("\n梯度爆炸:")
    print("  - 问题: 梯度变得非常大")
    print("  - 原因: 连续的大数相乘")
    print("  - 示例: 2^10 = {}".format(2 ** 10))
    print("  - 解决: 梯度裁剪、权重初始化、归一化")
    
    # nanochat 的解决方案
    print("\nnanochat 的解决方案:")
    print("  1. RMS Normalization (每层归一化)")
    print("  2. 残差连接 (Residual Connection)")
    print("  3. 合理的权重初始化")
    print("  4. 学习率 Warmup")


# =============================================================================
# 综合演示
# =============================================================================

def demo():
    """运行所有微积分演示"""
    derivative_basics()
    partial_derivatives()
    chain_rule()
    backpropagation_demo()
    gradient_descent()
    learning_rate_schedule()
    gradient_problems()
    
    print("\n" + "=" * 60)
    print("微积分与梯度演示完成！")
    print("=" * 60)
    print("\n关键概念:")
    print("  1. 梯度指向函数增长最快的方向")
    print("  2. 链式法则是反向传播的数学基础")
    print("  3. 梯度下降通过负梯度方向更新参数")
    print("  4. 学习率调度影响训练效果")


if __name__ == "__main__":
    demo()
