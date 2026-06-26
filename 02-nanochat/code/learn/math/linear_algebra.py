"""
线性代数基础模块

包含 LLM/Transformer 所需的线性代数知识：
- 向量操作
- 矩阵操作
- 矩阵乘法（Attention 的核心）
- 转置、逆矩阵
- 特征值与特征向量

与 nanochat 的关联：
- Embedding 层：向量表示
- Attention 机制：矩阵乘法 Q @ K^T
- 线性层：矩阵乘法 X @ W
"""

import numpy as np
from typing import Tuple, List


# =============================================================================
# 向量操作
# =============================================================================

def vector_basics():
    """向量基础操作演示"""
    print("【向量基础】")
    
    # 创建向量
    v1 = np.array([1, 2, 3])
    v2 = np.array([4, 5, 6])
    
    print(f"v1 = {v1}")
    print(f"v2 = {v2}")
    print(f"v1 形状: {v1.shape}")  # (3,)
    
    # 向量加法
    v_add = v1 + v2
    print(f"v1 + v2 = {v_add}")
    
    # 标量乘法
    scalar = 2
    v_scale = scalar * v1
    print(f"{scalar} * v1 = {v_scale}")
    
    # 点积（Dot Product）- Attention 的核心操作
    dot_product = np.dot(v1, v2)
    print(f"v1 · v2 = {dot_product}")
    # 手动计算: 1*4 + 2*5 + 3*6 = 4 + 10 + 18 = 32
    
    # 向量范数（Norm）
    l2_norm = np.linalg.norm(v1)
    print(f"||v1||2 = {l2_norm:.4f}")
    
    # 向量归一化
    v_normalized = v1 / l2_norm
    print(f"v1 归一化 = {v_normalized}")
    print(f"归一化后范数 = {np.linalg.norm(v_normalized):.4f}")
    
    return v1, v2


# =============================================================================
# 矩阵操作
# =============================================================================

def matrix_basics():
    """矩阵基础操作演示"""
    print("\n【矩阵基础】")
    
    # 创建矩阵
    A = np.array([[1, 2, 3],
                  [4, 5, 6]])  # 2x3 矩阵
    
    B = np.array([[7, 8],
                  [9, 10],
                  [11, 12]])  # 3x2 矩阵
    
    print(f"A (2x3) =\n{A}")
    print(f"B (3x2) =\n{B}")
    print(f"A 形状: {A.shape}")
    
    # 矩阵转置
    A_T = A.T
    print(f"\nA^T (3x2) =\n{A_T}")
    
    # 矩阵加法
    C = np.array([[1, 1, 1],
                  [1, 1, 1]])
    A_add = A + C
    print(f"\nA + C =\n{A_add}")
    
    # 逐元素乘法（Hadamard Product）
    A_hadamard = A * C
    print(f"\nA ⊙ C =\n{A_hadamard}")
    
    return A, B


# =============================================================================
# 矩阵乘法 - Attention 的核心
# =============================================================================

def matrix_multiplication():
    """矩阵乘法演示 - 这是 Attention 机制的核心"""
    print("\n【矩阵乘法 - Attention 核心】")
    
    # 矩阵乘法: C = A @ B
    A = np.array([[1, 2, 3],
                  [4, 5, 6]])  # 2x3
    
    B = np.array([[7, 8],
                  [9, 10],
                  [11, 12]])  # 3x2
    
    # 矩阵乘法
    C = A @ B  # 等价于 np.matmul(A, B)
    print(f"A (2x3) @ B (3x2) = C (2x2)")
    print(f"C =\n{C}")
    
    # 手动计算 C[0,0]:
    # C[0,0] = A[0,0]*B[0,0] + A[0,1]*B[1,0] + A[0,2]*B[2,0]
    #        = 1*7 + 2*9 + 3*11 = 7 + 18 + 33 = 58
    print(f"\n手动验证 C[0,0] = 1*7 + 2*9 + 3*11 = {1*7 + 2*9 + 3*11}")
    
    # Attention 中的矩阵乘法示例
    print("\n【Attention 中的矩阵乘法】")
    print("假设:")
    print("  Q (Query): [seq_len, d_model]")
    print("  K (Key):   [seq_len, d_model]")
    print("  V (Value): [seq_len, d_model]")
    print()
    print("Attention 计算:")
    print("  1. scores = Q @ K^T  // [seq_len, seq_len]")
    print("  2. weights = softmax(scores / sqrt(d_model))")
    print("  3. output = weights @ V  // [seq_len, d_model]")
    
    # 模拟 Attention 计算
    seq_len, d_model = 4, 8
    Q = np.random.randn(seq_len, d_model)
    K = np.random.randn(seq_len, d_model)
    V = np.random.randn(seq_len, d_model)
    
    # Step 1: 计算 scores
    scores = Q @ K.T  # [seq_len, seq_len]
    print(f"\nQ 形状: {Q.shape}")
    print(f"K 形状: {K.shape}")
    print(f"scores = Q @ K^T 形状: {scores.shape}")
    
    # Step 2: 缩放
    scale = np.sqrt(d_model)
    scores_scaled = scores / scale
    print(f"缩放因子 sqrt(d_model) = {scale:.2f}")
    
    return Q, K, V, scores_scaled


# =============================================================================
# Softmax - 注意力权重计算
# =============================================================================

def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """Softmax 函数 - 将 logits 转换为概率分布
    
    与 nanochat 的关联：
    - nanochat/gpt.py:472 - logits 的 softmax
    - Attention 权重计算
    """
    # 减去最大值防止数值溢出
    exp_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)


def attention_demo():
    """完整的 Attention 计算演示"""
    print("\n【完整 Attention 计算】")
    
    seq_len, d_model = 4, 8
    np.random.seed(42)
    
    Q = np.random.randn(seq_len, d_model)
    K = np.random.randn(seq_len, d_model)
    V = np.random.randn(seq_len, d_model)
    
    # Step 1: 计算注意力分数
    scores = Q @ K.T / np.sqrt(d_model)
    print(f"1. 分数矩阵 (scaled):\n{scores}")
    
    # Step 2: 应用 Softmax
    attention_weights = softmax(scores, axis=-1)
    print(f"\n2. 注意力权重 (softmax):\n{attention_weights}")
    print(f"   每行和: {attention_weights.sum(axis=-1)}")  # 应该都是 1
    
    # Step 3: 加权求和
    output = attention_weights @ V
    print(f"\n3. 输出 = weights @ V:")
    print(f"   输出形状: {output.shape}")
    
    return attention_weights, output


# =============================================================================
# 矩阵的逆
# =============================================================================

def matrix_inverse():
    """矩阵的逆 - 理解线性变换"""
    print("\n【矩阵的逆】")
    
    # 只有方阵才有逆矩阵
    A = np.array([[1, 2],
                  [3, 4]])
    
    print(f"A =\n{A}")
    
    # 计算逆矩阵
    try:
        A_inv = np.linalg.inv(A)
        print(f"\nA^(-1) =\n{A_inv}")
        
        # 验证: A @ A^(-1) = I
        I = A @ A_inv
        print(f"\nA @ A^(-1) =\n{I}")
        print(f"是否为单位矩阵: {np.allclose(I, np.eye(2))}")
    except np.linalg.LinAlgError:
        print("矩阵不可逆（奇异矩阵）")
    
    # 行列式
    det = np.linalg.det(A)
    print(f"\ndet(A) = {det}")


# =============================================================================
# 特征值与特征向量
# =============================================================================

def eigenvalues_eigenvectors():
    """特征值与特征向量 - PCA、降维的基础"""
    print("\n【特征值与特征向量】")
    
    # 对称矩阵
    A = np.array([[4, 2],
                  [2, 3]])
    
    print(f"A =\n{A}")
    
    # 计算特征值和特征向量
    eigenvalues, eigenvectors = np.linalg.eig(A)
    
    print(f"\n特征值: {eigenvalues}")
    print(f"特征向量:\n{eigenvectors}")
    
    # 验证: A @ v = λ * v
    for i in range(len(eigenvalues)):
        v = eigenvectors[:, i]
        λ = eigenvalues[i]
        Av = A @ v
        λv = λ * v
        print(f"\n验证第 {i+1} 个特征向量:")
        print(f"  A @ v = {Av}")
        print(f"  λ * v = {λv}")
        print(f"  是否相等: {np.allclose(Av, λv)}")


# =============================================================================
# 综合演示
# =============================================================================

def demo():
    """运行所有线性代数演示"""
    vector_basics()
    matrix_basics()
    matrix_multiplication()
    attention_demo()
    matrix_inverse()
    eigenvalues_eigenvectors()
    
    print("\n" + "=" * 60)
    print("线性代数基础演示完成！")
    print("=" * 60)
    print("\n关键概念:")
    print("  1. 向量是模型中 Token 的基本表示")
    print("  2. 矩阵乘法是 Attention 和线性层的核心")
    print("  3. Softmax 将分数转换为概率分布")
    print("  4. 转置操作在 Attention 中用于计算 Q @ K^T")


if __name__ == "__main__":
    demo()
