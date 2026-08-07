"""
线性注意力机制 —— 逐步举例说明
================================
用 L=3, d=2 的超小矩阵，每一步的形状和数值都打印出来，
对比「标准 Softmax 注意力」和「线性注意力（前缀累加）」。
"""

import sys
import io
# 强制 UTF-8 输出，解决 Windows GBK 编码问题
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import numpy as np

np.set_printoptions(precision=4, suppress=True)

# ============================================================
# 0. 设定：序列长度 L=3，特征维度 d=2
# ============================================================
L = 3
d = 2

# 手动定义 Q, K, V（不用随机数，方便跟踪每一步）
Q = np.array([
    [1.0, 0.0],   # token_0 的 Query
    [0.0, 1.0],   # token_1 的 Query
    [1.0, 1.0],   # token_2 的 Query
])

K = np.array([
    [1.0, 0.0],   # token_0 的 Key
    [0.0, 1.0],   # token_1 的 Key
    [1.0, 1.0],   # token_2 的 Key
])

V = np.array([
    [2.0, 0.0],   # token_0 的 Value
    [0.0, 2.0],   # token_1 的 Value
    [1.0, 1.0],   # token_2 的 Value
])

print("=" * 60)
print("Q (L×d):")
print(Q)
print("\nK (L×d):")
print(K)
print("\nV (L×d):")
print(V)

# ============================================================
# 1. 标准 Softmax 注意力（带因果 mask）
# ============================================================
print("\n" + "=" * 60)
print("1. 标准 Softmax 注意力 (因果)")
print("=" * 60)

# Step 1: 计算注意力分数矩阵 S = Q @ K^T
S = Q @ K.T   # (L, d) @ (d, L) = (L, L)
print("\nStep 1 — 注意力分数矩阵 S = Q @ K^T (L×L):")
print(S)
print("  解读: S[i][j] = token_i 对 token_j 的\"原始关注度\"")
print("  e.g. S[2][0] = 1.0 表示 token_2 query 和 token_0 key 的内积为 1.0")

# Step 2: 缩放
S_scaled = S / np.sqrt(d)
print(f"\nStep 2 — 缩放 (÷√{d}):")
print(S_scaled)

# Step 3: 构造因果 mask（下三角为 0，上三角为 -inf）
causal_mask = np.tril(np.ones((L, L)))  # 下三角全是 1
causal_mask_inf = np.where(causal_mask == 1, 0.0, -np.inf)
print("\nStep 3 — 因果 mask (0=允许看, -inf=禁止看):")
print(causal_mask_inf)

# Step 4: mask + softmax
S_masked = S_scaled + causal_mask_inf
print("\nStep 4 — 加 mask 后:")
print(S_masked)


def softmax(x, axis=-1):
    e = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e / e.sum(axis=axis, keepdims=True)


attn_weights = softmax(S_masked, axis=-1)
print("\nStep 5 — softmax 后的注意力权重 (每行是一个概率分布):")
print(attn_weights)
print("  解读: token_2 的权重 = [0.33, 0.33, 0.33]，均匀关注前三个 token")

# Step 6: 加权求和
Y_softmax = attn_weights @ V   # (L, L) @ (L, d) = (L, d)
print("\nStep 6 — 输出 Y = attn_weights @ V (L×d):")
print(Y_softmax)
print("  解读: 每行是对 V 的加权平均，权重由注意力分布决定")

# ============================================================
# 2. 线性注意力 —— 核心公式推导
# ============================================================
print("\n" + "=" * 60)
print("2. 线性注意力 —— 公式推导")
print("=" * 60)

print("""
标准注意力的核心是 softmax(Q @ K^T / √d) @ V
            ↑ 这个 (L×L) 矩阵是 O(L²) 的根源

线性注意力的思路:
  把 softmax(Q @ K^T) 近似为 φ(Q) @ φ(K)^T
  其中 φ 是一个「核函数」(kernel function)，例如 ReLU

那么:
  Attention ≈ φ(Q) @ φ(K)^T @ V
            = φ(Q) @ [φ(K)^T @ V]        ← 改变乘法顺序！
                    ↑ 先算这个！

  [φ(K)^T @ V] 的形状是 (d, d)，不随 L 增长！
  所以复杂度从 O(L²·d) 降到 O(L·d²)
""")


# ============================================================
# 3. 线性注意力 —— 逐步数值计算
# ============================================================
print("=" * 60)
print("3. 线性注意力 —— 逐步数值计算")
print("=" * 60)

# 核函数：ReLU
def phi(x):
    return np.maximum(x, 0)

# 对 Q, K, V 应用核函数
Q_phi = phi(Q)
K_phi = phi(K)
# V 不需要核函数变换

print("\nφ(Q) (L×d):")
print(Q_phi)
print("\nφ(K) (L×d):")
print(K_phi)
print("\nV (L×d):")
print(V)

# ----------------------------------------------------------
# 方式 A：全局计算（错误：偷看未来，仅用于教学）
# ----------------------------------------------------------
print("\n" + "-" * 40)
print("方式 A: 全局计算 (包含未来信息, 非因果)")
print("-" * 40)

KV_all = K_phi.T @ V   # (d, L) @ (L, d) = (d, d)
print("\nKV_all = φ(K)^T @ V  形状 (d×d):")
print(KV_all)
print("  解读: 这是把整条序列所有 token 的 K^T V 一次求和，包含了未来信息")

# 归一化因子（也是全局的）
Z_all = K_phi.sum(axis=0, keepdims=True).T   # (d, 1)
print("\nZ_all = Σ φ(K) 沿序列维度求和  形状 (d×1):")
print(Z_all)

# 分子: φ(Q) @ KV_all  形状 (L, d)
Numerator_all = Q_phi @ KV_all
print("\n分子 Numerator = φ(Q) @ KV_all  形状 (L×d):")
print(Numerator_all)

# 分母: φ(Q) @ Z_all  形状 (L, 1)
Denominator_all = Q_phi @ Z_all
print("\n分母 Denominator = φ(Q) @ Z_all  形状 (L×1):")
print(Denominator_all)

Y_global = Numerator_all / Denominator_all
print("\n输出 Y_global (逐元素除)  形状 (L×d):")
print(Y_global)
print("  ⚠️ token_0 能看到未来 token_1 和 token_2 的信息！")

# ----------------------------------------------------------
# 方式 B：前缀累加（正确：因果，自回归推理专用）
# ----------------------------------------------------------
print("\n" + "-" * 40)
print("方式 B: 前缀累加 (因果, 只看历史)")
print("-" * 40)

# 初始化累加器
S_cum = np.zeros((d, d))   # 累积 φ(K_i)^T @ V_i
Z_cum = np.zeros((d, 1))   # 累积 φ(K_i)

print(f"初始: S = 0 矩阵({d}×{d}), Z = 0 向量({d}×1)")

Y_causal = []

for i in range(L):
    # 取出第 i 个 token 的向量，转为列向量 (d, 1)
    ki = K_phi[i:i+1].T    # (d, 1)
    vi = V[i:i+1].T        # (d, 1)
    qi = Q_phi[i:i+1].T    # (d, 1)

    print(f"\n{'='*40}")
    print(f"第 {i} 步 (token_{i})")
    print(f"{'='*40}")
    print(f"  k_{i} = {ki.flatten()}")
    print(f"  v_{i} = {vi.flatten()}")
    print(f"  q_{i} = {qi.flatten()}")

    # 增量更新累加器（只加当前 token，不看未来）
    delta_S = ki @ vi.T   # (d, 1) @ (1, d) = (d, d)  ← 外积
    S_cum = S_cum + delta_S

    print(f"\n  k_{i} ⊗ v_{i} (外积) = ")
    print(f"  {delta_S}")
    print(f"\n  更新后 S (累积 K^T V):")
    print(f"  {S_cum}")

    Z_cum = Z_cum + ki
    print(f"\n  更新后 Z (累积 K):")
    print(f"  {Z_cum.flatten()}")

    # 计算当前 token 的输出
    # 分子: q_i^T @ S_cum，形状 (1, d)
    numerator = qi.T @ S_cum
    # 分母: q_i^T @ Z_cum，形状 (1, 1)
    denominator = qi.T @ Z_cum

    y_i = numerator / denominator   # (1, d)
    print(f"\n  分子 q_i^T @ S = {numerator.flatten()}")
    print(f"  分母 q_i^T @ Z = {denominator.flatten()}")
    print(f"  → y_{i} = {y_i.flatten()}")

    Y_causal.append(y_i.flatten())

Y_causal = np.array(Y_causal)

print("\n" + "-" * 40)
print("最终输出 (前缀累加):")
print(Y_causal)

# ============================================================
# 4. 三种方法对比
# ============================================================
print("\n" + "=" * 60)
print("4. 三种方法对比")
print("=" * 60)

print("\n标准 Softmax 注意力 (因果):")
print(Y_softmax)
print("\n线性注意力 — 全局计算 (含未来, 非因果):")
print(Y_global)
print("\n线性注意力 — 前缀累加 (因果):")
print(Y_causal)

print("\n差异: 全局线性 - 前缀累加线性 (未来信息泄露量):")
print(np.round(Y_global - Y_causal, 4))

# ============================================================
# 5. 复杂度对比总结
# ============================================================
print("\n" + "=" * 60)
print("5. 复杂度对比")
print("=" * 60)

print("""
┌──────────────────────┬──────────────────┬──────────────────────┐
│ 方法                 │ 每步计算复杂度    │ 总复杂度 (L步)       │
├──────────────────────┼──────────────────┼──────────────────────┤
│ 标准 Softmax 注意力  │ O(i·d)           │ O(L²·d)             │
│ 线性注意力(前缀累加) │ O(d²)            │ O(L·d²)             │
└──────────────────────┴──────────────────┴──────────────────────┘

关键洞察:
  - 标准注意力: 每步要重新计算整个 QK^T (i×i 矩阵)，总开销 ~L²
  - 线性注意力: 每步只需更新 (d×d) 的累加器 S，总开销 ~L·d²

  当 L >> d 时（比如 L=128K, d=128），线性注意力远快于标准注意力。
  当 d >> L 时（短序列高维度），线性注意力的优势不明显。

在实际 LLM 中：
  - d 通常为 64~128 (单头维度)
  - L 可以达到 128K 甚至 1M
  → 线性注意力在超长上下文场景下有巨大优势！

前缀累加在自回归推理中的意义:
  每生成一个新 token：
    1. 计算新 token 的 k, v
    2. S += k ⊗ v   (一个 (d,d) 的外积, O(d²))
    3. y = q^T @ S / (q^T @ Z)   (O(d²))
  不需要访问所有历史 token 的 KV Cache！
""")
