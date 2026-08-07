"""
外积 k ⊗ v 的含义 —— 逐步拆解
==============================
回答两个核心问题：
1. k ⊗ v 外积到底在做什么？
2. 为什么要把外积累加成 S？
"""

import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
import numpy as np

np.set_printoptions(precision=4, suppress=True)

d = 2

# ============================================================
# 第一部分：外积的基本含义
# ============================================================
print("=" * 60)
print("第一部分：外积 k ⊗ v 就是把一对 (key, value) 「写入」一个矩阵")
print("=" * 60)

# 假设有两个 token
k0 = np.array([1.0, 2.0]).reshape(d, 1)   # token_0 的 key (列向量)
v0 = np.array([3.0, 5.0]).reshape(d, 1)   # token_0 的 value (列向量)

k1 = np.array([2.0, 1.0]).reshape(d, 1)   # token_1 的 key
v1 = np.array([4.0, 6.0]).reshape(d, 1)   # token_1 的 value

print(f"\nk0 = {k0.flatten()}    v0 = {v0.flatten()}")
print(f"k1 = {k1.flatten()}    v1 = {v1.flatten()}")

# 外积
outer0 = k0 @ v0.T   # (2,1) @ (1,2) = (2,2)
outer1 = k1 @ v1.T

print(f"\nk0 ⊗ v0 (外积) = k0 @ v0^T =\n{outer0}")
print(f"\nk1 ⊗ v1 (外积) = k1 @ v1^T =\n{outer1}")

# 关键验证：外积矩阵 + 查询 = 检索
print("\n" + "-" * 40)
print("关键性质验证：q^T @ (k ⊗ v) = (q^T @ k) * v")
print("-" * 40)

q = np.array([0.8, 0.6]).reshape(d, 1)  # 某个查询
print(f"\n查询 q = {q.flatten()}")

# 方式1：外积矩阵 × 查询
result_matrix = q.T @ outer0
print(f"\n方式1: q^T @ (k0 ⊗ v0) = {result_matrix.flatten()}")
print(f"        即：先有矩阵 outer0，再用 q 去「读」")

# 方式2：先算相似度，再缩放 value
similarity = q.T @ k0
result_direct = similarity * v0.T
print(f"\n方式2: (q^T @ k0) * v0^T = {similarity.flatten()[0]:.4f} * {v0.flatten()} = {result_direct.flatten()}")
print(f"        即：先算 q 和 k0 的匹配度，再用这个分数缩放 v0")

print(f"\n两种方式结果完全一样！这就是外积的核心性质：")
print(f"  q^T @ (k ⊗ v) = (q^T @ k) * v")
print(f"   ↑ 从矩阵读取      ↑ 先匹配再取值")

# ============================================================
# 第二部分：累加器 S 的含义——把多个键值对「合并」进一个矩阵
# ============================================================
print("\n" + "=" * 60)
print("第二部分：S = Σ(k_i ⊗ v_i) 就是把所有键值对合并成一个「记忆矩阵」")
print("=" * 60)

S = outer0 + outer1
print(f"\nS = k0⊗v0 + k1⊗v1 =\n{S}")
print(f"\n这个 (2×2) 矩阵 S「记住」了两个 token 的所有信息！")

# 验证：用同一个 q 去查询 S
print("\n" + "-" * 40)
print("验证：用同一 q 查询累加器 S")
print("-" * 40)

# 方式 A：用累加器 S 一次性查询
result_S = q.T @ S
print(f"\n方式A (累加器): q^T @ S = {result_S.flatten()}")

# 方式 B：分别查询每个外积再求和
result_separate = q.T @ outer0 + q.T @ outer1
print(f"方式B (分别查):  q^T @ outer0 + q^T @ outer1 = {result_separate.flatten()}")

print(f"\n两种方式完全一样！这就是累加器 S 的威力：")
print(f"  一次性查询 S = 分别查询每个外积再求和")

# 方式 C：标准注意力的做法
result_attn = (q.T @ k0) * v0.T + (q.T @ k1) * v1.T
print(f"方式C (标准注意力): (q·k0)*v0 + (q·k1)*v1 = {result_attn.flatten()}")

# ============================================================
# 第三部分：图解——为什么这很重要
# ============================================================
print("\n" + "=" * 60)
print("第三部分：为什么累加器 S 能加速推理？")
print("=" * 60)

print("""
标准注意力（每生成一个新 token 需要做的事）：
============================================
  已有 token: t0, t1, t2, ..., t_{n-1}
  新 token:   t_n

  Q_new = [q_n]                    (1 × d)
  K_past = [k0, k1, ..., k_{n-1}]  (n × d)  ← n 越来越大！
  V_past = [v0, v1, ..., v_{n-1}]  (n × d)

  attention = softmax(q_n @ K_past^T) @ V_past
            = softmax( (1×d) @ (d×n) ) @ (n×d)
            = (1×n) @ (n×d)                          ← 涉及 n×n 的中间量
            = (1×d)

  每步要重新遍历所有 n 个历史 token，n 越大越慢！
  KV Cache 也随 n 线性增长，显存不断膨胀。


线性注意力（每生成一个新 token 需要做的事）：
============================================
  维护一个固定的累加器 S (d×d)，无论 n 多大，S 的大小不变！

  S_n = S_{n-1} + k_n ⊗ v_n     ← 把新 token 的信息"写入" S
       (d×d)     (d×d)外积

  y_n = q_n^T @ S_n / (q_n^T @ Z_n)     ← 从 S "读取"结果
       (1×d)    (d×d)                    ← 只涉及 (d×d) 矩阵！

  关键：
  - S 只存一个 (d×d) 矩阵，不需要存 n 个 KV 对
  - 每步计算量固定为 O(d²)，不随序列长度 n 增长
  - 显存占用固定，不膨胀
""")

# ============================================================
# 第四部分：数值演示——累加过程一步步看
# ============================================================
print("=" * 60)
print("第四部分：数值演示——S 如何一步步「记住」所有 token")
print("=" * 60)

# 用 3 个 token 演示
K_list = [np.array([1.0, 0.0]).reshape(d, 1),
          np.array([0.0, 1.0]).reshape(d, 1),
          np.array([1.0, 1.0]).reshape(d, 1)]

V_list = [np.array([2.0, 0.0]).reshape(d, 1),
          np.array([0.0, 2.0]).reshape(d, 1),
          np.array([1.0, 1.0]).reshape(d, 1)]

Q_list = [np.array([1.0, 0.0]).reshape(d, 1),
          np.array([0.0, 1.0]).reshape(d, 1),
          np.array([1.0, 1.0]).reshape(d, 1)]

S = np.zeros((d, d))
Z = np.zeros((d, 1))

for i in range(3):
    k, v, q = K_list[i], V_list[i], Q_list[i]

    print(f"\n{'='*40}")
    print(f"第 {i} 步：处理 token_{i}")
    print(f"{'='*40}")
    print(f"  k_{i} = {k.flatten()}")
    print(f"  v_{i} = {v.flatten()}")
    print(f"  q_{i} = {q.flatten()}")

    # 外积：把当前 token 的 (key,value) 对写入矩阵
    outer = k @ v.T
    print(f"\n  k_{i} ⊗ v_{i} =\n{outer}")
    print(f"  解读: 这个矩阵编码了「如果 query 和 k_{i}=[{k[0,0]:.0f},{k[1,0]:.0f}] 匹配，就读取 v_{i}=[{v[0,0]:.0f},{v[1,0]:.0f}]」")

    # 累加
    S_before = S.copy()
    S = S + outer
    Z = Z + k

    print(f"\n  S_before =\n{S_before}")
    print(f"  S_after  = S_before + 外积 =\n{S}")

    # 读取
    result = q.T @ S
    norm = q.T @ Z
    y = result / norm

    print(f"\n  查询 q_{i}^T @ S = {result.flatten()}")
    print(f"  归一化 q_{i}^T @ Z = {norm.flatten()}")
    print(f"  输出 y_{i} = {y.flatten()}")

    # 验证：用标准注意力的方式分步算
    if i == 2:
        print(f"\n  --- 验证：把 S 分解回各个外积 ---")
        print(f"  S = k0⊗v0 + k1⊗v1 + k2⊗v2")
        result_decomposed = q.T @ (K_list[0] @ V_list[0].T) + \
                            q.T @ (K_list[1] @ V_list[1].T) + \
                            q.T @ (K_list[2] @ V_list[2].T)
        print(f"  q^T @ (k0⊗v0) + q^T @ (k1⊗v1) + q^T @ (k2⊗v2) = {result_decomposed.flatten()}")
        print(f"  q^T @ S (一步) = {result.flatten()}")
        print(f"  完全相同！累加器 S 等价于分别查询每个外积再求和。")

# ============================================================
# 第五部分：最终总结
# ============================================================
print("\n" + "=" * 60)
print("总结：外积 + 累加器 = 固定大小的「压缩记忆」")
print("=" * 60)

print("""
1. 外积 k ⊗ v 的含义：
   - 把一对 (key, value) 「写入」一个 (d×d) 的矩阵
   - 之后用任何 q 去乘这个矩阵，等价于「匹配度 × 值」
   - 数学恒等式：q^T @ (k ⊗ v) = (q^T @ k) * v

2. 累加器 S = Σ(k_i ⊗ v_i) 的含义：
   - 把多个 key-value 对「合并」进同一个 (d×d) 矩阵
   - 之后查询 S 等价于对每对 key-value 分别查询再求和
   - 数学恒等式：q^T @ Σ(k_i ⊗ v_i) = Σ (q^T @ k_i) * v_i
   - 这就是注意力输出的本质！（只是少了 softmax 归一化）

3. 为什么能加速推理：
   - 标准注意力：每步要存 n 个 KV 对，逐个做点积，O(n)
   - 线性注意力：只存一个 (d×d) 的 S，O(d²)，与 n 无关！
   - S 就是整个历史的「压缩版 KV Cache」

4. 代价（为什么不是所有模型都用）：
   - 用 ReLU 等核函数近似 softmax，表达能力弱于真正的 softmax
   - 当 d 很大时（如 128 头 × 64 维 = 8192），O(d²) 也不小
   - 实际模型中 d 通常为 64~128（单头），此时 d² 远小于长序列的 n
""")
