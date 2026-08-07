"""
线性注意力输出公式推导 —— 从头拆解
===================================
公式: y_i = (q_i^T @ S) / (q_i^T @ Z)

回答三个问题：
1. 这个公式是怎么来的？
2. 分子 q_i^T @ S 在算什么？
3. 分母 q_i^T @ Z 在算什么？（为什么需要归一化？）
"""

import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
import numpy as np

np.set_printoptions(precision=4, suppress=True)

d = 2
L = 3

Q = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
K = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
V = np.array([[2.0, 0.0], [0.0, 2.0], [1.0, 1.0]])

def phi(x): return np.maximum(x, 0)
def softmax(x, axis=-1):
    e = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e / e.sum(axis=axis, keepdims=True)

# ============================================================
# 第零步：回顾标准 Softmax 注意力公式
# ============================================================
print("=" * 60)
print("第零步：标准 Softmax 注意力 —— 作为参照")
print("=" * 60)

print("""
标准自回归注意力的第 i 步输出:

    y_i = sum_{j=0}^{i}  softmax( q_i . k_j / sqrt(d) )_j  .  v_j

    其中 softmax 那一项是「归一化后的注意力权重」，
    满足: sum_j softmax(...)_j = 1  (所有历史 token 的权重之和为 1)

用矩阵写就是:

    y_i = softmax( q_i @ K_{0:i}^T / sqrt(d) ) @ V_{0:i}
""")

q2 = Q[2:3]                # (1, d)
K_prefix = K[0:3]          # (3, d)
V_prefix = V[0:3]          # (3, d)

scores = q2 @ K_prefix.T / np.sqrt(d)   # (1, 3)
weights = softmax(scores)                # (1, 3)
y2_std = weights @ V_prefix              # (1, d)

print(f"以 token_2 为例:")
print(f"  q_2 = {q2.flatten()}")
print(f"  注意力分数: {scores.flatten()}")
print(f"  注意力权重 (和为1): {weights.flatten()}  sum={weights.sum():.4f}")
print(f"  输出 y_2 = {y2_std.flatten()}")
print(f"\n  即: y_2 = {weights[0,0]:.4f}*v_0 + {weights[0,1]:.4f}*v_1 + {weights[0,2]:.4f}*v_2")

# ============================================================
# 第一步：从 softmax 到核函数 —— 公式是怎么来的
# ============================================================
print("\n" + "=" * 60)
print("第一步：用核函数 phi 替代 softmax —— 公式的推导过程")
print("=" * 60)

print("""
标准注意力:
    y = softmax( Q @ K^T / sqrt(d) ) @ V

    展开 softmax:
    y_i = sum_j  [ exp(q_i·k_j) / sum_{j'} exp(q_i·k_{j'}) ]  ·  v_j

        分子 = sum_j exp(q_i·k_j) · v_j        <-- 每个 v_j 按 exp(相似度) 加权
        分母 = sum_j exp(q_i·k_j)              <-- 所有权重之和（归一化）

线性注意力:
    用核函数 phi 近似 exp，即 exp(q_i·k_j) ~ phi(q_i) · phi(k_j)

    代入:
    分子 ~ sum_j ( phi(q_i)·phi(k_j) ) · v_j   =  phi(q_i) · sum_j phi(k_j)(x)v_j   =  phi(q_i) · S
    分母 ~ sum_j  phi(q_i)·phi(k_j)             =  phi(q_i) · sum_j phi(k_j)         =  phi(q_i) · Z

    所以:
    y_i ~ [ phi(q_i) · S ]  /  [ phi(q_i) · Z ]
               ^^分子               ^^分母

这就是公式 y_i = (q_i^T @ S) / (q_i^T @ Z) 的来源！
""")

# ============================================================
# 第二步：分子 q_i^T @ S 在算什么
# ============================================================
print("=" * 60)
print("第二步：分子 numerator = q_i^T @ S 在算什么？")
print("=" * 60)

Q_phi = phi(Q)
K_phi = phi(K)

print(f"\nphi(Q) =\n{Q_phi}")
print(f"phi(K) =\n{K_phi}")
print(f"V =\n{V}")

# 只考虑前 3 个 token
S = K_phi[0:1].T @ V[0:1] + K_phi[1:2].T @ V[1:2] + K_phi[2:3].T @ V[2:3]
print("\nS = phi(k0)^T @ v0 + phi(k1)^T @ v1 + phi(k2)^T @ v2 =")
print(S)
print("S 含义: 所有历史 token 的「键值记忆」矩阵, 形状固定为 (d,d)")

# 分子
q2_phi = phi(Q[2:3])
numerator = q2_phi @ S
print(f"\n分子 = phi(q_2) @ S = {q2_phi.flatten()} @ S = {numerator.flatten()}")

# 分解来看
print(f"\n分解:")
for j in range(3):
    kj = K_phi[j:j+1]
    vj = V[j:j+1]
    sim = q2_phi @ kj.T
    contrib = sim * vj
    print(f"  j={j}: phi(q_2)·phi(k_{j}) = {sim.flatten()[0]:.4f},  贡献 = {sim.flatten()[0]:.4f} x {vj.flatten()} = {contrib.flatten()}")

print(f"\n  分子 = 贡献之和 = {numerator.flatten()}")
print(f"  解读: 分子 = sum_j (q·k_j) x v_j，即「相似度 x 值」的加权和（未归一化）")


# ============================================================
# 第三步：分母 q_i^T @ Z 在算什么
# ============================================================
print("\n" + "=" * 60)
print("第三步：分母 denominator = q_i^T @ Z 在算什么？")
print("=" * 60)

Z = K_phi[0:1].T + K_phi[1:2].T + K_phi[2:3].T   # (d, 1): 所有 phi(k) 之和
print(f"\nZ = phi(k0)^T + phi(k1)^T + phi(k2)^T  = {Z.flatten()}")
print("Z 含义: 所有历史 phi(k) 的向量之和，用于归一化")

denominator = q2_phi @ Z
print(f"\n分母 = phi(q_2) @ Z = {q2_phi.flatten()} · {Z.flatten()} = {denominator.flatten()[0]:.4f}")

# 分解
print(f"\n分解:")
denom_sum = 0
for j in range(3):
    kj = K_phi[j:j+1]
    sim = q2_phi @ kj.T
    denom_sum += sim.flatten()[0]
    print(f"  j={j}: phi(q_2)·phi(k_{j}) = {sim.flatten()[0]:.4f}")

print(f"\n  分母 = 相似度之和 = {denom_sum:.4f}")
print(f"  解读: 分母 = sum_j (q·k_j)，即所有「相似度」的总和")


# ============================================================
# 第四步：分子 / 分母 = 归一化
# ============================================================
print("\n" + "=" * 60)
print("第四步：分子 / 分母 -> 为什么这就对了？")
print("=" * 60)

y2_linear = numerator / denominator
print(f"\n  y_2 = 分子 / 分母 = {numerator.flatten()} / {denominator.flatten()[0]:.4f}")
print(f"       = {y2_linear.flatten()}")

print(f"\n对比:")
print(f"  标准 Softmax 注意力:  y_2 = {y2_std.flatten()}")
print(f"  线性注意力:          y_2 = {y2_linear.flatten()}")

print("""
关键在于:
  分子 = sum_j (q·k_j) x v_j     <-- 每个 v_j 按「原始相似度」加权
  分母 = sum_j (q·k_j)           <-- 所有相似度的总和

  分子 / 分母 = sum_j [ (q·k_j) / sum_{j'}(q·k_{j'}) ] x v_j
              = sum_j (归一化权重) x v_j

这和 softmax 的结构一模一样！
  softmax:  sum_j [ exp(q·k_j) / sum exp(q·k) ] x v_j
  线性:     sum_j [ phi(q)·phi(k_j) / sum(phi(q)·phi(k)) ] x v_j

区别只是: exp(·) 换成了 phi(·)·phi(·)
         (核函数的「乘法分解」替代指数函数的「非线性」)
""")

# ============================================================
# 第五步：用具体的数字走完整个过程
# ============================================================
print("=" * 60)
print("第五步：完整公式走一遍（token_2 为例，数值全部展开）")
print("=" * 60)

q2_arr = np.array([1.0, 1.0])  # phi(q_2)

print(f"""
给定 phi(q_2) = {q2_arr}
累加器 S = {S.flatten()}
归一化器 Z = {Z.flatten()}

步骤:
  1. 分子 = q_2^T @ S
     = [{q2_arr[0]}, {q2_arr[1]}] @ [[{S[0,0]}, {S[0,1]}],
                                      [{S[1,0]}, {S[1,1]}]]
     第0维: {q2_arr[0]}*{S[0,0]} + {q2_arr[1]}*{S[1,0]} = {q2_arr[0]*S[0,0] + q2_arr[1]*S[1,0]}
     第1维: {q2_arr[0]}*{S[0,1]} + {q2_arr[1]}*{S[1,1]} = {q2_arr[0]*S[0,1] + q2_arr[1]*S[1,1]}
     = {numerator.flatten()}

  2. 分母 = q_2^T @ Z
     = [{q2_arr[0]}, {q2_arr[1]}] · [{Z[0,0]}, {Z[1,0]}]
     = {q2_arr[0]}*{Z[0,0]} + {q2_arr[1]}*{Z[1,0]}
     = {denominator.flatten()[0]:.4f}

  3. y_2 = 分子 / 分母
     = [{numerator[0,0]:.4f}, {numerator[0,1]:.4f}] / {denominator[0,0]:.4f}
     = [{numerator[0,0]/denominator[0,0]:.4f}, {numerator[0,1]/denominator[0,0]:.4f}]
     = {y2_linear.flatten()}
""")

# ============================================================
# 第六步：前缀累加中公式的工作方式
# ============================================================
print("=" * 60)
print("第六步：前缀累加中，公式每一步在做什么？")
print("=" * 60)

S_cum = np.zeros((d, d))
Z_cum = np.zeros((d, 1))

for i in range(L):
    ki = K_phi[i:i+1].T    # (d, 1)
    vi = V[i:i+1].T        # (d, 1)
    qi = Q_phi[i:i+1].T    # (d, 1)

    S_cum = S_cum + ki @ vi.T
    Z_cum = Z_cum + ki

    num = qi.T @ S_cum
    den = qi.T @ Z_cum
    y_i = num / den

    print(f"\n--- 第 {i} 步 (token_{i}) ---")
    print(f"  phi(k_{i}) = {ki.flatten()}")
    print(f"  v_{i} = {vi.flatten()}")
    print(f"  phi(q_{i}) = {qi.flatten()}")
    print(f"  S 更新后 = {S_cum.flatten()}")
    print(f"  Z 更新后 = {Z_cum.flatten()}")
    print(f"  分子 q_i^T @ S = {num.flatten()}")
    print(f"  分母 q_i^T @ Z = [{den[0,0]:.4f}]")

    # 解读分母：相似度之和
    sim_sum = sum(qi.T @ K_phi[j:j+1].T for j in range(i+1))
    print(f"  分母 = sum_j q_i·k_j = {sim_sum.flatten()[0]:.4f}")

    print(f"  => y_{i} = 分子/分母 = {y_i.flatten()}")

print("""
总结公式:

    y_i = (q_i^T @ S_i) / (q_i^T @ Z_i)

    其中:
      S_i = sum_{j=0}^{i}  phi(k_j) (x) v_j        <-- 历史的「键值记忆」
      Z_i = sum_{j=0}^{i}  phi(k_j)                <-- 历史的「键之和」（用于归一化）

    分子 = sum_j (q_i · k_j) x v_j                  <-- 加权值之和
    分母 = sum_j (q_i · k_j)                        <-- 相似度之和

    相除 = sum_j [ (q_i·k_j) / sum(q_i·k) ] x v_j  <-- 归一化后的加权值

    这等价于: softmax 中的 exp(q·k) 被替换为 phi(q)·phi(k)
""")
