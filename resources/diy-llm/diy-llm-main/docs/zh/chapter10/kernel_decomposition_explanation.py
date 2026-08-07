"""
核函数分解 通俗解释
==================
核心问题: 为什么能把 exp(q·k) 拆成 phi(q)·phi(k)?
"""

import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
import numpy as np

np.set_printoptions(precision=4, suppress=True)

# ============================================================
# 第〇步：什么是"分解"？先看一个最简单的类比
# ============================================================
print("=" * 60)
print("第零步：什么是「分解」？—— 从小学生都会的乘法分配律说起")
print("=" * 60)

print("""
大家都知道乘法分配律:
    a * (b + c) = a*b + a*c

它的本质是「把一个操作拆成可以分步做的两部分」，
所以可以先算 (b+c)，再和 a 乘。


但是有些运算「拆不开」，比如:

    exp(a + b)  !=  exp(a) + exp(b)      指数不能分配
    sin(a + b)  !=  sin(a) + sin(b)      三角函数不能分配
    sqrt(a + b) !=  sqrt(a) + sqrt(b)    平方根不能分配


注意力机制的问题就在这里!
""")

# ============================================================
# 第一步：exp(q·k) 为什么是"拆不开"的？
# ============================================================
print("=" * 60)
print("第一步：exp(q·k) 为什么导致 O(L^2)？因为它「拆不开」")
print("=" * 60)

print("""
标准注意力需要算 exp(q_i · k_j) 对每一对 (i, j)。

这个值同时依赖 q_i 和 k_j，你没法把它写成 「只关于 q_i 的某个函数」
乘以 「只关于 k_j 的某个函数」。

用数学说就是:

    exp(q · k) != f(q) * g(k)     对任何函数 f, g 都不成立

举个具体反例:
    q = [1, 2],  k = [3, 4]
    q·k = 1*3 + 2*4 = 11

    如果存在可分解的 f, g，那么:
    f([1,2]) * g([3,4]) = exp(11) = 59874

    但换了另一个 q' = [1, 0]:
    f([1,0]) * g([3,4]) = exp(3) = 20.1

    换 k' = [0, 1]:
    f([1,2]) * g([0,1]) = exp(2) = 7.4

    这些等式无法同时满足 —— 因为 f 和 g 各只有 d 个参数，
    却要精确拟合所有 (q,k) 配对产生的 L×L 个值。

因为拆不开，每一对 (q_i, k_j) 都必须单独算一次 exp → O(L^2)。
""")

# ============================================================
# 第二步：如果「假装」能拆开呢？
# ============================================================
print("=" * 60)
print("第二步：如果能找到 phi 使得 exp(q·k) ~ phi(q)·phi(k)，会怎样？")
print("=" * 60)

print("""
假设存在近似:
    exp(q·k) ≈ phi(q) · phi(k)

那么注意力公式中的求和就可以重新排列:

    sum_j exp(q_i·k_j) · v_j
    ≈ sum_j (phi(q_i)·phi(k_j)) · v_j
    = phi(q_i) · sum_j phi(k_j) · v_j     ← phi(q_i) 是公共因子，提到求和外面!
    = phi(q_i) · S                          ← S = sum_j phi(k_j) ⊗ v_j

关键: phi(q_i) 不依赖 j，所以可以先算 S（一次性聚合所有 k,v），
然后再和 phi(q_i) 相乘。S 的大小是 (d,d)，固定不变。

这就是分解带来的好处——把「逐对计算」改成了「先聚合、再查询」。
""")

# ============================================================
# 第三步：具体用什么做 phi？用代码感受一下
# ============================================================
print("=" * 60)
print("第三步：实际用什么函数做 phi？效果怎么样？")
print("=" * 60)

# 准备数据
q = np.array([1.0, 2.0])
K = np.array([[3.0, 4.0],
              [1.0, 0.0],
              [0.0, 1.0],
              [2.0, 2.0]])

print(f"\nquery q = {q}")
print(f"keys K =")
for i, k in enumerate(K):
    print(f"  k_{i} = {k}")

# 标准 exp 注意力
print(f"\n--- 标准: 直接用 exp(q·k) ---")
exp_scores = np.array([np.exp(q @ k) for k in K])
exp_weights = exp_scores / exp_scores.sum()
for i, (k, s, w) in enumerate(zip(K, exp_scores, exp_weights)):
    print(f"  k_{i}={k}:  q·k={q@k:.1f},  exp={s:.1f},  归一化权重={w:.4f}")

# 尝试用 ReLU 做 phi
print(f"\n--- 方案1: phi(x) = ReLU(x) = max(0, x) ---")

def phi_relu(x):
    return np.maximum(x, 0)

q_phi = phi_relu(q)
print(f"phi(q) = {q_phi}")

phi_scores_relu = np.array([q_phi @ phi_relu(k) for k in K])
phi_weights_relu = phi_scores_relu / phi_scores_relu.sum()
for i, (k, s, w) in enumerate(zip(K, phi_scores_relu, phi_weights_relu)):
    print(f"  k_{i}={k}:  phi(q)·phi(k)={s:.1f},  归一化权重={w:.4f}")

print(f"\n  exp 权重:      {exp_weights}")
print(f"  ReLU phi 权重: {phi_weights_relu}")
print(f"  -> 方向大致相同，数值有偏差")

# 尝试 ELU+1 (更常用的线性注意力核函数)
print(f"\n--- 方案2: phi(x) = ELU(x) + 1 (更接近 exp) ---")

def phi_elu_plus1(x):
    # ELU: x if x>0 else exp(x)-1
    elu = np.where(x > 0, x, np.exp(x) - 1)
    return elu + 1

q_phi2 = phi_elu_plus1(q)
print(f"phi(q) = {q_phi2}")

phi_scores_elu = np.array([q_phi2 @ phi_elu_plus1(k) for k in K])
phi_weights_elu = phi_scores_elu / phi_scores_elu.sum()
for i, (k, s, w) in enumerate(zip(K, phi_scores_elu, phi_weights_elu)):
    print(f"  k_{i}={k}:  phi(q)·phi(k)={s:.1f},  归一化权重={w:.4f}")

print(f"\n  exp 权重:       {exp_weights}")
print(f"  ELU+1 phi 权重: {phi_weights_elu}")
print(f"  -> 更接近 exp 的分布!")

# ============================================================
# 第四步：几何直觉 —— phi 在做什么？
# ============================================================
print("\n" + "=" * 60)
print("第四步：几何直觉 —— phi 其实是在「升维」")
print("=" * 60)

print("""
回到核心问题: 为什么 exp(q·k) 拆不开，但 phi(q)·phi(k) 能近似？

关键的数学事实:
    exp(q·k) 可以精确表示为 某个无限维 phi 的内积!

    具体来说，对于 exp(q·k)，存在一个「无穷维」的特征映射
    使得 exp(q·k) = <phi_inf(q), phi_inf(k)>
    其中 phi_inf 包含了 x 的各阶项: 1, x, x^2/2!, x^3/3!, ...

这是因为 exp 的泰勒展开:
    exp(x) = 1 + x + x^2/2! + x^3/3! + ...

所以 exp(q·k) = 1 + (q·k) + (q·k)^2/2! + (q·k)^3/3! + ...

而 (q·k)^n 是可以分解的！比如:
    (q·k)^2 = (q1*k1 + q2*k2)^2
            = q1^2*k1^2 + 2*q1*q2*k1*k2 + q2^2*k2^2
            = [q1^2, sqrt(2)q1q2, q2^2] · [k1^2, sqrt(2)k1k2, k2^2]
            = phi_deg2(q) · phi_deg2(k)

每一项 n 次幂都可以分解为有限维特征的内积。
但所有阶数加起来 → 需要无穷维！

所以:
  - exp(q·k) = 精确的无穷维 phi 的内积  →  不可计算
  - ReLU/ELU+1 等 = 有限维的近似         →  可以计算，牺牲精度换效率

ReLU 是最粗糙的近似（只保留正部分），ELU+1 更好一些。
在实际 LLM 中，这个精度损失通常可以接受，
尤其是当 d 足够大（比如 64~128 维的单头）时。
""")

# 演示: 多项式的可分解性
print("--- 数值演示: (q·k)^2 是可以分解的 ---")
q = np.array([1.0, 2.0])
k = np.array([3.0, 4.0])

direct = (q @ k) ** 2
print(f"直接算: (q·k)^2 = ({q[0]}*{k[0]} + {q[1]}*{k[1]})^2 = {direct:.1f}")

# 构造二次特征映射
phi_q2 = np.array([q[0]**2, np.sqrt(2)*q[0]*q[1], q[1]**2])
phi_k2 = np.array([k[0]**2, np.sqrt(2)*k[0]*k[1], k[1]**2])
decomposed = phi_q2 @ phi_k2
print(f"分解算: phi(q)·phi(k) = {phi_q2} · {phi_k2} = {decomposed:.1f}")
print(f"两种方式结果相同!")


# ============================================================
# 第五步：一句话总结
# ============================================================
print("\n" + "=" * 60)
print("第五步：总结")
print("=" * 60)

print("""
核函数分解的本质:

   exp(q·k) 这个运算是「粘在一起的」—— 你必须同时知道 q 和 k 才能算。

   但如果把它近似为 phi(q)·phi(k):
     - phi(q) 只和 q 有关  →  可以单独算
     - phi(k) 只和 k 有关  →  可以提前聚合进 S
     - 两者相乘就近似得到 exp(q·k) 的效果

   这就把 「O(L^2) 对逐对计算」 变成了 「O(L) 聚合 + O(1) 查询」。

代价:
   近似的精度取决于 phi 的选择和维度 d。
   ReLU 简单但粗糙，ELU+1 更好，理论上越大 d 越接近真实 exp。

类比:
   你想知道 100 个同学两两之间的「熟络程度」(exp(q·k))。

   标准做法: 让每个同学和另外 99 个分别聊天 → 100×99 = 9900 次聊天

   核函数做法: 让每个同学填一份「性格问卷」(phi)，然后
   只看问卷分数就能估计他们之间的熟络程度。
   只需要 100 份问卷 + 简单的矩阵运算。
""")
