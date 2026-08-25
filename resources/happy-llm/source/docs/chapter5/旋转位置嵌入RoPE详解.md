# 旋转位置嵌入（RoPE）详细解释

> 基于 `k_model.py` 中的实现，逐函数解析旋转位置嵌入的原理与代码。

---

## 1. 数学背景：RoPE 的核心思想

RoPE（Rotary Position Embedding）的核心思想是：**通过旋转矩阵来编码位置信息，使得两个 token 之间的注意力分数只依赖于它们的相对位置**。

对于位置 $m$ 上的一个向量 $\mathbf{x} \in \mathbb{R}^d$（$d$ 是 head_dim），将其按维度两两分组，形成 $d/2$ 个 2D 向量对。对第 $i$ 对 $(x_{2i}, x_{2i+1})$，施加一个旋转角度 $\theta_i = m \cdot \omega_i$（其中 $\omega_i = \theta^{-2i/d}$），即：

$$
\begin{pmatrix} x_{2i}' \\ x_{2i+1}' \end{pmatrix} =
\begin{pmatrix} \cos(m\omega_i) & -\sin(m\omega_i) \\ \sin(m\omega_i) & \cos(m\omega_i) \end{pmatrix}
\begin{pmatrix} x_{2i} \\ x_{2i+1} \end{pmatrix}
$$

用复数形式表示更简洁：将 $(x_{2i}, x_{2i+1})$ 视为复数 $x_{2i} + i \cdot x_{2i+1}$，旋转等价于乘以 $e^{i m \omega_i} = \cos(m\omega_i) + i \sin(m\omega_i)$。

**关键性质**：位置 $m$ 和 $n$ 的旋转后的向量做内积，结果只依赖于相对位置 $m-n$，这正是我们希望的位置编码性质。

---

## 2. `precompute_freqs_cis`（第 70-82 行）— 预计算频率表

```python
def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
```

### 参数说明

| 参数 | 含义 |
|------|------|
| `dim` | 每个 head 的维度（`dim // n_heads`） |
| `end` | 最大序列长度（`max_seq_len`） |
| `theta` | 基础频率参数，默认 10000.0 |

### 参数详细解释

#### `dim` — 每个注意力头的维度

```python
# 调用时传入的是 head_dim
freqs_cos, freqs_sin = precompute_freqs_cis(
    self.args.dim // self.args.n_heads,  # ← 这就是 dim 参数
    self.args.max_seq_len
)
```

**含义**：`dim = 模型总维度 ÷ 注意力头数 = head_dim`。

例如模型配置 `dim=768, n_heads=16`，则 `head_dim = 768/16 = 48`。

**为什么是 head_dim 而不是模型总维度？** RoPE 是在每个注意力头内部独立施加的。每个头的 Q/K 向量维度是 `head_dim`，旋转操作需要按维度两两分组（相邻两个元素视为一个"复数对"），所以总共产生 `head_dim / 2` 个"维度对"，每个维度对被分配一个不同的频率 $\omega_i$。

**频率的分配策略**：`torch.arange(0, dim, 2)` 生成 `[0, 2, 4, ..., dim-2]`，除以 `dim` 后得到归一化的指数 `[0/dim, 2/dim, 4/dim, ..., (dim-2)/dim]`，范围是 $[0, 1)$。这确保每个维度对获得从高到低的不同频率。

#### `end` — 最大序列长度

```python
t = torch.arange(end)  # [0, 1, 2, ..., end-1]
freqs = torch.outer(t, freqs)  # (end, dim//2)
```

**含义**：预计算的最大位置数，对应 `max_seq_len`。

**为什么需要预计算所有位置？** RoPE 需要为每个位置 $m \in [0, \text{seqlen})$ 计算 $\cos(m \cdot \omega_i)$ 和 $\sin(m \cdot \omega_i)$。与其每次前向传播时动态计算，不如一次性预计算 $m=0$ 到 $m=\text{max\_seq\_len}-1$ 的所有位置的频率值，然后根据实际序列长度截取：

```python
# 前向传播时截取
freqs_cos = self.freqs_cos[:seqlen]
freqs_sin = self.freqs_sin[:seqlen]
```

这样做的好处：
- 避免每次前向传播都重新计算三角函数（三角函数计算较昂贵）
- 作为 buffer 注册后随模型移动到 GPU，零额外开销
- `persistent=False` 表示不存入 `state_dict`，不占存储空间

外积 `torch.outer(t, freqs)` 的结果形状为 `(max_seq_len, head_dim//2)`，第 `[m][i]` 个元素 = $m \cdot \omega_i$，即位置 $m$ 在第 $i$ 个维度对上的旋转角度。

#### `theta` — 基础频率参数

```python
def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
```

**含义**：控制频率范围的基准值，默认 10000.0。

**频率计算公式**：

$$\omega_i = \theta^{-2i/d}, \quad i = 0, 1, 2, \dots, \frac{d}{2}-1$$

其中 $d$ 是 `head_dim`（即参数 `dim`）。

**频率范围**（以 `theta=10000` 为例）：

| $i$ | $2i/d$ | $\omega_i$ |
|-----|--------|------------------------------|
| 0   | 0      | $10000^0 = 1$（最高频） |
| $d/4$ | 0.5 | $10000^{-0.5} \approx 0.01$ |
| $d/2-1$ | $\approx 1$ | $\approx 1/10000 = 0.0001$（最低频） |

- **低维度对（$i$ 小）** → $\omega$ 接近 1 → 旋转角度变化快 → 对近距离位置**敏感** → **捕捉局部细节**
- **高维度对（$i$ 大）** → $\omega$ 接近 $1/\theta$ → 旋转角度变化慢 → 对远距离位置**有区分度** → **捕捉长程依赖**

**为什么选择 10000？** 这是 Transformer 原始论文中正弦位置编码的默认值。它让频率在对数尺度上均匀分布，覆盖了从 $1$ 到 $1/10000$ 的范围。不同研究也在探索其他取值（如 LLaMA 系使用 10000，部分变体使用更大的值以获得更好的长度外推能力）。

**调整 theta 的影响**：
- **增大 theta**（如 1000000）→ 整体频率降低 → 旋转更慢 → 更利于外推到更长序列
- **减小 theta**（如 100）→ 整体频率升高 → 旋转更快 → 更注重短距离位置区分

#### 三参数协作的完整视角

```
theta=10000 决定频率范围
      │
      ▼
ω_i = θ^(-2i/dim)    ← dim 决定有多少个不同频率
      │
      ▼
角度矩阵 = outer(位置[0..end], ω)    ← end 决定预计算多少个位置
      │
      ▼
cos 表 + sin 表 → apply_rotary_emb() → 旋转 Q 和 K
```

三个参数的关系可以理解为：

- **`theta`** 设定了频率的"基准线"
- **`dim`** 决定了有多少个等间隔分布于对数尺度上的频率点
- **`end`** 决定了提前算好多少个位置的 cos/sin 值

---

### 关于 $i$ — 维度对的索引

在 RoPE 中，`head_dim` 维的向量被**相邻两两分组**，形成 `head_dim/2` 个"维度对"（也叫"复数对"）。`i` 就是这些维度对的编号，从 0 到 `head_dim/2 - 1`。

假设 `head_dim = 8`，则共有 `8/2 = 4` 个维度对：

```
原始向量:  [x0,  x1,  x2,  x3,  x4,  x5,  x6,  x7]
             │    │    │    │    │    │    │    │
维度对:     └─┬──┘    └─┬──┘    └─┬──┘    └─┬──┘
             i=0       i=1       i=2       i=3
```

| `i` | 维度对 | 视为复数 | 频率 $\omega_i$（$\theta=10000$） |
|-----|--------|---------|----------------------------------|
| 0   | $(x_0, x_1)$ | $x_0 + i \cdot x_1$ | $10000^{-0/8} = 1$ |
| 1   | $(x_2, x_3)$ | $x_2 + i \cdot x_3$ | $10000^{-2/8} \approx 0.1$ |
| 2   | $(x_4, x_5)$ | $x_4 + i \cdot x_5$ | $10000^{-4/8} = 0.01$ |
| 3   | $(x_6, x_7)$ | $x_6 + i \cdot x_7$ | $10000^{-6/8} \approx 0.001$ |

代码中的 `torch.arange(0, dim, 2)` 就是在枚举 $2i$，其中 **$i = 0, 1, 2, 3$**。

**`i` 的物理意义**：

| `i` 的值 | 频率高低 | 擅长捕捉 |
|----------|---------|---------|
| **小**（$i=0,1,\dots$） | 高频率，旋转快 | **近距离位置关系**（相邻 token 的局部语法） |
| **大**（$i=\dots,d/2-1$） | 低频率，旋转慢 | **远距离位置关系**（跨段落的长程语义） |

不同的 $i$ 给不同的维度对分配不同的"旋转速度"，使得模型可以**同时关注不同尺度的位置关系**——这就是 RoPE 多频率设计的精妙之处。

---

### token 之间的距离是如何编码的

距离信息不是由一个单独的参数决定的，而是由**位置索引 `m` 和 `n` 的差值**自然产生的。具体体现在代码中的 `t`：

```python
t = torch.arange(end)  # [0, 1, 2, 3, ..., end-1]
```

**编码过程**：假设 token A 在位置 `m=5`，token B 在位置 `n=3`，它们的距离就是 `5 - 3 = 2`。

在 RoPE 中，Q 和 K 分别按各自的位置旋转：

- Q 在第 $i$ 个维度对旋转角度：$m \cdot \omega_i = 5 \omega_i$
- K 在第 $i$ 个维度对旋转角度：$n \cdot \omega_i = 3 \omega_i$

计算注意力分数 $QK^T$ 时，两者的内积自动依赖于 **旋转角度之差**：

$$(m \cdot \omega_i) - (n \cdot \omega_i) = (m - n) \cdot \omega_i$$

**距离 `m-n` 自然出现了**，不需要额外参数。

**`theta` 的作用：控制距离的"尺度感知"**。虽然位置索引直接给出距离，但不同的 `theta` 会影响模型对"远"和"近"的敏感度：

```
theta=10000, 序列长度=512

位置差 1   × ω₀(=1.0)     → 旋转 1 弧度     → 很大的角度变化 → 清楚区分相邻token
位置差 1   × ω₃(=0.001)   → 旋转 0.001 弧度  → 几乎不变化    → 对近距离不敏感
位置差 100 × ω₃(=0.001)   → 旋转 0.1 弧度    → 才能区分出远距离token
```

三个参数的角色总结：

| 参数 | 作用 |
|------|------|
| **位置索引 `[0, 1, 2, ...]`** | 提供 token 之间的**绝对距离**（`m - n`） |
| **`theta`** | 控制频率范围，决定什么叫"近"什么叫"远" |
| **`dim`** | 控制有多少个不同的频率刻度，在不同尺度上同时感知距离 |

三者配合，让模型在**每一个维度对上都以不同的敏感度感知 token 之间的距离**。

---

### 逐步解析

**Step 1 — 计算每个维度对的频率 $\omega_i$**（第 73 行）：

```python
freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
```

- `torch.arange(0, dim, 2)` 生成 `[0, 2, 4, ..., dim-2]`
- 除以 `dim` 得到 `[0/dim, 2/dim, 4/dim, ..., (dim-2)/dim]`
- `theta ** (...)` 计算 $\theta^{2i/d}$，即 $\text{theta}^{0/d}, \text{theta}^{2/d}, \text{theta}^{4/d}, \dots$
- 取倒数得到 $\omega_i = \theta^{-2i/d}$，维度 `(dim//2,)`

频率从高到低变化：低维度对对应高频（$\omega$ 接近 1），高维度对对应低频（$\omega$ 接近 $1/\theta$）。

> **`torch.arange(0, dim, 2)[: (dim // 2)]` 详解**
>
> 逐层拆解这个表达式：
>
> **第 1 层 `torch.arange(0, dim, 2)` — 取偶数索引**：从 `0` 开始每次加 `2`，直到 `< dim` 为止。`dim = 8` 时得到 `[0, 2, 4, 6]`。步长是 2 是因为 RoPE 把 `head_dim` 两两配对成 `dim//2` 个「维度对」，每对 `(2i, 2i+1)` 共用一个频率，所以只需取每对的起始索引 `0, 2, 4, ..., dim-2`。
>
> **第 2 层 `dim // 2` — 维度对数量**：`//` 是整除（向下取整），`dim = 8` 时得到 `4`。
>
> **第 3 层 `[: (dim // 2)]` — 切片**：`arange(0, dim, 2)` 的元素个数其实是 `ceil(dim/2)`（向上取整），而我们要的是 `floor(dim/2)`。`dim` 为偶数时两者相等，切片是**空操作**；`dim` 为奇数时才真正起作用，砍掉落单的最后一个偶数索引：
>
> | dim | `arange(0, dim, 2)` | 元素个数 | `dim // 2` | 切片后 |
> |-----|---------------------|---------|-----------|--------|
> | **8（偶）** | `[0, 2, 4, 6]` | 4 | 4 | `[0, 2, 4, 6]`（无变化） |
> | **9（奇）** | `[0, 2, 4, 6, 8]` | 5 | 4 | `[0, 2, 4, 6]`（砍掉末尾 `8`） |
>
> 奇数时索引 `8` 没有配对的另一半，所以不该给它分配频率。这是一种**防御性写法**——实际工程中 `head_dim` 几乎总是偶数，切片几乎从不真正触发，但能保证任何 `dim` 下都得到严格 `dim//2` 个频率，避免后续 `reshape_for_broadcast` 形状对不上。

**Step 2 — 计算每个位置 × 每个频率的角度**（第 75-77 行）：

```python
t = torch.arange(end, device=freqs.device)  # [0, 1, 2, ..., end-1]
freqs = torch.outer(t, freqs).float()        # (end, dim//2)
```

外积结果 `freqs[m][i] = m * ω_i`，即位置 $m$ 在第 $i$ 个维度对上的旋转角度。

> **`torch.outer` 的直观理解**
>
> `torch.outer(a, b)` 计算两个一维向量的**外积**：若 `a` 形状为 `(n,)`、`b` 形状为 `(m,)`，结果是 `(n, m)` 的矩阵，第 `[i][j]` 个元素 = `a[i] * b[j]`。它把「`a` 的每个元素」和「`b` 的每个元素」两两相乘铺成矩阵。
>
> 在这里就是「每个位置」×「每个频率」的笛卡尔积式组合，一次算出所有位置在所有维度对上的角度。

**具体数字例子**：取 `dim = 8`（则 `dim//2 = 4`）、`end = 3`（位置 0、1、2）、`theta = 10000`。

第一步算出频率 `freqs`：

```
freqs = 1 / (10000 ^ [0/8, 2/8, 4/8, 6/8])
      = 1 / (10000 ^ [0, 0.25, 0.5, 0.75])
      = 1 / [1, 10, 100, 1000]
      = [1.0, 0.1, 0.01, 0.001]
```

位置向量 `t = [0, 1, 2]`。外积 `torch.outer(t, freqs)` 得到形状 `(3, 4)` 的矩阵：

| 位置 m \ 频率 i | ω₀=1.0 | ω₁=0.1 | ω₂=0.01 | ω₃=0.001 |
|---|---|---|---|---|
| **m=0** | 0×1.0 = **0** | 0×0.1 = **0** | 0×0.01 = **0** | 0×0.001 = **0** |
| **m=1** | 1×1.0 = **1** | 1×0.1 = **0.1** | 1×0.01 = **0.01** | 1×0.001 = **0.001** |
| **m=2** | 2×1.0 = **2** | 2×0.1 = **0.2** | 2×0.01 = **0.02** | 2×0.001 = **0.002** |

```
[[0.0,  0.0,   0.0,   0.0  ],
 [1.0,  0.1,   0.01,  0.001],
 [2.0,  0.2,   0.02,  0.002]]
```

- **每一行** = 一个位置的完整信息：该位置在 4 个维度对上的 4 个角度。
- **每一列** = 同一个频率在所有位置上的值（随位置线性增长）。

**Step 3 — 计算 cos 和 sin**（第 79-81 行）：

```python
freqs_cos = torch.cos(freqs)  # (end, dim//2)
freqs_sin = torch.sin(freqs)  # (end, dim//2)
```

这就是复数 $e^{i \cdot m \cdot \omega}$ 的实部和虚部，分别对应旋转矩阵中的 $\cos$ 和 $\sin$ 分量。

---

## 3. `reshape_for_broadcast`（第 85-95 行）— 形状对齐

```python
def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
```

在 `apply_rotary_emb` 中，`freqs_cos/freqs_sin` 的形状是 `(seq_len, dim//2)`，而 `xq_r/xq_i` 的形状是 `(batch, seq_len, n_heads, dim//2)`。

此函数将频率张量的形状调整为 `(1, seq_len, 1, dim//2)`，以便通过广播机制与 query/key 张量相乘：

```python
shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
# 对 (batch, seq_len, n_heads, dim//2)，得到 [1, seq_len, 1, dim//2]
return freqs_cis.view(shape)
```

广播后的效果：

```
freqs_cos: (1, seq_len, 1, dim//2)
xq_r:      (batch, seq_len, n_heads, dim//2)
→ 相乘时自动广播为 (batch, seq_len, n_heads, dim//2)
```

---

## 4. `apply_rotary_emb`（第 97-122 行）— 施加旋转

这是核心函数，对 query 和 key 施加旋转位置编码。

```python
def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cos: torch.Tensor,
    freqs_sin: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
```

### Step 1 — 分离实部和虚部（第 105-106 行）

```python
xq_r, xq_i = xq.float().reshape(xq.shape[:-1] + (-1, 2)).unbind(-1)
xk_r, xk_i = xk.float().reshape(xk.shape[:-1] + (-1, 2)).unbind(-1)
```

- 输入形状：`(batch, seq_len, n_heads, head_dim)`
- `.reshape(..., -1, 2)` 将最后维度按相邻两个元素分组：`(batch, seq_len, n_heads, head_dim//2, 2)`
- `.unbind(-1)` 沿最后一维拆分为两个张量，分别对应"实部"和"虚部"
- 结果：`xq_r, xq_i` 形状均为 `(batch, seq_len, n_heads, head_dim//2)`

### Step 2 — 广播频率张量（第 109-110 行）

```python
freqs_cos = reshape_for_broadcast(freqs_cos, xq_r)  # (1, seq_len, 1, head_dim//2)
freqs_sin = reshape_for_broadcast(freqs_sin, xq_r)
```

### Step 3 — 复数旋转（核心）（第 113-116 行）

```python
xq_out_r = xq_r * freqs_cos - xq_i * freqs_sin  # 实部：a*cos - b*sin
xq_out_i = xq_r * freqs_sin + xq_i * freqs_cos  # 虚部：a*sin + b*cos
xk_out_r = xk_r * freqs_cos - xk_i * freqs_sin
xk_out_i = xk_r * freqs_sin + xk_i * freqs_cos
```

这直接实现了复数乘法的旋转公式。将 $(x_r, x_i)$ 视为复数 $z = x_r + i \cdot x_i$，乘以 $e^{i\theta} = \cos\theta + i \sin\theta$：

$$z' = z \cdot e^{i\theta} = (x_r + i x_i)(\cos\theta + i \sin\theta)$$

$$= (x_r\cos\theta - x_i\sin\theta) + i(x_r\sin\theta + x_i\cos\theta)$$

这就是上面四行代码的数学含义。

### Step 4 — 合并回原始形状（第 119-120 行）

```python
xq_out = torch.stack([xq_out_r, xq_out_i], dim=-1).flatten(3)
xk_out = torch.stack([xk_out_r, xk_out_i], dim=-1).flatten(3)
```

- `torch.stack(..., dim=-1)` → `(batch, seq_len, n_heads, head_dim//2, 2)`
- `.flatten(3)` → `(batch, seq_len, n_heads, head_dim)`

最终将交错排列的实部/虚部重新展平，恢复原始维度。

```python
return xq_out.type_as(xq), xk_out.type_as(xk)
```

最后转回原始数据类型（如 bf16/fp16），保持与输入一致。

### 具体数字例子

取 `batch=1, seq_len=2`（两个 token），`n_heads=1, head_dim=4`（即 `head_dim//2 = 2` 个维度对）。

**输入 `xq`**（形状 `(1, 2, 1, 4)`）：

| 位置 | 4 个 head_dim 元素 |
|---|---|
| m=0 | `[1, 2, 3, 4]` |
| m=1 | `[5, 6, 7, 8]` |

**预计算的频率**（`theta=10000, head_dim=4` → 2 个频率）：

```
ω = [ω₀, ω₁] = [1/(10000^0), 1/(10000^0.5)] = [1.0, 0.01]
```

每个位置的角度 `m·ω`：

| 位置 | 角度 `[m·ω₀, m·ω₁]` | cos | sin |
|---|---|---|---|
| m=0 | `[0, 0]` | `[1, 1]` | `[0, 0]` |
| m=1 | `[1, 0.01]` | `[0.540, 1.000]` | `[0.842, 0.010]` |

**Step 1 — 分离实部和虚部**：`reshape(1,2,1,2,2).unbind(-1)` 把相邻两元素拆成 `(实部, 虚部)`：

| 位置 | 原始 | 实部 `xq_r` | 虚部 `xq_i` | 当作复数 |
|---|---|---|---|---|
| m=0 | `[1,2,3,4]` | `[1,3]` | `[2,4]` | `(1+2i), (3+4i)` |
| m=1 | `[5,6,7,8]` | `[5,7]` | `[6,8]` | `(5+6i), (7+8i)` |

**Step 2 — 广播频率**：`freqs_cos/sin` 从 `(2, 2)` 广播成 `(1, 2, 1, 2)`，每个位置拿到自己那一行角度。

**Step 3 — 复数旋转**：公式 `新实部 = x_r·cosθ − x_i·sinθ`、`新虚部 = x_r·sinθ + x_i·cosθ`。

- 位置 m=0：θ=0 → 实部虚部都不变（旋转角为 0）。
- 位置 m=1，维度对 0（复数 `5+6i`，θ=1 弧度）：

```
新实部 = 5·cos(1) − 6·sin(1) = 5×0.540 − 6×0.842 = 2.702 − 5.049 = −2.348
新虚部 = 5·sin(1) + 6·cos(1) = 5×0.842 + 6×0.540 = 4.208 + 3.242 = 7.449
→ 旋转后：−2.348 + 7.449i
```

- 位置 m=1，维度对 1（复数 `7+8i`，θ=0.01 弧度）：

```
新实部 = 7·cos(0.01) − 8·sin(0.01) = 7×1.000 − 8×0.010 = 6.920
新虚部 = 7·sin(0.01) + 8·cos(0.01) = 7×0.010 + 8×1.000 = 8.070
→ 旋转后：6.920 + 8.070i
```

**Step 4 — 合并回原形状**：`stack([新实部, 新虚部], dim=-1).flatten(3)` 把实部/虚部交错拼回 `head_dim`：

| 位置 | 旋转前 | 旋转后 |
|---|---|---|
| m=0 | `[1, 2, 3, 4]` | `[1, 2, 3, 4]`（不变） |
| m=1 | `[5, 6, 7, 8]` | `[−2.348, 7.449, 6.920, 8.070]` |

从这个例子能看出的关键点：

1. **位置 0 不旋转**：`m=0` 时 `m·ω = 0`，cos=1、sin=0，是恒等变换。
2. **不同维度对转速不同**：维度对 0（高频 `ω₀=1`）转 1 弧度 ≈ 57°，维度对 1（低频 `ω₁=0.01`）只转 0.01 弧度 ≈ 0.57°——即 RoPE「低维转得快、高维转得慢」的多尺度感知。
3. **只改方向、不改模长**：复数乘以 `e^{iθ}` 是纯旋转，`|z|` 保持不变。
4. **同一位置所有 head 共享同一组角度**：靠广播实现，避免重复存储。

---

## 5. 在模型中的调用流程

### 5.1 预计算阶段（`Transformer.__init__`，第 339-341 行）

```python
freqs_cos, freqs_sin = precompute_freqs_cis(
    self.args.dim // self.args.n_heads,  # head_dim
    self.args.max_seq_len                # 最大长度
)
self.register_buffer("freqs_cos", freqs_cos, persistent=False)
self.register_buffer("freqs_sin", freqs_sin, persistent=False)
```

在模型初始化时一次性预计算出所有位置（0 到 max_seq_len-1）的频率的 cos 和 sin 值，并注册为 buffer（不参与梯度计算，但随模型一起移动到 GPU）。

### 5.2 截取当前序列长度（`Transformer.forward`，第 425-426 行）

```python
freqs_cos = self.freqs_cos[:seqlen]
freqs_sin = self.freqs_sin[:seqlen]
```

每次前向传播时根据实际序列长度截取对应的频率表。

### 5.3 在注意力层中施加旋转（`Attention.forward`，第 194 行）

```python
xq, xk = apply_rotary_emb(xq, xk, freqs_cos, freqs_sin)
```

**注意**：RoPE 只作用于 Query 和 Key，不作用于 Value。这是因为位置信息只需要影响注意力分数的计算（$QK^T$），而 RoPE 恰好保证了 $Q_m K_n^T$ 的结果只依赖于相对位置 $m-n$。

---

## 6. 整体数据流图

```
precompute_freqs_cis(head_dim, max_seq_len)
    │
    ├── freqs_cos: (max_seq_len, head_dim//2)  → 注册为 buffer
    └── freqs_sin: (max_seq_len, head_dim//2)  → 注册为 buffer
            │
            │  [:seqlen]  截取实际序列长度
            ▼
    (seq_len, head_dim//2)
            │
            │  reshape_for_broadcast()
            ▼
    (1, seq_len, 1, head_dim//2)
            │
            │  apply_rotary_emb()
            ▼
    xq: (batch, seq_len, n_heads, head_dim)  →  reshape →  实部/虚部分离
    xk: (batch, seq_len, n_kv_heads, head_dim)  →  reshape →  实部/虚部分离
            │
            │  复数旋转: z' = z * e^{iθ}
            ▼
    旋转后的 xq, xk（形状不变）
            │
            ▼
    继续计算 Attention(Q, K, V)
```

---

## 7. RoPE 的优势总结

| 特性 | 说明 |
|------|------|
| **相对位置编码** | 注意力分数只依赖相对位置，天然适合外推到更长序列 |
| **不增加参数量** | 频率表预计算，无需学习参数 |
| **与 Attention 融合** | 旋转直接作用在 Q/K 上，与自注意力机制无缝结合 |
| **维度分组** | 不同维度对使用不同频率，同时捕捉短距离和长距离依赖 |
| **计算高效** | 预计算 + 广播，实际计算量很小 |
