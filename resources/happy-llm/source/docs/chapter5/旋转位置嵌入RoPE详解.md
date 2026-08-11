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

**Step 2 — 计算每个位置 × 每个频率的角度**（第 75-77 行）：

```python
t = torch.arange(end, device=freqs.device)  # [0, 1, 2, ..., end-1]
freqs = torch.outer(t, freqs).float()        # (end, dim//2)
```

外积结果 `freqs[m][i] = m * ω_i`，即位置 $m$ 在第 $i$ 个维度对上的旋转角度。

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
