# 旋转位置编码 RoPE 详解（结合具体矩阵示例）

> 对应源码：`model/model_minimind.py`

RoPE（Rotary Position Embedding，旋转位置编码）的完整实现集中在三个地方：

- `precompute_freqs_cis`：[model_minimind.py:62-78](../model/model_minimind.py#L62-L78)
- `apply_rotary_pos_emb` + `rotate_half`：[model_minimind.py:80-84](../model/model_minimind.py#L80-L84)
- 调用链：`Attention.forward` 的 [model_minimind.py:118-119](../model/model_minimind.py#L118-L119)，以及 `MiniMindModel.forward` 的 [model_minimind.py:280](../model/model_minimind.py#L280)

本文从「思想 → 每行代码 → 具体矩阵」三层展开。

---

## 1. 核心思想（一句话）

传统绝对位置编码是把位置信息**加**到向量上（`x + pos_embed`）；RoPE 是把位置信息**旋转**到向量上——对第 `m` 个位置的向量，把它的每一对维度做一个角度为 `m·θ_i` 的二维旋转。关键性质是：

> 位置 `m` 的 query 与位置 `n` 的 key 做点积时，**结果只依赖相对位置 (n−m)**，而不是绝对位置。

所以 RoPE 是「用绝对位置的方式实现，却得到相对位置的语义」，还天然适配 KV cache（推理时逐 token 生成，位置编码可增量应用）。

---

## 2. `precompute_freqs_cis`：先算好 cos/sin 查表

```python
freqs = 1.0 / (rope_base ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
```

**L63** 计算每个「维度对」的旋转角频率（即每对维度转多快）：

$$θ_i = \text{base}^{-2i/\text{dim}},\quad i = 0,1,\dots,\frac{\text{dim}}{2}-1$$

- `torch.arange(0, dim, 2)` 得到 `[0, 2, 4, …]`，切片 `[: dim//2]` 后共 `dim/2` 个频率，对应 `dim/2` 对维度。
- `base`（本文件默认 `rope_theta = 1e6`）越大，高频维度的波长越短、转得越慢（长文本更稳定）。

```python
t = torch.arange(end, device=freqs.device)
freqs = torch.outer(t, freqs).float()
```

**L74-75**：`t` 是所有位置 `[0, 1, 2, …, end-1]`，`torch.outer` 得到相位角矩阵 `freqs[m, i] = m · θ_i`，形状 `[max_pos, dim/2]`，即「位置 × 频率」。

```python
freqs_cos = torch.cat([torch.cos(freqs), torch.cos(freqs)], dim=-1)
freqs_sin = torch.cat([torch.sin(freqs), torch.sin(freqs)], dim=-1)
```

**L76-77**：对相位角取 cos/sin，再把 `[dim/2]` 拼成 `[dim]`（前后两半相同）。最终 `cos`、`sin` 形状均为 `[max_pos, dim]`。

> 函数名里的 `cis` 来自复数的极坐标记法 `cos θ + i·sin θ = e^{iθ} = cis(θ)`。RoPE 本质是把每个维度对看成一个复数 `x_0 + i·x_1`，旋转就是乘 `e^{i·m·θ}`。

这两个表在 `MiniMindModel.__init__`（[L266-268](../model/model_minimind.py#L266-L268)）里被注册为 buffer（`persistent=False`，不存进 checkpoint），推理时直接切片，不用每次重算。

---

## 3. `apply_rotary_pos_emb`：用 `rotate_half` 完成旋转

```python
def rotate_half(x):
    return torch.cat((-x[..., x.shape[-1] // 2:], x[..., : x.shape[-1] // 2]), dim=-1)
```

对长度为 `d` 的向量，切成前后两半 `a`（前 `d/2`）和 `b`（后 `d/2`），返回 `[-b, a]`。

```python
q_embed = (q * cos) + (rotate_half(q) * sin)
```

这一行等价于对每个「维度对」做标准的二维旋转。为什么能这样写，下面用矩阵看最清楚。

---

## 4. 具体矩阵例子（dim = 4，base = 10000）

真实默认 `head_dim = 96`，这里为手算缩小到 `head_dim = 4`，并取 `base = 10000`（默认 `1e6`）让数字更直观。

### 4.1 频率 θ

```
arange(0, 4, 2)  → [0, 2]  → /4 → [0.0, 0.5]
base ** [0.0, 0.5] = [10000^0, 10000^0.5] = [1, 100]
1.0 / [1, 100]   = [1.0, 0.01]
```

所以 `θ_0 = 1.0`（第 0/2 维这对转得快），`θ_1 = 0.01`（第 1/3 维这对转得慢）。

### 4.2 cos/sin 表（位置 m = 0 和 m = 1）

| 位置 m | cos 向量（长度 4） | sin 向量（长度 4） |
|--------|------------------|------------------|
| m = 0 | `[1, 1, 1, 1]` | `[0, 0, 0, 0]` |
| m = 1 | `[0.5403, 0.99995, 0.5403, 0.99995]` | `[0.8415, 0.0100, 0.8415, 0.0100]` |

注意 m=0 时 sin 全为 0、cos 全为 1 → 位置 0 的向量**不旋转**（恒等变换），符合直觉。

### 4.3 对位置 m=1 的 query 做旋转

设某个 head 的 query 向量 `q = [1, 2, 3, 4]`（长度 4）。

```
rotate_half(q) = [-q[2:], q[:2]] = [-3, -4, 1, 2]

q * cos            = [1×0.5403, 2×0.99995, 3×0.5403, 4×0.99995]
                   = [0.5403,   1.9999,     1.6209,   3.9998 ]
rotate_half(q)*sin = [-3×0.8415, -4×0.0100, 1×0.8415, 2×0.0100]
                   = [-2.5245,   -0.0400,   0.8415,   0.0200 ]

q_embed = 两者相加 = [-1.9842, 1.9599, 2.4624, 4.0198]
```

### 4.4 对照标准二维旋转公式验证

维度对是**跨半区配对**的：`(q_0, q_2)` 一对、`(q_1, q_3)` 一对。

对 `(q_0, q_2) = (1, 3)`，角度 `θ_0 = 1`：

$$q_0' = 1\cdot\cos 1 - 3\cdot\sin 1 = 0.5403 - 2.5245 = -1.9842$$

$$q_2' = 1\cdot\sin 1 + 3\cdot\cos 1 = 0.8415 + 1.6209 = 2.4624$$

对 `(q_1, q_3) = (2, 4)`，角度 `θ_1 = 0.01`：

$$q_1' = 2\cos 0.01 - 4\sin 0.01 = 1.9599$$

$$q_3' = 2\sin 0.01 + 4\cos 0.01 = 4.0198$$

结果与 `q_embed` 完全一致 ✅

### 4.5 写成旋转矩阵 R₁

位置 m=1 的整个操作可写成一个 4×4 矩阵（代入数值）：

$$R_1 =
\begin{bmatrix}
\cos 1 & 0 & -\sin 1 & 0 \\
0 & \cos 0.01 & 0 & -\sin 0.01 \\
\sin 1 & 0 & \cos 1 & 0 \\
0 & \sin 0.01 & 0 & \cos 0.01
\end{bmatrix}
=
\begin{bmatrix}
0.5403 & 0 & -0.8415 & 0 \\
0 & 0.99995 & 0 & -0.0100 \\
0.8415 & 0 & 0.5403 & 0 \\
0 & 0.0100 & 0 & 0.99995
\end{bmatrix}$$

`q_embed = R₁ · q`。可见它是**分块对角**结构——每个维度对一个独立的 2×2 旋转块，这正是 RoPE 旋转矩阵的通用形式。`apply_rotary_pos_emb` 里的 `rotate_half` 技巧只是**用 element-wise 乘加替代显式矩阵乘法**，数值完全相同，但省掉了构建大矩阵、也避免了复数运算。

---

## 5. 几个要点补充

### 5.1 配对方式是「跨半区」，不是「相邻」

这里 `q_0↔q_2`、`q_1↔q_3` 配对（前半对后半），是 LLaMA 系的标准做法。另一种常见写法（GPT-NeoX）是 `q_0↔q_1`、`q_2↔q_3` 相邻配对。两者只是维度的排列，最终 attention 点积结果等价，但**不能混用**——cos/sin 表怎么拼，`rotate_half` 就必须怎么配。

### 5.2 `unsqueeze_dim=1` 是广播用的（L80）

`q` 形状是 `[batch, seq, heads, head_dim]`，`cos` 是 `[seq, dim]`。`cos.unsqueeze(1)` 变成 `[seq, 1, dim]`，再广播到 batch 和 heads 两个维度，让同一位置的所有 head 用同一套旋转角。

### 5.3 KV cache 时靠切片取位置（`MiniMindModel.forward` L280）

```python
start_pos = past_key_values[0][0].shape[1] if past_key_values[0] is not None else 0
position_embeddings = (self.freqs_cos[start_pos:start_pos+seq_length], ...)
```

生成第 2 个 token 时，`start_pos=1`，只取 cos/sin 表第 1 行给新 token——正是 RoPE「绝对位置编码可逐 token 增量应用」的优势。

### 5.4 YaRN 外推（L64-73，可选）

当 `inference_rope_scaling=True` 且序列长度超过 `original_max_position_embeddings`（2048）时启用，用于把训练时 2048 的窗口外推到更长的上下文。核心公式 `freqs * (1 - ramp + ramp/factor)`：对高频维度（波长短）做「插值」`freqs/factor`，对低频维度保持原样「外推」，`ramp` 是两者之间的线性过渡系数。

---

## 总结

一条链路串起来：

> **`precompute_freqs_cis` 一次性算好「位置 × 频率 → cos/sin」的查表，`apply_rotary_pos_emb` 用 `q*cos + rotate_half(q)*sin` 这个技巧，把每个 head 的 query/key 按各自位置做一次分块对角旋转，且点积只依赖相对位置。**
