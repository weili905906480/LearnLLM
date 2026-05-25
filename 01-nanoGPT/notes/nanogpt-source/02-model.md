# nanoGPT model.py 前向传播解析

> 源码：https://github.com/karpathy/nanoGPT/blob/master/model.py
> 约 300 行，实现完整 GPT 架构

---

## 类结构总览

```
GPT
 └── Block × N
      ├── CausalSelfAttention   注意力子层
      └── MLP                   前馈子层
```

## 数据流总览

```
输入 token ids: [B, T]
        ↓ Token Embedding + Position Embedding
[B, T, C]   C = n_embd = 768
        ↓ × N 个 Block
[B, T, C]
        ↓ LayerNorm (最终)
[B, T, C]
        ↓ lm_head (Linear)
[B, T, vocab_size]  ← logits
        ↓ cross_entropy
  scalar loss
```

- `B` = batch_size
- `T` = sequence length（最大 1024）
- `C` = n_embd（GPT-2 Small = 768）

---

## 1. GPTConfig

```python
@dataclass
class GPTConfig:
    block_size: int = 1024    # 最大序列长度
    vocab_size: int = 50304   # 词表大小（50257 向上补到 64 的倍数）
    n_layer:    int = 12      # Transformer Block 层数
    n_head:     int = 12      # 注意力头数
    n_embd:     int = 768     # 嵌入维度
    dropout:  float = 0.0
    bias:      bool = True
```

---

## 2. CausalSelfAttention（因果自注意力）

### 初始化

```python
self.c_attn = nn.Linear(n_embd, 3 * n_embd)   # Q/K/V 合并计算
self.c_proj = nn.Linear(n_embd, n_embd)        # 输出投影

# 因果掩码（下三角矩阵）
self.register_buffer("bias",
    torch.tril(torch.ones(block_size, block_size))
          .view(1, 1, block_size, block_size))
```

### 前向传播

```python
def forward(self, x):
    B, T, C = x.size()

    # Step 1: 计算 Q、K、V（一次线性变换，效率高）
    q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
    # q, k, v: [B, T, C]

    # Step 2: 拆分多头
    hs = C // self.n_head                              # head_size = 768/12 = 64
    k = k.view(B, T, self.n_head, hs).transpose(1, 2) # [B, nh, T, hs]
    q = q.view(B, T, self.n_head, hs).transpose(1, 2)
    v = v.view(B, T, self.n_head, hs).transpose(1, 2)

    # Step 3: 计算注意力
    if self.flash:
        # Flash Attention（PyTorch 2.0+，内存高效）
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
    else:
        # 标准注意力（手动实现）
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(hs))
        att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        y = att @ v    # [B, nh, T, hs]

    # Step 4: 合并多头
    y = y.transpose(1, 2).contiguous().view(B, T, C)  # [B, T, C]

    # Step 5: 输出投影
    y = self.c_proj(y)
    return y
```

### 注意力计算可视化

```
x: [B, T, C]
  ↓ c_attn (Linear C→3C)
split → Q[B,T,C]  K[B,T,C]  V[B,T,C]
  ↓ reshape + transpose
Q,K,V: [B, n_head=12, T, head_size=64]

att = Q @ K^T / sqrt(64)    [B, 12, T, T]
  ↓ masked_fill（未来位置 → -inf）
  ↓ softmax（每行归一化为概率）
y = att @ V                 [B, 12, T, 64]
  ↓ transpose + reshape
[B, T, 768]
  ↓ c_proj
[B, T, 768]
```

### 为什么除以 sqrt(head_size)？

```
不除：Q@K^T 量级 ≈ head_size = 64
     softmax 极端饱和 → 梯度消失

除以 sqrt(64)=8：值归一化到合理量级
```

### 因果掩码

```
位置 i 只能看到 0 ~ i 的 token：

[[1, 0, 0, 0, 0],
 [1, 1, 0, 0, 0],
 [1, 1, 1, 0, 0],
 [1, 1, 1, 1, 0],
 [1, 1, 1, 1, 1]]

0 的位置填 -inf → softmax 后概率为 0
```

---

## 3. MLP（前馈网络）

```python
class MLP(nn.Module):
    def __init__(self, config):
        self.c_fc   = nn.Linear(n_embd, 4 * n_embd)   # 升维 4x
        self.gelu   = nn.GELU()
        self.c_proj = nn.Linear(4 * n_embd, n_embd)   # 降回原维度
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.c_fc(x)     # [B, T, C] → [B, T, 4C]
        x = self.gelu(x)     # 非线性激活
        x = self.c_proj(x)   # [B, T, 4C] → [B, T, C]
        return self.dropout(x)
```

**分工：**
- Attention：token 之间的**信息交换**（谁关注谁）
- MLP：每个 token **独立**的非线性变换（知识存储在这里）

**GELU vs ReLU：**
```
ReLU:  max(0, x)    → x<0 时梯度=0（死神经元）
GELU:  x·Φ(x)      → x<0 时有小梯度，更平滑
```

---

## 4. Block（完整 Transformer 层）

```python
class Block(nn.Module):
    def forward(self, x):
        x = x + self.attn(self.ln_1(x))   # Pre-Norm + 残差
        x = x + self.mlp(self.ln_2(x))
        return x
```

### Pre-Norm vs Post-Norm

```
原始 Transformer (Post-Norm):
  x → Attention → Add → LayerNorm → ...

nanoGPT (Pre-Norm):
  x → LayerNorm → Attention → Add → ...
      ↑
  先 Norm 再计算，梯度可直接通过残差路径流回，训练更稳定
```

### 残差连接的意义

```
x_new = x + f(LayerNorm(x))

初始时 f ≈ 0 → x_new ≈ x
梯度可以"跳过" Block 直接流回早期层
→ 解决深层网络训练困难
```

---

## 5. GPT（顶层模型）

### 初始化

```python
self.transformer = nn.ModuleDict(dict(
    wte  = nn.Embedding(vocab_size, n_embd),   # Token Embedding
    wpe  = nn.Embedding(block_size, n_embd),   # Position Embedding
    drop = nn.Dropout(dropout),
    h    = nn.ModuleList([Block(config) for _ in range(n_layer)]),
    ln_f = nn.LayerNorm(n_embd),               # 最终 LayerNorm
))
self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)

# ⭐ 权重共享：lm_head 与 token embedding 共享权重
self.transformer.wte.weight = self.lm_head.weight
```

**权重共享（Weight Tying）：**
```
Token Embedding: vocab_size × n_embd  (50304 × 768 ≈ 3850 万参数)
LM Head:         n_embd × vocab_size  (共享，节省 3850 万参数)

好处：语义一致性 + 减少参数量
```

### 权重初始化

```python
# 默认：正态分布 std=0.02
torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

# 残差路径上的投影层：缩小初始化（GPT-2 原始论文技巧）
std = 0.02 / math.sqrt(2 * n_layer)
# 防止 N 层残差叠加后数值爆炸
```

### 前向传播

```python
def forward(self, idx, targets=None):
    B, T = idx.size()

    # Step 1: 位置索引
    pos = torch.arange(0, T, dtype=torch.long, device=device)

    # Step 2: Embedding
    tok_emb = self.transformer.wte(idx)   # [B, T, C] token 语义
    pos_emb = self.transformer.wpe(pos)   # [T, C]    位置信息
    x = self.transformer.drop(tok_emb + pos_emb)

    # Step 3: 逐层 Block
    for block in self.transformer.h:
        x = block(x)

    # Step 4: 最终 LayerNorm
    x = self.transformer.ln_f(x)

    # Step 5: 输出
    if targets is not None:
        # 训练：计算所有位置的 loss
        logits = self.lm_head(x)                     # [B, T, vocab_size]
        loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            targets.view(-1)
        )
    else:
        # 推理：只取最后一个位置（效率优化）
        logits = self.lm_head(x[:, [-1], :])         # [B, 1, vocab_size]
        loss = None

    return logits, loss
```

---

## 与原始 Transformer 的对比

| 方面 | 原始论文（2017） | nanoGPT |
|------|----------------|---------|
| 架构 | Encoder + Decoder | **仅 Decoder** |
| Norm 位置 | Post-Norm | **Pre-Norm** |
| 位置编码 | 正弦固定编码 | **可学习 Embedding** |
| 激活 | ReLU | **GELU** |
| 注意力 | 手动实现 | **Flash Attention** |
| 权重共享 | 无 | **lm_head ↔ wte** |

---

## 关键设计总结

| 设计 | 选择 | 理由 |
|------|------|------|
| Norm 位置 | Pre-Norm | 梯度流更稳定 |
| 激活函数 | GELU | 比 ReLU 更平滑 |
| 注意力 | Flash Attention | 节省显存，速度快 |
| 权重共享 | lm_head ↔ wte | 减少参数，语义一致 |
| 残差初始化 | 0.02/sqrt(2N) | 防止深层叠加爆炸 |
| 推理优化 | 只取最后一位 logits | 推理时不需要所有位置 |

---

## 我的理解 / 疑问

<!-- 在这里记录学习笔记 -->
