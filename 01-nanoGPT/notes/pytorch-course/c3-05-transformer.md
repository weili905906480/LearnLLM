# C3 第5章：Transformer 架构

> **C3 PyTorch: Advanced Architectures and Deployment** — DeepLearning.AI  
> 本文档覆盖：Self-Attention / 多头注意力 / 位置编码 / Transformer Block / 完整实现

---

## 核心概念速览

```
Transformer 的核心思想：
  让序列中的每个元素都能"看到"其他所有元素
  通过 Attention 机制动态决定关注谁

对比 RNN：
  RNN：信息必须一步步传递，距离远的依赖难以学习
  Transformer：任意两个位置直接计算注意力，无距离限制

          x1  x2  x3  x4
           ↓   ↓   ↓   ↓
RNN：  [h1→h2→h3→h4]     x1 想影响 h4，要传 3 步

Transformer：  ┌────────────┐
               │ Attention  │  每个位置直接和所有位置交互
               └────────────┘
```

---

## 1. Self-Attention（自注意力）

### 1.1 Q / K / V 的含义

```
三个角色的比喻（图书馆查资料）：

  Q（Query）：你的查询需求  "我要找深度学习的书"
  K（Key）：  每本书的标签  "本书内容：深度学习"
  V（Value）：书的实际内容  书里面的知识

计算步骤：
  1. Q 和每个 K 做匹配 → 得到相似度分数
  2. Softmax 归一化 → 注意力权重（加起来=1）
  3. 用权重对所有 V 加权求和 → 最终输出
```

### 1.2 单头 Self-Attention

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

def self_attention(Q, K, V, mask=None):
    """
    参数：
      Q: [B, T, d_k]  Query 矩阵
      K: [B, T, d_k]  Key   矩阵
      V: [B, T, d_v]  Value 矩阵
      mask: [B, T, T] 可选，用于 Causal（因果）掩码
    返回：
      out:  [B, T, d_v]  注意力输出
      attn: [B, T, T]    注意力权重矩阵
    """
    d_k = Q.size(-1)

    # Step 1: 计算注意力分数
    # Q @ K^T → [B, T, T]，除以 √d_k 防止 softmax 饱和
    scores = Q @ K.transpose(-2, -1) / math.sqrt(d_k)

    # Step 2: 应用掩码（因果注意力：只看自己和之前的位置）
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float('-inf'))

    # Step 3: Softmax → 注意力权重
    attn = F.softmax(scores, dim=-1)   # 每行和=1

    # Step 4: 加权求和 Value
    out = attn @ V   # [B, T, T] @ [B, T, d_v] = [B, T, d_v]

    return out, attn
```

```
注意力分数计算可视化（T=4 个 token）：

Q: [B, 4, d_k]   K^T: [B, d_k, 4]
  ┌──────────┐         ┌──────────────────┐
  │ q0 q1 q2 │    @    │ k0 k1 k2 k3      │
  │ q0 q1 q2 │         │ k0 k1 k2 k3      │
  │ q0 q1 q2 │         │ k0 k1 k2 k3      │
  │ q0 q1 q2 │         │ k0 k1 k2 k3      │
  └──────────┘         └──────────────────┘
      [B, 4, d_k]           [B, d_k, 4]

结果 scores: [B, 4, 4]
  ┌─────────────────┐
  │ s00 s01 s02 s03 │  ← token0 对所有位置的相似度
  │ s10 s11 s12 s13 │  ← token1 对所有位置的相似度
  │ s20 s21 s22 s23 │
  │ s30 s31 s32 s33 │
  └─────────────────┘

除以 √d_k 的原因：
  d_k 越大，点积值越大，Softmax 越趋近 one-hot
  梯度消失 → 除以 √d_k 归一化，保持梯度稳定
```

### 1.3 因果掩码（Causal Mask）

```python
def causal_mask(T):
    """
    生成下三角掩码，位置 i 只能看到 0~i 的 token
    用于语言模型（预测下一个词时不能看到未来）

    参数：T (int) 序列长度
    返回：[T, T] bool 矩阵
    """
    mask = torch.tril(torch.ones(T, T))
    return mask  # 1=可以看，0=不能看（会被填为 -inf）
```

```
T=5 时的因果掩码：

     pos: 0  1  2  3  4
  0  ┌───┬──┬──┬──┬──┐
     │ 1 │0 │0 │0 │0 │  ← pos0 只能看自己
  1  ├───┼──┼──┼──┼──┤
     │ 1 │1 │0 │0 │0 │  ← pos1 能看 0,1
  2  ├───┼──┼──┼──┼──┤
     │ 1 │1 │1 │0 │0 │  ← pos2 能看 0,1,2
  3  ├───┼──┼──┼──┼──┤
     │ 1 │1 │1 │1 │0 │
  4  ├───┼──┼──┼──┼──┤
     │ 1 │1 │1 │1 │1 │  ← pos4 能看所有
     └───┴──┴──┴──┴──┘

0 的位置填 -inf，Softmax 后概率为 0
→ 彻底阻止"看到未来"
```

---

## 2. 多头注意力（Multi-Head Attention）

### 2.1 为什么需要多头

```
单头注意力的局限：
  只能学习一种"关注方式"

多头注意力：
  同时用多个不同的 Q/K/V 投影矩阵
  每个"头"学习不同的关注模式

  Head 1：关注语法关系  （"主语" 关注 "动词"）
  Head 2：关注语义关系  （"代词" 关注 "指代对象"）
  Head 3：关注位置关系  （关注相邻的 token）
  ...

最后把所有头的结果拼接起来，用一个线性层整合
```

### 2.2 完整实现

```python
class MultiHeadAttention(nn.Module):
    """
    多头自注意力层

    参数：
      d_model (int)：输入/输出的特征维度（如 512）
      n_heads (int)：注意力头数（如 8）
      dropout (float)：注意力权重的 dropout 概率

    输入：
      x:    [B, T, d_model]  输入序列
      mask: [B, T, T]        可选，因果掩码或填充掩码

    输出：
      [B, T, d_model]  与输入同 shape
    """
    def __init__(self, d_model, n_heads, dropout=0.0):
        super().__init__()
        assert d_model % n_heads == 0, "d_model 必须能被 n_heads 整除"

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k     = d_model // n_heads   # 每个头的维度

        # Q / K / V / Output 的投影矩阵（合并为一个大矩阵，计算更高效）
        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        B, T, C = x.shape    # B=batch, T=seq_len, C=d_model

        # Step 1: 投影 + 拆分多头
        # [B, T, d_model] → [B, T, n_heads, d_k] → [B, n_heads, T, d_k]
        Q = self.W_q(x).view(B, T, self.n_heads, self.d_k).transpose(1, 2)
        K = self.W_k(x).view(B, T, self.n_heads, self.d_k).transpose(1, 2)
        V = self.W_v(x).view(B, T, self.n_heads, self.d_k).transpose(1, 2)
        # 此时 Q, K, V: [B, n_heads, T, d_k]

        # Step 2: 计算注意力分数
        scores = Q @ K.transpose(-2, -1) / math.sqrt(self.d_k)
        # scores: [B, n_heads, T, T]

        # Step 3: 应用掩码
        if mask is not None:
            # mask: [B, T, T] → 扩展为 [B, 1, T, T] 广播到所有头
            scores = scores.masked_fill(mask.unsqueeze(1) == 0, float('-inf'))

        # Step 4: Softmax
        attn = F.softmax(scores, dim=-1)   # [B, n_heads, T, T]
        attn = self.dropout(attn)

        # Step 5: 加权求和 Value
        out = attn @ V   # [B, n_heads, T, T] @ [B, n_heads, T, d_k]
                         # = [B, n_heads, T, d_k]

        # Step 6: 合并多头 + 输出投影
        # [B, n_heads, T, d_k] → [B, T, n_heads, d_k] → [B, T, d_model]
        out = out.transpose(1, 2).contiguous().view(B, T, self.d_model)
        out = self.W_o(out)   # [B, T, d_model]

        return out
```

```
多头拆分/合并 shape 变化：

输入 x: [B, T, d_model]   假设 B=2, T=10, d_model=512, n_heads=8, d_k=64

W_q(x):       [B, T, d_model]  = [2, 10, 512]
.view(...):   [B, T, n_heads, d_k] = [2, 10, 8, 64]
.transpose(): [B, n_heads, T, d_k] = [2, 8, 10, 64]   ← Q

Q @ K^T:      [2, 8, 10, 64] @ [2, 8, 64, 10] = [2, 8, 10, 10]  ← 注意力矩阵
attn @ V:     [2, 8, 10, 10] @ [2, 8, 10, 64] = [2, 8, 10, 64]

合并多头：
.transpose(): [2, 10, 8, 64]
.view():      [2, 10, 512]   ← 回到原始 d_model
W_o:          [2, 10, 512]   ← 最终输出
```

---

## 3. 位置编码（Positional Encoding）

```python
class PositionalEncoding(nn.Module):
    """
    正弦/余弦位置编码（原始 Transformer 论文方式）

    参数：
      d_model (int)：特征维度
      max_len (int)：支持的最大序列长度
      dropout (float)：dropout 概率

    输入：  x: [B, T, d_model]
    输出：  [B, T, d_model]  加上位置编码后的张量
    """
    def __init__(self, d_model, max_len=5000, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        # 预计算所有位置的编码：[max_len, d_model]
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1).float()       # [max_len, 1]
        div = torch.exp(
            torch.arange(0, d_model, 2).float()
            * (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(pos * div)   # 偶数维度用 sin
        pe[:, 1::2] = torch.cos(pos * div)   # 奇数维度用 cos

        pe = pe.unsqueeze(0)   # [1, max_len, d_model]，方便广播
        self.register_buffer('pe', pe)   # 不参与训练，但会保存到 state_dict

    def forward(self, x):
        # x: [B, T, d_model]
        x = x + self.pe[:, :x.size(1), :]   # 只取前 T 个位置的编码
        return self.dropout(x)
```

```
位置编码的公式：

  PE(pos, 2i)   = sin(pos / 10000^(2i/d))
  PE(pos, 2i+1) = cos(pos / 10000^(2i/d))

  pos：token 在序列中的位置（0, 1, 2, ...）
  i：  维度索引

为什么用 sin/cos？
  不同维度用不同频率的波，每个位置有唯一的"指纹"
  sin/cos 的组合可以用线性变换表示相对位置关系
  → 模型可以泛化到训练时没见过的序列长度

GPT-2（nanoGPT）的做法：用可学习的位置 Embedding
  wpe = nn.Embedding(block_size, n_embd)
  优点：更灵活；缺点：无法外推到更长序列
```

---

## 4. 前馈网络（FFN）

```python
class FeedForward(nn.Module):
    """
    Transformer 中的前馈网络（每个 token 独立处理）

    参数：
      d_model (int)：输入/输出维度
      d_ff    (int)：中间层维度，通常 = 4 × d_model
      dropout (float)：dropout 概率

    输入/输出：[B, T, d_model]（shape 不变，只改变值）
    """
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),    # 升维：d_model → d_ff
            nn.GELU(),                   # 非线性激活
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),    # 降维：d_ff → d_model
        )

    def forward(self, x):
        return self.net(x)
```

```
FFN 的作用：

  Attention 负责：token 之间的信息交换（谁关注谁）
  FFN    负责：每个 token 独立的非线性变换（存储和提取知识）

  研究表明：Transformer 的大量知识存储在 FFN 的权重里
  FFN ≈ 键值存储（Key-Value Memory）

shape 变化：
  [B, T, 512] → Linear → [B, T, 2048]（升维 4x）
               → GELU  → [B, T, 2048]
               → Linear → [B, T, 512]（降回原维度）
```

---

## 5. 完整 Transformer Block

```python
class TransformerBlock(nn.Module):
    """
    一个完整的 Transformer 解码器层（GPT 风格，Pre-Norm）

    参数：
      d_model  (int)：特征维度
      n_heads  (int)：注意力头数
      d_ff     (int)：FFN 中间维度（通常 4×d_model）
      dropout  (float)：dropout 概率

    输入/输出：[B, T, d_model]
    """
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        self.ln1  = nn.LayerNorm(d_model)
        self.attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.ln2  = nn.LayerNorm(d_model)
        self.ff   = FeedForward(d_model, d_ff, dropout)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # Pre-Norm + 残差（先归一化再计算，梯度更稳定）
        x = x + self.drop(self.attn(self.ln1(x), mask))   # Attention 子层
        x = x + self.drop(self.ff(self.ln2(x)))           # FFN 子层
        return x
```

```
Transformer Block 完整数据流：

输入 x: [B, T, d_model]
  │
  ├─── LayerNorm ──→ MultiHeadAttention ──→ Dropout ──→ + x ──→ 中间 x
  │                                                         ↑
  │                                                    残差连接
  │
  └─── LayerNorm ──→ FeedForward ──→ Dropout ──→ + 中间x ──→ 输出 x

输出 x: [B, T, d_model]   ← shape 完全不变

Pre-Norm vs Post-Norm：
  Post-Norm（原论文）：Attention → Add → LayerNorm
  Pre-Norm（GPT/现代LLM）：LayerNorm → Attention → Add
  Pre-Norm 训练更稳定，是现代 LLM 的标准做法
```

---

## 6. 完整 GPT 风格模型

```python
class GPT(nn.Module):
    """
    完整的 GPT 风格语言模型

    参数：
      vocab_size (int)：词表大小
      d_model    (int)：特征维度（如 512）
      n_heads    (int)：注意力头数（如 8）
      n_layers   (int)：Transformer Block 层数（如 6）
      max_len    (int)：最大序列长度
      dropout    (float)：dropout 概率
    """
    def __init__(self, vocab_size, d_model, n_heads,
                 n_layers, max_len, dropout=0.1):
        super().__init__()
        d_ff = 4 * d_model   # FFN 中间维度惯例：4倍

        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb   = nn.Embedding(max_len, d_model)   # 可学习位置编码
        self.drop      = nn.Dropout(dropout)

        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])
        self.ln_f    = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

        # 权重共享：lm_head 和 token_emb 共享权重（节省参数）
        self.lm_head.weight = self.token_emb.weight

    def forward(self, idx, targets=None):
        """
        参数：
          idx:     [B, T]  输入 token 索引
          targets: [B, T]  目标 token 索引（训练时传入，推理时不传）
        返回：
          logits: [B, T, vocab_size]
          loss:   标量（训练时），None（推理时）
        """
        B, T = idx.shape
        device = idx.device

        # Token Embedding + Position Embedding
        tok = self.token_emb(idx)                                # [B, T, d_model]
        pos = self.pos_emb(torch.arange(T, device=device))      # [T, d_model]
        x   = self.drop(tok + pos)                               # [B, T, d_model]

        # 生成因果掩码
        mask = torch.tril(torch.ones(T, T, device=device))      # [T, T]

        # 逐层 Transformer Block
        for block in self.blocks:
            x = block(x, mask)   # [B, T, d_model]

        # 最终 LayerNorm + 输出 logits
        x      = self.ln_f(x)               # [B, T, d_model]
        logits = self.lm_head(x)            # [B, T, vocab_size]

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),   # [B*T, vocab_size]
                targets.view(-1)                    # [B*T]
            )

        return logits, loss
```

```
完整数据流（B=4, T=128, d_model=512, n_heads=8, n_layers=6）：

输入 idx:       [4, 128]        token 索引

Token Emb:      [4, 128, 512]   每个 token 变成 512 维向量
Pos Emb:        [128, 512]   → 广播加到 token emb
x:              [4, 128, 512]

× 6 层 Block：
  LayerNorm:    [4, 128, 512]
  MHA:          [4, 128, 512]   8个头，每头 64 维
  + 残差:       [4, 128, 512]
  LayerNorm:    [4, 128, 512]
  FFN:          [4, 128, 512]   内部扩到 2048 再降回 512
  + 残差:       [4, 128, 512]

Final LN:       [4, 128, 512]
lm_head:        [4, 128, 50257]  logits（词表大小）

loss = CrossEntropy(logits, targets)
```

---

## 7. 推理：自回归生成文本

```python
@torch.no_grad()
def generate(model, idx, max_new_tokens, temperature=1.0, top_k=None):
    """
    自回归生成：每次生成一个 token，拼接到序列末尾

    参数：
      idx:            [B, T]  起始 token 序列
      max_new_tokens: int     要生成的 token 数量
      temperature:    float   >1 更随机，<1 更确定，=1 正常采样
      top_k:          int     只从概率最高的 k 个 token 中采样

    返回：[B, T + max_new_tokens]
    """
    model.eval()
    for _ in range(max_new_tokens):
        # 取最近的上下文（不超过 max_len）
        idx_cond = idx[:, -model.pos_emb.num_embeddings:]

        # 前向传播，只需要最后一个位置的 logits
        logits, _ = model(idx_cond)
        logits = logits[:, -1, :] / temperature   # [B, vocab_size]

        # Top-k 过滤（可选）
        if top_k is not None:
            v, _ = torch.topk(logits, top_k)
            logits[logits < v[:, [-1]]] = float('-inf')

        # 采样下一个 token
        probs = F.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)   # [B, 1]

        # 拼接到序列
        idx = torch.cat([idx, next_token], dim=1)   # [B, T+1]

    return idx
```

```
自回归生成过程（生成 4 个 token）：

初始: [hello, world]               T=2

Step1: model([hello, world])
       取最后位置 logits → 采样 → "!"
       序列: [hello, world, !]      T=3

Step2: model([hello, world, !])
       取最后位置 logits → 采样 → "I"
       序列: [hello, world, !, I]   T=4

...每步只生成 1 个 token，但利用了所有历史上下文
   通过 KV Cache 可以避免重复计算历史 token 的注意力
```

---

## 关键参数对比

```
不同规模的 Transformer 配置：

              d_model  n_heads  n_layers  d_ff   参数量
GPT-2 Small:   768      12       12       3072   124M
GPT-2 Medium:  1024     16       24       4096   345M
GPT-2 Large:   1280     20       36       5120   774M
GPT-2 XL:      1600     25       48       6400   1.5B
GPT-3:        12288     96       96      49152   175B

规律：
  d_ff = 4 × d_model（FFN 升维 4 倍）
  d_k  = d_model / n_heads（每个头的维度）
  参数量 ≈ 12 × n_layers × d_model²
```

---

## 常见问题速查

| 问题 | 解决方案 |
|------|---------|
| 注意力分数全是一样大 | 检查是否忘了除以 `√d_k`，或 d_k 太小 |
| 生成时看到了未来 token | 检查因果掩码是否正确（应为下三角矩阵） |
| 训练不稳定，loss 爆炸 | 用 Pre-Norm（LayerNorm 放在 Attention 前） |
| 序列超过训练长度 | 用正弦位置编码（可外推），或扩展 `max_len` |
| 多头 shape 报错 | 确认 `d_model % n_heads == 0` |
| 推理太慢 | 实现 KV Cache（复用历史 K/V，不重复计算） |

---

## 我的理解 / 疑问

<!-- 在这里记录学习笔记 -->
