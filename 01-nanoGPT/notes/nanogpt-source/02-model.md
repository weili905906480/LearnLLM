# model.py 逐行深度解析

> nanoGPT 的灵魂文件：**~300 行实现完整 GPT 模型**
>
> 参考：
> - [OpenAI GPT-2 官方实现](https://github.com/openai/gpt-2/blob/master/src/model.py)
> - [HuggingFace GPT-2 PyTorch 实现](https://github.com/huggingface/transformers/blob/main/src/transformers/models/gpt2/modeling_gpt2.py)

---

## 文件总览

```python
model.py 包含 6 个类：

1. LayerNorm           — 自定义 LayerNorm（支持关闭 bias）
2. CausalSelfAttention — 因果自注意力（核心中的核心）
3. MLP                 — 前馈网络（升维 → 激活 → 降维）
4. Block               — 一个 Transformer 层 = Attention + MLP
5. GPTConfig           — 配置数据类
6. GPT                 — 主模型类（组装一切）
```

它们之间的组合关系：

```
GPT
├── transformer (ModuleDict)
│   ├── wte: Embedding (token)
│   ├── wpe: Embedding (position)
│   ├── drop: Dropout
│   ├── h: ModuleList of Block × n_layer
│   │   └── Block
│   │       ├── ln_1: LayerNorm
│   │       ├── attn: CausalSelfAttention
│   │       ├── ln_2: LayerNorm
│   │       └── mlp: MLP
│   └── ln_f: LayerNorm (final)
└── lm_head: Linear (输出层，与 wte 共享权重)
```



---

## 一、依赖导入

```python
import math                    # sqrt, cos 等数学函数
import inspect                 # 检查函数签名（用于判断 fused AdamW 是否可用）
from dataclasses import dataclass  # 数据类装饰器

import torch
import torch.nn as nn          # 神经网络模块（Linear, Embedding, Dropout...）
from torch.nn import functional as F  # 函数式 API（softmax, cross_entropy...）
```

**设计选择**：只用 PyTorch 标准库，不依赖任何第三方模型库。

---

## 二、LayerNorm — 自定义层归一化

### 源码

```python
class LayerNorm(nn.Module):
    """LayerNorm but with an optional bias. PyTorch doesn't support simply bias=False"""

    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))      # γ 缩放参数，初始化为 1
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None  # β 偏移，可选

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)
```

### 数学公式

```
对每个 token 的嵌入向量 x = [x₁, x₂, ..., x_d]：

1. 计算均值：μ = (1/d) Σ xᵢ
2. 计算方差：σ² = (1/d) Σ (xᵢ - μ)²
3. 归一化：  x̂ᵢ = (xᵢ - μ) / √(σ² + ε)      ← ε=1e-5 防止除零
4. 仿射变换：yᵢ = γᵢ · x̂ᵢ + βᵢ              ← γ=weight, β=bias（可学习）
```

### 为什么自己写？

PyTorch 原生 `nn.LayerNorm` 不支持 `bias=False`（至少在写 nanoGPT 时）。
现代研究发现去掉 bias 稍好一些（RMSNorm 干脆连 mean 都不减），所以需要这个自定义版本。

### 形态变化

```
输入:  (B, T, n_embd)     如 (8, 1024, 768)
输出:  (B, T, n_embd)     形状不变，只是每个向量被归一化了

weight: (n_embd,)          如 (768,)
bias:   (n_embd,) or None
```

### 直觉

```
归一化前：有的维度值很大(100)，有的很小(0.001)
          → 梯度在不同维度差异巨大，训练不稳定

归一化后：所有维度拉到统一尺度（~N(0,1)）
          → 梯度更均匀，训练更稳定
```



---

## 三、CausalSelfAttention — 因果自注意力

### 这是什么？

注意力机制让模型在生成每个 token 时，能"回头看"前面所有 token，并决定哪些更重要。

**"Causal"**（因果）意味着：位置 i 只能看到 ≤i 的 token，不能偷看未来。

### 源码 — __init__

```python
class CausalSelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0

        # 一个 Linear 同时计算 Q, K, V（高效：只做一次矩阵乘法）
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        # 输出投影
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        # 正则化
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout

        # Flash Attention 检测
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
        if not self.flash:
            # 手动创建因果 mask（下三角矩阵）
            self.register_buffer("bias",
                torch.tril(torch.ones(config.block_size, config.block_size))
                     .view(1, 1, config.block_size, config.block_size))
```

### 参数形态

```
c_attn:  Linear(n_embd, 3 * n_embd)
         权重形状: (3*768, 768) = (2304, 768)
         一次前向把 x 投影成 Q, K, V 三份

c_proj:  Linear(n_embd, n_embd)
         权重形状: (768, 768)
         把多头拼接结果投影回原始维度
```

### 源码 — forward（核心）

```python
def forward(self, x):
    B, T, C = x.size()  # batch, seq_len, n_embd

    # ═══════════════════════════════════════════════════════
    # Step 1: 计算 Q, K, V
    # ═══════════════════════════════════════════════════════
    q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
    # x:        (B, T, C)
    # c_attn(x):(B, T, 3C)  → split → q,k,v 各 (B, T, C)

    # ═══════════════════════════════════════════════════════
    # Step 2: 多头 reshape
    # ═══════════════════════════════════════════════════════
    k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
    q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
    v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
    # 变换过程：
    # (B, T, C) → (B, T, n_head, head_dim) → (B, n_head, T, head_dim)
    # 例: (8, 1024, 768) → (8, 1024, 12, 64) → (8, 12, 1024, 64)

    # ═══════════════════════════════════════════════════════
    # Step 3: 注意力计算
    # ═══════════════════════════════════════════════════════
    if self.flash:
        # PyTorch 2.0+ Flash Attention（高效 CUDA 内核）
        y = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=None,
            dropout_p=self.dropout if self.training else 0,
            is_causal=True
        )
    else:
        # 手动实现（教学版，更清晰）
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        # (B,nh,T,hs) @ (B,nh,hs,T) → (B,nh,T,T)  注意力分数矩阵
        att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
        # 因果 mask：上三角填 -inf → softmax 后变 0
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)
        y = att @ v  # (B,nh,T,T) @ (B,nh,T,hs) → (B,nh,T,hs)

    # ═══════════════════════════════════════════════════════
    # Step 4: 多头合并 + 输出投影
    # ═══════════════════════════════════════════════════════
    y = y.transpose(1, 2).contiguous().view(B, T, C)
    # (B,nh,T,hs) → (B,T,nh,hs) → (B,T,C)  把所有头拼起来
    y = self.resid_dropout(self.c_proj(y))
    return y
```

### 注意力计算的数学

```
Attention(Q, K, V) = softmax(Q·K^T / √d_k) · V

其中：
  Q: 查询（"我在找什么？"）
  K: 键  （"我有什么内容？"）
  V: 值  （"我的实际信息"）
  d_k = head_dim = n_embd / n_head = 768/12 = 64

缩放因子 √d_k：
  防止点积值过大（d_k=64时，点积可能到几十甚至上百）
  → softmax 饱和 → 梯度消失
  除以 √64=8 后，值保持在合理范围
```

### 多头注意力图解（详细版）

#### 为什么 Q, K, V 要先"合起来"计算，再"分开"给各个头？

**答案：效率。**

```
方案 A（直觉但低效）：每个头各有一个独立的 Linear
  head_0: Q₀ = Linear_q0(x), K₀ = Linear_k0(x), V₀ = Linear_v0(x)
  head_1: Q₁ = Linear_q1(x), K₁ = Linear_k1(x), V₁ = Linear_v1(x)
  ...
  head_11: Q₁₁ = Linear_q11(x), ...
  → 需要 12×3 = 36 个 Linear 层！
  → 36 次矩阵乘法 → GPU 利用率低（每个矩阵太小）

方案 B（nanoGPT 实际做法）：一个大 Linear，算完再拆
  [Q_all; K_all; V_all] = 一个大Linear(x)    ← 只做 1 次矩阵乘法
  然后 split → reshape → 拆分给 12 个头

  本质上等价于方案 A，但：
  - 1 次大矩阵乘法 vs 36 次小矩阵乘法
  - GPU 更擅长大矩阵 → 硬件利用率高 2-5 倍
```

数学上完全等价的原因：

```
大 Linear 的权重矩阵 W_attn shape = (768, 2304)

实际上等于把 36 个小矩阵拼在一起：
W_attn = [ W_q0 | W_q1 | ... | W_q11 | W_k0 | ... | W_k11 | W_v0 | ... | W_v11 ]
            64      64          64       64          64       64          64

y = x @ W_attn  ←→  分别计算 x @ W_q0, x @ W_q1, ... 然后拼起来
```

---

#### view 和 transpose 的作用（用具体数字举例）

以 **B=2, T=4, n_embd=12, n_head=3, head_dim=4** 为例（缩小版便于理解）：

```python
# 假设 c_attn 计算后，q 的形状是：
q.shape = (2, 4, 12)
# 含义：2 个样本，每个 4 个 token，每个 token 有 12 维

# 我们要把 12 维拆给 3 个头，每个头分到 4 维
```

**Step 1: view — 把最后一维"切开"**

```python
q = q.view(2, 4, 3, 4)
# (B, T, n_head, head_dim) = (2, 4, 3, 4)
```

```
view 前：q[sample=0, token=0] = [a b c d | e f g h | i j k l]
                                  ←───12维的一整条────────────→

view 后：q[sample=0, token=0] = [[a b c d],   ← head 0 的 4 维
                                  [e f g h],   ← head 1 的 4 维
                                  [i j k l]]   ← head 2 的 4 维

本质：只是"重新解读"内存布局，把 12 维看成 3×4，没有移动任何数据。
```

**Step 2: transpose(1, 2) — 交换 T 和 n_head 维度**

```python
q = q.transpose(1, 2)
# (B, T, n_head, head_dim) → (B, n_head, T, head_dim)
# (2, 4, 3, 4)            → (2, 3, 4, 4)
```

```
transpose 前 (B, T, n_head, head_dim):
  样本0:
    token0: [head0=[a b c d], head1=[e f g h], head2=[i j k l]]
    token1: [head0=[m n o p], head1=[q r s t], head2=[u v w x]]
    token2: ...
    token3: ...

transpose 后 (B, n_head, T, head_dim):
  样本0:
    head0: [token0=[a b c d], token1=[m n o p], token2=..., token3=...]
    head1: [token0=[e f g h], token1=[q r s t], token2=..., token3=...]
    head2: [token0=[i j k l], token1=[u v w x], token2=..., token3=...]

本质：把"按 token 组织"变成"按 head 组织"
     每个 head 现在拥有所有 token 的信息 → 可以独立做注意力
```

**为什么要 transpose？**

```
注意力计算需要的格式：
  Q @ K^T = (T, head_dim) @ (head_dim, T) → (T, T)

如果不 transpose（保持 B, T, n_head, head_dim）：
  矩阵乘法的最后两维是 (n_head, head_dim) @ (head_dim, n_head)
  → 这不是我们想要的！

transpose 后（B, n_head, T, head_dim）：
  矩阵乘法的最后两维是 (T, head_dim) @ (head_dim, T) = (T, T)
  → 这正是每个 head 的注意力分数矩阵 ✓

GPU 的 batched matmul 会对前面的维度 (B, n_head) 并行
→ 所有样本、所有头同时计算 → 极快
```

---

#### 完整变换过程（真实数字 B=8, T=1024, n_embd=768, n_head=12）

```
输入 x: (8, 1024, 768)

Step 1: c_attn 计算
  x @ W_attn = (8, 1024, 768) @ (768, 2304) → (8, 1024, 2304)
                                                    一次矩阵乘

Step 2: split 成 Q, K, V
  (8, 1024, 2304) → split(768, dim=2) → q, k, v 各 (8, 1024, 768)

Step 3: view（拆成多头）
  q: (8, 1024, 768) → view(8, 1024, 12, 64) = (8, 1024, 12, 64)
     ← 768 = 12头 × 64维/头

Step 4: transpose（head 维提前）
  q: (8, 1024, 12, 64) → transpose(1,2) → (8, 12, 1024, 64)
     ← 现在格式是：每个 head 拥有 1024 个 token 的 64 维向量

Step 5: 注意力计算
  att = q @ k.T = (8, 12, 1024, 64) @ (8, 12, 64, 1024) → (8, 12, 1024, 1024)
                   ← 每个 head 一个 1024×1024 的注意力矩阵
  y = att @ v = (8, 12, 1024, 1024) @ (8, 12, 1024, 64) → (8, 12, 1024, 64)

Step 6: transpose 回来 + view 合并
  y: (8, 12, 1024, 64) → transpose(1,2) → (8, 1024, 12, 64)
  y: (8, 1024, 12, 64) → contiguous().view(8, 1024, 768)
     ← 12头 × 64维 = 768维，重新合并成一个向量

Step 7: 输出投影
  y @ W_proj = (8, 1024, 768) @ (768, 768) → (8, 1024, 768)
```

---

#### 为什么需要 Step 7 输出投影（c_proj）？

**核心问题**：如果没有 c_proj，12 个头的结果只是简单拼接，头与头之间互相不知道对方算了什么。

```
没有 c_proj 时：
  输出 = [head0的64维 | head1的64维 | ... | head11的64维]
  → 12 份独立结果简单堆叠，没有任何"交流"

有 c_proj 时：
  输出 = [h0|h1|...|h11] @ W_proj    (768×768 矩阵)
  → 输出的每一维 = 从所有 12 个头中加权融合信息
  → 头与头之间的发现被综合在一起
```

**类比**：

```
12 个侦探各自调查了案件的不同线索：
  侦探0：看了指纹 → 输出64维特征
  侦探1：看了监控 → 输出64维特征
  侦探2：看了证词 → 输出64维特征
  ...

没有 c_proj = 把 12 份报告订在一起交给法官
  → 法官（后续层）看到的是分裂的、未综合的信息

有 c_proj = 12 个侦探开会讨论，输出一份综合报告
  → 后续层拿到的是"已经综合过的"信息
```

**具体效果**：

```
假设：
  head 0 发现：位置3的token 是主语
  head 1 发现：位置7的token 是动词
  head 2 发现：位置3和7有强关联

c_proj 的 W_proj 学到：
  把 head0(主语) + head1(动词) + head2(关联)
  融合成："位置3是主语，它的动词在位置7"
  → 这个综合后的表示对下一层更有用
```

**从梯度角度看**：

```
反向传播时：
  c_proj 把 loss 的梯度"分发"回 12 个头
  W_proj 的权重决定了每个头对最终输出的贡献比例
  → 间接引导每个头学不同的关注模式

没有 c_proj：
  梯度按拼接位置直接回传
  head0 永远只影响前 64 维，head11 永远只影响后 64 维
  → 没有跨头梯度信号 → 各头无法协同
```

**参数量**：768 × 768 = 589,824，学的就是"如何把 12 个头的独立观察融合成统一表示"。

> **一句话**：c_proj 是多头注意力的"总结层"——没有它，多头只是"多个独立单头的拼接"，有了它，12 个头才能真正协同工作。

---

#### 为什么需要 .contiguous()？

```python
y = y.transpose(1, 2).contiguous().view(B, T, C)
```

```
transpose 只是改变了"如何解读内存"的元信息（stride），
并没有真正移动内存中的数据。

view 要求内存必须是连续的（contiguous）。
如果 transpose 后直接 view，会报错：
  RuntimeError: view size is not compatible with input tensor's
  size and stride (at least one dimension spans across two
  contiguous subspaces)

.contiguous() = 按照新的维度顺序，真正重新排列内存中的数据
               之后就可以安全地 view 了

代价：一次内存拷贝
      但相比矩阵乘法的开销，可以忽略不计
```

---

#### 多头注意力的直觉类比

```
类比：一篇文章，12 个人同时阅读

  Head 0 ("语法专家")：关注主语和动词的对应关系
    "猫  坐在  垫子  上"
     ↑───┘
     主语→动词

  Head 1 ("位置专家")：关注相邻 token 的关系
    "猫  坐在  垫子  上"
         ↑───┘
         相邻词依赖

  Head 2 ("语义专家")：关注远距离语义关联
    "那只昨天在花园里追蝴蝶的  猫  现在  坐在  垫子  上"
     ↑──────────────────────────────────────┘
     远距离共指

  ... 12 个头各有各的"关注点"
  
  最后：把 12 个头的观察结果拼起来 → 输出投影 → 综合所有视角的信息
```

每个头只用 64 维（而非 768 维）做注意力：
- **减少计算量**：768 维做一次注意力 vs 64 维做 12 次，FLOPs 相同但并行度更高
- **增加表达力**：12 个头可以学到不同模式，一个 768 维大头只能学一种模式

### 因果 Mask 可视化

```
对于 T=5 的序列：

     t=0  t=1  t=2  t=3  t=4
t=0 [ 1    0    0    0    0  ]   ← 第0个token只能看自己
t=1 [ 1    1    0    0    0  ]   ← 第1个能看 0,1
t=2 [ 1    1    1    0    0  ]   ← 第2个能看 0,1,2
t=3 [ 1    1    1    1    0  ]
t=4 [ 1    1    1    1    1  ]   ← 最后一个能看所有

0 的位置填 -inf → softmax 后变 0 → 看不到未来
```

### Flash Attention vs 手动实现

```
手动实现：
  1. 计算完整 T×T 注意力矩阵 → 占 O(T²) 显存
  2. 存储 softmax 输出 → 再乘 V
  显存: O(T²)，T=1024 时 = 1M 个 float

Flash Attention：
  1. 分块计算（tiling），不完整存储注意力矩阵
  2. 利用 GPU SRAM（快但小）替代 HBM（慢但大）
  显存: O(T)，速度快 2-4x
  is_causal=True 自动应用因果 mask
```



---

## 四、MLP — 前馈网络

### 源码

```python
class MLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.c_fc    = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu    = nn.GELU()
        self.c_proj  = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.c_fc(x)       # 升维：768 → 3072
        x = self.gelu(x)       # 非线性激活
        x = self.c_proj(x)     # 降维：3072 → 768
        x = self.dropout(x)    # 正则化
        return x
```

### 形态变化

```
输入:   (B, T, 768)
  ↓ c_fc (Linear)
        (B, T, 3072)     ← 升维 4 倍
  ↓ GELU
        (B, T, 3072)     ← 非线性，形状不变
  ↓ c_proj (Linear)
        (B, T, 768)      ← 降回原始维度
  ↓ Dropout
输出:   (B, T, 768)
```

### 为什么升维 4 倍？

```
Attention 做的是"信息整合"（不同 token 之间交流）
MLP 做的是"信息处理"（对每个 token 独立做非线性变换）

升维 4 倍 = 给模型一个更大的"思考空间"
在这个高维空间里做非线性运算，再压回原维度
```

### GELU 激活函数

```
GELU(x) = x · Φ(x)   其中 Φ 是标准正态的 CDF

与 ReLU 的对比：
  ReLU(x)  = max(0, x)        ← 硬截断，x<0 直接变 0
  GELU(x)  = x · Φ(x)        ← 软截断，x<0 时平滑趋向 0

GELU 更平滑 → 梯度更连续 → 训练更稳定
GPT-2/3/4 等大模型都用 GELU
```

### 参数量

```
c_fc:   768 × 3072 = 2,359,296 参数
c_proj: 3072 × 768 = 2,359,296 参数
总计:   ~4.7M 参数 / 每个 MLP 模块

12 层 MLP 总计: ~56M 参数（占 GPT-2 124M 的近一半！）
```

---

## 五、Block — 一个 Transformer 层

### 源码

```python
class Block(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))    # 残差 + Pre-Norm Attention
        x = x + self.mlp(self.ln_2(x))     # 残差 + Pre-Norm MLP
        return x
```

### 数据流图解

```
输入 x: (B, T, 768)
    │
    ├──────────────────────────┐
    │                          │ (残差连接)
    ↓                          │
  LayerNorm (ln_1)             │
    ↓                          │
  CausalSelfAttention          │
    ↓                          │
    +  ←───────────────────────┘
    │
    ├──────────────────────────┐
    │                          │ (残差连接)
    ↓                          │
  LayerNorm (ln_2)             │
    ↓                          │
  MLP                          │
    ↓                          │
    +  ←───────────────────────┘
    │
输出: (B, T, 768)
```

### Pre-Norm vs Post-Norm

```
Post-Norm (原始 Transformer 论文):
  x = LayerNorm(x + Attention(x))    ← Norm 在加法之后

Pre-Norm (GPT-2 / nanoGPT 用的):
  x = x + Attention(LayerNorm(x))    ← Norm 在子层之前

Pre-Norm 的好处：
  - 训练更稳定（梯度流更顺畅）
  - 不容易梯度爆炸
  - 不需要 learning rate warmup 就能训练（虽然还是用了）
```

### 残差连接的直觉

```
没有残差：x = f(x)
  → 信息必须经过每一层的变换，深层模型信息衰减

有残差：x = x + f(x)
  → f(x) 只学"增量修正"
  → 梯度可以直接通过 +x 回传（跳过 f）
  → 深层模型也能训练
```



---

## 六、GPTConfig — 模型配置

### 源码

```python
@dataclass
class GPTConfig:
    block_size: int = 1024    # 最大序列长度（上下文窗口）
    vocab_size: int = 50304   # 词表大小
    n_layer: int = 12         # Transformer 层数
    n_head: int = 12          # 注意力头数
    n_embd: int = 768         # 嵌入维度
    dropout: float = 0.0      # Dropout 概率
    bias: bool = True         # 是否使用 bias
```

### 各参数详解

| 参数 | 含义 | 为什么这个值？ |
|------|------|----------------|
| block_size=1024 | 最多看 1024 个 token 的历史 | GPT-2 原始设置 |
| vocab_size=50304 | 词表大小 | GPT-2 实际是 50257，向上取整到 64 的倍数，GPU 运算更高效 |
| n_layer=12 | 12 层 Transformer | GPT-2 small 设置 |
| n_head=12 | 12 个注意力头 | head_dim = 768/12 = 64，常见选择 |
| n_embd=768 | 嵌入维度 | GPT-2 small 设置 |
| dropout=0.0 | 预训练不用 dropout | 数据量足够大时，模型不容易过拟合 |
| bias=True | 默认有 bias | 设为 False 稍好一些（更现代的做法） |

### GPT-2 系列配置

```python
config_args = {
    'gpt2':        dict(n_layer=12, n_head=12, n_embd=768),   # 124M
    'gpt2-medium': dict(n_layer=24, n_head=16, n_embd=1024),  # 350M
    'gpt2-large':  dict(n_layer=36, n_head=20, n_embd=1280),  # 774M
    'gpt2-xl':     dict(n_layer=48, n_head=25, n_embd=1600),  # 1558M
}
```

### 参数量粗算（GPT-2 124M）

```
Embedding:     vocab_size × n_embd = 50304 × 768 = 38.6M
Position Emb:  block_size × n_embd = 1024 × 768  = 0.8M (不计入有效参数)

每层 Block:
  c_attn:   768 × 2304 = 1.77M
  c_proj:   768 × 768  = 0.59M
  c_fc:     768 × 3072 = 2.36M
  c_proj2:  3072 × 768 = 2.36M
  LayerNorm × 2:        = ~0.003M
  小计: ~7.1M

12 层: 7.1M × 12 = 85.2M
ln_f:  ~0.0015M
lm_head: 与 wte 共享，不额外计算

总计: 38.6M + 85.2M ≈ 124M ✓
```

---

## 七、GPT 主类 — __init__（模型组装）

### 源码

```python
class GPT(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config

        # ═══ 组装所有子模块 ═══
        self.transformer = nn.ModuleDict(dict(
            wte  = nn.Embedding(config.vocab_size, config.n_embd),  # token embedding
            wpe  = nn.Embedding(config.block_size, config.n_embd),  # position embedding
            drop = nn.Dropout(config.dropout),
            h    = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = LayerNorm(config.n_embd, bias=config.bias),     # final layer norm
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # ═══ Weight Tying ═══
        self.transformer.wte.weight = self.lm_head.weight
        # 输入嵌入和输出投影共享同一个权重矩阵！
        # 直觉：token→向量 和 向量→token 本质是同一个映射的正反方向
        # 效果：减少 38.6M 参数，且效果更好

        # ═══ 权重初始化 ═══
        self.apply(self._init_weights)
        # 对残差投影层特殊处理
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config.n_layer))
```

### Weight Tying 深入理解

```
不共享时：
  wte:     (50304, 768)   ← 38.6M 参数
  lm_head: (768, 50304)   ← 38.6M 参数
  总共 77.2M 参数

共享后：
  wte = lm_head.weight（同一块内存）
  只有 38.6M 参数

为什么可以共享？
  wte:     token_id → 向量（"苹果" → [0.1, -0.3, ...]）
  lm_head: 向量 → token_id 的分数

  如果两个 token 的嵌入向量接近 → 它们在输出时也应该容易互相替代
  这个假设在实践中成立，共享权重甚至比不共享效果更好
```

### 权重初始化策略

```python
def _init_weights(self, module):
    if isinstance(module, nn.Linear):
        torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        if module.bias is not None:
            torch.nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
```

```
普通层：     N(0, 0.02)
残差投影层： N(0, 0.02 / √(2 × n_layer))

为什么残差投影要缩小？
  每层有 2 个残差加法（attention + mlp）
  12 层 = 24 次累加
  如果每次加的方差都是 0.02² = 4e-4
  累加后方差会变成 24 × 4e-4 → 太大

  缩放 std /= √(2×12) ≈ 4.9
  → 累加后方差保持在合理范围
  → 防止深层模型初始输出爆炸
```



---

## 八、GPT.forward() — 前向传播（详细版）

### 源码

```python
def forward(self, idx, targets=None):
    """
    前向传播：token id 序列 → logits + loss
    
    参数:
        idx:     (batch, seq_len) 输入 token id
        targets: (batch, seq_len) 目标 token id（训练时提供，推理时为 None）
    返回:
        logits:  (batch, seq_len, vocab_size) 每个位置对所有词的预测分数
        loss:    标量，交叉熵损失（推理时为 None）
    """
    device = idx.device
    b, t = idx.size()                    # b=批大小, t=序列长度
    assert t <= self.config.block_size   # 序列不能超过上下文窗口(1024)
    pos = torch.arange(0, t, dtype=torch.long, device=device)  # 位置索引 [0,1,...,t-1]

    # ═══ 嵌入：把整数 id 变成连续向量 ═══
    tok_emb = self.transformer.wte(idx)  # 查词嵌入表: (b,t) → (b,t,768)
    pos_emb = self.transformer.wpe(pos)  # 查位置嵌入表: (t,) → (t,768)
    x = self.transformer.drop(tok_emb + pos_emb)  # 语义+位置 相加，再 Dropout

    # ═══ 核心：12 层 Transformer Block 逐层处理 ═══
    for block in self.transformer.h:
        x = block(x)                     # 每层: Attention + MLP, 形状不变 (b,t,768)

    # ═══ 最终归一化 ═══
    x = self.transformer.ln_f(x)         # LayerNorm 稳定数值

    # ═══ 输出头：768 维向量 → 50304 个词的分数 ═══
    if targets is not None:
        # 训练模式：所有位置都算 loss
        logits = self.lm_head(x)         # (b,t,768) → (b,t,50304)
        loss = F.cross_entropy(          # 交叉熵：预测分布 vs 正确答案
            logits.view(-1, logits.size(-1)),  # 展平为 (b*t, 50304)
            targets.view(-1),                   # 展平为 (b*t,)
            ignore_index=-1                     # 跳过 padding 位置
        )
    else:
        # 推理模式：只算最后一个位置（省 1024 倍计算）
        logits = self.lm_head(x[:, [-1], :])   # (b,1,768) → (b,1,50304)
        loss = None

    return logits, loss
```

### 这个函数做了什么？

forward() 是整个 GPT 模型的"主线路"：把 token id 进去，logits 和 loss 出来。

```
输入：一批 token id 序列  →  输出：每个位置对下一个 token 的预测分数
```

### idx 中的 token id 是如何来的？

**分词器（Tokenizer）把原始文本切成子词片段，每个片段对应一个固定整数 id。**

```
原始文本:  "The cat sat on the mat"
              ↓  分词器（GPT-2 BPE Tokenizer）
token 列表: ["The", " cat", " sat", " on", " the", " mat"]
              ↓  查词表（vocab_size=50257）
token ids:  [464,   3797,   3332,  319,   262,   2603]
              ↓  组成 tensor
idx:        torch.tensor([[464, 3797, 3332, 319, 262, 2603]])  shape=(1,6)
```

#### nanoGPT 中的实际来源

token id **不是训练时实时分词的**，而是提前离线处理好的：

```python
# ═══ 第一步：离线分词（data/openwebtext/prepare.py，只跑一次）═══
import tiktoken
enc = tiktoken.get_encoding("gpt2")         # 加载 GPT-2 BPE 分词器

# 把整个 OpenWebText 分词，存成二进制
ids = enc.encode_ordinary(text)             # "The cat" → [464, 3797]
np.array(ids, dtype=np.uint16).tofile('train.bin')
# 输出: train.bin (~17GB, 9B 个 token id)

# ═══ 第二步：训练时直接读取（train.py: get_batch()）═══
data = np.memmap('train.bin', dtype=np.uint16, mode='r')  # 内存映射，不全部加载
ix = torch.randint(len(data) - block_size, (batch_size,)) # 随机选起始位置
x = torch.stack([data[i:i+block_size] for i in ix])       # → idx  (8, 1024)
y = torch.stack([data[i+1:i+1+block_size] for i in ix])   # → targets (8, 1024)
# x 和 y 相差一位：x=[The,cat,sat], y=[cat,sat,on]
```

#### BPE 分词器如何工作？

```
GPT-2 使用 Byte Pair Encoding (BPE)，词表 50257 个子词

规则：
  1. 最初每个字节是一个 token
  2. 统计训练语料中最常见的相邻 token 对
  3. 合并最常见的对 → 形成新 token
  4. 重复直到词表达到目标大小

结果：
  常见词不拆:   "The"(464), " the"(262), " is"(318)
  罕见词拆开:   "ChatGPT" → ["Chat", "G", "PT"]
  子词片段:     "ing"(278), "tion"(1009), "un"(403)
  单字符保底:   "a"(64), "z"(89)  ← 任何文本都能编码

为什么用 BPE 不按字/按词分？
  按字符: 词表小(65) 但序列太长（"hello"=5个token）
  按词:   词表巨大(100万+)，新词 OOV
  BPE:    词表适中(50257)，任何文本都能编码，效率和覆盖率的平衡点
```

#### BPE 分词器是怎么来的？

BPE 分词器**不是神经网络**，它是一个纯统计算法，在训练语言模型**之前**单独跑的：

```
输入：大量原始文本（比如 WebText，约 40GB）

Step 1: 初始状态
  每个字节/字符作为最小单元，词表 = 256 个基础字节

Step 2: 统计相邻对频率
  在整个语料里找最常同时出现的两个相邻片段
  比如 ('t', 'h') 出现 1000 万次 → 合并成 'th'

Step 3: 合并，更新词表，重复
  GPT-2 目标词表大小 50257
  → 合并了约 50000 次，每次生成一个新 token

产物：
  词表文件（id ↔ 字符串）
  合并规则文件（按优先级排列的合并顺序）
  这两个文件就是"分词器"
```

GPT-2 的分词器是 OpenAI 用 WebText 语料预先训练好的，nanoGPT 直接加载使用：

```python
enc = tiktoken.get_encoding("gpt2")  # 加载现成的，不需要自己训练
```

nanochat 则重新训练了一个专门针对对话数据的分词器（词表 32768，含 `<|user|>` 等特殊 token）。

---

#### 分词器就是词表吗？

**不完全是，但密切相关。**

```
词表（Vocabulary）：
  一个 id → 字符串 的映射表
  id=464   → "The"
  id=3797  → " cat"
  id=50256 → "<|endoftext|>"
  共 50257 条，只是一个查找表

分词器（Tokenizer）：
  词表 + 合并规则（merge rules）

  光有词表不够，还需要知道：
  "遇到新文本，如何按优先级合并成 token？"

类比：
  词表  = 字典（查每个词什么意思）
  分词器 = 字典 + 断词规则（不认识的词怎么拆）
```

分词器的词表大小直接决定了模型的 `vocab_size`，两者必须严格对应，不能混用。

---

#### 词表中有 768 维向量吗？

**没有。词表和嵌入向量是完全独立的两个东西。**

```
词表（分词器里）：
  id=464  → "The"       ← 只有 id 到字符串的映射，没有向量

wte.weight（模型里）：
  id=464  → [0.12, -0.03, 0.87, ...]   ← 只有向量，没有字符串
  id=3797 → [-0.21, 0.55, 0.09, ...]

两者通过 id 关联，但物理上完全独立：
  分词器  → 存在 tiktoken/tokenizer 文件里（固定，训练模型前就确定）
  wte     → 存在模型的 .pt 权重文件里（可学习，随模型训练更新）
```

```
流程：
  "The cat"
    ↓  分词器（字符串 → id）
  [464, 3797]
    ↓  wte.weight（id → 768维向量，查表）
  [[0.12,-0.03,...], [-0.21,0.55,...]]
    ↓  进入 Transformer

分词器做完就退出，后面模型只和 id 与向量打交道
```

---

#### 完整链路总结

```
┌─────────────────────────────────────────────────────────┐
│  训练前（离线，只做一次）                                 │
│    原始文本 → BPE分词 → id数组 → train.bin              │
├─────────────────────────────────────────────────────────┤
│  训练时（每一步）                                        │
│    train.bin → 随机取1024个连续id → idx (8,1024)        │
│    idx右移一位 → targets (8,1024)                       │
│    → model.forward(idx, targets) → loss → backward     │
├─────────────────────────────────────────────────────────┤
│  推理时                                                  │
│    用户输入"The cat" → 分词器 → [464, 3797]             │
│    → model.forward(idx) → logits → 采样 → 新id         │
│    → 拼到idx后面 → 重复                                 │
│    → 所有id → 分词器.decode() → 输出文字                │
└─────────────────────────────────────────────────────────┘
```

---

### 源码逐行解析

```python
def forward(self, idx, targets=None):
```

**参数解释**：
```
idx:     (b, t) 的 LongTensor，b=batch_size, t=序列长度
         每个元素是 0~50303 的整数（token id）
         例：idx[0] = [464, 3797, 319, ...]  → "The cat on ..."

targets: (b, t) 的 LongTensor，和 idx 形状完全相同
         targets = idx 向右移一位
         即 targets[i] = idx 的下一个 token

         例：idx     = [The, cat, sat, on, the]
             targets = [cat, sat, on, the, mat]
         
         如果 targets=None → 推理模式（不计算 loss）
```

---

```python
    device = idx.device
    b, t = idx.size()
    assert t <= self.config.block_size
```

```
device: idx 在哪个设备上（cpu/cuda），后续创建的 tensor 要放同一个设备
b, t:   batch_size, 序列长度
assert: 序列不能超过上下文窗口（1024），否则位置编码会越界
```

---

```python
    pos = torch.arange(0, t, dtype=torch.long, device=device)  # shape (t,)
```

```
生成位置索引：[0, 1, 2, 3, ..., t-1]
用于查询位置嵌入表

为什么需要位置信息？
  Attention 本身是"集合运算"——不关心顺序
  "猫追狗" 和 "狗追猫" 在没有位置信息时，注意力看来是一样的！
  位置嵌入告诉模型：这个 token 在第几个位置
```

---

#### 嵌入层（把 id 变成向量）

```python
    tok_emb = self.transformer.wte(idx)   # (b, t) → (b, t, n_embd)
```

```
Token Embedding：每个 token id 查嵌入表，得到一个 768 维向量

嵌入表 wte.weight 的形状：(50304, 768)
  → 50304 个 token，每个有一个 768 维的"身份证"

例：
  id=464 ("The") → wte.weight[464] = [0.12, -0.03, 0.87, ...]  (768维)
  id=3797("cat") → wte.weight[3797] = [-0.21, 0.55, 0.09, ...]  (768维)

这些向量是训练出来的，语义相近的词向量也接近：
  "cat" 的向量 ≈ "dog" 的向量（都是动物）
  "cat" 的向量 ≠ "car" 的向量（语义不同）
```

**wte.weight 是训练出来的吗？**

是的，和模型所有参数一样，通过反向传播更新：

```
训练前（随机初始化）：
  wte.weight[464] ("The") = [0.013, -0.008, 0.021, ...]  ← 随机噪声
  wte.weight[3797]("cat") = [-0.005, 0.019, -0.011, ...] ← 随机噪声
  "The" 和 "cat" 的向量毫无语义关系

训练后：
  wte.weight[3797] ("cat") = [-0.21, 0.55, 0.09, ...]
  wte.weight[40535]("dog") = [-0.19, 0.51, 0.11, ...]  ← 和 cat 很接近！
  wte.weight[2018] ("car") = [0.45, -0.12, -0.33, ...]  ← 和 cat 不像
  语义相近的词，向量自然聚在一起（这不是设计出来的，是训练涌现的）
```

每步训练时，只有本 batch 出现过的 token id 对应的行才会被更新，
没出现过的行梯度为 0，保持不变。这也是"罕见词嵌入质量差"的原因。

**词表中有 768 维向量吗？**

没有。词表（BPE 分词器的产物）和嵌入矩阵是完全独立的两个东西：

```
词表（分词器文件里）：
  id=464  → "The"       ← 只有 id ↔ 字符串映射，没有向量

wte.weight（模型权重文件里）：
  id=464  → [0.12, -0.03, ...]  ← 只有向量，没有字符串

两者通过 id 关联：
  分词器：字符串 → id   （训练模型前固定，不随模型训练变化）
  wte：   id → 向量      （随模型训练更新）
```

---

```python
    pos_emb = self.transformer.wpe(pos)   # (t,) → (t, n_embd)
```

```
Position Embedding：每个位置查位置嵌入表

嵌入表 wpe.weight 的形状：(1024, 768)
  → 1024 个位置，每个有一个 768 维的"位置标记"

例：
  位置 0 → wpe.weight[0] = [0.01, 0.05, -0.02, ...]
  位置 1 → wpe.weight[1] = [0.03, -0.01, 0.04, ...]

位置向量也是训练出来的：
  模型自己学会"位置 0 通常是句子开头"
  "相邻位置的向量比较像"等模式
```

---

```python
    x = self.transformer.drop(tok_emb + pos_emb)
```

```
Token Embedding + Position Embedding = 模型的输入

相加的含义：
  tok_emb 告诉模型"这个 token 是什么"（语义信息）
  pos_emb 告诉模型"这个 token 在哪里"（位置信息）
  两者相加 → 同时包含"是什么"和"在哪里"

广播机制：
  tok_emb: (b, t, 768)   ← 每个样本每个位置有自己的 token 向量
  pos_emb: (t, 768)      ← 位置向量对所有样本一样
  相加时 pos_emb 自动广播到 (b, t, 768)

Dropout：
  训练时随机把一些维度置零（概率=dropout=0.0，预训练时不用）
  作用：正则化，防止过拟合
```

---

#### Transformer 层堆叠（核心计算）

```python
    for block in self.transformer.h:
        x = block(x)                      # (b, t, 768) → (b, t, 768)
```

```
12 个 Block 依次处理：
  Block 0: x → Attention(看周围) → MLP(非线性变换) → x'
  Block 1: x' → Attention → MLP → x''
  ...
  Block 11: → 最终的 x

每一层形状都不变 (b, t, 768)，但内容越来越"丰富"：
  浅层（0-3）：学到语法、词性、局部依赖
  中层（4-7）：学到语义关系、句子结构
  深层（8-11）：学到高级推理、长距离关联

类比：
  Block 0 = 认字  （"这是个名词"）
  Block 5 = 理解句意  （"猫坐在垫子上"）
  Block 11 = 预测下文  （"接下来应该说什么"）
```

---

#### 最终归一化

```python
    x = self.transformer.ln_f(x)          # (b, t, 768) → (b, t, 768)
```

```
最后一个 LayerNorm：
  12 层残差累加后，数值可能偏大或不稳定
  ln_f 把输出归一化到合理范围
  → lm_head 的输入更稳定 → 训练更容易
```

---

#### 输出层（向量 → 词表分数）

```python
    if targets is not None:
        # ═══ 训练模式 ═══
        logits = self.lm_head(x)          # (b, t, 768) → (b, t, 50304)
        loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)),  # (b*t, 50304)
            targets.view(-1),                   # (b*t,)
            ignore_index=-1
        )
    else:
        # ═══ 推理模式 ═══
        logits = self.lm_head(x[:, [-1], :])  # (b, 1, 50304)
        loss = None

    return logits, loss
```

---

### lm_head 在做什么？

#### 源码（在 __init__ 中定义）

```python
# 定义（在 GPT.__init__ 中）
self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
# Linear(768, 50304, bias=False)
# 权重形状: (50304, 768) — 每个 token 对应一个 768 维的"判别向量"

# Weight Tying: 和输入嵌入共享同一个权重
self.transformer.wte.weight = self.lm_head.weight
# wte.weight 形状也是 (50304, 768)
# 共享后，只存一份权重，省 38.6M 参数
```

#### 前向计算

```python
# 训练时（对所有位置计算）：
logits = self.lm_head(x)
# 等价于: logits = x @ lm_head.weight.T
# x:      (b, t, 768)
# weight: (50304, 768) → 转置后 (768, 50304)
# logits: (b, t, 768) @ (768, 50304) = (b, t, 50304)

# 推理时（只对最后一个位置计算）：
logits = self.lm_head(x[:, [-1], :])
# x[:, [-1], :]: (b, 1, 768)
# logits:        (b, 1, 50304)
```

#### 它在做什么（直觉）

```
lm_head 本质上是在做"相似度匹配"：

  对于 Transformer 输出的每个 768 维向量 h：
    logits[i] = h · wte.weight[i]   （点积）
    
  即：把 h 和词表中所有 50304 个 token 的嵌入向量逐个做点积
  点积越大 = h 和这个 token 的嵌入越"像" = 模型越觉得下一个词是它

因为 Weight Tying，lm_head.weight == wte.weight：
  输入时：token_id → 查表 → 768维向量（"苹果是什么"）
  输出时：768维向量 → 和所有token做点积 → 最像谁（"这最像苹果"）
  本质是同一张表的正反查询
```

**为什么 `logits[i] = h · wte.weight[i]` 得到的是下一个 token 的分数？**

直觉上有个疑惑：h 是"当前位置"的向量，为什么点积能预测"下一个"词？

关键在于：**h 不是"当前 token 的表示"，而是"模型对下一个 token 的预测编码"。**

```
输入: "The cat sat on the"
              ↓ 12层 Transformer 处理
h = 最后位置("the")的输出向量

但 h 不是 "the" 这个词本身的向量
它是：模型读完整个序列后，对"the"之后应该出现什么词的预测编码

类比：
  你读完 "The cat sat on the"
  你脑子里 "the" 位置的状态
  不是 "the 是什么词"
  而是 "接下来大概率出现 mat / floor / sofa..."
```

这个语义是**训练出来的**，不是设计出来的：

```
训练时 loss = -log(P[正确的下一个token])
梯度反复告诉模型：
  "h 要和 'mat' 的嵌入向量更接近，和其他词更远"
  
训练足够多步后：
  h 自然指向"正确的下一个 token 在嵌入空间中的位置"
  h · wte["mat"] 就自然地成为最大值
```

```
训练前：h 随机，和所有 token 点积差不多  →  loss 很大
训练后：h 指向正确答案方向            →  loss 很小
```

**那么 h 是怎么"被 MLP 变成"预测向量的？**

MLP **不直接**生成"对下一个 token 的预测"——它只是数据处理的中间一环。
`h` 是经过 12 层 **Attention + MLP 反复处理**后自然形成的，没有哪一层是专门负责"预测下一个词"的。

```
输入: x = tok_emb + pos_emb    ← "the 在位置 4"

Block 0:
  Attention: 收集上下文信息（看前面所有词，决定关注谁）
  MLP:       对汇总信息做非线性变换（消化吸收）

Block 1~11: 同上，每层建立在前一层的基础上

→ h = 12 层处理后的最终 768 维向量
```

**Attention 和 MLP 各自干什么：**

```
Attention：token 之间的"信息交流"
  位置4的"the"去问前面的词：
    "位置0的'The'对我贡献30%"
    "位置1的'cat'对我贡献50%"
    "位置2的'sat'对我贡献20%"
  → x_after_attn 包含整个序列的上下文摘要

MLP：对每个位置独立做"知识加工"
  不做跨 token 的交流，只对当前位置的向量做非线性变换
  实验表明 MLP 层存储了大量事实知识：
  "cat 后面接 sat，sat 后面可能接 on..."
  这类规律就编码在 MLP 的 5 亿个权重数字里
```

**关键是训练目标，不是某个模块的设计：**

```
梯度传播链：
  loss → lm_head → Block 11 的 MLP 和 Attention
       → Block 10 的 MLP 和 Attention
       ...
       → Block 0 的 MLP 和 Attention
       → 嵌入层

每次梯度都在说：
  "你的 h 和 'mat' 的向量要更接近"
  "Block 11 的 MLP，你的权重要这样调整"
  "Block 11 的 Attention，你对各 token 的注意力要这样调整"
  ...

经过数十亿次更新后：
  12 层 Attention 和 MLP 自动协同分工
  → 整体把输入变成"指向下一个 token 方向"的 h
  这个能力不是任何一层设计出来的，是从数据涌现出来的
```

**类比：**

```
厨师把食材做成美食，不是哪一步"切菜"或"翻炒"单独负责美味
而是整套工序（12道）协同的结果。

训练 = 试菜反复改进菜谱：
  每次做完尝一口（计算 loss）
  如果不好吃，调整每一步的做法（反向传播）
  反复数十亿次后，整套工序学会做出好菜

最终的 h = 那道"美食"——它是整个过程的产物，没有哪一步单独负责
```

#### 具体数字示例

```
假设经过 12 层后，最后位置的向量 h = [0.5, -0.2, 0.8, ...]  (768维)

lm_head 计算：
  logits[0]     = h · wte["the"]   = 0.5×0.1 + (-0.2)×0.3 + ... = 2.1
  logits[1]     = h · wte["a"]     = 0.5×0.05 + (-0.2)×(-0.1) + ... = 1.8
  ...
  logits[464]   = h · wte["The"]   = ... = -0.5
  logits[12345] = h · wte["mat"]   = ... = 12.3  ← 最高分！
  ...
  logits[50303] = h · wte["▅"]     = ... = -3.2

结果：50304 个分数，"mat" 得分最高
→ softmax 后 "mat" 概率最大
→ 模型预测下一个词是 "mat"
```

#### 为什么 bias=False？

```python
self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
#                                                         ^^^^^^^^^^^^
```

```
如果有 bias：logits[i] = h · weight[i] + bias[i]
  → 每个 token 有一个额外的"基础偏好分"
  → bias[i] 大的 token 天然更容易被预测到（不管上下文是什么）
  → 这不合理：词频信息应该从数据中学到，而不是硬编码

实际影响：
  有 bias：相当于给高频词一个"免费加分" → 模型倾向于重复高频词
  无 bias：完全基于上下文相似度预测 → 更公平

另外，因为 Weight Tying，wte 没有 bias，lm_head 也不应该有
```

#### 参数量

```
lm_head.weight: (50304, 768) = 38,633,472 参数 = 38.6M

但由于 Weight Tying，这 38.6M 和 wte 是同一块内存
→ 实际不额外占参数
→ 如果不共享，模型总参数会多出 38.6M（从 124M 变成 163M）
```

#### 计算量

```
训练时（所有 1024 个位置）：
  (8, 1024, 768) @ (768, 50304) = (8, 1024, 50304)
  FLOPs: 8 × 1024 × 768 × 50304 × 2 ≈ 633 亿次浮点运算
  → 这是单次前向中计算量最大的一步！

推理时（只算最后 1 个位置）：
  (8, 1, 768) @ (768, 50304) = (8, 1, 50304)
  FLOPs: 8 × 1 × 768 × 50304 × 2 ≈ 6.2 亿次
  → 比训练时少 1024 倍
```

---

### 训练模式 vs 推理模式

```
训练模式（targets 不为 None）：
  ┌─────────────────────────────────────────────────────────┐
  │  输入: [The, cat, sat, on, the]                          │
  │  目标: [cat, sat, on, the, mat]                          │
  │                                                          │
  │  对每个位置都算 logits → 和目标对比 → 求 loss            │
  │  位置0：模型预测"cat"的分数高不高？                       │
  │  位置1：模型预测"sat"的分数高不高？                       │
  │  ...                                                     │
  │  所有位置的 loss 取平均 → 一个标量                        │
  └─────────────────────────────────────────────────────────┘

推理模式（targets=None）：
  ┌─────────────────────────────────────────────────────────┐
  │  输入: [The, cat, sat, on, the]                          │
  │                                                          │
  │  只算最后一个位置的 logits                                │
  │  → "the" 后面最可能是什么？                               │
  │  → logits = [... mat:12.3, floor:11.8 ...]              │
  │                                                          │
  │  前面 4 个位置的 logits 没用（已经生成过了）              │
  │  → 不算它们，省计算                                       │
  └─────────────────────────────────────────────────────────┘
```

---

### 为什么推理时只算最后一位？

```python
logits = self.lm_head(x[:, [-1], :])  # 只取最后位置
```

```
lm_head 的计算量：
  全部位置: (b, 1024, 768) @ (768, 50304) = (b, 1024, 50304)
  → 1024 × 768 × 50304 ≈ 396 亿次乘加

  只要最后一个: (b, 1, 768) @ (768, 50304) = (b, 1, 50304)
  → 1 × 768 × 50304 ≈ 3860 万次乘加

  节省：1024 倍计算量！

注意：Transformer 层本身还是要全部计算的（因为注意力需要所有位置），
     但 lm_head 这一步可以省掉（只需要最后位置的 logits）。
```

---

### x[:, [-1], :] vs x[:, -1, :] 的区别

```python
x.shape = (8, 1024, 768)

x[:, -1, :]     # shape = (8, 768)        ← 降维了，变成 2D
x[:, [-1], :]   # shape = (8, 1, 768)     ← 保持 3D

为什么要保持 3D？
  后续代码统一按 (batch, seq_len, features) 处理
  如果变成 2D 还要 unsqueeze，不如一开始就保持一致
```

---

### Cross Entropy Loss 完整解释

```python
loss = F.cross_entropy(
    logits.view(-1, logits.size(-1)),   # (b*t, vocab_size) = (8192, 50304)
    targets.view(-1),                    # (b*t,) = (8192,)
    ignore_index=-1
)
```

#### 为什么要 view(-1, ...)？

```
cross_entropy 要求输入是 2D：(样本数, 类别数)

原始 logits: (8, 1024, 50304) → 3D，不能直接用
view(-1, 50304): 把前两维拉平 → (8×1024, 50304) = (8192, 50304)

targets: (8, 1024) → view(-1) → (8192,)
  每个元素是 0~50303 的正确答案

相当于：8192 个独立的"50304 分类"问题
```

#### Cross Entropy 的数学

```
对每个位置 i：
  1. logits_i = [s₀, s₁, ..., s₅₀₃₀₃]    ← 50304 个原始分数
  2. P_i = softmax(logits_i)               ← 转成概率分布（和为1）
     P_i[j] = exp(s_j) / Σ exp(s_k)
  3. loss_i = -log(P_i[target_i])          ← 正确答案的概率越高，loss 越小

最终 loss = (1/N) Σ loss_i               ← 所有位置取平均

直觉：
  如果模型对正确答案很确信（P=0.9）→ loss = -log(0.9) = 0.105  很小 ✓
  如果模型不确定（P=0.1）         → loss = -log(0.1) = 2.302  很大 ✗
  如果完全乱猜（P=1/50304）       → loss = -log(1/50304) = 10.8  极大 ✗✗
```

#### ignore_index=-1 的作用

```
有些位置不需要计算 loss（比如 padding）：
  targets 中设为 -1 的位置会被跳过

例：
  targets = [cat, sat, -1, -1, -1]
  → 只有前 2 个位置参与 loss 计算

在 nanoGPT 的预训练中不常用（数据是连续的，没有 padding）
但微调时可能用到
```

---

### 完整数据流（带具体数字）

```
输入: idx = (8, 1024)   ← 8 个样本，每个 1024 个 token id

Step 1: Token Embedding
  wte(idx): 每个 id 查表 → (8, 1024, 768)
  例：id=464 → 768 维向量 [0.12, -0.03, ...]

Step 2: Position Embedding
  wpe([0,1,...,1023]): 每个位置查表 → (1024, 768)
  广播加到 tok_emb 上

Step 3: 相加 + Dropout
  x = tok_emb + pos_emb → (8, 1024, 768)
  每个 token 的向量 = "我是什么词" + "我在第几位"

Step 4-15: 12 个 Transformer Block
  每个 Block：
    x → LayerNorm → Attention(整合上下文信息) → 残差 →
      → LayerNorm → MLP(非线性变换) → 残差 → x
  形状始终是 (8, 1024, 768)

Step 16: Final LayerNorm
  x → ln_f → (8, 1024, 768)

Step 17: lm_head
  训练: x → Linear(768→50304) → logits (8, 1024, 50304)
  推理: x[:,-1,:] → Linear → logits (8, 1, 50304)

Step 18: Loss（仅训练时）
  cross_entropy(logits, targets) → 标量 loss
  例：loss = 3.2（训练初期），loss = 2.85（训练结束）
```

---

### 训练时 loss 的含义

```
loss = 2.85 意味着什么？

cross_entropy loss = -log(P_correct)
  → P_correct = exp(-2.85) ≈ 0.058

即：平均来看，模型给正确答案约 5.8% 的概率

对比：
  随机猜测：loss = log(50304) ≈ 10.8，P = 1/50304 ≈ 0.002%
  GPT-2 训练好后：loss ≈ 2.85，P ≈ 5.8%
  
  看似只有 5.8%，但这是在 50304 个选项中！
  相比随机猜测提升了 29 倍。

另一种理解：
  困惑度 (perplexity) = exp(loss) = exp(2.85) ≈ 17.3
  意思是：模型在每个位置平均"犹豫"于 17 个候选词
  （随机猜测时"犹豫"于 50304 个词）
```

---

### forward 的设计巧妙之处

```
1. 训练和推理共用一个函数
   → targets=None 时自动切到推理模式
   → 不需要维护两套代码

2. 推理时的 mini-optimization
   → 只对最后一个位置计算 lm_head
   → 省 1024 倍的输出层计算

3. logits 和 loss 一起返回
   → 训练循环直接拿 loss.backward()
   → 推理时拿 logits 去采样

4. 位置编码作为"可选注入"
   → 通过简单的加法把位置信息融入
   → 如果以后换成 RoPE，只需改这一处
```

---

## 九、GPT.generate() — 自回归文本生成

### 源码

```python
@torch.no_grad()
def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
    for _ in range(max_new_tokens):
        # 裁剪到上下文窗口
        idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]

        # 前向得到 logits
        logits, _ = self(idx_cond)

        # 取最后一个位置 + temperature 缩放
        logits = logits[:, -1, :] / temperature

        # Top-K 过滤
        if top_k is not None:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = -float('Inf')

        # Softmax → 概率
        probs = F.softmax(logits, dim=-1)

        # 从概率分布中采样
        idx_next = torch.multinomial(probs, num_samples=1)

        # 拼接到序列末尾
        idx = torch.cat((idx, idx_next), dim=1)

    return idx
```

### 生成过程图解

```
初始 prompt: "The cat"  → token ids: [464, 3797]

Step 1:
  输入: [464, 3797]
  模型输出最后位置 logits → softmax → 采样 → "sat" (id=3332)

Step 2:
  输入: [464, 3797, 3332]
  模型输出最后位置 logits → softmax → 采样 → "on" (id=319)

Step 3:
  输入: [464, 3797, 3332, 319]
  ...

每步用前面所有已生成的 token 作为输入，产出下一个 token
这就是"自回归"（auto-regressive）
```

### Temperature 的效果

```
temperature = 1.0（默认）：原始概率分布
temperature = 0.5（低）：分布更尖锐 → 更确定性 → 更"安全"但可能重复
temperature = 1.5（高）：分布更平坦 → 更随机 → 更"创意"但可能胡说

数学：logits / T → softmax
  T 小 → logits 差距放大 → 高概率 token 更突出
  T 大 → logits 差距缩小 → 所有 token 概率更均匀
```

### Top-K 采样

```
Top-K = 50：
  只保留概率最高的 50 个 token
  其余全部设为 -inf → softmax 后变 0
  → 从这 50 个中采样

作用：防止采样到极低概率的"垃圾" token
```



---

## 十、GPT.from_pretrained() — 加载预训练权重

### 源码（简化注释版）

```python
@classmethod
def from_pretrained(cls, model_type, override_args=None):
    """从 HuggingFace 下载 GPT-2 权重并加载到 nanoGPT 模型"""

    # 1. 创建 nanoGPT 模型（随机权重）
    config = GPTConfig(**config_args)
    model = GPT(config)
    sd = model.state_dict()

    # 2. 从 HuggingFace 下载 GPT-2 权重
    from transformers import GPT2LMHeadModel
    model_hf = GPT2LMHeadModel.from_pretrained(model_type)
    sd_hf = model_hf.state_dict()

    # 3. 逐层复制权重
    transposed = ['attn.c_attn.weight', 'attn.c_proj.weight',
                  'mlp.c_fc.weight', 'mlp.c_proj.weight']
    for k in sd_keys_hf:
        if any(k.endswith(w) for w in transposed):
            sd[k].copy_(sd_hf[k].t())   # Conv1D → Linear 需要转置
        else:
            sd[k].copy_(sd_hf[k])        # 直接复制
    return model
```

### 为什么需要转置？

```
OpenAI 的 GPT-2 用了 Conv1D：
  权重形状: (in_features, out_features) = (768, 2304)

PyTorch 的 Linear：
  权重形状: (out_features, in_features) = (2304, 768)

它们数学上等价（y = xW^T vs y = Wx）
但存储时转置了，所以加载时要 .t()

涉及的层：c_attn, c_proj, c_fc, c_proj（所有做矩阵乘法的层）
```

---

## 十一、GPT.configure_optimizers() — 优化器配置

### 源码

```python
def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
    param_dict = {pn: p for pn, p in self.named_parameters() if p.requires_grad}

    # 按维度分组：2D 以上（矩阵）做 weight decay，1D 以下（bias/norm）不做
    decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
    nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]

    optim_groups = [
        {'params': decay_params, 'weight_decay': weight_decay},      # 0.1
        {'params': nodecay_params, 'weight_decay': 0.0},             # 0.0
    ]

    # 优先用 fused AdamW（CUDA 优化版，更快）
    fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
    use_fused = fused_available and device_type == 'cuda'
    optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas,
                                   fused=use_fused)
    return optimizer
```

### 为什么区分 decay / no-decay？

```
Weight Decay = 每步让权重衰减一点点: w = w - wd * w

应该 decay 的（2D 矩阵权重）：
  - Linear 的 weight（如 c_attn.weight, c_fc.weight）
  - Embedding 的 weight
  → 防止权重过大，起正则化作用

不应该 decay 的（1D 参数）：
  - bias 参数
  - LayerNorm 的 weight/bias
  → 这些参数本身就小，decay 会干扰训练
```

### Fused AdamW

```
普通 AdamW：
  for param in params:
    m = beta1 * m + (1-beta1) * grad
    v = beta2 * v + (1-beta2) * grad²
    param -= lr * m / (√v + eps) + wd * param
  → 每个参数一次 GPU kernel 调用

Fused AdamW：
  把上面所有运算融合成一个 kernel
  → 减少 kernel launch 开销
  → 减少 GPU 内存读写次数
  → 训练快 5-10%
```

---

## 十二、GPT.estimate_mfu() — 浮点利用率估算

### 源码

```python
def estimate_mfu(self, fwdbwd_per_iter, dt):
    """估算相对于 A100 峰值的浮点利用率"""
    N = self.get_num_params()
    cfg = self.config
    L, H, Q, T = cfg.n_layer, cfg.n_head, cfg.n_embd//cfg.n_head, cfg.block_size

    flops_per_token = 6*N + 12*L*H*Q*T
    flops_per_fwdbwd = flops_per_token * T
    flops_per_iter = flops_per_fwdbwd * fwdbwd_per_iter

    flops_achieved = flops_per_iter / dt          # 实际 FLOPS
    flops_promised = 312e12                        # A100 bf16 峰值: 312 TFLOPS
    mfu = flops_achieved / flops_promised
    return mfu
```

### FLOPs 公式解释（来自 PaLM 论文）

```
每个 token 的 FLOPs:
  6N        ← 矩阵乘法的 FLOPs（前向 2N + 反向 4N = 6N）
  + 12LHQT  ← 注意力矩阵的 FLOPs（QK^T 和 att@V）

对于 GPT-2 124M (N=124M, L=12, H=12, Q=64, T=1024):
  6N = 744M
  12LHQT = 12×12×12×64×1024 = 113M
  每 token: ~857M FLOPs

每次迭代:
  857M × 1024(T) × batch_size = ... 很大的数

MFU = 实际 / A100 峰值(312T)
  好的实现: MFU ~40-60%
  说明 GPU 在忙着算，不是等数据
```

---

## 十三、GPT.crop_block_size() — 模型裁剪

### 源码

```python
def crop_block_size(self, block_size):
    """缩小上下文窗口（模型手术）"""
    assert block_size <= self.config.block_size
    self.config.block_size = block_size
    # 截断位置嵌入
    self.transformer.wpe.weight = nn.Parameter(
        self.transformer.wpe.weight[:block_size]
    )
    # 截断因果 mask
    for block in self.transformer.h:
        if hasattr(block.attn, 'bias'):
            block.attn.bias = block.attn.bias[:,:,:block_size,:block_size]
```

### 使用场景

```
加载 GPT-2 checkpoint（block_size=1024）
但想用更短的上下文（如 256）进行快速实验

crop_block_size(256):
  wpe: (1024, 768) → (256, 768)   ← 只保留前 256 个位置
  mask: (1,1,1024,1024) → (1,1,256,256)

好处：推理更快、显存更少
代价：不能看超过 256 个 token 的历史
```

---

## 总结：model.py 的设计精髓

### 1. 极简但完整

```
300 行代码 = 工业级 GPT 模型
没有一行多余代码
没有一个多余的抽象层
```

### 2. 关键设计决策

| 决策 | 选择 | 原因 |
|------|------|------|
| Norm 位置 | Pre-Norm | 训练更稳定 |
| 激活函数 | GELU | 比 ReLU 更平滑 |
| 位置编码 | 学习式（Embedding） | 简单，GPT-2 原始方案 |
| Weight Tying | 是 | 减参、提效 |
| Flash Attention | 自动检测 | 有就用，没有退回手动 |
| bias | 可选（默认True） | 关闭稍好，更现代 |

### 3. 与更现代架构的对比

| 特性 | nanoGPT (GPT-2) | 现代 LLM (Llama/nanochat) |
|------|-----------------|--------------------------|
| 位置编码 | 学习式 | RoPE |
| Norm | LayerNorm | RMSNorm |
| 激活函数 | GELU | SwiGLU |
| 注意力 | MHA | GQA / MQA |
| FFN 倍数 | 4× | 8/3× (SwiGLU) |
| bias | 有 | 无 |

nanoGPT 选择了 GPT-2 的经典架构，目的是**教学清晰**而非最优性能。
