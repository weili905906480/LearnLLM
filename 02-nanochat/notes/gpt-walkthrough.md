# nanochat/gpt.py 源码逐行详解

> 源文件：https://github.com/karpathy/nanochat/blob/master/nanochat/gpt.py
>
> 这是 nanochat 的**核心模型文件**，定义了完整的 GPT Transformer 架构。
> 相比原版 GPT-2，引入了多项现代改进：RoPE 位置编码、GQA/MQA、QK Norm、relu² 激活、Value Embedding、Smear、Backout 等。

---

## 文件头注释

```python
"""
GPT model (rewrite, a lot simpler)
Notable features:
- rotary embeddings (and no positional embeddings)
- QK norm
- untied weights for token embedding and lm_head
- relu^2 activation in MLP
- norm after token embedding
- no learnable params in rmsnorm
- no bias in linear layers
- Group-Query Attention (GQA) support for more efficient inference
- Flash Attention 3 integration
"""
```

> 列出了本文件相比原始 GPT-2 的所有改进点：
> - **rotary embeddings**：用 RoPE 替代可学习的绝对位置编码，相对位置自然涌现，长度外推更好
> - **QK norm**：对 Q 和 K 做 RMSNorm，防止注意力 logits 过大，训练更稳定
> - **untied weights**：embedding 矩阵和 lm_head 矩阵不共享权重（GPT-2 共享），各自独立优化
> - **relu² activation**：MLP 使用 `relu(x)²` 替代 GELU，更稀疏、效果接近但更简单
> - **norm after token embedding**：token embed 之后立即做一次 RMSNorm
> - **no learnable params in rmsnorm**：RMSNorm 不含可学习的 scale 参数（更简洁）
> - **no bias**：所有 Linear 层不使用 bias（减少参数、略提速）
> - **GQA**：Group-Query Attention，K/V 头数少于 Q 头数，推理 KV Cache 更小
> - **Flash Attention 3**：H100 上使用 FA3，其他设备 fallback 到 PyTorch SDPA

---

## 第一部分：导入依赖

```python
from functools import partial
from dataclasses import dataclass
```
> - `partial`：创建带部分参数的函数，本文件中备用
> - `dataclass`：用于定义 `GPTConfig` 配置类，自动生成 `__init__`、`__repr__` 等

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
```
> PyTorch 核心三件套：
> - `torch`：张量操作
> - `nn`：神经网络模块基类（`Module`、`Linear`、`Embedding` 等）
> - `F`：无状态函数（`rms_norm`、`relu`、`cross_entropy`、`softmax` 等）

```python
from nanochat.common import get_dist_info, print0, COMPUTE_DTYPE
```
> - `get_dist_info`：获取分布式训练信息（rank、world_size 等），用于 `setup_optimizer`
> - `print0`：只在 rank 0 打印，避免多卡重复输出
> - `COMPUTE_DTYPE`：全局计算精度（bf16/fp32/fp16），模型激活值使用此精度

```python
from nanochat.optim import MuonAdamW, DistMuonAdamW
```
> - `MuonAdamW`：单 GPU 版组合优化器（矩阵权重用 Muon，其余用 AdamW）
> - `DistMuonAdamW`：多 GPU 分布式版本

```python
from nanochat.flash_attention import flash_attn
```
> 自定义 Flash Attention 模块：在 Hopper+ GPU（H100）上自动使用 FA3，其他硬件 fallback 到 PyTorch SDPA。

---

## 第二部分：GPTConfig 配置类

```python
@dataclass
class GPTConfig:
    sequence_len: int = 2048
    vocab_size: int = 32768
    n_layer: int = 12
    n_head: int = 6        # number of query heads
    n_kv_head: int = 6     # number of key/value heads (GQA)
    n_embd: int = 768
    window_pattern: str = "SSSL"
```

> 模型超参数配置，使用 `@dataclass` 装饰器自动生成构造函数：
>
> | 参数 | 默认值 | 说明 |
> |------|--------|------|
> | `sequence_len` | 2048 | 最大上下文长度（token 数） |
> | `vocab_size` | 32768 | 词表大小（2^15，BPE 分词器） |
> | `n_layer` | 12 | Transformer 层数（`--depth` 参数） |
> | `n_head` | 6 | Query 注意力头数 |
> | `n_kv_head` | 6 | Key/Value 注意力头数（GQA：可 < n_head） |
> | `n_embd` | 768 | 模型隐藏维度（embedding 维度） |
> | `window_pattern` | "SSSL" | 滑动窗口模式：S=短窗口，L=长窗口（完整上下文） |
>
> **GQA 说明**：当 `n_kv_head < n_head` 时启用 GQA。极端情况 `n_kv_head=1` 即 MQA（Multi-Query Attention）。
> 推理时 KV Cache 大小 = `n_kv_head / n_head` 倍的 MHA，大幅减少显存占用。
>
> **window_pattern 说明**：字符串在层间循环铺设，最后一层强制为 L（完整上下文）。
> 例如 "SSSL"：第 0 层短窗口，第 1 层短窗口，第 2 层短窗口，第 3 层长窗口，第 4 层短窗口...

---

## 第三部分：基础工具函数

### norm — RMSNorm

```python
def norm(x):
    return F.rms_norm(x, (x.size(-1),))
```

> 对张量最后一维做 **RMSNorm**（Root Mean Square Normalization）：
> $$\text{RMSNorm}(x) = \frac{x}{\sqrt{\frac{1}{d}\sum_{i=1}^d x_i^2 + \epsilon}}$$
>
> 相比 LayerNorm 的特点：
> - **无均值中心化**（不减 mean），计算更简单
> - **无可学习参数**（注释说明 `no learnable params in rmsnorm`），省去 γ、β
> - `F.rms_norm` 是 PyTorch 原生实现，在 bf16 下运行，性能好
>
> 这个 `norm` 函数在整个文件中被高频调用：token embedding 后、每个 Block 的 pre-norm 位置、最终输出前。

### Linear — 自定义线性层

```python
class Linear(nn.Linear):
    """nn.Linear that casts weights to match input dtype in forward.
    Replaces autocast: master weights stay fp32 for optimizer precision,
    but matmuls run in the activation dtype (typically bf16 from embeddings)."""
    def forward(self, x):
        return F.linear(x, self.weight.to(dtype=x.dtype))
```

> 继承自 `nn.Linear`，重写 `forward`，核心是 `self.weight.to(dtype=x.dtype)`：
>
> **设计动机**：nanochat 不使用 `torch.amp.autocast`，而是显式管理精度：
> - 权重以 **fp32** 存储 → 优化器（Adam 动量、Muon）精度高，避免权重噪声
> - 前向传播时临时转为输入的 dtype（通常是 **bf16**）→ 矩阵乘法在 bf16 上运行，利用张量核心
> - 注意：无 bias 参数（`bias=False`），所以不需要 `F.linear(x, w, b)` 中的第三个参数
>
> 这样既保持了 autocast 的精度收益，又有完全显式的精度控制。

### has_ve — Value Embedding 层判断

```python
def has_ve(layer_idx, n_layer):
    """Returns True if GPT layer should have Value Embedding (alternating, last layer always included)."""
    return layer_idx % 2 == (n_layer - 1) % 2
```

> 判断某一层是否拥有 **Value Embedding**（ResFormer 风格）：
> - 偶数层和奇数层交替启用 VE，确保约一半的层有 VE
> - 最后一层（`n_layer - 1`）**总是**启用 VE
>
> 例如 12 层模型（`n_layer=12`，最后一层 idx=11，奇数）：奇数层启用 VE → 层 1,3,5,7,9,11 有 VE。
>
> Value Embedding 是什么？见 `CausalSelfAttention.forward` 中的解释。

### apply_rotary_emb — RoPE 旋转位置编码

```python
def apply_rotary_emb(x, cos, sin):
    assert x.ndim == 4  # multihead attention
    d = x.shape[3] // 2
    x1, x2 = x[..., :d], x[..., d:]   # split up last dim into two halves
    y1 = x1 * cos + x2 * sin           # rotate pairs of dims
    y2 = x1 * (-sin) + x2 * cos
    return torch.cat([y1, y2], 3)
```

> 对形状为 `(B, T, H, D)` 的 Q 或 K 张量施加 RoPE 旋转：
>
> **原理**：将 head_dim 维度两两配对，视为复数实部/虚部，乘以旋转矩阵：
> $$\begin{pmatrix} y_1 \\ y_2 \end{pmatrix} = \begin{pmatrix} \cos\theta & \sin\theta \\ -\sin\theta & \cos\theta \end{pmatrix} \begin{pmatrix} x_1 \\ x_2 \end{pmatrix}$$
>
> - `x.ndim == 4` 断言：确保输入是 `(B, T, H, D)` 的 4D 张量（多头注意力布局）
> - `d = D // 2`：将 head_dim 对半分
> - `cos, sin` 形状为 `(1, T, 1, D//2)`，通过广播作用于所有 batch 和 head
> - `torch.cat([y1, y2], 3)`：合并回完整的 head_dim
>
> **关键性质**：RoPE 使得 `Q[m] · K[n] = f(m-n)`，即注意力分数只依赖相对位置差，而非绝对位置。


---

## 第四部分：CausalSelfAttention — 因果自注意力

### `__init__`

```python
class CausalSelfAttention(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.layer_idx = layer_idx
        self.n_head = config.n_head
        self.n_kv_head = config.n_kv_head
        self.n_embd = config.n_embd
        self.head_dim = self.n_embd // self.n_head
```
> 保存基础配置：
> - `layer_idx`：当前层编号，用于 KV Cache 索引和 VE 开关判断
> - `head_dim`：每个注意力头的维度 = `n_embd / n_head`（例如 768/6 = 128）

```python
        assert self.n_embd % self.n_head == 0
        assert self.n_kv_head <= self.n_head and self.n_head % self.n_kv_head == 0
```
> - 断言 1：embedding 维度必须能被 Q 头数整除
> - 断言 2：KV 头数 ≤ Q 头数，且 Q 头数必须是 KV 头数的整数倍（GQA 的分组要求）

```python
        self.c_q = Linear(self.n_embd, self.n_head * self.head_dim, bias=False)
        self.c_k = Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.c_v = Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.c_proj = Linear(self.n_embd, self.n_embd, bias=False)
```
> 四个投影矩阵（均无 bias）：
> - `c_q`：`[n_embd → n_head × head_dim]`，生成所有 Q 头
> - `c_k`：`[n_embd → n_kv_head × head_dim]`，生成 K（GQA 时比 Q 少）
> - `c_v`：`[n_embd → n_kv_head × head_dim]`，生成 V（同 K）
> - `c_proj`：`[n_embd → n_embd]`，注意力输出投影回残差流
>
> **GQA 参数对比**：MHA 中 K/V 输出维度 = `n_head × head_dim`，GQA 中 = `n_kv_head × head_dim`，参数量减少 `(1 - n_kv_head/n_head)` 倍。

```python
        self.ve_gate_channels = 12
        self.ve_gate = Linear(self.ve_gate_channels, self.n_kv_head, bias=False) if has_ve(layer_idx, config.n_layer) else None
```
> Value Embedding Gate（仅在 `has_ve` 为 True 的层创建）：
> - 输入：token embedding 的前 12 个通道（廉价的输入依赖信号）
> - 输出：`n_kv_head` 个 gate 值，控制每个 KV 头混入多少 Value Embedding
> - 不满足 `has_ve` 的层设为 `None`，节省参数

---

### `forward`

```python
    def forward(self, x, ve, cos_sin, window_size, kv_cache):
        B, T, C = x.size()
```
> 参数说明：
> - `x`：输入张量，形状 `(B, T, C)`，已经过 pre-norm
> - `ve`：Value Embedding 张量（若该层有 VE 则传入，否则 None）
> - `cos_sin`：RoPE 旋转矩阵 `(cos, sin)` 的元组
> - `window_size`：注意力窗口大小 `(left, right)` 元组
> - `kv_cache`：推理时的 KV 缓存对象，训练时为 None

```python
        q = self.c_q(x).view(B, T, self.n_head, self.head_dim)
        k = self.c_k(x).view(B, T, self.n_kv_head, self.head_dim)
        v = self.c_v(x).view(B, T, self.n_kv_head, self.head_dim)
```
> 将输入投影为 Q、K、V，并 reshape 为 `(B, T, H, D)` 格式（FA3 原生布局，无需转置）：
> - Q 形状：`(B, T, n_head, head_dim)`
> - K/V 形状：`(B, T, n_kv_head, head_dim)`（GQA 时 n_kv_head < n_head）

```python
        if ve is not None:
            ve = ve.view(B, T, self.n_kv_head, self.head_dim)
            gate = 3 * torch.sigmoid(self.ve_gate(x[..., :self.ve_gate_channels]))
            v = v + gate.unsqueeze(-1) * ve
```
> **Value Embedding（ResFormer 风格）**：
> - `ve` 是来自 token embedding table 的静态 value 向量（不经过注意力动态计算）
> - `gate`：根据输入前 12 个通道动态计算混合系数，范围 (0, 3)（sigmoid 输出乘 3）
> - `v = v + gate * ve`：将 VE 加权混入动态计算的 V
>
> **为什么有效**？VE 提供了 token 的"先验"语义信息，使得浅层注意力不需要从头学习 token 含义，加速收敛。

```python
        cos, sin = cos_sin
        q, k = apply_rotary_emb(q, cos, sin), apply_rotary_emb(k, cos, sin)
```
> 对 Q 和 K 施加 RoPE 旋转编码，注入相对位置信息。V 不做旋转（RoPE 只作用于决定"关注谁"的 Q/K，不影响"取出什么"的 V）。

```python
        q, k = norm(q), norm(k)   # QK norm
        q = q * 1.2
        k = k * 1.2
```
> **QK Norm**：对旋转后的 Q 和 K 分别做 RMSNorm：
> - 防止 Q·K 点积结果过大（导致 softmax 饱和、梯度消失）
> - 训练更稳定，尤其在深层模型或长序列时
> - 乘以 `1.2`：QK Norm 后向量范数变为 1，需要手动放大以恢复注意力的"锐度"（attention sharpness）
> - 注释 `split scale between Q and K`：传统 scaled dot-product 在 softmax 前除以 `√d_head`，这里改为在 Q 和 K 上各乘 1.2（等效于乘以 1.44，比 √128≈11.3 小得多，但 QK Norm 后不需要那么大的缩放）

```python
        if kv_cache is None:
            y = flash_attn.flash_attn_func(q, k, v, causal=True, window_size=window_size)
        else:
            k_cache, v_cache = kv_cache.get_layer_cache(self.layer_idx)
            y = flash_attn.flash_attn_with_kvcache(
                q, k_cache, v_cache,
                k=k, v=v,
                cache_seqlens=kv_cache.cache_seqlens,
                causal=True,
                window_size=window_size,
            )
            if self.layer_idx == kv_cache.n_layers - 1:
                kv_cache.advance(T)
```
> **注意力计算的两种路径**：
>
> **训练路径**（`kv_cache is None`）：
> - 调用 `flash_attn_func`，输入完整序列 Q/K/V
> - `causal=True`：因果掩码（不能看未来 token）
> - `window_size`：滑动窗口限制（S 层只关注最近 window 个 token）
>
> **推理路径**（有 KV Cache）：
> - `flash_attn_with_kvcache`：将当前步的 K/V 追加到 Cache 中，然后用完整的 Cache K/V 计算注意力
> - `cache_seqlens`：各 batch 当前已缓存的长度（支持不同长度的 batch）
> - 最后一层处理完后调用 `kv_cache.advance(T)` 更新位置指针

```python
        y = y.contiguous().view(B, T, -1)
        y = self.c_proj(y)
        return y
```
> - `.contiguous()`：确保内存连续（FA3 输出可能不连续）
> - `.view(B, T, -1)`：合并所有头 `(B, T, n_head, head_dim)` → `(B, T, n_embd)`
> - `c_proj`：输出投影，将多头注意力结果映射回残差流维度

---

## 第五部分：MLP — 前馈网络

```python
class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = Linear(config.n_embd, 4 * config.n_embd, bias=False)
        self.c_proj = Linear(4 * config.n_embd, config.n_embd, bias=False)

    def forward(self, x):
        x = self.c_fc(x)
        x = F.relu(x).square()
        x = self.c_proj(x)
        return x
```

> 两层全连接网络，中间维度放大 4 倍（GPT 经典设计）：
>
> **激活函数 `relu²`**：
> - `F.relu(x).square()` = `max(0, x)²`
> - 相比 GELU（GPT-2 原用）：更稀疏（负数输出为 0）、梯度更简洁
> - 相比普通 ReLU：输出更平滑（导数连续），有助于梯度流动
> - 无 bias：减少参数，现代 LLM 普遍做法
>
> **维度流**：`n_embd → 4*n_embd → relu² → n_embd`

---

## 第六部分：Block — Transformer 块

```python
class Block(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.attn = CausalSelfAttention(config, layer_idx)
        self.mlp = MLP(config)

    def forward(self, x, ve, cos_sin, window_size, kv_cache):
        x = x + self.attn(norm(x), ve, cos_sin, window_size, kv_cache)
        x = x + self.mlp(norm(x))
        return x
```

> 标准 **Pre-Norm Transformer Block**（GPT-2 以来的主流设计）：
>
> **数据流**：
> ```
> x → norm → Attention → + x   (残差连接)
>   → norm → MLP       → + x   (残差连接)
> ```
>
> **Pre-Norm vs Post-Norm**：
> - Pre-Norm（本实现）：先 norm 再计算，梯度更稳定，训练更容易
> - Post-Norm（原 Transformer）：计算后 norm，理论上表达能力稍强但梯度不稳
>
> 注意：Block 本身不含 `resid_lambdas` 和 `x0_lambdas` 的乘法，这些在 `GPT.forward` 的主循环中处理（见后文）。

---

## 第七部分：GPT — 主模型类

### `__init__`

```python
class GPT(nn.Module):
    def __init__(self, config, pad_vocab_size_to=64):
        """
        NOTE a major footgun: this __init__ function runs in meta device context (!!)
        Therefore, any calculations inside here are shapes and dtypes only, no actual data.
        => We actually initialize all data (parameters, buffers, etc.) in init_weights() instead.
        """
```
> **重要警告**：`__init__` 在 meta device 上下文中执行（来自 `base_train.py` 的 `with torch.device("meta")`）。
> Meta device 只有形状和 dtype，没有真实数据——所有实际数值初始化必须在 `init_weights()` 中完成。

```python
        self.config = config
        self.window_sizes = self._compute_window_sizes(config)
```
> 预计算每层的滑动窗口大小列表（`list of (left, right) tuples`），避免 forward 时重复计算。

```python
        padded_vocab_size = ((config.vocab_size + pad_vocab_size_to - 1) // pad_vocab_size_to) * pad_vocab_size_to
        if padded_vocab_size != config.vocab_size:
            print0(f"Padding vocab_size from {config.vocab_size} to {padded_vocab_size} for efficiency")
```
> **词表填充**：将 vocab_size 向上对齐到 64 的倍数。
> - 原因：矩阵乘法在维度对齐到 64（或 128）时能更好利用 GPU 张量核心
> - DDP 多卡时矩阵维度对齐也能避免通信开销
> - 实际 forward 中会裁剪回真实 vocab_size，不影响输出

```python
        self.transformer = nn.ModuleDict({
            "wte": nn.Embedding(padded_vocab_size, config.n_embd),
            "h": nn.ModuleList([Block(config, layer_idx) for layer_idx in range(config.n_layer)]),
        })
        self.lm_head = Linear(config.n_embd, padded_vocab_size, bias=False)
```
> 核心模型结构：
> - `wte`（word token embedding）：将 token ID 映射为向量，形状 `[vocab_size, n_embd]`
> - `h`：`n_layer` 个 Transformer Block 列表
> - `lm_head`：输出投影，将 `n_embd` 维隐状态映射为词表 logits
>
> 注意：`wte` 和 `lm_head` 权重**不共享**（`untied weights`），各自独立优化。

```python
        self.resid_lambdas = nn.Parameter(torch.ones(config.n_layer))
        self.x0_lambdas = nn.Parameter(torch.zeros(config.n_layer))
```
> **Per-layer 可学习标量**（受 modded-nanogpt 启发）：
> - `resid_lambdas[i]`：第 i 层前对残差流乘以该系数（初始化为 1，不改变初始行为）
> - `x0_lambdas[i]`：第 i 层混入初始 embedding `x0` 的系数（初始化为 0，初始禁用）
>
> 这两个参数让模型学会动态调整每层对残差流的贡献强度，类似于 "skip connection 的学习门控"。

```python
        self.smear_gate = Linear(24, 1, bias=False)
        self.smear_lambda = nn.Parameter(torch.zeros(1))
```
> **Smear（涂抹）机制**：
> - 将前一个 token 的 embedding 混入当前 token（提供廉价的 bigram 信息）
> - `smear_gate`：根据当前 token embedding 前 24 个通道计算混合门控
> - `smear_lambda`：全局缩放系数（初始化为 0，初始禁用）
>
> 这是一个低成本的"前向看"机制：在进入 Transformer 之前就提供一点上下文信息。

```python
        self.backout_lambda = nn.Parameter(0.2 * torch.ones(1))
```
> **Backout（撤出）机制**：
> - 在最终 norm 和 lm_head 之前，从输出中减去中间层（第 n_layer//2 层）的残差
> - 目的：去除低层特征（语法、表层信息），保留高层语义特征，提升 logit 质量
> - 初始化为 0.2（稍微撤出中间层信息）

```python
        head_dim = config.n_embd // config.n_head
        kv_dim = config.n_kv_head * head_dim
        self.value_embeds = nn.ModuleDict({
            str(i): nn.Embedding(padded_vocab_size, kv_dim)
            for i in range(config.n_layer) if has_ve(i, config.n_layer)
        })
```
> **Value Embedding 表**：
> - 每个 VE 层有一个独立的 Embedding 表，维度为 `n_kv_head × head_dim`（与 V 投影输出维度相同）
> - 用字符串层编号 `str(i)` 作为 key（ModuleDict 要求字符串 key）
> - 约一半的层有 VE，另一半为 None

```python
        self.rotary_seq_len = config.sequence_len * 10
        head_dim = config.n_embd // config.n_head
        cos, sin = self._precompute_rotary_embeddings(self.rotary_seq_len, head_dim)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)
```
> **预计算 RoPE 表**：
> - 预计算 `sequence_len × 10` 长度的 cos/sin 表（10 倍过量，支持未来可能的长序列）
> - `register_buffer`：注册为模型 buffer（跟随模型移动设备，但不是参数，不参与梯度计算）
> - `persistent=False`：不保存到 checkpoint（每次加载模型时根据序列长度重新计算）


---

### `init_weights` — 参数初始化

```python
    @torch.no_grad()
    def init_weights(self):
```
> `@torch.no_grad()` 装饰器：初始化不需要计算梯度，节省内存和计算。

```python
        torch.nn.init.normal_(self.transformer.wte.weight, mean=0.0, std=0.8)
        torch.nn.init.normal_(self.lm_head.weight, mean=0.0, std=0.001)
```
> - **wte（词嵌入）**：正态分布，std=0.8。较大的 std 让不同 token 在初始 embedding 空间中分离明显
> - **lm_head（输出投影）**：正态分布，std=0.001。极小的 std 让初始 logits 接近均匀分布，初始 loss ≈ log(vocab_size)，训练起始点稳定

```python
        n_embd = self.config.n_embd
        s = 3**0.5 * n_embd**-0.5
        for block in self.transformer.h:
            torch.nn.init.uniform_(block.attn.c_q.weight, -s, s)
            torch.nn.init.uniform_(block.attn.c_k.weight, -s, s)
            torch.nn.init.uniform_(block.attn.c_v.weight, -s, s)
            torch.nn.init.zeros_(block.attn.c_proj.weight)
            torch.nn.init.uniform_(block.mlp.c_fc.weight, -s * 0.4, s * 0.4)
            torch.nn.init.zeros_(block.mlp.c_proj.weight)
```
> **Transformer Block 初始化策略**：
>
> - `s = √3 × n_embd^{-0.5}`：均匀分布 `Uniform(-s, s)` 的标准差 = `s/√3 = n_embd^{-0.5}`，与 `Normal(0, n_embd^{-0.5})` 标准差相同，但**无极端值**（均匀分布无尾巴，避免初始权重离群点）
> - **c_q/c_k/c_v**：标准均匀初始化
> - **c_proj（注意力输出）**：**全零初始化** → 每个 Block 初始输出为 0，等效于残差连接初始为恒等映射，训练更稳定（深层网络初始化技巧）
> - **mlp.c_fc**：缩小 0.4 倍的均匀初始化，减小 MLP 初始输出幅度
> - **mlp.c_proj**：同样全零初始化

```python
        n_layer = self.config.n_layer
        for i in range(n_layer):
            self.resid_lambdas.data[i] = 1.15 - (0.10 * i / max(n_layer - 1, 1))
        for i in range(n_layer):
            self.x0_lambdas.data[i] = 0.20 - (0.15 * i / max(n_layer - 1, 1))
```
> **Per-layer 标量的精细初始化**：
>
> `resid_lambdas`（残差缩放）：从 1.15 线性衰减到 1.05（最后一层）
> - 浅层残差系数稍大：早期层信息变化更剧烈，需要更大的残差权重
> - 深层稍小：深层趋于精细调整，残差贡献减小
>
> `x0_lambdas`（初始嵌入混合）：从 0.20 线性衰减到 0.05（最后一层）
> - 浅层更多混入初始 token embedding：早期层需要更多 token 本身的词义信息
> - 深层逐渐减少：深层更关注上下文整合，不需要那么多原始 token 信息

```python
        torch.nn.init.zeros_(self.smear_lambda)
        torch.nn.init.constant_(self.backout_lambda, 0.2)
        torch.nn.init.uniform_(self.smear_gate.weight, 0.0, 0.02)
```
> - `smear_lambda = 0`：初始不激活 Smear，训练中逐渐学习
> - `backout_lambda = 0.2`：初始稍微减去中间层特征
> - `smear_gate`：小正数初始化（0 ~ 0.02），使初始 gate 输出略正，稳定初期训练

```python
        for ve in self.value_embeds.values():
            torch.nn.init.uniform_(ve.weight, -s, s)
        for block in self.transformer.h:
            if block.attn.ve_gate is not None:
                torch.nn.init.uniform_(block.attn.ve_gate.weight, 0.0, 0.02)
```
> - Value Embedding 表：与 c_v 相同的初始化方式（均匀分布，std = n_embd^{-0.5}）
> - VE Gate 权重：小正数初始化，使 gate 开始时输出接近 0（不混入 VE），逐渐学习

```python
        cos, sin = self._precompute_rotary_embeddings(self.rotary_seq_len, head_dim)
        self.cos, self.sin = cos, sin
```
> 在真实设备上重新计算 RoPE 表（`__init__` 在 meta device 上运行，这里才是真实计算）。

```python
        if COMPUTE_DTYPE != torch.float16:
            self.transformer.wte.to(dtype=COMPUTE_DTYPE)
            for ve in self.value_embeds.values():
                ve.to(dtype=COMPUTE_DTYPE)
```
> 将 Embedding 层（wte 和 value_embeds）转为计算精度（通常 bf16）：
> - Embedding 的梯度通常比矩阵权重梯度更稳定，可以接受低精度
> - 节省显存：embedding 表可能很大（vocab_size × n_embd = 32768 × 768 ≈ 75M 参数）
> - **fp16 例外**：fp16 训练时 GradScaler 无法正确处理 fp16 embedding 梯度，故保持 fp32

---

### `_precompute_rotary_embeddings` — 预计算 RoPE

```python
    def _precompute_rotary_embeddings(self, seq_len, head_dim, base=100000, device=None):
        if device is None:
            device = self.transformer.wte.weight.device
```
> 自动从 wte embedding 推断设备（保证 RoPE 表与模型在同一设备上）。

```python
        channel_range = torch.arange(0, head_dim, 2, dtype=torch.float32, device=device)
        inv_freq = 1.0 / (base ** (channel_range / head_dim))
```
> 计算每对维度的旋转频率 $\theta_i$：
> $$\theta_i = \frac{1}{base^{2i/d}} = \frac{1}{100000^{2i/d}}$$
> - `channel_range`：`[0, 2, 4, ..., head_dim-2]`，对应每对维度的索引
> - `base=100000`：RoPE 基础频率（越大低频越多，长度外推越好；注释建议可更大如 500K）
> - 低维度（小 i）频率高（旋转快），捕捉短程位置关系；高维度频率低，捕捉长程关系

```python
        t = torch.arange(seq_len, dtype=torch.float32, device=device)
        freqs = torch.outer(t, inv_freq)
        cos, sin = freqs.cos(), freqs.sin()
```
> - `t`：位置序列 `[0, 1, 2, ..., seq_len-1]`
> - `torch.outer(t, inv_freq)`：外积，得到每个位置每对维度的旋转角 `θ = t × inv_freq`，形状 `[seq_len, head_dim/2]`
> - 取 cos 和 sin：得到旋转矩阵的两个分量

```python
        cos, sin = cos.to(COMPUTE_DTYPE), sin.to(COMPUTE_DTYPE)
        cos, sin = cos[None, :, None, :], sin[None, :, None, :]
        return cos, sin
```
> - 转换为计算精度（bf16/fp32）
> - 增加 batch 维度和 head 维度：`[1, seq_len, 1, head_dim/2]`，后续通过广播作用于所有 batch 和 head

---

### `_compute_window_sizes` — 滑动窗口大小计算

```python
    def _compute_window_sizes(self, config):
        pattern = config.window_pattern.upper()
        assert all(c in "SL" for c in pattern), ...
        long_window = config.sequence_len
        short_window = -(-long_window // 4 // 128) * 128  # ceil to FA3 tile size
```
> - `long_window`：完整上下文长度（例如 2048）
> - `short_window`：短窗口 = 长窗口 ÷ 4，向上对齐到 128（FA3 tile size）
>   - 例如 seq_len=2048：`2048/4=512` → 向上对齐到 `512`（已是 128 的倍数）
>   - `-(-x // 4 // 128) * 128` 是 Python 中向上取整到 128 倍数的惯用写法

```python
        char_to_window = {
            "L": (long_window, 0),
            "S": (short_window, 0),
        }
        window_sizes = []
        for layer_idx in range(config.n_layer):
            char = pattern[layer_idx % len(pattern)]
            window_sizes.append(char_to_window[char])
        window_sizes[-1] = (long_window, 0)  # Final layer always gets full context
        return window_sizes
```
> - `(window, 0)`：`window` 是向左（过去）的注意力范围，0 是向右（未来，因果所以为 0）
> - `(-1, 0)` 表示无限左窗口（完整上下文）
> - 循环铺设：`pattern[layer_idx % len(pattern)]` 将模式字符串在所有层上循环
> - **最后一层强制 L**：确保最后一层能看到完整上下文，保留全局信息

---

### `estimate_flops` — FLOPs 估算

```python
    def estimate_flops(self):
        nparams = sum(p.numel() for p in self.parameters())
        nparams_exclude = (self.transformer.wte.weight.numel() + value_embeds_numel +
                          self.resid_lambdas.numel() + ...)
        attn_flops = 0
        for window_size in self.window_sizes:
            window = window_size[0]
            effective_seq = t if window < 0 else min(window, t)
            attn_flops += 12 * h * q * effective_seq
        num_flops_per_token = 6 * (nparams - nparams_exclude) + attn_flops
        return num_flops_per_token
```
> 估算每个 token 的训练 FLOPs（前向 + 反向）：
>
> **矩阵乘法 FLOPs**：`6 × (可训练矩阵参数数)`
> - 每个参数在前向贡献 2 FLOPs（乘 + 加）
> - 反向传播 = 2 × 前向（梯度计算）
> - 合计 `2 + 4 = 6` FLOPs/参数/token
> - 排除 embedding、标量参数（这些不是矩阵乘法）
>
> **注意力 FLOPs**：`∑_layer 12 × n_head × head_dim × effective_seq`
> - 每层注意力的 Q×K 和 Attention×V 矩阵乘法
> - 滑动窗口层用 `min(window, seq_len)` 而非完整 seq_len，精确估算
> - 公式来自 PaLM 论文（Ref 注释中有链接）

---

### `num_scaling_params` — 参数统计

```python
    def num_scaling_params(self):
        wte = ...
        value_embeds = ...
        lm_head = ...
        transformer_matrices = ...
        scalars = ...
        return {'wte': wte, 'value_embeds': value_embeds, 'lm_head': lm_head,
                'transformer_matrices': transformer_matrices, 'scalars': scalars, 'total': total}
```
> 按组别统计参数量，用于 scaling law 分析：
> - **wte**：词嵌入（Kaplan 等排除，Chinchilla 包含）
> - **value_embeds**：Value Embedding 表（nanochat 新增）
> - **lm_head**：输出投影（Chinchilla 包含）
> - **transformer_matrices**：所有 Block 参数（QKV、FFN 等，核心参数）
> - **scalars**：resid_lambdas、x0_lambdas、smear 等（极少）
>
> 用途：`base_train.py` 用 `transformer_matrices + lm_head` 的参数量来计算 Chinchilla 最优训练 token 数。

---

### `setup_optimizer` — 优化器配置

```python
    def setup_optimizer(self, unembedding_lr=0.004, embedding_lr=0.2, matrix_lr=0.02, weight_decay=0.0, scalar_lr=0.5):
        model_dim = self.config.n_embd
        ddp, rank, local_rank, world_size = get_dist_info()
```
> 接收各组别学习率，根据是否 DDP 选择优化器类型。

```python
        matrix_params = list(self.transformer.h.parameters())
        value_embeds_params = list(self.value_embeds.parameters())
        embedding_params = list(self.transformer.wte.parameters())
        lm_head_params = list(self.lm_head.parameters())
        resid_params = [self.resid_lambdas]
        x0_params = [self.x0_lambdas]
        smear_params = [self.smear_gate.weight, self.smear_lambda, self.backout_lambda]
```
> 将模型参数分成 7 组，不同组使用不同超参数（LR、beta、weight_decay）。

```python
        dmodel_lr_scale = (model_dim / 768) ** -0.5
```
> **muP 风格的 LR 缩放**：
> $$\text{lr} \propto \frac{1}{\sqrt{d_{model}/768}}$$
> - 基准模型维度 768（d12 模型），其他大小模型的 LR 按 `1/√(d/768)` 缩放
> - 原理（来自 maximal update parametrization / muP）：更宽的模型需要更小的 LR 才能保持相同的激活值更新幅度

```python
        param_groups = [
            dict(kind='adamw', params=lm_head_params,      lr=unembedding_lr * dmodel_lr_scale, betas=(0.8, 0.96),  eps=1e-10, weight_decay=0.01),
            dict(kind='adamw', params=embedding_params,    lr=embedding_lr * dmodel_lr_scale,   betas=(0.8, 0.995), eps=1e-10, weight_decay=0.001),
            dict(kind='adamw', params=value_embeds_params, lr=embedding_lr * dmodel_lr_scale * 0.5, ...),
            dict(kind='adamw', params=resid_params,        lr=scalar_lr * 0.01, betas=(0.8, 0.95), ...),
            dict(kind='adamw', params=x0_params,           lr=scalar_lr, betas=(0.96, 0.95), ...),
            dict(kind='adamw', params=smear_params,        lr=0.2, ...),
        ]
```
> **AdamW 参数组设计**：
>
> | 组别 | LR | beta1 | beta2 | weight_decay | 说明 |
> |------|-----|-------|-------|-------------|------|
> | lm_head | unembedding_lr × scale | 0.8 | 0.96 | 0.01 | 输出投影，中等正则 |
> | wte | embedding_lr × scale | 0.8 | 0.995 | 0.001 | 词嵌入，弱正则（词汇稀疏） |
> | value_embeds | embedding_lr × scale × 0.5 | 0.8 | 0.995 | 0.01 | VE 用一半 LR，更保守 |
> | resid_lambdas | scalar_lr × 0.01 | 0.8 | 0.95 | 0.05 | 残差标量，极小 LR（敏感参数） |
> | x0_lambdas | scalar_lr | 0.96 | 0.95 | 0 | 嵌入混合，高 beta1 平滑 |
> | smear | 0.2（固定） | 0.8 | 0.95 | 0 | Smear 机制 |

```python
        for shape in sorted({p.shape for p in matrix_params}):
            group_params = [p for p in matrix_params if p.shape == shape]
            param_groups.append(dict(
                kind='muon', params=group_params, lr=matrix_lr,
                momentum=0.95, ns_steps=5, beta2=0.9, weight_decay=weight_decay,
            ))
```
> **Muon 参数组**：将 Transformer 矩阵按形状分组（Muon 需要对同形状参数做批量 Newton-Schulz 正交化）：
> - `momentum=0.95`：Muon 动量（会在训练过程中从 0.85 warmup 到 0.95/0.97）
> - `ns_steps=5`：Newton-Schulz 正交化迭代次数（5 步足以收敛）
> - `beta2=0.9`：用于 Muon 内部的二阶矩估计

```python
        Factory = DistMuonAdamW if ddp else MuonAdamW
        optimizer = Factory(param_groups)
        for group in optimizer.param_groups:
            group["initial_lr"] = group["lr"]
        return optimizer
```
> - DDP 模式用分布式版优化器（Muon 的正交化需要跨 rank 通信）
> - 为每个参数组保存 `initial_lr`，供学习率调度器乘以 multiplier 使用


---

### `forward` — 前向传播

```python
    def forward(self, idx, targets=None, kv_cache=None, loss_reduction='mean'):
        B, T = idx.size()
```
> 参数说明：
> - `idx`：token ID 张量，形状 `(B, T)`，dtype=int
> - `targets`：训练目标 token ID（-1 表示 ignore），为 None 时进入推理模式
> - `kv_cache`：推理时的 KV 缓存对象，训练时为 None
> - `loss_reduction`：loss 归约方式（`'mean'` 或 `'none'`，RL 训练时用 `'none'`）

```python
        assert T <= self.cos.size(1), ...
        assert idx.device == self.cos.device, ...
        assert self.cos.dtype == COMPUTE_DTYPE, ...
```
> 三条断言保证基本约束：序列长度不超过 RoPE 缓存、设备一致、RoPE 精度正确。

```python
        T0 = 0 if kv_cache is None else kv_cache.get_pos()
        cos_sin = self.cos[:, T0:T0+T], self.sin[:, T0:T0+T]
```
> KV Cache 偏移：推理时从缓存位置 `T0` 开始取 RoPE，而非从 0 开始。
> 这保证了第 `T0+t` 个 token 使用位置 `T0+t` 的旋转角，与训练时一致。

```python
        x = self.transformer.wte(idx)
        x = x.to(COMPUTE_DTYPE)
        x = norm(x)
```
> **Token Embedding + 初始 Norm**：
> - `wte(idx)`：查表得到 token 向量，形状 `(B, T, n_embd)`
> - `.to(COMPUTE_DTYPE)`：确保激活精度正确（fp16 路径下 embedding 是 fp32，需要转换）
> - `norm(x)`：**在 embedding 上做 RMSNorm**（GPT-2 没有这一步）
>   - 保证进入 Transformer 时特征幅度一致，避免 token 向量幅度差异影响注意力计算

```python
        if kv_cache is None:
            assert T > 1
            gate = self.smear_lambda.to(x.dtype) * torch.sigmoid(self.smear_gate(x[:, 1:, :24]))
            x = torch.cat([x[:, :1], x[:, 1:] + gate * x[:, :-1]], dim=1)
        else:
            x_pre_smear = kv_cache.prev_embedding
            kv_cache.prev_embedding = x[:, -1:, :]
            if T > 1:
                gate = self.smear_lambda.to(x.dtype) * torch.sigmoid(self.smear_gate(x[:, 1:, :24]))
                x = torch.cat([x[:, :1], x[:, 1:] + gate * x[:, :-1]], dim=1)
            elif x_pre_smear is not None:
                gate = self.smear_lambda.to(x.dtype) * torch.sigmoid(self.smear_gate(x[:, :, :24]))
                x = x + gate * x_pre_smear
```
> **Smear 机制**（两种路径）：
>
> **训练路径**：
> - `gate = smear_lambda × sigmoid(smear_gate(x[..., :24]))`：用前 24 个 embedding 通道计算混合门控
> - 第 0 个 token 不 smear（没有前一个），第 1+ 个 token：`x[t] += gate[t] × x[t-1]`
> - 效果：每个 token 知道一点前一个 token 的信息（廉价 bigram）
>
> **推理路径（KV Cache）**：
> - `x_pre_smear`：从 cache 读取上一步的 embedding（decode 时每步只有 1 个 token，无法用 slice 取前一个）
> - prefill（T>1）：与训练相同
> - decode（T=1）：用缓存的前一步 embedding 做 smear

```python
        x0 = x
        n_layer = self.config.n_layer
        backout_layer = n_layer // 2
        x_backout = None
        for i, block in enumerate(self.transformer.h):
            x = self.resid_lambdas[i] * x + self.x0_lambdas[i] * x0
            ve = self.value_embeds[str(i)](idx).to(x.dtype) if str(i) in self.value_embeds else None
            x = block(x, ve, cos_sin, self.window_sizes[i], kv_cache)
            if i == backout_layer:
                x_backout = x
```
> **Transformer 主循环**：
>
> - `x0 = x`：保存初始 embedding（归一化后），供所有层的 `x0_lambdas` 使用
> - `backout_layer = n_layer // 2`：中间层编号（用于 backout）
>
> **每层的操作**：
> 1. `resid_lambdas[i] * x`：对残差流幅度做可学习缩放
> 2. `+ x0_lambdas[i] * x0`：混入初始 embedding（让每层都能"看到"原始 token 信息）
> 3. 查 Value Embedding 表（若该层有 VE）
> 4. 通过 Block（注意力 + MLP + 残差）
> 5. 缓存中间层输出（用于 backout）

```python
        if x_backout is not None:
            x = x - self.backout_lambda.to(x.dtype) * x_backout
        x = norm(x)
```
> **Backout**：最终输出前减去中间层残差，去除低层特征（提升 logit 质量）。
> 然后做最终 RMSNorm。

```python
        softcap = 15
        logits = self.lm_head(x)
        logits = logits[..., :self.config.vocab_size]
        logits = logits.float()
        logits = softcap * torch.tanh(logits / softcap)
```
> **Logit 计算与 SoftCap**：
> - `lm_head(x)`：投影到词表空间，形状 `(B, T, padded_vocab_size)`
> - `[..., :vocab_size]`：裁剪掉 padding 的部分
> - `.float()`：转为 fp32，确保后续 cross_entropy 数值稳定
> - **SoftCap**：`15 × tanh(logits/15)`
>   - 将 logits 平滑限制在 `(-15, 15)` 范围内
>   - 防止极端 logit 值（如 100）导致 softmax 数值溢出
>   - `tanh` 在 |x| < 15 时近似线性，超出后平滑饱和（类似 clip 但可微）
>   - Gemini 1.5 等模型也使用了类似技术

```python
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1, reduction=loss_reduction)
            return loss
        else:
            return logits
```
> **训练 vs 推理的输出分叉**：
>
> **训练模式**（`targets is not None`）：
> - `logits.view(-1, vocab_size)`：展平为 `(B×T, vocab_size)`
> - `targets.view(-1)`：展平为 `(B×T,)`
> - `ignore_index=-1`：跳过 mask 掉的位置（用户输入、padding）
> - 返回标量 loss（或 `(B×T,)` 向量，当 `loss_reduction='none'` 时）
>
> **推理模式**（`targets is None`）：
> - 直接返回 logits `(B, T, vocab_size)`

---

### `generate` — 朴素自回归推理

```python
    @torch.inference_mode()
    def generate(self, tokens, max_tokens, temperature=1.0, top_k=None, seed=42):
```
> `@torch.inference_mode()`：推理模式，关闭所有梯度计算和自动微分追踪，比 `torch.no_grad()` 更彻底，节省显存和计算。

```python
        assert isinstance(tokens, list)
        device = self.get_device()
        rng = None
        if temperature > 0:
            rng = torch.Generator(device=device)
            rng.manual_seed(seed)
```
> - 输入 tokens 为 Python 列表（简单接口）
> - 确定性生成（`temperature=0`）不需要随机数生成器
> - `torch.Generator`：独立的随机数生成器，指定 seed 保证可复现

```python
        ids = torch.tensor([tokens], dtype=torch.long, device=device)
        for _ in range(max_tokens):
            logits = self.forward(ids)        # (B, T, vocab_size)
            logits = logits[:, -1, :]         # (B, vocab_size) 只取最后一个位置
```
> **朴素实现（无 KV Cache）**：每步都重新计算完整序列的前向传播。
> 复杂度 O(T²)，适合简单场景；高效推理使用 `Engine`（有 KV Cache）。

```python
            if top_k is not None and top_k > 0:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
```
> **Top-K 采样**：只保留概率最高的 K 个 token，其余设为 `-∞`（softmax 后概率为 0）。
> 防止模型采样到低概率的奇怪 token。

```python
            if temperature > 0:
                logits = logits / temperature
                probs = F.softmax(logits, dim=-1)
                next_ids = torch.multinomial(probs, num_samples=1, generator=rng)
            else:
                next_ids = torch.argmax(logits, dim=-1, keepdim=True)
```
> **温度采样 vs 贪心解码**：
> - `temperature > 0`：除以温度后做 softmax，再多项式采样
>   - 高温（>1）：分布更均匀，输出更多样（但可能乱）
>   - 低温（<1）：分布更尖锐，输出更保守（但可能重复）
> - `temperature = 0`：贪心解码（argmax），完全确定性

```python
            ids = torch.cat((ids, next_ids), dim=1)
            token = next_ids.item()
            yield token
```
> - 将新 token 拼接到序列尾部（下一步用）
> - `yield token`：生成器函数，逐 token 流式输出（调用方可以边生成边处理）

---

## 整体架构总结

```
输入 token IDs [B, T]
        ↓
  wte embedding → norm
        ↓
  Smear（混入前一token信息）
        ↓
  保存 x0（初始嵌入）
  ┌─────────────────────────────────────────┐
  │  for i in range(n_layer):               │
  │    x = resid_λ[i] * x + x0_λ[i] * x0  │  ← 可学习残差缩放 + 初始嵌入混合
  │    ve = value_embeds[i](idx)            │  ← Value Embedding（部分层）
  │    x = Block(x, ve, RoPE, window)       │  ← 注意力 + MLP
  │      ├── norm(x) → Attention(Q,K,V)    │
  │      │   ├── c_q/c_k/c_v 投影          │
  │      │   ├── RoPE 旋转 Q/K             │
  │      │   ├── QK Norm + ×1.2            │
  │      │   ├── VE gate + 混入 V          │
  │      │   └── Flash Attention（滑窗）   │
  │      └── norm(x) → MLP(relu²)          │
  │    if i == n_layer//2: x_backout = x   │  ← 缓存中间层
  └─────────────────────────────────────────┘
        ↓
  x = x - backout_λ * x_backout           ← 去除低层特征
        ↓
  norm(x)
        ↓
  lm_head → softcap(logits, 15)
        ↓
  训练: cross_entropy(logits, targets)
  推理: return logits
```

## 与 GPT-2 的主要架构差异对比

| 特性 | GPT-2 | nanochat/gpt.py |
|------|-------|-----------------|
| 位置编码 | 可学习绝对位置编码 | RoPE（无参数，相对位置） |
| 注意力类型 | MHA（所有头独立K/V） | GQA/MQA（K/V头数更少） |
| 注意力稳定 | 无 | QK Norm + ×1.2 |
| 注意力窗口 | 完整上下文 | 滑动窗口（SSSL模式） |
| 激活函数 | GELU | relu²（更稀疏） |
| 权重共享 | wte = lm_head（tied） | 不共享（untied） |
| Norm | LayerNorm（有γ/β） | RMSNorm（无参数） |
| Norm 位置 | Pre-Norm | Pre-Norm + embedding后Norm |
| 快速注意力 | 无 | Flash Attention 3 |
| 残差机制 | 标准 +1 | resid_λ 可学习缩放 |
| Token上下文 | 无 | x0混入 + Smear + VE |
| Logit处理 | 无 | SoftCap(15) |
| Bias | 有 | 全部去除 |
| 精度管理 | autocast | 显式 COMPUTE_DTYPE |
