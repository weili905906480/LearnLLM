# nanoGPT 源码结构详解

> 仓库地址：[github.com/karpathy/nanoGPT](https://github.com/karpathy/nanoGPT)

## 目录总览

```
nanoGPT/
├── model.py                 # 🧠 核心：GPT 模型定义（全部在这一个文件）
├── train.py                 # 🏋️ 训练脚本（支持单卡/DDP 多卡）
├── sample.py                # 💬 推理/采样脚本（从训好的模型生成文本）
├── configurator.py          # ⚙️ 配置系统（命令行参数覆盖机制）
├── bench.py                 # 📊 性能基准测试
├── scaling_laws.ipynb       # 📈 缩放定律实验笔记本
├── transformer_sizing.ipynb # 📐 Transformer 参数计算笔记本
│
├── config/                  # 预设配置文件
│   ├── train_gpt2.py               # 训练 GPT-2 (124M) 的配置
│   ├── train_shakespeare_char.py   # 训练字符级 Shakespeare 模型
│   ├── finetune_shakespeare.py     # 微调 GPT-2 在 Shakespeare 上
│   ├── eval_gpt2.py                # 评估 GPT-2 (124M)
│   ├── eval_gpt2_medium.py         # 评估 GPT-2 Medium (350M)
│   ├── eval_gpt2_large.py          # 评估 GPT-2 Large (774M)
│   └── eval_gpt2_xl.py             # 评估 GPT-2 XL (1558M)
│
├── data/                    # 数据准备脚本
│   ├── openwebtext/
│   │   ├── prepare.py              # 下载并处理 OpenWebText (9B tokens)
│   │   └── readme.md
│   ├── shakespeare/
│   │   ├── prepare.py              # Shakespeare + GPT-2 BPE 分词
│   │   └── readme.md
│   └── shakespeare_char/
│       ├── prepare.py              # Shakespeare + 字符级分词
│       └── readme.md
│
└── assets/                  # README 用到的图片等资源
```

---

## 一、model.py — 模型定义（最核心）

**~300 行，完整定义了一个 GPT 模型**。

### 整体架构

```
输入 token ids: (batch, seq_len)
        ↓
┌─────────────────────────────────────────────┐
│  Token Embedding (wte)  +  Position Embedding (wpe)  │
│       vocab_size × n_embd      block_size × n_embd   │
└─────────────────────────────────────────────┘
        ↓
    Dropout
        ↓
┌─────────────────────────────────────────────┐
│  Transformer Block × n_layer                         │
│  ┌─────────────────────────────────────────┐        │
│  │  LayerNorm → CausalSelfAttention → residual     │
│  │  LayerNorm → MLP (FFN)            → residual     │
│  └─────────────────────────────────────────┘        │
└─────────────────────────────────────────────┘
        ↓
    LayerNorm (ln_f)
        ↓
    Linear (lm_head): n_embd → vocab_size
        ↓
输出 logits: (batch, seq_len, vocab_size)
```

### 组件详解

#### 1. GPTConfig（数据类）

```python
@dataclass
class GPTConfig:
    block_size: int = 1024      # 最大序列长度（上下文窗口）
    vocab_size: int = 50304     # 词表大小（GPT-2 50257 向上取整到 64 倍数）
    n_layer: int = 12           # Transformer 层数
    n_head: int = 12            # 注意力头数
    n_embd: int = 768           # 嵌入维度
    dropout: float = 0.0        # Dropout 比例
    bias: bool = True           # LayerNorm 和 Linear 是否用 bias
```

对应 GPT-2 各尺寸：
| 名称 | n_layer | n_head | n_embd | 参数量 |
|------|---------|--------|--------|--------|
| gpt2 | 12 | 12 | 768 | 124M |
| gpt2-medium | 24 | 16 | 1024 | 350M |
| gpt2-large | 36 | 20 | 1280 | 774M |
| gpt2-xl | 48 | 25 | 1600 | 1558M |

#### 2. LayerNorm

```python
class LayerNorm(nn.Module):
    """支持可选 bias 的 LayerNorm（PyTorch 原生不支持关闭 bias）"""
    def __init__(self, ndim, bias):
        self.weight = nn.Parameter(torch.ones(ndim))    # 缩放参数 γ
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None  # 偏移 β

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)
```

**作用**：对每个 token 的嵌入向量做归一化，稳定训练。

#### 3. CausalSelfAttention

```python
class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        # 一个 Linear 同时产出 Q, K, V（节省开销）
        self.c_attn = nn.Linear(n_embd, 3 * n_embd)  # → (Q, K, V)
        # 输出投影
        self.c_proj = nn.Linear(n_embd, n_embd)

    def forward(self, x):
        B, T, C = x.size()
        # 1. 计算 Q, K, V
        q, k, v = self.c_attn(x).split(n_embd, dim=2)
        # 2. 多头：reshape 为 (B, n_head, T, head_dim)
        q = q.view(B, T, n_head, C // n_head).transpose(1, 2)
        k = k.view(B, T, n_head, C // n_head).transpose(1, 2)
        v = v.view(B, T, n_head, C // n_head).transpose(1, 2)
        # 3. 注意力计算（自动用 Flash Attention 如果 PyTorch >= 2.0）
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        # 4. 合并多头，投影输出
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.c_proj(y)
        return y
```

**关键点**：
- **Causal mask**：确保位置 i 只能看到 ≤i 的 token（自回归）
- **Flash Attention**：PyTorch 2.0+ 自动启用，显存更少、速度更快
- **多头并行**：所有头共用一个大 Linear 计算 Q/K/V，再按头数 reshape

#### 4. MLP（前馈网络）

```python
class MLP(nn.Module):
    def __init__(self, config):
        self.c_fc   = nn.Linear(n_embd, 4 * n_embd)    # 升维 4 倍
        self.gelu   = nn.GELU()                         # 激活函数
        self.c_proj = nn.Linear(4 * n_embd, n_embd)     # 降回原维度

    def forward(self, x):
        x = self.c_fc(x)       # (B, T, n_embd) → (B, T, 4*n_embd)
        x = self.gelu(x)       # 非线性
        x = self.c_proj(x)     # (B, T, 4*n_embd) → (B, T, n_embd)
        return x
```

**作用**：对每个 token 独立做非线性变换，增加模型表达能力。

#### 5. Block（一个 Transformer 层）

```python
class Block(nn.Module):
    def forward(self, x):
        x = x + self.attn(self.ln_1(x))   # Pre-Norm + Attention + Residual
        x = x + self.mlp(self.ln_2(x))    # Pre-Norm + MLP + Residual
        return x
```

**Pre-LayerNorm**：先归一化再做注意力/MLP，训练更稳定。

#### 6. GPT 主类

关键方法：
| 方法 | 功能 |
|------|------|
| `__init__` | 组装所有组件，初始化权重，实施 weight tying |
| `forward(idx, targets)` | 前向传播，计算 logits 和 loss |
| `generate(idx, max_new_tokens)` | 自回归生成文本 |
| `from_pretrained(model_type)` | 加载 OpenAI GPT-2 预训练权重 |
| `configure_optimizers(...)` | 创建 AdamW，区分 decay/no-decay 参数 |
| `crop_block_size(block_size)` | 裁剪位置编码，减小上下文窗口 |
| `estimate_mfu(...)` | 估算模型浮点利用率（相对 A100 峰值） |

**Weight Tying**：
```python
self.transformer.wte.weight = self.lm_head.weight
# 输入嵌入和输出投影共享同一个权重矩阵
# 减少参数量，提升效果
```

**权重初始化**：
```python
# 普通层：N(0, 0.02)
# 残差投影层：N(0, 0.02 / sqrt(2 * n_layer))  ← 随深度缩小
```

---

## 二、train.py — 训练脚本

**~250 行，完成从数据加载到模型训练的全部逻辑。**

### 训练流程

```
1. 解析配置（configurator.py 覆盖默认值）
2. 初始化 DDP（如果多卡）
3. 加载数据（memmap 读取 .bin 文件）
4. 初始化模型（scratch / resume / gpt2 预训练权重）
5. torch.compile() 编译模型
6. 训练循环：
   ┌──────────────────────────────────────────┐
   │  for iter in range(max_iters):            │
   │    - 设置学习率（cosine with warmup）      │
   │    - 梯度累积 N 步                         │
   │    - 前向 → loss → 反向                    │
   │    - 梯度裁剪                              │
   │    - optimizer.step()                      │
   │    - 定期 eval + save checkpoint           │
   └──────────────────────────────────────────┘
```

### 关键设计

#### 数据加载（极简）

```python
def get_batch(split):
    data = np.memmap('data/openwebtext/train.bin', dtype=np.uint16, mode='r')
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([torch.from_numpy(data[i:i+block_size]) for i in ix])
    y = torch.stack([torch.from_numpy(data[i+1:i+1+block_size]) for i in ix])
    return x, y
```

- 不用 DataLoader，直接 memmap 随机采样
- 输入 x 和目标 y 相差一个位置（next-token prediction）
- `pin_memory()` + `non_blocking=True` 加速 CPU→GPU 传输

#### 学习率调度（Cosine with Warmup）

```python
def get_lr(it):
    if it < warmup_iters:           # 阶段1：线性升温
        return lr * (it+1) / (warmup_iters+1)
    if it > lr_decay_iters:         # 阶段3：保持最低 lr
        return min_lr
    # 阶段2：余弦衰减
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    coeff = 0.5 * (1.0 + cos(π * decay_ratio))
    return min_lr + coeff * (lr - min_lr)
```

```
学习率
  ↑
  │    /‾‾‾‾‾\
  │   /        \
  │  /          \‾‾‾‾‾  ← min_lr
  │ /
  └──────────────────→ 训练步数
    warmup   decay
```

#### 梯度累积

```python
for micro_step in range(gradient_accumulation_steps):
    logits, loss = model(X, Y)
    loss = loss / gradient_accumulation_steps  # 缩放 loss
    loss.backward()                            # 累积梯度
# 所有 micro_step 结束后再 optimizer.step()
```

**作用**：显存不够时，用多个小 batch 模拟大 batch。

#### DDP（分布式数据并行）

```python
# 启动：torchrun --nproc_per_node=8 train.py
if ddp:
    model = DDP(model, device_ids=[ddp_local_rank])
    # 自动把梯度在所有 GPU 之间平均
```

#### 混合精度训练

```python
ctx = torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16)
scaler = torch.cuda.amp.GradScaler(enabled=(dtype == 'float16'))

with ctx:
    logits, loss = model(X, Y)
scaler.scale(loss).backward()
scaler.step(optimizer)
```

- bfloat16：直接用，无需 scaler
- float16：需要 GradScaler 防止梯度下溢

### 默认超参数（训练 GPT-2 124M）

| 参数 | 值 | 说明 |
|------|---|------|
| batch_size | 12 | micro batch |
| gradient_accumulation_steps | 40 | 模拟大 batch |
| block_size | 1024 | 上下文长度 |
| learning_rate | 6e-4 | 最大学习率 |
| max_iters | 600,000 | 总步数 |
| warmup_iters | 2,000 | 升温步数 |
| weight_decay | 0.1 | 权重衰减 |
| grad_clip | 1.0 | 梯度裁剪 |
| dropout | 0.0 | 预训练不用 dropout |

---

## 三、sample.py — 采样/推理

**~80 行，加载模型后生成文本。**

### 流程

```
1. 加载 checkpoint 或 GPT-2 预训练权重
2. 编码 prompt → token ids
3. 调用 model.generate() 自回归生成
4. 解码 token ids → 文字输出
```

### 关键参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| temperature | 0.8 | <1 更确定，>1 更随机 |
| top_k | 200 | 只从概率最高的 k 个 token 中采样 |
| max_new_tokens | 500 | 最多生成多少个 token |
| num_samples | 10 | 生成几个样本 |

### generate() 的核心逻辑

```python
for _ in range(max_new_tokens):
    # 1. 裁剪到 block_size（超出上下文窗口时）
    idx_cond = idx[:, -block_size:]
    # 2. 前向得到 logits
    logits, _ = model(idx_cond)
    # 3. 取最后一个位置的 logits，除以 temperature
    logits = logits[:, -1, :] / temperature
    # 4. Top-K 过滤
    if top_k:
        v, _ = torch.topk(logits, top_k)
        logits[logits < v[:, [-1]]] = -inf
    # 5. Softmax → 概率分布
    probs = F.softmax(logits, dim=-1)
    # 6. 采样下一个 token
    idx_next = torch.multinomial(probs, num_samples=1)
    # 7. 拼接到序列末尾
    idx = torch.cat((idx, idx_next), dim=1)
```

---

## 四、configurator.py — 配置系统

**~30 行，极简但巧妙的设计。**

### 机制

```python
# 在 train.py 中通过 exec() 执行：
exec(open('configurator.py').read())
```

支持两种覆盖方式：

```bash
# 方式1：指定配置文件（exec 执行，覆盖全局变量）
python train.py config/train_gpt2.py

# 方式2：命令行 --key=value
python train.py --batch_size=32 --learning_rate=1e-4

# 组合使用
python train.py config/train_gpt2.py --batch_size=64
```

**设计哲学**：Karpathy 认为传统的配置框架（argparse、hydra）太复杂。直接用 Python 变量 + exec 覆盖，最简单也最灵活。

---

## 五、bench.py — 性能基准测试

**用途**：测试单次迭代速度和 MFU（模型浮点利用率）。

### 两种模式

1. **简单计时**：跑 20 步，计算平均每步时间 + MFU
2. **PyTorch Profiler**：生成详细的 profiling trace（可在 TensorBoard 查看）

### MFU 计算

```python
def estimate_mfu(self, fwdbwd_per_iter, dt):
    # FLOPs 估算（参考 PaLM 论文）
    flops_per_token = 6*N + 12*L*H*Q*T
    flops_per_iter = flops_per_token * T * fwdbwd_per_iter
    # 与 A100 峰值的比值
    mfu = (flops_per_iter / dt) / 312e12  # A100 bf16 = 312 TFLOPS
    return mfu
```

好的 MFU 约 40-60%，说明硬件利用率高。

---

## 六、config/ — 预设配置

### train_gpt2.py（完整训练）

```python
# 8×A100 训练约 5 天
batch_size = 12
block_size = 1024
gradient_accumulation_steps = 5 * 8  # = 40
# 总 batch: 12 * 1024 * 40 = ~491K tokens/iter
max_iters = 600000                   # 600K步 × 491K = ~300B tokens
```

### train_shakespeare_char.py（快速实验）

```python
# 字符级小模型，MacBook 也能跑
n_layer = 6, n_head = 6, n_embd = 384  # "baby GPT"
block_size = 256
batch_size = 64
max_iters = 5000  # 几分钟跑完
dropout = 0.2     # 小数据集需要正则化
```

### finetune_shakespeare.py（微调）

```python
init_from = 'gpt2-xl'        # 加载 GPT-2 XL (1.5B) 预训练权重
learning_rate = 3e-5          # 微调用更小的 lr
max_iters = 20                # 只需 20 步（数据很小）
decay_lr = False              # 不衰减
```

---

## 七、data/ — 数据准备

### openwebtext/prepare.py

```python
# 1. 从 HuggingFace 下载 OpenWebText (8M 文档)
dataset = load_dataset("openwebtext")
# 2. 用 tiktoken GPT-2 BPE 分词
ids = enc.encode_ordinary(text)
# 3. 拼接所有文档，存为 uint16 二进制文件
# 输出：train.bin (~17GB, 9B tokens)
#       val.bin   (~8.5MB, 4M tokens)
```

### shakespeare/prepare.py

```python
# 1. 下载 Tiny Shakespeare (~1MB)
# 2. 用 GPT-2 BPE 分词
# 3. 90/10 train/val 划分
# 输出：train.bin (301K tokens), val.bin (36K tokens)
```

### shakespeare_char/prepare.py

```python
# 1. 下载 Tiny Shakespeare
# 2. 字符级分词（65 个唯一字符）
# 3. 保存 meta.pkl（含 stoi/itos 映射）
# 输出：train.bin (1M chars), val.bin (111K chars)
```

**数据格式统一**：所有数据都存为 `np.uint16` 的 flat binary 文件，训练时用 `np.memmap` 直接读取，零额外开销。

---

## 八、设计哲学总结

| 原则 | 体现 |
|------|------|
| **极简** | 模型 300 行，训练 250 行，整个项目 <1000 行有效代码 |
| **透明** | 不用任何高层封装，所有细节都可见 |
| **可跑** | Shakespeare 实验几分钟就能跑通 |
| **可扩展** | 同一份代码无修改即可从 MacBook 扩到 8×A100 |
| **实用** | 能复现 GPT-2 级别性能（loss ~2.85） |

### 代码量统计

| 文件 | 行数 | 职责 |
|------|------|------|
| model.py | ~300 | 模型定义 |
| train.py | ~250 | 训练逻辑 |
| sample.py | ~80 | 推理采样 |
| configurator.py | ~30 | 配置系统 |
| bench.py | ~100 | 性能测试 |
| data/*/prepare.py | ~50 each | 数据准备 |
| **总计** | **~900** | **一个完整的 GPT 训练框架** |

---

## 九、如何运行

```bash
# 1. 准备数据（字符级 Shakespeare，最快）
python data/shakespeare_char/prepare.py

# 2. 训练（单 GPU）
python train.py config/train_shakespeare_char.py

# 3. 采样
python sample.py --out_dir=out-shakespeare-char

# 4. 或者直接用 GPT-2 预训练权重微调
python train.py config/finetune_shakespeare.py

# 5. 评估 GPT-2
python train.py config/eval_gpt2.py
```

---

## 十、整体数据流

```
原始文本（Shakespeare / OpenWebText）
        ↓  data/*/prepare.py
二进制 token 文件（train.bin, val.bin）
        ↓  train.py: get_batch()
随机采样 (batch_size, block_size) 的 token 序列
        ↓  model.forward()
Token Embedding + Position Embedding
        ↓
n_layer × (Attention + MLP)
        ↓
logits → CrossEntropyLoss(logits, targets)
        ↓  loss.backward()
梯度 → AdamW.step() → 更新权重
        ↓  重复 600K 次
训练好的 checkpoint (ckpt.pt)
        ↓  sample.py
加载模型 → generate() → 输出文本
```
