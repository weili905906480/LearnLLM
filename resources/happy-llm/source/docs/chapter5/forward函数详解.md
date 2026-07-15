# Chapter5 forward 函数详解

## 1. forward 函数的调用位置

在 `k_model.py` 中，`forward` 函数有以下调用位置：

### 1.1 Transformer.forward (第403行)

**内部调用**：
- `Transformer.generate` 方法：`self(idx_cond, attention_mask=mask_cond).logits` (第492行)
- `Transformer._beam_search` 方法：`self(beam_cond)` (第615行)
- `Transformer.generate_super` 方法：`self(idx_cond, attention_mask=mask_cond).logits` (第742行)
- `__main__` 测试代码：`output = model(X, Y)` (第803行)

### 1.2 DecoderLayer.forward (第301行)

- 在 `Transformer.forward` 中通过 `layer(h, freqs_cos, freqs_sin, attention_mask=attention_mask)` 调用 (第430行)

### 1.3 Attention.forward 和 MLP.forward

- 在 `DecoderLayer.forward` 中显式调用：`self.attention.forward(...)` 和 `self.feed_forward.forward(...)` (第305-306行)

### 1.4 其他文件中的调用

- `ddp_pretrain.py`：`out = model(X, Y)` (第101行)
- `ddp_sft_full.py`：`out = model(X, Y)` (第66行)
- `model_sample.py`：通过 `self.model.generate(...)` 间接调用 (第94、129行)

---

## 2. X, Y 的形状

### 2.1 数据集处理 (`dataset.py` 第42-44行)

```python
input_id = tokenizer(text)[:max_length]  # 长度为 max_length (默认512)
X = input_id[:-1]  # 长度为 511
Y = input_id[1:]   # 长度为 511
```

### 2.2 DataLoader batch 后 (`ddp_pretrain.py` 第86行)

- `X`: `(batch_size, max_seq_len - 1)` → `(64, 511)`
- `Y`: `(batch_size, max_seq_len - 1)` → `(64, 511)`
- `loss_mask`: `(batch_size, max_seq_len - 1)` → `(64, 511)`

其中 `max_seq_len = lm_config.max_seq_len = 512`（`k_model.py` 第26行定义）。

---

## 3. 为什么 targets 不需要经过 embeddings

`targets` 是**标签（token ID）**，不是输入，不需要经过 embedding。

```python
# tokens 是输入，需要 embedding
h = self.tok_embeddings(tokens)

# 模型输出 logits
logits = self.output(h)  # shape: (batch, seq_len, vocab_size)

# targets 是标签，直接用于计算交叉熵损失
self.last_loss = F.cross_entropy(
    logits.view(-1, logits.size(-1)),  # (batch*seq_len, vocab_size)
    targets.view(-1),                   # (batch*seq_len,) - 类别索引
    ignore_index=ignore_index
)
```

`F.cross_entropy` 需要的是：
- `logits`: 模型预测的概率分布 `(N, C)`
- `target`: 正确类别的**索引** `(N,)`，即 token ID

---

## 4. tokens, targets, logits 形状举例

假设 `batch_size=2`, `seq_len=4`, `vocab_size=6144`：

**tokens** (输入)：`shape = (2, 4)`
```
[[101, 2054, 102, 0],    # "你好吗<pad>"
 [101, 3054, 102, 0]]    # "今天好<pad>"
```

**targets** (标签，即 tokens 右移一位)：`shape = (2, 4)`
```
[[2054, 102, 0, 0],      # "好吗<pad><pad>"
 [3054, 102, 0, 0]]      # "天好<pad><pad>"
```

**logits** (模型输出，每个位置对 vocab_size 的预测)：`shape = (2, 4, 6144)`
```
[
  [  # 第1个样本
    [0.1, -0.3, 0.5, ...],   # 位置0预测 → 对比 target=2054
    [0.2, 0.8, -0.1, ...],   # 位置1预测 → 对比 target=102
    [-0.5, 0.3, 0.7, ...],   # 位置2预测 → 对比 target=0 (pad)
    [0.1, -0.2, 0.4, ...],   # 位置3预测 → 对比 target=0 (pad)
  ],
  [  # 第2个样本
    ...
  ]
]
```

---

## 5. logits.view(-1, logits.size(-1)) 操作

假设 `logits.shape = (2, 4, 6144)`：

**操作前**：3D 张量
```
logits = [
  [  # batch 0
    [0.1, -0.3, ...],   # 位置0: 6144个值
    [0.2, 0.8, ...],    # 位置1: 6144个值
    [-0.5, 0.3, ...],   # 位置2: 6144个值
    [0.1, -0.2, ...],   # 位置3: 6144个值
  ],
  [  # batch 1
    [0.3, -0.1, ...],   # 位置0: 6144个值
    [0.4, 0.6, ...],    # 位置1: 6144个值
    [-0.2, 0.5, ...],   # 位置2: 6144个值
    [0.7, -0.4, ...],   # 位置3: 6144个值
  ]
]
```

**`logits.view(-1, logits.size(-1))`**：把前两维合并

- `logits.size(-1)` = 6144
- `-1` 表示自动计算：`2 * 4 = 8`

**操作后**：2D 张量 `shape = (8, 6144)`
```
logits = [
  [0.1, -0.3, ...],   # batch0-位置0
  [0.2, 0.8, ...],    # batch0-位置1
  [-0.5, 0.3, ...],   # batch0-位置2
  [0.1, -0.2, ...],   # batch0-位置3
  [0.3, -0.1, ...],   # batch1-位置0
  [0.4, 0.6, ...],    # batch1-位置1
  [-0.2, 0.5, ...],   # batch1-位置2
  [0.7, -0.4, ...],   # batch1-位置3
]
```

**目的**：`F.cross_entropy` 要求输入是 `(N, C)` 格式，N 是样本数，C 是类别数。这样每个位置的预测可以和对应的 target 计算损失。

---

## 6. targets 的 reshape

代码中 targets 也做了 reshape：

```python
self.last_loss = F.cross_entropy(
    logits.view(-1, logits.size(-1)),  # (8, 6144)
    targets.view(-1),                   # (8,)
    ignore_index=ignore_index,
    reduction='none'
)
```

**示例**：

`targets.shape = (2, 4)`
```
targets = [
  [2054, 102, 0, 0],    # batch 0
  [3054, 102, 0, 0],    # batch 1
]
```

`targets.view(-1)` → `shape = (8,)`
```
targets = [2054, 102, 0, 0, 3054, 102, 0, 0]
```

**对应关系**：
```
logits[0] → targets[0] = 2054   # batch0-位置0 的预测 vs 真实
logits[1] → targets[1] = 102    # batch0-位置1 的预测 vs 真实
...
logits[7] → targets[7] = 0      # batch1-位置3 的预测 vs 真实
```

两者 reshape 后长度一致，才能逐元素计算交叉熵损失。

---

## 7. 6144 维向量与 targets 的交叉熵计算

`F.cross_entropy` 内部会自动处理，过程如下：

**输入**：
```
logits[0] = [0.1, -0.3, 0.5, ..., 0.8, ...]  # 6144个分数
targets[0] = 2054                              # 正确词的索引
```

**计算步骤**：

1. **Softmax**：将 6144 个分数转为概率分布
```
probs = softmax(logits[0]) = [0.02, 0.01, 0.03, ..., 0.05, ...]
                              索引: 0    1    2   ...  2054 ...
```

2. **取出目标概率**：取索引 2054 处的概率
```
p = probs[2054] = 0.05
```

3. **计算负对数似然**：
```
loss = -log(p) = -log(0.05) ≈ 3.0
```

**直观理解**：
```
logits:    "我认为下一个词是各个词的分数"
targets:   "正确答案是第2054号词"
cross_entropy: "你给第2054号词打的分越高，loss越小"
```

---

## 8. targets 的数据类型

`targets` 支持两种格式：

### 8.1 长整型索引（代码中使用的方式）

```python
# shape: (N,)
targets = torch.tensor([2054, 102, 0, 0], dtype=torch.long)

F.cross_entropy(logits, targets)
# logits.shape: (N, 6144)
# targets.shape: (N,)
```

### 8.2 one-hot 编码（也可以）

```python
# shape: (N, 6144)
targets_onehot = torch.zeros(4, 6144)
targets_onehot[0][2054] = 1
targets_onehot[1][102] = 1

F.cross_entropy(logits, targets_onehot)
# logits.shape: (N, 6144)
# targets_onehot.shape: (N, 6144)
```

### 8.3 PyTorch 内部判断

```python
if targets.dim() == 1:
    # 长整型索引模式
    return nll_loss(log_softmax(logits), targets)
else:
    # one-hot 或软标签模式
    return nll_loss(log_softmax(logits), targets.argmax(dim=1))
```

**推荐用索引**：更省内存，计算更快。代码中 `targets` 的 dtype 是 `torch.int64`（即 long）。
