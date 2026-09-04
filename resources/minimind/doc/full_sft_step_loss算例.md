# `train_full_sft.py` 一次 step 的 loss 生成算例（矩阵形状变化）

> 对应源码：`trainer/train_full_sft.py` 的 `train_epoch` 函数
> 配套文档：`train_full_sft详解.md`

本文用**具体数字**把 `train_epoch` 里一次 step 从数据到 loss 的完整过程走一遍，每一步都标出张量形状。为了能在一行放下，故意把 batch 和序列取小（`B=2, S=10`），括号里标注真实默认值。

---

## 0. 参数设定（小数字便于手算，括号内为真实默认）

| 符号 | 取值 | 含义 |
|---|---|---|
| `B` (batch_size) | 2（真实 16） | 一个 step 的样本数 |
| `S` (max_seq_len) | 10（真实 768） | 每条序列长度 |
| `V` (vocab_size) | 6400 | 词表大小 |
| `H` (hidden_size) | 768 | 隐藏维 |
| `L` (num_hidden_layers) | 8 | 层数 |
| `accumulation_steps` | 1 | 不累积 |

MiniMind 的特殊 token：`<|im_start|>`=1、`<|im_end|>`=2、`<pad>`=0；`bos_id`（回答段开头记号 `assistant\n`）=`[1,10]`，`eos_id`=`[2]`。

---

## 1. DataLoader 产出：`input_ids` 和 `labels`

`for step, (input_ids, labels) in enumerate(loader)` 拿到两个 `long` 张量。

**两条样本**（已经是 SFTDataset 处理完、补齐到长度 10 的结果）：

```
样本0 "你好 → 你好"：
  <|im_start|> user 你 好 <|im_end|> <|im_start|> assistant 你 好 <|im_end|>
  id:  1  20 30 40  2   1  10 30 40  2

样本1 "你是谁 → 我"：
  <|im_start|> user 你 是 谁 <|im_end|> <|im_start|> assistant 我 <|im_end|>
  id:  1  22 30 31 32  2   1  10 33  2
```

```python
input_ids: shape [B, S] = [2, 10]
labels   : shape [B, S] = [2, 10]
```

具体矩阵（labels 里只有 assistant 回答段填真实 id，其余全 -100）：

```
input_ids = [[ 1, 20, 30, 40,  2,  1, 10, 30, 40,  2],   # 样本0
             [ 1, 22, 30, 31, 32,  2,  1, 10, 33,  2]]   # 样本1

labels    = [[-100,-100,-100,-100,-100,-100,-100, 30, 40,  2],  # 样本0：idx7,8,9 真实
             [-100,-100,-100,-100,-100,-100,-100,-100, 33,  2]] # 样本1：idx8,9 真实
```

---

## 2. 数据上卡

```python
input_ids = input_ids.to(args.device)   # [2,10] 形状不变，只是搬到 GPU
labels    = labels.to(args.device)      # [2,10]
```

---

## 3. 学习率（标量，不涉及矩阵）

```python
lr = get_lr(epoch * iters + step, args.epochs * iters, args.learning_rate)
# 例：epoch=0, iters=100, step=1 -> lr = 1e-5*(0.1+0.45*(1+cos(π*1/200))) ≈ 0.9999e-5
```

---

## 4. 前向：`res = model(input_ids, labels=labels)`

进入 `MiniMindForCausalLM.forward`，**形状变化链**如下：

### 4.1 词嵌入：`embed_tokens`（`nn.Embedding(6400, 768)`）

```python
hidden_states: [2, 10]  →  [2, 10, 768]
# 每个 token id 查表变成 768 维向量：2×10 个 id，每个 768 维
```

### 4.2 穿过 8 层 Transformer（每层输出形状不变）

每层内部：

```python
注意力: [2,10,768] → Q/K/V 投影 → 注意力 → o_proj → [2,10,768]
        (q_proj: [2,10,768]→[2,10,768]；k/v_proj: [2,10,768]→[2,10,384]，kv_heads=4×head_dim=96)
MLP:    [2,10,768] → gate/up_proj → [2,10,2432] → down_proj → [2,10,768]
        (intermediate_size = ceil(768·π/64)·64 = 2432)
```

经过 8 层后：

```python
hidden_states: [2, 10, 768]   # 形状自始至终不变
```

### 4.3 最终 RMSNorm

```python
hidden_states = self.norm(hidden_states)   # [2,10,768] → [2,10,768]
```

### 4.4 LM head：`nn.Linear(768, 6400)`

```python
logits: [2, 10, 768]  →  [2, 10, 6400]
# 每个位置、每个样本都得到 6400 个「词表分数」，softmax 后即下一个 token 的概率分布
```

### 4.5 aux_loss（MoE 负载均衡损失）

非 MoE 时（`use_moe=0`）：`aux_loss` 是 **0-d 标量张量**，值为 0。

```python
aux_loss: shape []  (标量 0)
```

---

## 5. next-token 错位（loss 的核心）

```python
x, y = logits[..., :-1, :], labels[..., 1:]
x = x.contiguous()          # [2, 9, 6400]
y = y.contiguous()          # [2, 9]
```

- `logits` 去掉最后一个位置（位置 9 没有「下一个 token」可预测）→ `[2, 9, 6400]`
- `labels` 去掉第一个位置（位置 0 是已知输入 `<|im_start|>`，不用预测）→ `[2, 9]`

**逐位置对齐**（样本 0，用位置 i 预测位置 i+1 的 token）：

| x 的行号 i | 看过哪个 token (input_ids[i]) | 预测 y[i]=labels[i+1] | 是否算 loss |
|---|---|---|---|
| 0 | `<|im_start|>` (1) | labels[1]=-100 | ❌ 忽略 |
| 1 | user (20) | -100 | ❌ |
| 2 | 你 (30) | -100 | ❌ |
| 3 | 好 (40) | -100 | ❌ |
| 4 | `<|im_end|>` (2) | -100 | ❌ |
| 5 | `<|im_start|>` (1) | -100 | ❌ |
| **6** | **assistant (10)** | **labels[7]=30「你」** | ✅ |
| **7** | **你 (30)** | **labels[8]=40「好」** | ✅ |
| **8** | **好 (40)** | **labels[9]=2 `<|im_end|>`** | ✅ |

> 关键点：`logits[6]` 是在「看到了 `<|im_start|>assistant`」之后产生的预测，它要对准「回答的第一个字『你』」。这正是 `labels[7]`，因为错位后 `y[6]=labels[7]`。其余用户/系统提示位置因为 labels=-100 全部被 `ignore_index` 跳过。

两个样本合计：9×2 = 18 个预测位置，其中只有 **5 个是真实的**（样本0 有 3 个 + 样本1 有 2 个）。

---

## 6. 交叉熵损失（展平计算）

```python
loss = F.cross_entropy(x.view(-1, x.size(-1)), y.view(-1), ignore_index=-100)
```

形状变化：

```python
x.view(-1, 6400): [2,9,6400]  →  [18, 6400]   # 18 个位置、每个 6400 维分数
y.view(-1)      : [2,9]       →  [18]         # 18 个目标 token id
```

`ignore_index=-100` 会把 18 个位置里 labels=-100 的 13 个位置丢掉，只对剩下 **5 个真实位置**求交叉熵并取平均。

**单个位置的数值算例**（为手算把词表从 6400 简化到 5）：

假设某真实位置的 logits = `[2.0, 1.0, 0.1, 0.5, -1.0]`，目标类 = 0：

```
softmax 分母 = e^2.0 + e^1.0 + e^0.1 + e^0.5 + e^-1.0
            = 7.389 + 2.718 + 1.105 + 1.649 + 0.368 = 13.229
p(类0) = 7.389 / 13.229 = 0.5586
该位置 loss = -log(0.5586) = 0.582
```

真实训练里：对 5 个真实位置各算一个 `-log(p)`，求平均 → `res.loss`（**0-d 标量**）。

---

## 7. 加 aux_loss、除累积步数

```python
loss = res.loss + res.aux_loss        # 标量 + 标量 0 = 标量（本例 aux_loss=0）
loss = loss / args.accumulation_steps # 标量 / 1 = 标量（本例不变）
```

```python
res.loss    : shape []  (标量，比如 2.314)
res.aux_loss: shape []  (标量 0)
loss        : shape []  (标量 2.314)
```

---

## 8. 反向、裁剪、更新（形状都是标量/无形状的梯度）

```python
scaler.scale(loss).backward()      # loss 是标量，backward 给每个参数算出 .grad
scaler.unscale_(optimizer)         # 梯度反缩放
clip_grad_norm_(model.parameters(), 1.0)  # 所有参数梯度按范数整体裁剪
scaler.step(optimizer); scaler.update(); optimizer.zero_grad()
```

---

## 9. 全链路形状变化总表

| 阶段 | 操作 | 输入形状 → 输出形状 |
|---|---|---|
| 取数据 | `enumerate(loader)` | — → `input_ids [2,10]`, `labels [2,10]` |
| 上卡 | `.to(device)` | `[2,10]` → `[2,10]`（不变） |
| 词嵌入 | `embed_tokens` | `[2,10]` → `[2,10,768]` |
| 8 层 Transformer | 注意力 + MLP | `[2,10,768]` → `[2,10,768]` |
| 最终归一化 | `RMSNorm` | `[2,10,768]` → `[2,10,768]` |
| LM head | `lm_head(768→6400)` | `[2,10,768]` → `[2,10,6400]` |
| 错位 | `logits[:-1]`, `labels[1:]` | `[2,10,6400]`→`[2,9,6400]`；`[2,10]`→`[2,9]` |
| 展平 | `.view(-1, ...)` | `[2,9,6400]`→`[18,6400]`；`[2,9]`→`[18]` |
| 求 loss | `cross_entropy(ignore=-100)` | `[18,6400]`+`[18]` → **标量 `[]`** |
| 加 aux、除累积 | `loss+aux_loss`、`/N` | 标量 `[]` → 标量 `[]` |

**一句话总结形状的主线**：`[2,10]`（token id）→ `[2,10,768]`（嵌入/隐藏）→ `[2,10,6400]`（词表分数）→ 错位成 `[2,9,6400]` 与 `[2,9]` 对齐 → 展平成 `[18,6400]` vs `[18]` → **1 个标量 loss**。
