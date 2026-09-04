# 一次 step 的 loss 生成过程（含 model 内部变化）

> 对应源码：`trainer/train_pretrain.py`（train_epoch）、`model/model_minimind.py`（forward）
> 关联：`train_pretrain详解.md`

本文用一条真实数据，把「一次 step 里张量从 `input_ids` 一路变成 loss 标量」的完整过程串起来。

---

## 0. 前置：`PretrainDataset` 已经产出的东西

在 `train_epoch` 拿到 batch 之前，`PretrainDataset.__getitem__` 已经把原始文本变成了 `(input_ids, labels)`。

假设 `max_seq_len=8`（真实默认 340，这里缩小便于演示），一条文本 `"今天天气不错"`：

```
tokenize("今天天气不错", add_special_tokens=False) → [10, 20, 30, 40]   # 示意 id
包 bos/eos:  [1] + [10, 20, 30, 40] + [2] = [1, 10, 20, 30, 40, 2]     # 6 个
右补齐:      [1, 10, 20, 30, 40, 2, 0, 0]                              # 补 2 个 pad(0)

input_ids = [1, 10, 20, 30, 40, 2, 0, 0]
labels    = [1, 10, 20, 30, 40, 2, -100, -100]   # pad 位置置 -100（不参与 loss）
```

`DataLoader` 把 32 条这样的样本堆成 batch：

```
input_ids : [32, 8]   （真实 [32, 340]）
labels    : [32, 8]
```

这就是 `train_epoch` 里 `for step, (input_ids, labels) in enumerate(loader, ...)` 拿到的两个张量。

---

## 1. 这一步在 `train_epoch` 里的位置

```python
input_ids = input_ids.to(args.device)     # 搬到 GPU
labels = labels.to(args.device)
lr = get_lr(...)                           # 算学习率

with autocast_ctx:                         # 混合精度前向
    res = model(input_ids, labels=labels)  # ★ 核心：这里就是下面要拆的 model 内部
    loss = res.loss + res.aux_loss         # aux_loss 非 MoE 时 = 0
    loss = loss / args.accumulation_steps  # 除以 8

scaler.scale(loss).backward()              # 反向传播
```

关键就是 `model(input_ids, labels=labels)` 这一行。它内部完成「`input_ids` → logits → loss」的全部计算。

---

## 2. `model.forward` 内部：`input_ids` → `logits`

### 2.1 词嵌入：`[32, 8]` → `[32, 8, 768]`

```python
# MiniMindModel.forward
hidden_states = self.dropout(self.embed_tokens(input_ids))
```

`embed_tokens` 是一张 `[6400, 768]` 的查表矩阵，第 i 行是词表第 i 个 token 的向量。

```
input_ids [32, 8]  --查表-->  hidden_states [32, 8, 768]
   每条样本 8 个 token，每个 token 展开成 768 维向量
```

以第一条样本为例，8 个 token id `[1, 10, 20, 30, 40, 2, 0, 0]` 分别取嵌入表第 1/10/20/30/40/2/0/0 行，堆叠成 `[8, 768]`。

### 2.2 一层 `MiniMindBlock` 内部（×8 层重复）

每一层做两件事：**注意力子层** + **FFN 子层**，各带一次残差。数据流（`hidden=768`）：

```
输入 x [32, 8, 768]
  ├─ 第1子层：注意力
  │    residual = x                                      (保存副本)
  │    attn_out = Attention( RMSNorm(x) )  → [32,8,768]
  │    x = attn_out + residual                           (残差相加)
  ├─ 第2子层：FFN
  │    mlp_out = FFN( RMSNorm(x) )  → [32,8,768]
  │    x = mlp_out + x                                   (残差相加)
  └─ 输出 [32, 8, 768]
```

**注意力子层内部**（GQA，8 个 Q 头 / 4 个 KV 头 / head_dim=96）：

```
x [32, 8, 768]
  --q_proj--> xq [32, 8, 8×96]  --view--> [32, 8, 8, 96]   (8 个 Q 头)
  --k_proj--> xk [32, 8, 4×96]  --view--> [32, 8, 4, 96]   (4 个 K 头)
  --v_proj--> xv [32, 8, 4×96]  --view--> [32, 8, 4, 96]   (4 个 V 头)
  --RMSNorm(head_dim)--> 保持形状
  --RoPE 旋转位置编码--> 保持形状          (给 Q/K 注入位置信息)
  --repeat_kv(xk, 2)--> [32, 8, 8, 96]     (4 个 K/V 头各复制 2 份对齐 8 个 Q 头)
  --transpose--> Q/K/V [32, 8, 8, 96]      (头维提前)

  scores = Q·Kᵀ / √96          → [32, 8, 8, 8]    (每个头一个 8×8 的注意力分数矩阵)
  scores += 因果 mask          → 上三角置 -inf     (当前 token 看不到未来)
  attn  = softmax(scores)      → [32, 8, 8, 8]     (每行和为 1 的注意力权重)
  output = attn · V            → [32, 8, 8, 96]
  --transpose+reshape--> [32, 8, 768] --o_proj--> [32, 8, 768]
```

**注意力打分（`scores`）的具体数值例子**（源码里的示例，单 head、head_dim=2、S=3，便于手算）：

```
Q = [[1,0],[0,1],[1,1]]   K = [[1,0],[1,1],[0,1]]

Q·Kᵀ      = [[1,1,0],[0,1,1],[1,2,1]]
÷√2(≈1.414) = [[0.707,0.707,0.000],[0.000,0.707,0.707],[0.707,1.414,0.707]]
加因果mask  = [[0.707,-inf,-inf],[0.000,0.707,-inf],[0.707,1.414,0.707]]
softmax    = [[1.000,0.000,0.000],[0.330,0.670,0.000],[0.248,0.504,0.248]]
                                  ↑ token0 只能看自己；token1 看 0、1；token2 看 0、1、2
output = softmax · V   # 每个 token 的输出 = 按权重加权求和各 token 的 V
```

真实场景就是把 head 数从 1 放大到 8、head_dim 从 2 放大到 96、S 从 3 放大到 340，原理完全相同。

**FFN 子层内部**（SwiGLU，`intermediate≈2432`）：

```
x [32, 8, 768]
  --gate_proj--> gate [32, 8, 2432]  --SiLU--> [32,8,2432]
  --up_proj----> up   [32, 8, 2432]
  --> SiLU(gate) ⊙ up --> [32, 8, 2432]
  --down_proj--> [32, 8, 768]
```

（SwiGLU 数值例子：`SiLU(z)=z·sigmoid(z)`，如 `SiLU(2.0)≈1.761`，负值会反转 up 的符号，见 `FeedForward` 源码注释。）

### 2.3 ×8 层 + 最终 RMSNorm：`[32, 8, 768]`

```
hidden_states [32, 8, 768]
  --MiniMindBlock × 8--> [32, 8, 768]   (形状不变，内容逐层变换)
  --final RMSNorm--> [32, 8, 768]
```

（若 `use_moe=1`，每层 FFN 换成 MoE，并额外累积 `aux_loss`；这里默认非 MoE，`aux_loss=0`。）

### 2.4 LM head：`[32, 8, 768]` → `[32, 8, 6400]` logits

```python
# MiniMindForCausalLM.forward
logits = self.lm_head(hidden_states)   # Linear(768, 6400)
```

```
hidden_states [32, 8, 768]  @  lm_head.weight^T [768, 6400]  =  logits [32, 8, 6400]
```

`logits[b, s, v]` = 第 b 条样本、第 s 个位置、「下一个 token 是词表第 v 个」的**未归一化分数**。取 softmax 就是概率分布。

---

## 3. loss 计算：`logits` → 标量（核心）

### 3.1 错位对齐（next-token 预测的关键）

```python
x, y = logits[..., :-1, :].contiguous(), labels[..., 1:].contiguous()
```

用「位置 i 预测位置 i+1」，所以 logits 去尾、labels 去头：

```
以一条样本为例（S=8）：
  位置        :  0    1    2    3    4    5    6    7
  input_ids   : [1,   10,  20,  30,  40,  2,   0,   0]
  labels      : [1,   10,  20,  30,  40,  2, -100,-100]

  x = logits[0:7]  → [7, 6400]   位置 0..6 的预测（去掉最后 1 个，因为它没有"下一个 token"）
  y = labels[1:8]  → [7]         位置 1..7 的真实标签（去掉第 1 个，它是"已知输入"）

  对齐关系：x[0] 预测 y[0](=10)、x[1] 预测 y[1](=20) ... x[4] 预测 y[4](=2)、x[5]/x[6] 对应 -100(被忽略)
```

### 3.2 展平 + 交叉熵（具体数值）

```python
loss = F.cross_entropy(x.view(-1, 6400), y.view(-1), ignore_index=-100)
```

```
x.view(-1, 6400) = [32×7, 6400] = [224, 6400]    (真实默认 S=340 时是 [32×339, 6400])
y.view(-1)       = [224]
```

对**每一个非 -100 的位置**，算一次「6400 分类的交叉熵」。举一个位置的例子（简化 vocab=5，真实 6400）：

```
某个位置 s 的 logits = [1.2, 0.8, 2.5, -0.3, 0.1]     (5 个类别的未归一化分数)
真实标签 y = 2                                        (第 3 个类别是对的)

softmax 归一化：
  e^1.2 = 3.320,  e^0.8 = 2.226,  e^2.5 = 12.183,  e^-0.3 = 0.741,  e^0.1 = 1.105
  分母 = 19.574
  概率 p = [0.1696, 0.1137, 0.6224, 0.0379, 0.0565]

交叉熵 = -ln(p[真实类别]) = -ln(0.6224) = 0.474
```

**直觉**：模型对正确答案的预测概率越高，loss 越小；预测概率 1.0 → loss=0，概率趋近 0 → loss 趋近 +∞。反向传播就是「提高正确答案的 logits、压低其它 6399 个」。

### 3.3 padding 忽略 + 求平均

`ignore_index=-100` 让 label 为 -100 的位置**跳过**，不参与求和也不计入分母。

```
一条样本错位后 7 个位置：
  y = [10, 20, 30, 40, 2, -100, -100]
            ↑ 前 5 个算 loss        ↑ 后 2 个(pad)被忽略

F.cross_entropy 默认 reduction='mean'：
  loss_单样本 = (CE0 + CE1 + CE2 + CE3 + CE4) / 5    ← 除以"有效位置数"5，不是 7
```

最终 `res.loss` 是整个 batch 所有有效位置的平均交叉熵，一个**标量**（0 维张量）。

### 3.4 `F.cross_entropy` 的输出形状

`F.cross_entropy` 的输出形状**取决于 `reduction` 参数**，MiniMind 代码里没传该参数，用的是默认值 `'mean'`，所以结果是**一个 0 维标量**（`torch.Size([])`）。

```python
loss = F.cross_entropy(x.view(-1, x.size(-1)), y.view(-1), ignore_index=-100)
#                                                      └─ 没传 reduction，默认 'mean'
```

- 输入 `x.view(-1, 6400)`：形状 `[N, C]` = `[10848, 6400]`（N 个位置，每个位置对 6400 个类打分）
- 输入 `y.view(-1)`：形状 `[N]` = `[10848]`（每个位置的真实类别索引）
- **输出 `loss`：标量，形状 `torch.Size([])`，即 0 维张量**

在 `train_epoch` 里能直接 `loss.item()`、`loss / 8`、`loss.backward()`，就是因为它是标量。

三个 `reduction` 模式对比（`cross_entropy` 内部 = `log_softmax` + `nll_loss`）：

| `reduction` | 输出形状 | 含义 |
|-------------|---------|------|
| `'none'`（不归约） | `[N]` | 每个位置各自的 loss，如 `[0.474, 1.203, 0.851, ...]` |
| `'sum'`（求和） | `[]` 标量 | 所有位置 loss 相加 |
| `'mean'`（平均，**默认**） | `[]` 标量 | 所有位置 loss 的平均（MiniMind 用的就是这个） |

具体例子（某一行 logits 简化为 5 类，`C=5`，真实类别索引 2）：

```python
logits = torch.tensor([1.2, 0.8, 2.5, -0.3, 0.1])   # [5] → 展平成 [1, 5]
target = torch.tensor([2])                            # 真实类别是第 2 类（索引 2）

F.cross_entropy(logits.unsqueeze(0), target, reduction='none')   # → tensor([0.4741])  形状 [1]
F.cross_entropy(logits.unsqueeze(0), target, reduction='sum')    # → tensor(0.4741)    形状 []
F.cross_entropy(logits.unsqueeze(0), target, reduction='mean')   # → tensor(0.4741)    形状 []
```

（`0.4741 = -ln(0.6224)`，见上文 3.2。）

当 N=10848 时，三种模式分别是：

```
reduction='none' → [10848]   每个位置一个 loss 值
reduction='sum'  → []        10848 个 loss 相加的总和
reduction='mean' → []        总和 / 有效位置数（忽略 -100 的位置不计入分母）
```

`ignore_index=-100` **不改变形状**（始终是 `[N]` 或标量），它只影响**数值**：label 等于 -100 的位置，该位置 loss 被置 0 且**不计入 `'mean'` 的分母**。

```
例（N=7，其中 2 个是 pad）：
  y = [10, 20, 30, 40, 2, -100, -100]
  loss_none = [0.47, 0.92, 1.10, 0.35, 0.63, 0.00, 0.00]   # 形状 [7]，pad 位置为 0
  loss_mean = (0.47+0.92+1.10+0.35+0.63) / 5 = 0.694       # 除以 5，不是 7
  loss_sum  = 0.47+0.92+1.10+0.35+0.63 = 3.47              # pad 位置贡献 0
```

### 3.5 为什么 mean/sum 之后反向传播也有效

**核心一句话**：反向传播本质要求 loss 必须是**标量**，而 `mean`/`sum` 都是「把向量归约成标量」的可微线性操作，梯度能顺畅流回去——它们只差一个常数因子，不影响优化方向。

#### (1) 为什么必须先归约成标量才能 `backward()`

`reduction='none'` 得到的是 `[N]` 的向量 loss。直接对这个**向量** `backward()`，PyTorch 会报错：

```
RuntimeError: grad can be implicitly created only for scalar outputs
```

原因：更新模型参数 `w` 需要的是**一个梯度向量** `∂L/∂w`（和 `w` 同形状，给每个参数指一个方向）。

- 若 `L` 是标量 → `∂L/∂w` 是梯度（向量），方向唯一，直接 `w -= lr·∂L/∂w`。
- 若 `L` 是 `[N]` 向量 → `∂L/∂w` 变成 **Jacobian 矩阵**（形状 `[N, |w|]`），N 个位置各给一个「更新方向」，互相矛盾，没有单一下降方向。

（PyTorch 支持对非标量 `backward(gradient=v)`，但那本质是让你手动指定 `v`，先算 `v·L` 把它点积成**标量**再反传——兜一圈还是要标量。）

所以 `mean`/`sum` 的作用，就是把 N 个位置的 loss「合成一个数」，从而得到一个统一的梯度方向。

#### (2) mean 和 sum 都是可微的线性归约

设 N 个位置的 loss 是 `l0, l1, ..., l_{N-1}`，每个都依赖参数 `w`：

```
sum:  L_sum  = l0 + l1 + ... + l_{N-1}
mean: L_mean = (l0 + l1 + ... + l_{N-1}) / N
```

两者都是「对 l_i 的线性组合」，由链式法则：

```
∂L_sum/∂w  = ∂l0/∂w + ∂l1/∂w + ... + ∂l_{N-1}/∂w      （每个位置权重 1）
∂L_mean/∂w = (1/N)(∂l0/∂w + ... + ∂l_{N-1}/∂w)         （每个位置权重 1/N）
```

**关键结论**：

```
∂L_mean/∂w = (1/N) · ∂L_sum/∂w
```

两者梯度**方向完全相同**，只差常数 `1/N`。

#### (3) 为什么差常数不影响优化

梯度下降更新公式：

```
w_new = w - lr · ∂L/∂w
```

梯度整体缩小 `1/N` 倍，只要把学习率放大 `N` 倍，就得到**完全一样**的更新：

```
w - lr · ∂L_mean/∂w = w - lr · (1/N)·∂L_sum/∂w = w - (lr/N) · ∂L_sum/∂w
```

所以「mean + 学习率 lr」和「sum + 学习率 lr/N」是**同一个优化过程**。常数因子不改变下降方向，只等价于换了个学习率尺度。

#### (4) 为什么实践中默认用 mean

| | `sum` | `mean` |
|---|---|---|
| 梯度尺度 | 随 N（batch×序列长）变化 | 与 N 无关，稳定 |
| 换 batch_size | 需重调学习率 | 不用重调 |
| 数值 | 大 batch 时 loss 巨大 | 始终在合理范围 |

`mean` 把 loss 归一化到「每个位置的平均损失」，无论 batch 是 32 还是 256，loss 量级都与「每个 token 平均学得多好」直接对应，学习率也通用。这也是几乎所有框架默认 `reduction='mean'` 的原因。

#### (5) 具体数值例子

假设 2 个位置，loss 依赖参数 `w`（`l0 = w²`，`l1 = (w-1)²`）：

```
sum:  L_sum  = w² + (w-1)²        → ∂L_sum/∂w  = 2w + 2(w-1) = 4w - 2
mean: L_mean = [w² + (w-1)²]/2    → ∂L_mean/∂w = (4w - 2)/2 = 2w - 1
```

在 `w=0` 处：`∂L_sum/∂w = -2`，`∂L_mean/∂w = -1`，正好差 2 倍（=N）；方向一致（都是负，往正方向走），两者通过不同 `lr` 能达到相同 `w`。

#### (6) 回到 MiniMind 代码

```python
loss = F.cross_entropy(x.view(-1, x.size(-1)), y.view(-1), ignore_index=-100)
#                     ↓ 默认 reduction='mean' → 标量
```

`cross_entropy` 内部 = `log_softmax(logits)` + `nll_loss`。`mean` 在这里是「对所有**有效**位置（非 -100）的平均」。梯度经 `log_softmax` → `logits` → `lm_head` → 主干网络一路回传，每个位置的 softmax 都会「抬高正确答案 logits、压低其它 6399 个」，再被 `1/N` 平均。

---

## 4. 回到 `train_epoch`：loss → 反向 → 累积 → 更新

```
res.loss          (标量，如 4.2351)
+ res.aux_loss    (非 MoE = 0)
= 4.2351
÷ accumulation_steps(8)
= 0.5294          ← 这一步真正 backward 的 loss

scaler.scale(0.5294).backward()
    ↓ 反向传播，梯度累加到所有参数的 .grad 上
    ↓（step 1~7 只累积不更新）

step == 8 时：
    unscale → clip_grad_norm(1.0) → optimizer.step() → zero_grad()
```

---

## 附：完整形状链速查表（默认配置，batch=32）

| 阶段 | 张量 | 形状 | 说明 |
|------|------|------|------|
| 数据 | `input_ids` / `labels` | `[32, 340]` | token id / 标签（pad 处 -100） |
| 词嵌入 | `hidden_states` | `[32, 340, 768]` | 查表 `[6400,768]` |
| 注意力 Q | `xq` | `[32, 340, 8, 96]` | 8 头 |
| 注意力 K/V | `xk` / `xv` | `[32, 340, 4, 96]` | 4 头（GQA，各复制 2 份） |
| 注意力分数 | `scores` | `[32, 8, 340, 340]` | Q·Kᵀ/√96，因果 mask 后 softmax |
| 注意力输出 | `output` | `[32, 340, 768]` | 加权求和 + o_proj |
| FFN 中间 | `gate`/`up` | `[32, 340, 2432]` | SwiGLU 升维 |
| ×8 层 + norm | `hidden_states` | `[32, 340, 768]` | 主干输出 |
| LM head | `logits` | `[32, 340, 6400]` | 每位置对词表打分 |
| 错位 | `x` / `y` | `[32, 339, 6400]` / `[32, 339]` | 去尾 / 去头 |
| 展平 | `x.view` / `y.view` | `[10848, 6400]` / `[10848]` | 10848 = 32×339 |
| 损失 | `loss` | `[]`（标量） | 平均交叉熵 |
