# dpo_loss 数值算例：一个 step 的完整追踪（batch=1, len=4, vocab_size=8）

本文用一个具体的小例子，把 `train_epoch` 中**一个训练 step** 从头到尾走一遍：数据 → 前向 logits → `logits_to_log_probs`（log_softmax + gather）→ `dpo_loss`（mask 求和 → 切开 chosen/rejected → 对数几率差 → sigmoid → loss）。

> 相关文档：损失函数逐行注释见 [[dpo_loss详解]]，数据集见 [[DPODataset详解]]，log_softmax 算法见 [[log_softmax详解]]。

## 设定

- **词表 8 个 token**（id 0~7）：`3 = 坏答案`、`7 = 好答案`、`5/6 = 上下文词`
- **真实 batch=1** → 拼接后 2 行，第 0 行 chosen、第 1 行 rejected
- **len=4**：4 个位置，其中回答段 = 位置 2、3（mask=1），位置 0、1 是 prompt（mask=0）

---

## 第 1 步：取 batch 数据（`train_epoch` 开头）

```python
x_chosen   = [3,5,6,7]   y_chosen   = [5,6,7,7]   mask_chosen   = [0,0,1,1]
x_rejected = [3,5,6,3]   y_rejected = [5,6,3,3]   mask_rejected = [0,0,1,1]
```

## 第 2 步：拼接（`torch.cat`）

```python
x = torch.cat([x_chosen, x_rejected], dim=0)     # (2,4)
y = torch.cat([y_chosen, y_rejected], dim=0)     # (2,4)
mask = torch.cat([mask_chosen, mask_rejected], dim=0)
```

```
x    = [[3,5,6,7],      y    = [[5,6,7,7],      mask = [[0,0,1,1],
       [3,5,6,3]]             [5,6,3,3]]             [0,0,1,1]]
```

## 第 3 步：前向出 logits（`ref_model(x)` / `model(x)`）

输出 logits shape `(2, 4, 8)`。为便于手算，每位置的 logit 向量固定：

```
ref_logits    每位置 [0,1,2,3,4,5,6,7]
policy_logits chosen 行  [0,1,2,3,4,5,6,9]   ← token7(好) +2
              rejected 行 [0,1,2,1,4,5,6,7]  ← token3(坏) -2
```

## 第 4 步：`logits_to_log_probs`

### 4.1 log_softmax（dim=2，词表维度）

三种 logit 向量的 logsumexp 完整展开如下：

**① ref `[0,1,2,3,4,5,6,7]`：**

```
logsumexp = ln( e^0 + e^1 + e^2 + e^3 + e^4 + e^5 + e^6 + e^7 )
         = ln( 1.000000 + 2.718282 + 7.389056 + 20.085537
             + 54.598150 + 148.413159 + 403.428793 + 1096.633158 )
         = ln( 1734.266135 )
         = 7.45834
```

**② policy chosen `[0,1,2,3,4,5,6,9]`：**

```
logsumexp = ln( e^0 + e^1 + e^2 + e^3 + e^4 + e^5 + e^6 + e^9 )
         = ln( 1.000000 + 2.718282 + 7.389056 + 20.085537
             + 54.598150 + 148.413159 + 403.428793 + 8103.083928 )
         = ln( 8740.716905 )
         = 9.07575
```

> `e^9 = 8103` 一项就占了总和 8740 的 93%，logsumexp ≈ 9.0757 主要被 token 7（logit 9）撑起来。

**③ policy rejected `[0,1,2,1,4,5,6,7]`：**

```
logsumexp = ln( e^0 + e^1 + e^2 + e^1 + e^4 + e^5 + e^6 + e^7 )
         = ln( 1.000000 + 2.718282 + 7.389056 + 2.718282
             + 54.598150 + 148.413159 + 403.428793 + 1096.633158 )
         = ln( 1716.898880 )
         = 7.44827
```

> token 3 的 logit 从 3 降到 1（`e^3=20.09` → `e^1=2.72`），但 token 7 的 `e^7=1096.63` 仍是最大项，所以 logsumexp 比 ref 的 7.4583 略小。

三个对照：

| logit 向量 | 求和 Σ e^v | logsumexp |
|---|---|---|
| ref `[0..7]` | 1734.266 | **7.45834** |
| policy chosen `[0,1,2,3,4,5,6,9]` | 8740.717 | **9.07575** |
| policy rejected `[0,1,2,1,4,5,6,7]` | 1716.899 | **7.44827** |

`log_softmax[v] = logit[v] − logsumexp`，原始 logits 与 log_softmax 逐 token 对应如下：

**ref `[0..7]`（logsumexp = 7.45834）：**

```
logit       = [0, 1, 2, 3, 4, 5, 6, 7]
log_softmax = [-7.4583, -6.4583, -5.4583, -4.4583, -3.4583, -2.4583, -1.4583, -0.4583]
               ↑token0   ↑token1   ↑token2   ↑token3            ↑token6  ↑token7
```

**policy chosen `[0,1,2,3,4,5,6,9]`（logsumexp = 9.075747）：**

```
logit       = [0, 1, 2, 3, 4, 5, 6, 9]   ← token7 被抬到 9
log_softmax = [-9.0757, -8.0757, -7.0757, -6.0757, -5.0757, -4.0757, -3.0757, -0.0757]
               ↑token0   ↑token1   ↑token2   ↑token3   ↑token4   ↑token5   ↑token6  ↑token7
```

**policy rejected `[0,1,2,1,4,5,6,7]`（logsumexp = 7.44827）：**

```
logit       = [0, 1, 2, 1, 4, 5, 6, 7]   ← token3 被压到 1
log_softmax = [-7.4483, -6.4483, -5.4483, -6.4483, -3.4483, -2.4483, -1.4483, -0.4483]
               ↑token0   ↑token1   ↑token2   ↑token3   ↑token4  ↑token5  ↑token6  ↑token7
```

> 对应关系一目了然：每个位置的 log_softmax = 该位置 logit − logsumexp。注意 policy chosen 的 token7（logit 9）得到最高的 −0.0758；policy rejected 的 token3（logit 1）得到 −6.4483，和 token1 相同（因为两者 logit 都是 1）。

### 4.2 gather 按 y 抠出真实 token 的 log 概率

`torch.gather(log_softmax, dim=2, index=y.unsqueeze(2))` 的作用：对每个位置 `(r, t)`，用 y 里的 token id 当索引，去该位置 log_softmax 的**词表维度**里查对应值：

```
log_probs[r][t] = log_softmax[r][t][ y[r][t] ]
```

先看 y 的数值（2×4，每个元素是一个 token id）：

```
y = [[5, 6, 7, 7],   ← chosen 行
     [5, 6, 3, 3]]   ← rejected 行
```

**ref 分支**：ref 的 log_softmax 每位置都一样 `[-7.4583, -6.4583, -5.4583, -4.4583, -3.4583, -2.4583, -1.4583, -0.4583]`，按 y 逐个查表：

```
chosen 行  y = [5, 6, 7, 7]:
  pos0: y=5 → log_softmax[5] = -2.4583
  pos1: y=6 → log_softmax[6] = -1.4583
  pos2: y=7 → log_softmax[7] = -0.4583
  pos3: y=7 → log_softmax[7] = -0.4583
  → ref_log_probs[0] = [-2.4583, -1.4583, -0.4583, -0.4583]

rejected 行 y = [5, 6, 3, 3]:
  pos0: y=5 → log_softmax[5] = -2.4583
  pos1: y=6 → log_softmax[6] = -1.4583
  pos2: y=3 → log_softmax[3] = -4.4583
  pos3: y=3 → log_softmax[3] = -4.4583
  → ref_log_probs[1] = [-2.4583, -1.4583, -4.4583, -4.4583]
```

**policy 分支**：policy 的 log_softmax 分两套——chosen 行用 `[-9.0757, -8.0757, -7.0757, -6.0757, -5.0757, -4.0757, -3.0757, -0.0757]`，rejected 行用 `[-7.4483, -6.4483, -5.4483, -6.4483, -3.4483, -2.4483, -1.4483, -0.4483]`：

```
chosen 行  y = [5, 6, 7, 7]（查 policy_chosen 的 log_softmax）:
  pos0: y=5 → log_softmax[5] = -4.0757
  pos1: y=6 → log_softmax[6] = -3.0757
  pos2: y=7 → log_softmax[7] = -0.0757   ← token7 被抬到 logit 9，概率最高
  pos3: y=7 → log_softmax[7] = -0.0757
  → policy_log_probs[0] = [-4.0757, -3.0757, -0.0757, -0.0757]

rejected 行 y = [5, 6, 3, 3]（查 policy_rejected 的 log_softmax）:
  pos0: y=5 → log_softmax[5] = -2.4483
  pos1: y=6 → log_softmax[6] = -1.4483
  pos2: y=3 → log_softmax[3] = -6.4483   ← token3 被压到 logit 1，概率很低
  pos3: y=3 → log_softmax[3] = -6.4483
  → policy_log_probs[1] = [-2.4483, -1.4483, -6.4483, -6.4483]
```

> 关键：gather 只是「按 y 查表」。chosen 行在位置 2、3 的 y=7，查到 log_softmax[7]（最高值）；rejected 行在位置 2、3 的 y=3，查到 log_softmax[3]（很低的值）。同一个 log_softmax，因为 y 不同而抠出不同的值——这正是后续 `pi_logratios` / `ref_logratios` 里偏好差异的来源。

## 第 5 步：`dpo_loss`

### 5.1 乘 mask 再求和（`(log_probs * mask).sum(dim=1)`）

**ref 分支：**

```
ref_log_probs        = [[-2.4583, -1.4583, -0.4583, -0.4583],
                        [-2.4583, -1.4583, -4.4583, -4.4583]]
ref_log_probs * mask = [[ 0,       0,      -0.4583, -0.4583],
                        [ 0,       0,      -4.4583, -4.4583]]
.sum(dim=1)          → ref_sum = [-0.9167, -8.9167]
```

**policy 分支：**

```
policy_log_probs        = [[-4.0757, -3.0757, -0.0757, -0.0757],
                           [-2.4483, -1.4483, -6.4483, -6.4483]]
policy_log_probs * mask = [[ 0,       0,      -0.0757, -0.0757],
                           [ 0,       0,      -6.4483, -6.4483]]
.sum(dim=1)             → pol_sum = [-0.1515, -12.8966]
```

### 5.2 切开 chosen / rejected（`batch_size//2 = 1`）

```
chosen_ref = -0.9167   reject_ref = -8.9167   → ref_logratios = -0.9167 − (-8.9167)  = 8.0
chosen_pol = -0.1515   reject_pol = -12.8966  → pi_logratios  = -0.1515 − (-12.8966) = 12.745
```

### 5.3 算 logits 与 loss

```
logits = pi_logratios - ref_logratios = 12.745 - 8.0 = 4.745
beta·logits = 0.15 × 4.745 = 0.7118
σ(0.7118) = 1/(1+e^-0.7118) = 0.6708
loss = -log(0.6708) = 0.3993
```

---

## 结果解读

| 量 | 值 | 含义 |
|---|---|---|
| `pi_logratios` | +12.745 | policy 强烈偏好好词 7（−0.0757）胜过坏词 3（−6.4483） |
| `ref_logratios` | +8.0 | ref 也偏好 chosen，但幅度较小 |
| `logits` | +4.745 | policy 比 ref 多偏好 chosen 了 4.745，方向正确 |
| `loss` | 0.399 | 低于中性 0.693，模型表现好、惩罚小 |

**一句话**：x/y/mask 进模型 → 前向出 logits `(2,4,8)` → log_softmax 在词表维归一化 → gather 抠出 `(2,4)` 的 log 概率 → 乘 mask 只留回答段求和 → 切开 chosen/rejected 算两组对数几率差 → 相减得 `logits=+4.745` → 过 sigmoid 再取负 log 得 `loss=0.399`。

---

## 附：为什么 `ref_sum = [-0.9167, -8.9167]` 两个数不一样？

**不是因为 ref 模型不同（ref 对两行完全一样），而是两行的目标 token（y）不同，gather 时抠到的位置不同。**

ref 的 logits 每位置都是 `[0..7]`（单调递增斜坡），所以 log_softmax 也是同一个。关键看 token 3 和 token 7 在斜坡上的位置：

| token | logit | log_softmax |
|---|---|---|
| **7（好词）** | 7（最大） | **−0.4583**（最高） |
| **3（坏词）** | 3（靠后） | **−4.4583**（很低） |

两行 gather 结果不同：

```
chosen  行的 y = [5,6,7,7]  → 回答段抠 token 7 → -0.4583, -0.4583 → 求和 -0.9167
rejected 行的 y = [5,6,3,3] → 回答段抠 token 3 → -4.4583, -4.4583 → 求和 -8.9167
```

所以 `ref_sum` 的差异**完全来自「回答段里是哪个 token」**：

- chosen 回答段填 token 7（ref 认为概率高）→ −0.9167
- rejected 回答段填 token 3（ref 认为概率低）→ −8.9167

两者相减 `ref_logratios = 8.0` 衡量的正是 **ref 模型本身对「好词 7 vs 坏词 3」的固有偏好强度**。ref 不是中性的（logits 是斜坡，天然更喜欢 token 7）；policy 要做的，是在这个固有偏好之上把差距拉得更大（policy 的 12.745 > ref 的 8.0），这才是 `logits > 0` 的含义。

> 若想让 ref 完全中性，就该让 ref 对 token 3 和 token 7 打同样的分（如都设 2.0），那样 `ref_logratios = 0`。
