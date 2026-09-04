# grpo_train_epoch 一个 step 的 loss 计算算例

本文用一个最小可手算的例子，把 `train_grpo.py` 里 `grpo_train_epoch` 的**一个训练 step** 从 rollout 到 `loss.backward()` 完整走一遍，每一步都给出张量形状和具体数值。**重点是阶段 4：从 logits 到 per_token_logps 的完整变化过程。**

> 相关文档：函数整体讲解见 [[train_grpo详解]]，奖励计算见 [[calculate_rewards数值算例]]，log_softmax + gather 的逐位算例见 [[dpo_loss数值算例]]。

---

## 一、设定

```python
B = 1, G = 2, N = B*G = 2        # batch=1, 每组采样 2 条
P = 3, R = 4                      # prompt 3 token，最多生成 4 token
V = 8                             # 词表 8 个 token（id 0~7），够手算
pad_token_id = 0
eos_token_id = 2
beta = 0.1
epsilon_high = 5.0
loss_type = "cispo"（默认）
aux_loss = 0（非 MoE）
accumulation_steps = 1
```

一个 step 的数据流（括号是 `grpo_train_epoch` 里的注释编号）：

```
编码 prompt → rollout 采样 → 算奖励/advantage → policy 前向 → ref 前向
   → KL/ratio → 逐 token loss → mask 归一化 → backward
```

---

## 二、逐阶段矩阵变化

### 阶段 1：Rollout 输出（第 177–188 行）

唯一一条 prompt 的 token 为 `[1, 5, 6]`（左填充下无 pad，P=3），采样 G=2 次：

```python
completion_ids [N,R] = [[4, 7, 2, 0],   # 生成 4、7 两个真 token → EOS(2) → pad(0)
                        [3, 5, 6, 7]]   # 生成满 4 个 token，无 EOS

output_ids [N,P+R]   = [[1, 5, 6, 4, 7, 2, 0],
                        [1, 5, 6, 3, 5, 6, 7]]
```

同时拿到采样时的 `old_per_token_logps [N,R]` 和解码文本 `completions`（算奖励用）。

### 阶段 2：full_mask 与 logp_pos（第 191、197 行）

```python
full_mask = (output_ids != 0).long()        # pad 位置为 0
#        = [[1,1,1,1,1,1,0],
#           [1,1,1,1,1,1,1]]

prompt_lens = [3, 3]                        # 每条 prompt 的实际长度
logp_pos = prompt_lens.unsqueeze(1) - 1 + arange(R)
         = [[2, 3, 4, 5],
            [2, 3, 4, 5]]
```

`logp_pos` 含义：生成 token 在完整序列第 `P..P+R-1` 位，由位置 `P-1..` 的 logits 预测，所以索引从 `P-1=2` 起步（模型位置 t 的 logits 预测 t+1 的 token）。

### 阶段 3：rewards → advantages（第 200 行，第 241–245 行）

沿用 [[calculate_rewards数值算例]] 的例子，两条回答奖励 `rewards = [4.25, -2.25]`。组内标准化：

```python
grouped = rewards.view(-1, 2)                    # [[4.25, -2.25]]
mean_r  = (4.25 + (-2.25)) / 2 = 1.0
std_r   = sqrt(((4.25-1)² + (-2.25-1)²)/2) = 3.25
advantages = (rewards - mean_r) / (std_r + 1e-4)
          = [1.0, -1.0]          # 组内互相对比：好回答 +1，坏回答 -1
```

---

### 阶段 4：从 logits 到 per_token_logps（第 206–217 行）—— 完整变化过程

这是整个 step 里最关键、也最容易算错的一步。对应代码（第 213 行）：

```python
per_token_logps = F.log_softmax(res.logits[:, :-1, :], dim=-1) \
    .gather(2, outputs[:, 1:].unsqueeze(-1)).squeeze(-1) \
    .gather(1, logp_pos)
```

#### 4.1 形状流水线

```
res.logits                                 [N, P+R, V]      = [2, 7, 8]   完整序列每个位置的词表 logits
   ↓ [:, :-1, :]  丢掉最后一个位置
                                             [N, P+R-1, V]   = [2, 6, 8]   位置 t 的 logits 预测 t+1 的 token
   ↓ log_softmax(dim=-1)  词表维归一化
                                             [2, 6, 8]                     每个位置变成 log 概率分布
   ↓ gather(2, outputs[:, 1:])  抠真实 next-token 的 log 概率
                                             [2, 6, 1] → squeeze → [2, 6]  每位置只剩一个数
   ↓ gather(1, logp_pos)  只挑生成段的位置
                                             [2, 4]                        per_token_logps
```

关键理解：**`logits[:, :-1]` 和 `outputs[:, 1:]` 是「错位对齐」**——位置 t 的 logits 预测的就是 `outputs[t+1]`，所以 logits 少最后一位、outputs 少第一位，两者一一对应。之后 `logp_pos` 再从中筛出「生成段」的那 4 个位置。

#### 4.2 三个模型的 logits（每位置相同，V=8）

三个模型各自前向一次，得到各自的 logits。为便于手算，每个位置的 logit 向量固定（8 维）：

| 模型 | logits（token 0~7） | 含义 |
|---|---|---|
| **policy（当前）** | `[0,1,2,3,4,5,6,7]` | 标准斜坡 |
| **old（采样时）** | `[0,1,2,3,4,5,6,6]` | token 7 的 logit 从 7 降到 6（采样时对 token 7 没那么自信） |
| **ref（冻结 SFT）** | `[0,1,2,3,4,5,6,8]` | token 7 的 logit 从 7 升到 8（基准更偏爱 token 7） |

#### 4.3 log_softmax（dim=-1）

`log_softmax[v] = logit[v] − logsumexp`，其中 `logsumexp = ln(Σ e^logit)`。

**① policy `[0..7]`：**

```
Σ e^v = 1 + 2.7183 + 7.3891 + 20.0855 + 54.5982 + 148.4132 + 403.4288 + 1096.6332
      = 1734.2661
logsumexp = ln(1734.2661) = 7.4583
```

**② old `[0,1,2,3,4,5,6,6]`：**

```
Σ e^v = (e^0+…+e^6) + e^6 = 637.6330 + 403.4288 = 1041.0618
logsumexp = ln(1041.0618) = 6.9480
```

**③ ref `[0,1,2,3,4,5,6,8]`：**

```
Σ e^v = (e^0+…+e^6) + e^8 = 637.6330 + 2980.9580 = 3618.5910
logsumexp = ln(3618.5910) = 8.1938
```

三个模型的 log_softmax 表（`log_softmax[token] = logit[token] − logsumexp`）：

| token | policy（−7.4583） | old（−6.9480） | ref（−8.1938） |
|---|---|---|---|
| 0 | -7.4583 | -6.9480 | -8.1938 |
| 1 | -6.4583 | -5.9480 | -7.1938 |
| 2 | -5.4583 | -4.9480 | -6.1938 |
| 3 | -4.4583 | -3.9480 | -5.1938 |
| 4 | -3.4583 | -2.9480 | -4.1938 |
| 5 | -2.4583 | -1.9480 | -3.1938 |
| 6 | -1.4583 | -0.9480 | -2.1938 |
| 7 | -0.4583 | -0.9480 | -0.1938 |

> 注意两个细节：① token 7 在 old 里被压到 -0.9480（和 token 6 一样，因为 logit 都是 6），在 ref 里被抬到 -0.1938（logit 8）；② logsumexp 变了会**牵一发动全身**——old 的 logsumexp 比 policy 小，所以除了 token 7 之外，old 每个 token 的 log_softmax 都比 policy 高了 0.5103。

#### 4.4 第一次 gather：抠真实 next-token 的 log 概率

`outputs[:, 1:]` 是每个位置要预测的目标 token：

```
outputs[:, 1:] = [[5, 6, 4, 7, 2, 0],    ← 行0：预测 output_ids[0] 的第 1..6 位
                  [5, 6, 3, 5, 6, 7]]    ← 行1：预测 output_ids[1] 的第 1..6 位
```

以 **policy** 为例，逐位置查 log_softmax 表（`log_softmax[目标token]`）：

```
行0 目标 [5,6,4,7,2,0]:
  5→-2.4583, 6→-1.4583, 4→-3.4583, 7→-0.4583, 2→-5.4583, 0→-7.4583
  → [-2.4583, -1.4583, -3.4583, -0.4583, -5.4583, -7.4583]   [6 个位置]

行1 目标 [5,6,3,5,6,7]:
  5→-2.4583, 6→-1.4583, 3→-4.4583, 5→-2.4583, 6→-1.4583, 7→-0.4583
  → [-2.4583, -1.4583, -4.4583, -2.4583, -1.4583, -0.4583]   [6 个位置]
```

得到 `[N, 6] = [2, 6]`。每个位置的语义是「**完整序列第 t+1 个 token 的 log 概率**」，此时还没区分 prompt 段和生成段。

#### 4.5 第二次 gather：只挑生成段的位置

`logp_pos = [[2,3,4,5],[2,3,4,5]]`，从上一步的 `[2,6]` 里抠出第 2、3、4、5 列：

```
行0 第 2,3,4,5 列 → [-3.4583, -0.4583, -5.4583, -7.4583]
行1 第 2,3,4,5 列 → [-4.4583, -2.4583, -1.4583, -0.4583]
```

这 4 个值正好对应**生成段 4 个 token**（`completion_ids`）的 log 概率：

```
per_token_logps（policy） = [[-3.4583, -0.4583, -5.4583, -7.4583],   ← token 4、7、2、0
                             [-4.4583, -2.4583, -1.4583, -0.4583]]   ← token 3、5、6、7
```

#### 4.6 三套结果汇总

old 和 ref 走**完全相同的流水线**，只是 log_softmax 表不同（见 4.3），gather 抠的目标 token 相同。三套 `[2,4]` 结果：

```python
per_token_logps     = [[-3.4583, -0.4583, -5.4583, -7.4583],   # 当前 policy
                       [-4.4583, -2.4583, -1.4583, -0.4583]]
old_per_token_logps = [[-2.9480, -0.9480, -4.9480, -6.9480],   # 采样时（旧 policy）
                       [-3.9480, -1.9480, -0.9480, -0.9480]]
ref_per_token_logps = [[-4.1938, -0.1938, -6.1938, -8.1938],   # 冻结 ref
                       [-5.1938, -3.1938, -2.1938, -0.1938]]
```

> 这三套矩阵就是阶段 5 的输入。第 0 行末位（token 0，pad 位置）也填了值，但后面会被 `completion_mask` 置 0，不影响 loss。

---

### 阶段 5：KL 散度与 ratio（第 260–266 行）

```python
kl_div       = ref_per_token_logps - per_token_logps
per_token_kl = exp(kl_div) - kl_div - 1        # k3 估计器
ratio        = exp(per_token_logps - old_per_token_logps)
```

#### 5.1 kl_div 与 per_token_kl

```
kl_div = ref - per = [[-0.7355,  0.2645, -0.7355, -0.7355],
                      [-0.7355, -0.7355, -0.7355,  0.2645]]

per_token_kl = exp(kl_div) - kl_div - 1
             = [[0.2148, 0.0383, 0.2148, 0.2148],   # exp(-0.7355)+0.7355-1 = 0.2148
                [0.2148, 0.2148, 0.2148, 0.0383]]   # exp(0.2645)-0.2645-1   = 0.0383
```

`per_token_kl` 用 **k3 估计器** `e^d - d - 1`，是 KL(π‖π_ref) 的始终 ≥0 下界近似。

#### 5.2 ratio 的完整计算

`ratio` 是重要性采样比率 `π_current / π_old`，衡量当前 policy 相对采样时（old）对每个生成 token 的偏好变化。逐元素两步：

**第 1 步：算对数差 `per - old`（`exp` 里面的减法）**

```
per - old = [[-3.4583-(-2.9480), -0.4583-(-0.9480), -5.4583-(-4.9480), -7.4583-(-6.9480)],
             [-4.4583-(-3.9480), -2.4583-(-1.9480), -1.4583-(-0.9480), -0.4583-(-0.9480)]]

          = [[-0.5103, +0.4897, -0.5103, -0.5103],
             [-0.5103, -0.5103, -0.5103, +0.4897]]
```

**第 2 步：取指数 `exp(...)`**

```
ratio = exp(per - old) = [[e^-0.5103, e^+0.4897, e^-0.5103, e^-0.5103],
                          [e^-0.5103, e^-0.5103, e^-0.5103, e^+0.4897]]

      = [[0.6003, 1.6318, 0.6003, 0.6003],
         [0.6003, 0.6003, 0.6003, 1.6318]]
```

**规律解读**：对数差只有两个值——**非 token-7 位置都是 −0.5103，token-7 位置都是 +0.4897**，且 +0.4897 恰好落在两条回答里 token=7 的地方（行 0 第 1 位、行 1 第 3 位，对应 `completion_ids` 里的 7）。

回到阶段 4.3 的 log_softmax 表，看这两个值从哪来：

| 位置 | policy log_softmax | old log_softmax | 差 `per - old` |
|---|---|---|---|
| token t（≠7） | t − 7.4583 | t − 6.9480 | **−0.5103** |
| token 7 | 7 − 7.4583 = −0.4583 | 6 − 6.9480 = −0.9480 | **+0.4897** |

- **非 token-7**：两个模型的 logit 都是 t，差异**纯粹来自 logsumexp 不同**（policy 7.4583 vs old 6.9480），所以恒等于 −0.5103。
- **token 7**：old 把它的 logit 从 7 降到 6（见 4.2），除了同样承受 logsumexp 差 −0.5103，还额外多了 +1.0 的 logit 差，合计 −0.5103 + 1.0 = **+0.4897**。

于是：非 token-7 位置 `e^-0.5103 = 0.6003`（policy 比 old **更不自信**），token-7 位置 `e^+0.4897 = 1.6318`（policy 比 old **更自信**）。这正是 old 把 token 7 降权带来的连锁效应。

#### 5.3 ratio 的直观理解

`ratio` 本质是**新旧两个策略对同一个 token 的概率之比**：

```
ratio = π_current(该 token) / π_old(该 token)
```

一句话记住它：**同一个 token，现在的 policy 比采样时（old）更想要它，还是更不想要它？**

| ratio | 含义 | 直观说法 |
|---|---|---|
| `= 1` | 概率没变 | 新旧策略对它的态度一致 |
| `> 1` | 现在的概率更高 | 新策略比采样时「更喜欢」这个 token |
| `< 1` | 现在的概率更低 | 新策略比采样时「更嫌弃」这个 token |

**为什么需要它（重要性采样）**：采样用的是旧策略 `π_old`，更新的是当前策略 `π_cur`，两个分布已经不一样了（policy 每步都在变）。`ratio` 就是**概率的汇率**——把「从旧策略抽出来的样本」重新加权，换算成「等价于从当前策略抽出来的样本」。样本来自旧分布，乘上 ratio 后期望就回到当前分布。

**回到本例的两个数**：`ratio = 0.6003`（非 token-7）表示现在比采样时更不想要这些 token；`ratio = 1.6318`（token 7）表示现在比采样时更想要 token 7。原因就是 old 当初把 token 7 降了权，现在的 policy 把它恢复了。

**在 loss 里的角色**：`ratio` 是 advantage 的权重——advantage 决定方向（该涨还是该跌），ratio 决定力度（这个 token 现在值多少）。正 advantage 的回答里，`ratio > 1` 的 token 被抬得更用力。注意 cispo 里 ratio 被 `clamp(...).detach()`，只当**停止梯度的缩放权重**，梯度实际只走 `logp`。

#### 5.4 per_token_kl 的直观理解

`per_token_kl` 是**当前 policy 离参考模型 ref 有多远**的逐 token 度量。它和 ratio 是两个维度：ratio 比「现在 vs 采样时」，kl 比「现在 vs 参考基准」。

**ref 是谁、为什么需要"缰绳"**：ref 是 SFT 后的模型（冻结），代表"安全、正常、会说人话"的基准。GRPO 靠 reward 打分训练，但 reward 是代理目标，模型会 **reward hacking**（刷格式骗高分、内容却崩坏）。KL 就是缰绳：不管 reward 怎么诱惑，都不让 policy 跑离 ref 太远。`beta=0.1` 是这条缰绳的松紧。

**为什么用 k3 = e^d - d - 1 而不是直接相减**：直接算 `log π - log π_ref` 逐项有正有负、方差大。k3 有两个好性质：① **恒 ≥ 0**（`e^d ≥ 1+d`，等号只在 d=0 取到）；② 是 KL(π‖π_ref) 的**无偏估计**。

| d = log(π_ref/π) | 含义 | k3 |
|---|---|---|
| `= 0` | π 和 ref 一样 | **0**（没偏离） |
| `> 0` | ref 比 policy 更看好这个 token | `> 0`，随 d 指数增长 |
| `< 0` | policy 比 ref 更看好这个 token | 仍然 `> 0` |

关键：**不管 policy 比 ref「更激进」还是「更保守」，只要偏离就 >0、都要付代价**——偏离的方向不重要，偏离这件事本身被惩罚。

**回到本例的两个数**：token 7 处 `d=+0.2645` → k3 = **0.0383**（分歧小）；非 token-7 处 `d=−0.7355` → k3 = **0.2148**（policy 偏离 ref 较大）。

**在 loss 里的角色**：`per_token_loss = -ratio·adv·logp + beta·per_token_kl`。取负后 kl 项变成 `+beta·kl`，**KL 越大 loss 越大**，梯度把 policy 往「靠近 ref」的方向拉。前面 `ratio·adv·logp` 项往高 reward 推，后面 `+beta·kl` 项往回拽，两者平衡，policy 就不至于跑飞。

### 阶段 6：completion_mask（第 249–257 行）

生成可能提前遇 EOS，后面全是 pad，这些位置不参与 loss：

```python
completion_pad_mask = [[1,1,1,0], [1,1,1,1]]       # pad 位置为 0
is_eos = (completion_ids==2) & completion_pad_mask
       = [[0,0,1,0], [0,0,0,0]]
eos_idx = [2, 3]      # 第0行第一个 EOS 在位置2；第1行无 EOS 取 R-1=3
completion_mask = (arange(R) <= eos_idx) & completion_pad_mask
                = [[1,1,1,0], [1,1,1,1]]           # EOS 及之前保留，pad 丢弃
```

### 阶段 7：逐 token loss（cispo，第 273–275 行）

```python
clamped_ratio = clamp(ratio, max=5.0).detach()     # 本例 ratio 都 ≤1.63，clamp 不变，但 detach 阻断梯度
per_token_loss = -(clamped_ratio * adv.unsqueeze(1) * per_token_logps - beta * per_token_kl)
```

`adv.unsqueeze(1) = [[1.0], [-1.0]]` 广播到 `[2,4]`。逐元素展开：

**第 0 行（advantage=+1.0，好回答）**：

```
ratio*adv*logp = [0.6003×1×(-3.4583), 1.6318×1×(-0.4583), 0.6003×1×(-5.4583), 0.6003×1×(-7.4583)]
               = [-2.0762, -0.7478, -3.2766, -4.4775]
- beta*kl      = [-0.0215, -0.0038, -0.0215, -0.0215]
inside         = [-2.0977, -0.7516, -3.2981, -4.4990]
per_token_loss = [ 2.0977,  0.7516,  3.2981,  4.4990]   ← 取负
```

**第 1 行（advantage=-1.0，坏回答）**：

```
ratio*adv*logp = [0.6003×(-1)×(-4.4583), 0.6003×(-1)×(-2.4583), 0.6003×(-1)×(-1.4583), 1.6318×(-1)×(-0.4583)]
               = [ 2.6763,  1.4757,  0.8754,  0.7478]
- beta*kl      = [-0.0215, -0.0215, -0.0215, -0.0038]
inside         = [ 2.6548,  1.4542,  0.8539,  0.7440]
per_token_loss = [-2.6548, -1.4542, -0.8539, -0.7440]   ← 取负
```

最终：

```
per_token_loss [2,4] = [[ 2.0977,  0.7516,  3.2981,  4.4990],
                        [-2.6548, -1.4542, -0.8539, -0.7440]]
```

### 阶段 8：mask 归一化 → loss → backward（第 284–286 行）

```python
policy_loss = ((per_token_loss * completion_mask).sum(1) / completion_mask.sum(1).clamp(min=1)).mean()
loss = (policy_loss + aux_loss) / accumulation_steps
loss.backward()
```

```
per_token_loss * mask = [[ 2.0977,  0.7516,  3.2981,  0.0000],
                         [-2.6548, -1.4542, -0.8539, -0.7440]]
.sum(dim=1)           = [ 6.1474, -5.7069]
/ 有效token数(3,4)     = [ 2.0491, -1.4267]
.mean()               = (2.0491 + (-1.4267)) / 2 = 0.3112
```

```python
loss = 0.3112          # aux_loss=0（非MoE），accumulation_steps=1
```

---

## 三、结果解读

| 量 | 值 | 含义 |
|---|---|---|
| `advantages` | `[+1.0, -1.0]` | 好回答要抬、坏回答要压 |
| `ratio` | 非 token-7 位置 `0.6003`，token-7 位置 `1.6318` | policy 相对旧策略：token 7 更自信、其余略不自信 |
| `per_token_kl` | 0.04~0.21 | KL 较小，policy 还没偏离 ref 太远 |
| `per_token_loss` 第 0 行 | 全正 `[2.10,0.75,3.30,4.50]` | 好回答的每个 token 都贡献正 loss |
| `per_token_loss` 第 1 行 | 全负 `[-2.65,-1.45,-0.85,-0.74]` | 坏回答的 token 贡献负 loss |
| `policy_loss` | **0.3112** | 好/坏回答平均后的总损失 |

---

## 四、两个关键点

1. **loss 的正负不代表好坏**。这是 policy gradient，不是分类交叉熵。梯度方向由 `d(loss)/d(logp) = -clamped_ratio·adv` 决定：advantage 为正（好回答）时梯度为负，`optimizer.step()` 沿负梯度方向走 → **增大**该 token 的 logp；advantage 为负（坏回答）则相反。所以第 0 行往「更可能生成」更新、第 1 行往「更不可能生成」更新，与 loss 本身是正是负无关。

2. **cispo 与 grpo 的区别只在阶段 7**：cispo 只 `clamp(ratio, max=5.0).detach()`（ratio 当停止梯度的权重，梯度只走 `per_token_logps`）；grpo 则用 `torch.min(ratio·adv, clip(ratio,0.8,1.2)·adv)` 做双向 PPO 裁剪。前 6 个阶段完全一致。

---

## 五、一句话总结

`grpo_train_epoch` 一个 step：rollout 出 `[N,R]` 的生成 → 三个模型各前向一次，`logits [N,P+R,V]` 经 `log_softmax → gather(真实 next-token) → gather(logp_pos)` 得三套 `[N,R]` log 概率 → 组内标准化得 advantage → `ratio=exp(per-old)`、`per_token_kl=k3(ref-per)` → cispo 逐 token loss `-(clamp(ratio)·adv·logp − β·kl)` → 乘 `completion_mask` 只统计有效 token 求均值 → 得 `policy_loss=0.3112` 再 `backward()`。
