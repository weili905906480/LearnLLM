# dpo_loss 详解

`dpo_loss` 定义在 `trainer/train_dpo.py`，是 DPO（Direct Preference Optimization）训练的核心损失函数。输入 ref 和 policy 各自对整条序列的逐 token log 概率，输出一个标量 loss。

> 相关文档：训练脚本见 [[train_dpo详解]]，数据集见 [[DPODataset详解]]，log_softmax 算法见 [[log_softmax详解]]。

## 一、函数签名与输入

```python
def dpo_loss(ref_log_probs, policy_log_probs, mask, beta):
```

四个参数：

| 参数 | shape | 含义 |
|---|---|---|
| `ref_log_probs` | `(batch_size, seq_len)` | 冻结参考模型对每个位置真实 token 的 log 概率（由 `logits_to_log_probs` 算好） |
| `policy_log_probs` | `(batch_size, seq_len)` | 可训练策略模型对每个位置真实 token 的 log 概率 |
| `mask` | `(batch_size, seq_len)` | 只有 assistant 回答段为 1，其余（提示/padding）为 0 |
| `beta` | 标量 | 温度/强度系数，控制偏好强度与偏离 ref 的程度 |

**关键约定**：batch 里前一半行是 chosen，后一半行是 rejected（因为 `train_epoch` 里 `x = torch.cat([x_chosen, x_rejected], dim=0)` 按行拼接）。

---

## 二、逐行注释

```python
# 第 1 行：把「逐 token 的 log 概率」乘以 mask，只保留回答段，再沿 seq 维度求和
#         得到「整段回答的对数概率」log P(回答|上下文)。用户提示和 padding 被 mask 屏蔽掉。
ref_log_probs = (ref_log_probs * mask).sum(dim=1)
```
- `ref_log_probs * mask`：mask 为 1 的位置保留原 log-prob，为 0 的位置变成 0（相当于丢弃）。
- `.sum(dim=1)`：沿序列维度求和，把 `(B, S)` 压缩成 `(B,)`——每条样本得到一个标量「整段回答的 log 概率」。
- 因为 log 概率相乘 = 相加（`log P = Σ log p_t`），所以「求和」就是「把回答段每个 token 的 log-prob 累加成整段的 log 概率」。

```python
# 第 2 行：对策略模型做同样的 mask + 求和，得到每条样本整段回答的 log 概率
policy_log_probs = (policy_log_probs * mask).sum(dim=1)
```

```python
# 第 3 行：记录拼接后的总行数（= 2 × 真实 batch_size，因为 chosen 和 rejected 拼在一起）
batch_size = ref_log_probs.shape[0]
```

```python
# 第 4~7 行：按「前一半 chosen、后一半 rejected」的约定切开，取出四个标量向量（各 shape (B,)）
chosen_ref_log_probs    = ref_log_probs[:batch_size // 2]      # ref 对 chosen 的整段 log 概率
reject_ref_log_probs    = ref_log_probs[batch_size // 2:]      # ref 对 rejected 的整段 log 概率
chosen_policy_log_probs = policy_log_probs[:batch_size // 2]   # policy 对 chosen 的整段 log 概率
reject_policy_log_probs = policy_log_probs[batch_size // 2:]   # policy 对 rejected 的整段 log 概率
```

```python
# 第 8 行：policy 模型对「chosen 相对 rejected」的偏好差（对数几率）
#         log π_θ(chosen) - log π_θ(rejected) = log[ π_θ(chosen) / π_θ(rejected) ]
pi_logratios = chosen_policy_log_probs - reject_policy_log_probs
```

```python
# 第 9 行：ref 模型对「chosen 相对 rejected」的偏好差（作为基准锚点）
#         log π_ref(chosen) - log π_ref(rejected)
ref_logratios = chosen_ref_log_probs - reject_ref_log_probs
```

```python
# 第 10 行：最终 logits = policy 的偏好差 − ref 的偏好差
#          这是 DPO 的核心量：衡量「policy 相对 ref 多涨了多少对 chosen 的偏好」
#          logits > 0 表示 policy 比 ref 更偏好 chosen（好），< 0 表示更偏好 rejected（坏）
logits = pi_logratios - ref_logratios
```

```python
# 第 11 行：DPO 损失 = -log σ(β · logits)
#          σ 是 sigmoid；F.logsigmoid(x) = log σ(x)
#          β·logits 越大 → σ→1 → log→0 → loss→0（模型表现好，几乎不惩罚）
#          β·logits 越小(负) → σ→0 → log→ -∞ → loss→ +∞（模型偏好反了，狠狠惩罚）
loss = -F.logsigmoid(beta * logits)
```

```python
# 第 12 行：对 batch 内所有偏好对取平均，得到标量 loss
return loss.mean()
```

---

## 三、完整数值例子（从头算到尾）

设 `真实 batch_size = 1`（1 个 chosen + 1 个 rejected），所以拼接后 `batch_size = 2`。假设 seq_len = 6（已错位），用示意值：

**输入（policy 的逐 token log 概率）：**
```
policy_log_probs = [  # shape (2, 6)
    [-0.1, -0.2, -0.3, -0.4, -0.5, -0.6],   # 第 0 行 = chosen
    [-0.3, -0.4, -0.5, -0.6, -0.7, -0.8],   # 第 1 行 = rejected
]
```

**输入（ref 的逐 token log 概率）：**
```
ref_log_probs = [
    [-0.2, -0.3, -0.4, -0.5, -0.6, -0.7],   # chosen
    [-0.2, -0.3, -0.4, -0.5, -0.6, -0.7],   # rejected
]
```

**mask（只有最后 3 个位置是回答段）：**
```
mask = [
    [0, 0, 0, 1, 1, 1],
    [0, 0, 0, 1, 1, 1],
]
```

**逐行执行：**

```python
ref_log_probs = (ref_log_probs * mask).sum(dim=1)
```
```
ref_log_probs * mask =
    [0, 0, 0, -0.5, -0.6, -0.7]   # chosen 行
    [0, 0, 0, -0.5, -0.6, -0.7]   # rejected 行
.sum(dim=1) → [-1.8, -1.8]
```

```python
policy_log_probs = (policy_log_probs * mask).sum(dim=1)
```
```
policy_log_probs * mask =
    [0, 0, 0, -0.4, -0.5, -0.6]   # chosen
    [0, 0, 0, -0.6, -0.7, -0.8]   # rejected
.sum(dim=1) → [-1.5, -2.1]
```

```python
batch_size = 2
chosen_ref_log_probs    = ref_log_probs[:1]      = [-1.8]
reject_ref_log_probs    = ref_log_probs[1:]      = [-1.8]
chosen_policy_log_probs = policy_log_probs[:1]   = [-1.5]
reject_policy_log_probs = policy_log_probs[1:]   = [-2.1]
```

```python
pi_logratios  = [-1.5] - [-2.1] = [0.6]   # policy 明显更偏好 chosen（log 几率 0.6 > 0）
ref_logratios = [-1.8] - [-1.8] = [0.0]   # ref 对两者无差别（锚点为 0）
logits        = [0.6] - [0.0]   = [0.6]   # policy 相对 ref 多涨了 0.6 的偏好
```

```python
beta = 0.15
beta * logits = [0.09]
F.logsigmoid(0.09) = log σ(0.09) = log(0.5225) ≈ -0.649
loss = -(-0.649) = 0.649
loss.mean() = 0.649
```

**结论**：policy 已经在正确方向上跑赢 ref（偏好差 0.6 > 0），所以损失不大（0.649）。反过来，如果 policy 更偏好 rejected（`pi_logratios` 变负），`β·logits` 是很负的数，`σ→0`，`-log σ` 会非常大，被狠狠惩罚。

---

## 四、完整公式回顾

$$\mathcal{L}_{DPO} = -\mathbb{E}\left[\log \sigma\left(\beta \left[\log \frac{\pi_\theta(y_w|x)}{\pi_{ref}(y_w|x)} - \log \frac{\pi_\theta(y_l|x)}{\pi_{ref}(y_l|x)}\right]\right)\right]$$

代码里拆成两步：

$$\text{logits} = \underbrace{(\log \pi_\theta(y_w) - \log \pi_\theta(y_l))}_{\text{pi\_logratios}} \;-\; \underbrace{(\log \pi_{ref}(y_w) - \log \pi_{ref}(y_l))}_{\text{ref\_logratios}}$$

$$\text{loss} = -\log\sigma(\beta \cdot \text{logits})$$

**直觉**：当 policy 对 chosen 的相对偏好恰好等于 ref（即 `logits = 0`），loss = `-log σ(0) = log 2 ≈ 0.693`，这是 DPO 的「中性起点」；训练就是让 policy 的 `logits` 越推越大（更偏好 chosen），loss 越压越低。

---

## 五、β·logits 的作用

`β·logits` 是 DPO 损失里喂进 sigmoid 的那个「打分」，理解了它就理解了整个 DPO 在优化什么。

### 1. 它到底是什么

```python
loss = -F.logsigmoid(beta * logits)   # 即 -log σ(β · logits)
```

`σ(β·logits)` 是一个 **0~1 的概率**，含义是「模型认为 chosen 比 rejected 更好」的置信度；`-log σ(...)` 就是对这个概率做的**二元交叉熵**——想让这个概率尽量趋近 1。

所以 `β·logits` 是 sigmoid 的输入，它的大小和符号直接决定 loss：

| `β·logits` 的值 | `σ(β·logits)` | `loss = -log σ` | 含义 |
|---|---|---|---|
| 很大正数（如 +5） | ≈ 0.993 | ≈ 0.007 | 模型强烈偏好 chosen，几乎不惩罚 |
| = 0 | 0.5 | ≈ 0.693 | 中性，chosen/rejected 五五开（起点） |
| 很大负数（如 -5） | ≈ 0.007 | ≈ 5.0 | 模型偏好反了（偏向 rejected），狠狠惩罚 |

### 2. `logits` 负责「方向 + 大小」，`β` 负责「强度缩放」

`β·logits` 里两个因子各司其职：

**`logits`（方向 + 原始大小）** = `pi_logratios - ref_logratios`，表示「policy 相对 ref 多偏好 chosen 了多少」：
- `logits > 0` → policy 比 ref 更偏向 chosen，方向正确
- `logits < 0` → policy 反而更偏向 rejected，方向错了

**`β`（缩放强度）**：一个标量温度系数，把 `logits` 放大或缩小后再过 sigmoid。

- `β` 越大 → 同样的 `logits` 被放大 → sigmoid 更陡 → 梯度更强、优化更激进，但也更容易**跑偏、忘掉 SFT 学的东西**。
- `β` 越小 → 信号被压缩 → 优化更保守、更稳，更贴近 ref，但可能**学不动偏好**。

代码里默认 `β = 0.15`，配合极小的学习率（`4e-8`），就是「温和地拉开偏好、别忘记已有能力」的取向。

### 3. β 的数学出处（为什么是它而不是别的）

DPO 是从带 KL 约束的 RLHF 目标推导出来的：

$$\max_{\pi_\theta} \ \mathbb{E}\left[ r(x,y) \right] - \beta \cdot \text{KL}(\pi_\theta \| \pi_{ref})$$

这里的 `β` 就是 **KL 散度惩罚项的系数**，本质是「**允许 policy 偏离 ref 多远**」的开关：

- `β` 小 → KL 惩罚权重大 → 把 policy 牢牢拴在 ref 附近（**防遗忘**，但偏好学得慢）；
- `β` 大 → KL 惩罚弱 → policy 可以大步离开 ref（**偏好拉得开**，但可能过拟合偏好对、退化）。

所以 `β·logits` 里的 `β` 不是随便加的温度，而是这个 KL 约束系数在闭式解里自然落进了 sigmoid 的参数位。

### 4. 数值对照（固定 logits=0.6）

| β | β·logits | σ(β·logits) | loss |
|---|---|---|---|
| 0.05 | 0.03 | 0.5075 | 0.678 |
| 0.15 | 0.09 | 0.5225 | 0.649 |
| 1.0 | 0.6 | 0.6457 | 0.437 |
| 5.0 | 3.0 | 0.9526 | 0.049 |

同一个 `logits=0.6`（policy 已经正确偏向 chosen），`β` 从 0.05 加到 5，loss 从 0.678 降到 0.049——**β 越大，模型「已经做对了」这件事得到的奖励越明显**，梯度反馈越强。

**一句话总结**：`β·logits` 是 DPO 的「偏好打分」——`logits` 决定方向（是否正确偏好 chosen）和大小，`β` 决定这个偏好的**强度/激进程度**；两者一起进 sigmoid 变成「chosen 优于 rejected」的概率，损失就是逼这个概率往 1 走，同时用 `β` 约束 policy 别离 ref 太远。

---

## 六、σ（sigmoid）的公式与计算

`σ` 就是 **sigmoid（S 型/逻辑斯蒂）函数**：

$$\sigma(x) = \frac{1}{1 + e^{-x}}$$

它把任意实数 `x ∈ (-∞, +∞)` 挤压到 `(0, 1)` 区间，正好可以当「概率」用。

### 1. 公式与性质

| 性质 | 值/说明 |
|---|---|
| 定义 | `σ(x) = 1 / (1 + e^(-x))` |
| 值域 | `(0, 1)`，永远开区间，不会恰好等于 0 或 1 |
| `σ(0)` | `1 / (1+1) = 0.5`（中点） |
| `x → +∞` | `e^(-x)→0`，`σ→1`（饱和到 1） |
| `x → -∞` | `e^(-x)→+∞`，`σ→0`（饱和到 0） |
| 对称性 | `σ(-x) = 1 - σ(x)`（关于点 (0, 0.5) 中心对称） |
| 导数 | `σ'(x) = σ(x)·(1 - σ(x))`（最大 0.25，在 x=0 处） |

图像是一条 S 形曲线：x=0 附近最陡，往两端越来越平（梯度消失区）。

### 2. 手算步骤（以 DPO 里的 `σ(0.09)` 为例）

分三步：**取负 → 求指数 → 倒数**。

$$x = 0.09$$

```
① -x = -0.09
② e^(-0.09) = 0.9139      （自然指数，e ≈ 2.71828）
③ 1 + 0.9139 = 1.9139
④ σ(0.09) = 1 / 1.9139 = 0.5225
```

这正好是 `dpo_loss` 例子里 `β·logits = 0.09` 对应的概率 0.5225——含义是「模型认为 chosen 优于 rejected」的置信度约 52.25%（略高于一半，所以 loss 不大但还有提升空间）。

再算两个基准点对照：

```
σ(0)    = 1 / (1 + e^0) = 1/2 = 0.5        # 中性，五五开
σ(3.0)  = 1 / (1 + e^-3.0) = 1/(1+0.0498) = 0.9526   # 强烈正向
σ(-3.0) = 1 / (1 + e^3.0)  = 1/(1+20.09)  = 0.0474   # 强烈负向（= 1 - 0.9526）
```

### 3. PyTorch 里怎么算（数值稳定版）

代码里用的是 `F.logsigmoid(beta * logits)`，即 `log σ(x)`，不是先算 σ 再取 log。它内部用 **softplus** 关系来保证稳定：

$$\log\sigma(x) = -\log(1 + e^{-x}) = -\text{softplus}(-x)$$

而 `softplus(x) = log(1 + e^x)` 也有自己的稳定实现（x 很大时直接用 `log(1+e^x) ≈ x` 避免 `e^x` 上溢）。所以：

- 直接算 `σ(x)`：用 `1/(1+e^-x)`，`x` 很负时 `e^-x` 上溢成 inf，结果反而正确趋近 0，基本没大问题；
- 算 `log σ(x)`：**必须**走 `-softplus(-x)` 这条稳定路径，否则 `σ` 先趋近 0、再 `log(0) = -inf` 会丢精度。

```python
import torch
import torch.nn.functional as F
x = torch.tensor(0.09)
print(torch.sigmoid(x))        # 0.5225   —— σ 本身
print(F.logsigmoid(x))         # -0.6490  —— log σ(0.09)，dpo_loss 用的就是这个
# 验证等价性
print(torch.log(torch.sigmoid(x)))  # -0.6490  （此处数值相近，但大 |x| 时后者不稳）
```

### 4. 放回 DPO 语境

```python
loss = -F.logsigmoid(beta * logits)   # = -log σ(β·logits)
```

- `σ(β·logits)` 把「偏好打分」映射成「chosen 优于 rejected 的概率」；
- 取负 log 就是对这个概率做交叉熵：`概率→1` 时 loss→0，`概率→0` 时 loss→+∞。

**一句话**：`σ(x) = 1/(1+e^-x)`，把分数压成 (0,1) 的概率；在 DPO 里它把 `β·logits` 变成「模型偏好 chosen 的置信度」，损失就是逼这个置信度逼近 1。
