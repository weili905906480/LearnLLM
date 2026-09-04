# log_softmax 算法详解

本文解释 `train_dpo.py` 中 `logits_to_log_probs` 函数用到的 `F.log_softmax` 背后的算法：**log-sum-exp trick（对数-和-指数技巧）**。

`logits_to_log_probs` 里只有一行用到它：

```python
log_probs = F.log_softmax(logits, dim=2)
```

作用：把模型输出的原始分数（logits）转成**每个 token 的 log 概率**。

## 〇、为什么要对 softmax 取 log（动机）

核心原因可以归结为**「把连乘变成加法」+「数值稳定」+「损失函数本来就定义在 log 上」**三点：

### 1. 语言模型要算的是「整句概率」= 连乘

生成一句话 `y = (y_1, y_2, ..., y_T)` 的概率，是每个 token 的**条件概率连乘**：

$$P(y|x) = \prod_{t=1}^{T} p_\theta(y_t \mid y_{<t})$$

每个 `p_t` 都是 softmax 输出的概率，值在 `(0, 1)` 之间。**连乘 T 个小于 1 的数会迅速趋近 0**——比如每个 token 概率 0.5，乘 100 个就是 `0.5^100 ≈ 8×10^-31`，已经快到浮点数能表示的下限了；再乘就直接下溢成 0，信息全丢。

**取 log 后，乘法变加法：**

$$\log P(y|x) = \sum_{t=1}^{T} \log p_\theta(y_t \mid y_{<t})$$

一堆负数相加，绝不会下溢，数值上稳得多。这正是 `dpo_loss` 里 `(log_probs * mask).sum(dim=1)` 为什么是「求和」而不是「求积」的原因。

### 2. 训练目标本身就叫「负对数似然（NLL）」

最大似然估计（MLE）要最大化 `P(y|x)`。因为 log 是**单调递增**函数，最大化 `P` 完全等价于最大化 `log P`，等价于最小化 `-log P`。这个 `-log P` 就是负对数似然（Negative Log Likelihood），也是 CrossEntropyLoss 的本质：

$$\text{CE} = -\sum_t \log p_\theta(y_t \mid y_{<t})$$

所以**损失函数天然就在 log 概率空间里定义**，直接产出 log 概率反而省了一步，还避免了 `log(softmax)` 的数值问题。

### 3. 具体到 DPO：需要的是「概率比」，log 后变成「差」

DPO 的损失里有个关键项：

$$\log \frac{\pi_\theta(y_w|x)}{\pi_\theta(y_l|x)}$$

这是两个概率的**比值**。取 log 后，比值变成减法：

$$\log \frac{\pi_\theta(y_w|x)}{\pi_\theta(y_l|x)} = \log \pi_\theta(y_w|x) - \log \pi_\theta(y_l|x)$$

这正是 `dpo_loss` 里的 `pi_logratios = chosen_policy_log_probs - reject_policy_log_probs`——**两个 log 概率直接相减**即可。如果手里只有概率本身，还得先除再取 log，既麻烦又容易下溢。

### 4. 附带好处：优化更平滑

- `log` 是凹函数，log-likelihood 曲面比原始概率更平滑、梯度更稳定，收敛更平稳。
- 概率极小时（如 `10^-10`），log 会把它放大到可感知的 `-23`，让模型对「预测错」的小概率事件也能产生有意义的梯度，而不是被忽略。

**一句话总结**：取 log 是为了把「连乘概率」变成「加法 log 概率」——既避免了下溢，又正好匹配了「负对数似然」这个训练目标；在 DPO 里还额外让「概率比」退化成「log 概率相减」，让整个 loss 都落在稳定的对数空间里。

---

## 一、从 softmax 到 log_softmax：数学定义

**softmax**（把 logits 变成概率分布，和 = 1）：

$$\text{softmax}(x_i) = \frac{e^{x_i}}{\sum_j e^{x_j}}$$

**log_softmax**（对 softmax 再取自然对数，因为 DPO 要的是 log 概率，所以直接一步到位）：

$$\text{log\_softmax}(x_i) = \log\left(\frac{e^{x_i}}{\sum_j e^{x_j}}\right) = x_i - \log\sum_j e^{x_j}$$

## 二、为什么不能直接 `log(softmax(x))`：数值不稳定

理论上可以先算 softmax 再取 log，但数值上会出问题：

1. **上溢（overflow）**：`exp(x)` 增长极快。`x=1000` 时 `e^1000` 直接爆成 `inf`，`inf/inf` 变 `nan`。
2. **下溢（underflow）**：`x` 很负时 `e^x → 0`，softmax 后变成 0，再 `log(0) = -inf`。

所以 PyTorch（以及所有框架）用的是 **log-sum-exp trick**，全程不直接算裸的 `exp(x)`。

## 三、稳定算法：log-sum-exp trick

先定义 **logsumexp**（对一堆指数求和再取 log 的稳定版）：

$$\text{logsumexp}(x) = \max(x) + \log\sum_j e^{x_j - \max(x)}$$

于是 log_softmax 等价写成：

$$\text{log\_softmax}(x_i) = \underbrace{(x_i - \max(x))}_{\text{① 平移}} \;-\; \underbrace{\log\sum_j e^{x_j - \max(x)}}_{\text{② logsumexp 的修正项}}$$

**逐行伪代码：**

```python
m = x.max()                          # ① 先找这一行 logits 的最大值
shifted = x - m                      # ② 所有元素减去 max（平移）
logsumexp = m + log(shifted.exp().sum())  # ③ 修正项：max + log(Σ exp(平移后))
log_softmax = x - logsumexp          # ④ 等价于 shifted - log(Σ exp(shifted))
```

## 四、为什么「减 max」是对的、且能保证稳定

**正确性（平移不变性）**：softmax 对整体平移不变，因为分子分母同时乘 `e^{-m}`：

$$\frac{e^{x_i}}{\sum_j e^{x_j}} = \frac{e^{x_i - m}}{\sum_j e^{x_j - m}}$$

减 max 不改变 softmax 结果，所以也不改变 log_softmax。

**稳定性（为什么不炸）**：
- 平移后最大元素 = 0，所以 `exp(0) = 1`，其余元素 `exp(负数) ∈ (0, 1]`；
- 于是 `Σ exp(shifted) ≥ 1`，`log(Σ) ≥ 0` 永远有定义（不会 `log(0)`）；
- 所有 `exp` 的自变量都 ≤ 0，`exp` 结果 ≤ 1，**永远不会上溢**（最大就是 1）。

## 五、具体数值例子

取 vocab_size=3 的某位置 logits：

```
x = [1.0, 2.0, 3.0]
```

**① 平移：** `m = max(x) = 3.0`

```
shifted = x - 3.0 = [-2.0, -1.0, 0.0]
```

**② 求 logsumexp：**

```
exp(shifted) = [e^-2.0, e^-1.0, e^0.0] = [0.1353, 0.3679, 1.0000]
sum          = 1.5032
log(sum)     = 0.4076
logsumexp    = 3.0 + 0.4076 = 3.4076
```

**③ 得到 log_softmax：**

```
log_softmax = x - logsumexp
            = [1.0-3.4076, 2.0-3.4076, 3.0-3.4076]
            = [-2.4076, -1.4076, -0.4076]
```

**验证**：反推 softmax = `[0.0900, 0.2447, 0.6652]`，三者相加 = 1.0 ✓；再取 log = `[-2.4076, -1.4076, -0.4076]`，与上面一致 ✓。

**性质检查**：三个 log_softmax 值全 ≤ 0（因为概率 ≤ 1，取 log 必为负），且 `exp` 回去求和 = 1。

## 六、放回 `logits_to_log_probs` 的上下文

```python
log_probs = F.log_softmax(logits, dim=2)  # 在 vocab 维度(dim=2)上归一化
log_probs_per_token = torch.gather(log_probs, dim=2, index=labels.unsqueeze(2)).squeeze(-1)
```

- `logits` shape `(batch, seq, vocab_size)`，`dim=2` 指定在**词表维度**上做 log_softmax——即对每个位置、每个 batch，独立地把该位置的 `vocab_size` 个分数归一化成 log 概率。
- `F.log_softmax` 内部就是用上面这套 log-sum-exp trick 实现的（PyTorch 的 `F.log_softmax` 和 `torch.logsumexp` 底层都走这个稳定公式）。
- 得到 `log_probs` 后，`torch.gather` 再按 `labels`（真实 token id）把对应位置的 log 概率抠出来，得到每个位置「真实 token 的 log 概率」。

---

**小结一句话**：log_softmax = 「先减 max 防溢出 → 用 log-sum-exp 算归一化分母的 log → 用 `x - logsumexp` 得到每个 token 的 log 概率」。它等价于 `log(softmax(x))`，但数值上稳定得多。

> 相关文档：DPO 训练脚本见 [[train_dpo详解]]，数据集见 [[DPODataset详解]]。
