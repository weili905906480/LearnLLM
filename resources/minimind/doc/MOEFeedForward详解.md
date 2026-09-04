# MOEFeedForward 详解（含具体矩阵举例）

> 源码位置：`model/model_minimind.py:148-176`
> 说明：下面用一组**具体数值的矩阵**把 `MOEFeedForward.forward` 从头到尾走一遍。为便于手算，把维度缩小为 `hidden_size=4`、`num_experts=3`、`num_experts_per_tok=1`（top-1，与默认一致），输入 `batch=1, seq_len=4`（即 4 个 token）。真实代码里是 `768 / 4 专家`，机制完全一样。

## 完整代码

```python
class MOEFeedForward(nn.Module):
    def __init__(self, config: MiniMindConfig):
        super().__init__()
        self.config = config
        self.gate = nn.Linear(config.hidden_size, config.num_experts, bias=False)
        self.experts = nn.ModuleList([FeedForward(config, intermediate_size=config.moe_intermediate_size) for _ in range(config.num_experts)])
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        batch_size, seq_len, hidden_dim = x.shape
        x_flat = x.view(-1, hidden_dim)
        scores = F.softmax(self.gate(x_flat), dim=-1)
        topk_weight, topk_idx = torch.topk(scores, k=self.config.num_experts_per_tok, dim=-1, sorted=False)
        if self.config.norm_topk_prob:
            topk_weight = topk_weight / (topk_weight.sum(dim=-1, keepdim=True) + 1e-20)
        y = torch.zeros_like(x_flat)
        for i, expert in enumerate(self.experts):
            mask = (topk_idx == i)
            if mask.any():
                token_idx = mask.any(dim=-1).nonzero().flatten()
                weight = topk_weight[mask].view(-1, 1)
                y.index_add_(0, token_idx, (expert(x_flat[token_idx]) * weight).to(y.dtype))
            elif self.training:
                y[0, 0] += 0 * sum(p.sum() for p in expert.parameters())
        if self.training and self.config.router_aux_loss_coef > 0:
            load = F.one_hot(topk_idx, self.config.num_experts).float().mean(0)
            self.aux_loss = (load * scores.mean(0)).sum() * self.config.num_experts * self.config.router_aux_loss_coef
        else:
            self.aux_loss = scores.new_zeros(1).squeeze()
        return y.view(batch_size, seq_len, hidden_dim)
```

---

## 一、`__init__`：三个成员

```python
self.gate = nn.Linear(hidden_size, num_experts, bias=False)
```

**门控 / 路由器（Router）**：输入 `hidden_size` 维，输出 `num_experts`（默认 4）维。它给每个 token 的每个专家打一个分数。`bias=False` 是为了少一点参数、少一点 rank 倾向。

```python
self.experts = nn.ModuleList([
    FeedForward(config, intermediate_size=config.moe_intermediate_size)
    for _ in range(config.num_experts)
])
```

**专家集合**：`num_experts` 个独立的 `FeedForward`。注意每个专家内部就是标准的 **SwiGLU FFN**（`gate_proj` + `up_proj` + `down_proj`，silu 激活）。所以 MoE 的本质只是——**把"一个 FFN"换成"多个 FFN + 一个路由"，每个 token 只走其中被选中的少量专家**。

```python
self.act_fn = ACT2FN[config.hidden_act]
```

这一行实际是冗余的（专家内部已各自用了 `act_fn`，这里没被引用），属于遗留代码，可忽略。

---

## 二、`forward` 逐步流程（配矩阵举例）

### 0. 输入 x

形状 `[1, 4, 4]`，4 个 token，每个 token 是 4 维向量：

```
token 0 (t0): [1, 0, 0, 1]
token 1 (t1): [0, 1, 1, 0]
token 2 (t2): [1, 1, 0, 0]
token 3 (t3): [0, 0, 1, 1]
```

### 1. 展平

```python
x_flat = x.view(-1, hidden_dim)   # [1,4,4] -> [4,4]
```

MoE 路由是 **token 级**的，所以把 batch 和 seq 合并，得到 `N=4` 行、每行一个 token：

```
x_flat =              （行号 = token 序号）
  [1, 0, 0, 1]    <- t0（行 0）
  [0, 1, 1, 0]    <- t1（行 1）
  [1, 1, 0, 0]    <- t2（行 2）
  [0, 0, 1, 1]    <- t3（行 3）
```

### 2. 门控打分 gate

```python
self.gate = nn.Linear(4, 3, bias=False)
scores = F.softmax(self.gate(x_flat), dim=-1)
```

假设 gate 的权重矩阵 `W_gate` 形状 `[4, 3]`（4 个输入维度 → 3 个专家）为：

```
       e0   e1   e2
dim0: [ 2,   0,  -1 ]
dim1: [-1,   2,   0 ]
dim2: [ 0,  -1,   2 ]
dim3: [ 1,   1,  -2 ]
```

先算 logits `= x_flat @ W_gate`，形状 `[4, 3]`：

```
t0: [1,0,0,1] -> [3,  1, -3]     # 1*2+1*1=3 ; 0+0+1*1=1 ; -1+(-2)=-3
t1: [0,1,1,0] -> [-1, 1,  2]
t2: [1,1,0,0] -> [1,  2, -1]
t3: [0,0,1,1] -> [1,  0,  0]
```

再 softmax（按行归一化），得到 `scores` `[4, 3]`：

```
          e0      e1      e2
t0:  [ 0.879,  0.119,  0.002 ]
t1:  [ 0.035,  0.260,  0.705 ]
t2:  [ 0.260,  0.705,  0.035 ]
t3:  [ 0.576,  0.212,  0.212 ]
```

> 每一行就是"这个 token 应该被分给各专家的概率分布"。

### 3. Top-k 选专家

```python
topk_weight, topk_idx = torch.topk(scores, k=1, dim=-1, sorted=False)
```

每行取最大的 1 个（列索引即专家编号）：

```
topk_idx    = [[0],     # t0 -> 专家 0
               [2],     # t1 -> 专家 2
               [1],     # t2 -> 专家 1
               [0]]     # t3 -> 专家 0

topk_weight = [[0.879],
               [0.705],
               [0.705],
               [0.576]]
```

### 4. 归一化 top-k 权重

```python
if config.norm_topk_prob:
    topk_weight = topk_weight / (topk_weight.sum(dim=-1, keepdim=True) + 1e-20)
```

**在 k=1 时，每个 token 只选了 1 个专家**，分母就是它自己，除完恒等于 1：

```
topk_weight -> [[1.0], [1.0], [1.0], [1.0]]
```

> 所以 top-1 下这步实际上不起作用，权重统一变 1（每个 token 只走一个专家，系数天然为 1）。它是为 `k>1` 预留的——当 top-2 时，两个专家的权重会被重新归一化成和为 1。

### 5. 逐专家分桶 + 前向 + index_add_

这是最核心的循环。为便于手算，把每个专家**简化成一个 4×4 的线性变换矩阵**（真实代码是 SwiGLU FFN，机制相同）：

```
专家 0 (W0) = 恒等矩阵      -> 输出 = 输入
专家 1 (W1) = 2 × 单位矩阵   -> 输出 = 2 × 输入
专家 2 (W2) = 全 0 矩阵      -> 输出 = 0
```

先根据 `topk_idx` 做**分桶**（哪个 token 进哪个专家）：

| 专家 | 被路由到的 token（行号） |
|------|------------------------|
| 0 | t0（行0）、t3（行3） |
| 1 | t2（行2） |
| 2 | t1（行1） |

初始化输出为全零：

```python
y = torch.zeros_like(x_flat)   # 4×4 全 0
```

#### 专家 0 迭代

```python
mask = (topk_idx == 0)          # [[T],[F],[F],[T]]
token_idx = mask.any(dim=-1).nonzero().flatten()  # [0, 3]
```

即"行 0 和行 3 被路由到专家 0"。取出这两行，过专家 0（恒等），乘权重（1.0）：

```
expert0(x_flat[[0,3]]) = [[1,0,0,1],   # t0 不变
                          [0,0,1,1]]   # t3 不变
```

```python
y.index_add_(0, [0,3], [[1,0,0,1],[0,0,1,1]])
```

`index_add_(0, idx, src)` 的语义是 `y[idx[i]] += src[i]`，于是：

```
y =
  [1, 0, 0, 1]    <- 行0 写入 t0
  [0, 0, 0, 0]
  [0, 0, 0, 0]
  [0, 0, 1, 1]    <- 行3 写入 t3
```

#### 专家 1 迭代

```python
mask = (topk_idx == 1)   # [[F],[F],[T],[F]]
token_idx = [2]          # 只有行 2
```

```
expert1(x_flat[[2]]) = 2 × [1,1,0,0] = [2,2,0,0]
```

```python
y.index_add_(0, [2], [[2,2,0,0]])
```

```
y =
  [1, 0, 0, 1]
  [0, 0, 0, 0]
  [2, 2, 0, 0]    <- 行2 写入 t2（×2）
  [0, 0, 1, 1]
```

#### 专家 2 迭代

```python
mask = (topk_idx == 2)   # [[F],[T],[F],[F]]
token_idx = [1]
```

```
expert2(x_flat[[1]]) = 0 × [0,1,1,0] = [0,0,0,0]
```

加 0，`y` 不变：

```
y =
  [1, 0, 0, 1]    <- t0 走了专家 0（恒等）
  [0, 0, 0, 0]    <- t1 走了专家 2（零映射）
  [2, 2, 0, 0]    <- t2 走了专家 1（放大 2 倍）
  [0, 0, 1, 1]    <- t3 走了专家 0（恒等）
```

#### 还原形状

```python
return y.view(batch_size, seq_len, hidden_dim)   # [4,4] -> [1,4,4]
```

**关键结论**：每个 token 只经过自己被选中的那一个专家，结果按"行号"散落回输出对应位置。`index_add_` 就是做这个"按行索引归位"的。如果是 top-2，同一行会被多个专家累加。

---

## 三、负载均衡 aux loss（训练时）

```python
load = F.one_hot(topk_idx, num_experts).float().mean(0)   # [3]
self.aux_loss = (load * scores.mean(0)).sum() * num_experts * router_aux_loss_coef
```

**`load`（每个专家实际被选中的平均频率）**：把 `topk_idx = [[0],[2],[1],[0]]` 做 one-hot 再对 4 个 token 平均：

```
one_hot:  [1,0,0] / [0,0,1] / [0,1,0] / [1,0,0]
load  = ( [2,1,1] ) / 4 = [0.50, 0.25, 0.25]
```

**`scores.mean(0)`（门控对各专家的平均打分）**：

```
e0: (0.879+0.035+0.260+0.576)/4 = 0.4375
e1: (0.119+0.260+0.705+0.212)/4 = 0.3240
e2: (0.002+0.705+0.035+0.212)/4 = 0.2385
-> [0.4375, 0.3240, 0.2385]
```

**aux loss 计算**（`num_experts=3`，`router_aux_loss_coef=5e-4`）：

```
load * scores.mean(0) = [0.5×0.4375, 0.25×0.324, 0.25×0.2385]
                      = [0.21875, 0.081, 0.05963]
sum                   = 0.35938
× num_experts (3)     = 1.0781
× coef (5e-4)         = 0.000539
```

$$L_{aux} = N \cdot \alpha \cdot \sum_i \underbrace{f_i}_{\text{实际负载}} \cdot \underbrace{P_i}_{\text{门控均值}} = 5.39\times10^{-4}$$

它惩罚"路由不均"：如果所有 token 都挤到专家 0（load 变成 `[1,0,0]`），这一项会变大，训练时通过梯度把 router 拉向更均匀的分配，避免其他专家"饿死"。

---

## 四、aux_loss 反向传播机制（如何鼓励均匀）

### 1. 反向传播的完整链路

`self.aux_loss` 并不在原地 backward，而是逐级向上汇总，最后与 CE loss 加在一起才 backward：

- **第 1 步**：`MOEFeedForward.forward` 计算并暂存 `self.aux_loss`
- **第 2 步**：`MiniMindModel.forward` 汇总所有 MoE 层（`model_minimind.py:231`）

```python
aux_loss = sum([l.mlp.aux_loss for l in self.layers if isinstance(l.mlp, MOEFeedForward)], ...)
```

- **第 3 步**：`MiniMindForCausalLM.forward` 放进输出 `MoeCausalLMOutputWithPast(aux_loss=...)`
- **第 4 步**：训练脚本加到总 loss 再 backward（`train_pretrain.py:37-40`）

```python
res = model(input_ids, labels=labels)
loss = res.loss + res.aux_loss        # CE + aux
loss = loss / args.accumulation_steps
scaler.scale(loss).backward()
```

即：`CE loss` 与 `aux_loss` 一起求导、梯度同时回传，`aux_loss` 只是总 loss 的一个加性正则项。

### 2. 梯度流向哪些参数？（只更新 router，不更新 expert）

```python
load  = F.one_hot(topk_idx, ...).float().mean(0)   # ← 常数，无梯度
scores = F.softmax(self.gate(x_flat), dim=-1)      # ← 有梯度
```

- `load` 来自 `topk` 的整数索引 → `one_hot` → `float` → `mean`，全是不可微的离散操作，是叶子常数，**不产生梯度**。
- `scores` 来自 `softmax(gate(x))`，`gate` 可训练，**有梯度**。

因此梯度只能沿 `scores → gate → gate.weight` 回传，**aux loss 只更新 router 权重，不动专家/注意力等其它层**。

### 3. 梯度长什么样（数学推导）

展开公式（`P_i = (1/N) Σ_j s_{j,i}`）：

$$L_{aux} = N\alpha\sum_i f_i P_i = N\alpha\sum_i f_i\left(\frac1N\sum_j s_{j,i}\right) = \alpha\sum_i\sum_j f_i\, s_{j,i}$$

> `N` 与 `1/N` 抵消，净剩 `coef`，所以梯度形式很简洁。

对打分求偏导（先视 softmax 为独立变量）：

$$\frac{\partial L_{aux}}{\partial s_{j,i}} = \alpha\, f_i$$

**梯度正比于专家 i 当前的负载 `f_i`**。再经 softmax 链式法则传到 logits $z_{j,i}$：

$$\frac{\partial L_{aux}}{\partial z_{j,i}} = \alpha\, s_{j,i}\left(f_i - \sum_k f_k s_{j,k}\right)$$

直觉：当专家 i 的实际负载 `f_i` 高于"该 token 打分加权后的平均负载"时，梯度为正 → 优化器压低 gate 对专家 i 的打分；反之抬高。

### 4. 为什么能"鼓励均匀"

**① 函数形状：越均匀值越小**

| 场景 | load = [f₀,f₁,f₂] | P = [P₀,P₁,P₂] | Σ f·P | ×N×α |
|------|------|------|-------|-------|
| 完全均匀 | [1/3,1/3,1/3] | [1/3,1/3,1/3] | 1/3 | =1 |
| 轻度不均 | [0.5,0.25,0.25] | [0.44,0.32,0.24] | 0.359 | =1.08 |
| 极端不均（全挤专家0） | [1,0,0] | [1,0,0] | 1 | =3 |

> 乘 `num_experts` 正是为了让均匀时的下界恒为 1，量级稳定，便于 `coef` 统一控强。

**② 梯度方向：给过热专家降温**

`∂L_aux/∂s ∝ f_i`：负载高的专家梯度大且为正 → 下调其 logits；负载低的专家惩罚弱 → 相对上调。一轮后 gate 对过热专家降温、冷门专家提温，下一轮 top-k 选中分布随之改变，`load` 更新，形成"打分 ↔ 负载"负反馈，最终收敛到均匀。

### 5. 为什么是 `load ⊙ scores` 这种形式

`load` 来自 argmax，**本身不可微**，无法直接写 `minimize(entropy(load))` 去优化它。于是把**不可微的 `load` 当常数权重**、**可微的 `scores` 当被优化量**相乘：既让梯度沿 `scores` 回传，又把"当前负载"信息编码进梯度方向。这正是 Switch Transformer（Fedus et al., 2021）的 load-balancing loss 原版做法。

**一句话**：`aux_loss` 作为总 loss 正则项一起 backward，梯度只经 `scores→gate` 流回 router；梯度方向正比于各专家当前负载，被过度使用的专家降温、被冷落的提温，经多轮负反馈让路由趋于均匀。

---

## 五、一张图总结整个数据流

```
x [1,4,4]
  │ view
  ▼
x_flat [4,4]  ──gate(4→3)──▶ logits [4,3] ──softmax──▶ scores [4,3]
                                                          │ topk(1)
                                                          ▼
                                              topk_idx [4,1]  topk_weight [4,1]
                                                          │
        ┌──────────────┬──────────────┬───────────────────┘（按专家分桶）
        ▼              ▼              ▼
    专家0(行0,3)    专家1(行2)     专家2(行1)
        │              │              │
        └────── index_add_ 累加 ──────┘
                          ▼
                    y [4,4] ──view──▶ [1,4,4]

训练时另算：load ⊙ scores.mean(0) → aux_loss
```

---

## 六、一句话总结

**`gate` 打分 → top-k 路由 → 每个 token 只进被选中的专家 → `index_add_` 按行号归位 → 训练时再加负载均衡 aux loss。**

设计亮点：

1. **稀疏激活**：top-1 下每个 token 只算一个专家，用 `198M` 总参数换 `~64M` 激活参数，容量更大但推理更慢（比 dense 慢约 50%）。
2. **token 级分桶**：用 `index_add_` 而非显式 gather/scatter，是纯 PyTorch 下的简洁写法。
3. **负载均衡兜底**：既算 aux loss 鼓励均匀，又用 `0 * sum(params)` 防止饿死专家丢失梯度。
