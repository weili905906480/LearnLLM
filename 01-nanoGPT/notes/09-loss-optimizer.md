# C1 第4章：损失函数与优化器

> **C1 PyTorch Fundamentals** — DeepLearning.AI  
> 本文档覆盖：常用 Loss 函数 / 优化器 / 学习率 / 参数分组 / 标准训练步骤

---

## 核心概念

```
训练 = 不断缩小"预测值"和"真实值"之间的差距

  模型输出 ──→ 损失函数 ──→ loss（标量）──→ .backward() ──→ 优化器更新参数
                  ↑                                              ↑
              衡量差距                                      决定怎么更新

  Loss 函数：告诉模型"你离正确答案有多远"
  优化器：   告诉模型"参数往哪个方向调、调多少"
```

---

## 1. 损失函数（Loss Function）

### 1.1 `nn.MSELoss` — 均方误差（回归任务）

```python
criterion = nn.MSELoss()

y_pred = torch.tensor([2.5, 3.0, 4.1])   # 模型预测
y_true = torch.tensor([3.0, 3.0, 4.0])   # 真实值

loss = criterion(y_pred, y_true)
# loss = ((2.5-3)² + (3-3)² + (4.1-4)²) / 3 = (0.25 + 0 + 0.01) / 3 = 0.0867
```

| 参数 | 类型 | 默认 | 说明 |
|------|------|------|------|
| `reduction` | str | `'mean'` | `'mean'`=取平均 / `'sum'`=求和 / `'none'`=不归约 |

```
公式：MSE = (1/N) × Σ(y_pred - y_true)²

适用场景：回归任务（预测连续数值）
  房价预测
  温度预测
  股票收益预测

输入 shape：
  y_pred: [B, *]   任意 shape
  y_true: [B, *]   必须和 y_pred 相同
  输出:   标量（reduction='mean' 时）
```

---

### 1.2 `nn.CrossEntropyLoss` — 交叉熵（多分类任务）

```python
criterion = nn.CrossEntropyLoss()

# 模型输出 logits（未经 softmax 的原始分数）
logits = torch.tensor([[2.0, 1.0, 0.5],    # 样本1：3类的分数
                       [0.5, 2.5, 0.3]])   # 样本2：3类的分数
labels = torch.tensor([0, 1])              # 样本1属于类0，样本2属于类1

loss = criterion(logits, labels)
```

| 参数 | 类型 | 默认 | 说明 |
|------|------|------|------|
| `weight` | Tensor | `None` | 各类别的权重（类别不平衡时用） |
| `ignore_index` | int | `-100` | 忽略该标签（padding 时用） |
| `reduction` | str | `'mean'` | 同上 |
| `label_smoothing` | float | `0.0` | 标签平滑（0~1之间） |

```
内部做了什么（不需要你手动做）：

  logits [B, C]    C=类别数
     ↓ Softmax（转为概率）
  probs [B, C]     每行和=1
     ↓ Log
  log_probs [B, C]
     ↓ 取 labels 对应位置的值
  loss = -mean(log_probs[i, labels[i]])

所以：
  ⚠️ 模型最后一层 不要加 Softmax！
  CrossEntropyLoss 内部已经包含了

输入 shape：
  logits: [B, C]      B=batch, C=类别数
  labels: [B]         每个样本的类别索引（int，0~C-1）
  输出:   标量
```

```
可视化：

  logits:  [[2.0, 1.0, 0.5],    labels: [0, 1]
            [0.5, 2.5, 0.3]]

  Softmax后：
  probs:   [[0.59, 0.22, 0.19],  ← 样本1，预测类0概率0.59
            [0.13, 0.76, 0.11]]  ← 样本2，预测类1概率0.76

  loss = -( log(0.59) + log(0.76) ) / 2
       = -( -0.53 + -0.27 ) / 2
       = 0.40

  含义：预测概率越接近1，loss越小
```

---

### 1.3 `nn.BCEWithLogitsLoss` — 二分类

```python
criterion = nn.BCEWithLogitsLoss()

logits = torch.tensor([1.5, -0.5, 2.0])   # 模型输出（未经 sigmoid）
labels = torch.tensor([1.0, 0.0, 1.0])    # 真实标签（0或1）

loss = criterion(logits, labels)
```

| 参数 | 类型 | 默认 | 说明 |
|------|------|------|------|
| `pos_weight` | Tensor | `None` | 正样本权重（正负样本不平衡时用） |
| `reduction` | str | `'mean'` | 同上 |

```
内部做了什么：
  logits → sigmoid → 概率 → BCE loss

  等价于 nn.BCELoss(nn.Sigmoid(logits), labels)
  但数值更稳定（避免 log(0) 的问题）

输入 shape：
  logits: [B, *]   任意 shape
  labels: [B, *]   相同 shape，值为 0.0 或 1.0
  输出:   标量

使用场景：
  二分类：是/否，正/负
  多标签分类：一张图同时有"猫"和"狗"标签
```

---

### 1.4 Loss 选择速查

```
┌──────────────────┬─────────────────────────┬────────────────────┐
│ 任务类型          │ 推荐 Loss               │ 模型最后一层        │
├──────────────────┼─────────────────────────┼────────────────────┤
│ 回归（连续值）    │ MSELoss                 │ Linear（无激活）   │
│ 二分类            │ BCEWithLogitsLoss       │ Linear（无激活）   │
│ 多分类（单标签）  │ CrossEntropyLoss        │ Linear（无激活）   │
│ 多标签分类        │ BCEWithLogitsLoss       │ Linear（无激活）   │
└──────────────────┴─────────────────────────┴────────────────────┘

共同规则：模型最后一层输出 logits（原始分数），不要手动加激活函数
```

---

## 2. 优化器（Optimizer）

### 2.1 `torch.optim.SGD` — 随机梯度下降

```python
optimizer = torch.optim.SGD(
    model.parameters(),
    lr=0.01,
    momentum=0.9,
    weight_decay=1e-4
)
```

| 参数 | 类型 | 默认 | 说明 |
|------|------|------|------|
| `params` | iterable | 必填 | 模型参数，通常 `model.parameters()` |
| `lr` | float | 必填 | 学习率（步长大小） |
| `momentum` | float | `0` | 动量（加速收敛，减少震荡） |
| `weight_decay` | float | `0` | L2正则化系数（防过拟合） |

```
更新公式：

  无 momentum：
    w = w - lr × grad

  有 momentum（v 是速度，记住历史方向）：
    v = momentum × v + grad
    w = w - lr × v

  直觉：
    无 momentum：每步都是新的方向，容易震荡
    有 momentum：像滚球一样，保留惯性，跑更快更稳

  ┌─ 无 momentum ─┐      ┌─ 有 momentum ─┐
  │    / \ / \ /   │      │    ╲          │
  │   /   V   V    │      │     ╲         │
  │  到达最低点慢   │      │      → 直达   │
  └────────────────┘      └───────────────┘
```

---

### 2.2 `torch.optim.Adam` — 自适应学习率（最常用）

```python
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=1e-3,
    betas=(0.9, 0.999),
    eps=1e-8,
    weight_decay=0
)
```

| 参数 | 类型 | 默认 | 说明 |
|------|------|------|------|
| `lr` | float | `1e-3` | 学习率 |
| `betas` | tuple | `(0.9, 0.999)` | 一阶/二阶矩估计的衰减率 |
| `eps` | float | `1e-8` | 防止除零 |
| `weight_decay` | float | `0` | L2正则化 |

```
Adam 的核心思想：每个参数有自己的学习率

  SGD：所有参数用同一个 lr
  Adam：
    更新频繁的参数 → 自动减小步长
    更新稀疏的参数 → 自动增大步长
    → 每个参数自适应调节

  实际效果：
    收敛更快
    对 lr 的选择不那么敏感
    大多数场景直接用 lr=1e-3 就行
```

---

### 2.3 `torch.optim.AdamW` — 解耦 Weight Decay（推荐）

```python
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=1e-3,
    weight_decay=0.01
)
```

```
Adam vs AdamW 的区别：

  Adam + weight_decay：
    把 L2 正则项加到 loss 里再求梯度
    → 正则化效果被自适应学习率"稀释"

  AdamW：
    直接在参数更新时减去 weight_decay × w
    → 正则化效果独立于梯度，更稳定

  结论：用 AdamW 替代 Adam（几乎所有现代模型都用 AdamW）
```

---

### 2.4 优化器选择建议

```
┌──────────────────┬──────────────────────────────┐
│ 场景              │ 推荐优化器                    │
├──────────────────┼──────────────────────────────┤
│ 刚开始实验        │ Adam(lr=1e-3)                │
│ 想要最好效果      │ AdamW(lr=1e-3, wd=0.01)     │
│ 大规模LLM训练    │ AdamW + cosine LR schedule   │
│ 简单小模型        │ SGD(lr=0.01, momentum=0.9)   │
│ 微调预训练模型    │ AdamW(lr=1e-5~5e-5)          │
└──────────────────┴──────────────────────────────┘
```

---

## 3. 学习率（Learning Rate）

```python
# 学习率太大：
lr = 1.0       # loss 震荡不收敛，甚至爆炸 → NaN

# 学习率太小：
lr = 1e-7      # loss 几乎不动，训练极慢

# 合适的学习率：
lr = 1e-3      # Adam 默认值，大多数场景可用
lr = 1e-4~5e-5 # 微调预训练模型时
```

```
学习率对训练的影响：

loss
 ↑
 │╲  lr太大：loss爆炸或震荡
 │ ╲___
 │     ╲___  lr合适：平稳下降
 │         ╲____
 │              ╲______  lr太小：下降极慢
 │
 └──────────────────────→ epoch

经验起点：
  Adam:  lr = 1e-3
  SGD:   lr = 0.01 ~ 0.1
  微调:   lr = 1e-5 ~ 5e-5
```

---

## 4. 不同参数组用不同学习率

```python
# 场景：微调预训练模型
# 预训练层学习率小（保护已学知识），新增层学习率大（快速学新任务）

model = nn.Sequential(
    nn.Linear(784, 256),   # 假设这是预训练层
    nn.ReLU(),
    nn.Linear(256, 10)     # 新增的分类层
)

optimizer = torch.optim.Adam([
    {'params': model[0].parameters(), 'lr': 1e-5},   # 预训练层：小 lr
    {'params': model[2].parameters(), 'lr': 1e-3},   # 新增层：大 lr
])
```

| 参数 | 说明 |
|------|------|
| `params` | 该组包含的参数 |
| `lr` | 该组的学习率（覆盖全局 lr） |
| `weight_decay` | 该组的正则化系数 |

```
微调策略示意：

  ┌───────────────────────────────────────┐
  │ 预训练层（已学好）                      │
  │   → 小 lr (1e-5)，轻微调整即可        │
  │   → 知识已经在里面，不要破坏           │
  ├───────────────────────────────────────┤
  │ 新增层（随机初始化）                    │
  │   → 大 lr (1e-3)，需要快速学习         │
  │   → 从零开始，需要大步更新             │
  └───────────────────────────────────────┘
```

---

## 5. 标准训练步骤（每个 iteration）

```python
for epoch in range(num_epochs):
    for X, y in train_loader:
        # ① 清零梯度
        optimizer.zero_grad()

        # ② 前向传播
        logits = model(X)

        # ③ 计算损失
        loss = criterion(logits, y)

        # ④ 反向传播
        loss.backward()

        # ⑤ 参数更新
        optimizer.step()
```

```
顺序不能乱！必须严格遵守：

  ① zero_grad()    清掉上一步残留梯度
       ↓
  ② model(X)       前向传播得到预测
       ↓
  ③ criterion()    计算 loss
       ↓
  ④ backward()     反向传播计算梯度
       ↓
  ⑤ step()         用梯度更新参数

常见错误：
  忘了 zero_grad() → 梯度累积，越来越大
  backward() 后才计算 loss → 报错
  step() 放在 backward() 前 → 用的是旧梯度
```

---

## 6. 完整训练+评估示例

```python
model     = MLP(784, 256, 10).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

for epoch in range(10):
    # ── 训练 ──
    model.train()
    train_loss = 0
    for X, y in train_loader:
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()
        loss = criterion(model(X), y)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    # ── 评估 ──
    model.eval()
    correct = 0
    with torch.no_grad():
        for X, y in val_loader:
            X, y = X.to(device), y.to(device)
            pred = model(X).argmax(dim=1)
            correct += (pred == y).sum().item()

    acc = correct / len(val_loader.dataset)
    print(f"Epoch {epoch}: loss={train_loss/len(train_loader):.4f}, acc={acc:.3f}")
```

```
训练 vs 评估模式的区别：

  model.train() + optimizer：
    Dropout 工作，BatchNorm 用 batch 统计量
    计算梯度，更新参数

  model.eval() + torch.no_grad()：
    Dropout 关闭，BatchNorm 用全局统计量
    不计算梯度 → 省内存省计算
    只算指标，不更新
```

---

## 快速速查表

| 组件 | 推荐选择 | 关键参数 |
|------|---------|---------|
| 回归 Loss | `MSELoss()` | reduction='mean' |
| 分类 Loss | `CrossEntropyLoss()` | 输入是 logits，不要加 Softmax |
| 二分类 Loss | `BCEWithLogitsLoss()` | 输入是 logits，不要加 Sigmoid |
| 优化器 | `AdamW(lr=1e-3, wd=0.01)` | 大多数场景首选 |
| 微调 lr | `1e-5 ~ 5e-5` | 预训练模型用小 lr |
| 新层 lr | `1e-3` | 新增层用默认 lr |

---

## 我的理解 / 疑问

<!-- 在这里记录学习笔记 -->
