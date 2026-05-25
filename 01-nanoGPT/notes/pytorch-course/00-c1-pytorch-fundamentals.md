# C1 PyTorch Fundamentals — 课程总览

> **DeepLearning.AI — PyTorch for Deep Learning Professional Certificate**  
> 第一门课：PyTorch 基础，从零搭建并训练第一个神经网络

---

## 课程目标

```
学完本课你能做到：

  ✅ 熟练操作 Tensor（创建、索引、变形、运算）
  ✅ 理解自动微分原理，手写梯度下降
  ✅ 用 nn.Module 搭建任意结构的神经网络
  ✅ 选择合适的 Loss 函数和优化器
  ✅ 构建 Dataset/DataLoader 加载任意数据
  ✅ 完成完整的训练 + 评估循环
```

---

## 章节导航

| 章节 | 文件 | 核心内容 | 关键 API |
|------|------|---------|---------|
| 第1章 | [06-tensors.md](./06-tensors.md) | 张量基础 | `torch.tensor` / `reshape` / `@` |
| 第2章 | [07-autograd.md](./07-autograd.md) | 自动微分 | `requires_grad` / `.backward()` / `no_grad` |
| 第3章 | [08-nn-module.md](./08-nn-module.md) | 构建神经网络 | `nn.Module` / `nn.Linear` / `Dropout` |
| 第4章 | [09-loss-optimizer.md](./09-loss-optimizer.md) | 损失函数与优化器 | `CrossEntropyLoss` / `AdamW` |
| 第5章 | [10-dataset-dataloader.md](./10-dataset-dataloader.md) | 数据加载 | `Dataset` / `DataLoader` / `transforms` |
| 第6章 | [11-training-loop.md](./11-training-loop.md) | 完整训练循环 | 训练 + 评估 + checkpoint |

---

## 各章核心知识点

---

### 第1章：Tensor 张量基础

```python
# 创建
t = torch.tensor([[1.0, 2.0], [3.0, 4.0]])   # shape (2, 2)
t = torch.zeros(3, 4)     # shape (3, 4)
t = torch.rand(2, 3)      # shape (2, 3)

# 属性
t.shape    # torch.Size([2, 3])
t.dtype    # torch.float32
t.device   # cpu / cuda

# 形状变换
t.reshape(3, -1)           # -1 自动推断
t.squeeze(0)               # 去掉 size=1 的维度
t.unsqueeze(0)             # 插入 size=1 的维度
t.permute(0, 2, 3, 1)     # 重排维度 NCHW → NHWC

# 运算
A @ B                      # 矩阵乘法
t.sum(dim=0)               # 沿第0维求和，该维消失
t.sum(dim=0, keepdim=True) # 保留维度（变为1）
```

```
dim 核心规则：dim=N → 第N维消失，其余维度保留

  shape (2, 3)：
    sum(dim=0) → (3,)    第0维(大小=2)消失
    sum(dim=1) → (2,)    第1维(大小=3)消失

shape (3,) 的写法：Python 一维元组必须加逗号
  (3)  → int 3（不是 shape！）
  (3,) → tuple，表示只有1个维度、大小为3
```

---

### 第2章：Autograd 自动微分

```python
x = torch.tensor(2.0, requires_grad=True)
y = x ** 2 + 3 * x        # 构建计算图

y.backward()               # 反向传播
print(x.grad)              # dy/dx = 2x+3 = 7

# 每次 backward 前必须清零！
optimizer.zero_grad()      # 否则梯度累积

# 推理/更新时关闭追踪
with torch.no_grad():
    w -= lr * w.grad

y.detach()                 # 从计算图中分离
```

```
训练循环标准顺序（每次 iteration）：

  ① optimizer.zero_grad()    清零梯度
  ② logits = model(X)        前向传播
  ③ loss = criterion(...)    计算 loss
  ④ loss.backward()          反向传播
  ⑤ optimizer.step()         更新参数
```

---

### 第3章：nn.Module 构建神经网络

```python
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1  = nn.Linear(784, 256)
        self.fc2  = nn.Linear(256, 10)
        self.relu = nn.ReLU()
        self.drop = nn.Dropout(0.3)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.drop(x)
        return self.fc2(x)      # 最后一层不加激活！

model.train()    # 训练模式（Dropout 工作）
model.eval()     # 评估模式（Dropout 关闭）
```

```
数据流 shape：

  [B, 784] → Linear(784→256) → [B, 256]
           → ReLU             → [B, 256]
           → Dropout          → [B, 256]
           → Linear(256→10)   → [B, 10]  logits
```

---

### 第4章：损失函数与优化器

```python
# Loss
criterion = nn.CrossEntropyLoss()      # 多分类（最常用）
criterion = nn.MSELoss()               # 回归
criterion = nn.BCEWithLogitsLoss()     # 二分类 / 多标签

# 优化器
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=1e-3,
    weight_decay=0.01
)
```

```
Loss 选择：
  回归任务    → MSELoss          输入：连续值
  多分类任务  → CrossEntropy     输入：logits (B, C)，标签 (B,)
  二分类任务  → BCEWithLogits    输入：logits (B,)，标签 (B,) float

优化器选择：
  默认首选    → AdamW(lr=1e-3, wd=0.01)
  微调预训练  → AdamW(lr=1e-5 ~ 5e-5)
```

---

### 第5章：Dataset 和 DataLoader

```python
class MyDataset(Dataset):
    def __init__(self, X, y):
        self.X, self.y = X, y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

loader = DataLoader(dataset, batch_size=32, shuffle=True)

for X_batch, y_batch in loader:
    ...   # X_batch: [32, features]
```

```
Dataset 三要素：
  __init__     加载 / 存储数据
  __len__      返回数据集大小
  __getitem__  返回第 idx 个样本 (x, y)
```

---

### 第6章：完整训练循环

```python
def train(model, loader, optimizer, criterion, device):
    model.train()
    for X, y in loader:
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()
        loss = criterion(model(X), y)
        loss.backward()
        optimizer.step()

@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    correct = 0
    for X, y in loader:
        X, y = X.to(device), y.to(device)
        correct += (model(X).argmax(1) == y).sum().item()
    return correct / len(loader.dataset)

# 主循环 + 保存最优
for epoch in range(num_epochs):
    train(model, train_loader, optimizer, criterion, device)
    acc = evaluate(model, val_loader, device)
    if acc > best_acc:
        best_acc = acc
        torch.save(model.state_dict(), 'best.pth')
```

---

## 知识点关系图

```
Tensor（数据容器）
    ↓ requires_grad=True
Autograd（自动计算梯度）
    ↓
nn.Module（组织网络结构）
    ↓ 输出 logits
Loss Function（衡量误差）
    ↓ .backward()
Optimizer（更新参数）
    ↑
Dataset / DataLoader（喂数据）

五者合在一起 = 完整训练流程
```

---

## 常见错误速查

| 错误现象 | 原因 | 解决方案 |
|---------|------|---------|
| loss 越来越大 / NaN | 学习率过大 | 减小 lr，试试 1e-3 |
| 梯度越来越大 | 没有清零梯度 | 添加 `optimizer.zero_grad()` |
| val loss 远高于 train loss | 过拟合 | 加 Dropout、减小模型 |
| train/val loss 都很高 | 欠拟合 | 增大模型、增加层数 |
| `.numpy()` 报错 | tensor 有梯度 | 改用 `.detach().numpy()` |
| CUDA out of memory | 显存不足 | 减小 batch_size |
| 预测全是同一类 | lr 太小或结构问题 | 检查 lr 和最后一层 |
| RuntimeError: device mismatch | 数据和模型设备不同 | 统一 `.to(device)` |

---

## 推荐学习顺序

```
第1章 Tensor ──→ 第2章 Autograd ──→ 第3章 nn.Module
  数据表示          梯度计算            网络搭建
      ↓                                    ↓
第5章 DataLoader ←── 第6章 训练循环 ←── 第4章 Loss + Optimizer
  数据加载              全部拼在一起        如何训练
```

---

## 下一步：C2 进阶技术

```
C1 完成后，C2 将学习：
  ✦ 学习率调度（LR Scheduler）
  ✦ 超参数自动搜索（Optuna）
  ✦ 数据增强（transforms）
  ✦ 迁移学习（Transfer Learning）
  ✦ 训练监控（TensorBoard）
  ✦ 模型评估指标（Precision / Recall / F1）
```
