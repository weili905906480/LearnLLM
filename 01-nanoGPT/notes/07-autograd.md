# C1 第2章：Autograd 自动微分

> **C1 PyTorch Fundamentals** — DeepLearning.AI  
> 本文档覆盖 PyTorch 自动微分的核心机制：计算图、反向传播、梯度操作

---

## 核心概念速览

```
前向传播                          反向传播
x ──→ y = x² + 2x ──→ loss      loss ──→ dy/dx（自动计算）

PyTorch 在前向传播时自动构建计算图
调用 .backward() 时沿图反向传播梯度
```

---

## 1. `requires_grad` — 开启梯度追踪

```python
# 叶子节点（需要计算梯度的参数）
x = torch.tensor(3.0, requires_grad=True)
w = torch.tensor(2.0, requires_grad=True)
b = torch.tensor(1.0, requires_grad=True)

# 普通张量（不追踪）
data = torch.tensor([1.0, 2.0, 3.0])          # requires_grad=False（默认）
```

| 参数 | 类型 | 默认 | 说明 |
|------|------|------|------|
| `requires_grad` | bool | `False` | `True` = 追踪该张量上的所有操作，用于计算梯度 |

```
requires_grad=True 的张量：
  ┌──────────────────────────────┐
  │  x = tensor(3.0)             │
  │  requires_grad = True        │
  │  grad = None（反向传播前）     │
  │  grad_fn = None（叶子节点）   │
  └──────────────────────────────┘

经过运算后产生的中间张量：
  ┌──────────────────────────────┐
  │  y = x² = tensor(9.0)       │
  │  requires_grad = True        │
  │  grad_fn = <PowBackward0>    │  ← 记录了"如何反向传播"
  └──────────────────────────────┘
```

---

## 2. 计算图（Computational Graph）

```python
x = torch.tensor(2.0, requires_grad=True)
w = torch.tensor(3.0, requires_grad=True)
b = torch.tensor(1.0, requires_grad=True)

z = w * x + b    # z = 3*2 + 1 = 7
```

```
计算图（前向传播时自动构建）：

x(2) ──┐
       ├──→ [mul] ──→ wx(6) ──→ [add] ──→ z(7)
w(3) ──┘                   ↑
                      b(1) ─┘

每个节点记录：
  - 输入值
  - 操作类型（mul / add / ...）
  - 如何计算梯度（grad_fn）
```

**反向传播后的梯度：**

```python
z.backward()

# dz/dw = x = 2
# dz/dx = w = 3
# dz/db = 1

print(w.grad)   # tensor(2.)
print(x.grad)   # tensor(3.)
print(b.grad)   # tensor(1.)
```

```
反向传播（链式法则）：

z = wx + b
∂z/∂w = x = 2    ← w 的梯度
∂z/∂x = w = 3    ← x 的梯度
∂z/∂b = 1        ← b 的梯度
```

---

## 3. `.backward()` — 反向传播

```python
x = torch.tensor([[1.0, 2.0],
                  [3.0, 4.0]], requires_grad=True)

y = x ** 2          # y = x²，逐元素
z = y.sum()         # z = sum(x²)，标量

z.backward()        # 反向传播
```

| 参数 | 类型 | 默认 | 说明 |
|------|------|------|------|
| `gradient` | Tensor | `None` | 非标量输出时需传入梯度张量（通常为全1） |
| `retain_graph` | bool | `False` | `True` = 保留计算图，可多次反向传播 |
| `create_graph` | bool | `False` | `True` = 创建高阶梯度图 |

```
x:                   y = x²:              z = y.sum()：
┌─────┬─────┐        ┌─────┬─────┐
│  1  │  2  │  ──→   │  1  │  4  │  ──→  scalar(30)
│  3  │  4  │        │  9  │ 16  │
└─────┴─────┘        └─────┴─────┘
shape (2, 2)         shape (2, 2)

x.grad = dz/dx = 2x：
┌─────┬─────┐
│  2  │  4  │   ← 2×1, 2×2
│  6  │  8  │   ← 2×3, 2×4
└─────┴─────┘
shape (2, 2)
```

> ⚠️ `.backward()` 只能对**标量**（0维张量）直接调用  
> 若 z 是向量/矩阵，需传入 `gradient` 参数

---

## 4. `.grad` — 梯度存储与累积

```python
x = torch.tensor(2.0, requires_grad=True)

# 第一次反向传播
y = x ** 2
y.backward()
print(x.grad)    # tensor(4.)   dy/dx = 2x = 4

# 第二次反向传播（不清零）
y = x ** 2
y.backward()
print(x.grad)    # tensor(8.)  ← 梯度累加了！4 + 4 = 8

# 正确做法：每次反向传播前清零
x.grad.zero_()
y = x ** 2
y.backward()
print(x.grad)    # tensor(4.)  ← 正确
```

```
梯度累积问题（训练循环中的常见错误）：

iteration 1：loss.backward() → x.grad = 4
iteration 2：loss.backward() → x.grad = 8  ← 应该是 4！
iteration 3：loss.backward() → x.grad = 12

解决方案：每次 backward() 前调用 optimizer.zero_grad()
         或手动 x.grad.zero_()
```

| 操作 | 说明 |
|------|------|
| `x.grad` | 存储梯度值，反向传播前为 `None` |
| `x.grad.zero_()` | 原地清零（`_` 后缀 = in-place） |
| `optimizer.zero_grad()` | 清零所有参数的梯度（训练循环标准写法） |

---

## 5. 停止梯度追踪

### 5.1 `torch.no_grad()` — 推理时关闭追踪

```python
x = torch.tensor(2.0, requires_grad=True)

# 推理 / 参数更新时使用
with torch.no_grad():
    y = x ** 2
    print(y.requires_grad)   # False
    print(y.grad_fn)          # None

# 装饰器写法（评估函数）
@torch.no_grad()
def evaluate(model, data):
    return model(data)
```

```
使用场景：
  ┌─────────────────────────────────────────────┐
  │  推理（inference）：不需要梯度，节省内存       │
  │  参数更新：                                  │
  │    with torch.no_grad():                    │
  │        w -= lr * w.grad   ← 不追踪更新步骤  │
  │  评估（eval loop）                           │
  └─────────────────────────────────────────────┘
```

### 5.2 `.detach()` — 从计算图中分离

```python
x = torch.tensor(2.0, requires_grad=True)
y = x ** 2              # y 在计算图中

y_detached = y.detach() # y_detached 脱离计算图
print(y_detached.requires_grad)   # False
print(y_detached.grad_fn)          # None

# 常用于：把 tensor 转成 numpy，或当作常数使用
loss_value = loss.detach().item()   # 获取 loss 数值（不影响计算图）
```

| 方法 | 说明 | 使用场景 |
|------|------|---------|
| `torch.no_grad()` | 上下文管理器，块内所有操作都不追踪 | 推理、参数更新 |
| `.detach()` | 返回共享数据但脱离计算图的新张量 | 获取中间值、转 numpy |
| `.item()` | 0维张量 → Python 标量 | 打印 loss 值 |

---

## 6. 完整训练循环示例

```python
import torch
import torch.nn as nn

# 数据：拟合 y = 2x + 1
x_data = torch.linspace(0, 10, 100)
y_data = 2 * x_data + 1 + torch.randn(100) * 0.5

# 参数（叶子节点）
w = torch.tensor(0.0, requires_grad=True)
b = torch.tensor(0.0, requires_grad=True)

lr = 0.01

for epoch in range(200):
    # ── Step 1：前向传播 ──
    y_pred = w * x_data + b
    loss = ((y_pred - y_data) ** 2).mean()    # MSE

    # ── Step 2：反向传播 ──
    loss.backward()

    # ── Step 3：参数更新（no_grad 包裹） ──
    with torch.no_grad():
        w -= lr * w.grad
        b -= lr * b.grad

    # ── Step 4：清零梯度 ──
    w.grad.zero_()
    b.grad.zero_()
```

```
训练循环中的梯度流：

epoch N:

x_data ──→ [y_pred = w*x+b] ──→ [loss = MSE] ── .backward() ──→
                                                              ↓
                                                    w.grad, b.grad
                                                              ↓
                                               w -= lr * w.grad
                                               b -= lr * b.grad
                                                              ↓
                                                   w.grad.zero_()
                                                   b.grad.zero_()
                                                              ↓
                                                         epoch N+1
```

**标准顺序（每次循环必须严格遵守）：**

```
① optimizer.zero_grad()    清零梯度  ← 如果用 optimizer
② loss = model(x)          前向传播
③ loss.backward()          反向传播
④ optimizer.step()         参数更新
```

---

## 7. 使用 `nn.Module` + `optimizer` 的标准写法

```python
model = nn.Linear(1, 1)                          # y = w*x + b
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
criterion = nn.MSELoss()

for epoch in range(200):
    optimizer.zero_grad()                         # ① 清零梯度

    y_pred = model(x_data.unsqueeze(1))           # ② 前向传播
    loss = criterion(y_pred, y_data.unsqueeze(1)) # ③ 计算 loss

    loss.backward()                               # ④ 反向传播
    optimizer.step()                              # ⑤ 更新参数
```

```
对比手动梯度下降 vs optimizer：

手动：                          optimizer：
  w -= lr * w.grad               optimizer.step()
  b -= lr * b.grad               （内部自动处理所有参数）
  w.grad.zero_()
  b.grad.zero_()

  优点：透明可控                   优点：简洁，支持 Adam 等高级优化器
  缺点：参数多时繁琐               缺点：不易看出内部发生了什么
```

---

## 8. 多层计算图示例

```python
x  = torch.tensor(2.0, requires_grad=True)
w1 = torch.tensor(3.0, requires_grad=True)
w2 = torch.tensor(4.0, requires_grad=True)

h  = w1 * x         # 隐藏层：h = 6
y  = w2 * h          # 输出层：y = 24
loss = (y - 10) ** 2 # loss = (24-10)² = 196

loss.backward()
```

```
计算图：

x(2) ──→ [×w1] ──→ h(6) ──→ [×w2] ──→ y(24) ──→ [loss=(y-10)²] ──→ 196

反向传播（链式法则）：

∂loss/∂y  = 2(y-10) = 2×14 = 28
∂loss/∂h  = ∂loss/∂y × ∂y/∂h  = 28 × w2 = 28 × 4 = 112
∂loss/∂w2 = ∂loss/∂y × ∂y/∂w2 = 28 × h  = 28 × 6 = 168
∂loss/∂w1 = ∂loss/∂h × ∂h/∂w1 = 112 × x = 112 × 2 = 224
∂loss/∂x  = ∂loss/∂h × ∂h/∂x  = 112 × w1 = 112 × 3 = 336

w1.grad = 224
w2.grad = 168
x.grad  = 336
```

---

## 关键概念总结

| 概念 | 说明 |
|------|------|
| `requires_grad=True` | 标记该张量需要计算梯度，会参与计算图构建 |
| `grad_fn` | 记录该张量是由哪种操作产生的（叶子节点为 None） |
| `.backward()` | 从当前张量开始反向传播，计算所有叶子节点的梯度 |
| `.grad` | 叶子节点存储梯度的地方（非叶子节点默认不保留） |
| `.grad.zero_()` | 清零梯度，每次 backward 前必须执行 |
| `torch.no_grad()` | 关闭梯度追踪，用于推理和参数更新 |
| `.detach()` | 从计算图中分离，得到不追踪梯度的张量副本 |

---

## 常见错误速查

| 错误现象 | 原因 | 解决方案 |
|---------|------|---------|
| 梯度越来越大 | 没有在每步清零梯度 | 添加 `optimizer.zero_grad()` 或 `x.grad.zero_()` |
| `RuntimeError: element 0 of tensors does not require grad` | 对 `requires_grad=False` 的张量调用 backward | 确认叶子节点设置了 `requires_grad=True` |
| `loss.backward()` 报错非标量 | loss 不是标量 | 确保 loss 是 `.mean()` 或 `.sum()` 后的结果 |
| `.numpy()` 报错 | 张量有梯度追踪 | 改用 `.detach().numpy()` |
| 参数没有更新 | 参数更新写在 `requires_grad` 追踪范围内 | 用 `with torch.no_grad():` 包裹更新步骤 |

---

## 我的理解 / 疑问

<!-- 在这里记录学习笔记 -->
