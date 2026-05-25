# C1 第3章：nn.Module 构建神经网络

> **C1 PyTorch Fundamentals** — DeepLearning.AI  
> 本文档覆盖：nn.Module / nn.Sequential / 常用层 / 激活函数 / 模型信息 / 保存加载

---

## 核心概念

```
PyTorch 搭网络的方式：

方式1：nn.Sequential   简单线性堆叠，适合快速搭建
方式2：继承 nn.Module  灵活自定义，生产环境首选

两个必须实现的方法：
  __init__()    定义所有有参数的层
  forward()     定义数据如何流过这些层
```

---

## 1. `nn.Sequential` — 快速堆叠

```python
import torch
import torch.nn as nn

model = nn.Sequential(
    nn.Linear(784, 256),   # 输入784维 → 输出256维
    nn.ReLU(),
    nn.Linear(256, 128),
    nn.ReLU(),
    nn.Linear(128, 10)     # 输出10类
)

x = torch.rand(32, 784)   # 32个样本，每个784维
out = model(x)
```

| 参数 | 说明 |
|------|------|
| `*args` | 按顺序传入各层，数据从第一层依次流向最后一层 |

```
数据流（shape 变化）：

输入 x:          [32, 784]
  ↓ Linear(784→256)
                 [32, 256]
  ↓ ReLU
                 [32, 256]  ← shape 不变，只改值
  ↓ Linear(256→128)
                 [32, 128]
  ↓ ReLU
                 [32, 128]
  ↓ Linear(128→10)
输出 out:        [32, 10]   ← 每个样本10个类别的分数（logits）
```

---

## 2. 继承 `nn.Module` — 自定义网络

```python
class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()                          # 必须调用父类初始化
        # 在 __init__ 里定义所有有参数的层
        self.fc1     = nn.Linear(input_size, hidden_size)
        self.fc2     = nn.Linear(hidden_size, hidden_size)
        self.fc3     = nn.Linear(hidden_size, output_size)
        self.relu    = nn.ReLU()
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x):
        # 在 forward 里定义数据流动路径
        x = self.relu(self.fc1(x))   # 线性 + 激活
        x = self.dropout(x)          # 随机丢弃（训练时防过拟合）
        x = self.relu(self.fc2(x))
        x = self.fc3(x)              # 最后一层不加激活
        return x

# 实例化和使用
model = MLP(input_size=784, hidden_size=256, output_size=10)
x   = torch.rand(32, 784)
out = model(x)               # 自动调用 forward()
```

```
__init__ 的职责：                 forward 的职责：
  注册层（让 PyTorch 知道           定义数据路径
  这些层属于这个模型）              控制哪些层参与运算
  ┌──────────────────┐            ┌──────────────────┐
  │ self.fc1 = ...   │            │ x = self.fc1(x)  │
  │ self.fc2 = ...   │            │ x = relu(x)      │
  │ self.relu = ...  │            │ x = self.fc2(x)  │
  └──────────────────┘            └──────────────────┘
```

> ⚠️ 层必须在 `__init__` 里用 `self.xxx =` 赋值，PyTorch 才能追踪其参数

---

## 3. 常用层

### 3.1 `nn.Linear` — 全连接层

```python
layer = nn.Linear(in_features=256, out_features=128, bias=True)
```

| 参数 | 类型 | 默认 | 说明 |
|------|------|------|------|
| `in_features` | int | 必填 | 输入维度 |
| `out_features` | int | 必填 | 输出维度 |
| `bias` | bool | `True` | 是否添加偏置项 |

```
运算：y = x @ W.T + b

输入 x:  [32, 256]
权重 W:  [128, 256]   （out × in）
输出 y:  [32, 128]

参数量 = 256 × 128 + 128 = 32,896
               ↑        ↑
           W矩阵      bias
```

---

### 3.2 `nn.Dropout` — 随机丢弃

```python
dropout = nn.Dropout(p=0.5)   # 训练时随机将 50% 的神经元置 0
```

| 参数 | 类型 | 默认 | 说明 |
|------|------|------|------|
| `p` | float | `0.5` | 置零的概率，范围 [0, 1) |

```
训练模式（model.train()）：     评估模式（model.eval()）：
  随机将 p 比例的元素置 0          全部保留，不丢弃
  剩余元素除以 (1-p) 补偿幅度

  输入:  [1, 2, 3, 4, 5, 6]      输入: [1, 2, 3, 4, 5, 6]
  p=0.5 随机置0 + 缩放:            输出: [1, 2, 3, 4, 5, 6]
  输出:  [0, 4, 0, 8, 10, 0]
```

> 💡 **作用**：防止过拟合。相当于每次训练一个不同的子网络，增强泛化能力

---

### 3.3 `nn.BatchNorm1d` / `nn.BatchNorm2d` — 批归一化

```python
bn1 = nn.BatchNorm1d(num_features=256)    # 用于全连接层后（1D特征）
bn2 = nn.BatchNorm2d(num_features=64)     # 用于卷积层后（2D特征图）
```

| 参数 | 类型 | 说明 |
|------|------|------|
| `num_features` | int | 特征维度数（全连接）或通道数（卷积） |

```
作用：对每个 batch 的特征进行归一化，加速训练，缓解梯度问题

输入  [32, 256]
  ↓ 对256个特征各自做归一化（均值=0，方差=1）
输出  [32, 256]  ← shape 不变，值被归一化
```

---

### 3.4 `nn.Flatten` — 展平层

```python
flatten = nn.Flatten(start_dim=1)   # 从第1维开始展平（保留batch维）
```

| 参数 | 默认 | 说明 |
|------|------|------|
| `start_dim` | `1` | 从哪一维开始展平 |
| `end_dim` | `-1` | 展平到哪一维结束 |

```
CNN 输出接全连接层时必须用：

卷积输出:  [32, 64, 7, 7]
  ↓ Flatten(start_dim=1)
展平后:    [32, 64×7×7] = [32, 3136]
              ↑
         batch维度保留
```

---

## 4. 激活函数

```python
nn.ReLU()       # max(0, x)
nn.Sigmoid()    # 1 / (1 + e^-x)     → 输出 (0, 1)
nn.Tanh()       # (e^x - e^-x) / (e^x + e^-x)  → 输出 (-1, 1)
nn.GELU()       # x · Φ(x)，比 ReLU 更平滑，Transformer 常用
nn.Softmax(dim=1)  # 各类概率之和=1，多分类输出层用
nn.LeakyReLU(negative_slope=0.01)   # 负区间有小梯度，解决死神经元
```

```
各激活函数曲线对比：

     ReLU          Sigmoid          GELU
      │  /           │    ___        │   /
      │ /            │   /           │  /
──────│/─────   ─────│──/──────  ────│─/──────
      │              │ /             │/
                     │/          __/
                                   ↑负区间有小梯度

使用建议：
  隐藏层默认用   ReLU（简单高效）
  需要平滑时用   GELU（Transformer、BERT）
  二分类输出用   Sigmoid
  多分类输出不在网络内加 Softmax（CrossEntropyLoss 内部已包含）
```

---

## 5. `train()` / `eval()` 模式

```python
model.train()   # 训练模式：Dropout 随机丢弃，BatchNorm 用当前 batch 统计
model.eval()    # 评估模式：Dropout 关闭，BatchNorm 用全局统计量
```

```
影响的层：
  ┌──────────────┬──────────────┬──────────────┐
  │ 层           │ train()      │ eval()       │
  ├──────────────┼──────────────┼──────────────┤
  │ Dropout      │ 随机置零     │ 全部保留      │
  │ BatchNorm    │ 用当前batch  │ 用训练期统计  │
  │ Linear/Conv  │ 无变化       │ 无变化        │
  └──────────────┴──────────────┴──────────────┘

标准使用模式：
  训练：model.train() → loss.backward() → optimizer.step()
  验证：model.eval()  → with torch.no_grad(): → 计算指标
```

---

## 6. 查看模型信息

### 6.1 打印模型结构

```python
model = MLP(784, 256, 10)
print(model)
```

```
输出：
MLP(
  (fc1): Linear(in_features=784, out_features=256, bias=True)
  (fc2): Linear(in_features=256, out_features=256, bias=True)
  (fc3): Linear(in_features=256, out_features=10, bias=True)
  (relu): ReLU()
  (dropout): Dropout(p=0.5, inplace=False)
)
```

### 6.2 遍历参数

```python
for name, param in model.named_parameters():
    print(f"{name:20s}  shape={tuple(param.shape)}")
```

```
输出：
fc1.weight           shape=(256, 784)
fc1.bias             shape=(256,)
fc2.weight           shape=(256, 256)
fc2.bias             shape=(256,)
fc3.weight           shape=(10, 256)
fc3.bias             shape=(10,)
```

### 6.3 统计参数量

```python
total     = sum(p.numel() for p in model.parameters())
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f"总参数量:     {total:,}")
print(f"可训练参数量: {trainable:,}")
```

```
各层参数量：
  fc1: 784×256 + 256 = 200,960
  fc2: 256×256 + 256 =  65,792
  fc3: 256×10  +  10 =   2,570
  总计:                 269,322
```

---

## 7. 保存与加载模型

```python
# ✅ 推荐：只保存参数（state_dict）
torch.save(model.state_dict(), 'model.pth')

# 加载参数
model_new = MLP(784, 256, 10)             # 必须先创建相同结构的模型
model_new.load_state_dict(torch.load('model.pth'))
model_new.eval()

# ❌ 不推荐：保存整个模型（依赖类定义，迁移性差）
torch.save(model, 'model_full.pth')
loaded = torch.load('model_full.pth')
```

| 方法 | 文件大小 | 迁移性 | 推荐度 |
|------|---------|--------|--------|
| `state_dict()` | 小（只有参数） | 高 | ✅ 推荐 |
| 整个模型 | 大（含结构） | 低（依赖类） | ❌ 不推荐 |

```
state_dict 的内容：
{
  'fc1.weight': tensor([...]),   shape (256, 784)
  'fc1.bias':   tensor([...]),   shape (256,)
  'fc2.weight': tensor([...]),   shape (256, 256)
  ...
}
```

---

## 8. 完整示例：MNIST 分类网络

```python
class MNISTNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()              # (B, 1, 28, 28) → (B, 784)
        self.fc1     = nn.Linear(784, 256)
        self.bn1     = nn.BatchNorm1d(256)
        self.fc2     = nn.Linear(256, 128)
        self.fc3     = nn.Linear(128, 10)
        self.relu    = nn.ReLU()
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        x = self.flatten(x)                     # [B, 784]
        x = self.relu(self.bn1(self.fc1(x)))    # [B, 256]
        x = self.dropout(x)
        x = self.relu(self.fc2(x))              # [B, 128]
        x = self.fc3(x)                         # [B, 10]  ← logits
        return x
```

```
完整数据流（batch_size=32）：

输入图像:    [32, 1, 28, 28]
  ↓ Flatten
             [32, 784]
  ↓ Linear(784→256) + BN + ReLU
             [32, 256]
  ↓ Dropout(0.3)
             [32, 256]
  ↓ Linear(256→128) + ReLU
             [32, 128]
  ↓ Linear(128→10)
输出 logits: [32, 10]   ← 10个类别的原始分数，不过 Softmax
```

> 💡 **为什么最后不加 Softmax？**  
> `nn.CrossEntropyLoss` 内部已经做了 LogSoftmax，如果再加 Softmax 会计算两次，导致结果错误

---

## 快速速查表

| 层 | 参数 | 输入 shape | 输出 shape |
|----|------|-----------|-----------|
| `Linear(in, out)` | in=输入维, out=输出维 | `(B, in)` | `(B, out)` |
| `ReLU()` | 无 | `(B, N)` | `(B, N)` |
| `Dropout(p)` | p=丢弃率 | `(B, N)` | `(B, N)` |
| `BatchNorm1d(n)` | n=特征数 | `(B, N)` | `(B, N)` |
| `Flatten(start)` | start=起始维 | `(B, C, H, W)` | `(B, C×H×W)` |
| `Sigmoid()` | 无 | `(B, N)` | `(B, N)` 值域(0,1) |
| `Softmax(dim)` | dim=归一化维 | `(B, N)` | `(B, N)` 和=1 |

---

## 我的理解 / 疑问

<!-- 在这里记录学习笔记 -->
