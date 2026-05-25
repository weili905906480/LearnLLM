# C1 第1章：Tensor 张量基础

> **C1 PyTorch Fundamentals** — DeepLearning.AI  
> 本文档覆盖 Tensor 的创建、属性、类型转换、索引、形状变换、运算、广播、GPU

---

## 1. 创建张量

### 1.1 从列表创建 `torch.tensor()`

```python
import torch

t1 = torch.tensor([1, 2, 3])
t2 = torch.tensor([[1.0, 2.0, 3.0],
                   [4.0, 5.0, 6.0]])
```

| 参数 | 类型 | 说明 |
|------|------|------|
| `data` | list / ndarray | 输入数据 |
| `dtype` | torch.dtype | 可选，不填则自动推断 |
| `device` | str | 可选，`"cpu"` 或 `"cuda"` |

**矩阵形态：**

```
t1 → shape (3,)          一维向量
┌───┬───┬───┐
│ 1 │ 2 │ 3 │
└───┴───┴───┘

t2 → shape (2, 3)        二维矩阵 2行×3列
     col:  0    1    2
       ┌──────────────┐
row 0  │ 1.0  2.0  3.0│
row 1  │ 4.0  5.0  6.0│
       └──────────────┘

三维 shape (2, 2, 3) → 2个 2×3 矩阵堆叠
layer 0:            layer 1:
┌───┬───┬───┐       ┌───┬───┬───┐
│ 1 │ 2 │ 3 │       │ 7 │ 8 │ 9 │
│ 4 │ 5 │ 6 │       │10 │11 │12 │
└───┴───┴───┘       └───┴───┴───┘
```

> 💡 **dtype 自动推断**：整数列表 → `int64`，含小数点 → `float32`

---

### 1.2 初始化函数

```python
torch.zeros(3, 4)          # 全 0
torch.ones(2, 3)           # 全 1
torch.rand(2, 3)           # 均匀分布 [0, 1)
torch.randn(2, 3)          # 标准正态 N(0,1)
torch.eye(4)               # 单位矩阵
torch.arange(0, 10, 2)     # 等差序列
torch.linspace(0, 1, 5)    # 等间距序列
```

| 函数 | 参数 | 输出 shape | 说明 |
|------|------|-----------|------|
| `zeros(m, n)` | m=行数, n=列数 | `(m, n)` | 全 0 |
| `ones(m, n)` | 同上 | `(m, n)` | 全 1 |
| `rand(m, n)` | 同上 | `(m, n)` | U[0,1) |
| `randn(m, n)` | 同上 | `(m, n)` | N(0,1) |
| `eye(n)` | n=大小 | `(n, n)` | 对角线为1 |
| `arange(start, end, step)` | end 不含 | `(⌈(end-start)/step⌉,)` | 等差 |
| `linspace(start, end, steps)` | end 含 | `(steps,)` | 等间距 |

**矩阵形态示例：**

```
torch.zeros(3, 4) → shape (3, 4)
┌─────────────────┐
│ 0.  0.  0.  0.  │
│ 0.  0.  0.  0.  │
│ 0.  0.  0.  0.  │
└─────────────────┘

torch.eye(4) → shape (4, 4)
┌───────────────┐
│ 1.  0.  0.  0.│
│ 0.  1.  0.  0.│
│ 0.  0.  1.  0.│
│ 0.  0.  0.  1.│
└───────────────┘

torch.arange(0, 10, 2) → shape (5,)
┌───┬───┬───┬───┬───┐
│ 0 │ 2 │ 4 │ 6 │ 8 │
└───┴───┴───┴───┴───┘
    step=2, end(10)不含
```

---

## 2. 张量属性

```python
t = torch.rand(8, 3, 224, 224)

t.shape    # torch.Size([8, 3, 224, 224])
t.dtype    # torch.float32
t.device   # device(type='cpu')
t.ndim     # 4
t.numel()  # 8 × 3 × 224 × 224 = 1,204,224
```

| 属性 | 返回类型 | 说明 |
|------|---------|------|
| `.shape` | `torch.Size` | 各维度大小 |
| `.dtype` | `torch.dtype` | 数据类型 |
| `.device` | `torch.device` | 存储位置（cpu / cuda） |
| `.ndim` | `int` | 维度数，等于 `len(t.shape)` |
| `.numel()` | `int` | 元素总数，等于各维度之积 |

**深度学习中 4D 图像张量的含义：**

```
shape (8, 3, 224, 224)
       │  │   │    │
       │  │   │    └── W：图片宽度（像素）
       │  │   └─────── H：图片高度（像素）
       │  └─────────── C：通道数（RGB=3）
       └────────────── N：batch 中的图片数量

可视化一张图片 (3, H, W)：
  R通道: [224×224 矩阵]
  G通道: [224×224 矩阵]
  B通道: [224×224 矩阵]
```

---

## 3. 数据类型转换

```python
x = torch.tensor([1, 2, 3])   # int64

x.float()    # → float32   常用于训练
x.half()     # → float16   省显存（半精度）
x.double()   # → float64   高精度
x.bool()     # → bool
x.int()      # → int32
x.long()     # → int64

# 通用写法：同时改类型和设备
x.to(dtype=torch.float32, device="cuda")
```

| 方法 | 等价写法 | 输出 dtype | 用途 |
|------|---------|-----------|------|
| `.float()` | `.to(torch.float32)` | float32 | 训练默认 |
| `.half()` | `.to(torch.float16)` | float16 | 省显存，需 GradScaler |
| `.double()` | `.to(torch.float64)` | float64 | 高精度科学计算 |
| `.bool()` | `.to(torch.bool)` | bool | 掩码操作 |
| `.long()` | `.to(torch.int64)` | int64 | 标签、索引 |

**`.to()` 参数说明：**

| 参数 | 类型 | 说明 |
|------|------|------|
| `dtype` | torch.dtype | 目标数据类型（可选） |
| `device` | str / device | `"cpu"` / `"cuda"` / `"cuda:0"` |

---

## 4. 索引与切片

```python
t = torch.tensor([[1, 2, 3],
                  [4, 5, 6],
                  [7, 8, 9]])
```

```
矩阵 t，shape (3, 3)：

     col  0   1   2
        ┌───┬───┬───┐
row 0   │ 1 │ 2 │ 3 │
        ├───┼───┼───┤
row 1   │ 4 │ 5 │ 6 │
        ├───┼───┼───┤
row 2   │ 7 │ 8 │ 9 │
        └───┴───┴───┘
```

### 4.1 基础索引

```python
t[0]       # → tensor([1, 2, 3])    整行，shape (3,)
t[-1]      # → tensor([7, 8, 9])    最后一行
t[0, 1]    # → tensor(2)            第0行第1列，标量
t[0][1]    # → tensor(2)            等价写法
```

### 4.2 切片 `[start:end:step]`（end 不含）

```python
t[0:2]        # 第0~1行，shape (2, 3)
t[:, 1]       # 所有行的第1列，shape (3,)
t[0:2, 1:3]   # 子矩阵，shape (2, 2)
t[::2]        # 每隔1行取，shape (2, 3)
```

```
t[0:2] 取前2行：            t[:, 1] 取第1列：
┌───┬───┬───┐               ┌───┐
│ 1 │ 2 │ 3 │               │ 2 │
│ 4 │ 5 │ 6 │               │ 5 │
└───┴───┴───┘               │ 8 │
shape (2, 3)                └───┘
                            shape (3,)

t[0:2, 1:3] 子矩阵：
┌───┬───┐
│ 2 │ 3 │  ← row0, col1~2
│ 5 │ 6 │  ← row1, col1~2
└───┴───┘
shape (2, 2)
```

### 4.3 布尔索引

```python
mask = t > 4          # shape (3,3) bool 矩阵
t[mask]               # → tensor([5, 6, 7, 8, 9])  1D，所有满足条件的元素
t[t > 4]              # 等价简写
```

```
mask = t > 4：
┌───────┬───────┬───────┐
│ False │ False │ False │  row 0
├───────┼───────┼───────┤
│ False │ True  │ True  │  row 1
├───────┼───────┼───────┤
│ True  │ True  │ True  │  row 2
└───────┴───────┴───────┘
结果：展平为 1D → [5, 6, 7, 8, 9]
```

---

## 5. 形状变换

### 5.1 `reshape()`

```python
t = torch.arange(12)    # shape (12,)

t.reshape(3, 4)         # shape (3, 4)
t.reshape(2, 2, 3)      # shape (2, 2, 3)
t.reshape(3, -1)        # shape (3, 4)，-1 自动推断
t.reshape(-1)           # shape (12,) 展平
```

| 参数 | 说明 |
|------|------|
| `*shape` | 新形状，各维度大小 |
| `-1` | 该维度自动推断（只能有一个 -1） |

> ⚠️ **约束**：新旧形状的元素总数必须相等

```
原始 shape (12,)：
[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]

reshape(3, 4) → shape (3, 4)：   reshape(2,2,3) → shape (2,2,3)：
┌────────────┐                   layer 0:        layer 1:
│  0  1  2  3│                   ┌──────┐         ┌──────┐
│  4  5  6  7│                   │0  1  2│         │6  7  8│
│  8  9 10 11│                   │3  4  5│         │9 10 11│
└────────────┘                   └──────┘         └──────┘
```

---

### 5.2 `squeeze()` / `unsqueeze()`

```python
t = torch.rand(1, 3, 1, 4)   # shape (1, 3, 1, 4)

t.squeeze()        # shape (3, 4)    去掉所有 size=1 的维度
t.squeeze(0)       # shape (3, 1, 4) 只去掉第0维
t.squeeze(2)       # shape (1, 3, 4) 只去掉第2维

s = torch.rand(3, 4)          # shape (3, 4)
s.unsqueeze(0)     # shape (1, 3, 4) 在第0位插入
s.unsqueeze(1)     # shape (3, 1, 4) 在第1位插入
s.unsqueeze(-1)    # shape (3, 4, 1) 在末尾插入
```

| 方法 | 参数 `dim` | 作用 |
|------|-----------|------|
| `squeeze(dim)` | 可选；不填=去掉所有1 | 移除 size=1 的维度 |
| `unsqueeze(dim)` | 必填；支持负数 | 在指定位置插入 size=1 的维度 |

```
形态变化链：

(1, 3, 1, 4)
    ↓ squeeze()          去掉所有 size=1
(3, 4)
    ↓ unsqueeze(0)       在最前插入
(1, 3, 4)
    ↓ unsqueeze(-1)      在最后插入
(1, 3, 4, 1)

实际用途：
  向量 (512,) → unsqueeze(0) → (1, 512)   模拟 batch_size=1
  向量 (512,) → unsqueeze(-1) → (512, 1)  变为列向量
```

---

### 5.3 `permute()` / `transpose()`

```python
t = torch.rand(8, 3, 224, 224)   # NCHW

t.permute(0, 2, 3, 1)  # shape (8, 224, 224, 3)  → NHWC
t.transpose(0, 1)       # shape (3, 8, 224, 224)  → 交换前两维
```

| 方法 | 参数 | 说明 |
|------|------|------|
| `permute(*dims)` | 新的维度顺序（全部维度） | 任意重排 |
| `transpose(dim0, dim1)` | 两个维度的下标 | 只交换两个维度 |

```
permute(0, 2, 3, 1) 的含义：

原始维度:  dim0  dim1  dim2  dim3
           N(8)  C(3)  H(224) W(224)
            ↓     ↓      ↓     ↓
新位置:    [0]   [2]    [3]   [1]
            ↓     ↓      ↓     ↓
新shape:  N(8) H(224) W(224) C(3)

PyTorch NCHW  →  NumPy/matplotlib NHWC
```

> ⚠️ `permute`/`transpose` 后若要用 `.view()`，需先调用 `.contiguous()`

---

## 6. 数学运算

### 6.1 逐元素运算（Element-wise）

```python
a = torch.tensor([1., 2., 3.])
b = torch.tensor([4., 5., 6.])

a + b          # [5., 7., 9.]
a - b          # [-3., -3., -3.]
a * b          # [4., 10., 18.]   ← 不是矩阵乘法！
a / b          # [0.25, 0.4, 0.5]
a ** 2         # [1., 4., 9.]
torch.sqrt(a)  # [1., 1.414, 1.732]
torch.exp(a)   # [e, e², e³]
torch.log(a)   # [0., 0.693, 1.099]
```

```
逐元素运算示意 (a + b)：

a: ┌───┬───┬───┐    b: ┌───┬───┬───┐
   │ 1 │ 2 │ 3 │       │ 4 │ 5 │ 6 │
   └───┴───┴───┘       └───┴───┴───┘
          +
         ↓ 对应位置运算
      ┌───┬───┬───┐
      │ 5 │ 7 │ 9 │
      └───┴───┴───┘
```

---

### 6.2 矩阵乘法 `@` / `torch.matmul()`

```python
A = torch.rand(2, 3)   # shape (2, 3)
B = torch.rand(3, 4)   # shape (3, 4)

C = A @ B              # shape (2, 4)
C = torch.matmul(A, B) # 等价
C = torch.mm(A, B)     # 仅限 2D
```

> **规则：A `(m×k)` @ B `(k×n)` = C `(m×n)`**  
> A 的列数必须等于 B 的行数

```
矩阵乘法形态变化：

A (2×3)        B (3×4)           C (2×4)
┌─────────┐   ┌───────────┐     ┌───────────┐
│a00 a01 a02│  │b00 b01 b02 b03│  │c00 c01 c02 c03│
│a10 a11 a12│  │b10 b11 b12 b13│  │c10 c11 c12 c13│
└─────────┘   │b20 b21 b22 b23│  └───────────┘
              └───────────┘
  (2, 3)    @    (3, 4)      =      (2, 4)
      └──── k=3 必须相等 ────┘

batch 矩阵乘法：
  (8, 4, 3) @ (8, 3, 5) = (8, 4, 5)
   batch=8 保持不变，对每个 batch 独立做矩阵乘
```

---

### 6.3 统计运算（`dim` 参数）

```python
t = torch.tensor([[1., 2., 3.],
                  [4., 5., 6.]])

t.sum()              # tensor(21.)    所有元素
t.sum(dim=0)         # tensor([5., 7., 9.])    沿行方向（压缩行）
t.sum(dim=1)         # tensor([6., 15.])       沿列方向（压缩列）
t.mean(dim=0)        # tensor([2.5, 3.5, 4.5])
t.max(dim=0)         # values & indices
t.argmax(dim=1)      # tensor([2, 2])  每行最大值的列索引
t.sum(dim=0, keepdim=True)  # shape (1, 3)，保持维度
```

| 参数 | 类型 | 说明 |
|------|------|------|
| `dim` | int | 沿哪个维度计算；不填=对全部元素 |
| `keepdim` | bool | `True`=保持维度（结果 shape 不变） |

```
原始矩阵 t，shape (2, 3)：

核心规则：dim=N → 第N维消失，其他维度保留
  原始 (2, 3)
         ↑  ↑
       dim=0 dim=1

  sum(dim=0) → ( , 3) → shape (3,)   第0维消失
  sum(dim=1) → (2,  ) → shape (2,)   第1维消失

──────────────────────────────────────────────────────────────
dim=0：第0维（行,大小=2）消失，沿行方向↓ 压缩
       每一列的所有行叠加 → 输出 shape (3,)
──────────────────────────────────────────────────────────────

      col:  0    1    2
          ┌────┬────┬────┐
  row 0   │  1 │  2 │  3 │  ↓ 各列独立
          ├────┼────┼────┤  ↓ 叠加所有行
  row 1   │  4 │  5 │  6 │  ↓
          └────┴────┴────┘
            ↓    ↓    ↓
  sum:    [ 5,   7,   9 ]     shape (3,)   ← 3列保留，2行消失

──────────────────────────────────────────────────────────────
dim=1：第1维（列,大小=3）消失，沿列方向→ 压缩
       每一行的所有列叠加 → 输出 shape (2,)
──────────────────────────────────────────────────────────────

      col:  0    1    2
          ┌────┬────┬────┐
  row 0   │  1 │  2 │  3 │  →  1+2+3 =  6  ┐
          ├────┼────┼────┤                  │ 各行独立
  row 1   │  4 │  5 │  6 │  →  4+5+6 = 15  ┘ 叠加所有列
          └────┴────┴────┘

  sum:    [ 6,  15 ]           shape (2,)   ← 2行保留，3列消失

──────────────────────────────────────────────────────────────
keepdim=True：被压缩的维度变为1，而不是直接消失
──────────────────────────────────────────────────────────────

  sum(dim=0)                → shape (3,)     行维度(2)直接消失
  sum(dim=0, keepdim=True)  → shape (1, 3)   行维度变为1，保留位置

  sum(dim=1)                → shape (2,)     列维度(3)直接消失
  sum(dim=1, keepdim=True)  → shape (2, 1)   列维度变为1，保留位置

keepdim=True 的实际用途（广播）：
  t - t.mean(dim=1)                # (2,3) - (2,) → ❌ 无法广播
  t - t.mean(dim=1, keepdim=True)  # (2,3) - (2,1) → ✅ 广播成功
```

---

## 7. 广播机制（Broadcasting）

```python
# 标量 + 矩阵
t = torch.ones(3, 4)
t + 10             # shape (3, 4)，每个元素 +10

# 行向量 + 列向量
a = torch.tensor([[1, 2, 3]])      # shape (1, 3)
b = torch.tensor([[1], [2], [3]])  # shape (3, 1)
a + b                               # shape (3, 3)

# 深度学习常见：batch + bias
batch = torch.rand(8, 512)   # shape (8, 512)
bias  = torch.rand(512)      # shape (512,)
batch + bias                 # shape (8, 512)，bias 自动扩展到每一行

# 图像归一化
images = torch.rand(8, 3, 224, 224)
mean = torch.tensor([0.485, 0.456, 0.406]).reshape(1, 3, 1, 1)
(images - mean)              # shape (8, 3, 224, 224)
```

**广播规则（从右往左对齐）：**

```
规则：
  1. 维度数不同 → 左侧补 1
  2. 某维度为 1 → 自动扩展到对方的大小
  3. 两者都不是 1 且大小不同 → 报错

示例：(1, 3) + (3, 1) → (3, 3)

a shape (1, 3):              b shape (3, 1):
┌───┬───┬───┐                ┌───┐
│ 1 │ 2 │ 3 │                │ 1 │
└───┴───┴───┘                │ 2 │
                             │ 3 │
                             └───┘
广播后各自变成 (3, 3)：

a 扩展：                     b 扩展：
┌───┬───┬───┐               ┌───┬───┬───┐
│ 1 │ 2 │ 3 │               │ 1 │ 1 │ 1 │
│ 1 │ 2 │ 3 │               │ 2 │ 2 │ 2 │
│ 1 │ 2 │ 3 │               │ 3 │ 3 │ 3 │
└───┴───┴───┘               └───┴───┴───┘
               a + b：
               ┌───┬───┬───┐
               │ 2 │ 3 │ 4 │
               │ 3 │ 4 │ 5 │
               │ 4 │ 5 │ 6 │
               └───┴───┴───┘
```

---

## 8. 与 NumPy 互转

```python
import numpy as np

# Tensor → NumPy（共享内存）
t = torch.tensor([1., 2., 3.])
n = t.numpy()           # 共享内存：修改 t 会影响 n

# 有梯度时的安全写法
t2 = torch.tensor([1., 2.], requires_grad=True)
n2 = t2.detach().numpy()

# NumPy → Tensor
n = np.array([4., 5., 6.])
t_shared = torch.from_numpy(n)  # 共享内存
t_copy   = torch.tensor(n)      # 复制，互不影响
```

| 方式 | 内存 | 说明 |
|------|------|------|
| `tensor.numpy()` | 共享 | 仅 CPU，修改一方另一方也变 |
| `tensor.detach().numpy()` | 共享 | 有 `requires_grad` 时用这个 |
| `torch.from_numpy(arr)` | 共享 | 修改 numpy 会影响 tensor |
| `torch.tensor(arr)` | 复制 | 完全独立，推荐用于安全场景 |

---

## 9. GPU 加速

```python
device = "cuda" if torch.cuda.is_available() else "cpu"

# 方式1：创建时指定
t = torch.rand(3, 4, device=device)

# 方式2：创建后移动
t = torch.rand(3, 4)
t = t.to(device)           # 移到 GPU
t = t.to("cpu")            # 移回 CPU
t = t.cuda()               # 等价于 .to("cuda")
```

| 方法 | 参数 | 说明 |
|------|------|------|
| `.to(device)` | `"cpu"` / `"cuda"` / `"cuda:0"` | 通用写法 |
| `.cuda()` | 无 | 移到默认 GPU |
| `.cpu()` | 无 | 移回 CPU |

> ⚠️ **不同设备的 Tensor 不能直接运算**
>
> ```python
> t_cpu + t_gpu   # ❌ RuntimeError
> t_cpu.to(device) + t_gpu  # ✅ 正确
> ```

---

## 快速速查表

| 操作 | 示例 | 输入 shape | 输出 shape |
|------|------|-----------|-----------|
| 创建 | `torch.rand(2, 3)` | — | `(2, 3)` |
| reshape | `t.reshape(3, -1)` | `(12,)` | `(3, 4)` |
| 展平 | `t.flatten(1)` | `(8,64,7,7)` | `(8, 3136)` |
| 去维 | `t.squeeze(0)` | `(1, 3, 4)` | `(3, 4)` |
| 加维 | `t.unsqueeze(0)` | `(3, 4)` | `(1, 3, 4)` |
| 转置 | `t.permute(0,2,3,1)` | `(8,3,H,W)` | `(8,H,W,3)` |
| 矩阵乘 | `A @ B` | `(2,3)` `(3,4)` | `(2, 4)` |
| 按行求和 | `t.sum(dim=0)` | `(2, 3)` | `(3,)` |
| 按列求和 | `t.sum(dim=1)` | `(2, 3)` | `(2,)` |
| 保留维度 | `t.sum(dim=0, keepdim=True)` | `(2, 3)` | `(1, 3)` |
