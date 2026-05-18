"""
C1 Chapter 1: Tensor 张量基础
==============================
PyTorch for Deep Learning - DeepLearning.AI

本文件涵盖：
  1. 张量创建（多种方式）
  2. 张量属性（shape / dtype / device）
  3. 数据类型转换
  4. 索引与切片
  5. 形状变换（reshape / squeeze / permute）
  6. 数学运算
  7. 广播机制
  8. 与 NumPy 互转
  9. GPU 加速

每个操作都附有：
  - 参数说明
  - 输入/输出矩阵形态图示
  - 详细注释
"""

import torch
import numpy as np

# 打印分隔线辅助函数
def section(title):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")

def show_tensor(name, t):
    """可视化张量的形态、类型、值"""
    print(f"\n  [{name}]")
    print(f"    shape  : {tuple(t.shape)}")
    print(f"    dtype  : {t.dtype}")
    print(f"    device : {t.device}")
    print(f"    ndim   : {t.ndim}  (维度数)")
    print(f"    numel  : {t.numel()}  (元素总数)")
    if t.numel() <= 24:
        print(f"    values :\n{t}")



# ============================================================
# 第1节：创建张量
# ============================================================
section("1. 创建张量")

# ── 1.1 从 Python 列表创建 ──────────────────────────────────
print("\n  ▶ 1.1 从列表创建")
print("""
  形态示意：
    一维 (1D): [1, 2, 3]
    ┌───┬───┬───┐
    │ 1 │ 2 │ 3 │    shape = (3,)
    └───┴───┴───┘

    二维 (2D): [[1,2,3],[4,5,6]]
    ┌───┬───┬───┐
    │ 1 │ 2 │ 3 │    shape = (2, 3)
    ├───┼───┼───┤    2行 × 3列
    │ 4 │ 5 │ 6 │
    └───┴───┴───┘

    三维 (3D): 2个 2×3 矩阵堆叠
    shape = (2, 2, 3)
    层0:              层1:
    ┌───┬───┬───┐     ┌───┬───┬───┐
    │ 1 │ 2 │ 3 │     │ 7 │ 8 │ 9 │
    ├───┼───┼───┤     ├───┼───┼───┤
    │ 4 │ 5 │ 6 │     │10 │11 │12 │
    └───┴───┴───┘     └───┴───┴───┘
""")

t1d = torch.tensor([1, 2, 3])
# 参数：data (Python list / NumPy array)
# dtype 自动推断：整数 → int64，浮点 → float32

t2d = torch.tensor([[1.0, 2.0, 3.0],
                     [4.0, 5.0, 6.0]])
# 二维，float 后缀让 dtype 变为 float32

t3d = torch.tensor([[[1, 2, 3], [4, 5, 6]],
                     [[7, 8, 9], [10,11,12]]])

show_tensor("t1d (1D)", t1d)
show_tensor("t2d (2D)", t2d)
show_tensor("t3d (3D)", t3d)

# ── 1.2 常用初始化函数 ──────────────────────────────────────
print("\n  ▶ 1.2 常用初始化函数")
print("""
  torch.zeros(rows, cols)   全0矩阵
  torch.ones(rows, cols)    全1矩阵
  torch.rand(rows, cols)    均匀分布 [0, 1)
  torch.randn(rows, cols)   标准正态分布 N(0,1)
  torch.eye(n)              单位矩阵（对角线为1）
  torch.arange(start, end, step)  等差序列
  torch.linspace(start, end, steps) 等间距序列
""")

zeros = torch.zeros(3, 4)
# 参数：
#   size: int 或 tuple，如 (3,4) 或 3, 4
# 输出形态：
#   ┌─────────────────┐
#   │ 0  0  0  0      │
#   │ 0  0  0  0      │  shape=(3,4)
#   │ 0  0  0  0      │
#   └─────────────────┘
show_tensor("zeros(3,4)", zeros)

ones = torch.ones(2, 3)
show_tensor("ones(2,3)", ones)

rand = torch.rand(2, 3)
# 每个元素从均匀分布 U[0,1) 采样
show_tensor("rand(2,3)", rand)

randn = torch.randn(2, 3)
# 每个元素从标准正态 N(0,1) 采样
show_tensor("randn(2,3)", randn)

eye = torch.eye(4)
# 参数 n：方阵大小
# 输出：
#   ┌───┬───┬───┬───┐
#   │ 1 │ 0 │ 0 │ 0 │
#   ├───┼───┼───┼───┤
#   │ 0 │ 1 │ 0 │ 0 │  shape=(4,4)
#   ├───┼───┼───┼───┤
#   │ 0 │ 0 │ 1 │ 0 │
#   ├───┼───┼───┼───┤
#   │ 0 │ 0 │ 0 │ 1 │
#   └───┴───┴───┴───┘
show_tensor("eye(4)", eye)

arange = torch.arange(0, 10, 2)
# 参数：start=0, end=10(不含), step=2
# 输出: [0, 2, 4, 6, 8]   shape=(5,)
show_tensor("arange(0,10,2)", arange)

linspace = torch.linspace(0, 1, 6)
# 参数：start=0, end=1(含), steps=6
# 输出: [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]   shape=(6,)
show_tensor("linspace(0,1,6)", linspace)

# ── 1.3 指定 dtype 创建 ─────────────────────────────────────
print("\n  ▶ 1.3 指定 dtype 创建")
t_int   = torch.zeros(2, 3, dtype=torch.int32)
t_float = torch.zeros(2, 3, dtype=torch.float32)
t_half  = torch.zeros(2, 3, dtype=torch.float16)
t_bool  = torch.zeros(2, 3, dtype=torch.bool)

print(f"  int32  : {t_int.dtype}")
print(f"  float32: {t_float.dtype}  ← 默认浮点类型")
print(f"  float16: {t_half.dtype}   ← 半精度，省显存")
print(f"  bool   : {t_bool.dtype}")



# ============================================================
# 第2节：张量属性
# ============================================================
section("2. 张量属性")

t = torch.randn(2, 3, 4)
print(f"""
  张量 t = torch.randn(2, 3, 4)

  形态示意（三维张量）：
  ┌─────────────────────────────────────┐
  │  维度0 (batch)  = 2                  │
  │  维度1 (行)     = 3                  │
  │  维度2 (列)     = 4                  │
  │                                      │
  │  可理解为：2 个 3×4 的矩阵            │
  │                                      │
  │  层 0:          层 1:                 │
  │  [*][*][*][*]   [*][*][*][*]         │
  │  [*][*][*][*]   [*][*][*][*]         │
  │  [*][*][*][*]   [*][*][*][*]         │
  └─────────────────────────────────────┘

  t.shape   = {t.shape}   ← 各维度大小
  t.dtype   = {t.dtype}      ← 数据类型
  t.device  = {t.device}           ← 存储位置
  t.ndim    = {t.ndim}             ← 维度数（=len(t.shape)）
  t.numel() = {t.numel()}            ← 元素总数（=2×3×4）
""")

# 深度学习常见的 4D 张量（图像 batch）
t4d = torch.rand(8, 3, 224, 224)
print(f"""
  图像数据常用 4D 张量：torch.rand(8, 3, 224, 224)

  形态含义：
  ┌──────────────────────────────────────────┐
  │  维度0 (N)  = 8     ← batch 中图片数量    │
  │  维度1 (C)  = 3     ← 通道数 (RGB=3)     │
  │  维度2 (H)  = 224   ← 图片高度（像素）    │
  │  维度3 (W)  = 224   ← 图片宽度（像素）    │
  └──────────────────────────────────────────┘

  t4d.shape   = {t4d.shape}
  t4d.numel() = {t4d.numel():,}   ← 8×3×224×224
""")



# ============================================================
# 第3节：数据类型转换
# ============================================================
section("3. 数据类型转换")

print("""
  PyTorch 常见数据类型：
  ┌──────────────┬──────────────┬────────────────────────────┐
  │ 类型名        │ 字节数       │ 说明                        │
  ├──────────────┼──────────────┼────────────────────────────┤
  │ torch.float32│ 4 bytes      │ 默认浮点，训练常用           │
  │ torch.float16│ 2 bytes      │ 半精度，省显存，需 GradScaler │
  │ torch.bfloat16│ 2 bytes     │ 脑浮点，LLM 训练常用         │
  │ torch.float64│ 8 bytes      │ 双精度，精度最高，速度慢     │
  │ torch.int32  │ 4 bytes      │ 整数                        │
  │ torch.int64  │ 8 bytes      │ 整数，torch.long             │
  │ torch.bool   │ 1 byte       │ 布尔，True/False             │
  └──────────────┴──────────────┴────────────────────────────┘
""")

x = torch.tensor([1, 2, 3])                # 默认 int64

x_f32  = x.float()                         # → float32  (等价于 x.to(torch.float32))
x_f16  = x.half()                          # → float16  (等价于 x.to(torch.float16))
x_f64  = x.double()                        # → float64
x_bool = x.bool()                          # → bool
x_int  = x_f32.int()                       # → int32
x_long = x_f32.long()                      # → int64

print(f"  原始 x:      {x.dtype}   值={x.tolist()}")
print(f"  .float()   : {x_f32.dtype}   值={x_f32.tolist()}")
print(f"  .half()    : {x_f16.dtype}   值={x_f16.tolist()}")
print(f"  .double()  : {x_f64.dtype}  值={x_f64.tolist()}")
print(f"  .bool()    : {x_bool.dtype}     值={x_bool.tolist()}")
print(f"  .int()     : {x_int.dtype}    值={x_int.tolist()}")

# 使用 .to() 方法（更通用）
device = "cuda" if torch.cuda.is_available() else "cpu"
x_cuda = x.to(device=device, dtype=torch.float32)
# 参数：
#   device : 目标设备，如 "cpu" / "cuda" / "cuda:0"
#   dtype  : 目标数据类型
print(f"\n  .to(device={device!r}, dtype=float32): {x_cuda.dtype} on {x_cuda.device}")



# ============================================================
# 第4节：索引与切片
# ============================================================
section("4. 索引与切片")

t = torch.tensor([[1, 2, 3],
                   [4, 5, 6],
                   [7, 8, 9]])

print(f"""
  矩阵 t（shape={tuple(t.shape)}）：

  列索引:  0   1   2
         ┌───┬───┬───┐
  行 0   │ 1 │ 2 │ 3 │
         ├───┼───┼───┤
  行 1   │ 4 │ 5 │ 6 │
         ├───┼───┼───┤
  行 2   │ 7 │ 8 │ 9 │
         └───┴───┴───┘
""")

# 单元素索引
print("  ── 单元素索引 ──")
print(f"  t[0]      = {t[0]}      ← 第0行（返回1D tensor）")
print(f"  t[1]      = {t[1]}      ← 第1行")
print(f"  t[-1]     = {t[-1]}      ← 最后一行（负索引）")
print(f"  t[0, 1]   = {t[0, 1]}          ← 第0行第1列 → 标量")
print(f"  t[0][1]   = {t[0][1]}          ← 等价写法")

# 切片
print("\n  ── 切片 [start:end:step]（end不含）──")
print(f"  t[0:2]       shape={t[0:2].shape} ← 第0~1行\n{t[0:2]}")
print(f"  t[:, 1]      shape={t[:,1].shape}  ← 第1列（所有行）\n{t[:,1]}")
print(f"  t[0:2, 1:3]  shape={t[0:2,1:3].shape} ← 子矩阵\n{t[0:2,1:3]}")
print(f"  t[::2]       shape={t[::2].shape} ← 每隔1行取一行\n{t[::2]}")

# 布尔索引
print("\n  ── 布尔索引 ──")
mask = t > 4
print(f"  mask (t>4):\n{mask}")
print(f"  t[t>4] = {t[t>4]}   ← 返回满足条件的所有元素（1D）")

# 高级索引（Fancy Indexing）
print("\n  ── 高级索引（整数数组索引）──")
idx = torch.tensor([0, 2])             # 取第0行和第2行
print(f"  t[[0,2]]  shape={t[idx].shape}:\n{t[idx]}")

rows = torch.tensor([0, 1, 2])
cols = torch.tensor([0, 1, 2])
print(f"  对角线元素 t[[0,1,2],[0,1,2]] = {t[rows, cols]}")

# 3D 张量的索引
print("\n  ── 3D 张量索引 ──")
t3 = torch.arange(24).reshape(2, 3, 4)
print(f"""
  t3 = torch.arange(24).reshape(2, 3, 4)

  形态：(batch=2, rows=3, cols=4)
  t3[0]:          ← 第0个 batch
{t3[0]}
  t3[1]:          ← 第1个 batch
{t3[1]}
  t3[0, 1, 2] = {t3[0, 1, 2]}   ← batch0, 第1行, 第2列
  t3[:, 0, :]  shape={t3[:,0,:].shape} ← 所有batch，第0行，所有列
{t3[:,0,:]}
""")



# ============================================================
# 第5节：形状变换
# ============================================================
section("5. 形状变换")

# ── 5.1 reshape ─────────────────────────────────────────────
print("\n  ▶ 5.1 reshape — 改变形状，不改变数据")
print("""
  原则：新形状的元素总数必须等于原形状的元素总数

  例：shape (2, 6) → reshape 后的多种形状：
  (2,6)→(3,4)→(4,3)→(12,)→(1,12)→(2,2,3) 均合法
  因为都是 12 个元素
""")

t = torch.arange(12).float()
print(f"  原始 t: shape={tuple(t.shape)}\n  {t}")

t_2x6 = t.reshape(2, 6)
print(f"""
  reshape(2, 6): shape=(2,6)
  ┌─────────────────────┐
  │  0  1  2  3  4  5   │
  │  6  7  8  9 10 11   │
  └─────────────────────┘
{t_2x6}""")

t_3x4 = t.reshape(3, 4)
print(f"""
  reshape(3, 4): shape=(3,4)
  ┌─────────────┐
  │  0  1  2  3 │
  │  4  5  6  7 │
  │  8  9 10 11 │
  └─────────────┘
{t_3x4}""")

t_auto = t.reshape(3, -1)
# 参数：-1 表示该维度自动推断
# 3 × ? = 12 → ? = 4
print(f"  reshape(3, -1): -1 自动推断为 {t_auto.shape[1]}，shape={tuple(t_auto.shape)}")

t_3d = t.reshape(2, 2, 3)
print(f"""
  reshape(2, 2, 3): shape=(2,2,3)
  层0:        层1:
  ┌─────┐     ┌─────┐
  │0 1 2│     │6 7 8│
  │3 4 5│     │9 10 11│
  └─────┘     └─────┘
{t_3d}""")

# ── 5.2 view（与 reshape 类似，要求内存连续）─────────────────
print("\n  ▶ 5.2 view — 与 reshape 类似，要求内存连续")
t_c = t_3x4.contiguous()       # 先确保内存连续
t_view = t_c.view(6, 2)
print(f"  view(6,2): shape={tuple(t_view.shape)}\n{t_view}")
print("  注意：view 失败时改用 reshape 即可")

# ── 5.3 flatten ──────────────────────────────────────────────
print("\n  ▶ 5.3 flatten — 展平为一维")
print("""
  常用于 CNN 输出接 Linear 层之前：

  (batch=8, C=64, H=7, W=7)
           ↓ flatten(start_dim=1)
  (batch=8, 64*7*7=3136)
""")

t = torch.rand(8, 64, 7, 7)
t_flat = t.flatten(start_dim=1)
# 参数：
#   start_dim : 从哪个维度开始展平（默认0=全部展平）
#   end_dim   : 展平到哪个维度（默认-1=最后）
print(f"  flatten(start_dim=1): {tuple(t.shape)} → {tuple(t_flat.shape)}")

# ── 5.4 squeeze / unsqueeze ──────────────────────────────────
print("\n  ▶ 5.4 squeeze / unsqueeze — 删除/增加大小为1的维度")
print("""
  squeeze:   去掉所有大小为1的维度
  unsqueeze: 在指定位置插入大小为1的新维度

  形态变化示意：
  (1, 3, 1, 4)
      ↓ squeeze()         去掉所有1
  (3, 4)
      ↓ unsqueeze(0)      在第0维插入
  (1, 3, 4)
      ↓ unsqueeze(-1)     在最后插入
  (1, 3, 4, 1)
""")

t = torch.rand(1, 3, 1, 4)
print(f"  原始: shape={tuple(t.shape)}")

t_sq = t.squeeze()
# 参数：dim（可选），指定只去掉特定维度
print(f"  .squeeze()           → {tuple(t_sq.shape)}  去掉所有 size=1 的维度")

t_sq_dim = t.squeeze(0)
print(f"  .squeeze(0)          → {tuple(t_sq_dim.shape)}  只去掉第0维")

t_usq = t_sq.unsqueeze(0)
# 参数：dim，在该位置插入新维度
print(f"  .unsqueeze(0)        → {tuple(t_usq.shape)}  在第0位插入")

t_usq2 = t_sq.unsqueeze(1)
print(f"  .unsqueeze(1)        → {tuple(t_usq2.shape)}  在第1位插入")

# 实际用途：给 batch 维度
x = torch.tensor([1.0, 2.0, 3.0])  # shape (3,)
x_batch = x.unsqueeze(0)            # shape (1, 3) ← 模拟 batch_size=1
x_feat  = x.unsqueeze(-1)           # shape (3, 1) ← 特征列向量
print(f"\n  实际用途:")
print(f"  向量 (3,) → unsqueeze(0) → {tuple(x_batch.shape)}  (batch=1,feat=3)")
print(f"  向量 (3,) → unsqueeze(-1)→ {tuple(x_feat.shape)}  (feat=3,1) 列向量")

# ── 5.5 permute / transpose ──────────────────────────────────
print("\n  ▶ 5.5 permute / transpose — 维度换位")
print("""
  permute(dims): 按 dims 指定的顺序重排所有维度
  transpose(dim0, dim1): 只交换两个维度

  图像格式转换示意：
  PyTorch: (N, C, H, W) = (8, 3, 224, 224)  ← CHW 格式
                ↓ permute(0, 2, 3, 1)
  NumPy:   (N, H, W, C) = (8, 224, 224, 3)   ← HWC 格式（matplotlib 用）
""")

t = torch.rand(8, 3, 224, 224)    # PyTorch 图像格式 NCHW
print(f"  原始 NCHW: {tuple(t.shape)}")

t_nhwc = t.permute(0, 2, 3, 1)
# 参数：dims (tuple)，指定新的维度顺序
# (0→0, 2→1, 3→2, 1→3) 即把第1维移到最后
print(f"  permute(0,2,3,1) NHWC: {tuple(t_nhwc.shape)}")

# transpose：只交换两个维度
t2 = torch.rand(3, 4)
t_T = t2.transpose(0, 1)
# 参数：dim0, dim1  要交换的两个维度
print(f"\n  矩阵转置 transpose(0,1): {tuple(t2.shape)} → {tuple(t_T.shape)}")
print(f"  也可写成 t2.T: {tuple(t2.T.shape)}")

# 注意：permute/transpose 返回视图，需要 .contiguous() 才能用 view
t_cont = t_nhwc.contiguous()
print(f"\n  注意：permute 后若要用 view，需先 .contiguous()")



# ============================================================
# 第6节：数学运算
# ============================================================
section("6. 数学运算")

# ── 6.1 逐元素运算 ───────────────────────────────────────────
print("\n  ▶ 6.1 逐元素运算（element-wise）")
print("""
  两个相同形状的张量，对应位置逐个运算：

  a = [1, 2, 3]     b = [4, 5, 6]
  ┌───┬───┬───┐     ┌───┬───┬───┐
  │ 1 │ 2 │ 3 │  +  │ 4 │ 5 │ 6 │
  └───┴───┴───┘     └───┴───┴───┘
           ↓
  ┌───┬───┬───┐
  │ 5 │ 7 │ 9 │   对应位置相加
  └───┴───┴───┘
""")

a = torch.tensor([1.0, 2.0, 3.0])
b = torch.tensor([4.0, 5.0, 6.0])

print(f"  a = {a.tolist()}")
print(f"  b = {b.tolist()}")
print(f"  a + b      = {(a+b).tolist()}")
print(f"  a - b      = {(a-b).tolist()}")
print(f"  a * b      = {(a*b).tolist()}   ← 逐元素乘（非矩阵乘！）")
print(f"  a / b      = {(a/b).tolist()}")
print(f"  a ** 2     = {(a**2).tolist()}   ← 逐元素幂次")
print(f"  torch.sqrt(a) = {torch.sqrt(a).tolist()}")
print(f"  torch.exp(a)  = {torch.exp(a).tolist()}")
print(f"  torch.log(a)  = {torch.log(a).tolist()}")
print(f"  torch.abs(torch.tensor([-1.,-2.,3.])) = {torch.abs(torch.tensor([-1.,-2.,3.])).tolist()}")

# 原地操作（in-place）：加下划线后缀
print(f"\n  原地操作（改变自身，节省内存）：")
c = torch.tensor([1.0, 2.0, 3.0])
c.add_(10)       # 等价 c = c + 10，但不创建新张量
print(f"  c.add_(10)  → {c.tolist()}")
c.mul_(2)
print(f"  c.mul_(2)   → {c.tolist()}")

# ── 6.2 矩阵乘法 ─────────────────────────────────────────────
print("\n  ▶ 6.2 矩阵乘法（Matrix Multiplication）")
print("""
  规则：A(m×k) @ B(k×n) = C(m×n)
  A 的列数必须等于 B 的行数！

  例：(2×3) @ (3×4) = (2×4)

  A(2×3):         B(3×4):
  ┌───────────┐   ┌────────────────┐
  │ a b c     │   │ g h i j       │
  │ d e f     │   │ k l m n       │
  └───────────┘   │ o p q r       │
                  └────────────────┘
        ↓
  C(2×4):
  ┌────────────────────┐
  │ ag+bk+co  ...      │
  │ dg+ek+fo  ...      │
  └────────────────────┘
""")

A = torch.rand(2, 3)
B = torch.rand(3, 4)
C = A @ B                    # 最推荐的写法
C2 = torch.matmul(A, B)      # 等价
C3 = torch.mm(A, B)          # 仅支持 2D 矩阵

print(f"  A: shape={tuple(A.shape)}")
print(f"  B: shape={tuple(B.shape)}")
print(f"  A @ B = C: shape={tuple(C.shape)}")

# batch 矩阵乘法（3D）
print("""
  批量矩阵乘法（Batch Matrix Multiply）：
  (B, m, k) @ (B, k, n) = (B, m, n)
  对每个 batch 独立做矩阵乘法
""")
A_batch = torch.rand(8, 4, 3)   # 8个 4×3 矩阵
B_batch = torch.rand(8, 3, 5)   # 8个 3×5 矩阵
C_batch = A_batch @ B_batch     # 8个 4×5 矩阵
print(f"  ({tuple(A_batch.shape)}) @ ({tuple(B_batch.shape)}) = {tuple(C_batch.shape)}")

# ── 6.3 统计运算 ─────────────────────────────────────────────
print("\n  ▶ 6.3 统计运算")
print("""
  关键参数 dim（沿哪个维度计算）：

  t = [[1, 2, 3],
       [4, 5, 6]]    shape=(2,3)

  dim=0（沿行方向，压缩行维度）：   dim=1（沿列方向，压缩列维度）：
  ↓↓↓                              → → →
  [1,2,3]                           [1,2,3] → sum=6
  [4,5,6]                           [4,5,6] → sum=15
  = [5,7,9]  shape=(3,)             = [6, 15]  shape=(2,)
""")

t = torch.tensor([[1.0, 2.0, 3.0],
                   [4.0, 5.0, 6.0]])

print(f"  t:\n{t}\n")
print(f"  t.sum()        = {t.sum()}         ← 所有元素求和")
print(f"  t.sum(dim=0)   = {t.sum(dim=0)}   ← 沿行方向（每列求和）shape={tuple(t.sum(dim=0).shape)}")
print(f"  t.sum(dim=1)   = {t.sum(dim=1)}       ← 沿列方向（每行求和）shape={tuple(t.sum(dim=1).shape)}")
print(f"  t.mean()       = {t.mean()}")
print(f"  t.mean(dim=0)  = {t.mean(dim=0)}")
print(f"  t.max()        = {t.max()}")
print(f"  t.max(dim=0)   values={t.max(dim=0).values}, indices={t.max(dim=0).indices}")
# max(dim) 返回 (values, indices) 两个张量
print(f"  t.argmax()     = {t.argmax()}         ← 最大值的全局索引（展平后）")
print(f"  t.argmax(dim=1)= {t.argmax(dim=1)}       ← 每行最大值的列索引")

# keepdim=True：保持维度不变
print(f"\n  keepdim=True（保持维度）：")
print(f"  t.sum(dim=0)            shape={tuple(t.sum(dim=0).shape)}")
print(f"  t.sum(dim=0, keepdim=True) shape={tuple(t.sum(dim=0, keepdim=True).shape)} ← 保留维度")



# ============================================================
# 第7节：广播机制（Broadcasting）
# ============================================================
section("7. 广播机制（Broadcasting）")

print("""
  广播规则（从右往左对齐维度）：
  1. 维度数不同 → 左边补 1
  2. 某个维度大小为 1 → 自动扩展复制到对方的大小
  3. 维度大小既不相同又都不是 1 → 报错

  ──────────────────────────────────────────────────
  示例1：(3,1) + (3,)
    (3,1)     维度对齐后 (3,1)
    (3,)   →            (1,3)   → 广播为 (3,3)
  ──────────────────────────────────────────────────
  示例2：(1,3) + (3,1)
    (1,3)
    (3,1)   → 广播为 (3,3)

    a = [1, 2, 3]  shape(1,3)     b = [[1],  shape(3,1)
                                        [2],
                                        [3]]
    ┌───┬───┬───┐     ┌───┬───┬───┐
    │ 1 │ 2 │ 3 │  +  │ 1 │ 1 │ 1 │ ← b 列扩展
    │ 1 │ 2 │ 3 │     │ 2 │ 2 │ 2 │ ← a 行扩展
    │ 1 │ 2 │ 3 │     │ 3 │ 3 │ 3 │
    └───┴───┴───┘     └───┴───┴───┘
             ↓
    ┌───┬───┬───┐
    │ 2 │ 3 │ 4 │
    │ 3 │ 4 │ 5 │  shape=(3,3)
    │ 4 │ 5 │ 6 │
    └───┴───┴───┘
  ──────────────────────────────────────────────────
""")

# 示例1：标量 + 矩阵
t = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
result = t + 10
print(f"  矩阵 + 标量：shape{tuple(t.shape)} + scalar")
print(f"  {t.tolist()} + 10 = {result.tolist()}")

# 示例2：行向量 + 列向量
a = torch.tensor([[1, 2, 3]])          # shape (1, 3)
b = torch.tensor([[1], [2], [3]])      # shape (3, 1)
result = a + b
print(f"\n  行向量 + 列向量：{tuple(a.shape)} + {tuple(b.shape)} → {tuple(result.shape)}")
print(f"\n  a:\n{a}")
print(f"  b:\n{b}")
print(f"  a + b:\n{result}")

# 示例3：深度学习常见场景——对 batch 加 bias
batch = torch.rand(8, 512)             # 8个样本，每个512维
bias  = torch.rand(512)                # bias 向量
result = batch + bias                  # (8,512) + (512,) → (8,512)
print(f"\n  batch + bias：{tuple(batch.shape)} + {tuple(bias.shape)} → {tuple(result.shape)}")
print(f"  bias 自动广播到每一行 ✓")

# 示例4：图像归一化
images = torch.rand(8, 3, 224, 224)   # (N, C, H, W)
mean   = torch.tensor([0.485, 0.456, 0.406]).reshape(1, 3, 1, 1)  # (1,C,1,1)
std    = torch.tensor([0.229, 0.224, 0.225]).reshape(1, 3, 1, 1)
normalized = (images - mean) / std
print(f"\n  图像归一化：{tuple(images.shape)} - mean{tuple(mean.shape)} → {tuple(normalized.shape)}")
print(f"  每个通道减去各自的均值 ✓")



# ============================================================
# 第8节：与 NumPy 互转
# ============================================================
section("8. 与 NumPy 互转")

print("""
  关键规则：
  ┌──────────────────────────────────────────────────────┐
  │  torch.from_numpy(arr)  共享内存（修改一方另一方也变）  │
  │  torch.tensor(arr)      复制数据（互不影响）           │
  │  tensor.numpy()         共享内存（仅 CPU tensor 可用） │
  │  tensor.detach().numpy()  安全写法（requires_grad 时） │
  └──────────────────────────────────────────────────────┘
""")

# Tensor → NumPy
t = torch.tensor([1.0, 2.0, 3.0])
n = t.numpy()                           # 共享内存
print(f"  tensor: {t.tolist()}")
print(f"  numpy:  {n}")
t[0] = 99
print(f"  修改 t[0]=99 后，numpy 也变了: {n}  ← 共享内存！")

# 安全写法（有梯度时）
t2 = torch.tensor([1.0, 2.0], requires_grad=True)
n2 = t2.detach().numpy()               # 先 detach 脱离计算图
print(f"\n  有梯度的 tensor → .detach().numpy(): {n2}")

# NumPy → Tensor
import numpy as np
n = np.array([4.0, 5.0, 6.0])

t_shared = torch.from_numpy(n)         # 共享内存
t_copy   = torch.tensor(n)             # 复制

n[0] = 100
print(f"\n  修改 numpy n[0]=100 后：")
print(f"  from_numpy (共享): {t_shared.tolist()}")
print(f"  tensor    (复制):  {t_copy.tolist()}")

# ============================================================
# 第9节：GPU 加速
# ============================================================
section("9. GPU 加速")

print("""
  GPU 加速原理：
  ┌─────────────────────────────────────────────────────┐
  │  CPU: 几十个核心，擅长复杂逻辑，串行计算              │
  │  GPU: 数千个小核心，擅长大量简单并行计算              │
  │                                                     │
  │  矩阵乘法正好适合 GPU 大规模并行                      │
  │  (8,3,224,224) 的张量运算：                          │
  │    CPU: 串行计算每个元素                              │
  │    GPU: 所有元素同时并行计算                          │
  └─────────────────────────────────────────────────────┘
""")

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"  当前可用设备: {device}")

# 方式1：创建时直接指定
t_gpu = torch.rand(3, 4, device=device)
print(f"\n  直接在 {device} 创建: shape={tuple(t_gpu.shape)}, device={t_gpu.device}")

# 方式2：.to(device)
t_cpu = torch.rand(3, 4)
t_gpu2 = t_cpu.to(device)
print(f"  .to('{device}'): {t_cpu.device} → {t_gpu2.device}")

# 方式3：.cuda() / .cpu()
# t.cuda()   等价于 t.to("cuda")
# t.cpu()    等价于 t.to("cpu")

# GPU 上的运算
A = torch.rand(1000, 1000, device=device)
B = torch.rand(1000, 1000, device=device)
C = A @ B   # 矩阵乘法在 GPU 上执行
print(f"\n  GPU 矩阵乘法: {tuple(A.shape)} @ {tuple(B.shape)} = {tuple(C.shape)}")

# 结果移回 CPU（用于 NumPy / matplotlib）
C_cpu = C.cpu()
print(f"  结果移回 CPU: {C_cpu.device}")

# 注意：不同设备的 tensor 不能直接运算
print("""
  ⚠️  常见错误：
  t_cpu = torch.rand(3)          # CPU
  t_gpu = torch.rand(3).cuda()   # GPU
  t_cpu + t_gpu   → RuntimeError: expected all tensors to be on the same device

  ✅  解决：确保所有 tensor 在同一设备
  t_cpu.to(device) + t_gpu  ← 正确
""")

# ============================================================
# 总结
# ============================================================
section("总结：张量操作速查表")
print("""
  ┌──────────────────────┬────────────────────────────────────┐
  │  操作                 │  说明                               │
  ├──────────────────────┼────────────────────────────────────┤
  │  torch.tensor(data)  │  从数据创建，复制                    │
  │  torch.zeros/ones    │  全0/全1，参数为 shape               │
  │  torch.rand/randn    │  均匀/正态分布随机                   │
  │  torch.arange        │  等差序列 [start, end)              │
  ├──────────────────────┼────────────────────────────────────┤
  │  t.shape             │  各维度大小                          │
  │  t.dtype             │  数据类型                            │
  │  t.device            │  存储设备                            │
  │  t.numel()           │  元素总数                            │
  ├──────────────────────┼────────────────────────────────────┤
  │  t.reshape(...)      │  重塑，-1 自动推断                   │
  │  t.view(...)         │  重塑（需内存连续）                   │
  │  t.flatten(start)    │  从 start 维开始展平                 │
  │  t.squeeze(dim)      │  去除 size=1 的维度                  │
  │  t.unsqueeze(dim)    │  在 dim 插入 size=1 的维度           │
  │  t.permute(dims)     │  重排所有维度                        │
  │  t.transpose(d0,d1)  │  交换两个维度                       │
  ├──────────────────────┼────────────────────────────────────┤
  │  a + b / a @ b       │  逐元素加 / 矩阵乘法                 │
  │  t.sum(dim)          │  沿 dim 求和                         │
  │  t.mean/max/argmax   │  均值/最大值/最大值索引              │
  │  keepdim=True        │  保持维度不变                        │
  ├──────────────────────┼────────────────────────────────────┤
  │  t.to(device/dtype)  │  改变设备/数据类型                   │
  │  t.numpy()           │  转 NumPy（共享内存）                │
  │  torch.from_numpy()  │  转 Tensor（共享内存）               │
  │  torch.tensor(arr)   │  转 Tensor（复制）                   │
  └──────────────────────┴────────────────────────────────────┘
""")
