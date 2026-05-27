# 在 M1 Mac 上运行 nanoGPT

> 涵盖：sample.py / train.py 在 M1 上的配置、中文数据集、内存限制、数据量建议

---

## 一、M1 Mac 能跑吗？

可以。M1 支持 **MPS（Metal Performance Shaders）** GPU 加速。

```
CUDA  → NVIDIA GPU（Linux/Windows）
MPS   → Apple Silicon GPU（M1/M2/M3/M4）
CPU   → 任何设备的兜底方案
```

---

## 二、运行 sample.py

### 关键参数

| 参数 | M1 必须设置 | 原因 |
|------|------------|------|
| `--device=mps` | ✅ | 用 M1 GPU 加速 |
| `--dtype=float32` | ✅ | MPS 不完全支持 bf16/fp16 |
| `--compile=False` | ✅ | torch.compile 在 MPS 不稳定 |

### 命令

```bash
# 直接用 GPT-2 预训练模型
python sample.py \
    --init_from=gpt2 \
    --device=mps \
    --dtype=float32 \
    --compile=False \
    --start="Once upon a time" \
    --max_new_tokens=200 \
    --num_samples=3
```

### 修改源码默认值（省得每次敲参数）

```python
# sample.py 顶部改这两行
device  = 'mps' if torch.backends.mps.is_available() else 'cpu'
dtype   = 'float32'
compile = False
```

### 速度参考

```
A100 GPU：    ~300 token/s
M1 MPS：      ~30-50 token/s
Intel Mac CPU：~5-10 token/s
```

---

## 三、运行 train.py

### 命令

```bash
# 训练莎士比亚字符级模型（最适合本地跑）
python train.py config/train_shakespeare_char.py \
    --device=mps \
    --dtype=float32 \
    --compile=False
```

### 修改源码默认值

```python
# train.py 顶部改这三行
device  = 'mps' if torch.backends.mps.is_available() else 'cpu'
dtype   = 'float32'
compile = False
```

### 训练速度参考

```
shakespeare_char 小模型：  ~500ms/iter
GPT-2 124M：               ~2-3s/iter（不建议在 M1 全量训练）
```

---

## 四、train.bin / val.bin / meta.pkl 是什么

### 三个文件的来源

```
python data/shakespeare_char/prepare.py
              ↓
  train.bin   ← 90% 数据（token id 序列）
  val.bin     ← 10% 数据（token id 序列）
  meta.pkl    ← 词表（字符 ↔ id 映射）
```

### 存储的内容

```
原始文本：  "First Citizen:\nBefore we proceed..."
              ↓ 字符级 tokenize
token ids：  [18, 47, 56, 57, 58, 1, 15, 47, ...]
              ↓ 存成二进制（uint16，每个 token 2 bytes）
train.bin：  [18, 47, 56, 57, 58, 1, 15, 47, ...]
```

### meta.pkl 的内容

```python
{
    'vocab_size': 65,
    'stoi': {'\n': 0, ' ': 1, '!': 2, ... 'z': 64},  # 字符 → id
    'itos': {0: '\n', 1: ' ', 2: '!', ... 64: 'z'},   # id → 字符
}
```

### 谁用谁不用

```
train.py  → 只读 train.bin（直接读数字，不需要 tokenizer）
sample.py → 用 meta.pkl encode/decode（字符级模型）
            用 tiktoken encode/decode（GPT-2 模型，没有 meta.pkl）
```

### 为什么存 .bin 不存 .txt

```
.txt：每次训练都要重新 tokenize，慢
.bin：已经 tokenize 好，直接读数字
      用 np.memmap 按需读取，不占 RAM
      随机访问任意位置是 O(1)
```

---

## 五、为什么字符可以作为 token

token 只是"把文本切成片段喂给模型"的方式，没有规定必须是什么粒度。

### 不同粒度对比

```
原始文本："Hello!"

字节级：  [72, 101, 108, 108, 111, 33]    6 个 token
字符级：  ['H','e','l','l','o','!']        6 个 token
子词级：  ['Hello', '!']                   2 个 token（BPE）
词级：    ['Hello!']                        1 个 token
```

### 字符级的优势

```
1. 词表极小（几十到几百）
   BPE 需要 50000+ 的 embedding 矩阵
   字符级只需要 65 个

2. 没有 OOV 问题
   BPE 遇到新词（如 "ChatGPT"）可能拆分不好
   字符级任何文字都由已知字符组成，永远不会 OOV

3. 实现极简
   prepare.py 只需要 20 行
```

### 字符级的代价

```
"To be or not to be"

字符级：18 个 token
BPE：    6 个 token

同样的 block_size=1024：
  字符级能看约 170 个英文单词的上下文
  BPE    能看约 700 个英文单词的上下文

需要更长序列才能捕捉同样的语义
→ 训练更难，需要更大模型或更多步数
```

---

## 六、换成中文数据集

字符级模型对语言无感知，汉字和字母对模型来说没有本质区别。

### 步骤

```bash
# 1. 准备中文文本
mkdir data/chinese_char
# 把中文 txt 放到 data/chinese_char/input.txt

# 2. 运行 prepare.py
cd data/chinese_char
python prepare.py

# 3. 训练
cd ../../
python train.py config/train_chinese_char.py \
    --device=mps \
    --dtype=float32 \
    --compile=False

# 4. 采样
python sample.py \
    --out_dir=out-chinese-char \
    --device=mps \
    --dtype=float32 \
    --compile=False \
    --start="红楼梦"
```

### prepare.py（中文版）

```python
import os
import pickle
import numpy as np

with open('input.txt', 'r', encoding='utf-8') as f:
    data = f.read()

chars     = sorted(list(set(data)))
vocab_size = len(chars)
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}

ids = [stoi[c] for c in data]
n   = len(ids)

np.array(ids[:int(n*0.9)], dtype=np.uint16).tofile('train.bin')
np.array(ids[int(n*0.9):], dtype=np.uint16).tofile('val.bin')

pickle.dump({'vocab_size': vocab_size, 'itos': itos, 'stoi': stoi},
            open('meta.pkl', 'wb'))
```

### 中英文对比

| | 英文（莎士比亚） | 中文 |
|---|---|---|
| vocab_size | 65 | 2000~5000 |
| 每个 token | 1 个字母 | 1 个汉字 |
| 语义密度 | 低 | 高（每个字有意义） |
| vocab 对内存影响 | 可忽略 | 仍然很小，不是瓶颈 |

---

## 七、M1 最大能训练多大的模型

### 内存计算公式

```
训练总内存 ≈ 参数量 × 22 bytes（float32）

分解：
  模型参数：  × 4 bytes
  梯度：      × 4 bytes
  AdamW m/v： × 8 bytes
  激活值：    × 6 bytes（估算）
```

### 参数量估算

```
参数量 ≈ 12 × n_layer × n_embd²

n_layer  n_head  n_embd   参数量    训练内存
  4        4      128      0.8M      ~18MB
  6        6      384      10M       ~220MB    ← M1 推荐
  8        8      512      25M       ~550MB
  12       12     768      85M       ~1.9GB    ← M1 上限
  12       12     1024     125M      ~2.8GB    ⚠️  勉强
  24       16     1024     350M      ~7.7GB    ❌  M1 OOM
```

### 各版本 M1 安全上限

```
M1 基础版（8GB）：  85M 参数
M1 Pro  （16GB）：  350M 参数
M1 Max  （32GB）：  1.5B 参数
M1 Ultra（64GB）：  3B 参数
```

### M1 基础版推荐配置

```python
# 保守配置（快，稳定）
n_layer = 6
n_head  = 6
n_embd  = 384    # 约 10M 参数，内存 ~220MB

# 接近上限（效果更好）
n_layer = 12
n_head  = 12
n_embd  = 768    # 约 85M 参数，内存 ~1.9GB
batch_size = 16
gradient_accumulation_steps = 8
```

---

## 八、最大能支持多大的训练文本

### 结论：文本大小几乎不限制内存

```python
# train.py 用 np.memmap 读数据

data = np.memmap('train.bin', dtype=np.uint16, mode='r')
# 不把文件读进内存
# 只建立"文件位置 → 虚拟地址"的映射
# 用到哪块才从磁盘加载那块

# train.bin 有 17GB  → 内存占用几乎 0
# train.bin 有 1TB   → 内存占用还是几乎 0
```

**文本大小 和 模型大小 是两个独立的问题。**

### 红楼梦的数字

```
红楼梦全文：
  约 73 万汉字
  原始文件约 2.7MB（UTF-8）

tokenize 后：
  730,000 个 token
  train.bin ≈ 1.46MB
  val.bin   ≈ 0.16MB

结论：硬盘和内存完全不是问题
```

### 莎士比亚数据集的数字

```
字符数：  1,115,394 个字符  ≈ 1.1MB
train.bin：约 1MB（90% = 1,003,854 tokens）
val.bin：  约 0.1MB（10% = 111,540 tokens）

和红楼梦对比：
  莎士比亚：1,115,394 tokens
  红楼梦：    730,000 tokens
  莎士比亚比红楼梦多约 50% 的 token
```

莎士比亚是专门用来**快速验证模型能跑通**的玩具数据集，不追求生成质量。

### 数据量够不够的问题

```
GPT-3 论文建议：参数量 × 20 的 token 数

模型参数    建议 token 数    对应中文字数        举例
  1M          20M tokens      约 2000 万汉字     红楼梦 × 27 本
  10M         200M tokens     约 2 亿汉字        红楼梦 × 274 本
  85M         1.7B tokens     约 17 亿汉字       红楼梦 × 2329 本

红楼梦只有 73 万字 → 只够训练约 1M 参数的小模型
```

### 实际建议

```
只有红楼梦一本（73万字）：
  用小模型防过拟合
  n_layer=4, n_head=4, n_embd=128（约 1M 参数）
  dropout=0.3
  能学到文风，但质量有限

想要更好效果：
  红楼梦 + 三国演义 + 水浒传 + 西游记 ≈ 300 万字
  或找古典文学语料库，目标 1 亿字以上
```

---

## 九、完整流程总结

```
准备数据
  python data/chinese_char/prepare.py
  → train.bin / val.bin / meta.pkl

训练
  python train.py config/train_chinese_char.py \
      --device=mps --dtype=float32 --compile=False
  → out-chinese-char/ckpt.pt

采样
  python sample.py \
      --out_dir=out-chinese-char \
      --device=mps --dtype=float32 --compile=False \
      --start="第一回"
  → 生成文本输出到终端
```
