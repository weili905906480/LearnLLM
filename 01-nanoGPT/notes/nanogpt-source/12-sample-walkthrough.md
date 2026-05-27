# sample.py 逐行深度解析

> nanoGPT 的推理引擎：**~80 行实现完整 GPT 文本生成流程**
>
> 源码：[github.com/karpathy/nanoGPT/blob/master/sample.py](https://github.com/karpathy/nanoGPT/blob/master/sample.py)
>
> 配合阅读：[02-model.md](./02-model.md)（模型结构）/ [04-sampling.md](./04-sampling.md)（快速概览）

---

## 文件总览

```
sample.py 的执行顺序：

 ① 超参数配置     — 定义生成行为的所有旋钮
 ② configurator  — 命令行/配置文件覆盖
 ③ 随机种子 & 精度 — 可复现性 + 混合精度上下文
 ④ 模型加载       — resume / gpt2 两种模式
 ⑤ Tokenizer 准备 — meta.pkl 或 tiktoken
 ⑥ prompt 编码    — 文本 → token ids
 ⑦ 自回归生成循环 — model.generate() × num_samples
 ⑧ 解码 & 输出    — token ids → 文本
```



---

## 一、导入与超参数配置

### 源码

```python
"""
Sample from a trained model
"""
import os
import pickle
from contextlib import nullcontext
import torch
import tiktoken
from model import GPTConfig, GPT
```

### 逐行解析

```python
import os
```
- 用于路径拼接（`os.path.join`）和文件存在性检查（`os.path.exists`）
- 比手写字符串路径更跨平台

```python
import pickle
```
- 用于加载 `meta.pkl`（字符级 tokenizer 的词表文件）
- pickle 是 Python 原生的对象序列化格式，可以保存任意 Python 对象

```python
from contextlib import nullcontext
```
- `nullcontext()` 是一个"什么都不做"的上下文管理器
- 作用：当设备是 CPU 时，不需要 `autocast`，但代码结构要统一
- 用法：`with nullcontext(): ...` 等价于 `...`（直接执行，无任何包装）

```python
import torch
```
- PyTorch 核心库，提供张量操作、自动微分、模型加载等一切功能

```python
import tiktoken
```
- OpenAI 开源的 BPE tokenizer 库
- 当没有自定义词表时，使用 GPT-2 的 tokenizer（50257 词）
- BPE（Byte Pair Encoding）：将常见字符组合合并成子词 token

```python
from model import GPTConfig, GPT
```
- `GPTConfig`：dataclass，保存模型超参数（n_layer, n_head, n_embd 等）
- `GPT`：GPT 模型类，包含 `forward()` 和 `generate()` 方法

---

### 超参数源码

```python
init_from = 'resume'
out_dir = 'out'
start = "\n"
num_samples = 10
max_new_tokens = 500
temperature = 0.8
top_k = 200
seed = 1337
device = 'cuda'
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16'
compile = False
exec(open('configurator.py').read())
```

### 逐行解析

```python
init_from = 'resume'
```
- 两种模式：
  - `'resume'`：从本地 checkpoint（`out/ckpt.pt`）加载自己训练的模型
  - `'gpt2'` / `'gpt2-medium'` / `'gpt2-large'` / `'gpt2-xl'`：直接加载 OpenAI 预训练权重



```python
out_dir = 'out'
```
- 当 `init_from == 'resume'` 时，从这个目录查找 `ckpt.pt`
- 当 `init_from` 是 gpt2 变体时，此参数被忽略
- 默认 `'out'` 对应 `train.py` 的输出目录

```python
start = "\n"
```
- 生成的起始 prompt（上下文/前缀）
- `"\n"` 表示以换行符开始（常用于字符级模型，如莎士比亚）
- 也可以是 `"<|endoftext|>"`（GPT-2 的文档开始 token）
- 或具体文本，如 `"Once upon a time"`
- 特殊语法：`"FILE:prompt.txt"` 表示从文件读取 prompt

```python
num_samples = 10
```
- 生成多少个独立的文本样本
- 每个样本都从相同的 `start` prompt 开始，但因为采样的随机性，结果各不相同
- 用于观察模型输出的多样性

```python
max_new_tokens = 500
```
- 每个样本最多生成多少个新 token（不含 prompt）
- token ≠ 字符：对于 GPT-2 tokenizer，英文平均 1 token ≈ 0.75 个单词
- 500 tokens ≈ 375 个英文单词 ≈ 一段短文

```python
temperature = 0.8
```
- 控制生成的随机程度
- `1.0`：原始 logits 分布，正常采样
- `< 1.0`（如 0.8）：logits 除以小数 → 各 token 概率差距拉大 → 更倾向高概率词 → 输出更确定、保守
- `> 1.0`：logits 除以大数 → 各 token 概率差距缩小 → 更均匀 → 输出更随机、有创意
- 极端情况：`temperature → 0` 等价于贪心解码（每次选最大概率 token）

```python
top_k = 200
```
- 只保留概率最高的前 k 个 token，其余概率设为 0（再 softmax）
- 防止低概率的"垃圾 token"被偶然采样到
- `k=200`：保留选择空间较大，生成多样
- `k=1`：等价于贪心解码
- `k=50000`（约等于词表大小）：等价于不过滤

```python
seed = 1337
```
- 随机种子，用于结果可复现
- 同样的 seed + 同样的模型 + 同样的 prompt → 每次运行生成结果完全相同
- `1337` 是 Karpathy 惯用的种子（leet speak 里 "1337" = "leet"，黑客文化）

```python
device = 'cuda'
```
- 模型和张量运行在哪个设备上
- 常见值：`'cpu'`、`'cuda'`、`'cuda:0'`（第 0 块 GPU）、`'mps'`（Apple Silicon）

```python
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16'
```
- 自动选择最佳浮点精度：
  - A100/H100 等新 GPU：支持 BF16 → 选 `bfloat16`（范围大，训练稳定）
  - 老 GPU（V100/T4）：不支持 BF16 → 选 `float16`（速度快但范围小）
  - CPU 时：默认不进入 autocast，这行值被 `ctx = nullcontext()` 忽略

```python
compile = False
```
- 是否用 `torch.compile()` 加速模型
- 推理时默认关闭：推理通常只跑一次，编译开销（~1分钟）得不偿失
- 如果要批量推理（如评测），可以打开

```python
exec(open('configurator.py').read())
```
- 读取并执行 `configurator.py` 的源码，允许从命令行覆盖上面任意参数
- 例：`python sample.py --temperature=1.2 --top_k=40 --start="To be or not to be"`
- 不需要提前声明参数，任何全局变量都可以被覆盖



---

## 二、随机种子与混合精度上下文

### 源码

```python
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
device_type = 'cuda' if 'cuda' in device else 'cpu'
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)
```

### 逐行解析

```python
torch.manual_seed(seed)
```
- 设置 CPU 侧的全局随机种子
- 影响所有 CPU 上的随机操作（如 `torch.randint`、dropout 等）
- 保证在相同 seed 下，每次运行生成结果完全一致

```python
torch.cuda.manual_seed(seed)
```
- 设置当前 GPU 的随机种子
- 与 `manual_seed` 独立：CPU 和 GPU 有各自的随机数生成器
- 影响 GPU 上的随机操作（如 `torch.multinomial` 在 GPU 上的采样）
- 注意：如果用多 GPU（`manual_seed_all`），这里只设置当前卡

```python
torch.backends.cuda.matmul.allow_tf32 = True
```
- 允许矩阵乘法使用 TF32 格式（Tensor Float 32）
- TF32 是 A100 引入的格式：精度略低于 FP32，但速度约 8 倍快
- 内部格式：10 位尾数（FP32 是 23 位），8 位指数（和 FP32 一样）
- 推理时精度损失可忽略不计，但速度提升明显
- 默认 PyTorch 1.7+ 中这个选项是 True，这里显式设置以防被关闭

```python
torch.backends.cudnn.allow_tf32 = True
```
- 允许 cuDNN（处理卷积的 CUDA 库）使用 TF32
- nanoGPT 不用卷积，但这是标准配置，一并打开

```python
device_type = 'cuda' if 'cuda' in device else 'cpu'
```
- 从设备字符串中提取设备类型
- 处理 `device='cuda:1'` 这样的情况：`'cuda' in 'cuda:1'` → `True` → `device_type='cuda'`
- 后续 `torch.amp.autocast` 需要传 `'cuda'` 或 `'cpu'`，不能传 `'cuda:1'`

```python
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
```
- 把字符串形式的 dtype（如 `'bfloat16'`）转换成 PyTorch 的 dtype 对象（`torch.bfloat16`）
- 字典查找：`dtype` 字符串作为 key，`torch.dtype` 对象作为 value
- `autocast` 需要 `torch.dtype` 对象而不是字符串

```python
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)
```
- 根据设备类型选择混合精度上下文：
  - CPU：`nullcontext()`，即什么都不做，直接用 FP32 运行
  - GPU：`torch.amp.autocast()`，自动将矩阵乘法等操作降精度到 BF16/FP16

#### autocast 做了什么？

```
模型所有参数始终保存为 FP32（精确）
autocast 上下文内运行 forward 时：
  矩阵乘法 (Linear, Attention)   → 自动转为 BF16 计算（快 2-3x）
  softmax, layer_norm, loss 计算  → 保持 FP32（精度敏感，不降）

结果：速度接近纯 BF16，数值稳定接近纯 FP32
推理时不需要 GradScaler（那是训练时处理 FP16 梯度下溢用的）
```



---

## 三、模型加载

### 源码

```python
if init_from == 'resume':
    ckpt_path = os.path.join(out_dir, 'ckpt.pt')
    checkpoint = torch.load(ckpt_path, map_location=device)
    gptconf = GPTConfig(**checkpoint['model_args'])
    model = GPT(gptconf)
    state_dict = checkpoint['model']
    unwanted_prefix = '_orig_mod.'
    for k,v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
elif init_from.startswith('gpt2'):
    model = GPT.from_pretrained(init_from, dict(dropout=0.0))
```

### 逐行解析（resume 分支）

```python
ckpt_path = os.path.join(out_dir, 'ckpt.pt')
```
- 拼接 checkpoint 文件路径，默认为 `'out/ckpt.pt'`
- `os.path.join` 会自动处理不同操作系统的路径分隔符（Windows 用 `\`，Linux/Mac 用 `/`）

```python
checkpoint = torch.load(ckpt_path, map_location=device)
```
- 将 checkpoint 文件加载到内存
- `map_location=device`：把存储在文件里的张量映射到目标设备
  - 如果 checkpoint 是在 `cuda:0` 上保存的，但现在要在 `cpu` 推理 → 自动重映射
  - 没有这个参数，会报错"CUDA error: no kernel image is available for execution on the device"

- checkpoint 是一个字典，包含：
  ```
  {
    'model':       模型参数 state_dict,
    'optimizer':   优化器状态（sample.py 不需要，但 train.py 保存了），
    'model_args':  {'n_layer': 12, 'n_head': 12, 'n_embd': 768, ...},
    'iter_num':    训练到第几步,
    'best_val_loss': 历史最低 val loss,
    'config':      完整的训练超参数
  }
  ```

```python
gptconf = GPTConfig(**checkpoint['model_args'])
```
- 从 checkpoint 中读取模型结构参数，重建 `GPTConfig` 对象
- `**` 解包字典：`GPTConfig(n_layer=12, n_head=12, n_embd=768, ...)`
- **关键**：必须用 checkpoint 里的 `model_args` 而不是当前文件的默认值
  - 否则如果模型结构不匹配（比如训练的是 6 层，但默认是 12 层），`load_state_dict` 会报错

```python
model = GPT(gptconf)
```
- 用配置对象创建一个**随机初始化**的 GPT 模型（参数全是随机值）
- 接下来会用 `load_state_dict` 把训练好的参数填入

```python
state_dict = checkpoint['model']
```
- 从 checkpoint 字典中取出模型参数字典
- state_dict 的结构：
  ```
  {
    'transformer.wte.weight': tensor(...),   # token embedding
    'transformer.wpe.weight': tensor(...),   # position embedding
    'transformer.h.0.ln_1.weight': tensor(...),  # 第0层 LayerNorm
    ...
    'lm_head.weight': tensor(...),           # 输出层（与 wte 共享权重）
  }
  ```

```python
unwanted_prefix = '_orig_mod.'
for k,v in list(state_dict.items()):
    if k.startswith(unwanted_prefix):
        state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
```
- 处理 `torch.compile()` 引起的 key 前缀污染问题
- **问题**：`train.py` 里如果 `compile=True`，保存 checkpoint 时 state_dict 的 key 会被加上 `'_orig_mod.'` 前缀：
  ```
  训练时（compile=True）保存的 key:
    '_orig_mod.transformer.wte.weight'
  
  未编译的模型期望的 key:
    'transformer.wte.weight'
  ```
- **解决**：遍历所有 key，如果有该前缀，创建去掉前缀的新 key，删除旧 key
- `list(state_dict.items())`：必须先转成列表再遍历，否则遍历中修改字典会报错
- `k[len(unwanted_prefix):]`：`k[10:]`，切掉前 10 个字符（`'_orig_mod.'` 长度为 10）

```python
model.load_state_dict(state_dict)
```
- 将训练好的参数加载进模型
- 严格匹配：state_dict 中的每个 key 必须在模型中有对应参数，反之亦然
- 如果不匹配（多了或少了参数），会报 `RuntimeError`



### 逐行解析（gpt2 分支）

```python
elif init_from.startswith('gpt2'):
    model = GPT.from_pretrained(init_from, dict(dropout=0.0))
```

- `init_from.startswith('gpt2')` 匹配所有 GPT-2 变体：
  - `'gpt2'`：124M 参数（12 层，768 维）
  - `'gpt2-medium'`：345M 参数（24 层，1024 维）
  - `'gpt2-large'`：774M 参数（36 层，1280 维）
  - `'gpt2-xl'`：1.5B 参数（48 层，1600 维）

- `GPT.from_pretrained(init_from, dict(dropout=0.0))`：
  - 从 HuggingFace Hub 下载 OpenAI 官方预训练权重
  - `dropout=0.0`：推理时关闭 dropout（训练时可能是 0.1 等）
  - 第一次调用会下载权重文件（缓存在 `~/.cache/huggingface/`），之后直接读缓存

---

## 四、模型推理模式配置

### 源码

```python
model.eval()
model.to(device)
if compile:
    model = torch.compile(model)
```

### 逐行解析

```python
model.eval()
```
- 将模型切换到评估模式（与 `model.train()` 相对）
- **主要效果**：
  - Dropout 层：`eval` 模式下关闭（所有神经元都参与计算，没有随机 mask）
  - BatchNorm 层：使用训练期间积累的全局统计量（nanoGPT 没有 BatchNorm，但保持好习惯）
- **注意**：`model.eval()` 本身**不关闭**梯度计算，需要搭配 `torch.no_grad()` 才能节省显存

```python
model.to(device)
```
- 将模型的所有参数从 CPU 转移到目标设备（GPU）
- 为什么不在 `torch.load` 时就直接加载到 GPU？
  - `torch.load(map_location=device)` 将张量数据加载到目标设备
  - 但 `GPT(gptconf)` 创建的是空模型（参数在 CPU）
  - `load_state_dict` 后参数已经在目标设备（因为 checkpoint 里的张量在目标设备）
  - 这里的 `model.to(device)` 是保险措施，确保所有参数（包括 Buffer 等）都在正确设备

```python
if compile:
    model = torch.compile(model)
```
- 可选：用 PyTorch 2.0 编译模型为更高效的形式
- 推理时默认 `compile=False`：
  - 编译需要约 1 分钟（JIT 编译生成优化的 CUDA kernel）
  - 如果只生成 10 个样本，编译时间 >> 生成时间，得不偿失
  - 适合需要大量推理的场景（如评测几千个样本）



---

## 五、Tokenizer 准备

### 源码

```python
load_meta = False
if init_from == 'resume' and 'config' in checkpoint and 'dataset' in checkpoint['config']:
    meta_path = os.path.join('data', checkpoint['config']['dataset'], 'meta.pkl')
    load_meta = os.path.exists(meta_path)
if load_meta:
    print(f"Loading meta from {meta_path}...")
    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)
    stoi, itos = meta['stoi'], meta['itos']
    encode = lambda s: [stoi[c] for c in s]
    decode = lambda l: ''.join([itos[i] for i in l])
else:
    print("No meta.pkl found, assuming GPT-2 encodings...")
    enc = tiktoken.get_encoding("gpt2")
    encode = lambda s: enc.encode(s, allowed_special={"<|endoftext|>"})
    decode = lambda l: enc.decode(l)
```

### 逐行解析

```python
load_meta = False
```
- 初始化 flag，默认不加载 meta（使用 GPT-2 tokenizer）

```python
if init_from == 'resume' and 'config' in checkpoint and 'dataset' in checkpoint['config']:
```
- 三个条件同时满足才尝试加载 meta：
  1. `init_from == 'resume'`：只有加载本地 checkpoint 才可能有自定义 tokenizer（gpt2 变体固定用 tiktoken）
  2. `'config' in checkpoint`：兼容旧版 checkpoint（早期的 nanoGPT 可能没保存 config 字段）
  3. `'dataset' in checkpoint['config']`：兼容旧版（早期 config 可能没有 dataset 字段）

```python
meta_path = os.path.join('data', checkpoint['config']['dataset'], 'meta.pkl')
```
- 根据训练时用的数据集名称，拼接 meta 文件路径
- 例：`dataset='shakespeare_char'` → `'data/shakespeare_char/meta.pkl'`
- `meta.pkl` 是 `data/prepare.py` 在数据预处理时生成的，包含字符到 id 的映射

```python
load_meta = os.path.exists(meta_path)
```
- 检查文件是否实际存在
- 即使路径拼接成功，文件不存在也不能加载（避免 `FileNotFoundError`）

```python
with open(meta_path, 'rb') as f:
    meta = pickle.load(f)
```
- 以二进制读取模式（`'rb'`）打开文件
- `pickle.load` 反序列化文件内容，恢复成 Python 字典
- `meta` 的结构：
  ```python
  {
    'vocab_size': 65,        # 字符数量（莎士比亚数据集有 65 个不同字符）
    'itos': {0: '\n', 1: ' ', 2: '!', ...},   # id → 字符
    'stoi': {'\n': 0, ' ': 1, '!': 2, ...},   # 字符 → id
  }
  ```

```python
stoi, itos = meta['stoi'], meta['itos']
```
- `stoi`：String To Index，字典，`{'a': 0, 'b': 1, ...}`
- `itos`：Index To String，字典（或列表），`{0: 'a', 1: 'b', ...}`

```python
encode = lambda s: [stoi[c] for c in s]
```
- 字符级编码函数：把字符串转成 token id 列表
- 逐字符查 `stoi` 字典
- 例：`encode("hi")` → `[stoi['h'], stoi['i']]` → `[39, 46]`（莎士比亚词表中的 id）
- **限制**：只能处理训练数据中出现过的字符，遇到未知字符会 `KeyError`

```python
decode = lambda l: ''.join([itos[i] for i in l])
```
- 字符级解码函数：把 token id 列表转回字符串
- 逐 id 查 `itos` 字典，得到字符列表，再 `join` 成字符串
- 例：`decode([39, 46])` → `['h', 'i']` → `"hi"`

```python
enc = tiktoken.get_encoding("gpt2")
```
- 加载 GPT-2 使用的 BPE tokenizer
- 词表大小：50,257 个 token（包括 256 个字节 token + 合并的子词 token + 1 个特殊 token）
- 首次调用会下载词表文件（约几 MB），缓存在本地

```python
encode = lambda s: enc.encode(s, allowed_special={"<|endoftext|>"})
```
- 用 tiktoken 编码字符串为 token id 列表
- `allowed_special={"<|endoftext|>"}`：允许 `<|endoftext|>` 这个特殊 token 被编码
  - 默认 tiktoken 会报错，因为特殊 token 可能被用户误写
  - 显式允许它，这样 `start="<|endoftext|>"` 能正确编码为 `[50256]`

```python
decode = lambda l: enc.decode(l)
```
- 用 tiktoken 解码 token id 列表为字符串
- BPE 解码直接查合并表，不是逐字符的



#### 字符级 tokenizer vs BPE tokenizer 对比

```
字符级（meta.pkl）：
  单位：单个字符（'a', 'b', 'c', ...）
  词表：小（65 个字符 for 莎士比亚）
  优点：无 OOV（词表外）问题，简单直观
  缺点：序列长，一个单词 = 多个 token，需要更大的 context window

BPE（tiktoken GPT-2）：
  单位：子词（'he', 'llo', ' world' 等）
  词表：大（50,257 个 token）
  优点：序列短，'hello world' = ['hello', ' world'] = 2 个 token
  缺点：词表复杂，编码需要 Byte Pair Encoding 算法

哪个模型用哪个 tokenizer：
  自己训练字符级模型（shakespeare_char）→ meta.pkl
  微调 GPT-2 或加载 GPT-2 权重          → tiktoken (gpt2 编码)
```

---

## 六、Prompt 编码

### 源码

```python
if start.startswith('FILE:'):
    with open(start[5:], 'r', encoding='utf-8') as f:
        start = f.read()
start_ids = encode(start)
x = (torch.tensor(start_ids, dtype=torch.long, device=device)[None, ...])
```

### 逐行解析

```python
if start.startswith('FILE:'):
    with open(start[5:], 'r', encoding='utf-8') as f:
        start = f.read()
```
- 特殊协议：如果 `start` 以 `'FILE:'` 开头，说明 prompt 来自文件
- `start[5:]`：切掉前 5 个字符（`'FILE:'` 的长度），得到真实文件路径
  - 例：`start = "FILE:prompt.txt"` → 读取 `prompt.txt` 的内容
- `encoding='utf-8'`：显式指定编码，避免在 Windows 上因默认 GBK 编码读取错误
- 读取后 `start` 被替换为文件内容，后续处理和普通字符串一致

```python
start_ids = encode(start)
```
- 调用前面定义好的 `encode` 函数，把 prompt 字符串转成 token id 列表
- 例：
  - 字符级：`encode("\n")` → `[0]`（换行符对应 id 0）
  - BPE：`encode("\n")` → `[198]`（tiktoken 里换行符的 id）

```python
x = (torch.tensor(start_ids, dtype=torch.long, device=device)[None, ...])
```
- **分三步理解**：

  **Step 1**：`torch.tensor(start_ids, dtype=torch.long, device=device)`
  - 把 Python 列表 `start_ids` 转成 PyTorch 张量
  - `dtype=torch.long`：`int64` 类型，embedding 层的索引必须是整数
  - `device=device`：直接在目标设备上创建，避免 CPU→GPU 的拷贝
  - 形状：`(T,)`，T 是 prompt 的 token 数量（一维）

  **Step 2**：`[None, ...]`
  - `None` 在方括号里等价于 `unsqueeze(0)`，在第 0 维增加一个维度
  - `...`（Ellipsis）表示保留剩余所有维度
  - 形状变化：`(T,)` → `(1, T)`
  - **为什么要加维度**？模型 `forward()` 接受的输入形状是 `(B, T)`（batch × sequence），即使 batch=1 也要有这个维度

  **结果**：`x` 的形状为 `(1, T)`，如 `(1, 1)`（prompt 是 `"\n"`，1 个 token）



---

## 七、自回归生成循环

### 源码

```python
with torch.no_grad():
    with ctx:
        for k in range(num_samples):
            y = model.generate(x, max_new_tokens, temperature=temperature, top_k=top_k)
            print(decode(y[0].tolist()))
            print('---------------')
```

### 逐行解析

```python
with torch.no_grad():
```
- **最关键的推理优化**：禁用梯度计算
- **没有这行会怎样**？
  - 每次 forward 时，PyTorch 会追踪所有操作，构建计算图（用于反向传播）
  - 所有中间激活值（attention 输出、FFN 输出等）都被保存在显存中
  - 生成 500 个 token 就要保存 500 步的计算图 → 显存爆炸
- **有这行**：不构建计算图，中间激活值用完即释放 → 显存占用只有模型参数本身
- 速度也更快（少了 autograd 的追踪开销）

```python
with ctx:
```
- 进入之前配置的混合精度上下文
- CPU：`nullcontext()`，什么都不做
- GPU：`torch.amp.autocast(dtype=torch.bfloat16)`，自动降精度加速矩阵运算
- 与 `torch.no_grad()` 嵌套使用：两者相互独立，可以任意组合

```python
for k in range(num_samples):
```
- 循环 `num_samples`（默认 10）次，每次生成一个独立的文本样本
- 每次从同样的 `x`（prompt）开始，但因为采样（`temperature` + `top_k`）有随机性，结果不同
- 如果 `temperature=0`（greedy），所有 10 个样本会完全相同

```python
y = model.generate(x, max_new_tokens, temperature=temperature, top_k=top_k)
```
- 调用 GPT 模型的 `generate` 方法，执行自回归生成
- **参数**：
  - `x`：起始 token 序列，形状 `(1, T_prompt)`
  - `max_new_tokens`：最多生成多少新 token（不含 prompt）
  - `temperature`：采样温度
  - `top_k`：只保留前 k 个候选 token
- **返回**：`y`，形状 `(1, T_prompt + max_new_tokens)`，包含 prompt + 生成内容

#### model.generate() 内部逻辑（在 model.py 中）

```python
@torch.no_grad()
def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
    for _ in range(max_new_tokens):
        # 1. 截断：如果序列超过 block_size，只取最后 block_size 个 token
        idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]

        # 2. 前向传播，得到 logits
        logits, _ = self(idx_cond)

        # 3. 只取最后一个位置的 logits（预测下一个 token）
        logits = logits[:, -1, :] / temperature

        # 4. Top-k 过滤
        if top_k is not None:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = -float('Inf')

        # 5. Softmax 转概率
        probs = F.softmax(logits, dim=-1)

        # 6. 按概率采样
        idx_next = torch.multinomial(probs, num_samples=1)

        # 7. 拼接到序列末尾
        idx = torch.cat((idx, idx_next), dim=1)
    return idx
```

```
步骤分解（以第一步为例，prompt = [198]，即'\n'）：

输入 idx: [[198]]，形状 (1, 1)

Step 1 截断：1 <= 1024，不需要截断，idx_cond = [[198]]

Step 2 前向：logits 形状 (1, 1, 50257)
  模型读取 token [198]，在所有 50257 个词上输出分数

Step 3 取最后位置 + 除以 temperature：
  logits = logits[:, -1, :] / 0.8
  形状变为 (1, 50257)
  除以 0.8 使分布更尖锐（高分更高，低分更低）

Step 4 Top-k 过滤：
  找出前 200 个最高分的 token
  第 200 名的分数作为阈值
  低于阈值的位置设为 -inf（softmax 后概率≈0）

Step 5 Softmax：
  (1, 50257) 的 logits → (1, 50257) 的概率分布，总和为 1

Step 6 采样：
  torch.multinomial 按概率随机抽取 1 个 token
  比如抽到 id=198（'\n'），或 id=785（'The'）

Step 7 拼接：
  idx = cat([[198], [785]], dim=1) = [[198, 785]]
  形状 (1, 2)

下一步用 [[198, 785]] 作为输入，预测第 3 个 token...
如此反复，直到生成 500 个新 token
```



```python
print(decode(y[0].tolist()))
```
- **分三步**：
  1. `y[0]`：取 batch 中第 0 个样本（因为 batch_size=1，就是整个序列），形状从 `(1, T)` 变为 `(T,)`
  2. `.tolist()`：把 GPU 上的 PyTorch 张量转换为 Python 整数列表，同时数据从 GPU 传回 CPU
  3. `decode(...)`：把 token id 列表转回人类可读的字符串

```python
print('---------------')
```
- 在每个样本之间打印分隔线，方便区分不同的生成结果
- 每生成完一个样本就立即打印（而不是等全部生成完），这样可以实时看到输出

---

## 八、完整数据流图

```
prompt 字符串
  "Once upon a time"
        ↓ encode()
  token ids: [7454, 2402, 257, 640]
        ↓ torch.tensor()[None, ...]
  张量 x: shape (1, 4)  ← 在 GPU 上

  ┌─────────────────────────────────────────────────────┐
  │  生成循环（max_new_tokens=500 步）                    │
  │                                                      │
  │  每步：                                              │
  │    ① 截断序列到 block_size（1024）                   │
  │    ② forward(x) → logits (1, T, 50257)              │
  │    ③ 取最后位置 logits (1, 50257)                   │
  │    ④ 除以 temperature (0.8)                         │
  │    ⑤ top-k 过滤（保留前 200 个）                    │
  │    ⑥ softmax → 概率分布                             │
  │    ⑦ multinomial 采样 → 1 个 token id               │
  │    ⑧ 拼接到序列末尾                                 │
  │                                                      │
  └─────────────────────────────────────────────────────┘
        ↓ y: shape (1, 504)   ← 4 个 prompt token + 500 个新 token
  y[0].tolist()
  → [7454, 2402, 257, 640, 290, 257, ...]
        ↓ decode()
  "Once upon a time and a ..."
        ↓ print
  输出到终端
```

---

## 九、关键概念深度解析

### Temperature 的数学原理

```
模型输出的 logits（未归一化的分数）：
  假设词表 4 个词，logits = [2.0, 1.0, 0.5, 0.1]

temperature=1.0（不变）：
  softmax → [0.576, 0.212, 0.128, 0.085]   差距明显

temperature=0.5（更低）：
  logits / 0.5 = [4.0, 2.0, 1.0, 0.2]
  softmax → [0.839, 0.114, 0.042, 0.005]   高概率词更占优势，分布更尖

temperature=2.0（更高）：
  logits / 2.0 = [1.0, 0.5, 0.25, 0.05]
  softmax → [0.358, 0.296, 0.272, 0.244]   接近均匀分布，更随机

temperature→0（极限）：
  logits / 0 → 概率全给最大 logit 的词 → 贪心解码
```

### Top-k 过滤的实现细节

```python
v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
logits[logits < v[:, [-1]]] = -float('Inf')
```

```
假设 top_k=3，logits = [2.0, 0.5, 1.5, 3.0, 0.1]

torch.topk(logits, 3) 返回：
  v = [3.0, 2.0, 1.5]  ← 前3大的值
  _ = [3, 0, 2]         ← 对应的索引（不使用）

v[:, [-1]] = [[1.5]]    ← 第3名的分数（阈值）

logits < 1.5 的位置：索引 1（0.5）和索引 4（0.1）
这些位置设为 -inf

结果 logits = [2.0, -inf, 1.5, 3.0, -inf]
softmax 后 -inf 位置概率为 0，不会被采样

min(top_k, logits.size(-1))：
  防止 top_k > 词表大小（vocab_size=50257）
  如果 top_k=99999，就取全部 50257 个词（等于不过滤）
```

### torch.multinomial 采样

```python
idx_next = torch.multinomial(probs, num_samples=1)
```

```
probs = [0.6, 0.0, 0.3, 0.1, 0.0]（经过 top-k 和 softmax 后）

torch.multinomial：按权重随机抽样
  60% 的概率返回索引 0
  0%  的概率返回索引 1
  30% 的概率返回索引 2
  10% 的概率返回索引 3
  0%  的概率返回索引 4

num_samples=1：只抽一个
返回：tensor([[2]])，形状 (1, 1)

对比 argmax（贪心）：
  argmax → 永远返回索引 0（最大概率）
  multinomial → 按概率随机，增加多样性
```

### 序列截断（context window 限制）

```python
idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
```

```
GPT 模型的 position embedding 只有 block_size（1024）个位置
超过 1024 个 token 的序列无法直接处理（没有对应的位置编码）

解决方案：滑动窗口
  - 当序列长度 ≤ 1024：完整输入
  - 当序列长度 > 1024：只取最后 1024 个 token

代价：
  前面的内容（超过 1024 token 之前的部分）被"遗忘"
  这是经典 Transformer 的根本限制
  （解决方案：RoPE/ALiBi 等相对位置编码，或更大的 context window）
```



---

## 十、常见使用场景

### 从自己训练的模型采样

```bash
# 基本用法（默认参数）
python sample.py

# 自定义参数
python sample.py \
    --out_dir=out-shakespeare-char \
    --start="ROMEO:" \
    --num_samples=5 \
    --max_new_tokens=200 \
    --temperature=0.9 \
    --top_k=50
```

### 从 GPT-2 预训练模型采样

```bash
# 使用 GPT-2 小版本（124M）
python sample.py --init_from=gpt2 --start="The future of AI is"

# 使用 GPT-2 XL（1.5B，效果最好）
python sample.py --init_from=gpt2-xl --start="To be or not to be"
```

### 从文件读取 prompt

```bash
# 创建 prompt 文件
echo "In a galaxy far, far away" > myprompt.txt

# 使用文件作为 prompt
python sample.py --start="FILE:myprompt.txt"
```

---

## 十一、参数调优实验指南

```
目标：观察不同参数对生成质量的影响

实验 1：Temperature
  temperature=0.5   → 生成更保守、重复、"安全"，但可能单调
  temperature=0.8   → 推荐默认，平衡多样性和连贯性
  temperature=1.0   → 原始分布，可能产生意外词汇
  temperature=1.5   → 容易产生不连贯的"乱码"

实验 2：Top-k
  top_k=1    → 贪心解码，完全确定，无随机性
  top_k=10   → 保守，只在前10个词里选
  top_k=200  → 默认，较大的选择空间
  top_k=None → 不过滤（等价于 top_k=vocab_size），可能偶尔出现低质量词

实验 3：温度 + top-k 组合
  temperature=0.8, top_k=200   → 默认，适合大多数场景
  temperature=1.0, top_k=40    → top-k 限制更严，适合需要连贯性时
  temperature=0.6, top_k=20    → 非常保守，高度连贯但缺乏创意
  temperature=1.2, top_k=500   → 更有创意，适合创意写作
```

---

## 十二、完整代码注释版

```python
"""
Sample from a trained model
从训练好的模型中采样生成文本
"""
import os
import pickle
from contextlib import nullcontext   # 空上下文管理器，CPU 时用
import torch
import tiktoken                       # OpenAI 的 BPE tokenizer
from model import GPTConfig, GPT

# ── 超参数配置 ─────────────────────────────────────────────
init_from = 'resume'      # 'resume'=加载本地ckpt / 'gpt2'=加载OpenAI权重
out_dir = 'out'           # resume 时的 checkpoint 目录
start = "\n"              # prompt 起始文本（也可 "FILE:xxx.txt"）
num_samples = 10          # 生成几个独立的文本样本
max_new_tokens = 500      # 每个样本最多生成多少新 token
temperature = 0.8         # <1 更确定，>1 更随机
top_k = 200               # 只在概率最高的 k 个 token 中采样
seed = 1337               # 随机种子，保证可复现
device = 'cuda'           # 运行设备
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16'
compile = False           # 是否用 torch.compile 加速（推理一般不开）

exec(open('configurator.py').read())  # 允许命令行覆盖以上任意参数

# ── 随机种子和精度设置 ─────────────────────────────────────
torch.manual_seed(seed)               # CPU 随机种子
torch.cuda.manual_seed(seed)          # GPU 随机种子

torch.backends.cuda.matmul.allow_tf32 = True  # 矩阵乘允许 TF32（A100 更快）
torch.backends.cudnn.allow_tf32 = True        # cuDNN 允许 TF32

device_type = 'cuda' if 'cuda' in device else 'cpu'  # 提取设备类型

# 字符串 → PyTorch dtype 对象
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]

# CPU 用空上下文，GPU 用混合精度 autocast
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

# ── 模型加载 ──────────────────────────────────────────────
if init_from == 'resume':
    ckpt_path = os.path.join(out_dir, 'ckpt.pt')         # 拼接路径
    checkpoint = torch.load(ckpt_path, map_location=device)  # 加载到目标设备
    gptconf = GPTConfig(**checkpoint['model_args'])        # 用 ckpt 里的结构参数建 config
    model = GPT(gptconf)                                   # 创建随机初始化的模型
    state_dict = checkpoint['model']                       # 取出训练好的参数

    # 去掉 torch.compile 留下的 '_orig_mod.' 前缀
    unwanted_prefix = '_orig_mod.'
    for k,v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)

    model.load_state_dict(state_dict)                      # 加载训练好的参数

elif init_from.startswith('gpt2'):
    model = GPT.from_pretrained(init_from, dict(dropout=0.0))  # 从 HuggingFace 下载

model.eval()        # 关闭 dropout，切换推理模式
model.to(device)    # 移到 GPU

if compile:
    model = torch.compile(model)   # 可选编译加速（通常推理不需要）

# ── Tokenizer 准备 ────────────────────────────────────────
load_meta = False
# 检查是否有自定义字符级 tokenizer（来自本地训练的模型）
if init_from == 'resume' and 'config' in checkpoint and 'dataset' in checkpoint['config']:
    meta_path = os.path.join('data', checkpoint['config']['dataset'], 'meta.pkl')
    load_meta = os.path.exists(meta_path)

if load_meta:
    print(f"Loading meta from {meta_path}...")
    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)                       # 加载字符词表
    stoi, itos = meta['stoi'], meta['itos']         # 字符↔id 映射字典
    encode = lambda s: [stoi[c] for c in s]         # 字符串 → token ids
    decode = lambda l: ''.join([itos[i] for i in l])  # token ids → 字符串
else:
    print("No meta.pkl found, assuming GPT-2 encodings...")
    enc = tiktoken.get_encoding("gpt2")             # 加载 BPE tokenizer
    encode = lambda s: enc.encode(s, allowed_special={"<|endoftext|>"})
    decode = lambda l: enc.decode(l)

# ── Prompt 编码 ───────────────────────────────────────────
if start.startswith('FILE:'):                        # 从文件读取 prompt
    with open(start[5:], 'r', encoding='utf-8') as f:
        start = f.read()

start_ids = encode(start)                            # 字符串 → token ids 列表
x = (torch.tensor(start_ids, dtype=torch.long, device=device)[None, ...])
# torch.tensor(...): Python 列表 → 张量，形状 (T,)
# [None, ...]:       增加 batch 维度，形状 (T,) → (1, T)

# ── 自回归生成 ────────────────────────────────────────────
with torch.no_grad():   # 禁用梯度，节省显存，加快推理
    with ctx:            # 混合精度上下文（GPU 时 autocast）
        for k in range(num_samples):  # 生成 num_samples 个独立样本
            y = model.generate(x, max_new_tokens, temperature=temperature, top_k=top_k)
            # y 形状: (1, T_prompt + max_new_tokens)
            print(decode(y[0].tolist()))  # y[0]: 去掉 batch 维，tolist(): 转 Python 列表
            print('---------------')
```

