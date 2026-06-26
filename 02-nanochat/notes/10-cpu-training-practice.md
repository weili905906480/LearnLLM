# nanochat CPU 训练实践记录

---

## 环境配置

**硬件：**
- CPU: AMD Ryzen 7 5800（8核16线程）
- 内存: 32 GB
- GPU: GeForce GT 730（不可用，太旧）

**软件：**
- OS: Windows
- Python: 3.10（uv 自动安装）
- PyTorch: 2.9.1+cpu

**安装步骤：**
```bash
pip install uv
uv sync --extra cpu
```

---

## 问题与解决方案

### 1. HuggingFace 网络问题

**问题：** 无法连接到 huggingface.co（国内网络）

**解决：** 修改 `nanochat/dataset.py:23` 使用镜像：
```python
# 原始
BASE_URL = "https://huggingface.co/datasets/karpathy/climbmix-400b-shuffle/resolve/main"
# 修改为
BASE_URL = "https://hf-mirror.com/datasets/karpathy/climbmix-400b-shuffle/resolve/main"
```

运行时设置环境变量：
```powershell
$env:HF_ENDPOINT = "https://hf-mirror.com"
$env:NO_PROXY = "*"
```

### 2. wandb Unicode 编码错误

**问题：** Windows GBK 编码无法打印 Unicode 字符

**解决：** 禁用 wandb 并设置 UTF-8：
```powershell
$env:WANDB_MODE = "disabled"
$env:PYTHONIOENCODING = "utf-8"
```

### 3. torch.compile 缺少 C++ 编译器

**问题：** `RuntimeError: Compiler: cl is not found`

**解决：** 禁用 torch.compile：
```powershell
$env:TORCH_COMPILE_DISABLE = "1"
```

### 4. SFT NaN Loss（关键 Bug）

**问题：** SFT 训练第 3 步 loss 变成 NaN

**根因：** 部分对话（如 MMLU 多选题）的 assistant 回复在序列末尾，当截断到 `max_seq_len` 时，所有训练目标被截断，导致 batch 没有有效目标，`F.cross_entropy` 返回 NaN。

**解决：** 修改 `scripts/chat_sft.py:211` 的 `refill_buffer()`，过滤掉超长对话：
```python
def refill_buffer():
    nonlocal cursor, epoch
    while len(conv_buffer) < buffer_size:
        conversation = dataset[cursor]
        ids, mask = tokenizer.render_conversation(conversation)
        # 新增：跳过太长的对话，防止没有训练目标
        if len(ids) <= row_capacity:
            conv_buffer.append((ids, mask))
        cursor += ddp_world_size
        if cursor >= dataset_size:
            cursor = cursor % dataset_size
            epoch += 1
```

### 5. Windows signal.SIGALRM 不可用

**问题：** `engine.py` 使用 Unix 信号实现超时，Windows 不支持

**解决：** 修改 `nanochat/engine.py:25-44`，使用 threading.Timer 替代：
```python
@contextmanager
def timeout(duration, formula):
    if hasattr(signal, 'SIGALRM'):
        # Unix
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(duration)
        yield
        signal.alarm(0)
    else:
        # Windows - use threading
        import threading
        timer = threading.Timer(duration, lambda: (_ for _ in ()).throw(Exception(...)))
        timer.start()
        try:
            yield
        finally:
            timer.cancel()
```

---

## 训练流程

### Step 1: 下载预训练数据

**数据来源：** HuggingFace 上的 `karpathy/climbmix-400b-shuffle` 数据集

**相关代码：** `nanochat/dataset.py`

**下载命令：**
```powershell
python -m nanochat.dataset -n 8  # 下载 8 个训练 shard + 1 个验证 shard
```

**数据内容：**
- 原始数据：NVIDIA 的 `Nemotron-ClimbMix` 数据集（约 400B tokens）
- 格式：Parquet 文件，每个约 100MB，包含 `text` 列
- 内容：英文网页文本（新闻、百科、博客等）
- 预处理：打乱顺序后重新打包，参考 `dev/repackage_data_reference.py`

**数据统计：**
| 项目 | 值 |
|------|-----|
| 总 shard 数 | 6543（`MAX_SHARD = 6542`） |
| 已下载 | 9 个（8 训练 + 1 验证） |
| 每个 shard | 约 84 个 row_group，每个 1024 个文档 |
| 存储位置 | `02-nanochat/.cache/nanochat/base_data_climbmix/` |

**代码结构：**

| 函数 | 作用 |
|------|------|
| `download_single_file(index)` | 下载单个 shard，支持重试和断点续传 |
| `list_parquet_files()` | 列出本地所有 parquet 文件 |
| `parquets_iter_batched(split)` | 按 row_group 批量迭代数据，用于训练 |

**数据示例：**
```
Protect your business and employee's during flu season

Starting in 2005, the Center for Disease Control (CDC) established the
National Influenza Vaccination week "to highlight the importance of
continuing flu vaccination through the holiday season and beyond."
...
```

结果：下载 9 个 shard 到 `02-nanochat/.cache/nanochat/base_data_climbmix/`

### Step 2: 训练分词器

**相关代码：** `scripts/tok_train.py`、`nanochat/tokenizer.py`

**训练命令：**
```powershell
python -m scripts.tok_train --max-chars=2000000000
```

**参数说明：**

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--max-chars` | 2,000,000,000 | 训练使用的最大字符数（2B） |
| `--doc-cap` | 10,000 | 每个文档截断的最大字符数 |
| `--vocab-size` | 32,768 | 词表大小（2^15） |

**训练流程：**
1. 从预训练数据中迭代文本（最多 2B 字符）
2. 每个文档截断到 10,000 字符
3. 使用 `rustbpe` 库训练 BPE 分词器
4. 保存分词器和 token_bytes 映射

**分词器类型：** `RustBPETokenizer`（`nanochat/tokenizer.py:163`）
- 训练：使用 `rustbpe` 库（Rust 实现，速度快）
- 推理：使用 `tiktoken` 库（高效编码/解码）
- 风格：GPT-4 分词器风格

**特殊 token：**

| Token | 用途 |
|-------|------|
| `<\|bos\|>` | 文档开始标记 |
| `<\|user_start\|>` / `<\|user_end\|>` | 用户消息边界 |
| `<\|assistant_start\|>` / `<\|assistant_end\|>` | 助手消息边界 |
| `<\|python_start\|>` / `<\|python_end\|>` | Python 工具调用边界 |
| `<\|output_start\|>` / `<\|output_end\|>` | Python 输出边界 |

**输出文件：**
| 文件 | 作用 |
|------|------|
| `tokenizer.pkl` | 分词器模型（RustBPE 格式） |
| `token_bytes.pt` | token_id → 字节数映射（用于计算 bits per byte） |

**结果：**
- 耗时：55 秒
- 词表大小：32,768（含 8 个特殊 token）
- 输出：`02-nanochat/.cache/nanochat/tokenizer/`

### Step 3: 训练 Base 模型

**环境变量设置：**
```powershell
$env:NO_PROXY = "*"
$env:HTTP_PROXY = ""
$env:HTTPS_PROXY = ""
$env:WANDB_MODE = "disabled"
$env:PYTHONIOENCODING = "utf-8"
$env:TORCH_COMPILE_DISABLE = "1"
```

**训练命令：**
```powershell
python -m scripts.base_train --depth=4 --head-dim=64 --window-pattern=L --max-seq-len=256 --device-batch-size=4 --total-batch-size=4096 --eval-every=200 --eval-tokens=16384 --core-metric-every=-1 --sample-every=-1 --save-every=-1 --num-iterations=5000 --run=dummy
```

**参数说明：**

| 参数 | 值 | 说明 |
|------|-----|------|
| `--depth` | 4 | Transformer 层数，控制模型大小（4层≈36M参数） |
| `--head-dim` | 64 | 每个注意力头的维度 |
| `--window-pattern` | L | 滑动窗口注意力模式（L=全注意力，S=短窗口） |
| `--max-seq-len` | 256 | 最大序列长度，越大越耗内存 |
| `--device-batch-size` | 4 | 每个设备的 batch size |
| `--total-batch-size` | 4096 | 总 batch size（tokens），用于梯度累积 |
| `--eval-every` | 200 | 每 N 步评估一次验证集 |
| `--eval-tokens` | 16384 | 评估时使用的 token 数量 |
| `--core-metric-every` | -1 | 禁用 CORE 指标评估（节省时间） |
| `--sample-every` | -1 | 禁用采样生成（节省时间） |
| `--save-every` | -1 | 禁用中间 checkpoint 保存 |
| `--num-iterations` | 5000 | 训练总步数 |
| `--run` | dummy | wandb 运行名（dummy=禁用 wandb） |

**结果：**
- 耗时：约 20 分钟（5000 步）
- 模型：4层，36M 参数
- Val bpb：3.18 → 1.84
- 输出：`02-nanochat/.cache/nanochat/base_checkpoints/d4/`

### Step 4: SFT 微调

**下载身份对话数据：**
```powershell
Invoke-WebRequest -Uri "https://karpathy-public.s3.us-west-2.amazonaws.com/identity_conversations.jsonl" -OutFile "$env:NANOCHAT_BASE_DIR\identity_conversations.jsonl"
```

**训练命令：**
```powershell
python -m scripts.chat_sft --max-seq-len=256 --device-batch-size=4 --total-batch-size=4096 --eval-every=100 --eval-tokens=16384 --num-iterations=500 --embedding-lr=0.05 --unembedding-lr=0.001 --matrix-lr=0.005 --load-optimizer=0 --chatcore-every=-1
```

**参数说明：**

| 参数 | 值 | 说明 |
|------|-----|------|
| `--max-seq-len` | 256 | 最大序列长度，超过此长度的对话会被过滤 |
| `--device-batch-size` | 4 | 每个设备的 batch size |
| `--total-batch-size` | 4096 | 总 batch size（tokens） |
| `--eval-every` | 100 | 每 N 步评估一次验证集 |
| `--eval-tokens` | 16384 | 评估时使用的 token 数量 |
| `--num-iterations` | 500 | 训练总步数 |
| `--embedding-lr` | 0.05 | Embedding 层学习率（默认继承 pretrain） |
| `--unembedding-lr` | 0.001 | 输出层学习率（默认继承 pretrain） |
| `--matrix-lr` | 0.005 | Transformer 矩阵参数学习率（Muon 优化器） |
| `--load-optimizer` | 0 | 不继承 pretrain 的优化器状态（0=否，1=是） |
| `--chatcore-every` | -1 | 禁用 ChatCORE 指标评估（节省时间） |

**SFT 数据混合：**
- SmolTalk: 460K 通用对话
- CustomJSON: 1K 身份对话（加载 2 遍）
- MMLU: 100K 多选题（3 epochs）
- GSM8K: 8K 数学题（4 epochs）
- SimpleSpelling: 200K 单词拼写
- SpellingBee: 80K 字母计数

**结果：**
- 耗时：5 分钟（125 步）
- Loss：3.77 → 2.52
- Val bpb：2.00 → 1.58
- 输出：`02-nanochat/.cache/nanochat/chatsft_checkpoints/d4/`

---

## SFT 数据详解

**相关代码：** `scripts/chat_sft.py:165-173`、`tasks/` 目录

**数据加载入口：** `scripts/chat_sft.py:174`
```python
train_dataset = TaskMixture(train_tasks)
```

### 数据混合机制

**基类：** `tasks/common.py:Task`
- 所有任务继承自 `Task` 基类
- 核心方法：`get_example(index)` 返回 conversation dict
- 格式：`{"messages": [{"role": "user", "content": ...}, {"role": "assistant", "content": ...}]}`

**混合器：** `tasks/common.py:TaskMixture`
- 将多个任务合并为一个数据集
- 使用确定性 shuffle（seed=42）混合所有任务
- 重复传入同一任务可实现过采样

### 6 种 SFT 任务

#### 1. SmolTalk（`tasks/smoltalk.py`）— 460K 通用对话

**数据来源：** HuggingFace `HuggingFaceTB/smol-smoltalk`

**数据格式：**
```json
{
  "messages": [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "What is Python?"},
    {"role": "assistant", "content": "Python is a programming language..."}
  ]
}
```

**特点：**
- 多轮对话（user/assistant 交替）
- 可选 system 消息
- assistant content 为纯字符串
- 训练集 460K 行，测试集 24K 行

#### 2. CustomJSON（`tasks/customjson.py`）— 1K 身份对话

**数据来源：** `identity_conversations.jsonl`（Karpathy 提供）

**数据格式：**
```json
[
  {"role": "user", "content": "Who are you?"},
  {"role": "assistant", "content": "I am nanochat, a language model..."}
]
```

**特点：**
- 每行一个 JSON 数组（不是 JSON 对象）
- 用于给模型注入身份信息
- 在 train_tasks 中加载 2 遍（过采样）
- assistant content 为纯字符串

#### 3. MMLU（`tasks/mmlu.py`）— 100K 多选题

**数据来源：** HuggingFace `cais/mmlu`

**数据格式：**
```json
{
  "messages": [
    {"role": "user", "content": "Multiple Choice question: What is 2+2?\n- 3=A\n- 4=B\n- 5=C\n- 6=D\n\nRespond only with the letter of the correct answer."},
    {"role": "assistant", "content": "B"}
  ],
  "subject": "elementary_mathematics",
  "letters": ["A","B","C","D"]
}
```

**特点：**
- 57 个学科类别
- 选项格式：`- {choice}={letter}`（小模型更易绑定）
- assistant content 为单个字母（A/B/C/D）
- 训练集 auxiliary_train 约 100K 行

#### 4. GSM8K（`tasks/gsm8k.py`）— 8K 数学题（含 tool call）

**数据来源：** HuggingFace `openai/gsm8k`

**数据格式：**
```json
{
  "messages": [
    {"role": "user", "content": "Weng earns $12 an hour for babysitting..."},
    {"role": "assistant", "content": [
      {"type": "text", "text": "Weng earns 12/60 = $"},
      {"type": "python", "text": "12/60"},
      {"type": "python_output", "text": "0.2"},
      {"type": "text", "text": "0.2 per minute..."},
      {"type": "python", "text": "0.2*50"},
      {"type": "python_output", "text": "10"},
      {"type": "text", "text": "10.\n#### 10"}
    ]}
  ]
}
```

**特点：**
- assistant content 为 parts 列表（不是纯字符串）
- 3 种 part type：`text`、`python`、`python_output`
- 原始数据中 `<<expr=result>>` 格式被解析为结构化 parts
- 答案标记：`#### {number}`

#### 5. SimpleSpelling（`tasks/spellingbee.py:233`）— 200K 单词拼写

**数据来源：** 本地生成（`words_alpha.txt`，370K 英文单词）

**数据格式：**
```json
{
  "messages": [
    {"role": "user", "content": "Spell the word: apple"},
    {"role": "assistant", "content": "apple:a,p,p,l,e"}
  ]
}
```

**特点：**
- 简化任务，专门练习 token → 字符序列映射
- 从 370K 单词中随机采样
- assistant content 为纯字符串
- 格式：`{word}:{letter1},{letter2},...`

#### 6. SpellingBee（`tasks/spellingbee.py:115`）— 80K 字母计数（含 tool call）

**数据来源：** 本地生成（`words_alpha.txt`）

**数据格式：**
```json
{
  "messages": [
    {"role": "user", "content": "How many r are in strawberry?"},
    {"role": "assistant", "content": [
      {"type": "text", "text": "We are asked to find the number 'r' in the word 'strawberry'..."},
      {"type": "python", "text": "'strawberry'.count('r')"},
      {"type": "python_output", "text": "3"},
      {"type": "text", "text": "\n\nPython gives us 3.\n\nMy final answer is:\n\n#### 3"}
    ]}
  ]
}
```

**特点：**
- 30+ 种用户提问模板（含中日韩法德西等语言）
- 90% 从单词中选字母，10% 随机字母
- assistant 先手动逐字母计数，再用 Python 验证
- 包含完整的推理过程

### Content 类型总结

| 类型 | 格式 | 示例任务 |
|-----|------|---------|
| 纯字符串 | `"content": "text"` | SmolTalk, CustomJSON, MMLU, SimpleSpelling |
| parts 列表 | `"content": [{"type": ..., "text": ...}]` | GSM8K, SpellingBee |

Parts 中的 type：
- `text`：普通文本
- `python`：Python 表达式（tool call）
- `python_output`：Python 执行结果

### 训练目标的 Mask 设计

通过 `tokenizer.render_conversation()`（`tokenizer.py:266`）：

| 内容类型 | mask | 说明 |
|---------|------|------|
| user 消息 | 0 | 不训练 |
| assistant 文本 | 1 | 训练目标 |
| assistant python 调用 | 1 | 训练目标 |
| python_output（工具返回） | 0 | 不训练，运行时由环境产生 |

**效果：** 模型只在 assistant 的回复内容（不含 python_output）上计算 loss。

---

## 模型测试结果

**环境变量设置：**
```powershell
$env:NO_PROXY = "*"
$env:HTTP_PROXY = ""
$env:HTTPS_PROXY = ""
$env:ALL_PROXY = ""
$env:HF_HOME = "E:\Project\LLM\LearnLLM\02-nanochat\.cache\huggingface"
$env:NANOCHAT_BASE_DIR = "E:\Project\LLM\LearnLLM\02-nanochat\.cache\nanochat"
$env:WANDB_MODE = "disabled"
$env:PYTHONIOENCODING = "utf-8"
$env:TORCH_COMPILE_DISABLE = "1"
$env:HF_ENDPOINT = "https://hf-mirror.com"
```

**测试命令：**
```powershell
# 单次提问
python -m scripts.chat_cli -p "Who are you?" -i sft

# 交互模式（可多轮对话）
python -m scripts.chat_cli -i sft

# 其他测试示例
python -m scripts.chat_cli -p "What is the capital of France?" -i sft
python -m scripts.chat_cli -p "Spell the word: hello" -i sft
python -m scripts.chat_cli -p "What is 2+2?" -i sft
```

**参数说明：**
| 参数 | 说明 |
|------|------|
| `-p` | 单次提问模式，输出后退出 |
| `-i sft` | 使用 SFT 模型（而非 base 模型） |
| 无 `-p` | 交互模式，可持续对话 |

**结果：** 模型效果很差，所有问题都生成 SpellingBee 格式的随机回答。

**原因：**
- 模型太小（4层，36M 参数）
- Base 训练不足（500 步 vs 推荐 5000 步）
- SFT 训练不足（125 步 vs 推荐 1500 步）

---

## 改进建议

| 参数 | 当前值 | 推荐值 | 预计耗时 |
|------|--------|--------|----------|
| --depth | 4 | 6 | +50% |
| Base 步数 | 500 | 5000 | +10x |
| SFT 步数 | 125 | 1500 | +12x |
| 总训练时间 | ~25 分钟 | ~2 小时 | - |

**完整训练命令（参考 runcpu.sh）：**
```powershell
# Base 模型
python -m scripts.base_train --depth=6 --head-dim=64 --window-pattern=L --max-seq-len=512 --device-batch-size=32 --total-batch-size=16384 --eval-every=100 --eval-tokens=524288 --core-metric-every=-1 --sample-every=100 --num-iterations=5000

# SFT
python -m scripts.chat_sft --max-seq-len=512 --device-batch-size=32 --total-batch-size=16384 --eval-every=200 --eval-tokens=524288 --num-iterations=1500
```

---

## 修改的文件清单

| 文件 | 修改内容 |
|------|----------|
| `nanochat/dataset.py:23` | BASE_URL 改为 hf-mirror.com |
| `scripts/chat_sft.py:211-221` | 添加超长对话过滤 |
| `nanochat/engine.py:25-44` | Windows 兼容的 timeout 实现 |

---

## 总结

在 Windows CPU 环境下成功运行 nanochat 全流程（数据下载 → 分词器训练 → Base 训练 → SFT 微调 → 模型测试），但受限于硬件（CPU + 小模型 + 少量训练），模型效果很差。主要收获：

1. 理解了 nanochat 的代码结构和训练流程
2. 发现并修复了 SFT 的 NaN loss bug（超长对话截断问题）
3. 解决了 Windows 兼容性问题（signal、编码、网络）
4. 验证了 CPU 路径的可行性，但效果远不如 GPU
