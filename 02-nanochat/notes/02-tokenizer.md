# Tokenizer — BPE 分词器详解

> nanochat Stage 1：训练 BPE 分词器。
>
> 源码：`nanochat/tokenizer.py`

---

## 什么是 Tokenizer

将原始文本转换为模型可以处理的数字序列（token ids）。

```
"Hello world" → [15496, 995]
```

nanochat 提供两套实现：

| 实现 | 训练 | 推理 | 特点 |
|------|------|------|------|
| `HuggingFaceTokenizer` | HuggingFace tokenizers | 同上 | 功能完整但 API 复杂 |
| `RustBPETokenizer` | rustbpe (Rust) | tiktoken (Rust) | 训练可靠、推理快，**nanochat 默认** |

---

## BPE（Byte Pair Encoding）算法

### 核心思想

从单个字节（256 个）出发，不断合并出现频率最高的相邻 pair，直到词表达到目标大小。

```
初始词表：256 个字节（0x00 ~ 0xFF）

循环：
  1. 统计所有相邻 token pair 的出现频次
  2. 找到频次最高的 pair
  3. 将该 pair 合并为一个新 token，加入词表
  4. 重复直到词表大小 = 目标大小
```

### 算法示例

```
原始："aaabbc"
字节：[a, a, a, b, b, c]

第1轮：最高频 pair = (a,a) → 合并为 "aa" (新 token id=256)
结果：[aa, a, b, b, c]

第2轮：最高频 pair = (b,b) → 合并为 "bb" (新 token id=257)
结果：[aa, a, bb, c]

...
```

---

## 编码过程详解（以 "Hello, world! 2024" 为例）

编码分为三个阶段：**预分词 → 字节编码 → BPE 合并**。

### Step 1：预分词 — Split Pattern 正则切分

`SPLIT_PATTERN` 将文本切成独立的 chunk，BPE 合并只在 chunk 内部进行，绝不跨 chunk。

```
SPLIT_PATTERN 正则各分支:
┌──────────────────┬─────────────────────────────────────────────────┐
│ 正则分支          │ 匹配内容                                        │
├──────────────────┼─────────────────────────────────────────────────┤
│ '(?i:[sdmt]|ll…) │ 英文缩写: 's, 'd, 'll, 're...                  │
│ \p{L}+           │ 连续 Unicode 字母                                │
│ \p{N}{1,2}       │ 1~2 位连续数字（GPT-4 用的是 {1,3}）             │
│ [^\s\p{L}\p{N}]+ │ 连续符号/标点                                    │
│ \s+              │ 空白字符                                         │
└──────────────────┴─────────────────────────────────────────────────┘
```

**示例切分**：

```
输入: "Hello, world! 2024"

切分结果（8 个独立 chunk）:
  chunk 1: "Hello"     ← 匹配 \p{L}+
  chunk 2: ","         ← 匹配标点分支
  chunk 3: " "         ← 匹配 \s+
  chunk 4: "world"     ← 匹配 \p{L}+
  chunk 5: "!"         ← 匹配标点分支
  chunk 6: " "         ← 匹配 \s+
  chunk 7: "20"        ← 匹配 \p{N}{1,2}
  chunk 8: "24"        ← 匹配 \p{N}{1,2}
```

> **为什么数字只用 {1,2} 而非 GPT-4 的 {1,3}？**
> 小词表（32K）下，{1,3} 会产生太多数字 token 变体（000~999），浪费 token 空间。
> 经实验验证，2 是 32K 词表的最佳选择。

### Step 2：字节编码 — 字符 → UTF-8 字节

```
"Hello" → ['H','e','l','l','o'] → [72, 101, 108, 108, 111]
","     → [',']                  → [44]
" "     → [' ']                  → [32]
"world" → ['w','o','r','l','d']  → [119, 111, 114, 108, 100]
"!"     → ['!']                  → [33]
"20"    → ['2','0']              → [50, 48]
"24"    → ['2','4']              → [50, 52]

初始 token ids（17 个字节 token）:
  [72, 101, 108, 108, 111, 44, 32, 119, 111, 114, 108, 100, 33, 32, 50, 48, 50, 52]
```

对于中文：`"你好"` → UTF-8 `[228,189,160, 229,165,189]` → 6 个字节 token（训练后高频字会合并为 1 个 token）。

### Step 3：BPE 贪婪合并 — 反复替换最高优先级的相邻对

使用训练好的 `mergeable_ranks`（`{bytes → rank}`，rank 越小优先级越高），贪婪合并。

```
以 chunk "Hello" 为例（初始 5 个字节: [72, 101, 108, 108, 111]）:

假设 mergeable_ranks:
  b'll'    → rank=256 (最小)
  b'lo'    → rank=320
  b'el'    → rank=415
  b'llo'   → rank=720
  b'ello'  → rank=840
  b'Hello' → rank=1050

第 1 轮：扫描相邻对
  (72,101)→b"He"  不存在
  (101,108)→b"el" rank=415
  (108,108)→b"ll" rank=256  ← 最小！合并 "l"+"l"→"ll"(id=256)
  (108,111)→b"lo" rank=320
  → 结果: [72, 101, 256, 111]   (4 tokens)

第 2 轮：
  (72,101)→b"He" 不存在
  (101,256)→b"ell" rank=840
  (256,111)→b"llo" rank=720  ← 最小！合并 "ll"+"o"→"llo"(id=720)
  → 结果: [72, 101, 720]   (3 tokens)

第 3 轮：
  (72,101)→b"He" 不存在
  (101,720)→b"ello" rank=840  ← 唯一可选，合并
  → 结果: [72, 840]   (2 tokens)

第 4 轮：
  (72,840)→b"Hello" rank=1050  ← 合并
  → 结果: [1050]   (1 token!)

第 5 轮：只剩 1 个 token → 结束
```

**最终结果**：

```
"Hello, world! 2024"
  → [1050, 44, 32, 1200, 33, 32, 490, 491]
     Hello  ,  spa world !   spa  20   24

原始 17 字符 → 8 个 token，压缩比 > 50%
```

### 完整编码链路（带特殊 token）

```python
tokenizer.encode("Hello, world! 2024",
                  prepend="<|bos|>",
                  append="<|assistant_start|>")

# 内部流程:
# 1. "<|bos|>"             → encode_special → 65527
# 2. "Hello, world! 2024"  → encode_ordinary → [1050,44,32,1200,33,32,490,491]
# 3. "<|assistant_start|>" → encode_special → 65530

# 最终:
ids = [65527, 1050, 44, 32, 1200, 33, 32, 490, 491, 65530]
```

---

## 词表大小决定了什么

词表大小 `vocab_size` 是 LLM 中最重要的超参数之一，直接影响三个维度：

### 1. Embedding 矩阵参数量

模型输入 embedding 和输出 LM Head 共享权重 `[vocab_size × d_model]`：

| vocab_size | d_model=1024 时的参数量 | 对 150M 小模型的影响 |
|------------|------------------------|---------------------|
| 256 | 262K | 可忽略 |
| 32,768 (32K) | 33.5M | 占 22% |
| 65,536 (64K) | 67.1M | 占 45% |
| 128,000 (128K) | 131M | 占 87%（不合理） |

**对小模型来说，词表不能太大**——否则大部分参数都在 embedding 层而非 transformer 层。

### 2. 文本压缩率

```
"The transformer architecture revolutionized natural language processing."

vocab_size = 256:   ≈ 62 tokens  (每个字符 ~2 字节)
vocab_size = 32K:   ≈ 12 tokens
vocab_size = 64K:   ≈ 9 tokens
vocab_size = 128K:  ≈ 7 tokens
```

更大的词表 = 更短的序列 = 同样的上下文窗口覆盖更多文本。但每个 token 携带更多信息，学习难度增加。

### 3. 计算开销

序列越短 → 注意力矩阵越小 → KV Cache 显存越小：

```
1000 字符文本:
  vocab_size=256 → seq_len≈1000（几乎无压缩）
  vocab_size=32K → seq_len≈350
  vocab_size=65K → seq_len≈280
```

### 不同模型的选择

| 模型 | vocab_size | 考量 |
|------|-----------|------|
| GPT-2 | 50,257 | 早期探索 |
| GPT-3/4 | 100,000+ | 大模型不惧 embedding 开销 |
| Llama 2 | 32,000 | 平衡压缩率和效率 |
| Llama 3 | 128,000 | 提升多语言效率 |
| Qwen | ~152,000 | 中文优先，需更多汉字 token |
| **nanochat** | **65,536** | 平衡压缩率和模型大小 |

---

## 词表全景：65,536 个 token 里有什么

```
vocab_size = 65536 (64K)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
│█████████████████████████████████████████████████████████████│
│█ 256 字节基座 ██  65271 个 BPE 子词 token  ██ 9 个特殊 token █│
│█████████████████████████████████████████████████████████████│
│                                                            │
│  id 0~255:       单字节 token (0x00~0xFF)                  │
│  id 256~65526:   BPE 合并产生的子词 token                   │
│  id 65527~65535: 特殊 token                                │
└────────────────────────────────────────────────────────────┘
```

### 第一层：256 个字节级 token（所有词表的基石）

```
id │ 字节  │ 含义
───┼───────┼──────
0  │ 0x00  │ NUL（基本不会出现）
...
32 │ 0x20  │ 空格 ' '
48 │ 0x30  │ '0'
65 │ 0x41  │ 'A'
97 │ 0x61  │ 'a'
...
128│ 0x80  │ UTF-8 多字节序列前导字节（中文首字节在此区域）
...
255│ 0xFF  │
```

这 256 个 token **永远不会被移除**，保证任何 UTF-8 文本都能通过字节序列兜底编码。

### 第二层：BPE 合并产生的子词 token（主体，~65K）

#### 类型 A：完整高频词

```
"the"  → 1 个 token
"and"  → 1 个 token
"of"   → 1 个 token
```

#### 类型 B：常见子词/词缀

```
"ing"   → 动名词后缀，如 "running" = "runn" + "ing"
"tion"  → 名词后缀，如 "attention" = "atten" + "tion"
"pre"   → 前缀
"ly"    → 副词后缀
```

这让模型可以用组合方式处理未见过的词：
```
"unbelievably" → "un" + "believ" + "ably"（3 tokens）
```

#### 类型 C：多语言字符/子词

```
"你"     → 1 个 token（高频汉字，3 字节 → 1 token）
"好"     → 1 个 token
"世界"   → 1 个 token（常见中文词组）
"用户"   → 1 个 token
"こんにちは" → 1 个 token（如果语料足够多）
```

> 如果训练语料以英文为主，中文 token 会比较少，一个中文句子可能每个字需要 1~3 个 token。

#### 类型 D：数字组合

```
"20"    → 1 个 token
"24"    → 1 个 token
"2024"  → 1 个 token（如果 "20"+"24" 这个 pair 频率够高）
"100"   → 1 个 token
```

SPLIT_PATTERN 用 `{1,2}` 限制数字组大小，但 BPE 后续合并仍可产生更长的数字 token。

#### 类型 E：代码片段

```
"def "     → Python 关键字
"import "  → Python 关键字
"    "     → 4 空格缩进
"return "  → Python 关键字
```

代码语料会产生大量编程语言特有的 token。

#### 类型 F：ByteLevel "怪异" token

不可见字节被映射到可见 Unicode 字符：
```
"À€" → 实际上代表某个 UTF-8 非法字节序列
```

这些通常对应训练语料中的二进制数据或乱码，实际使用中很少出现。

### 第三层：9 个特殊 token

```python
SPECIAL_TOKENS = [
    "<|bos|>",              # 65527 — 文档/序列开始标记
    "<|user_start|>",       # 65528 — 用户消息起始
    "<|user_end|>",         # 65529 — 用户消息结束
    "<|assistant_start|>",  # 65530 — 助手消息起始
    "<|assistant_end|>",    # 65531 — 助手消息结束
    "<|python_start|>",     # 65532 — Python 代码块起始
    "<|python_end|>",       # 65533 — Python 代码块结束
    "<|output_start|>",     # 65534 — 执行输出起始
    "<|output_end|>",       # 65535 — 执行输出结束
]
```

特殊 token 的 id 分配在词表末尾，不参与 BPE 训练。通过 `encode_special()` 做精确匹配查找。

---

## 对话渲染（render_conversation）

SFT 训练时将结构化对话转为 token 序列和 mask。

### 渲染格式

```
输入对话:
  user:      "What is 2+2?"
  assistant: "2+2 = 4"

渲染结果:
  <|bos|> <|user_start|> What is 2+2? <|user_end|>
  <|assistant_start|> 2+2 = 4 <|assistant_end|>
```

### Mask 规则

| 内容 | mask | 原因 |
|------|------|------|
| `<|bos|>` | 0 | 系统标记，不需要预测 |
| `<|user_start|>`, `<|user_end|>` | 0 | 对话结构标记 |
| 用户消息文本 | 0 | 用户输入，不是模型生成的 |
| `<|assistant_start|>` | 0 | 结构标记 |
| 助手消息文本 | **1** | 需要让模型学会生成的内容 |
| `<|assistant_end|>` | **1** | 让模型学会何时停止 |
| `<|python_start|>` `<|python_end|>` | **1** | Python 代码是模型生成的 |
| Python 输出内容 | 0 | 外部工具返回的，不是模型生成的 |

### 可视化调试

```python
ids, mask = tokenizer.render_conversation(conv)
print(tokenizer.visualize_tokenization(ids, mask, with_token_id=True))
# 绿色 = mask=1（计算 loss），红色 = mask=0（不计算 loss）
```

---

## 训练流程代码解析

```python
@classmethod
def train_from_iterator(cls, text_iterator, vocab_size):
    # 1) 创建 BPE 模型
    tokenizer = HFTokenizer(BPE(
        byte_fallback=True,   # 未知字符退回字节表示，保证无 UNK
        unk_token=None,
        fuse_unk=False,
    ))

    # 2) 预分词器：正则切分 → ByteLevel 编码
    tokenizer.pre_tokenizer = pre_tokenizers.Sequence([
        pre_tokenizers.Split(Regex(SPLIT_PATTERN),
                             behavior="isolated",  # 每个匹配独立成组
                             invert=False),        # 按匹配内容切分
        pre_tokenizers.ByteLevel(add_prefix_space=False),
    ])

    # 3) 解码器（与 ByteLevel 预分词配对使用）
    tokenizer.decoder = decoders.ByteLevel()

    # 4) BPE 训练
    trainer = BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=0,    # 不设最小频率阈值
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        special_tokens=SPECIAL_TOKENS,
    )
    tokenizer.train_from_iterator(text_iterator, trainer)
    return cls(tokenizer)
```

---

## nanochat 的 Tokenizer 配置

| 参数 | 值 | 说明 |
|------|------|------|
| 词表大小 | 65,536 (64K) | 平衡压缩率和模型开销 |
| 算法 | BPE | GPT-4 风格 |
| 数字分组 | `\p{N}{1,2}` | 比 GPT-4 的 `{1,3}` 更省 token 空间 |
| 特殊 token | 9 个 | 对话 + 工具调用标记 |
| 字节回退 | `byte_fallback=True` | 保证无 UNK token |
| 训练库 | rustbpe (Rust) | 比 HuggingFace 的训练更可靠 |
| 推理库 | tiktoken (Rust) | 支持多线程批量编码 |

---

## 与 GPT-2/4 Tokenizer 的区别

| 对比 | GPT-2 | GPT-4 | nanochat |
|------|-------|-------|----------|
| 词表大小 | 50,257 | ~100,000 | 65,536 |
| 数字分组 | `{1,3}` | `{1,3}` | `{1,2}` |
| 特殊 token | 1 个 | 多个 | 9 个 |
| 基础算法 | Byte-level BPE | Byte-level BPE | Byte-level BPE |
| 推理 | tiktoken | tiktoken | tiktoken |
| 训练 | ? | ? | rustbpe |

---

## 关键设计决策

- [x] 词表大小的选择：65K，平衡 embedding 开销和压缩效率
- [x] SPLIT_PATTERN 数字分组 `{1,2}` vs `{1,3}`：节省小词表空间
- [x] 特殊 token 定义：9 个，覆盖对话和工具调用
- [x] 字节级 fallback（`byte_fallback=True`）：处理任何 UTF-8 文本
- [x] 两套实现双轨制：HuggingFace 做实验，rustbpe+tiktoken 做生产
- [x] 训练数据的选择影响词表分布（英文为主 vs 中文为主）

---

## 核心代码文件

| 文件 | 作用 |
|------|------|
| `nanochat/tokenizer.py` | 分词器主实现（HuggingFaceTokenizer + RustBPETokenizer） |
| `nanochat/tok_train.py` | 分词器训练脚本 |
| `nanochat/tok_eval.py` | 分词器评估脚本 |
