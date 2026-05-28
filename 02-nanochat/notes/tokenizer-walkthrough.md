# nanochat/tokenizer.py 源码逐行详解

> 源文件：https://github.com/karpathy/nanochat/blob/master/nanochat/tokenizer.py
>
> 本文件实现了 nanochat 的**分词器（Tokenizer）**，采用 GPT-4 风格的 BPE（Byte Pair Encoding）分词。
> 提供两套实现：HuggingFace Tokenizer（训练+推理）和 RustBPE+tiktoken 组合（训练用 Rust 加速，推理用 tiktoken 高效执行）。

---

## 文件头注释

```python
"""
BPE Tokenizer in the style of GPT-4.

Two implementations are available:
1) HuggingFace Tokenizer that can do both training and inference but is really confusing
2) Our own RustBPE Tokenizer for training and tiktoken for efficient inference
"""
```

> 说明了两套实现的取舍：
> - **HuggingFace Tokenizer**：功能完整但 API 复杂，适合实验
> - **RustBPE + tiktoken**：
>   - 训练阶段用 `rustbpe`（Rust 实现，速度快）
>   - 推理阶段用 `tiktoken`（OpenAI 开源，极高效）
>   - 生产代码使用此方案

---

## 第一部分：导入与全局常量

```python
import os
import copy
from functools import lru_cache
```

> - `os`：文件路径操作（读写 tokenizer 文件）
> - `copy`：深拷贝对话数据，避免修改原始数据
> - `lru_cache`：最近最少使用缓存装饰器，用于缓存特殊 token 的编码结果

---

### SPECIAL_TOKENS — 特殊 token 列表

```python
SPECIAL_TOKENS = [
    "<|bos|>",
    "<|user_start|>",
    "<|user_end|>",
    "<|assistant_start|>",
    "<|assistant_end|>",
    "<|python_start|>",
    "<|python_end|>",
    "<|output_start|>",
    "<|output_end|>",
]
```

> nanochat 定义了 9 个特殊 token，构成对话格式的骨架：
>
> | Token | 作用 |
> |-------|------|
> | `<\|bos\|>` | Beginning of Sequence，每段对话的开头 |
> | `<\|user_start\|>` | 用户消息开始 |
> | `<\|user_end\|>` | 用户消息结束 |
> | `<\|assistant_start\|>` | 助手消息开始 |
> | `<\|assistant_end\|>` | 助手消息结束 |
> | `<\|python_start\|>` | 助手调用 Python REPL 工具的代码开始 |
> | `<\|python_end\|>` | Python 代码结束 |
> | `<\|output_start\|>` | Python 执行结果开始 |
> | `<\|output_end\|>` | Python 执行结果结束 |
>
> **对话渲染示例**：
> ```
> <|bos|>
> <|user_start|>你好<|user_end|>
> <|assistant_start|>你好！有什么可以帮你的？<|assistant_end|>
> ```
>
> SFT 训练时只对 `<|assistant_start|>...<|assistant_end|>` 之间的内容计算 loss（mask=1），其余 mask=0。

---

### SPLIT_PATTERN — 预分词正则

```python
SPLIT_PATTERN = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,2}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""
```

> GPT-4 风格的文本预分词正则，在 BPE 合并之前先将文本切割成合理的"单元"：
>
> **各子模式解析**：
>
> | 子模式 | 匹配内容 | 举例 |
> |--------|---------|------|
> | `'(?i:[sdmt]\|ll\|ve\|re)` | 英文缩写后缀 | `'s`, `'t`, `'ll`, `'ve`, `'re` |
> | `[^\r\n\p{L}\p{N}]?+\p{L}+` | 单词（含可选前缀标点） | ` hello`, `world` |
> | `\p{N}{1,2}` | **1~2 位数字**（GPT-4 原版是 1~3 位） | `12`, `7` |
> | ` ?[^\s\p{L}\p{N}]++[\r\n]*` | 标点符号串 | `!!`, `, ` |
> | `\s*[\r\n]` | 换行符 | `\n`, `\r\n` |
> | `\s+(?!\S)` | 尾部空格 | 行末空白 |
> | `\s+` | 剩余空白 | 空格序列 |
>
> **与 GPT-4 的关键区别**：数字正则从 `\p{N}{1,3}` 改为 `\p{N}{1,2}`（最多 2 位数字一个 token）。
> - 原因：对于 32K 词表的小模型，3 位数字太浪费 token 空间
> - 作者验证：2 位是最优值（1 位略差，3 位更差）



---

## 第二部分：HuggingFaceTokenizer

```python
from tokenizers import Tokenizer as HFTokenizer
from tokenizers import pre_tokenizers, decoders, Regex
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
```

> 导入 HuggingFace `tokenizers` 库（底层用 Rust 实现）：
> - `HFTokenizer`：分词器主类
> - `pre_tokenizers`：预分词器（在 BPE 前切割文本）
> - `decoders`：解码器（token IDs → 文本）
> - `Regex`：HuggingFace 要求正则必须用此包装
> - `BPE`：BPE 模型
> - `BpeTrainer`：BPE 训练器

---

### `__init__`

```python
class HuggingFaceTokenizer:
    """Light wrapper around HuggingFace Tokenizer for some utilities"""

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
```

> 轻量包装类，持有一个 HuggingFace `Tokenizer` 对象，统一对外暴露接口。

---

### `from_pretrained` — 从 HuggingFace Hub 加载

```python
    @classmethod
    def from_pretrained(cls, hf_path):
        tokenizer = HFTokenizer.from_pretrained(hf_path)
        return cls(tokenizer)
```

> 从 HuggingFace Hub 加载预训练分词器（如 `"gpt2"`），方便对比实验。

---

### `from_directory` — 从本地磁盘加载

```python
    @classmethod
    def from_directory(cls, tokenizer_dir):
        tokenizer_path = os.path.join(tokenizer_dir, "tokenizer.json")
        tokenizer = HFTokenizer.from_file(tokenizer_path)
        return cls(tokenizer)
```

> 从本地 `tokenizer.json` 加载，是训练后的正常使用路径。
> 文件由 `save()` 方法生成，路径约定为 `{base_dir}/tokenizer/tokenizer.json`。

---

### `train_from_iterator` — 训练分词器

```python
    @classmethod
    def train_from_iterator(cls, text_iterator, vocab_size):
        tokenizer = HFTokenizer(BPE(
            byte_fallback=True,
            unk_token=None,
            fuse_unk=False,
        ))
```

> 创建 BPE 模型：
> - `byte_fallback=True`：**关键参数**。当某个字节序列不在词表中时，回退到单字节表示。保证任何输入都能被编码（无需 `[UNK]` token）
> - `unk_token=None`：不设置未知 token（byte_fallback 保证无需 UNK）
> - `fuse_unk=False`：不合并连续的 UNK（与 byte_fallback 配合，此处无意义）

```python
        tokenizer.normalizer = None
```

> 不做任何文本归一化（如大小写转换、Unicode 规范化）。GPT-4 风格分词器不做归一化，保留原始文本。

```python
        gpt4_split_regex = Regex(SPLIT_PATTERN)
        tokenizer.pre_tokenizer = pre_tokenizers.Sequence([
            pre_tokenizers.Split(pattern=gpt4_split_regex, behavior="isolated", invert=False),
            pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False)
        ])
```

> **两阶段预分词**（HuggingFace 要求包在 `Regex()` 中）：
>
> 1. **Split**：用 `SPLIT_PATTERN` 将文本切割成初步单元（单词、数字、标点等），`behavior="isolated"` 表示匹配的子串作为独立单元
> 2. **ByteLevel**：将每个初步单元的每个字节映射到可打印的 Unicode 字符（GPT-2 的经典技巧）
>    - 例如：字节 0x00 → `Ā`，空格 → `Ġ`
>    - 保证词表中只含可打印字符，方便 tokenizer.json 序列化
>    - `add_prefix_space=False`：不自动在开头加空格

```python
        tokenizer.decoder = decoders.ByteLevel()
```

> ByteLevel 解码器：与 ByteLevel 预分词器配对，解码时把可打印 Unicode 字符映射回原始字节。

```python
        trainer = BpeTrainer(
            vocab_size=vocab_size,
            show_progress=True,
            min_frequency=0,
            initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
            special_tokens=SPECIAL_TOKENS,
        )
```

> BPE 训练器配置：
> - `vocab_size`：目标词表大小（nanochat 默认 32768）
> - `show_progress=True`：显示训练进度条
> - `min_frequency=0`：不设置最低频率阈值（所有 token pair 都参与合并候选）
> - `initial_alphabet=ByteLevel.alphabet()`：初始词表包含所有 256 个字节的 ByteLevel 表示（保证 byte_fallback 有效）
> - `special_tokens=SPECIAL_TOKENS`：将 9 个特殊 token 加入词表（不参与 BPE 合并，直接插入）

```python
        tokenizer.train_from_iterator(text_iterator, trainer)
        return cls(tokenizer)
```

> 从文本迭代器中训练，返回包装好的分词器实例。

---

### 工具方法

```python
    def get_vocab_size(self):
        return self.tokenizer.get_vocab_size()
```

> 返回词表大小（包含特殊 token）。

```python
    def get_special_tokens(self):
        special_tokens_map = self.tokenizer.get_added_tokens_decoder()
        special_tokens = [w.content for w in special_tokens_map.values()]
        return special_tokens
```

> 获取所有特殊 token 的字符串列表。`get_added_tokens_decoder()` 返回 `{id: AddedToken}` 字典，`.content` 取 token 文本。

```python
    def id_to_token(self, id):
        return self.tokenizer.id_to_token(id)
```

> 将 token ID 转换回 token 字符串（注意：这是 ByteLevel 编码后的字符串，不是原始文本）。

---

### `_encode_one` — 编码单个字符串

```python
    def _encode_one(self, text, prepend=None, append=None, num_threads=None):
        assert isinstance(text, str)
        ids = []
        if prepend is not None:
            prepend_id = prepend if isinstance(prepend, int) else self.encode_special(prepend)
            ids.append(prepend_id)
        ids.extend(self.tokenizer.encode(text, add_special_tokens=False).ids)
        if append is not None:
            append_id = append if isinstance(append, int) else self.encode_special(append)
            ids.append(append_id)
        return ids
```

> 编码一段文本，可选在头/尾插入特殊 token：
> - `prepend`/`append`：可以是 token ID（int）或特殊 token 字符串，自动解析
> - `add_special_tokens=False`：不自动添加任何特殊 token（由调用方显式控制）
> - `num_threads`：忽略（此参数只在 RustBPETokenizer 中有用）

```python
    def encode_special(self, text):
        return self.tokenizer.token_to_id(text)
```

> 精确匹配特殊 token，返回其 ID。与普通 `encode()` 的区别：special token 不经过 BPE 分词，直接查表。

```python
    def get_bos_token_id(self):
        bos = self.encode_special("<|bos|>")
        if bos is None:
            bos = self.encode_special("<|endoftext|>")
        assert bos is not None, "Failed to find BOS token in tokenizer"
        return bos
```

> 获取 BOS token ID，兼容两种命名：
> 1. 优先查找 `<|bos|>`（nanochat 自训练分词器）
> 2. 回退到 `<|endoftext|>`（GPT-2 风格 HuggingFace 分词器）
> 3. 两者都找不到则报错

```python
    def encode(self, text, *args, **kwargs):
        if isinstance(text, str):
            return self._encode_one(text, *args, **kwargs)
        elif isinstance(text, list):
            return [self._encode_one(t, *args, **kwargs) for t in text]
        else:
            raise ValueError(f"Invalid input type: {type(text)}")
```

> 批量或单个编码，自动分发：字符串 → `_encode_one`，列表 → 逐元素编码。

```python
    def __call__(self, *args, **kwargs):
        return self.encode(*args, **kwargs)
```

> 使分词器可调用：`tokenizer("hello")` 等价于 `tokenizer.encode("hello")`。

```python
    def decode(self, ids):
        return self.tokenizer.decode(ids, skip_special_tokens=False)
```

> 将 token IDs 解码回文本，`skip_special_tokens=False` 保留所有特殊 token（调试时方便看到对话结构）。

```python
    def save(self, tokenizer_dir):
        os.makedirs(tokenizer_dir, exist_ok=True)
        tokenizer_path = os.path.join(tokenizer_dir, "tokenizer.json")
        self.tokenizer.save(tokenizer_path)
        print(f"Saved tokenizer to {tokenizer_path}")
```

> 保存为 `tokenizer.json`（HuggingFace 标准格式，可跨语言读取）。



---

## 第三部分：RustBPETokenizer

```python
import pickle
import rustbpe
import tiktoken
```

> - `pickle`：Python 对象序列化，用于保存/加载 tiktoken 的 `Encoding` 对象
> - `rustbpe`：nanochat 自实现的 Rust BPE 训练库（速度快，支持 `SPLIT_PATTERN`）
> - `tiktoken`：OpenAI 开源的高效分词推理库（用 C 实现，比 HuggingFace 快约 3-5 倍）

---

### `__init__`

```python
class RustBPETokenizer:
    """Light wrapper around tiktoken (for efficient inference) but train with rustbpe"""

    def __init__(self, enc, bos_token):
        self.enc = enc
        self.bos_token_id = self.encode_special(bos_token)
```

> - `enc`：tiktoken `Encoding` 对象（推理引擎）
> - `bos_token_id`：缓存 BOS token 的 ID（高频使用，避免重复查找）
> - 注意：`encode_special` 有 `@lru_cache`，首次调用后自动缓存

---

### `train_from_iterator` — 训练分词器

```python
    @classmethod
    def train_from_iterator(cls, text_iterator, vocab_size):
        tokenizer = rustbpe.Tokenizer()
        vocab_size_no_special = vocab_size - len(SPECIAL_TOKENS)
        assert vocab_size_no_special >= 256, ...
        tokenizer.train_from_iterator(text_iterator, vocab_size_no_special, pattern=SPLIT_PATTERN)
```

> **训练阶段用 rustbpe**：
> - 先计算去掉 9 个特殊 token 后的 BPE 词表大小
> - `vocab_size_no_special >= 256`：至少需要 256 个位置存放所有单字节 token（byte-level BPE 的基础）
> - `pattern=SPLIT_PATTERN`：传入预分词正则，rustbpe 在训练前先用该正则切分文本

```python
        pattern = tokenizer.get_pattern()
        mergeable_ranks_list = tokenizer.get_mergeable_ranks()
        mergeable_ranks = {bytes(k): v for k, v in mergeable_ranks_list}
        tokens_offset = len(mergeable_ranks)
        special_tokens = {name: tokens_offset + i for i, name in enumerate(SPECIAL_TOKENS)}
        enc = tiktoken.Encoding(
            name="rustbpe",
            pat_str=pattern,
            mergeable_ranks=mergeable_ranks,
            special_tokens=special_tokens,
        )
        return cls(enc, "<|bos|>")
```

> **训练后转换为 tiktoken**：
>
> 1. `get_pattern()`：从 rustbpe 取回实际使用的正则模式
> 2. `get_mergeable_ranks()`：取回所有 BPE 合并规则，格式为 `(bytes, rank)` 列表
> 3. 转换为 `dict[bytes, int]`：tiktoken 要求的格式（token 字节序列 → 合并优先级排名）
> 4. `tokens_offset`：特殊 token 的 ID 从 BPE 词表结束位置之后开始，避免冲突
> 5. `special_tokens`：`{token_str: id}` 字典，9 个特殊 token 按顺序分配 ID
> 6. `tiktoken.Encoding`：用 BPE 规则和特殊 token 构建 tiktoken 编码器
>
> **为什么训练用 rustbpe、推理用 tiktoken？**
> - rustbpe 支持自定义正则 `SPLIT_PATTERN`，tiktoken 的内置分词器写死了正则
> - tiktoken 推理速度极快（C 实现），适合生产环境
> - 通过上述转换，可以享受两者的优势

---

### `from_directory` — 从磁盘加载

```python
    @classmethod
    def from_directory(cls, tokenizer_dir):
        pickle_path = os.path.join(tokenizer_dir, "tokenizer.pkl")
        with open(pickle_path, "rb") as f:
            enc = pickle.load(f)
        return cls(enc, "<|bos|>")
```

> 从 `tokenizer.pkl` 加载 tiktoken `Encoding` 对象（由 `save()` 序列化）。
> 使用 pickle 而非 JSON，因为 tiktoken `Encoding` 对象不支持 JSON 序列化。

---

### `from_pretrained` — 使用 tiktoken 内置分词器

```python
    @classmethod
    def from_pretrained(cls, tiktoken_name):
        enc = tiktoken.get_encoding(tiktoken_name)
        return cls(enc, "<|endoftext|>")
```

> 加载 tiktoken 内置的预训练分词器（如 `"cl100k_base"` 即 GPT-4 的分词器）：
> - BOS token 用 `"<|endoftext|>"`（tiktoken 的标准文档分隔符）
> - 注释解释了历史混淆：`<|endoftext|>` 常被放在文档开头（作为 BOS），但字面意义是"文本结束"，nanochat 统一改名为 `<|bos|>`

---

### 工具方法

```python
    def get_vocab_size(self):
        return self.enc.n_vocab
```

> tiktoken 的词表大小属性（包含特殊 token）。

```python
    def get_special_tokens(self):
        return self.enc.special_tokens_set
```

> 返回所有特殊 token 的集合（tiktoken 内置属性）。

```python
    def id_to_token(self, id):
        return self.enc.decode([id])
```

> 将单个 token ID 解码为字符串。tiktoken 没有直接的 `id_to_token` 方法，用单元素列表 decode 实现。

```python
    @lru_cache(maxsize=32)
    def encode_special(self, text):
        return self.enc.encode_single_token(text)
```

> 编码单个特殊 token：
> - `encode_single_token`：tiktoken 专用的特殊 token 编码方法（精确匹配）
> - `@lru_cache(maxsize=32)`：缓存最近 32 个调用结果。特殊 token 种类少（9 个），几乎所有调用都命中缓存，避免重复查表

---

### `encode` — 编码文本

```python
    def encode(self, text, prepend=None, append=None, num_threads=8):
        if prepend is not None:
            prepend_id = prepend if isinstance(prepend, int) else self.encode_special(prepend)
        if append is not None:
            append_id = append if isinstance(append, int) else self.encode_special(append)

        if isinstance(text, str):
            ids = self.enc.encode_ordinary(text)
            if prepend is not None:
                ids.insert(0, prepend_id)
            if append is not None:
                ids.append(append_id)
        elif isinstance(text, list):
            ids = self.enc.encode_ordinary_batch(text, num_threads=num_threads)
            if prepend is not None:
                for ids_row in ids:
                    ids_row.insert(0, prepend_id)
            if append is not None:
                for ids_row in ids:
                    ids_row.append(append_id)
        else:
            raise ValueError(f"Invalid input type: {type(text)}")

        return ids
```

> tiktoken 编码，支持单个字符串和列表批量：
>
> - `encode_ordinary(text)`：**不编码特殊 token**，只编码普通文本（避免用户输入意外触发特殊 token）
> - `encode_ordinary_batch(text, num_threads=8)`：多线程批量编码，`num_threads=8` 充分利用多核 CPU
> - `prepend`/`append`：在编码结果前/后插入特殊 token ID
> - 注意注释 `TODO: slightly inefficient here`：`ids.insert(0, prepend_id)` 是 O(n) 操作，大批量时有优化空间

---

### `decode`

```python
    def decode(self, ids):
        return self.enc.decode(ids)
```

> tiktoken 解码：将 token ID 列表转回字节串并解码为 UTF-8 字符串。
> tiktoken 自动处理多字节 UTF-8 字符（如中文、emoji），不会产生乱码。

---

### `save`

```python
    def save(self, tokenizer_dir):
        os.makedirs(tokenizer_dir, exist_ok=True)
        pickle_path = os.path.join(tokenizer_dir, "tokenizer.pkl")
        with open(pickle_path, "wb") as f:
            pickle.dump(self.enc, f)
        print(f"Saved tokenizer encoding to {pickle_path}")
```

> 将 tiktoken `Encoding` 对象序列化为 `tokenizer.pkl`。
> 与 HuggingFace 的 `tokenizer.json` 不同，pkl 是 Python 特有格式，但包含 tiktoken 对象所有必要信息。



---

## 第四部分：render_conversation — 对话渲染核心

> 这是整个文件最关键的方法，将结构化的对话数据转换为 token ID 序列和对应的 loss mask。

```python
    def render_conversation(self, conversation, max_tokens=2048):
        """
        Tokenize a single Chat conversation (which we call a "doc" or "document" here).
        Returns:
        - ids: list[int] is a list of token ids of this rendered conversation
        - mask: list[int] of same length, mask = 1 for tokens that the Assistant is expected to train on.
        """
```

> 返回值：
> - `ids`：token ID 列表
> - `mask`：同长度的掩码列表，1 = assistant 回复（参与 loss 计算），0 = 其他（用户输入、系统消息、工具输出，不参与 loss）

```python
        ids, mask = [], []
        def add_tokens(token_ids, mask_val):
            if isinstance(token_ids, int):
                token_ids = [token_ids]
            ids.extend(token_ids)
            mask.extend([mask_val] * len(token_ids))
```

> 辅助函数 `add_tokens`：同时向 `ids` 和 `mask` 追加内容：
> - 如果传入的是单个 int（特殊 token），自动包装成列表
> - `mask_val`：0 或 1，表示这批 token 是否参与 loss

```python
        if conversation["messages"][0]["role"] == "system":
            conversation = copy.deepcopy(conversation)
            messages = conversation["messages"]
            assert messages[1]["role"] == "user"
            messages[1]["content"] = messages[0]["content"] + "\n\n" + messages[1]["content"]
            messages = messages[1:]
        else:
            messages = conversation["messages"]
```

> **System 消息处理**：
> - nanochat 不单独渲染 system 消息，而是将其内容合并到第一条 user 消息前（换两行隔开）
> - 使用 `copy.deepcopy` 避免修改原始 conversation 数据（防止副作用）
> - `assert messages[1]["role"] == "user"`：确保 system 消息后必须紧跟 user 消息

```python
        assert len(messages) >= 1
        bos = self.get_bos_token_id()
        user_start, user_end = self.encode_special("<|user_start|>"), self.encode_special("<|user_end|>")
        assistant_start, assistant_end = self.encode_special("<|assistant_start|>"), self.encode_special("<|assistant_end|>")
        python_start, python_end = self.encode_special("<|python_start|>"), self.encode_special("<|python_end|>")
        output_start, output_end = self.encode_special("<|output_start|>"), self.encode_special("<|output_end|>")
```

> 预先获取所有 9 个特殊 token 的 ID（利用 lru_cache 缓存，开销极小）。

```python
        add_tokens(bos, 0)
        for i, message in enumerate(messages):
            must_be_from = "user" if i % 2 == 0 else "assistant"
            assert message["role"] == must_be_from
```

> - 每段对话以 BOS token 开头（mask=0，不参与 loss）
> - 严格检查消息轮次：偶数索引必须是 user，奇数索引必须是 assistant
> - 保证对话格式正确（user ↔ assistant 严格交替）

```python
            content = message["content"]

            if message["role"] == "user":
                assert isinstance(content, str)
                value_ids = self.encode(content)
                add_tokens(user_start, 0)
                add_tokens(value_ids, 0)
                add_tokens(user_end, 0)
```

> **User 消息渲染**（全部 mask=0，不训练）：
> ```
> <|user_start|> [用户文本 tokens] <|user_end|>
>      mask=0         mask=0            mask=0
> ```
> - 用户输入不参与 loss：模型不需要"学会生成用户输入"

```python
            elif message["role"] == "assistant":
                add_tokens(assistant_start, 0)
                if isinstance(content, str):
                    value_ids = self.encode(content)
                    add_tokens(value_ids, 1)
```

> **Assistant 纯文本消息渲染**（文本 mask=1，参与训练）：
> ```
> <|assistant_start|> [助手文本 tokens] <|assistant_end|>
>       mask=0              mask=1            mask=1
> ```
> - `assistant_start` 本身 mask=0：模型不需要学"何时生成 start token"（这由推理引擎控制）
> - 文本内容 mask=1：模型需要学会生成助手回复

```python
                elif isinstance(content, list):
                    for part in content:
                        value_ids = self.encode(part["text"])
                        if part["type"] == "text":
                            add_tokens(value_ids, 1)
                        elif part["type"] == "python":
                            add_tokens(python_start, 1)
                            add_tokens(value_ids, 1)
                            add_tokens(python_end, 1)
                        elif part["type"] == "python_output":
                            add_tokens(output_start, 0)
                            add_tokens(value_ids, 0)
                            add_tokens(output_end, 0)
```

> **Assistant 多部分消息渲染**（工具调用场景）：
>
> 助手内容可以是包含多个 `part` 的列表：
>
> | part["type"] | mask | 说明 |
> |-------------|------|------|
> | `"text"` | 1 | 普通文本，训练 |
> | `"python"` | 1 | Python 代码（助手写的），训练 |
> | `"python_output"` | 0 | Python 执行结果（来自外部），不训练 |
>
> 渲染示例（Python 工具调用）：
> ```
> <|assistant_start|>
>   让我算一下：          mask=1
>   <|python_start|>     mask=1
>   print(2+2)           mask=1
>   <|python_end|>       mask=1
>   <|output_start|>     mask=0
>   4                    mask=0
>   <|output_end|>       mask=0
>   答案是 4。            mask=1
> <|assistant_end|>      mask=1
> ```
>
> **关键设计**：Python 输出（`python_output`）是外部 REPL 的返回值，不是模型"生成"的，所以 mask=0。
> 模型只需学会"在什么情况下调用 Python 以及写什么代码"，不需要学会预测执行结果。

```python
                add_tokens(assistant_end, 1)
```

> `assistant_end` token mask=1：模型需要学会生成结束 token（知道什么时候停止回复）。

```python
        ids = ids[:max_tokens]
        mask = mask[:max_tokens]
        return ids, mask
```

> 截断到 `max_tokens`（默认 2048），防止超长对话导致 OOM。

---

## 第五部分：visualize_tokenization — 调试可视化

```python
    def visualize_tokenization(self, ids, mask, with_token_id=False):
        """Small helper function useful in debugging: visualize the tokenization of render_conversation"""
        RED = '\033[91m'
        GREEN = '\033[92m'
        RESET = '\033[0m'
        GRAY = '\033[90m'
```

> ANSI 颜色码：
> - 红色（RED）：mask=0（不参与训练）
> - 绿色（GREEN）：mask=1（参与训练）
> - 灰色（GRAY）：显示 token ID
> - 重置（RESET）：清除颜色

```python
        tokens = []
        for i, (token_id, mask_val) in enumerate(zip(ids, mask)):
            token_str = self.decode([token_id])
            color = GREEN if mask_val == 1 else RED
            tokens.append(f"{color}{token_str}{RESET}")
            if with_token_id:
                tokens.append(f"{GRAY}({token_id}){RESET}")
        return '|'.join(tokens)
```

> 将每个 token 用颜色标注后用 `|` 分隔，输出到终端：
> - 绿色 token = assistant 文字（训练目标）
> - 红色 token = 用户输入/特殊 token（不训练）
> - `with_token_id=True` 时在每个 token 后附上 ID（调试用）
>
> 示例输出：
> ```
> [红]<|bos|>|[红]<|user_start|>|[红]你好|[红]<|user_end|>|[红]<|assistant_start|>|[绿]你好！|[绿]<|assistant_end|>
> ```

---

## 第六部分：render_for_completion — RL 推理渲染

```python
    def render_for_completion(self, conversation):
        """
        Used during Reinforcement Learning. In that setting, we want to
        render the conversation priming the Assistant for a completion.
        Unlike the Chat SFT case, we don't need to return the mask.
        """
        conversation = copy.deepcopy(conversation)
        messages = conversation["messages"]
        assert messages[-1]["role"] == "assistant"
        messages.pop()
```

> RL 阶段的特殊需求：
> - 不需要 mask（RL 通过 reward 训练，不需要 teacher-forcing 的 mask）
> - 需要去掉最后一条 assistant 消息（这是 ground truth，在 RL 中由模型自己生成）
> - `copy.deepcopy`：避免修改原始对话数据

```python
        ids, mask = self.render_conversation(conversation)
        assistant_start = self.encode_special("<|assistant_start|>")
        ids.append(assistant_start)
        return ids
```

> 渲染去掉最后一条 assistant 消息后的对话，然后在末尾追加 `<|assistant_start|>`：
> - 这让模型处于"准备开始生成 assistant 回复"的状态
> - RL 训练器（`chat_rl.py`）接收这个 prefix，让模型自由采样生成回复，然后用外部 reward 函数评分

---

## 第七部分：全局便捷函数

### `get_tokenizer` — 加载分词器

```python
def get_tokenizer():
    from nanochat.common import get_base_dir
    base_dir = get_base_dir()
    tokenizer_dir = os.path.join(base_dir, "tokenizer")
    # return HuggingFaceTokenizer.from_directory(tokenizer_dir)
    return RustBPETokenizer.from_directory(tokenizer_dir)
```

> 全局分词器加载函数，从 `{base_dir}/tokenizer/` 加载：
> - 注释掉的行表示可以切换到 HuggingFace 实现（对比实验用）
> - 生产代码使用 `RustBPETokenizer`（推理更快）
> - `get_base_dir()` 通常返回用户主目录下的 nanochat 数据目录

---

### `get_token_bytes` — 加载每 token 字节数

```python
def get_token_bytes(device="cpu"):
    import torch
    from nanochat.common import get_base_dir
    base_dir = get_base_dir()
    tokenizer_dir = os.path.join(base_dir, "tokenizer")
    token_bytes_path = os.path.join(tokenizer_dir, "token_bytes.pt")
    assert os.path.exists(token_bytes_path), f"Token bytes not found at {token_bytes_path}? It gets written by tok_train.py"
    with open(token_bytes_path, "rb") as f:
        token_bytes = torch.load(f, map_location=device)
    return token_bytes
```

> 加载预计算的每 token 字节数张量（由 `tok_train.py` 在训练分词器时生成）：
> - 形状：`[vocab_size]`，每个元素是该 token 对应的 UTF-8 字节数
> - 特殊 token 字节数为 0（不计入 bits-per-byte 计算）
> - 用途：计算 bpb（bits-per-byte）损失指标 = `loss / mean(token_bytes)`，与词表大小无关，可跨模型横向比较
> - `map_location=device`：直接加载到目标设备（GPU/CPU），避免二次移动

---

## 整体架构总结

```
训练阶段（tok_train.py）
    ↓ text_iterator（原始文本）
  rustbpe.Tokenizer.train_from_iterator()
    ↓ BPE 合并规则（mergeable_ranks）
  tiktoken.Encoding 对象
    ↓ pickle.dump
  tokenizer.pkl（保存到磁盘）
  token_bytes.pt（每 token 字节数）

推理/训练阶段
    ↓ RustBPETokenizer.from_directory()
  tiktoken.Encoding 对象（高效推理）
    ↓
  encode(text)       → token IDs（普通文本）
  encode_special()   → token ID（特殊 token）
    ↓
  render_conversation(conversation)
    → ids + mask（SFT 训练用）
  render_for_completion(conversation)
    → ids（RL 训练用）
    ↓
  decode(ids) → 原始文本
```

## 对话格式模板总结

```
<|bos|>                               mask=0
<|user_start|>                        mask=0
  [用户文本]                           mask=0
<|user_end|>                          mask=0
<|assistant_start|>                   mask=0
  [助手文本]                           mask=1  ← 训练目标
  <|python_start|>                    mask=1
    [Python 代码]                      mask=1  ← 训练目标
  <|python_end|>                      mask=1
  <|output_start|>                    mask=0
    [REPL 输出]                        mask=0  ← 不训练
  <|output_end|>                      mask=0
  [后续助手文本]                        mask=1  ← 训练目标
<|assistant_end|>                     mask=1  ← 训练目标（学会停止）
<|user_start|>                        mask=0
  [第二轮用户文本]                      mask=0
...（多轮对话循环）
```

## 两套实现对比

| 特性 | HuggingFaceTokenizer | RustBPETokenizer |
|------|---------------------|-----------------|
| 训练后端 | HuggingFace tokenizers（Rust） | rustbpe（Rust） |
| 推理后端 | HuggingFace tokenizers | tiktoken（C） |
| 保存格式 | tokenizer.json（通用） | tokenizer.pkl（Python） |
| 推理速度 | 中等 | 快（tiktoken） |
| 批量推理 | 单线程（for 循环） | 多线程（num_threads=8） |
| 生产使用 | 否（实验/对比） | 是 |
| 加载方式 | from_directory / from_pretrained | from_directory / from_pretrained |
