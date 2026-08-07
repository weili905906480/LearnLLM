"""
BPE Tokenizer in the style of GPT-4.
GPT-4 风格的 BPE（Byte Pair Encoding）分词器。

================================================================================
概述 / Overview
================================================================================
BPE 是当前主流 LLM 使用的子词分词算法。它的核心思想是：
1. 从字节级（256 个基础 token）开始
2. 反复统计相邻 token 对出现的频率
3. 将最高频的 token 对合并成一个新 token
4. 重复步骤 2-3 直到达到目标词表大小

这样高频的字符组合会被合并为单个 token，低频的组合保持为字节。
例如："hello" 经过 BPE 后可能变成 ["h", "el", "lo"] 三个 token。

Two implementations are available:
1) HuggingFace Tokenizer that can do both training and inference but is really confusing
2) Our own RustBPE Tokenizer for training and tiktoken for efficient inference

本文件提供两种实现：
1) HuggingFaceTokenizer —— 基于 HuggingFace 库，可训练和推理，但内部逻辑复杂
2) RustBPETokenizer —— 使用 rustbpe（Rust 实现）训练，用 tiktoken 做推理，速度快

两种分词器都可以相互替换，因为它们的词表和算法是兼容的。
================================================================================
"""

import os
import copy
from functools import lru_cache

# ==============================================================================
# 特殊 Token 定义 / Special Tokens
# ==============================================================================
# 特殊 token 是预留给系统使用的 token，不会出现在训练文本中。
# 它们的 id 会被分配在词表的末尾位置，确保不会被普通文本的 token 占用。
# 在训练 BPE 时，这些 token 被作为 special_tokens 传入，确保它们一定会出现在词表中。

SPECIAL_TOKENS = [
    # every document begins with the Beginning of Sequence (BOS) token that delimits documents
    # BOS (Beginning of Sequence)：每个文档/对话开头的标记，LLM 用它来识别序列边界
    "<|bos|>",
    # tokens below are only used during finetuning to render Conversations into token ids
    # 以下 token 仅在微调（SFT/RL）阶段使用，用于将对话结构化为 token 序列
    "<|user_start|>",  # user messages - 标记用户消息开始
    "<|user_end|>",    # 标记用户消息结束
    "<|assistant_start|>",  # assistant messages - 标记助手/AI 回复开始
    "<|assistant_end|>",    # 标记助手/AI 回复结束
    "<|python_start|>",  # assistant invokes python REPL tool - 助手调用 Python 代码执行工具
    "<|python_end|>",    # Python 代码块结束
    "<|output_start|>",  # python REPL outputs back to assistant - Python 执行结果返回
    "<|output_end|>",    # 输出结果结束
]

# NOTE: this split pattern deviates from GPT-4 in that we use \p{N}{1,2} instead of \p{N}{1,3}
# I did this because I didn't want to "waste" too many tokens on numbers for smaller vocab sizes.
# I verified that 2 is the sweet spot for vocab size of 32K. 1 is a bit worse, 3 was worse still.

# ==============================================================================
# 预分词正则模式 / Split Pattern (Pre-tokenization Regex)
# ==============================================================================
# 这个正则表达式定义了 BPE 之前如何将文本切分成"组（chunk）"。
# BPE 合并只在同一组内进行，不会跨组合并。这是 GPT-4 使用的 split pattern。
#
# 正则解析（按 | 分隔的各分支）：
# 1. '(?i:[sdmt]|ll|ve|re)
#    匹配英文缩写，如 's, 'd, 'm, 't, 'll, 've, 're（大小写不敏感）
#    例: "don't" -> "don" + "'t"（分开处理，避免把缩写当成一个整体）
#
# 2. [^\r\n\p{L}\p{N}]?+\p{L}+
#    匹配一个连续的字母序列（\p{L} 是 Unicode 字母），前面可能有一个非字母数字符号
#    例: "hello" -> "hello";  "!hello" -> "!" + "hello"
#
# 3. \p{N}{1,2}
#    匹配 1-2 位连续数字（GPT-4 原始用 {1,3}，这里改成 {1,2} 以节省小词表的 token 空间）
#    例: "2024" -> "20" + "24"（每个2位数一个token，而不是3位数）
#
# 4. ?[^\s\p{L}\p{N}]++[\r\n]*
#    匹配空格（可选）+ 连续的非字母数字符号 + 可能的换行
#    例: "  ;;\n" -> 一个整体
#
# 5. \s*[\r\n]
#    匹配可选的空白 + 换行
#
# 6. \s+(?!\S)
#    匹配行尾的空白（后面没有非空白字符的空白）
#
# 7. \s+
#    匹配其他所有空白字符
#
# 设计原则：让 BPE 在不同的"语义边界"内进行合并，防止跨语义边界的无意义合并。

SPLIT_PATTERN = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,2}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""

# -----------------------------------------------------------------------------
# Generic GPT-4-style tokenizer based on HuggingFace Tokenizer
# 实现 1：基于 HuggingFace tokenizers 库的 GPT-4 风格分词器
# -----------------------------------------------------------------------------
# HuggingFace 的 tokenizers 库（注意是 tokenizers 不是 transformers）：
# - 用 Rust 编写，速度快
# - 提供了完整的 tokenizer pipeline：Normalizer -> PreTokenizer -> Model -> PostProcessor -> Decoder
# - 这里只使用其中的 PreTokenizer、Model(BPE)、Decoder 组件
#
# Tokenizer Pipeline 各组件的作用：
# - Normalizer（标准化器）：对文本做 Unicode 标准化、小写化等（本实现设为 None，不做标准化）
# - PreTokenizer（预分词器）：用正则将文本切分成 chunk
# - Model（模型）：在 chunk 上执行 BPE 合并
# - PostProcessor（后处理器）：添加特殊 token 如 BOS/EOS（本实现设为 None，手动处理）
# - Decoder（解码器）：将 token id 序列还原为文本字符串

from tokenizers import Tokenizer as HFTokenizer
from tokenizers import pre_tokenizers, decoders, Regex
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer

class HuggingFaceTokenizer:
    """
    Light wrapper around HuggingFace Tokenizer for some utilities.
    对 HuggingFace Tokenizer 的轻量封装，提供统一接口。

    使用方式：
    1. 训练新分词器：   HuggingFaceTokenizer.train_from_iterator(texts, vocab_size=65536)
    2. 加载预训练分词器：HuggingFaceTokenizer.from_pretrained("gpt2")
    3. 加载本地分词器：  HuggingFaceTokenizer.from_directory("out/tokenizer")
    """

    def __init__(self, tokenizer):
        """
        初始化分词器。
        Args:
            tokenizer: HuggingFace Tokenizer 实例（HFTokenizer 对象）
        """
        self.tokenizer = tokenizer

    @classmethod
    def from_pretrained(cls, hf_path):
        """
        从 HuggingFace Hub 加载预训练分词器。
        init from a HuggingFace pretrained tokenizer (e.g. "gpt2")

        Args:
            hf_path: HuggingFace 模型名称，如 "gpt2"
        Returns:
            HuggingFaceTokenizer 实例
        """
        tokenizer = HFTokenizer.from_pretrained(hf_path)
        return cls(tokenizer)

    @classmethod
    def from_directory(cls, tokenizer_dir):
        """
        从本地目录加载分词器。
        init from a local directory on disk (e.g. "out/tokenizer")

        Args:
            tokenizer_dir: 分词器目录路径，目录下需包含 tokenizer.json 文件
        Returns:
            HuggingFaceTokenizer 实例
        """
        tokenizer_path = os.path.join(tokenizer_dir, "tokenizer.json")
        tokenizer = HFTokenizer.from_file(tokenizer_path)
        return cls(tokenizer)

    @classmethod
    def train_from_iterator(cls, text_iterator, vocab_size):
        """
        从文本迭代器训练 BPE 分词器。
        train from an iterator of text

        训练流程：
        1. 创建 BPE 模型（支持 byte fallback，即未知字符退回字节表示）
        2. 设置预分词器：GPT-4 split pattern + ByteLevel 预处理
        3. 设置解码器：ByteLevel
        4. 在文本上运行 BPE 训练

        Args:
            text_iterator: 文本迭代器，每次 yield 一个字符串（可以是大型数据集的惰性迭代器）
            vocab_size: 目标词表大小（包含特殊 token），例如 65536
        Returns:
            HuggingFaceTokenizer 实例

        注意：
        - byte_fallback=True 确保每个字节都能被编码，不会有 UNK
        - min_frequency=0 表示即使只出现一次的 token 对也会被合并
        - 特殊 token（如 <|bos|>）会自动包含在词表中
        """
        # Configure the HuggingFace Tokenizer
        tokenizer = HFTokenizer(BPE(
            byte_fallback=True,  # needed! 字节回退：未知字符用字节序列表示，避免 UNK
            unk_token=None,      # 不设置 UNK token（因为 byte_fallback 保证不会有未知字符）
            fuse_unk=False,      # 不融合 UNK token
        ))
        # Normalizer: None
        # 不设置标准化器——保持原始文本不变
        tokenizer.normalizer = None
        # Pre-tokenizer: GPT-4 style
        # the regex pattern used by GPT-4 to split text into groups before BPE
        # NOTE: The pattern was changed from \p{N}{1,3} to \p{N}{1,2} because I suspect it is harmful to
        # very small models and smaller vocab sizes, because it is a little bit wasteful in the token space.
        # (but I haven't validated this! TODO)
        # 预分词器分两步：
        # Step 1: 按 GPT-4 split pattern 将文本切分成 chunk
        # Step 2: 对每个 chunk 做 ByteLevel 编码（将文本转为字节表示，再映射到可见字符）
        gpt4_split_regex = Regex(SPLIT_PATTERN)  # huggingface demands that you wrap it in Regex!!
        tokenizer.pre_tokenizer = pre_tokenizers.Sequence([
            pre_tokenizers.Split(pattern=gpt4_split_regex, behavior="isolated", invert=False),
            # behavior="isolated": 每个匹配项独立成组，匹配项之间的内容也是独立的组
            # invert=False: 按正则匹配的内容来切分（而非保留匹配内容）
            pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False)
            # ByteLevel: 将 Unicode 字符转换为字节级表示
            # add_prefix_space=False: 不在开头添加空格（GPT 风格）
        ])
        # Decoder: ByteLevel (it pairs together with the ByteLevel pre-tokenizer)
        # 解码器用 ByteLevel，与预分词器中的 ByteLevel 配对使用
        tokenizer.decoder = decoders.ByteLevel()
        # Post-processor: None
        # 不设置后处理器——BOS/EOS 等特殊 token 在 encode 时手动添加
        tokenizer.post_processor = None
        # Trainer: BPE
        # 配置 BPE 训练器参数
        trainer = BpeTrainer(
            vocab_size=vocab_size,          # 目标词表大小
            show_progress=True,             # 显示训练进度
            min_frequency=0,                # no minimum frequency - 不设最小频率阈值
            initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
            # 初始字母表：包含所有 256 个字节的 ByteLevel 表示
            special_tokens=SPECIAL_TOKENS,  # 特殊 token 列表，确保它们出现在词表中
        )
        # Kick off the training
        # 开始训练——遍历 text_iterator 中的文本，统计 token 对频率，反复合并
        tokenizer.train_from_iterator(text_iterator, trainer)
        return cls(tokenizer)

    def get_vocab_size(self):
        """返回词表大小。Get vocabulary size."""
        return self.tokenizer.get_vocab_size()

    def get_special_tokens(self):
        """返回特殊 token 列表。Get list of special tokens. """
        special_tokens_map = self.tokenizer.get_added_tokens_decoder()
        special_tokens = [w.content for w in special_tokens_map.values()]
        return special_tokens

    def id_to_token(self, id):
        """
        将 token id 转换为 token 字符串。
        例: id_to_token(1234) -> "hello"

        Args:
            id: token id (整数)
        Returns:
            token 字符串
        """
        return self.tokenizer.id_to_token(id)

    def _encode_one(self, text, prepend=None, append=None, num_threads=None):
        """
        编码单个字符串为 token id 序列（内部方法）。
        encode a single string

        编码流程：
        1. 如果设置了 prepend，先在开头插入指定 token
        2. 调用 tokenizer.encode 将文本转为 token id
        3. 如果设置了 append，在末尾追加指定 token

        Args:
            text: 要编码的文本字符串
            prepend: 可选，在开头添加的内容。可以是 token 字符串或 token id
            append:  可选，在末尾添加的内容。可以是 token 字符串或 token id
            num_threads: 多线程参数（此实现中忽略，仅为接口兼容）
        Returns:
            list[int]: token id 序列
        """
        # prepend/append can be either a string of a special token or a token id directly.
        # num_threads is ignored (only used by the nanochat Tokenizer for parallel encoding)
        assert isinstance(text, str)
        ids = []
        if prepend is not None:
            # 如果 prepend 是整数，直接作为 token id；否则通过 encode_special 查找
            prepend_id = prepend if isinstance(prepend, int) else self.encode_special(prepend)
            ids.append(prepend_id)
        # add_special_tokens=False: 不自动添加 BOS/EOS，因为已经在 prepend/append 中手动处理
        ids.extend(self.tokenizer.encode(text, add_special_tokens=False).ids)
        if append is not None:
            append_id = append if isinstance(append, int) else self.encode_special(append)
            ids.append(append_id)
        return ids

    def encode_special(self, text):
        """
        编码特殊 token（通过精确匹配查找 token id）。
        encode a single special token via exact match

        与普通文本编码不同，特殊 token 要求完全匹配。如果找不到则返回 None。

        Args:
            text: 特殊 token 字符串，如 "<|bos|>"
        Returns:
            token id (int) 或 None（如果词表中不存在该特殊 token）
        """
        return self.tokenizer.token_to_id(text)

    def get_bos_token_id(self):
        """
        获取 BOS (Beginning of Sequence) token 的 id。
        Different HuggingFace models use different BOS tokens and there is little consistency

        查找策略（按优先级）：
        1) 尝试找 <|bos|> token（nanochat 默认）
        2) 如果失败，尝试找 <|endoftext|> token（GPT-2 等模型使用）
        3) 如果都找不到，抛出 AssertionError

        历史注：GPT-2/3 和一些老模型用 <|endoftext|> 作为文档分隔符，
        虽然名字叫 "end of text"，但实际用途是同时标记结束和开始。
        nanochat 统一使用 <|bos|>，语义更清晰。

        Returns:
            BOS token 的整数 id
        """
        # 1) attempt to find a <|bos|> token
        bos = self.encode_special("<|bos|>")
        # 2) if that fails, attempt to find a <|endoftext|> token (e.g. GPT-2 models)
        if bos is None:
            bos = self.encode_special("<|endoftext|>")
        # 3) if these fail, it's better to crash than to silently return None
        assert bos is not None, "Failed to find BOS token in tokenizer"
        return bos

    def encode(self, text, *args, **kwargs):
        """
        编码文本为 token id 序列。支持单字符串和字符串列表。

        这是分词器的主要编码接口。
        - 输入单个字符串 -> 返回 list[int]
        - 输入字符串列表   -> 返回 list[list[int]]

        Args:
            text: str 或 list[str] —— 要编码的文本
            *args, **kwargs: 传递给 _encode_one 的参数（prepend, append 等）
        Returns:
            list[int] 或 list[list[int]]: token id(s)
        """
        if isinstance(text, str):
            return self._encode_one(text, *args, **kwargs)
        elif isinstance(text, list):
            return [self._encode_one(t, *args, **kwargs) for t in text]
        else:
            raise ValueError(f"Invalid input type: {type(text)}")

    def __call__(self, *args, **kwargs):
        """
        使分词器实例可以像函数一样直接调用。
        例: tokenizer("hello world") 等价于 tokenizer.encode("hello world")
        """
        return self.encode(*args, **kwargs)

    def decode(self, ids):
        """
        将 token id 序列解码为文本字符串。

        Args:
            ids: list[int]，token id 序列
        Returns:
            str: 解码后的文本
        """
        return self.tokenizer.decode(ids, skip_special_tokens=False)
        # skip_special_tokens=False: 不解码特殊 token，保留它们在输出中

    def save(self, tokenizer_dir):
        """
        将分词器保存到磁盘。
        save the tokenizer to disk

        保存为 HuggingFace 标准的 tokenizer.json 格式。

        Args:
            tokenizer_dir: 保存目录路径，会自动创建（如不存在）
        """
        os.makedirs(tokenizer_dir, exist_ok=True)
        tokenizer_path = os.path.join(tokenizer_dir, "tokenizer.json")
        self.tokenizer.save(tokenizer_path)
        print(f"Saved tokenizer to {tokenizer_path}")


# -----------------------------------------------------------------------------
# Tokenizer based on rustbpe + tiktoken combo
# 实现 2：基于 rustbpe（训练）+ tiktoken（推理）的分词器
# -----------------------------------------------------------------------------
# 为什么用两个库？
# - rustbpe：Rust 编写的 BPE 训练器，训练速度快，内存效率高
# - tiktoken：OpenAI 出品的高效分词推理库，支持批量编码和多线程
#
# 训练时用 rustbpe（因为 HuggingFace tokenizers 的 BPE 训练有时会有奇怪的行为），
# 推理时用 tiktoken（因为 tiktoken 的 encode_ordinary_batch 支持高效的多线程批量编码）。
#
# 两者通过相同的 split pattern 和 mergeable_ranks（合并优先级）保持一致性。

import pickle
import rustbpe
import tiktoken

class RustBPETokenizer:
    """
    Light wrapper around tiktoken (for efficient inference) but train with rustbpe.
    使用 rustbpe 训练、tiktoken 推理的分词器。

    这是 nanochat 的首选分词器实现（在 get_tokenizer() 中默认使用）。
    优势：
    1. rustbpe 训练更可靠
    2. tiktoken 推理支持多线程批量编码
    3. 序列化简单（pickle），加载速度快
    """

    def __init__(self, enc, bos_token):
        """
        初始化分词器。

        Args:
            enc: tiktoken.Encoding 实例（包含词表和编码逻辑）
            bos_token: BOS token 字符串，如 "<|bos|>" 或 "<|endoftext|>"
        """
        self.enc = enc
        self.bos_token_id = self.encode_special(bos_token)

    @classmethod
    def train_from_iterator(cls, text_iterator, vocab_size):
        """
        从文本迭代器训练分词器。
        训练流程：
        1) 使用 rustbpe 训练 BPE 合并规则
        2) 将训练结果转换为 tiktoken Encoding 对象用于推理

        关键步骤：
        - 从 vocab_size 中减去特殊 token 数量，得到实际训练的 BPE token 数
        - rustbpe 训练产生 mergeable_ranks（token 字节 -> 合并优先级）
        - 特殊 token 的 id 分配在 BPE token 之后（连续的 id 区域）
        - 将上述信息封装为 tiktoken.Encoding 对象

        Args:
            text_iterator: 文本迭代器
            vocab_size: 目标词表总大小（含特殊 token）
        Returns:
            RustBPETokenizer 实例

        Raises:
            AssertionError: 如果 vocab_size 减去特殊 token 后小于 256（至少要保留所有字节）
        """
        # 1) train using rustbpe
        # Step 1: 用 rustbpe 训练
        tokenizer = rustbpe.Tokenizer()
        # the special tokens are inserted later in __init__, we don't train them here
        # 计算 BPE 部分需要的 token 数（总词表 - 特殊 token 数）
        vocab_size_no_special = vocab_size - len(SPECIAL_TOKENS)
        # 必须 >= 256，因为至少需要表示所有单字节（0-255）
        assert vocab_size_no_special >= 256, f"vocab_size_no_special must be at least 256, got {vocab_size_no_special}"
        tokenizer.train_from_iterator(text_iterator, vocab_size_no_special, pattern=SPLIT_PATTERN)
        # 训练输出：每个合并步骤产生一个新的 token（由两个更小的 token 拼接而成）

        # 2) construct the associated tiktoken encoding for inference
        # Step 2: 构建 tiktoken Encoding 对象用于推理
        pattern = tokenizer.get_pattern()  # 获取 rustbpe 内部使用的 split pattern（应与 SPLIT_PATTERN 一致）
        mergeable_ranks_list = tokenizer.get_mergeable_ranks()
        # mergeable_ranks: 将 token 的字节表示映射到合并优先级（数字越小优先级越高/越早被合并）
        # 格式: [(bytes, rank), ...] -> {bytes: rank}
        mergeable_ranks = {bytes(k): v for k, v in mergeable_ranks_list}
        # 特殊 token 的 id 从 BPE token 数量之后开始分配
        # 例如: 如果有 65500 个 BPE token，则第一个特殊 token 的 id 是 65500
        tokens_offset = len(mergeable_ranks)
        special_tokens = {name: tokens_offset + i for i, name in enumerate(SPECIAL_TOKENS)}
        # 创建 tiktoken Encoding 对象
        enc = tiktoken.Encoding(
            name="rustbpe",
            pat_str=pattern,                    # split pattern 字符串
            mergeable_ranks=mergeable_ranks,    # dict[bytes, int] (token bytes -> merge priority rank)
            special_tokens=special_tokens,      # dict[str, int] (special token name -> token id)
        )
        return cls(enc, "<|bos|>")

    @classmethod
    def from_directory(cls, tokenizer_dir):
        """
        从本地目录加载分词器。
        从 pickle 文件中反序列化 tiktoken Encoding 对象。

        Args:
            tokenizer_dir: 分词器目录，需包含 tokenizer.pkl 文件
        Returns:
            RustBPETokenizer 实例
        """
        pickle_path = os.path.join(tokenizer_dir, "tokenizer.pkl")
        with open(pickle_path, "rb") as f:
            enc = pickle.load(f)
        return cls(enc, "<|bos|>")

    @classmethod
    def from_pretrained(cls, tiktoken_name):
        """
        从 tiktoken 加载 OpenAI 预训练分词器。
        例如: from_pretrained("gpt2") 会加载 GPT-2 的分词器。

        注意：OpenAI 的分词器用 <|endoftext|> 作为序列分隔符，
        nanochat 统一用 <|bos|> 这个概念，但这里做兼容处理。

        Args:
            tiktoken_name: tiktoken 分词器名称，如 "gpt2", "cl100k_base" 等
        Returns:
            RustBPETokenizer 实例
        """
        # https://github.com/openai/tiktoken/blob/eedc8563/tiktoken_ext/openai_public.py
        enc = tiktoken.get_encoding(tiktoken_name)
        # tiktoken calls the special document delimiter token "<|endoftext|>"
        # yes this is confusing because this token is almost always PREPENDED to the beginning of the document
        # it most often is used to signal the start of a new sequence to the LLM during inference etc.
        # so in nanoChat we always use "<|bos|>" short for "beginning of sequence", but historically it is often called "<|endoftext|>".
        # <|endoftext|> 在 GPT-2/3 中既是文档结束标记也是文档开始标记，历史命名有歧义
        # nanochat 使用 <|bos|> 语义更明确，但加载 tiktoken 分词器时用 <|endoftext|>
        return cls(enc, "<|endoftext|>")

    def get_vocab_size(self):
        """返回词表大小。Get vocabulary size."""
        return self.enc.n_vocab

    def get_special_tokens(self):
        """返回特殊 token 集合。Get the set of special tokens."""
        return self.enc.special_tokens_set

    def id_to_token(self, id):
        """
        将 token id 解码为对应的 token 字符串。
        例: id_to_token(1234) -> "hello"

        Args:
            id: token id (int)
        Returns:
            str: token 字符串
        """
        return self.enc.decode([id])

    @lru_cache(maxsize=32)
    def encode_special(self, text):
        """
        编码特殊 token（带 LRU 缓存）。
        因为特殊 token 种类少且反复使用，用 LRU 缓存避免重复查找。

        Args:
            text: 特殊 token 字符串
        Returns:
            int: 特殊 token 的 id
        """
        return self.enc.encode_single_token(text)

    def get_bos_token_id(self):
        """
        返回 BOS token 的 id。
        与 HuggingFaceTokenizer 不同，这里直接返回在 __init__ 中缓存的 bos_token_id。
        """
        return self.bos_token_id

    def encode(self, text, prepend=None, append=None, num_threads=8):
        """
        编码文本为 token id 序列。支持单字符串和字符串列表的批量编码。
        text can be either a string or a list of strings

        tiktoken 的 encode_ordinary_batch 支持多线程批量编码，速度很快。

        Args:
            text: str 或 list[str] —— 要编码的文本
            prepend: 可选，在开头添加的 token（字符串或 id）
            append:  可选，在末尾添加的 token（字符串或 id）
            num_threads: 批量编码时的线程数（默认 8）
        Returns:
            list[int] 或 list[list[int]]: token id 序列
        """
        # preprocess prepend/append：如果是字符串则转为 token id
        # 预处理 prepend / append——提前计算好 id，避免在循环内重复计算
        if prepend is not None:
            prepend_id = prepend if isinstance(prepend, int) else self.encode_special(prepend)
        if append is not None:
            append_id = append if isinstance(append, int) else self.encode_special(append)

        if isinstance(text, str):
            # 单字符串编码
            ids = self.enc.encode_ordinary(text)  # encode_ordinary: 只使用普通 token，不包含特殊 token
            if prepend is not None:
                ids.insert(0, prepend_id)  # TODO: slightly inefficient here? :( hmm
                # 在开头插入的效率问题：list.insert(0, ...) 是 O(n)，但对于单次插入影响可忽略
            if append is not None:
                ids.append(append_id)
        elif isinstance(text, list):
            # 批量编码 —— 使用多线程加速
            ids = self.enc.encode_ordinary_batch(text, num_threads=num_threads)
            if prepend is not None:
                for ids_row in ids:
                    ids_row.insert(0, prepend_id)  # TODO: same efficiency concern
            if append is not None:
                for ids_row in ids:
                    ids_row.append(append_id)
        else:
            raise ValueError(f"Invalid input type: {type(text)}")

        return ids

    def __call__(self, *args, **kwargs):
        """
        使分词器实例可以像函数一样直接调用。
        例: tokenizer("hello") 等价于 tokenizer.encode("hello")
        """
        return self.encode(*args, **kwargs)

    def decode(self, ids):
        """
        将 token id 序列解码为文本字符串。

        Args:
            ids: list[int]，token id 序列
        Returns:
            str: 解码后的文本
        """
        return self.enc.decode(ids)

    def save(self, tokenizer_dir):
        """
        将分词器保存到磁盘。
        save the encoding object to disk

        使用 pickle 序列化 tiktoken Encoding 对象。
        保存为 "tokenizer.pkl" 文件。

        Args:
            tokenizer_dir: 保存目录路径
        """
        os.makedirs(tokenizer_dir, exist_ok=True)
        pickle_path = os.path.join(tokenizer_dir, "tokenizer.pkl")
        with open(pickle_path, "wb") as f:
            pickle.dump(self.enc, f)
        print(f"Saved tokenizer encoding to {pickle_path}")

    def render_conversation(self, conversation, max_tokens=2048):
        """
        将对话（Conversation）渲染为 token id 序列和训练 mask。
        Tokenize a single Chat conversation (which we call a "doc" or "document" here).

        这是 SFT（Supervised Fine-Tuning）阶段的核心函数。
        它将结构化的对话数据转换为模型可以处理的 token 序列，
        并生成一个 mask 指示哪些 token 需要计算 loss。

        渲染（Tokenization）过程：
        输入对话格式示例:
        {
          "messages": [
            {"role": "user",      "content": "What is 2+2?"},
            {"role": "assistant", "content": "2+2 = 4"},
            {"role": "user",      "content": "Write Python code to compute 3*5"},
            {"role": "assistant", "content": [
              {"type": "text",   "text": "Here's the code:"},
              {"type": "python", "text": "print(3*5)"},
              {"type": "python_output", "text": "15"},
              {"type": "text",   "text": "The result is 15."}
            ]},
          ]
        }

        渲染后 token 序列:
        <|bos|> <|user_start|> What is 2+2? <|user_end|>
        <|assistant_start|> 2+2 = 4 <|assistant_end|>
        <|user_start|> Write Python code... <|user_end|>
        <|assistant_start|> Here's the code: <|python_start|> print(3*5) <|python_end|>
        <|output_start|> 15 <|output_end|> The result is 15. <|assistant_end|>

        Mask 规则：
        - mask = 0: 不计算 loss（用户消息、系统消息、Python 输出等非助手生成的 token）
        - mask = 1: 计算 loss（助手/AI 生成的 token，包括文本和 Python 代码）

        为什么 Python 输出不计算 loss？
          因为 Python 输出是外部工具（Python REPL）返回的结果，不是模型生成的。
          训练时不应该让模型去"预测" Python 的输出。

        为什么 Python 代码要计算 loss？
          因为 Python 代码是助手/模型决定调用的，属于模型生成的内容。

        系统消息处理：
          如果第一条消息是 system 角色的，会将其内容合并到紧随其后的 user 消息中。
          这是因为 nanochat 的 tokenizer 不单独支持 system 角色。

        Args:
            conversation: dict，包含 "messages" 键的对话字典
            max_tokens: 最大 token 数（超出部分截断），防止 OOM
        Returns:
            ids:  list[int] —— token id 序列
            mask: list[int] —— 训练 mask（1=需要计算loss, 0=不需要）
                ids 和 mask 长度相同
        """
        # ids, masks that we will return and a helper function to help build them up.
        # 用于收集结果的两个列表和一个辅助函数
        ids, mask = [], []

        def add_tokens(token_ids, mask_val):
            """
            辅助函数：将 token id(s) 添加到 ids 列表，并添加对应长度的 mask 值。
            如果 token_ids 是单个整数，先包装成列表。

            Args:
                token_ids: int 或 list[int] —— 要添加的 token id(s)
                mask_val: int (0 或 1) —— mask 值
            """
            if isinstance(token_ids, int):
                token_ids = [token_ids]
            ids.extend(token_ids)
            mask.extend([mask_val] * len(token_ids))

        # sometimes the first message is a system message...
        # => just merge it with the second (user) message
        # 处理系统消息：合并到第二个（用户）消息中
        if conversation["messages"][0]["role"] == "system":
            # some conversation surgery is necessary here for now...
            # 深拷贝避免修改原始数据
            conversation = copy.deepcopy(conversation)  # avoid mutating the original
            messages = conversation["messages"]
            # 确保第二条消息是 user 角色的
            assert messages[1]["role"] == "user", "System message must be followed by a user message"
            # 将系统消息内容追加到用户消息前面，用换行分隔
            messages[1]["content"] = messages[0]["content"] + "\n\n" + messages[1]["content"]
            # 移除系统消息（已合并到用户消息中）
            messages = messages[1:]
        else:
            messages = conversation["messages"]
        assert len(messages) >= 1, f"Conversation has less than 1 message: {messages}"

        # fetch all the special tokens we need
        # 获取所有需要的特殊 token id（提前计算避免重复查找）
        bos = self.get_bos_token_id()
        user_start, user_end = self.encode_special("<|user_start|>"), self.encode_special("<|user_end|>")
        assistant_start, assistant_end = self.encode_special("<|assistant_start|>"), self.encode_special("<|assistant_end|>")
        python_start, python_end = self.encode_special("<|python_start|>"), self.encode_special("<|python_end|>")
        output_start, output_end = self.encode_special("<|output_start|>"), self.encode_special("<|output_end|>")

        # now we can tokenize the conversation
        # 开始逐条处理对话消息
        add_tokens(bos, 0)  # BOS token —— 不计算 loss (mask=0)
        for i, message in enumerate(messages):

            # some sanity checking here around assumptions, to prevent footguns
            # 消息角色交替校验：偶数索引必须是 user，奇数索引必须是 assistant
            must_be_from = "user" if i % 2 == 0 else "assistant"
            assert message["role"] == must_be_from, f"Message {i} is from {message['role']} but should be from {must_be_from}"

            # content can be either a simple string or a list of parts (e.g. containing tool calls)
            # content 可以是简单字符串，也可以是包含多种类型的列表
            content = message["content"]

            if message["role"] == "user":
                # === 用户消息 ===
                # 用户消息简单直接：纯文本字符串，全部 mask=0
                assert isinstance(content, str), "User messages are simply expected to be strings"
                value_ids = self.encode(content)
                add_tokens(user_start, 0)    # <|user_start|> 不计算 loss
                add_tokens(value_ids, 0)     # 用户文本内容不计算 loss
                add_tokens(user_end, 0)      # <|user_end|> 不计算 loss

            elif message["role"] == "assistant":
                # === 助手消息 ===
                # 助手消息较复杂：可以是纯文本，也可以是包含工具调用的多部分列表
                add_tokens(assistant_start, 0)  # <|assistant_start|> 不计算 loss（只是标记）
                if isinstance(content, str):
                    # simple string => simply add the tokens
                    # 情况 1：纯文本回复
                    value_ids = self.encode(content)
                    add_tokens(value_ids, 1)  # 助手生成的文本 —— 计算 loss (mask=1)
                elif isinstance(content, list):
                    # 情况 2：多部分内容（可能包含代码执行和工具调用）
                    for part in content:
                        value_ids = self.encode(part["text"])
                        if part["type"] == "text":
                            # string part => simply add the tokens
                            # 纯文本部分 —— 计算 loss
                            add_tokens(value_ids, 1)
                        elif part["type"] == "python":
                            # python tool call => add the tokens inside <|python_start|> and <|python_end|>
                            # Python 代码调用 —— 计算 loss（代码是模型生成的）
                            add_tokens(python_start, 1)
                            add_tokens(value_ids, 1)
                            add_tokens(python_end, 1)
                        elif part["type"] == "python_output":
                            # python output => add the tokens inside <|output_start|> and <|output_end|>
                            # none of these tokens are supervised because the tokens come from Python at test time
                            # Python 执行输出 —— 不计算 loss
                            # 原因：输出来自外部 Python 解释器，不是模型生成的，不应让模型学习"预测"它
                            add_tokens(output_start, 0)
                            add_tokens(value_ids, 0)
                            add_tokens(output_end, 0)
                        else:
                            raise ValueError(f"Unknown part type: {part['type']}")
                else:
                    raise ValueError(f"Unknown content type: {type(content)}")
                # <|assistant_end|> 计算 loss
                # 原因是：让模型学会在回复结束时生成 <|assistant_end|>，从而知道何时停止
                add_tokens(assistant_end, 1)

        # truncate to max_tokens tokens MAX (helps prevent OOMs)
        # 截断到 max_tokens，防止显存溢出（OOM）
        ids = ids[:max_tokens]
        mask = mask[:max_tokens]
        return ids, mask

    def visualize_tokenization(self, ids, mask, with_token_id=False):
        """
        可视化 tokenization 结果，主要用于调试。
        Small helper function useful in debugging: visualize the tokenization of render_conversation

        输出带颜色的 token 序列：
        - 绿色：mask=1（需要计算 loss 的 token，即助手生成的 token）
        - 红色：mask=0（不计算 loss 的 token，如用户消息、特殊标记等）
        - 灰色（可选）：在 token 后显示其 id

        使用方式：
            ids, mask = tokenizer.render_conversation(conv)
            print(tokenizer.visualize_tokenization(ids, mask, with_token_id=True))

        Args:
            ids:  token id 序列
            mask: mask 序列（与 ids 等长）
            with_token_id: 是否在 token 旁边显示 id（默认 False）
        Returns:
            str: 带 ANSI 颜色代码的可视化字符串
        """
        RED = '\033[91m'      # 红色 —— mask=0（不计算 loss）
        GREEN = '\033[92m'    # 绿色 —— mask=1（计算 loss）
        RESET = '\033[0m'     # 重置颜色
        GRAY = '\033[90m'     # 灰色 —— token id 显示
        tokens = []
        for i, (token_id, mask_val) in enumerate(zip(ids, mask)):
            token_str = self.decode([token_id])
            color = GREEN if mask_val == 1 else RED
            tokens.append(f"{color}{token_str}{RESET}")
            if with_token_id:
                tokens.append(f"{GRAY}({token_id}){RESET}")
        return '|'.join(tokens)

    def render_for_completion(self, conversation):
        """
        为 RL（强化学习）阶段渲染对话，用于生成补全。
        Used during Reinforcement Learning. In that setting, we want to
        render the conversation priming the Assistant for a completion.
        Unlike the Chat SFT case, we don't need to return the mask.

        与 render_conversation 的区别：
        - render_conversation: 用于 SFT 训练，返回 (ids, mask)
        - render_for_completion: 用于 RL 推理，只需要 ids，不需要 mask
        - 会移除对话的最后一条 assistant 消息（因为那是待生成的补全目标）
        - 在序列末尾追加 <|assistant_start|>，引导模型开始生成

        处理流程：
        1. 深拷贝对话（避免修改原始数据）
        2. 移除最后一条消息（必须是 assistant 角色的）
        3. 渲染剩余对话
        4. 追加 <|assistant_start|> token，引导模型生成回复

        Args:
            conversation: dict，完整的对话数据
        Returns:
            list[int]: token id 序列（末尾是 <|assistant_start|>，模型从此处开始补全）
        """
        # We have some surgery to do: we need to pop the last message (of the Assistant)
        # 深拷贝避免修改原始对话
        conversation = copy.deepcopy(conversation)  # avoid mutating the original
        messages = conversation["messages"]
        # 确保最后一条消息是 assistant 的（RL 中最后一条是需要"重建"的 assistant 回复）
        assert messages[-1]["role"] == "assistant", "Last message must be from the Assistant"
        messages.pop()  # remove the last message (of the Assistant) inplace

        # Now tokenize the conversation
        # 用 render_conversation 渲染剩余对话，但只需要 ids
        ids, mask = self.render_conversation(conversation)

        # Finally, to prime the Assistant for a completion, append the Assistant start token
        # 追加 <|assistant_start|>，告诉模型"现在开始生成 assistant 回复"
        assistant_start = self.encode_special("<|assistant_start|>")
        ids.append(assistant_start)
        return ids


# -----------------------------------------------------------------------------
# nanochat-specific convenience functions
# nanochat 专用的便捷函数
# -----------------------------------------------------------------------------

def get_tokenizer():
    """
    获取默认分词器实例。
    从 base_dir/tokenizer/ 目录加载 RustBPETokenizer。

    这是 nanochat 项目中使用分词器的标准入口。
    默认使用 RustBPETokenizer（而非 HuggingFaceTokenizer），因为：
    1. tiktoken 推理速度更快
    2. 支持多线程批量编码
    3. 序列化/反序列化更简单

    Returns:
        RustBPETokenizer 实例
    """
    from nanochat.common import get_base_dir
    base_dir = get_base_dir()  # 获取 nanochat 项目的基础目录
    tokenizer_dir = os.path.join(base_dir, "tokenizer")  # 分词器子目录
    # return HuggingFaceTokenizer.from_directory(tokenizer_dir)
    # 注释掉的 HuggingFaceTokenizer 方案 —— 可以作为备选
    return RustBPETokenizer.from_directory(tokenizer_dir)

def get_token_bytes(device="cpu"):
    """
    获取 token 的字节表示张量。
    用于将 token 的字节序列加载为 PyTorch 张量，供 embedding 层使用。

    这个张量由 tok_train.py 训练脚本生成，保存在 token_bytes.pt 文件中。
    用途：在模型的 embedding 层中，可以用 token 的字节表示来初始化或增强 embedding。

    Args:
        device: PyTorch 设备，如 "cpu" 或 "cuda"（默认 "cpu"）
    Returns:
        torch.Tensor: token 字节表示张量
    Raises:
        AssertionError: 如果 token_bytes.pt 文件不存在
    """
    import torch
    from nanochat.common import get_base_dir
    base_dir = get_base_dir()
    tokenizer_dir = os.path.join(base_dir, "tokenizer")
    token_bytes_path = os.path.join(tokenizer_dir, "token_bytes.pt")
    assert os.path.exists(token_bytes_path), f"Token bytes not found at {token_bytes_path}? It gets written by tok_train.py"
    with open(token_bytes_path, "rb") as f:
        token_bytes = torch.load(f, map_location=device)
    return token_bytes
