"""
Train a tokenizer using our own BPE Tokenizer library.
In the style of GPT-4 tokenizer.
使用 rustbpe (Rust 实现) 训练 GPT-4 风格的 BPE 分词器。

================================================================================
脚本功能概述
================================================================================
本脚本是 nanochat 训练流程的第一阶段（Stage 1），负责：
1. 从 parquet 数据集中读取预训练文本
2. 训练 BPE 分词器
3. 保存分词器到磁盘
4. 执行编码/解码往返自检
5. 生成 token_bytes.pt（每个 token 的字节数，用于计算 bits per byte）
6. 写入训练报告

输出文件（保存在 base_dir/tokenizer/ 目录）:
- tokenizer.pkl:     tiktoken Encoding 对象（推理用）
- token_bytes.pt:    token id → 字节数的映射张量（评估用）

训练数据流:
  parquet 文件 → parquets_iter_batched() → text_iterator() → rustbpe 训练
================================================================================
"""
import os
import time
import argparse
import torch
from nanochat.tokenizer import RustBPETokenizer
from nanochat.common import get_base_dir
from nanochat.dataset import parquets_iter_batched

# -----------------------------------------------------------------------------
# Parse command line arguments
# 解析命令行参数
# -----------------------------------------------------------------------------

parser = argparse.ArgumentParser(description='Train a BPE tokenizer')
# --max-chars: 训练使用的最大字符数（达到后停止读取数据）
# 默认 2B（20 亿）字符，这个数量足以训练出高质量的 65K 词表
parser.add_argument('--max-chars', type=int, default=2_000_000_000,
                    help='Maximum characters to train on (default: 2B)')
# --doc-cap: 每篇文档截取的最大字符数
# 例如 doc_cap=10000 表示只取每篇文档的前 10000 个字符
# 目的：
#   1) 防止超长文档（如整本书）支配词频统计
#   2) 减少训练数据中的噪音（长文档的后半部分往往质量较低）
parser.add_argument('--doc-cap', type=int, default=10_000,
                    help='Maximum characters per document (default: 10,000)')
# --vocab-size: 目标词表大小
# 默认 32768 (2^15)，nanochat 官方推荐 65536 (2^16)
# 注意：其中的 9 个位置预留给特殊 token（<|bos|>、<|user_start|> 等）
parser.add_argument('--vocab-size', type=int, default=32768,
                    help='Vocabulary size (default: 32768 = 2^15)')
args = parser.parse_args()
print(f"max_chars: {args.max_chars:,}")
print(f"doc_cap: {args.doc_cap:,}")
print(f"vocab_size: {args.vocab_size:,}")

# -----------------------------------------------------------------------------
# Text iterator
# 文本迭代器：惰性地从 parquet 数据集中读取文本，逐篇 yield
# -----------------------------------------------------------------------------

def text_iterator():
    """
    文本迭代器——训练 BPE 的数据源。

    工作流程：
    1) Flatten the batches into a single iterator
       将 parquet 的批量数据"拍平"为逐文档的迭代器
    2) Crop every document to args.doc_cap characters
       每篇文档只取前 doc_cap 个字符
    3) Break when we've seen args.max_chars characters
       累计达到 max_chars 个字符后停止

    为什么设计成迭代器（而非一次性加载到内存）？
    - 训练数据可能高达几十 GB，无法全部放入内存
    - 惰性读取：每次只从磁盘读取一批数据，处理完就释放
    - BPE 训练需要多轮扫描文本，但不需要同时保留所有文本

    为什么需要截断文档（doc_cap）？
    - 防止超长文档主导词频统计。例如，如果训练集中有一本
      10 万字的书，它的词频模式会过度影响 BPE 合并决策
    - 让 tokenizer 接触到更多不同来源的文本开头部分
    """
    nchars = 0  # 已处理的字符总数（用于判断是否达到 max_chars 上限）
    for batch in parquets_iter_batched(split="train"):
        # parquets_iter_batched 每次返回一个 batch（list[str]）
        # 一个 batch 对应 parquet 文件中的一个 row_group，包含多条文档
        for doc in batch:
            # doc 是一篇完整的文本文档（字符串）
            doc_text = doc
            # 截断：只取前 doc_cap 个字符
            if len(doc_text) > args.doc_cap:
                doc_text = doc_text[:args.doc_cap]
            nchars += len(doc_text)
            yield doc_text  # 惰性产出：调用方拿到文本后才继续读下一篇
            if nchars > args.max_chars:
                # 达到字符上限，停止迭代
                # 注意：用的是 > 而非 >=，所以可能会超出一点（不超过最后一个文档的长度）
                return
# 创建迭代器实例
text_iter = text_iterator()

# -----------------------------------------------------------------------------
# Train the tokenizer
# 训练分词器
# -----------------------------------------------------------------------------
# 计时开始
t0 = time.time()
# 调用 RustBPETokenizer 的类方法 train_from_iterator 进行训练
# 内部流程：
#   1) 创建 rustbpe.Tokenizer()
#   2) 计算 BPE token 数量 = vocab_size - len(SPECIAL_TOKENS)
#      例如 vocab_size=65536, SPECIAL_TOKENS=9 → BPE 部分 = 65527
#   3) tokenizer.train_from_iterator(text_iter, 65527, pattern=SPLIT_PATTERN)
#      rustbpe 在 text_iter 上执行 BPE 合并，直到词表达到 65527 个 token
#   4) 获取 mergeable_ranks 和 pattern
#   5) 构造 tiktoken.Encoding 对象（用于后续的高效推理）
#   6) 返回 RustBPETokenizer 实例
tokenizer = RustBPETokenizer.train_from_iterator(text_iter, args.vocab_size)
t1 = time.time()
train_time = t1 - t0
print(f"Training time: {train_time:.2f}s")

# -----------------------------------------------------------------------------
# Save the tokenizer to disk
# 保存分词器到磁盘
# -----------------------------------------------------------------------------
# get_base_dir() 返回路径优先级：
#   1) 环境变量 NANOCHAT_BASE_DIR
#   2) 项目本地 .cache/nanochat/（如果存在）
#   3) ~/.cache/nanochat/
base_dir = get_base_dir()
tokenizer_dir = os.path.join(base_dir, "tokenizer")
# tokenizer.save() 会将 tiktoken.Encoding 对象 pickle 序列化到 tokenizer.pkl
tokenizer.save(tokenizer_dir)

# -----------------------------------------------------------------------------
# Quick inline sanity check
# 快速自检：编码→解码往返验证
# -----------------------------------------------------------------------------
# 这是一个完整性测试，确保分词器能正确编码并解码回原始文本
# 如果编解码不一致（例如某些字符被错误处理），这里会触发 AssertionError
test_text = """Hello world! This is a test.
Numbers: 123, 4567, 89
Contractions: I'm, you're, it's
Special chars: @#$%^&*()
Unicode: 你好世界 🌍"""
encoded = tokenizer.encode(test_text)    # 文本 → token ids
decoded = tokenizer.decode(encoded)      # token ids → 文本
assert decoded == test_text              # 必须完全一致
# 注意：这里 encode 用的是普通编码（不含特殊 token），
# 所以测试文本中不能包含特殊 token 字符串（如 "<|bos|>"），否则测试会失败。

# -----------------------------------------------------------------------------
# One more thing: we wish to cache a mapping from token id to number of bytes of that token
# for efficient evaluation of bits per byte. Unlike the typical mean loss, this
# allows us to report a loss that is invariant to the vocab size of the tokenizer.
# The bits per byte on the validation set is then one of the primary metrics we care about.
#
# 额外任务：为每个 token 缓存其对应的 UTF-8 字节数
# =============================================================================
# 为什么需要 token_bytes？
#
# 在评估模型性能时，常用的指标是"平均 loss"（交叉熵），但这个指标
# 依赖于词表大小——词表越大，每个 token 携带的信息越多，loss 自然更低，
# 导致不同词表大小的模型之间无法公平比较。
#
# 解决方案：bits per byte（每字节比特数）
#   1) 解码每个 token 得到原始文本
#   2) 统计文本的 UTF-8 字节数
#   3) 将 loss (nats) 除以字节数 → 得到 bits per byte
#
# bits per byte 是与词表大小无关的指标（因为它回到字节层面计算），
# 因此可以用来公平比较不同 tokenizer 和模型的压缩/预测能力。
# 这是 nanochat 在验证集上最关心的核心指标之一。
#
# 本段代码预先计算每个 token id 对应的字节数，保存为 token_bytes.pt，
# 后续评估时可以直接用 token_ids 索引查表，无需重复解码。
# =============================================================================

# 获取词表总大小（包含特殊 token）
vocab_size = tokenizer.get_vocab_size()
# 获取特殊 token 的字符串集合，如 {"<|bos|>", "<|user_start|>", ...}
# 特殊 token 的字节数设为 0——它们不携带信息，不应计入 bits per byte
special_set = set(tokenizer.get_special_tokens())
# 解码所有 token：遍历 0 ~ vocab_size-1，将每个 token id 转为字符串
# 例如: decode([1234]) → "the", decode([65527]) → "<|bos|>"
token_strings = [tokenizer.decode([token_id]) for token_id in range(vocab_size)]
# 计算每个 token 的 UTF-8 字节数
token_bytes = []
for token_id in range(vocab_size):
    token_str = token_strings[token_id]  # the Python string representation of this token
    if token_str in special_set:
        # 特殊 token 的字节数设为 0——它们是结构标记，不代表实际文本内容
        # 例: <|bos|> 只是一个序列分隔符，不代表任何语言信息
        token_bytes.append(0)  # special characters are not counted
    else:
        # 普通 token：计算其 UTF-8 编码的字节数
        # 例: "hello" → 5 字节, "你" → 3 字节, "🌍" → 4 字节
        id_bytes = len(token_str.encode("utf-8"))  # number of bytes that make up this token
        token_bytes.append(id_bytes)
# 转为 PyTorch 张量（int32 类型，CPU 存储）
# shape: [vocab_size]，例如 [65536]
# 内容示例: [1, 1, ..., 1, 5, 3, 2, ..., 0, 0, 0]
#             ↑ 256 个单字节 token    ↑ 多字节token  ↑ 9 个特殊token
token_bytes = torch.tensor(token_bytes, dtype=torch.int32, device='cpu')
# 保存到 tokenizer 目录
token_bytes_path = os.path.join(tokenizer_dir, "token_bytes.pt")
with open(token_bytes_path, "wb") as f:
    torch.save(token_bytes, f)
print(f"Saved token_bytes to {token_bytes_path}")

# Log to report
# 写入训练报告（供 report 系统汇总展示）
from nanochat.report import get_report
# 过滤掉字节数为 0 的特殊 token，计算普通 token 的字节数统计
token_bytes_nonzero = (token_bytes[token_bytes > 0]).to(dtype=torch.float32)
# get_report() 返回 Report 或 DummyReport 对象
# - rank 0（主进程）→ Report 实例，实际写入报告文件
# - 其他 rank（DDP 多卡训练）→ DummyReport 实例，什么都不做
# 日志会写入 base_dir/report/tokenizer-training.md
get_report().log(section="Tokenizer training", data=[
    vars(args),  # argparse command line arguments
                  # 展开为: {"max_chars": 2000000000, "doc_cap": 10000, "vocab_size": 32768}
    {"train_time": train_time},  # 训练耗时（秒）
    {"num_special_tokens": len(special_set)},  # 特殊 token 数量（固定为 9）
    {
        # 普通 token 的字节数统计
        "token_bytes_min": int(token_bytes_nonzero.min().item()),    # 最小字节数（通常为 1，单字节 token）
        "token_bytes_max": int(token_bytes_nonzero.max().item()),    # 最大字节数（最长的合并 token）
        "token_bytes_mean": token_bytes_nonzero.mean().item(),       # 平均字节数（反映 token 的平均"信息密度"）
        "token_bytes_std": token_bytes_nonzero.std().item(),         # 字节数标准差（反映 token 长度的离散程度）
    }
])
# 训练完成！
# 输出文件:
#   {base_dir}/tokenizer/tokenizer.pkl    — 分词器（推理用）
#   {base_dir}/tokenizer/token_bytes.pt   — token 字节数映射（评估用）
#   {base_dir}/report/tokenizer-training.md — 训练报告
