"""
通用本地 Qwen3-8B Agent 脚本。

这个脚本把已经下载好的 Qwen3-8B GGUF Q4_K_M 模型包装成一个简单的本地 Agent，
可以在 CPU 上完成以下几类任务：

1. 普通问答：
   python resources/happy-llm/source/docs/chapter6/code/qwen3_8b_local_agent.py "什么是 LoRA？"

2. 带文件上下文问答：
   python resources/happy-llm/source/docs/chapter6/code/qwen3_8b_local_agent.py \
       "解释这个脚本的作用" \
       --file resources/happy-llm/source/docs/chapter6/code/pretrain.py

3. 简单本地 RAG：
   python resources/happy-llm/source/docs/chapter6/code/qwen3_8b_local_agent.py \
       "第六章里 Pretrain 和 SFT 有什么区别？" \
       --rag-dir resources/happy-llm/source/docs/chapter6 \
       --top-k 4

依赖：
   pip install llama-cpp-python

模型文件默认位置：
   resources/happy-llm/source/docs/chapter6/models/Qwen3-8B-GGUF/Qwen3-8B-Q4_K_M.gguf

说明：
   - 这是一个“本地辅助 Agent”，不是 Claude Code 的主模型替代品。
   - 它默认只读文件，不会修改、删除或执行外部命令，适合作为安全的学习/总结/RAG 助手。
   - CPU 上 8B 模型推理速度较慢，第一次测试建议使用较短 prompt 和较小 max-tokens。
"""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


# 尽量让 Windows 终端输出中文时不要乱码。
# 某些终端仍然需要在运行时设置：PYTHONIOENCODING=utf-8
try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass


# 当前文件位于：chapter6/code/qwen3_8b_local_agent.py
# Path(__file__).resolve().parents[1] 对应 chapter6 目录。
CHAPTER6_DIR = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_PATH = CHAPTER6_DIR / "models" / "Qwen3-8B-GGUF" / "Qwen3-8B-Q4_K_M.gguf"


# RAG 默认扫描的文本文件类型。
# 这里覆盖教程、代码和常见配置文件；不扫描二进制和大模型文件。
DEFAULT_RAG_EXTENSIONS = {
    ".md",
    ".txt",
    ".py",
    ".json",
    ".jsonl",
    ".yaml",
    ".yml",
    ".sh",
}


# 扫描目录时跳过这些目录，避免误读模型、缓存、虚拟环境、git 元数据等大目录。
SKIP_DIR_NAMES = {
    ".git",
    ".cache",
    ".venv",
    "venv",
    "env",
    "__pycache__",
    ".ipynb_checkpoints",
    "models",
    "checkpoints",
    "output",
    "outputs",
    "runs",
    "wandb",
}


@dataclass
class TextChunk:
    """RAG 检索时使用的文本块。"""

    path: Path
    index: int
    text: str
    score: int = 0


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""

    parser = argparse.ArgumentParser(
        description="使用本地 Qwen3-8B GGUF Q4_K_M 模型进行 CPU 问答、文件问答和简单 RAG。"
    )

    parser.add_argument(
        "prompt",
        nargs="*",
        help="用户问题。如果不提供，会从标准输入读取。",
    )
    parser.add_argument(
        "--model",
        default=str(DEFAULT_MODEL_PATH),
        help="GGUF 模型文件路径。默认使用 chapter6/models 下的 Qwen3-8B-Q4_K_M.gguf。",
    )
    parser.add_argument(
        "--file",
        action="append",
        default=[],
        help="附加到上下文中的本地文件路径。可以重复传入多个 --file。",
    )
    parser.add_argument(
        "--rag-dir",
        action="append",
        default=[],
        help="启用简单 RAG：扫描指定目录，并检索与问题最相关的文本块。可以重复传入多个 --rag-dir。",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=4,
        help="RAG 模式下选取的相关文本块数量。默认 4。",
    )
    parser.add_argument(
        "--mode",
        choices=["assistant", "study", "code", "rag"],
        default="assistant",
        help="Agent 角色模式：assistant 通用助手；study 学习助手；code 代码解释；rag 基于检索上下文回答。",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=256,
        help="最多生成的新 token 数。CPU 首测建议 64~256。默认 256。",
    )
    parser.add_argument(
        "--ctx",
        type=int,
        default=2048,
        help="llama.cpp 上下文长度 n_ctx。越大越耗内存。默认 2048。",
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=12,
        help="CPU 推理线程数。你的机器可尝试 8、12、16。默认 12。",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="采样温度。越低越稳定，越高越发散。默认 0.7。",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=0.9,
        help="核采样 top_p。默认 0.9。",
    )
    parser.add_argument(
        "--think",
        action="store_true",
        help="启用 Qwen3 thinking。CPU 上会更慢；默认关闭。",
    )
    parser.add_argument(
        "--show-context",
        action="store_true",
        help="打印最终发送给模型的上下文，方便调试 RAG 或文件问答。",
    )

    return parser.parse_args()


def get_prompt(args: argparse.Namespace) -> str:
    """从命令行参数或标准输入获取用户问题。"""

    if args.prompt:
        return " ".join(args.prompt).strip()

    # 支持管道输入：echo "什么是 RAG" | python qwen3_8b_local_agent.py
    data = sys.stdin.read().strip()
    if data:
        return data

    raise SystemExit("请提供问题，例如：python qwen3_8b_local_agent.py \"什么是 RAG？\"")


def read_text_file(path: Path, max_chars: int = 12_000) -> str:
    """安全读取文本文件，并限制最大字符数。

    限制字符数是为了避免把超大文件一次性塞进上下文，导致 CPU 推理变慢或超过 n_ctx。
    """

    try:
        text = path.read_text(encoding="utf-8", errors="ignore")
    except Exception as exc:
        return f"[读取失败：{path}，原因：{exc}]"

    if len(text) > max_chars:
        return text[:max_chars] + f"\n\n[文件过长，已截断，只保留前 {max_chars} 个字符]"
    return text


def iter_candidate_files(dirs: Iterable[Path]) -> Iterable[Path]:
    """遍历 RAG 候选文件。"""

    for root in dirs:
        if not root.exists():
            continue
        if root.is_file():
            if root.suffix.lower() in DEFAULT_RAG_EXTENSIONS:
                yield root
            continue

        for path in root.rglob("*"):
            if not path.is_file():
                continue
            if path.suffix.lower() not in DEFAULT_RAG_EXTENSIONS:
                continue
            if any(part in SKIP_DIR_NAMES for part in path.parts):
                continue
            yield path


def split_text(text: str, chunk_size: int = 1_200, overlap: int = 150) -> list[str]:
    """把长文本切分成带重叠的短块。

    这里使用字符级切分，简单稳妥；中文、英文、代码都能处理。
    overlap 可以保留块之间的少量上下文，避免关键句子被切断。
    """

    text = text.strip()
    if not text:
        return []
    if len(text) <= chunk_size:
        return [text]

    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunks.append(text[start:end])
        if end == len(text):
            break
        start = max(0, end - overlap)
    return chunks


def query_terms(query: str) -> set[str]:
    """提取用于简单检索的关键词。

    为了避免额外依赖，这里不用 jieba 或向量模型，只做轻量规则：
    - 英文、数字、下划线片段作为词；
    - 长度大于 1 的连续中文片段作为词；
    - 同时加入部分中文 2 字/3 字片段，提升中文查询召回。
    """

    terms: set[str] = set()
    lower = query.lower()

    for token in re.findall(r"[a-zA-Z0-9_\-]+", lower):
        if len(token) >= 2:
            terms.add(token)

    chinese_segments = re.findall(r"[一-鿿]+", query)
    for segment in chinese_segments:
        if len(segment) >= 2:
            terms.add(segment)
        # 加入有限数量的 2-gram / 3-gram，避免集合过大。
        for n in (2, 3):
            for i in range(max(0, len(segment) - n + 1)):
                terms.add(segment[i : i + n])

    return terms


def score_chunk(chunk_text: str, terms: set[str]) -> int:
    """根据关键词重叠给文本块打分。"""

    if not terms:
        return 0
    lower = chunk_text.lower()
    score = 0
    for term in terms:
        if term and term.lower() in lower:
            # 出现一次给基础分；出现多次再给少量加分，避免长文本天然分高太多。
            count = lower.count(term.lower())
            score += 2 + min(count, 3)
    return score


def retrieve_chunks(query: str, rag_dirs: list[str], top_k: int) -> list[TextChunk]:
    """从指定目录做一个简单关键词 RAG 检索。"""

    dirs = [Path(item) for item in rag_dirs]
    terms = query_terms(query)
    chunks: list[TextChunk] = []

    for path in iter_candidate_files(dirs):
        text = read_text_file(path, max_chars=80_000)
        if text.startswith("[读取失败"):
            continue
        for idx, chunk_text in enumerate(split_text(text)):
            score = score_chunk(chunk_text, terms)
            if score > 0:
                chunks.append(TextChunk(path=path, index=idx, text=chunk_text, score=score))

    chunks.sort(key=lambda item: item.score, reverse=True)
    return chunks[: max(0, top_k)]


def mode_system_prompt(mode: str) -> str:
    """根据模式生成系统提示词。"""

    if mode == "study":
        return (
            "你是一个本地运行的 LearnLLM 学习助手。请用中文回答，"
            "重点解释概念、步骤和初学者容易混淆的点。回答要清晰、准确、适合学习笔记。"
        )
    if mode == "code":
        return (
            "你是一个本地运行的代码解释助手。请用中文回答，"
            "优先说明代码目的、核心流程、关键函数、重要参数和可能的注意事项。"
        )
    if mode == "rag":
        return (
            "你是一个本地 RAG 问答助手。请优先依据给定的检索上下文回答。"
            "如果上下文不足，请明确说明不足，不要编造来源。"
        )
    return (
        "你是一个本地运行的中文助手。请回答得简洁、准确、可操作。"
        "如果问题涉及代码或大模型学习，请给出清晰步骤。"
    )


def build_context(args: argparse.Namespace, user_prompt: str) -> str:
    """构造文件上下文和 RAG 上下文。"""

    context_parts: list[str] = []

    # 1. 显式文件上下文：用户指定 --file 时，直接读取这些文件。
    for file_name in args.file:
        path = Path(file_name)
        content = read_text_file(path)
        context_parts.append(
            f"## 文件上下文：{path}\n\n```text\n{content}\n```"
        )

    # 2. RAG 上下文：用户指定 --rag-dir 时，从目录中检索相关文本块。
    if args.rag_dir:
        chunks = retrieve_chunks(user_prompt, args.rag_dir, args.top_k)
        if chunks:
            rag_lines = ["## 检索到的相关上下文"]
            for rank, chunk in enumerate(chunks, start=1):
                rag_lines.append(
                    f"\n### 片段 {rank} | score={chunk.score} | {chunk.path} | chunk={chunk.index}\n"
                    f"```text\n{chunk.text}\n```"
                )
            context_parts.append("\n".join(rag_lines))
        else:
            context_parts.append("## 检索到的相关上下文\n\n[未检索到明显相关的文本片段]")

    return "\n\n".join(context_parts).strip()


def build_qwen3_prompt(system_prompt: str, user_prompt: str, context: str, enable_thinking: bool) -> str:
    """构造 Qwen3 chat template 风格的输入。"""

    if context:
        final_user_prompt = (
            "请结合下面的上下文回答问题。\n\n"
            f"{context}\n\n"
            "## 用户问题\n"
            f"{user_prompt}"
        )
    else:
        final_user_prompt = user_prompt

    # Qwen3 可通过 /no_think 关闭 thinking。CPU 上默认关闭，减少等待时间。
    if not enable_thinking:
        final_user_prompt = f"{final_user_prompt} /no_think"

    return (
        "<|im_start|>system\n"
        f"{system_prompt}<|im_end|>\n"
        "<|im_start|>user\n"
        f"{final_user_prompt}<|im_end|>\n"
        "<|im_start|>assistant\n"
    )


def clean_answer(answer: str) -> str:
    """清理模型输出中可能出现的空 thinking 标记。"""

    answer = answer.strip()
    # 常见输出是：<think>\n\n</think>\n\n正文
    answer = re.sub(r"^<think>\s*</think>\s*", "", answer, flags=re.DOTALL)
    return answer.strip()


def main() -> None:
    args = parse_args()
    user_prompt = get_prompt(args)
    model_path = Path(args.model)

    if not model_path.exists():
        raise SystemExit(f"模型文件不存在：{model_path}")

    try:
        from llama_cpp import Llama
    except ImportError as exc:
        raise SystemExit(
            "未检测到 llama-cpp-python，请先安装：\n"
            "    pip install llama-cpp-python"
        ) from exc

    system_prompt = mode_system_prompt(args.mode)
    context = build_context(args, user_prompt)
    prompt = build_qwen3_prompt(system_prompt, user_prompt, context, args.think)

    if args.show_context:
        print("========== 发送给模型的完整 Prompt ==========")
        print(prompt)
        print("========== Prompt 结束 ==========")

    print("加载本地 Qwen3-8B GGUF 模型...")
    print(f"模型：{model_path}")
    print(f"模式：{args.mode} | n_ctx={args.ctx} | threads={args.threads} | max_tokens={args.max_tokens}")

    llm = Llama(
        model_path=str(model_path),
        n_ctx=args.ctx,
        n_threads=args.threads,
        n_gpu_layers=0,
        verbose=False,
    )

    print("\n正在生成回答...\n")
    output = llm(
        prompt,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        stop=["<|im_end|>", "<|im_start|>"],
        echo=False,
    )

    answer = clean_answer(output["choices"][0]["text"])

    print("========== Qwen3-8B 本地 Agent 回答 ==========")
    print(answer)


if __name__ == "__main__":
    main()
