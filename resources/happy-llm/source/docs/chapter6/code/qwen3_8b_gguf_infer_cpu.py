"""
使用 llama.cpp 在本机 CPU 上加载 Qwen3-8B 的 GGUF Q4_K_M 量化模型并做一次推理。

运行位置：建议在项目根目录 E:/Project/LLM/LearnLLM 下运行。

运行命令：
    python resources/happy-llm/source/docs/chapter6/code/qwen3_8b_gguf_infer_cpu.py

依赖说明：
    本脚本使用 llama-cpp-python，它是 llama.cpp 的 Python 绑定。
    如果运行时报错找不到 llama_cpp，请先安装：

    pip install llama-cpp-python

    如果 Windows + Python 3.14 安装失败，建议新建 Python 3.11 环境后再安装：

    python3.11 -m venv .venv-llama-cpp
    source .venv-llama-cpp/bin/activate
    pip install --upgrade pip
    pip install llama-cpp-python

模型文件：
    resources/happy-llm/source/docs/chapter6/models/Qwen3-8B-GGUF/Qwen3-8B-Q4_K_M.gguf

注意：
    - Qwen3-8B Q4_K_M 文件约 4.68GB，你的 32GB 内存可以尝试 CPU 推理。
    - CPU 推理会比 GPU 慢很多，首次测试请控制 prompt 和 max_tokens。
    - n_ctx 越大，KV cache 占用越高；CPU 首测建议使用 2048。
"""

from __future__ import annotations

from pathlib import Path


# GGUF 模型的本地路径。
# 这里使用项目根目录下的相对路径；如果你从其他目录运行脚本，可以改成绝对路径。
MODEL_PATH = Path(
    "resources/happy-llm/source/docs/chapter6/models/Qwen3-8B-GGUF/Qwen3-8B-Q4_K_M.gguf"
)


# CPU 推理上下文长度。
# 含义：模型一次最多能看到多少 token 的上下文。
# 值越大，能处理的文本越长，但内存占用也越高；本机 CPU 首测推荐 2048。
N_CTX = 2048


# CPU 线程数。
# 你的机器检测到大约 16 个逻辑 CPU，这里先用 12，给系统和其他程序留一点余量。
# 如果想尽量提高速度，可以改成 16；如果电脑卡顿，可以改成 8。
N_THREADS = 12


# 生成的新 token 数。
# CPU 上 8B 模型生成速度会比较慢，所以第一次先设短一点。
MAX_TOKENS = 128


# Qwen3 支持 thinking 模式；CPU 首测建议关闭，减少生成长度和等待时间。
ENABLE_THINKING = False


# 测试问题。后续你可以直接修改这里来测试其他问题。
PROMPT = "你是什么模型,你有Agent能力吗"


def build_qwen3_prompt(user_prompt: str, enable_thinking: bool = False) -> str:
    """构造 Qwen3 Instruct/Chat 风格的输入文本。

    使用 transformers 时通常可以调用 tokenizer.apply_chat_template。
    但 GGUF + llama.cpp 推理时，我们通常直接拼接 chat template。

    Qwen 系列常见对话格式大致如下：
        <|im_start|>user
        用户问题<|im_end|>
        <|im_start|>assistant

    对 Qwen3，如果希望关闭 thinking，官方示例通常可以在 prompt 中加入 /no_think。
    这里把 /no_think 附加到用户问题后面，减少 CPU 推理时的生成量。
    """

    thinking_flag = "" if enable_thinking else " /no_think"
    return (
        "<|im_start|>user\n"
        f"{user_prompt}{thinking_flag}<|im_end|>\n"
        "<|im_start|>assistant\n"
    )


def main() -> None:
    """加载本地 GGUF 模型，并执行一次 CPU 推理。"""

    # llama-cpp-python 不是 transformers 的一部分，需要单独安装。
    # 把 import 放到 main 里，是为了在依赖缺失时给出更清晰的提示。
    try:
        from llama_cpp import Llama
    except ImportError as exc:
        raise SystemExit(
            "未检测到 llama-cpp-python。\n"
            "请先安装 llama.cpp 的 Python 绑定：\n\n"
            "    pip install llama-cpp-python\n\n"
            "如果你当前 Python 3.14 安装失败，建议使用 Python 3.10/3.11 新建虚拟环境后再安装。"
        ) from exc

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"没有找到模型文件：{MODEL_PATH}\n"
            "请确认 Qwen3-8B-Q4_K_M.gguf 已下载到 chapter6/models 目录。"
        )

    print("使用 llama.cpp / llama-cpp-python 加载 GGUF 模型")
    print(f"模型路径：{MODEL_PATH}")
    print(f"上下文长度 n_ctx：{N_CTX}")
    print(f"CPU 线程数 n_threads：{N_THREADS}")
    print(f"最大生成 token 数：{MAX_TOKENS}")

    # 初始化 llama.cpp 模型。
    # model_path：GGUF 文件路径。
    # n_ctx：上下文长度。
    # n_threads：CPU 推理线程数。
    # n_gpu_layers=0：明确使用纯 CPU，不把层卸载到 GPU。
    # verbose=False：减少 llama.cpp 底层日志输出，方便阅读结果。
    llm = Llama(
        model_path=str(MODEL_PATH),
        n_ctx=N_CTX,
        n_threads=N_THREADS,
        n_gpu_layers=0,
        verbose=False,
    )

    prompt = build_qwen3_prompt(PROMPT, enable_thinking=ENABLE_THINKING)

    print("\n========== Prompt ==========")
    print(PROMPT)
    print("\n正在生成，8B 模型在 CPU 上可能需要等待一段时间...\n")

    # 调用 llama.cpp 的 completion API。
    # stop 用于在模型输出到下一轮特殊标记时停止生成。
    output = llm(
        prompt,
        max_tokens=MAX_TOKENS,
        temperature=0.7,
        top_p=0.9,
        stop=["<|im_end|>", "<|im_start|>"],
        echo=False,
    )

    # llama-cpp-python 返回的是 OpenAI completion 风格结构：
    # {"choices": [{"text": "..."}], ...}
    answer = output["choices"][0]["text"].strip()

    print("========== 模型回答 ==========")
    print(answer)


if __name__ == "__main__":
    main()
