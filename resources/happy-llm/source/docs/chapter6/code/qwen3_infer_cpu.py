"""
在本机 CPU 上加载本地的 Qwen3-0.6B 模型并做一次简单推理。

使用方式：
    python resources/happy-llm/source/docs/chapter6/code/qwen3_infer_cpu.py

前置步骤：
    1. 先把 Qwen/Qwen3-0.6B 下载到本地目录：resources/happy-llm/source/docs/chapter7/models/Qwen3-0.6B
    2. 确保当前 Python 环境已经安装 torch 和 transformers

说明：
    - 本脚本专门面向“没有 CUDA / GPU 的本机 CPU 环境”。
    - CPU 推理速度会明显慢于 GPU，因此默认只生成较短回答。
    - 如果你后续换成 GPU 环境，可以再把 device 改为 cuda，并使用更合适的数据类型。
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


# 本地模型目录。
# 这里不要写成 "Qwen/Qwen3-0.6B"，否则 transformers 会尝试从 Hugging Face 在线下载。
# 我们先把模型下载到 resources/happy-llm/source/docs/chapter6/models/Qwen3-0.6B，再从这个本地目录加载。
MODEL_PATH = "resources/happy-llm/source/docs/chapter6/models/Qwen3-0.6B"


# CPU 上建议生成短一点，方便快速验证模型是否能正常加载和输出。
# 如果你的机器内存较大、可以接受更慢的速度，可以把这个值改成 128、256。
MAX_NEW_TOKENS = 96


# Qwen3 默认支持 thinking 模式。
# 在 CPU 上打开 thinking 会生成更多 token，速度会更慢。
# 这里先关闭 thinking，只验证普通问答推理流程。
ENABLE_THINKING = False


# 用一个简单中文问题做烟雾测试。
# 后续你可以直接修改这个 prompt，测试其他问题。
PROMPT = "习近平是谁。"


def main() -> None:
    """加载 tokenizer、加载模型，并在 CPU 上执行一次短文本生成。"""

    # 当前环境是 CPU 版 PyTorch，因此 device 固定为 cpu。
    # 如果以后换成 GPU，可以改成：device = "cuda"
    device = "cpu"

    print(f"使用设备：{device}")
    print(f"本地模型目录：{MODEL_PATH}")

    # 1. 加载 tokenizer。
    # tokenizer 负责把人类可读文本转换为模型能理解的 token id，
    # 也负责把模型输出的 token id 解码回文本。
    print("\n[1/4] 正在加载 tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_PATH,
        trust_remote_code=True,
    )

    # 2. 加载模型。
    # torch_dtype=torch.float32：CPU 环境下最稳妥，兼容性最好。
    # device_map=None：不使用 accelerate 自动切分设备，避免 CPU-only 环境出现复杂映射问题。
    # low_cpu_mem_usage=False：当前环境未必安装 accelerate；关闭后依赖更少。
    print("[2/4] 正在加载模型，这一步在 CPU 上可能需要一些时间...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.float32,
        device_map=None,
        low_cpu_mem_usage=False,
        trust_remote_code=True,
    )

    # 把模型明确放到 CPU，并切换到 eval 模式。
    # eval 模式会关闭 dropout 等训练专用行为，适合推理。
    model.to(device)
    model.eval()

    # 3. 构造 Qwen Chat 模板输入。
    # 对聊天模型来说，不建议直接把原始 prompt 喂给模型，
    # 更推荐使用 tokenizer.apply_chat_template 生成模型训练时熟悉的对话格式。
    messages = [
        {"role": "user", "content": PROMPT},
    ]

    print("[3/4] 正在构造模型输入...")

    # Qwen3 tokenizer 支持 enable_thinking 参数。
    # 为了兼容不同 transformers/tokenizer 版本，这里做一个 try/except：
    # - 如果当前 tokenizer 支持 enable_thinking，就关闭 thinking；
    # - 如果不支持，就退回到普通 chat_template 调用。
    try:
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=ENABLE_THINKING,
        )
    except TypeError:
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

    # return_tensors="pt" 表示返回 PyTorch Tensor。
    # 再通过 .to(device) 放到 CPU，与模型所在设备保持一致。
    inputs = tokenizer(text, return_tensors="pt").to(device)

    # 记录输入长度，后面只解码新增生成部分，避免把用户问题也打印出来。
    input_token_count = inputs["input_ids"].shape[-1]

    # 4. 执行生成。
    # torch.no_grad() 表示不计算梯度，节省内存和计算；推理时必须这样做。
    # do_sample=False 使用贪心解码，输出更稳定，也更适合首次烟雾测试。
    print("[4/4] 正在生成回答，CPU 上请耐心等待...")
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    # 只取模型新生成的 token，不包含输入 prompt。
    generated_ids = output_ids[0][input_token_count:]

    # skip_special_tokens=True 会去掉 <|im_end|> 等特殊 token，让输出更干净。
    response = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

    print("\n========== Prompt ==========")
    print(PROMPT)
    print("\n========== 模型回答 ==========")
    print(response)


if __name__ == "__main__":
    main()
