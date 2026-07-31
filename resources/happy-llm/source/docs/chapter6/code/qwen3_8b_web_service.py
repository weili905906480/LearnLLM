"""
Qwen3-8B GGUF 本地 Web 服务第一版。

功能：
1. 启动时只加载一次 GGUF 模型
2. 提供 GET /health
3. 提供 POST /chat
4. 提供 GET / 的最小网页
5. 默认只监听 127.0.0.1
6. 默认关闭 Qwen3 thinking，减少 CPU 推理等待时间
7. 使用全局锁串行化生成请求，避免 CPU 被多个请求同时打满

运行方式：
    python resources/happy-llm/source/docs/chapter6/code/qwen3_8b_web_service.py

依赖：
    pip install llama-cpp-python fastapi uvicorn
"""

from __future__ import annotations

import argparse
import re
import sys
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

try:
    from fastapi import FastAPI, HTTPException
    from fastapi.responses import HTMLResponse, JSONResponse
    from pydantic import BaseModel, Field
except ImportError as exc:  # pragma: no cover - 依赖缺失时直接失败
    raise SystemExit(
        "缺少 fastapi / pydantic 依赖，请先安装：\n"
        "    pip install fastapi uvicorn"
    ) from exc


try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass


DEFAULT_MODEL_PATH = Path(
    "resources/happy-llm/source/docs/chapter6/models/"
    "Qwen3-8B-GGUF/Qwen3-8B-Q4_K_M.gguf"
)
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8000
DEFAULT_CTX = 2048
DEFAULT_THREADS = 12
DEFAULT_MAX_TOKENS = 256
DEFAULT_TEMPERATURE = 0.7
DEFAULT_TOP_P = 0.9

GENERATION_LOCK = threading.Lock()


class ChatRequest(BaseModel):
    """/chat 接口的请求体。"""

    prompt: str = Field(..., description="用户问题")
    max_tokens: int = Field(default=DEFAULT_MAX_TOKENS, ge=1, le=4096)
    temperature: float = Field(default=DEFAULT_TEMPERATURE, ge=0.0, le=2.0)
    top_p: float = Field(default=DEFAULT_TOP_P, ge=0.0, le=1.0)
    think: bool = Field(default=False, description="是否启用 Qwen3 thinking")


class ChatResponse(BaseModel):
    """/chat 接口的响应体。"""

    answer: str
    prompt: str
    model: str


class OpenAIChatMessage(BaseModel):
    """OpenAI chat.completions 消息格式。"""

    role: str
    content: str


class OpenAIChatCompletionRequest(BaseModel):
    """/v1/chat/completions 请求体。"""

    model: str | None = None
    messages: list[OpenAIChatMessage] = Field(..., min_length=1)
    max_tokens: int | None = Field(default=None, ge=1, le=4096)
    temperature: float = Field(default=DEFAULT_TEMPERATURE, ge=0.0, le=2.0)
    top_p: float = Field(default=DEFAULT_TOP_P, ge=0.0, le=1.0)
    stream: bool = False
    think: bool = Field(default=False, description="是否启用 Qwen3 thinking")


class WebServiceConfig(BaseModel):
    """Web 服务启动配置。"""

    model_path: Path = DEFAULT_MODEL_PATH
    host: str = DEFAULT_HOST
    port: int = DEFAULT_PORT
    ctx: int = DEFAULT_CTX
    threads: int = DEFAULT_THREADS


def mode_system_prompt() -> str:
    """第一版固定使用通用中文助手提示词。"""

    return (
        "你是一个本地运行的中文助手。"
        "请回答得清晰、准确、可操作。"
    )


def build_qwen3_prompt(system_prompt: str, user_prompt: str, think: bool) -> str:
    """拼接 Qwen3 Chat 模板。"""

    final_user_prompt = user_prompt.strip()
    if not think:
        final_user_prompt = f"{final_user_prompt} /no_think"

    return (
        "<|im_start|>system\n"
        f"{system_prompt}<|im_end|>\n"
        "<|im_start|>user\n"
        f"{final_user_prompt}<|im_end|>\n"
        "<|im_start|>assistant\n"
    )


def build_qwen3_prompt_from_messages(
    messages: list[OpenAIChatMessage],
    think: bool,
) -> str:
    """把 OpenAI chat messages 转成 Qwen3 Chat 模板。"""

    prompt_parts = []
    for index, message in enumerate(messages):
        role = message.role.strip().lower()
        if role not in {"system", "user", "assistant"}:
            raise ValueError(f"不支持的消息角色：{message.role}")

        content = message.content.strip()
        if not think and role == "user" and index == len(messages) - 1:
            content = f"{content} /no_think"

        prompt_parts.append(f"<|im_start|>{role}\n{content}<|im_end|>")

    prompt_parts.append("<|im_start|>assistant\n")
    return "\n".join(prompt_parts)


def clean_answer(answer: str) -> str:
    """清理模型输出里的思考标签。"""

    answer = answer.strip()
    answer = re.sub(r"^<think>\s*</think>\s*", "", answer, flags=re.DOTALL)
    return answer.strip()


def build_homepage() -> str:
    """返回一个最小可用的聊天页面。"""

    return """
<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Qwen3-8B 本地聊天</title>
  <style>
    body { font-family: sans-serif; margin: 24px; max-width: 960px; }
    textarea, input { width: 100%; box-sizing: border-box; font: inherit; }
    textarea { min-height: 160px; }
    .row { margin: 12px 0; }
    .grid { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 12px; }
    button { padding: 10px 16px; font: inherit; cursor: pointer; }
    pre { white-space: pre-wrap; word-break: break-word; background: #f6f6f6; padding: 12px; }
    label { display: block; margin-bottom: 6px; }
  </style>
</head>
<body>
  <h1>Qwen3-8B 本地聊天</h1>
  <div class="row">
    <label for="prompt">问题</label>
    <textarea id="prompt" placeholder="输入你的问题"></textarea>
  </div>
  <div class="grid">
    <div>
      <label for="max_tokens">max_tokens</label>
      <input id="max_tokens" type="number" value="256" min="1" max="4096" />
    </div>
    <div>
      <label for="temperature">temperature</label>
      <input id="temperature" type="number" value="0.7" min="0" max="2" step="0.1" />
    </div>
  </div>
  <div class="grid">
    <div>
      <label for="top_p">top_p</label>
      <input id="top_p" type="number" value="0.9" min="0" max="1" step="0.05" />
    </div>
    <div style="display:flex; align-items:end;">
      <label style="display:flex; align-items:center; gap:8px; margin:0;">
        <input id="think" type="checkbox" />
        启用 thinking
      </label>
    </div>
  </div>
  <div class="row">
    <button id="send">发送</button>
  </div>
  <div class="row">
    <label>回答</label>
    <pre id="output"></pre>
  </div>
  <script>
    const send = async () => {
      const payload = {
        prompt: document.getElementById('prompt').value,
        max_tokens: Number(document.getElementById('max_tokens').value),
        temperature: Number(document.getElementById('temperature').value),
        top_p: Number(document.getElementById('top_p').value),
        think: document.getElementById('think').checked,
      };
      document.getElementById('output').textContent = '生成中...';
      const resp = await fetch('/chat', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify(payload),
      });
      const data = await resp.json();
      document.getElementById('output').textContent =
        resp.ok ? data.answer : (data.detail || JSON.stringify(data, null, 2));
    };
    document.getElementById('send').addEventListener('click', send);
  </script>
</body>
</html>
""".strip()


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""

    parser = argparse.ArgumentParser(description="Qwen3-8B GGUF 本地 Web 服务第一版")
    parser.add_argument(
        "--model",
        default=str(DEFAULT_MODEL_PATH),
        help="GGUF 模型文件路径",
    )
    parser.add_argument(
        "--host",
        default=DEFAULT_HOST,
        help="监听地址，默认只监听本机",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=DEFAULT_PORT,
        help="监听端口",
    )
    parser.add_argument(
        "--ctx",
        type=int,
        default=DEFAULT_CTX,
        help="上下文长度 n_ctx",
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=DEFAULT_THREADS,
        help="CPU 推理线程数",
    )
    return parser.parse_args()


def load_llm(config: WebServiceConfig) -> Any:
    """加载 llama.cpp 模型。"""

    try:
        from llama_cpp import Llama
    except ImportError as exc:
        raise SystemExit(
            "缺少 llama-cpp-python 依赖，请先安装：\n"
            "    pip install llama-cpp-python"
        ) from exc

    if not config.model_path.exists():
        raise FileNotFoundError(f"模型文件不存在：{config.model_path}")

    print("正在加载本地 Qwen3-8B GGUF 模型...")
    print(f"模型：{config.model_path}")
    print(f"n_ctx={config.ctx} | n_threads={config.threads}")

    return Llama(
        model_path=str(config.model_path),
        n_ctx=config.ctx,
        n_threads=config.threads,
        n_gpu_layers=0,
        verbose=False,
    )


def create_app(config: WebServiceConfig) -> FastAPI:
    """创建 FastAPI 应用。"""

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.config = config
        app.state.model_loaded = False
        app.state.llm = None
        app.state.generation_lock = GENERATION_LOCK

        app.state.llm = load_llm(config)
        app.state.model_loaded = True
        yield

    app = FastAPI(title="Qwen3-8B Local Web Service", lifespan=lifespan)

    @app.get("/", response_class=HTMLResponse)
    def index() -> str:
        return build_homepage()

    @app.get("/health")
    def health() -> dict[str, Any]:
        return {
            "status": "ok",
            "model_loaded": bool(app.state.model_loaded),
            "model_path": str(config.model_path),
        }

    @app.post("/chat", response_model=ChatResponse)
    def chat(req: ChatRequest) -> ChatResponse:
        if not app.state.model_loaded or app.state.llm is None:
            raise HTTPException(status_code=503, detail="model not loaded")

        system_prompt = mode_system_prompt()
        prompt = build_qwen3_prompt(system_prompt, req.prompt, req.think)

        with app.state.generation_lock:
            output = app.state.llm(
                prompt,
                max_tokens=req.max_tokens or DEFAULT_MAX_TOKENS,
                temperature=req.temperature,
                top_p=req.top_p,
                stop=["<|im_end|>", "<|im_start|>"],
                echo=False,
            )

        answer = clean_answer(output["choices"][0]["text"])
        return ChatResponse(answer=answer, prompt=req.prompt, model=config.model_path.name)

    @app.post("/v1/chat/completions")
    def openai_chat_completions(req: OpenAIChatCompletionRequest) -> dict[str, Any]:
        if req.stream:
            raise HTTPException(status_code=400, detail="stream is not supported")

        if not app.state.model_loaded or app.state.llm is None:
            raise HTTPException(status_code=503, detail="model not loaded")

        try:
            prompt = build_qwen3_prompt_from_messages(req.messages, req.think)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        with app.state.generation_lock:
            output = app.state.llm(
                prompt,
                max_tokens=req.max_tokens or DEFAULT_MAX_TOKENS,
                temperature=req.temperature,
                top_p=req.top_p,
                stop=["<|im_end|>", "<|im_start|>"],
                echo=False,
            )

        answer = clean_answer(output["choices"][0]["text"])
        prompt_tokens = max(1, len(prompt) // 4)
        completion_tokens = max(1, len(answer) // 4)

        return {
            "id": f"chatcmpl-{int(time.time() * 1000)}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": config.model_path.name,
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": answer,
                    },
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
            },
        }

    return app


def main() -> None:
    """命令行启动入口。"""

    args = parse_args()
    config = WebServiceConfig(
        model_path=Path(args.model),
        host=args.host,
        port=args.port,
        ctx=args.ctx,
        threads=args.threads,
    )

    try:
        import uvicorn
    except ImportError as exc:
        raise SystemExit(
            "缺少 uvicorn 依赖，请先安装：\n"
            "    pip install uvicorn"
        ) from exc

    app = create_app(config)
    uvicorn.run(app, host=config.host, port=config.port)


if __name__ == "__main__":
    main()
