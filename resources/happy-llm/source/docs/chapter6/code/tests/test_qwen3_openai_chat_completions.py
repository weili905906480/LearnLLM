from __future__ import annotations

import importlib.util
import sys
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace

from fastapi.testclient import TestClient


MODULE_PATH = Path(__file__).resolve().parents[1] / "qwen3_8b_web_service.py"


def load_module():
    spec = importlib.util.spec_from_file_location("qwen3_8b_web_service", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class FakeLlama:
    def __init__(self):
        self.calls = []

    def __call__(self, prompt, **kwargs):
        self.calls.append({"prompt": prompt, **kwargs})
        return {"choices": [{"text": "你好，我是本地模型。"}]}


class TestOpenAIChatCompletions(unittest.TestCase):
    def test_chat_completions_returns_openai_compatible_response(self):
        module = load_module()
        config = SimpleNamespace(model_path=Path("local-qwen3.gguf"))
        app = module.create_app(config)
        fake_llm = FakeLlama()
        app.state.model_loaded = True
        app.state.llm = fake_llm
        app.state.generation_lock = threading.Lock()

        client = TestClient(app)
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "qwen3-8b",
                "messages": [
                    {"role": "system", "content": "你是一个中文助手。"},
                    {"role": "user", "content": "介绍一下 Pandas。"},
                ],
                "max_tokens": 128,
                "temperature": 0.2,
                "top_p": 0.8,
            },
        )

        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["object"], "chat.completion")
        self.assertEqual(data["model"], "local-qwen3.gguf")
        self.assertEqual(data["choices"][0]["index"], 0)
        self.assertEqual(data["choices"][0]["finish_reason"], "stop")
        self.assertEqual(data["choices"][0]["message"]["role"], "assistant")
        self.assertEqual(data["choices"][0]["message"]["content"], "你好，我是本地模型。")
        self.assertIn("created", data)
        self.assertIn("usage", data)

        self.assertEqual(fake_llm.calls[0]["max_tokens"], 128)
        self.assertEqual(fake_llm.calls[0]["temperature"], 0.2)
        self.assertEqual(fake_llm.calls[0]["top_p"], 0.8)
        self.assertIn("<|im_start|>system\n你是一个中文助手。", fake_llm.calls[0]["prompt"])
        self.assertIn("<|im_start|>user\n介绍一下 Pandas。 /no_think", fake_llm.calls[0]["prompt"])

    def test_chat_completions_accepts_missing_max_tokens(self):
        module = load_module()
        config = SimpleNamespace(model_path=Path("local-qwen3.gguf"))
        app = module.create_app(config)
        fake_llm = FakeLlama()
        app.state.model_loaded = True
        app.state.llm = fake_llm
        app.state.generation_lock = threading.Lock()

        client = TestClient(app)
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "qwen3-8b",
                "messages": [{"role": "user", "content": "ping"}],
            },
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(fake_llm.calls[0]["max_tokens"], module.DEFAULT_MAX_TOKENS)

    def test_chat_completions_accepts_null_max_tokens(self):
        module = load_module()
        config = SimpleNamespace(model_path=Path("local-qwen3.gguf"))
        app = module.create_app(config)
        fake_llm = FakeLlama()
        app.state.model_loaded = True
        app.state.llm = fake_llm
        app.state.generation_lock = threading.Lock()

        client = TestClient(app)
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "qwen3-8b",
                "messages": [{"role": "user", "content": "ping"}],
                "max_tokens": None,
            },
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(fake_llm.calls[0]["max_tokens"], module.DEFAULT_MAX_TOKENS)


if __name__ == "__main__":
    unittest.main()
