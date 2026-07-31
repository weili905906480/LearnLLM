# Qwen3-8B GGUF 本地 Web 服务实现计划

## 1. 目标

基于当前已经下载并验证可用的本地模型：

```text
resources/happy-llm/source/docs/chapter6/models/Qwen3-8B-GGUF/Qwen3-8B-Q4_K_M.gguf
```

使用 `llama-cpp-python` + `FastAPI` 搭建一个本机可访问的 Web 服务，让浏览器或 HTTP 客户端可以调用 Qwen3-8B GGUF Q4_K_M 模型进行本地推理。

第一版目标是“简单、稳定、可学习、可扩展”，不追求复杂功能。

---

## 2. 推荐方案

采用方案一：自写 FastAPI 本地服务。

整体架构：

```text
浏览器 / curl / Python 客户端
        ↓ HTTP
FastAPI 本地 Web 服务
        ↓
llama-cpp-python
        ↓
Qwen3-8B-Q4_K_M.gguf
```

计划新增文件：

```text
resources/happy-llm/source/docs/chapter6/code/qwen3_8b_web_service.py
```

---

## 3. 第一版功能范围

### 3.1 必做功能

1. 启动服务时只加载一次 GGUF 模型。
2. 提供健康检查接口：

```text
GET /health
```

3. 提供普通聊天接口：

```text
POST /chat
```

4. 提供简单浏览器页面：

```text
GET /
```

5. 默认使用 CPU 推理。
6. 默认关闭 Qwen3 thinking，减少 CPU 推理等待时间。
7. 使用线程锁控制并发，一次只允许一个生成请求，避免 CPU 被多个请求同时打满。
8. 代码中加入详细中文注释，便于学习。

### 3.2 第一版暂不做

1. 暂不做流式输出。
2. 暂不做用户登录。
3. 暂不做多轮历史持久化。
4. 暂不做向量数据库 RAG。
5. 暂不监听局域网地址，默认只监听本机 `127.0.0.1`。

---

## 4. 接口设计

### 4.1 `GET /health`

用途：确认服务是否启动、模型是否已加载。

示例返回：

```json
{
  "status": "ok",
  "model_loaded": true,
  "model_path": "resources/happy-llm/source/docs/chapter6/models/Qwen3-8B-GGUF/Qwen3-8B-Q4_K_M.gguf"
}
```

---

### 4.2 `POST /chat`

用途：提交一个 prompt，返回模型回答。

请求示例：

```json
{
  "prompt": "请用三句话解释什么是 RAG。",
  "max_tokens": 128,
  "temperature": 0.7,
  "top_p": 0.9,
  "think": false
}
```

字段说明：

| 字段 | 类型 | 是否必填 | 默认值 | 说明 |
| --- | --- | --- | --- | --- |
| `prompt` | string | 是 | 无 | 用户问题 |
| `max_tokens` | int | 否 | 256 | 最大生成 token 数 |
| `temperature` | float | 否 | 0.7 | 采样温度 |
| `top_p` | float | 否 | 0.9 | nucleus sampling 参数 |
| `think` | bool | 否 | false | 是否启用 Qwen3 thinking |

返回示例：

```json
{
  "answer": "RAG 是一种结合检索和生成的技术...",
  "prompt": "请用三句话解释什么是 RAG。",
  "model": "Qwen3-8B-Q4_K_M.gguf"
}
```

---

### 4.3 `GET /`

用途：提供一个最小网页聊天界面。

页面包含：

1. 一个文本输入框；
2. 一个发送按钮；
3. 一个回答展示区域；
4. 简单提示：CPU 推理可能较慢，请耐心等待。

---

## 5. 模型加载设计

模型必须在服务启动时加载一次，而不是每次请求重新加载。

推荐伪代码：

```python
llm = Llama(
    model_path=str(MODEL_PATH),
    n_ctx=2048,
    n_threads=12,
    n_gpu_layers=0,
    verbose=False,
)
```

默认参数：

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `n_ctx` | 2048 | 上下文长度，越大越占内存 |
| `n_threads` | 12 | CPU 推理线程数 |
| `n_gpu_layers` | 0 | 纯 CPU 推理 |
| `verbose` | false | 减少底层日志 |

后续可通过命令行参数扩展：

```bash
python qwen3_8b_web_service.py --ctx 4096 --threads 16 --port 8000
```

---

## 6. Prompt 模板设计

使用 Qwen Chat 格式：

```text
<|im_start|>system
你是一个本地运行的中文助手。请回答得清晰、准确、可操作。<|im_end|>
<|im_start|>user
用户问题 /no_think<|im_end|>
<|im_start|>assistant
```

默认关闭 thinking：

```text
/no_think
```

如果请求中 `think=true`，则不追加 `/no_think`。

---

## 7. 并发控制

因为当前是 CPU 推理，多个请求同时生成会导致：

1. CPU 占用过高；
2. 每个请求都变慢；
3. 内存压力增加；
4. 甚至可能导致服务卡死。

第一版使用全局锁：

```python
generation_lock = threading.Lock()

with generation_lock:
    output = llm(...)
```

这样一次只处理一个生成请求。

---

## 8. 启动方式

直接运行 Python 文件：

```bash
python resources/happy-llm/source/docs/chapter6/code/qwen3_8b_web_service.py
```

默认监听：

```text
http://127.0.0.1:8000
```

浏览器访问：

```text
http://127.0.0.1:8000
```

健康检查：

```bash
curl http://127.0.0.1:8000/health
```

聊天接口：

```bash
curl -X POST http://127.0.0.1:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"prompt":"请用三句话解释什么是 LoRA。","max_tokens":128}'
```

---

## 9. 依赖

当前已安装：

```text
llama-cpp-python==0.3.34
```

还需要安装：

```bash
pip install fastapi uvicorn
```

如需换国内源：

```bash
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple fastapi uvicorn
```

---

## 10. 验证步骤

1. 检查依赖是否安装：

```bash
python -c "import fastapi, uvicorn, llama_cpp; print('ok')"
```

2. 启动服务：

```bash
python resources/happy-llm/source/docs/chapter6/code/qwen3_8b_web_service.py
```

3. 打开浏览器：

```text
http://127.0.0.1:8000
```

4. 测试 `/health`：

```bash
curl http://127.0.0.1:8000/health
```

5. 测试 `/chat`：

```bash
curl -X POST http://127.0.0.1:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"prompt":"用一句话说明 RAG 是什么","max_tokens":64}'
```

---

## 11. 后续扩展方向

第一版跑通后，可以逐步扩展：

1. 增加流式输出接口：

```text
POST /chat/stream
```

2. 增加多轮对话历史。
3. 增加 `mode` 参数：

```text
assistant / study / code / rag
```

4. 复用 `qwen3_8b_local_agent.py` 中的文件上下文能力。
5. 增加目录检索 RAG 能力。
6. 增加 OpenAI-compatible 接口：

```text
POST /v1/chat/completions
```

7. 接入 Open WebUI 或其他前端。

---

## 12. 风险与注意事项

1. CPU 推理速度较慢，网页请求可能需要等待较久。
2. 不建议第一版开放到局域网或公网。
3. 不建议同时处理多个生成请求。
4. `max_tokens` 不宜设置过大，首次建议 64 到 256。
5. `n_ctx` 不宜一开始设置过大，首次建议 2048。
6. 该服务只适合本地学习与实验，不建议直接作为生产服务。

---

## 13. 第一版完成标准

满足以下条件即认为第一版完成：

1. `qwen3_8b_web_service.py` 能正常启动；
2. 模型只在服务启动时加载一次；
3. 浏览器访问 `/` 能看到简单聊天页面；
4. `/health` 返回正常状态；
5. `/chat` 能返回 Qwen3-8B 的回答；
6. CPU 推理过程中服务不崩溃；
7. 代码包含清晰中文注释。
