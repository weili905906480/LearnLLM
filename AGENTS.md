# Repository Guidelines

## 项目结构与模块组织

本仓库用于系统学习 Karpathy 风格的 LLM 项目。`01-nanoGPT/` 包含 nanoGPT 源码副本与学习材料：代码在 `01-nanoGPT/code/`，笔记在 `01-nanoGPT/notes/`，其中包括 `nanogpt-source/`、`pytorch-course/` 和 `theory/`。`02-nanochat/` 是更完整的 Python 项目：核心包在 `02-nanochat/code/nanochat/`，可执行脚本在 `02-nanochat/code/scripts/`，实验脚本在 `02-nanochat/code/runs/`，测试在 `02-nanochat/code/tests/`，讲解笔记在 `02-nanochat/notes/`。通用资料放在 `resources/`。

## 构建、测试与开发命令

在 `02-nanochat/code` 目录下运行 nanochat 相关命令：

```bash
uv sync --extra cpu --group dev
uv sync --extra gpu --group dev
uv run pytest
uv run pytest -m "not slow"
uv run python -m scripts.chat_cli -p "hello"
uv run python -m scripts.chat_web
```

本地阅读和轻量调试优先使用 CPU 依赖；CUDA 训练或评测使用 GPU 依赖。`uv run pytest` 运行 `tests/` 中的测试；带 marker 的命令会跳过慢测试。训练示例参考 `runs/runcpu.sh`、`runs/speedrun.sh` 和 `runs/scaling_laws.sh`。

## 编码风格与命名约定

Python 代码使用 4 空格缩进，保持脚本简洁直接。可复用逻辑放在 `nanochat/` 包内，训练、评测、推理等流程入口放在 `scripts/`，文件名使用描述性的 snake_case，例如 `base_train.py`、`chat_eval.py`。测试文件遵循 `test_*.py`，测试函数使用 `test_*`。连续性学习笔记建议使用带编号的描述性文件名。

## 测试指南

修改 `nanochat/`、`scripts/` 或 `tasks/` 的行为时，应在 `02-nanochat/code/tests/` 添加或更新 pytest 测试。优先编写小而确定的测试，避免默认依赖数据下载或 GPU。耗时测试使用 `@pytest.mark.slow` 标记，便于通过 `uv run pytest -m "not slow"` 跳过。

## 提交与 Pull Request 规范

近期提交多使用简短祈使句，也会使用常见前缀，例如 `docs: add ...`、`chore: copy ...`、`add nanochat ... walkthrough`。默认使用中文 Git 提交信息，除非用户明确要求英文或外部项目规范另有要求。每次提交聚焦一个主题。PR 应说明修改范围、列出已运行命令，并注明训练或 GPU 相关的硬件假设。只有修改 `nanochat/ui.html` 或 Web 服务等可见界面时才需要截图。

## 语言与文档要求

默认使用中文回复协作问题，除非用户明确要求英文。新增或修改学习笔记、讲解文档、README 内容时，默认写中文文档；引用英文术语时可保留原文并给出中文解释。代码标识符、命令、路径和第三方 API 名称保持原样。

## 安全与配置提示

不要提交生成的 checkpoint、下载的数据集、虚拟环境或本地密钥。机器相关配置应放在环境变量中，例如 `NANOCHAT_DTYPE`。涉及非默认训练硬件或精度设置时，在 PR 描述中写清楚。
