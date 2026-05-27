# nanochat

> 原文来源：https://github.com/karpathy/nanochat/blob/master/README.md

nanochat 是用于训练大型语言模型（LLM）最简洁的实验框架。它设计为在单 GPU 节点上运行，代码极简易改造，涵盖 LLM 训练的所有主要阶段：分词、预训练、微调、评估、推理以及聊天 UI。例如，你可以用约 $48（约 2 小时 8×H100 GPU 节点）训练自己的 GPT-2 量级 LLM（该模型在 2019 年训练成本约 $43,000），然后通过熟悉的 ChatGPT 风格 Web UI 与之对话。如果使用 Spot 实例，总成本可低至 ~$15。

更通用地说，nanochat 开箱即用地支持通过设置唯一一个复杂度旋钮 `--depth`（GPT Transformer 的层数，GPT-2 量级约对应 depth=26）来训练一整个计算最优模型小系列。所有其他超参数（Transformer 宽度、注意力头数、学习率调整、训练步长、权重衰减等）都会以最优方式自动计算。

关于仓库问题，推荐使用 [DeepWiki](https://deepwiki.com/karpathy/nanochat)（Devin/Cognition 出品），或前往 [Discussions 标签页](https://github.com/karpathy/nanochat/discussions)，也可来 Discord 的 [#nanochat](https://discord.com/channels/1020383067459821711/1427295580895314031) 频道。

---

## 通往 GPT-2 的时间排行榜（Time-to-GPT-2 Leaderboard）

目前的开发重点是调优预训练阶段（耗计算最多）。受 modded-nanogpt 启发，为了激励进步与社区协作，nanochat 维护了一个 "GPT-2 极速挑战" 排行榜——即训练 nanochat 模型达到 GPT-2 量级能力（以 DCLM CORE 分数衡量）所需的挂钟时间。[runs/speedrun.sh](https://github.com/karpathy/nanochat/blob/master/runs/speedrun.sh) 脚本始终是训练 GPT-2 量级模型并与之对话的参考方式。当前排行榜如下：

| # | 时间 | val_bpb | CORE | 描述 | 日期 | Commit | 贡献者 |
|---|------|---------|------|------|------|--------|--------|
| 0 | 168 小时 | - | 0.2565 | 原始 OpenAI GPT-2 检查点 | 2019 | - | OpenAI |
| 1 | 3.04 | 0.74833 | 0.2585 | d24 基线，略微过训练 | 2026-01-29 | 348fbb3 | @karpathy |
| 2 | 2.91 | 0.74504 | 0.2578 | d26 略微欠训练 **+fp8** | 2026-02-02 | a67eba3 | @karpathy |
| 3 | 2.76 | 0.74645 | 0.2602 | 总批大小提升至 1M tokens | 2026-02-05 | 2c062aa | @karpathy |
| 4 | 2.02 | 0.71854 | 0.2571 | 数据集换为 NVIDIA ClimbMix | 2026-03-04 | 324e69c | @ddudek @karpathy |
| 5 | 1.80 | 0.71808 | 0.2690 | autoresearch 第一轮 | 2026-03-09 | 6ed7d1d | @karpathy |
| 6 | 1.65 | 0.71800 | 0.2626 | autoresearch 第二轮 | 2026-03-14 | a825e63 | @karpathy |

核心指标是"通往 GPT-2 的时间"——在 8×H100 GPU 节点上超越 GPT-2（1.6B）CORE 指标所需的挂钟时间，GPT-2 CORE 分数为 0.256525。2019 年训练 GPT-2 的成本约 $43,000，而如今经过 7 年间全栈的众多进步，我们可以用远低于 $100 的成本更快完成（当前 ~$3/GPU/hr，8×H100 节点约 $24/hr，2 小时约 $48）。

更多排行榜解读与贡献方式，见 [dev/LEADERBOARD.md](https://github.com/karpathy/nanochat/blob/master/dev/LEADERBOARD.md)。

---

## 快速开始

### 环境搭建

nanochat 使用 [uv](https://docs.astral.sh/uv/) 进行依赖管理：

```bash
uv sync --extra gpu    # CUDA 环境（A100/H100 等）
uv sync --extra cpu    # CPU-only 或 MPS 环境
source .venv/bin/activate
```

开发模式（含 pytest、matplotlib、ipykernel、transformers 等）：

```bash
uv sync --extra gpu --group dev
```

### 复现并与 GPT-2 对话

最有趣的事莫过于训练自己的 GPT-2 并与之对话。完整流程包含在 [runs/speedrun.sh](https://github.com/karpathy/nanochat/blob/master/runs/speedrun.sh) 中，适合在 8×H100 GPU 节点上运行。启动一台新的 8×H100 GPU 机器（例如作者推荐的 [Lambda](https://lambda.ai/service/gpu-cloud)），然后启动训练脚本：

```bash
bash runs/speedrun.sh
```

建议在 screen 会话中运行，整个过程约需 3 小时。完成后可通过 ChatGPT 风格的 Web UI 与之对话：

```bash
python -m scripts.chat_web
```

访问显示的 URL（Lambda 上请用节点公网 IP 加端口，如 `http://209.20.xxx.xxx:8000/`），然后像使用 ChatGPT 一样与你的 LLM 对话！由于 speedrun 模型的计算量约 4e19 FLOPs，能力有限，和它聊天有点像和幼儿园小朋友交流 :)

**一些补充说明：**

- 代码在 Ampere 8×A100 节点上同样可以运行，只是稍慢一些。
- 省去 `torchrun`，代码同样可在单 GPU 上运行（会自动切换为梯度累积），结果几乎一致，但耗时约为 8 倍。
- 若 GPU 显存小于 80GB，需调小 `--device-batch-size`（默认 32，可改为 16、8、4、2 乃至 1），否则可能 OOM。
- 代码基于标准 PyTorch，理论上支持 xpu、mps 等后端，但作者未全部测试，可能存在问题。

---

## 科研使用

如果你是研究者，希望改进 nanochat，可以关注 [runs/scaling_laws.sh](https://github.com/karpathy/nanochat/blob/master/runs/scaling_laws.sh) 和 [runs/miniseries.sh](https://github.com/karpathy/nanochat/blob/master/runs/miniseries.sh)。更多文档见 [Jan 7 miniseries v1](https://github.com/karpathy/nanochat/discussions/420)。

快速实验（~5 分钟预训练）推荐训练 12 层模型（GPT-1 大小）：

```bash
OMP_NUM_THREADS=1 torchrun --standalone --nproc_per_node=8 -m scripts.base_train -- \
    --depth=12 \
    --run="d12" \
    --model-tag="d12" \
    --core-metric-every=999999 \
    --sample-every=-1 \
    --save-every=-1 \
```

评估时推荐在 wandb 中监控：

1. `val_bpb`（验证损失，以词表大小无关的 bits per byte 为单位）随 `step`、`total_training_time`、`total_training_flops` 的变化
2. `core_metric`（DCLM CORE 分数）
3. VRAM 利用率、`train/mfu`（模型 FLOPS 利用率）、`train/tok_per_sec`（训练吞吐量）

nanochat 围绕单一复杂度旋钮设计——Transformer 的深度（`--depth`）。这一整数会自动决定所有其他超参数，使训练出的模型达到计算最优。用户无需手动设置这些，只需通过 `--depth` 指定模型大小，一切"开箱即用"。GPT-2 量级模型大约在 d24–d26 范围内。任何候选改动必须在所有 `--depth` 设置下都有原则性效果。

---

## 在 CPU / MPS 上运行

[runs/runcpu.sh](https://github.com/karpathy/nanochat/blob/master/runs/runcpu.sh) 展示了如何在 CPU 或 Apple Silicon 上运行的简单示例，会大幅缩小训练模型以适应合理的时间窗口（几十分钟），效果不佳但可用于学习。

---

## 精度 / 数据类型

nanochat 不使用 `torch.amp.autocast`，而是通过单一全局变量 `COMPUTE_DTYPE`（定义在 `nanochat/common.py`）显式管理精度，默认值根据硬件自动检测：

| 硬件 | 默认 dtype | 原因 |
|------|-----------|------|
| CUDA SM 80+（A100、H100 等） | `bfloat16` | 原生 bf16 张量核心 |
| CUDA SM < 80（V100、T4 等） | `float32` | 无 bf16；可通过 `NANOCHAT_DTYPE=float16` 启用 fp16（需 GradScaler） |
| CPU / MPS | `float32` | 无低精度张量核心 |

可通过 `NANOCHAT_DTYPE` 环境变量覆盖默认值：

```bash
NANOCHAT_DTYPE=float32 python -m scripts.chat_cli -p "hello"   # 强制 fp32
NANOCHAT_DTYPE=bfloat16 torchrun --nproc_per_node=8 -m scripts.base_train  # 强制 bf16
```

**工作原理：** 模型权重以 fp32 存储（用于优化器精度），自定义 `Linear` 层在前向传播时将其转为 `COMPUTE_DTYPE`。Embedding 直接以 `COMPUTE_DTYPE` 存储以节省内存。这与 autocast 的混合精度效果相同，但对精度有完全的显式控制。

注意：`float16` 训练会在 `base_train.py` 中自动启用 `GradScaler` 以防止梯度下溢。SFT 也支持，但 RL 目前不支持。fp16 推理在任何地方均可正常工作。

---

## 相关指南

作者发布的相关指南（按时间倒序）：

- [2026-02-01：以远低于 $100 的成本超越 GPT-2：nanochat 的历程](https://github.com/karpathy/nanochat/discussions/481)
- [Jan 7 miniseries v1](https://github.com/karpathy/nanochat/discussions/420)：第一批 nanochat 计算最优模型系列
- [指南：如何为 nanochat 添加新能力（以在 strawberry 中数 r 为例）](https://github.com/karpathy/nanochat/discussions/164)
- [指南：为你的 nanochat 注入身份](https://github.com/karpathy/nanochat/discussions/139)：通过合成数据生成和 SFT 阶段数据混合，定制 nanochat 的个性
- [2025-10-13：nanochat 原始介绍帖](https://github.com/karpathy/nanochat/discussions/1)（含部分已过时的信息）

---

## 文件结构

```
.
├── LICENSE
├── README.md
├── dev
│   ├── gen_synthetic_data.py       # 身份注入的合成数据示例
│   ├── generate_logo.html
│   ├── nanochat.png
│   └── repackage_data_reference.py # 预训练数据分片生成
├── nanochat
│   ├── __init__.py                 # 空文件
│   ├── checkpoint_manager.py       # 模型检查点保存/加载
│   ├── common.py                   # 杂项小工具
│   ├── core_eval.py                # 基础模型 CORE 分数评估（DCLM 论文）
│   ├── dataloader.py               # 分词分布式数据加载器
│   ├── dataset.py                  # 预训练数据下载/读取工具
│   ├── engine.py                   # 带 KV Cache 的高效模型推理
│   ├── execution.py                # 允许 LLM 以工具方式执行 Python 代码
│   ├── gpt.py                      # GPT nn.Module Transformer
│   ├── logo.svg
│   ├── loss_eval.py                # 评估 bits per byte（代替 loss）
│   ├── optim.py                    # AdamW + Muon 优化器，支持单 GPU 和分布式
│   ├── report.py                   # 编写 nanochat Report 的工具
│   ├── tokenizer.py                # GPT-4 风格的 BPE 分词器封装
│   └── ui.html                     # nanochat 前端 HTML/CSS/JS
├── pyproject.toml
├── runs
│   ├── miniseries.sh               # 小系列训练脚本
│   ├── runcpu.sh                   # CPU/MPS 上运行的简单示例
│   ├── scaling_laws.sh             # 缩放定律实验
│   └── speedrun.sh                 # 训练 ~$100 的 nanochat d20
├── scripts
│   ├── base_eval.py                # 基础模型：CORE 分数、bits per byte、采样
│   ├── base_train.py               # 基础模型：训练
│   ├── chat_cli.py                 # 聊天模型：CLI 交互
│   ├── chat_eval.py                # 聊天模型：评估任务
│   ├── chat_rl.py                  # 聊天模型：强化学习
│   ├── chat_sft.py                 # 聊天模型：SFT 训练
│   ├── chat_web.py                 # 聊天模型：Web UI 交互
│   ├── tok_eval.py                 # 分词器：评估压缩率
│   └── tok_train.py                # 分词器：训练
├── tasks
│   ├── arc.py                      # 多选科学题
│   ├── common.py                   # TaskMixture | TaskSequence
│   ├── customjson.py               # 从任意 jsonl 对话创建 Task
│   ├── gsm8k.py                    # 8K 小学数学题
│   ├── humaneval.py                # 简单 Python 编程任务
│   ├── mmlu.py                     # 宽泛主题多选题
│   ├── smoltalk.py                 # HuggingFace SmolTalk 聚合数据集
│   └── spellingbee.py              # 教模型拼写/数字母的任务
├── tests
│   └── test_engine.py
└── uv.lock
```

---

## 完整训练 Pipeline 源码对应

### Stage 1：Tokenizer 训练

**入口文件：** `scripts/tok_train.py`

- 从预训练数据中读取约 **2B 字符**，训练 BPE 分词器
- 词表大小 **32,768**，风格类似 GPT-4 分词器
- 依赖：`nanochat/tokenizer.py`（`RustBPETokenizer`）、`nanochat/dataset.py`（数据读取）

```bash
python -m scripts.tok_train
```

---

### Stage 2：预训练（Pretraining）

**入口文件：** `scripts/base_train.py`

- 最核心的训练文件，约 **400 行**
- 唯一旋钮 `--depth` 自动推算所有超参：模型宽度、batch size、学习率、weight decay
- 使用 **Muon + AdamW** 组合优化器，支持 FP8 训练（H100+）
- 依赖：`nanochat/gpt.py`（模型架构 MQA + RoPE）、`nanochat/optim.py`（Muon 优化器）、`nanochat/dataloader.py`

```bash
torchrun --nproc_per_node=8 -m scripts.base_train --depth=26
```

---

### Stage 3：中期训练（Midtraining）

**入口文件：** `scripts/base_train.py`（**同一个文件**，换数据集）

- 没有独立的 midtraining 脚本，通过 `--resume-from-step` 参数从预训练 checkpoint 继续训练
- 区别在于喂入更高质量的数据子集（在 `nanochat/dataset.py` 中切换）

---

### Stage 4：SFT 微调

**入口文件：** `scripts/chat_sft.py`

- 加载预训练 base model，在对话数据上微调
- **Loss masking**：只对 `<|assistant|>` 部分计算 loss，用户输入部分 mask 掉
- 数据混合：SmolTalk + MMLU + GSM8K + SpellingBee + 合成身份数据
- 依赖：`tasks/` 目录下各 task 文件（`smoltalk.py`、`gsm8k.py`、`mmlu.py` 等）

```bash
torchrun --nproc_per_node=8 -m scripts.chat_sft --device-batch-size=16
```

---

### Stage 5：强化学习（RL，可选）

**入口文件：** `scripts/chat_rl.py`

- 基于 **GRPO/REINFORCE** 变体（非标准 PPO，已简化）
- 在 **GSM8K 数学题**上做 RL，用答案是否正确作为 reward
- 加载 SFT checkpoint，对输出采样多次，用 reward 计算 advantage，做策略梯度更新

```bash
torchrun --nproc_per_node=8 -m scripts.chat_rl --run=default
```

---

### Stage 6：推理 / Web 服务

**入口文件：** `scripts/chat_web.py`（Web UI）或 `scripts/chat_cli.py`（命令行）

- FastAPI 服务，多 GPU worker pool 并行处理请求
- 流式输出（SSE），支持 ChatGPT 风格对话界面
- UI 在 `nanochat/ui.html`，推理引擎在 `nanochat/engine.py`（带 KV Cache）

```bash
python -m scripts.chat_web --num-gpus 4
```

---

### 核心支撑模块

| 文件 | 作用 |
|------|------|
| `nanochat/gpt.py` | GPT 模型定义（MQA + RoPE + LayerNorm） |
| `nanochat/optim.py` | Muon + AdamW 优化器实现 |
| `nanochat/tokenizer.py` | BPE 分词器封装 |
| `nanochat/engine.py` | KV Cache 推理引擎 |
| `nanochat/dataloader.py` | 分布式数据加载器 |
| `nanochat/checkpoint_manager.py` | checkpoint 保存/加载 |

---

## 贡献指南

nanochat 的目标是提升预算低于 $1000 的微型模型的 SOTA 水平，并让端到端工作流程触手可及。"可及"不仅指成本，还指认知复杂度——nanochat 不是一个可无限配置的 LLM "框架"，代码库中没有庞大的配置对象、模型工厂或 if-else 怪兽，而是一个单一、内聚、极简、可读、易改造、最大程度可 fork 的"强基线"代码库，从头跑到尾即可得到一个可以对话的 ChatGPT 模型。

目前最有意思的方向是加速到达 GPT-2 的延迟（即 CORE 分数超过 0.256525）。目前约需 3 小时，可通过改进预训练阶段进一步压缩。

**AI 使用声明政策：披露制。** 提交 PR 时，请说明哪些部分有 LLM 的大量贡献，或你未亲自编写或尚未完全理解的内容。

---

## 致谢

- 项目名称（nanochat）源自作者早期项目 [nanoGPT](https://github.com/karpathy/nanoGPT)，后者仅覆盖预训练。
- nanochat 同样受到 [modded-nanoGPT](https://github.com/KellerJordan/modded-nanogpt) 启发，该项目通过明确指标和排行榜将 nanoGPT 游戏化，nanochat 借鉴了其许多想法和预训练实现。
- 感谢 [HuggingFace](https://huggingface.co/) 提供 fineweb 和 smoltalk。
- 感谢 [Lambda](https://lambda.ai/service/gpu-cloud) 提供开发所用算力。
- 感谢首席 LLM 引导师 🧙‍♂️ Alec Radford 的建议与指导。
- 感谢仓库管理员 Sofie [@svlandeg](https://github.com/svlandeg) 协助管理 nanochat 的 issues、PR 和 Discussions。

---

## 引用

如果你在研究中发现 nanochat 有帮助，请按如下方式引用：

```bibtex
@misc{nanochat,
  author = {Andrej Karpathy},
  title = {nanochat: The best ChatGPT that \$100 can buy},
  year = {2025},
  publisher = {GitHub},
  url = {https://github.com/karpathy/nanochat}
}
```

---

## 许可证

MIT
