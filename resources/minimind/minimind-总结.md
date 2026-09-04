# MiniMind 仓库总结

> 仓库地址：<https://github.com/jingyaogong/minimind>
> 本地路径：`resources/minimind`
> 克隆提交：`ed5ea9e`（README 最新更新）
> 开源协议：Apache License 2.0

---

## 一、项目是什么

MiniMind 是一个**从 0 开始训练超小语言模型**的开源复现项目，核心理念是"大道至简"：

- 目标：仅用约 **3 块钱 GPU 成本** 与 **2 小时训练时间**，训练出一个 **64M** 参数的完整语言模型。
- 主线最小版本体积约为 GPT-3 的 **1/2700**，力求普通个人 GPU 也能快速复现。
- 同时开源了大模型的**极简结构**与**完整训练链路**，覆盖从预训练到强化学习的全过程。
- 所有核心算法代码均用 **PyTorch 原生实现**，不依赖 `transformers` / `trl` / `peft` 等第三方库的高层抽象接口。

它既是一个 LLM 全阶段开源复现项目，也是一套面向 LLM 入门与实践的教程。

---

## 二、模型系列

| 模型 | 参数量 | 发布时间 | 说明 |
|------|--------|----------|------|
| minimind-3 | 64M | 2026.04.01 | 主线 Dense 模型 |
| minimind-3-moe | 198M-A64M | 2026.04.01 | MoE 版本（4 experts / top-1） |
| minimind2-small | 26M | 2025.04.26 | 历史版本 |
| minimind2-moe | 145M | 2025.04.26 | 历史版本 |
| minimind2 | 104M | 2025.04.26 | 历史版本 |
| minimind-v1 系列 | 26M~108M | 2024 | 已下线，不再维护 |

> 主线结构已对齐 **Qwen3 / Qwen3-MoE** 生态，便于转换到 `transformers` / `llama.cpp` / `ollama` / `vllm` 等第三方生态。

---

## 三、模型架构

`minimind-3` Dense 采用 Transformer **Decoder-Only** 结构，关键配置：

- Pre-Norm + **RMSNorm**
- **SwiGLU** 激活函数
- **RoPE** 旋转位置编码（支持 YaRN 长文本外推）
- `q_heads=8`、`kv_heads=4`（GQA 分组查询注意力）
- `max_position_embeddings=32768`、`rope_theta=1e6`

`minimind-3-moe` 在相同结构上扩展 MoE 前馈层，兼容 Qwen3-MoE 风格（去除 shared expert），默认 `4 experts / top-1 routing`。

### 模型参数对照

| Model | params | len_vocab | max_pos | rope_theta | n_layers | d_model | kv_heads | q_heads |
|-------|--------|-----------|---------|------------|----------|---------|----------|---------|
| minimind-3 | 64M | 6400 | 32768 | 1e6 | 8 | 768 | 4 | 8 |
| minimind-3-moe | 198M-A64M | 6400 | 32768 | 1e6 | 8 | 768 | 4 | 8 |
| minimind2-small | 26M | 6400 | 32768 | 1e6 | 8 | 512 | 2 | 8 |
| minimind2 | 104M | 6400 | 32768 | 1e6 | 16 | 768 | 2 | 8 |

> 模型配置讨论（参考 MobileLLM 研究）：在小模型区间，"深而窄"（更多层、更窄维度）通常优于"矮而胖"，但 `d_model < 512` 时词嵌入过窄劣势会放大。

---

## 四、仓库目录结构

```
minimind/
├── dataset/               # 数据集加载（lm_dataset.py 等）
│   └── lm_dataset.py
├── model/                 # 模型结构定义
│   ├── model_minimind.py  # Dense 模型（主线）
│   ├── model_lora.py      # LoRA 实现（纯手写，不依赖 peft）
│   ├── tokenizer.json     # 自定义 tokenizer（词表 6400）
│   └── tokenizer_config.json
├── trainer/               # 全部训练脚本
│   ├── train_pretrain.py     # 预训练
│   ├── train_full_sft.py     # 全参指令微调
│   ├── train_lora.py         # LoRA 微调
│   ├── train_distillation.py # 白盒蒸馏
│   ├── train_dpo.py          # DPO（RLHF）
│   ├── train_ppo.py          # PPO（RLAIF）
│   ├── train_grpo.py         # GRPO / CISPO（RLAIF）
│   ├── train_agent.py        # Agentic RL（多轮 Tool-Use）
│   ├── rollout_engine.py     # 训推分离的 rollout 引擎
│   ├── train_tokenizer.py    # tokenizer 训练示例
│   └── trainer_utils.py      # 训练工具
├── scripts/               # 推理 / 部署 / 转换脚本
│   ├── eval_llm.py           # CLI 推理
│   ├── eval_toolcall.py      # Tool Calling 测试
│   ├── serve_openai_api.py   # OpenAI 兼容 API 服务
│   ├── chat_api.py           # API 客户端示例
│   ├── web_demo.py           # Streamlit WebUI
│   └── convert_model.py      # torch ↔ transformers 格式转换
├── eval_llm.py            # 推理入口（根目录）
├── requirements.txt
└── README.md
```

---

## 五、完整训练链路

训练顺序（`cd trainer` 目录执行）：

### 1. 预训练 Pretrain（必须）
- 目标：**学会词语接龙**，从海量文本学习事实知识与语言规律。
- 脚本：`train_pretrain.py`，输出 `pretrain_{dim}.pth`

### 2. 指令微调 SFT（必须）
- 目标：适应多轮对话模板、指令跟随、工具调用、思考标签。
- 脚本：`train_full_sft.py`，输出 `full_sft_{dim}.pth`
- 当前主线 SFT 数据已混入 Tool Calling 样本。

### 3. 知识蒸馏 KD（可选）
- **黑盒蒸馏**：对教师输出做监督微调（`CE` 损失）。
- **白盒蒸馏**：额外拟合教师 token 分布（`CE + KL` 混合损失 + 温度缩放）。
- 脚本：`train_distillation.py`

### 4. LoRA（可选）
- 纯手写实现低秩微调，不依赖 `peft`。
- 适合垂直领域 / 自我认知等私有数据适配。
- 脚本：`train_lora.py`，可经 `convert_model.py` 合并回基模。

### 5. 工具调用 & 自适应思考
- Tool Calling 能力已并入主线 SFT，默认 `full_sft` 即具备基础能力。
- 支持 `<tool_call>`、`<tool_response>`、`<think>` 等模板标记。
- 自适应思考通过 `open_thinking` 开关 + `chat_template` 控制，不再单独训练思考模型。

### 6. 强化学习（可选）

**统一视角**：所有 PO（Policy Optimization）算法本质是优化同一目标：

$$\mathcal{J}_{PO} = \mathbb{E}\left[ f(r_t) \cdot g(A_t) - h(\text{KL}_t) \right]$$

其中三个核心组件：策略项 $f(r_t)$、优势项 $g(A_t)$、正则项 $h(\text{KL}_t)$。

| 算法 | 类型 | 策略项 | 优势项 | 训练模型数 |
|------|------|--------|--------|-----------|
| DPO | RLHF (off-policy) | $\log r_w - \log r_l$ | 无显式优势 | 1（前向 2） |
| PPO | RLAIF (on-policy) | $\min(r, \text{clip}(r))$ | $R - V(s)$（需 Critic） | 2 |
| GRPO | RLAIF | $\min(r, \text{clip}(r))$ | $\frac{R-\mu}{\sigma}$（组内归一化） | 1 |
| CISPO | RLAIF | $\text{clip}(r) \cdot A \cdot \log\pi$ | $\frac{R-\mu}{\sigma}$ | 1 |

- **DPO**：`train_dpo.py`，无 Reward Model，显存低、收敛稳定，适合人类偏好对齐。
- **PPO**：`train_ppo.py`，Actor + Critic + GAE，显存约 1.5~2 倍，收敛较慢。
- **GRPO**：`train_grpo.py`，分组相对价值估计，无需 Critic，收敛更稳定、上限更高。
- **CISPO**：GRPO 的 loss 变体，解决 ratio 被 clip 后梯度截断问题（`loss_type=cispo`）。
- **Agentic RL**：`train_agent.py`，多轮 Tool-Use 场景的 GRPO/CISPO，延迟奖励，整条轨迹打分，训推分离（支持 sglang rollout）。

---

## 六、数据集

下载地址：[ModelScope](https://www.modelscope.cn/datasets/gongjy/minimind_dataset/files) | [HuggingFace](https://huggingface.co/datasets/jingyaogong/minimind_dataset)

| 文件 | 大小 | 用途 |
|------|------|------|
| pretrain_t2t_mini.jsonl ✨ | 1.2GB | 快速复现预训练 |
| pretrain_t2t.jsonl | 10GB | 主线预训练 |
| sft_t2t_mini.jsonl ✨ | 1.6GB | 快速复现 SFT（已混 Tool Call） |
| sft_t2t.jsonl | 14GB | 主线 SFT |
| rlaif.jsonl ✨ | 24MB | PPO/GRPO/CISPO 训练 |
| agent_rl.jsonl | 86MB | Agentic RL 主线 |
| agent_rl_math.jsonl | 18MB | Agentic RL 数学（RLVR） |
| dpo.jsonl | 53MB | DPO 偏好数据 |

> 快速复现 Zero 模型只需 `pretrain_t2t_mini.jsonl` + `sft_t2t_mini.jsonl`。

---

## 七、推理与部署

- **CLI 推理**：`python eval_llm.py --weight full_sft`（或 `--load_from` 指定 transformers 目录）
- **WebUI**：`cd scripts && streamlit run web_demo.py`
- **OpenAI 兼容 API**：`cd scripts && python serve_openai_api.py`（支持 `reasoning_content`、`tool_calls`、`open_thinking`）
- **第三方框架**：兼容 `llama.cpp`、`ollama`、`vllm`、`SGLang`、`MNN`、`Llama-Factory`。

---

## 八、关键亮点

1. **极低成本**：单卡 3090，约 2 小时 / 3 元即可从 0 训练出可对话的 64M 模型。
2. **全链路从 0 实现**：MoE、数据清洗、预训练、SFT、LoRA、DPO、PPO/GRPO/CISPO、Agentic RL、蒸馏等全过程，无第三方高层抽象。
3. **主线对齐 Qwen3 生态**：便于迁移到主流推理与训练框架。
4. **训推分离**：Agentic RL 引入 rollout engine，训练侧与推理侧解耦。
5. **自适应思考**：`<think>` 标签 + `open_thinking` 开关，模型动态决定是否显式思考。
6. **YaRN 长文本外推**：RoPE 免训练扩展上下文到 2048+。
7. **丰富的可视化**：SwanLab（国内友好，接口兼容 WandB）。

---

## 九、依赖环境（requirements.txt 要点）

- `torch`（建议 2.6.0，需自行安装 CUDA 版本）
- `transformers==4.57.6`、`trl==0.13.0`
- `swanlab==0.7.11`（训练可视化，替代 WandB）
- `modelscope==1.37.0`、`streamlit==1.50.0`
- `datasets`、`openai`、`flask`、`tiktoken` 等

作者参考配置：RTX 3090 (24GB) × 8、Ubuntu 20.04、CUDA 12.2、Python 3.10.16。

---

## 十、对 LearnLLM 的学习价值

这是一个非常适合**系统学习大语言模型全流程**的项目：

1. **可读性强**：核心结构仅数百行，去掉框架抽象，直接看到 Transformer、RoPE、MoE、KV-cache 等底层实现。
2. **全阶段覆盖**：从 tokenizer、预训练、SFT，到 LoRA、蒸馏、DPO、PPO/GRPO 强化学习，形成完整闭环。
3. **RL 统一视角**：README 用"策略项 / 优势项 / 正则项"三个组件统一解释 DPO/PPO/GRPO/CISPO，是理解 RLHF/RLAIF 的优秀入门材料。
4. **可复现**：低成本即可亲手跑通"从 0 训练一个语言模型"的完整过程。
5. **生态兼容**：训练结果可无缝接入 llama.cpp / ollama / vllm 等推理生态，贴近工程实践。

> 建议配合该仓库的 `model/model_minimind.py` 与 `trainer/` 下脚本逐行研读，是本项目最有价值的部分。
