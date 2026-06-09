# Datawhale Diy-LLM 课程总结

> 来源仓库：[datawhalechina/diy-llm](https://github.com/datawhalechina/diy-llm/tree/main)  
> 在线阅读：[datawhalechina.github.io/diy-llm](https://datawhalechina.github.io/diy-llm/)  
> 整理日期：2026-06-09  
> 总结范围：`README.md`、`docs/zh/` 课程目录、`coursework/` 6 个作业 README。

## 一句话概括

Diy-LLM 是 Datawhale 围绕 Stanford CS336 构建的中文化大模型系统课程。它不是单纯翻译原课，而是把 LLM 从分词器、Transformer 结构、训练系统、GPU 优化、分布式训练、数据工程、评估、推理到 SFT/RLHF/RLVR 对齐的完整流程拆成中文教材和可动手作业。

## CS336 官方地址

CS336 更像 Stanford 的公开课程主页，不是一个专门的纯在线视频课站点。官方公开页面主要在这里：

- 2025 春季主页：https://stanford-cs336.github.io/spring2025/
- 2024 归档页：https://stanford-cs336.github.io/spring2024/index.html

## 核心定位

这个项目更接近一套“从零构建并理解现代 LLM 工程系统”的课程，而不是 API 使用教程。它强调三个层次：

- 理论层：解释 BPE、RoPE、RMSNorm、SwiGLU、MoE、Scaling Laws、RLHF、GRPO 等关键概念。
- 工程层：覆盖 PyTorch 训练、混合精度、FLOPs/显存核算、FlashAttention、CUDA/Triton、分布式训练与推理优化。
- 实践层：配套 6 个作业，从手写 tokenizer 和小模型训练，逐步过渡到系统优化、数据处理、模型对齐和评估。

## 适合人群

- 已有 Python、PyTorch 和深度学习基础，希望系统学习 LLM 底层实现的人。
- 想补齐“大模型训练工程”知识的人，尤其是显存、吞吐、分布式和推理服务相关内容。
- 想用中文材料跟进 CS336 课程的人。
- 已经看过 nanoGPT/nanochat，但希望扩展到更完整 LLM 工程链路的人。

不太适合完全零基础学习者。课程默认学习者能读懂 Python 工程代码，并具备线性代数、概率统计、微积分和机器学习基础。

## 课程结构

| 模块 | 章节 | 重点 |
| --- | --- | --- |
| 导论与工具 | 前言、第 1 章 | 学习路线、实验追踪、W&B、超参数搜索、可视化面板 |
| 模型基础 | 第 2-5 章 | BPE 分词器、PyTorch 资源核算、Transformer 训练细节、MoE |
| 系统优化 | 第 6-8 章 | GPU 体系结构、FlashAttention、Kernel Fusion、CUDA/Triton、分布式训练、ZeRO/FSDP |
| 规模规律 | 第 9 章 | Chinchilla、IsoFLOPs、计算最优模型与数据规模、Scaling 实验设计 |
| 推理与数据 | 第 10-11 章 | KV Cache、投机解码、量化、PagedAttention、连续批处理、数据过滤、去重、PII 脱敏 |
| 评估与训练流程 | 第 12-14 章 | MMLU、HumanEval、HELM、CEval、预训练、SFT、DPO、RLHF、GRPO、RLVR |
| 扩展内容 | 第 15 章 | LLM 推理本质、LeCun 对 LLM 未来的观点等前沿讨论 |

## 作业概览

| 作业 | 主题 | 主要任务 |
| --- | --- | --- |
| 作业 1：手搓大模型 | 基础实现 | 实现 BPE tokenizer、Transformer、AdamW、训练循环，并训练极简语言模型 |
| 作业 2：系统优化 | 训练系统 | 做性能分析与基准测试，实现 FlashAttention-2、分布式训练和优化器分片相关代码 |
| 作业 3：Scaling Laws | 扩展定律 | 训练不同规模模型，拟合 FLOPs-Loss/Chinchilla 缩放关系，预测扩展效果 |
| 作业 4：数据处理 | 预训练数据工程 | 将 Common Crawl 类原始数据转为训练数据，做语言识别、过滤、去重等处理 |
| 作业 5：模型对齐 | SFT 与 RL | 基于数学推理任务做 zero-shot 基线、SFT、Expert Iteration、GRPO 与消融实验 |
| 作业 6：模型评估 | 评测框架 | 使用 lm-evaluation-harness、evalscope、Evalchemy、lighteval 做多维度评测 |

## 重点内容提要

### 1. 分词器与基础模型

课程从 BPE、Unicode 规范化和 tokenizer 训练切入，然后进入现代 Transformer 训练细节。模型部分覆盖 RoPE、RMSNorm、SwiGLU、Pre-Norm/Post-Norm、AdamW、学习率调度等现代 LLM 常用组件。作业 1 会把这些内容落到一个可以训练和生成文本的小型语言模型里。

### 2. 训练资源与系统效率

第 3、6、7、8 章是课程的工程主干。它们把“模型能不能训练起来”拆成可计算的问题：FLOPs、显存占用、内存带宽、计算强度、混合精度、Tensor Cores、FlashAttention、Triton Kernel、数据并行、模型并行、流水线并行、ZeRO 和 FSDP。作业 2 对应这一部分，适合系统性练习训练性能优化。

### 3. Scaling Laws

第 9 章和作业 3 关注模型规模、数据规模和计算预算之间的关系。课程不仅讲 Chinchilla 这类结论，也要求通过实验拟合曲线、分析 IsoFLOPs，并用小规模实验预测大规模训练表现。这部分对理解“为什么这样选模型大小和训练 token 数”很有价值。

### 4. 数据工程

第 11 章和作业 4 强调预训练数据不是简单下载文本，而是要经过抽取、语言识别、质量过滤、MinHash 去重、PII 脱敏、数据混合比例设计等步骤。它补上了很多纯模型教程容易跳过的关键环节。

### 5. 推理与评估

推理章节覆盖 KV Cache、投机解码、量化、PagedAttention 和 continuous batching 等部署前必须理解的概念。评估章节则把 MMLU、HumanEval、HELM、CEval、AlpacaEval、Arena 等评测体系放到一起，作业 6 用主流评测框架进行实践。

### 6. 对齐与可验证奖励

第 13、14 章从预训练之后的 SFT、DPO、RLHF/PPO 讲到 GRPO、rule-based verifier、outcome/process reward 和 RLVR。作业 5 以数学推理为载体，运行 zero-shot、SFT、Expert Iteration、GRPO 和多种消融实验，适合理解 DeepSeek-R1 之后常见的可验证奖励训练路线。

## 与本仓库学习路线的关系

本仓库当前主线偏 Karpathy 风格：先看 nanoGPT，再看 nanochat，重点是用较小代码量理解 GPT 和 ChatGPT 流程。Diy-LLM 可以作为后续扩展材料：

1. 学完 `01-nanoGPT/` 后，对照 Diy-LLM 的第 2-4 章和作业 1，可以补齐 tokenizer、现代 Transformer 组件和训练细节。
2. 学完 `02-nanochat/` 的 pretraining、SFT、RL、serving 后，对照 Diy-LLM 的第 10、12、13、14 章，可以扩展推理、评估和对齐算法视角。
3. 如果目标是训练系统工程，优先读 Diy-LLM 第 6-8 章和作业 2；这部分比 nanoGPT/nanochat 更系统地覆盖 GPU、Triton 和分布式。
4. 如果目标是完整大模型数据闭环，补读第 11 章和作业 4；这部分能弥补很多“小模型复现项目”对真实数据工程讨论不足的问题。

## 推荐学习顺序

1. 先读 README、前言和课程目录，确认自己要补的是模型基础、系统工程、数据工程还是对齐评估。
2. 如果还没手写过 GPT，先做第 2-4 章和作业 1。
3. 如果已经熟悉 nanoGPT，直接进入第 6-8 章和作业 2，重点训练 GPU/分布式系统思维。
4. 再读第 9 章，用 Scaling Laws 理解训练预算、模型规模和数据规模之间的取舍。
5. 最后按目标选择第 10-14 章：部署方向看推理，数据方向看数据工程，研究/产品评测看评估，对齐方向看 SFT、DPO、GRPO 和 RLVR。

## 项目亮点

- 中文资料完整，覆盖 Stanford CS336 的核心路线，并加入本土化说明。
- 理论和代码结合紧密，作业覆盖从 tokenizer 到评估对齐的完整链路。
- 系统优化部分较强，包含 FlashAttention、Triton、CUDA、分布式训练和显存/算力分析。
- 对近年中文学习者关心的 Qwen、DeepSeek、GRPO、RLVR 等内容更友好。
- 适合作为 nanoGPT/nanochat 之后的“系统工程升级材料”。

## 注意事项

- 完整训练和部分系统作业需要 GPU；CPU 更适合阅读、调试和小规模验证。
- `coursework/` 中包含数据、checkpoint、notebook、实验结果等内容，克隆或运行前要关注磁盘占用。
- 部分作业默认模型或路径可能需要按本机环境修改，例如作业 5 中的 Qwen2.5-Math 基座模型路径。
- 课程仍在持续更新，实际学习时应以 GitHub `main` 分支和在线阅读站为准。

## 总结

Diy-LLM 的价值在于把“理解大模型”从单点代码复现扩展成完整工程链路：分词、模型、训练、系统优化、数据、推理、评估和对齐。它适合作为本仓库 nanoGPT/nanochat 学习之后的进阶路线，尤其适合想进一步掌握 CS336 风格训练系统和现代 LLM 工程细节的学习者。
