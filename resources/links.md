# 学习资源汇总

## Karpathy 官方资源

### 视频课程

| 资源 | 链接 | 说明 |
|------|------|------|
| Neural Networks: Zero to Hero | https://karpathy.ai/zero-to-hero.html | 从零开始的完整系列，**强烈推荐** |
| build-nanogpt 视频 | https://github.com/karpathy/build-nanogpt | 逐行构建 GPT，配合 nanoGPT 看 |
| Let's build GPT (YouTube) | https://www.youtube.com/watch?v=kCc8FmEb1nY | 2小时完整讲解 |

### 代码仓库

| 仓库 | 链接 | 说明 |
|------|------|------|
| nanoGPT | https://github.com/karpathy/nanoGPT | GPT 预训练（本学习项目重点） |
| nanochat | https://github.com/karpathy/nanochat | 完整 ChatGPT 流程（本学习项目重点） |
| minbpe | https://github.com/karpathy/minbpe | BPE 分词器最小实现 |
| llm.c | https://github.com/karpathy/llm.c | 纯 C 语言实现 LLM 训练 |
| micrograd | https://github.com/karpathy/micrograd | 最小自动微分引擎 |
| makemore | https://github.com/karpathy/makemore | 字符级语言模型 |

---

## 前置基础资源

### Python & PyTorch

| 资源 | 链接 | 说明 |
|------|------|------|
| PyTorch 60min Blitz | https://pytorch.org/tutorials/beginner/deep_learning_60min_blitz.html | PyTorch 官方入门 |
| PyTorch 官方教程 | https://pytorch.org/tutorials/ | 完整教程列表 |

### 数学基础

| 资源 | 链接 | 说明 |
|------|------|------|
| 3Blue1Brown 线性代数 | https://www.3blue1brown.com/topics/linear-algebra | 可视化直觉 |
| 3Blue1Brown 微积分 | https://www.3blue1brown.com/topics/calculus | 导数直觉 |
| 3Blue1Brown 神经网络 | https://www.3blue1brown.com/topics/neural-networks | NN 可视化 |

---

## nanochat 深入资源

### 代码解析

| 资源 | 链接 | 说明 |
|------|------|------|
| Code-Level Tour of nanochat | https://medium.com/@writeronepagecode/the-100-chatgpt-a-code-level-tour-of-andrej-karpathys-nanochat-729490982bcc | 逐模块代码解析 |
| Notes on Training NanoChat | https://www.weideng.org/posts/nanochat/ | MQA、RoPE 技术细节 |
| 单 GPU 复现 nanochat | https://limcheekin.medium.com/reproducing-karpathys-nanochat-on-a-single-gpu-step-by-step-with-ai-tools-e9420aaee912 | 实操复现指南 |
| $100 and 2 Hours 实战 | https://therawaideas.substack.com/p/nanochat-d20-100-and-2-hours-from | Modal 平台快速跑通 |
| nanochat 官方介绍站 | https://nanochat-ai.com/ | 项目概览 |

### 训练平台

| 平台 | 链接 | 说明 |
|------|------|------|
| Together AI | https://docs.together.ai/docs/nanochat-on-instant-clusters | 官方 nanochat 部署指南 |
| Modal | https://modal.com/ | 按需 GPU 云平台 |
| Lambda Labs | https://lambdalabs.com/ | GPU 租赁 |

---

## 论文 & 理论

### 核心论文

| 论文 | 说明 |
|------|------|
| Attention Is All You Need (2017) | Transformer 原始论文 |
| Language Models are Unsupervised Multitask Learners (GPT-2) | GPT-2 论文 |
| Training language models to follow instructions (InstructGPT) | RLHF 对齐论文 |
| Direct Preference Optimization (DPO) | DPO 算法论文 |
| RoFormer: Enhanced Transformer with Rotary Position Embedding | RoPE 论文 |
| Fast Transformer Decoding: One Write-Head is All You Need | Multi-Query Attention 论文 |

### 博客 & 文章

| 资源 | 链接 | 说明 |
|------|------|------|
| A Recipe for Training NNs | https://karpathy.github.io/2019/04/25/recipe/ | Karpathy 训练经验总结 |
| The Illustrated Transformer | https://jalammar.github.io/illustrated-transformer/ | Transformer 可视化讲解 |
| Lilian Weng's Blog | https://lilianweng.github.io/ | 高质量 ML 总结博客 |

---

## Hugging Face 相关

| 资源 | 链接 | 说明 |
|------|------|------|
| nanochat 模型权重 | https://huggingface.co/sdobson/nanochat | 社区训练的 nanochat 模型 |
| GPT-2 模型 | https://huggingface.co/openai-community/gpt2 | 官方 GPT-2 权重 |

---

## 社区讨论

| 资源 | 链接 | 说明 |
|------|------|------|
| Hacker News 讨论 | https://news.ycombinator.com/item?id=45569350 | nanochat 发布讨论 |
| nanoGPT Issues | https://github.com/karpathy/nanoGPT/issues | 社区问答 |

---

## 推荐学习顺序

```
1. 3Blue1Brown 神经网络系列（建立直觉）
2. Karpathy Zero to Hero 前3集（micrograd → makemore → GPT）
3. PyTorch 60min Blitz（动手能力）
4. build-nanogpt 视频 + 代码（nanoGPT 深入）
5. nanochat Code-Level Tour 文章（nanochat 概览）
6. 动手跑 nanochat（实践）
```
