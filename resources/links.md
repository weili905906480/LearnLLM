# 学习资源汇总

## Karpathy 官方资源

### 视频课程

| 资源 | 链接 | 说明 |
|------|------|------|
| Neural Networks: Zero to Hero | [课程主页](https://karpathy.ai/zero-to-hero.html) / [GitHub](https://github.com/karpathy/nn-zero-to-hero) | 从零开始的完整系列，**强烈推荐** |
| Ep1: micrograd | [YouTube](https://www.youtube.com/watch?v=VMj-3S1tku0) | 反向传播 & 自动微分（2.5h） |
| Ep2: makemore Part 1 | [YouTube](https://www.youtube.com/watch?v=PaCmpygFfXo) | Bigram 字符级语言模型 |
| Ep3: makemore Part 2 | [YouTube](https://www.youtube.com/watch?v=TCH_1BHY58I) | MLP 语言模型 |
| Ep4: makemore Part 3 | [YouTube](https://www.youtube.com/watch?v=P6sfmUTpUmc) | BatchNorm 内部机制 |
| Ep5: makemore Part 4 | [YouTube](https://www.youtube.com/watch?v=t3YJ5hKiMQ0) | WaveNet 架构 |
| Ep6: makemore Part 5 | [YouTube](https://www.youtube.com/watch?v=q8SA3rM6ckI) | Backprop Ninja |
| Ep7: Let's build GPT | [YouTube](https://www.youtube.com/watch?v=kCc8FmEb1nY) | 从零构建 GPT（2h，**核心**） |
| Ep8: Tokenizer | [YouTube](https://www.youtube.com/watch?v=zduSFxRajkE) | BPE 分词器详解 |
| Ep9: GPT-2 复现 | [YouTube](https://www.youtube.com/watch?v=l8pRSuU81PU) | build-nanogpt 完整视频（4h） |
| build-nanogpt GitHub | [GitHub](https://github.com/karpathy/build-nanogpt) | 配套代码仓库 |

### 代码仓库

| 仓库 | 链接 | 说明 |
|------|------|------|
| nanoGPT | [github.com/karpathy/nanoGPT](https://github.com/karpathy/nanoGPT) | GPT 预训练（本学习项目重点） |
| nanochat | [github.com/karpathy/nanochat](https://github.com/karpathy/nanochat) | 完整 ChatGPT 流程（本学习项目重点） |
| nn-zero-to-hero | [github.com/karpathy/nn-zero-to-hero](https://github.com/karpathy/nn-zero-to-hero) | Zero to Hero 系列配套代码 |
| minbpe | [github.com/karpathy/minbpe](https://github.com/karpathy/minbpe) | BPE 分词器最小实现 |
| llm.c | [github.com/karpathy/llm.c](https://github.com/karpathy/llm.c) | 纯 C 语言实现 LLM 训练 |
| micrograd | [github.com/karpathy/micrograd](https://github.com/karpathy/micrograd) | 最小自动微分引擎 |
| makemore | [github.com/karpathy/makemore](https://github.com/karpathy/makemore) | 字符级语言模型 |

---

## 前置基础资源

### Python & PyTorch

| 资源 | 链接 | 说明 |
|------|------|------|
| PyTorch 60min Blitz | [pytorch.org/tutorials](https://pytorch.org/tutorials/beginner/deep_learning_60min_blitz.html) | PyTorch 官方入门 |
| PyTorch 官方教程 | [pytorch.org/tutorials](https://pytorch.org/tutorials/) | 完整教程列表 |
| Python 官方 Tutorial | [docs.python.org](https://docs.python.org/3/tutorial/) | Python 基础 |
| NumPy 快速入门 | [numpy.org](https://numpy.org/doc/stable/user/quickstart.html) | 数组操作基础 |

### 数学基础

| 资源 | 链接 | 说明 |
|------|------|------|
| 3Blue1Brown 线性代数 | [3blue1brown.com](https://www.3blue1brown.com/topics/linear-algebra) / [Bilibili](https://www.bilibili.com/video/BV1ys411472E) | 可视化直觉 |
| 3Blue1Brown 微积分 | [3blue1brown.com](https://www.3blue1brown.com/topics/calculus) / [Bilibili](https://www.bilibili.com/video/BV1qW411N7FU) | 导数直觉 |
| 3Blue1Brown 神经网络 | [3blue1brown.com](https://www.3blue1brown.com/topics/neural-networks) / [Bilibili](https://www.bilibili.com/video/BV1bx411M7Zx) | NN 可视化 |
| Matrix Calculus for DL | [explained.ai](https://explained.ai/matrix-calculus/) | 深度学习矩阵微积分 |

---

## nanochat 深入资源

### 代码解析

| 资源 | 链接 | 说明 |
|------|------|------|
| Code-Level Tour of nanochat | [Medium](https://medium.com/@writeronepagecode/the-100-chatgpt-a-code-level-tour-of-andrej-karpathys-nanochat-729490982bcc) / [Substack](https://onepagecode.substack.com/p/the-100-chatgpt-a-code-level-tour) | 逐模块代码解析 |
| Notes on Training NanoChat | [weideng.org](https://www.weideng.org/posts/nanochat/) | MQA、RoPE 技术细节 |
| 单 GPU 复现 nanochat | [Medium](https://limcheekin.medium.com/reproducing-karpathys-nanochat-on-a-single-gpu-step-by-step-with-ai-tools-e9420aaee912) | 实操复现指南 |
| $100 and 2 Hours 实战 | [Substack](https://therawaideas.substack.com/p/nanochat-d20-100-and-2-hours-from) | Modal 平台快速跑通 |
| Train an LLM from Scratch | [Trelis Substack](https://trelis.substack.com/p/train-an-llm-from-scratch-with-karpathys) | Trelis 详细教程 |
| nanochat 官方介绍站 | [nanochat-ai.com](https://nanochat-ai.com/) | 项目概览 |
| Build Your Own ChatGPT | [emelia.io](https://emelia.io/hub/nanochat-karpathy) | 综合介绍 + autoresearch 扩展 |

### 训练平台

| 平台 | 链接 | 说明 |
|------|------|------|
| Together AI | [docs.together.ai](https://docs.together.ai/docs/nanochat-on-instant-clusters) | 官方 nanochat 部署指南 |
| Modal | [modal.com](https://modal.com/) | 按需 GPU 云平台（适合快速实验） |
| Lambda Labs | [lambdalabs.com](https://lambdalabs.com/) | GPU 租赁 |
| Vast.ai | [vast.ai](https://vast.ai/) | 低价 GPU 租赁 |
| RunPod | [runpod.io](https://www.runpod.io/) | GPU 云平台 |

---

## 论文 & 理论

### 核心论文

| 论文 | 链接 | 说明 |
|------|------|------|
| Attention Is All You Need (2017) | [arxiv:1706.03762](https://arxiv.org/abs/1706.03762) | Transformer 原始论文 |
| Language Models are Unsupervised Multitask Learners | [OpenAI Blog](https://openai.com/research/better-language-models) / [GitHub](https://github.com/openai/gpt-2) | GPT-2 论文 |
| Training language models to follow instructions (InstructGPT) | [arxiv:2203.02155](https://arxiv.org/abs/2203.02155) | RLHF 对齐论文 |
| Direct Preference Optimization (DPO) | [arxiv:2305.18290](https://arxiv.org/abs/2305.18290) | DPO 算法论文 |
| RoFormer: Enhanced Transformer with Rotary Position Embedding | [arxiv:2104.09864](https://arxiv.org/abs/2104.09864) | RoPE 论文 |
| Fast Transformer Decoding: One Write-Head is All You Need | [arxiv:1911.02150](https://arxiv.org/abs/1911.02150) | Multi-Query Attention 论文 |
| Improving Language Understanding by Generative Pre-Training (GPT-1) | [OpenAI](https://openai.com/research/language-unsupervised) | GPT-1 原始论文 |
| Language Models are Few-Shot Learners (GPT-3) | [arxiv:2005.14165](https://arxiv.org/abs/2005.14165) | GPT-3 论文 |
| FlashAttention: Fast and Memory-Efficient Exact Attention | [arxiv:2205.14135](https://arxiv.org/abs/2205.14135) | Flash Attention 论文 |

### 博客 & 文章

| 资源 | 链接 | 说明 |
|------|------|------|
| A Recipe for Training NNs | [karpathy.github.io](https://karpathy.github.io/2019/04/25/recipe/) | Karpathy 训练经验总结 |
| The Illustrated Transformer | [jalammar.github.io](https://jalammar.github.io/illustrated-transformer/) | Transformer 可视化讲解 |
| The Illustrated GPT-2 | [jalammar.github.io](https://jalammar.github.io/illustrated-gpt2/) | GPT-2 可视化讲解 |
| Lilian Weng's Blog | [lilianweng.github.io](https://lilianweng.github.io/) | 高质量 ML 总结博客 |
| Lil'Log: Attention | [lilianweng.github.io](https://lilianweng.github.io/posts/2018-06-24-attention/) | 注意力机制全面总结 |
| Lil'Log: RLHF | [lilianweng.github.io](https://lilianweng.github.io/posts/2023-03-15-rlhf/) | RLHF 全面总结 |
| Hugging Face RLHF 博客 | [huggingface.co/blog](https://huggingface.co/blog/rlhf) | RLHF 入门讲解 |
| Simon Willison on nanochat | [simonwillison.net](http://feeds.simonwillison.net/random/andrej-karpathy/) | nanochat 评测笔记 |

---

## Hugging Face 相关

| 资源 | 链接 | 说明 |
|------|------|------|
| nanochat 模型权重 | [huggingface.co/sdobson/nanochat](https://huggingface.co/sdobson/nanochat) | 社区训练的 nanochat 561M 模型 |
| GPT-2 模型 | [huggingface.co/openai-community/gpt2](https://huggingface.co/openai-community/gpt2) | 官方 GPT-2 权重 |
| HF NLP Course | [huggingface.co/learn/nlp-course](https://huggingface.co/learn/nlp-course) | Hugging Face 官方 NLP 课程 |
| Tokenizers 文档 | [huggingface.co/docs/tokenizers](https://huggingface.co/docs/tokenizers) | Tokenizer 库文档 |

---

## 社区讨论

| 资源 | 链接 | 说明 |
|------|------|------|
| Hacker News: nanochat | [news.ycombinator.com](https://news.ycombinator.com/item?id=45569350) | nanochat 发布讨论 |
| nanoGPT Issues | [github.com/karpathy/nanoGPT/issues](https://github.com/karpathy/nanoGPT/issues) | 社区问答 |
| nanochat Issues | [github.com/karpathy/nanochat/issues](https://github.com/karpathy/nanochat/issues) | nanochat 社区问答 |
| r/MachineLearning | [reddit.com/r/MachineLearning](https://www.reddit.com/r/MachineLearning/) | ML 社区讨论 |
| r/LocalLLaMA | [reddit.com/r/LocalLLaMA](https://www.reddit.com/r/LocalLLaMA/) | 本地 LLM 社区 |

---

## 中文资源

| 资源 | 链接 | 说明 |
|------|------|------|
| 3Blue1Brown 线性代数（B站） | [Bilibili](https://www.bilibili.com/video/BV1ys411472E) | 官方中文字幕 |
| 3Blue1Brown 神经网络（B站） | [Bilibili](https://www.bilibili.com/video/BV1bx411M7Zx) | 官方中文字幕 |
| 李沐 动手学深度学习 | [d2l.ai/zh](https://zh.d2l.ai/) / [Bilibili](https://space.bilibili.com/1567748478) | 中文深度学习教材 + 视频 |
| 李宏毅 机器学习 | [Bilibili](https://www.bilibili.com/video/BV1Wv411h7kN) | 台大 ML 课程（中文） |
| Transformer 论文逐段精读 | [Bilibili: 沐神](https://www.bilibili.com/video/BV1pu411o7BE) | 李沐论文精读系列 |
| GPT 论文精读 | [Bilibili: 沐神](https://www.bilibili.com/video/BV1AF411b7xQ) | 李沐 GPT 系列精读 |

---

## 推荐学习顺序

```
1. 3Blue1Brown 神经网络系列（建立直觉）
2. Karpathy Zero to Hero Ep1-Ep3（micrograd → makemore → MLP）
3. PyTorch 60min Blitz（动手能力）
4. Zero to Hero Ep7: Let's build GPT（核心！）
5. Zero to Hero Ep8: Tokenizer（分词器）
6. Zero to Hero Ep9: GPT-2 复现（= build-nanogpt）
7. 阅读 nanoGPT 源码（对照视频）
8. nanochat Code-Level Tour 文章（nanochat 概览）
9. 动手跑 nanochat（实践）
```
