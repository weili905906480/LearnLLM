# 学习资源汇总

> 💡 优先提供中文资源；英文资源标注 `[英文，需翻译]`

---

## Karpathy 官方资源

### 视频课程

| 资源 | 链接 | 语言 | 说明 |
|------|------|------|------|
| Neural Networks: Zero to Hero | [课程主页](https://karpathy.ai/zero-to-hero.html) / [GitHub](https://github.com/karpathy/nn-zero-to-hero) / [**📄 中文翻译页面**](./zero-to-hero-zh.html) | 英文（含中文译版） | 从零开始的完整系列，**强烈推荐** |
| Ep1: micrograd | [YouTube](https://www.youtube.com/watch?v=VMj-3S1tku0) | 英文，需翻译 | 反向传播 & 自动微分（2.5h） |
| Ep2: makemore Part 1 | [YouTube](https://www.youtube.com/watch?v=PaCmpygFfXo) | 英文，需翻译 | Bigram 字符级语言模型 |
| Ep3: makemore Part 2 | [YouTube](https://www.youtube.com/watch?v=TCH_1BHY58I) | 英文，需翻译 | MLP 语言模型 |
| Ep4: makemore Part 3 | [YouTube](https://www.youtube.com/watch?v=P6sfmUTpUmc) | 英文，需翻译 | BatchNorm 内部机制 |
| Ep5: makemore Part 4 | [YouTube](https://www.youtube.com/watch?v=t3YJ5hKiMQ0) | 英文，需翻译 | WaveNet 架构 |
| Ep6: makemore Part 5 | [YouTube](https://www.youtube.com/watch?v=q8SA3rM6ckI) | 英文，需翻译 | Backprop Ninja |
| Ep7: Let's build GPT | [YouTube](https://www.youtube.com/watch?v=kCc8FmEb1nY) | 英文，需翻译 | 从零构建 GPT（2h，**核心**） |
| Ep8: Tokenizer | [YouTube](https://www.youtube.com/watch?v=zduSFxRajkE) | 英文，需翻译 | BPE 分词器详解 |
| Ep9: GPT-2 复现 | [YouTube](https://www.youtube.com/watch?v=l8pRSuU81PU) | 英文，需翻译 | build-nanogpt 完整视频（4h） |
| build-nanogpt GitHub | [GitHub](https://github.com/karpathy/build-nanogpt) | 英文 | 配套代码仓库 |

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

## 中文资源（按主题分类）

### 🎥 中文视频课程

| 资源 | 链接 | 说明 |
|------|------|------|
| 3Blue1Brown 线性代数 | [Bilibili](https://www.bilibili.com/video/BV1ys411472E) | 官方中文字幕，建立直觉 |
| 3Blue1Brown 微积分 | [Bilibili](https://www.bilibili.com/video/BV1qW411N7FU) | 官方中文字幕，导数直觉 |
| 3Blue1Brown 神经网络 | [Bilibili](https://www.bilibili.com/video/BV1bx411M7Zx) | 官方中文字幕，NN 可视化 |
| 李沐 动手学深度学习 | [d2l.ai 中文版](https://zh.d2l.ai/) / [Bilibili](https://space.bilibili.com/1567748478) | **中文深度学习最佳教材**，配套视频 |
| 李宏毅 机器学习 | [Bilibili](https://www.bilibili.com/video/BV1Wv411h7kN) | 台大 ML 课程，中文授课 |
| 李沐 Transformer 论文精读 | [Bilibili](https://www.bilibili.com/video/BV1pu411o7BE) | 逐段精读 Attention Is All You Need |
| 李沐 GPT/GPT-2/GPT-3 精读 | [Bilibili](https://www.bilibili.com/video/BV1AF411b7xQ) | GPT 系列论文精读 |
| 莫烦 PyTorch 教程 | [GitHub](https://github.com/MorvanZhou/PyTorch-Tutorial) / [morvanzhou.github.io](https://morvanzhou.github.io/tutorials/machine-learning/torch/) | 中文 PyTorch 入门视频 |

### 📖 Transformer & GPT 中文讲解

| 资源 | 链接 | 说明 |
|------|------|------|
| Transformer 模型详解（图解最完整版） | [博客园](https://www.cnblogs.com/kongen/p/18088002) | 中文图解 Transformer，非常详细 |
| 图解 Transformer（Datawhale 翻译） | [GitHub](https://github.com/datawhalechina/learn-nlp-with-transformers/blob/main/docs/%E7%AF%87%E7%AB%A02-Transformer%E7%9B%B8%E5%85%B3%E5%8E%9F%E7%90%86/2.2-%E5%9B%BE%E8%A7%A3transformer.md) | Illustrated Transformer 中文翻译 |
| Transformer 论文代码中文注释 | [GitHub](https://github.com/mcxiaoxiao/annotated-transformer-Chinese) | Annotated Transformer 中文版 |
| Transformer 从入门到精通 | [博客园](https://www.cnblogs.com/Icys/p/18119309/annotated-transformer-chinese) | 中文完整翻译 |
| Transformer 论文解读和源码解析 | [腾讯云](https://cloud.tencent.com/developer/article/2449805) | 论文解读 + PyTorch 源码 |

### 📖 Tokenizer / BPE 中文讲解

| 资源 | 链接 | 说明 |
|------|------|------|
| BPE（字节对编码）：原理、流程与应用详解 | [博客园](https://www.cnblogs.com/myh516/p/19055578) | 中文，BPE 算法完整讲解 |
| 深度学习——BPE 分词算法 | [博客园](https://www.cnblogs.com/smartljy/p/18820636) | 中文，面向 GPT/BERT 的 BPE |
| 大模型词元化处理详解：BPE、WordPiece、Unigram | [腾讯云](https://cloud.tencent.cn/developer/article/2628555) | 中文，三种分词算法对比 |
| HuggingFace BPE 教程 | [HF Course](https://huggingface.co/learn/nlp-course/en/chapter6/5) | 英文，需翻译；有代码实现 |

### 📖 SFT / RLHF / DPO 中文讲解

| 资源 | 链接 | 说明 |
|------|------|------|
| SFT/DPO/PPO/GRPO 训练全解析 | [博客园](https://www.cnblogs.com/yxysuanfa/p/19143050) | **中文**，四种训练算法全面对比 |
| 从0开发大模型之 SFT 训练 | [腾讯云](https://cloud.tencent.cn/developer/article/2500409) | 中文，SFT 实操详解 |
| RLHF 各种训练算法科普 | [博客园](https://www.cnblogs.com/xiaoxi666/p/18723121) | 中文，RLHF 算法全面科普 |
| RLHF 全景：从 PPO 到 DPO（中英双语） | [博客](https://normaluhr.github.io/2025/02/11/rlhf/) | 中英双语，理论推导清晰 |
| 中文大模型微调（LLM-SFT） | [GitHub](https://github.com/yongzhuo/LLM-SFT) | 中文，支持 ChatGLM/LLaMA/LoRA |

### 📖 RoPE 旋转位置编码中文讲解

| 资源 | 链接 | 说明 |
|------|------|------|
| 旋转位置编码 RoPE 详解 | [博客园](https://www.cnblogs.com/deephub/p/18107891) | **中文**，为什么比绝对/相对编码更好 |
| RoPE 详解、代码实现与应用 | [awesomeml.com](https://awesomeml.com/rotary-positional-embedding/) | 中文，含 PyTorch 代码实现 |

### 📖 nanochat 中文讲解

| 资源 | 链接 | 说明 |
|------|------|------|
| Karpathy 开源 NanoChat：100美元+8000行代码手搓 ChatGPT | [博客园](https://www.cnblogs.com/lab4ai/p/19157605) | **中文**，完整项目介绍和流程分析 |
| Code-Level Tour of nanochat | [Medium](https://medium.com/@writeronepagecode/the-100-chatgpt-a-code-level-tour-of-andrej-karpathys-nanochat-729490982bcc) / [Substack](https://onepagecode.substack.com/p/the-100-chatgpt-a-code-level-tour) | 英文，需翻译；逐模块代码解析 |
| Notes on Training NanoChat | [weideng.org](https://www.weideng.org/posts/nanochat/) | 英文，需翻译；MQA、RoPE 技术细节 |
| 单 GPU 复现 nanochat | [Medium](https://limcheekin.medium.com/reproducing-karpathys-nanochat-on-a-single-gpu-step-by-step-with-ai-tools-e9420aaee912) | 英文，需翻译；实操复现指南 |
| $100 and 2 Hours 实战 | [Substack](https://therawaideas.substack.com/p/nanochat-d20-100-and-2-hours-from) | 英文，需翻译；Modal 平台快速跑通 |
| Train an LLM from Scratch | [Trelis Substack](https://trelis.substack.com/p/train-an-llm-from-scratch-with-karpathys) | 英文，需翻译；详细教程 |
| nanochat 官方介绍站 | [nanochat-ai.com](https://nanochat-ai.com/) | 英文，需翻译；项目概览 |

---

## 前置基础资源

### Python & PyTorch

| 资源 | 链接 | 语言 | 说明 |
|------|------|------|------|
| PyTorch 官方中文教程 | [pytorch.ac.cn](https://pytorch.ac.cn/tutorials/beginner/pytorch_with_examples.html) | **中文** | PyTorch 中文文档站 |
| PyTorch 中文教程 | [pytorch-cn.com](https://pytorch-cn.com/tutorials/) | **中文** | 社区中文翻译 |
| PyTorch 官方中文教程 GitHub | [GitHub](https://github.com/fendouai/PyTorchDocs) | **中文** | 60分钟入门 + 强化教程 |
| 莫烦 PyTorch 教程 | [GitHub](https://github.com/MorvanZhou/PyTorch-Tutorial) | **中文** | 配套视频，适合新手 |
| PyTorch 60min Blitz 原版 | [pytorch.org](https://pytorch.org/tutorials/beginner/deep_learning_60min_blitz.html) | 英文，需翻译 | 官方入门 |
| Python 官方 Tutorial | [docs.python.org](https://docs.python.org/zh-cn/3/tutorial/) | **中文** | Python 官方中文文档 |
| NumPy 快速入门 | [numpy.org](https://numpy.org/doc/stable/user/quickstart.html) | 英文，需翻译 | 数组操作基础 |

### 数学基础

| 资源 | 链接 | 语言 | 说明 |
|------|------|------|------|
| 3Blue1Brown 线性代数 | [Bilibili](https://www.bilibili.com/video/BV1ys411472E) | **中文字幕** | 可视化直觉 |
| 3Blue1Brown 微积分 | [Bilibili](https://www.bilibili.com/video/BV1qW411N7FU) | **中文字幕** | 导数直觉 |
| 3Blue1Brown 神经网络 | [Bilibili](https://www.bilibili.com/video/BV1bx411M7Zx) | **中文字幕** | NN 可视化 |
| Matrix Calculus for DL | [explained.ai](https://explained.ai/matrix-calculus/) | 英文，需翻译 | 深度学习矩阵微积分 |

---

## 论文 & 理论

### 核心论文

| 论文 | 链接 | 中文资源 | 说明 |
|------|------|----------|------|
| Attention Is All You Need (2017) | [arxiv:1706.03762](https://arxiv.org/abs/1706.03762) | [李沐精读](https://www.bilibili.com/video/BV1pu411o7BE) | Transformer 原始论文 |
| GPT-2: Language Models are Unsupervised Multitask Learners | [GitHub](https://github.com/openai/gpt-2) | [李沐精读](https://www.bilibili.com/video/BV1AF411b7xQ) | GPT-2 论文 |
| InstructGPT (RLHF) | [arxiv:2203.02155](https://arxiv.org/abs/2203.02155) | [博客园 RLHF 科普](https://www.cnblogs.com/xiaoxi666/p/18723121) | RLHF 对齐论文 |
| Direct Preference Optimization (DPO) | [arxiv:2305.18290](https://arxiv.org/abs/2305.18290) | [SFT/DPO/PPO 全解析](https://www.cnblogs.com/yxysuanfa/p/19143050) | DPO 算法论文 |
| RoFormer (RoPE) | [arxiv:2104.09864](https://arxiv.org/abs/2104.09864) | [中文详解](https://www.cnblogs.com/deephub/p/18107891) | 旋转位置编码 |
| Multi-Query Attention | [arxiv:1911.02150](https://arxiv.org/abs/1911.02150) | 暂无中文，需翻译 | MQA 论文 |
| GPT-1 | [OpenAI](https://openai.com/research/language-unsupervised) | [李沐精读](https://www.bilibili.com/video/BV1AF411b7xQ) | GPT-1 原始论文 |
| GPT-3 | [arxiv:2005.14165](https://arxiv.org/abs/2005.14165) | [李沐精读](https://www.bilibili.com/video/BV1AF411b7xQ) | GPT-3 论文 |
| FlashAttention | [arxiv:2205.14135](https://arxiv.org/abs/2205.14135) | 暂无中文，需翻译 | Flash Attention |

### 博客 & 文章

| 资源 | 链接 | 语言 | 说明 |
|------|------|------|------|
| A Recipe for Training NNs | [karpathy.github.io](https://karpathy.github.io/2019/04/25/recipe/) | 英文，需翻译 | Karpathy 训练经验总结 |
| The Illustrated Transformer | [原文](https://jalammar.github.io/illustrated-transformer/) / [中文翻译](https://github.com/datawhalechina/learn-nlp-with-transformers/blob/main/docs/%E7%AF%87%E7%AB%A02-Transformer%E7%9B%B8%E5%85%B3%E5%8E%9F%E7%90%86/2.2-%E5%9B%BE%E8%A7%A3transformer.md) | 有中文翻译 | Transformer 可视化讲解 |
| The Illustrated GPT-2 | [jalammar.github.io](https://jalammar.github.io/illustrated-gpt2/) | 英文，需翻译 | GPT-2 可视化讲解 |
| Lilian Weng: Attention | [lilianweng.github.io](https://lilianweng.github.io/posts/2018-06-24-attention/) | 英文，需翻译 | 注意力机制全面总结 |
| Lilian Weng: RLHF | [lilianweng.github.io](https://lilianweng.github.io/posts/2023-03-15-rlhf/) | 英文，需翻译 | RLHF 全面总结 |
| Hugging Face RLHF 博客 | [huggingface.co/blog](https://huggingface.co/blog/rlhf) | 英文，需翻译 | RLHF 入门讲解 |

---

## 训练平台

| 平台 | 链接 | 说明 |
|------|------|------|
| Together AI | [docs.together.ai](https://docs.together.ai/docs/nanochat-on-instant-clusters) | 官方 nanochat 部署指南 |
| Modal | [modal.com](https://modal.com/) | 按需 GPU 云平台（适合快速实验） |
| Lambda Labs | [lambdalabs.com](https://lambdalabs.com/) | GPU 租赁 |
| Vast.ai | [vast.ai](https://vast.ai/) | 低价 GPU 租赁 |
| RunPod | [runpod.io](https://www.runpod.io/) | GPU 云平台 |
| AutoDL | [autodl.com](https://www.autodl.com/) | **国内** GPU 租赁，价格便宜 |

---

## Hugging Face 相关

| 资源 | 链接 | 语言 | 说明 |
|------|------|------|------|
| nanochat 模型权重 | [huggingface.co/sdobson/nanochat](https://huggingface.co/sdobson/nanochat) | — | 社区训练的 nanochat 561M 模型 |
| GPT-2 模型 | [huggingface.co/openai-community/gpt2](https://huggingface.co/openai-community/gpt2) | — | 官方 GPT-2 权重 |
| HF NLP Course | [huggingface.co/learn/nlp-course](https://huggingface.co/learn/nlp-course) | 英文，需翻译 | Hugging Face 官方 NLP 课程 |
| Tokenizers 文档 | [huggingface.co/docs/tokenizers](https://huggingface.co/docs/tokenizers) | 英文，需翻译 | Tokenizer 库文档 |
| HuggingFace Transformers 中文文档 | [GitHub](https://github.com/liuzard/transformers_zh_docs) | **中文** | 社区中文翻译 |

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

## 推荐学习顺序

```
1. 3Blue1Brown 神经网络系列 B站（建立直觉，中文字幕）
2. 李沐 动手学深度学习 前几章（中文，PyTorch基础）
3. 李沐 Transformer 论文精读（中文，理解核心架构）
4. Karpathy Zero to Hero Ep7: Let's build GPT（英文，核心！配合中文 Transformer 讲解）
5. Karpathy Zero to Hero Ep8: Tokenizer（英文，配合中文 BPE 博客）
6. Karpathy Zero to Hero Ep9: GPT-2 复现（英文，= build-nanogpt）
7. 阅读 nanoGPT 源码（对照视频 + 中文 Transformer 图解）
8. 博客园 nanochat 中文讲解（了解全流程）
9. nanochat Code-Level Tour 文章（英文，深入代码）
10. 动手跑 nanochat（实践，AutoDL 或 Modal）
```
