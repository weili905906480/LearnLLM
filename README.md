# Learn Karpathy LLM

> 系统学习 Andrej Karpathy 的 [nanoGPT](https://github.com/karpathy/nanoGPT) 和 [nanochat](https://github.com/karpathy/nanochat) 项目，从零理解大语言模型的训练全流程。

---

## 项目目标

通过阅读代码、做笔记、动手实验，深入理解：

1. **nanoGPT** — GPT 模型的预训练原理（Transformer 架构、训练循环）
2. **nanochat** — 从预训练到对话 AI 的完整流程（Tokenizer → Pretrain → SFT → RL → Serving）

---

## 目录结构

```
learn-karpathy-llm/
├── README.md                  # 本文件：项目说明 & 学习路线
├── 01-nanoGPT/                # nanoGPT 学习区
│   ├── notes/                 # 学习笔记
│   │   ├── 01-overview.md     # 项目总览
│   │   ├── 02-model.md        # model.py 解析
│   │   ├── 03-training.md     # train.py 解析
│   │   └── 04-sampling.md     # sample.py 解析
│   ├── code/                  # 自己动手写的代码
│   └── experiments/           # 实验记录
├── 02-nanochat/               # nanochat 学习区
│   ├── notes/                 # 学习笔记
│   │   ├── 01-overview.md     # 项目总览
│   │   ├── 02-tokenizer.md    # 分词器训练
│   │   ├── 03-pretraining.md  # 预训练阶段
│   │   ├── 04-midtraining.md  # 中期训练
│   │   ├── 05-sft.md          # 监督微调
│   │   ├── 06-rl.md           # 强化学习
│   │   └── 07-serving.md      # 部署推理
│   ├── code/                  # 自己动手写的代码
│   └── experiments/           # 实验记录
└── resources/                 # 学习资源汇总
    └── links.md               # 参考链接 & 推荐阅读
```

---

## 学习路线

### Phase 1：前置基础（1-2周）

| 主题 | 内容 | 资源 |
|------|------|------|
| Python | 类、装饰器、生成器 | 官方 Tutorial |
| PyTorch | Tensor、autograd、nn.Module | [60min Blitz](https://pytorch.org/tutorials/beginner/deep_learning_60min_blitz.html) |
| 数学 | 矩阵乘法、导数链式法则 | 3Blue1Brown 系列 |

### Phase 2：nanoGPT 深入（2-3周）

| 步骤 | 内容 | 方式 |
|------|------|------|
| 2.1 | 看 Zero to Hero 前几集 | 视频 + 笔记 |
| 2.2 | 看 build-nanogpt 视频 | 逐行对照代码 |
| 2.3 | 读 model.py | 画架构图，写笔记 |
| 2.4 | 读 train.py | 理解训练循环 |
| 2.5 | 跑 shakespeare 实验 | 动手训练 |

### Phase 3：nanochat 全流程（2-3周）

| 步骤 | 内容 | 方式 |
|------|------|------|
| 3.1 | 通读 nanochat 代码结构 | 画流程图 |
| 3.2 | 理解 Tokenizer 训练 | BPE 算法笔记 |
| 3.3 | 理解 Pretraining | 对比 nanoGPT 的异同 |
| 3.4 | 理解 SFT | 对话格式 & chat template |
| 3.5 | 理解 RL（可选） | RLHF/DPO 概念 |
| 3.6 | 尝试跑通全流程 | 小规模实验 |

---

## 快速开始

```bash
# 克隆本仓库
git clone https://github.com/weili905906480/learn-karpathy-llm.git

# 克隆学习对象
git clone https://github.com/karpathy/nanoGPT.git
git clone https://github.com/karpathy/nanochat.git
```

---

## 进度追踪

- [ ] Phase 1：前置基础
  - [ ] PyTorch 60min Blitz
  - [ ] 线性代数复习
- [ ] Phase 2：nanoGPT
  - [ ] Zero to Hero 系列
  - [ ] build-nanogpt 视频
  - [ ] model.py 笔记
  - [ ] train.py 笔记
  - [ ] shakespeare 实验
- [ ] Phase 3：nanochat
  - [ ] 代码结构总览
  - [ ] Tokenizer 笔记
  - [ ] Pretraining 笔记
  - [ ] SFT 笔记
  - [ ] RL 笔记
  - [ ] 全流程实验

---

## License

MIT
