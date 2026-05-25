# 01-nanoGPT 学习笔记索引

## 目录结构

```
notes/
├── nanogpt-source/      # nanoGPT 源码解析
├── pytorch-course/      # PyTorch 课程笔记（DeepLearning.AI）
└── theory/              # 理论/概念深入
```

---

## nanogpt-source/ — nanoGPT 源码解析

| 文件 | 内容 |
|------|------|
| [01-overview.md](nanogpt-source/01-overview.md) | 项目总览、核心文件、运行方式 |
| [02-model.md](nanogpt-source/02-model.md) | model.py 前向传播解析（GPT 架构） |
| [03-training.md](nanogpt-source/03-training.md) | train.py 训练流程逐段解析 |
| [04-sampling.md](nanogpt-source/04-sampling.md) | sample.py 推理/采样流程 |
| [10-source-code-walkthrough.md](nanogpt-source/10-source-code-walkthrough.md) | 完整源码目录结构 + 全文件详解 |

**阅读顺序**：01 → 10 → 02 → 03 → 04

---

## pytorch-course/ — PyTorch 课程笔记

### 课程总览

| 文件 | 内容 |
|------|------|
| [00-c1-pytorch-fundamentals.md](pytorch-course/00-c1-pytorch-fundamentals.md) | C1 课程总览：PyTorch 基础 |
| [00-c2-pytorch-techniques.md](pytorch-course/00-c2-pytorch-techniques.md) | C2 课程总览：进阶技术与生态工具 |
| [00-c3-advanced-architectures.md](pytorch-course/00-c3-advanced-architectures.md) | C3 课程总览：高级架构与部署 |

### C1 各章笔记

| 文件 | 内容 |
|------|------|
| [06-tensors.md](pytorch-course/06-tensors.md) | 第1章：Tensor 张量基础 |
| [07-autograd.md](pytorch-course/07-autograd.md) | 第2章：Autograd 自动微分 |
| [08-nn-module.md](pytorch-course/08-nn-module.md) | 第3章：nn.Module 构建神经网络 |
| [09-loss-optimizer.md](pytorch-course/09-loss-optimizer.md) | 第4章：损失函数与优化器 |

### C3 各章笔记

| 文件 | 内容 |
|------|------|
| [c3-05-transformer.md](pytorch-course/c3-05-transformer.md) | 第5章：Transformer 架构 |

---

## theory/ — 理论概念深入

| 文件 | 内容 |
|------|------|
| [05-positional-encoding.md](theory/05-positional-encoding.md) | 位置编码深度解析（正弦/余弦公式推导） |

---

## 学习路径建议

```
1. 先过 PyTorch 基础（pytorch-course/06→07→08→09）
2. 理解 Transformer（pytorch-course/c3-05 + theory/05）
3. 进入 nanoGPT 源码（nanogpt-source/01→10→02→03→04）
```
