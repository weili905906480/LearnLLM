# nanochat 项目总览

> 源码：https://github.com/karpathy/nanochat
> 口号："The best ChatGPT that $100 can buy."

## 项目定位

nanochat 是 nanoGPT 的精神续作，实现了从零训练一个类 ChatGPT 对话 AI 的完整端到端流程。
核心模型代码约 500 行，在单节点 8×H100 上约 4 小时（~$100）可跑通全流程。

## 与 nanoGPT 的区别

| 对比项 | nanoGPT | nanochat |
|--------|---------|----------|
| 范围 | 仅预训练 | 端到端全流程 |
| 产出 | 文本续写模型 | 可对话的聊天助手 |
| 阶段数 | 1（预训练） | 6（完整 pipeline） |
| 模型大小 | GPT-2 (124M-1.5B) | ~561M |
| 新技术 | 标准 MHA | MQA + RoPE + Muon |

## 完整训练 Pipeline

```
┌─────────────────────────────────────────────────────┐
│  Stage 1: Tokenizer Training                        │
│  → 训练 BPE 分词器 (32,768 vocab)                    │
├─────────────────────────────────────────────────────┤
│  Stage 2: Pretraining (Base Model)                  │
│  → 大规模文本上做 next-token prediction               │
├─────────────────────────────────────────────────────┤
│  Stage 3: Midtraining                               │
│  → 高质量数据继续训练，提升知识密度                     │
├─────────────────────────────────────────────────────┤
│  Stage 4: Supervised Fine-Tuning (SFT)              │
│  → 对话数据微调，教模型"怎么聊天"                      │
├─────────────────────────────────────────────────────┤
│  Stage 5: Reinforcement Learning (可选)              │
│  → 强化学习进一步对齐模型行为                          │
├─────────────────────────────────────────────────────┤
│  Stage 6: Evaluation + Web Serving                  │
│  → 评估 + ChatGPT 风格 Web 界面                      │
└─────────────────────────────────────────────────────┘
```

## 核心文件结构（推测）

| 文件/目录 | 作用 |
|-----------|------|
| `train.py` | 训练 orchestrator（调度各阶段） |
| `model.py` | 模型架构（MQA, RoPE, LayerNorm） |
| `tokenizer/` | BPE 分词器训练 |
| `data/` | 数据准备脚本 |
| `sft/` | SFT 相关代码 |
| `serve/` | Web UI 推理服务 |

## 关键技术点

- [ ] Multi-Query Attention (MQA)
- [ ] Rotary Position Embedding (RoPE)
- [ ] Muon 优化器
- [ ] `--depth` 一个参数控制所有超参
- [ ] DDP 多卡训练
- [ ] Chat Template 格式

## 我的理解/疑问

<!-- 在这里记录你的理解和疑问 -->

