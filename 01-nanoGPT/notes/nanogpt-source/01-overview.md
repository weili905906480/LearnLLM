# nanoGPT 项目总览

> 源码：https://github.com/karpathy/nanoGPT
> 配套视频：https://github.com/karpathy/build-nanogpt

## 项目定位

nanoGPT 是最简单、最快速的中等规模 GPT 训练/微调代码库。核心代码约 600 行。

## 核心文件

| 文件 | 行数 | 作用 |
|------|------|------|
| `model.py` | ~300 | GPT 模型定义（Transformer 架构） |
| `train.py` | ~300 | 训练循环、优化器、分布式 |
| `sample.py` | ~60 | 从模型生成文本 |
| `config/` | — | 各种训练配置文件 |
| `data/` | — | 数据准备脚本 |

## 架构概览

```
Input Tokens
    ↓
Token Embedding + Position Embedding
    ↓
┌─────────────────────────┐
│  Transformer Block × N  │
│  ┌───────────────────┐  │
│  │ Layer Norm        │  │
│  │ Multi-Head Attn   │  │
│  │ Residual Add      │  │
│  │ Layer Norm        │  │
│  │ Feed Forward(MLP) │  │
│  │ Residual Add      │  │
│  └───────────────────┘  │
└─────────────────────────┘
    ↓
Layer Norm
    ↓
Linear (vocab_size)
    ↓
Output Logits → Softmax → Next Token
```

## 关键概念

- [ ] Transformer Decoder-only 架构
- [ ] Self-Attention 机制 (Q, K, V)
- [ ] 位置编码 (Positional Encoding)
- [ ] Layer Normalization
- [ ] Residual Connection
- [ ] Causal Mask（防止看到未来 token）
- [ ] AdamW 优化器
- [ ] Learning Rate Schedule (warmup + cosine decay)

## 我的理解/疑问

<!-- 在这里记录你的理解和疑问 -->

