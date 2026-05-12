# Pretraining（预训练）

> nanochat Stage 2：在大规模文本上训练 Base Model

## 目标

让模型学会 **next-token prediction**：给定前面的文本，预测下一个 token。

```
输入: "The cat sat on the"
目标: "mat"（或任何合理的下一个词）
```

这个看似简单的任务，在大规模数据上训练后，模型会涌现出语言理解、知识存储、推理等能力。

## 与 nanoGPT 预训练的对比

| 对比项 | nanoGPT | nanochat |
|--------|---------|----------|
| 数据集 | OpenWebText (9B tokens) | ClimbMix-400B / FineWeb |
| 注意力 | 标准 MHA | Multi-Query Attention |
| 位置编码 | 学习式绝对位置 | RoPE（旋转位置编码） |
| 优化器 | AdamW | Muon + AdamW |
| Norm | Pre-LayerNorm | Pre-LayerNorm |

## 关键概念

### 1. 数据：ClimbMix / FineWeb

- [ ] 数据来源和规模
- [ ] 数据清洗和过滤流程
- [ ] token 化后的数据格式

### 2. Multi-Query Attention (MQA)

```
标准 MHA:  每个 head 有独立的 Q, K, V
MQA:       每个 head 有独立的 Q，但共享 K, V

好处：推理时 KV cache 内存大幅减少
代价：训练时表达能力略有损失（实际影响很小）
```

- [ ] MQA 的实现方式
- [ ] 相比 MHA 的 KV cache 节省比例
- [ ] 为什么对训练影响小但推理加速大

### 3. RoPE（Rotary Position Embedding）

```
核心思想：通过旋转向量来编码位置信息
- 相对位置通过向量夹角体现
- 天然支持长度外推
```

- [ ] RoPE 的数学直觉
- [ ] 相比学习式位置编码的优势
- [ ] 长文本外推能力

### 4. Muon 优化器

- [ ] Muon 是什么？与 AdamW 的区别
- [ ] 为什么 nanochat 选择 Muon
- [ ] 哪些参数用 Muon，哪些用 AdamW

### 5. `--depth` 参数

nanochat 通过一个 `--depth` 参数自动设定所有超参数：

- [ ] depth 越大 → 模型越大 → 训练越久
- [ ] 自动计算：n_layer, n_head, n_embd, batch_size, lr 等

## 训练配置

| 参数 | 值（参考） |
|------|-----------|
| 模型大小 | ~561M |
| 训练时间 | ~2-4 小时 (8×H100) |
| 序列长度 | — |
| 全局 batch size | — |
| 学习率 | — |

## 预训练结束后模型能做什么？

- ✅ 流畅的文本续写
- ✅ 存储世界知识
- ❌ 还不会"聊天"（不会遵循指令）
- ❌ 可能输出有害内容（未对齐）

## 代码笔记

```python
# TODO: 贴入预训练相关的关键代码并注释
```

## 我的理解/疑问

<!-- 在这里记录你的理解和疑问 -->

