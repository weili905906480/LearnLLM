# Tokenizer 训练

> nanochat Stage 1：训练 BPE 分词器

## 什么是 Tokenizer

将原始文本转换为模型可以处理的数字序列（token ids）。

```
"Hello world" → [15496, 995]
```

## BPE（Byte Pair Encoding）算法

### 核心思想

从单个字节（256 个）出发，不断合并出现频率最高的相邻 pair，直到词表达到目标大小。

### 算法步骤

```
初始词表：256 个字节

循环：
  1. 统计所有相邻 token pair 的出现频次
  2. 找到频次最高的 pair
  3. 将该 pair 合并为一个新 token，加入词表
  4. 重复直到词表大小 = 目标（nanochat: 32,768）
```

### 示例

```
原始："aaabbc"
字节：[a, a, a, b, b, c]

第1轮：最高频 pair = (a,a) → 合并为 "aa"
结果：[aa, a, b, b, c]

第2轮：最高频 pair = (b,b) → 合并为 "bb"
结果：[aa, a, bb, c]

...
```

## nanochat 的 Tokenizer 配置

| 参数 | 值 |
|------|------|
| 词表大小 | 32,768 |
| 算法 | BPE |
| 训练数据 | ~2B 字符的预训练数据子集 |
| 特殊 token | `<|user|>`, `<|assistant|>`, `<|endoftext|>` 等 |

## 关键设计决策

- [ ] 词表大小的选择（为什么是 32K？）
- [ ] 特殊 token 的定义
- [ ] 字节级 fallback（处理任何 UTF-8 文本）
- [ ] 训练数据的选择对 tokenizer 质量的影响

## 与 GPT-2 Tokenizer 的区别

| 对比 | GPT-2 | nanochat |
|------|-------|----------|
| 词表大小 | 50,257 | 32,768 |
| 基础 | Byte-level BPE | Byte-level BPE |
| 特殊 token | 1个 (`<|endoftext|>`) | 多个（对话标记） |

## 代码笔记

```python
# TODO: 贴入 tokenizer 训练的关键代码并注释
```

## 我的理解/疑问

<!-- 在这里记录你的理解和疑问 -->

