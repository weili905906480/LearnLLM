# Supervised Fine-Tuning (SFT)

> nanochat Stage 4：教模型"怎么聊天"

## 目标

将预训练的 base model 变成一个能**遵循指令、进行对话**的助手。

## 核心区别

```
Base Model（预训练后）：
  输入: "What is the capital of France?"
  输出: "What is the capital of Germany? What is..." （续写问题）

SFT 之后：
  输入: "What is the capital of France?"
  输出: "The capital of France is Paris."（回答问题）
```

**SFT 的本质：** 不是教模型新知识，而是教模型一种新的"行为模式"——看到问题要回答，而不是续写。

## Chat Template（对话格式）

```
<|user|>
你好，请介绍一下自己
<|assistant|>
你好！我是一个 AI 助手，很高兴认识你。有什么我可以帮助你的吗？
<|endoftext|>
```

### 关键设计

- [ ] 特殊 token 的定义（`<|user|>`, `<|assistant|>`, `<|endoftext|>`）
- [ ] 多轮对话的拼接方式
- [ ] Loss mask：只在 assistant 回复部分计算 loss

## Loss Masking

```
Tokens:     <|user|> 你 好 <|assistant|> 你 好 ！ <|endoftext|>
Loss mask:     0     0  0       0          1  1  1      1

只在 assistant 的回复部分计算 loss！
用户输入部分不参与 loss 计算。
```

- [ ] 为什么要 mask 用户输入？
- [ ] 如果不 mask 会怎样？

## SFT 数据

### nanochat 使用的数据集

| 数据集 | 说明 |
|--------|------|
| SmolTalk（或类似） | 高质量对话数据 |

### 数据格式示例

```json
{
  "messages": [
    {"role": "user", "content": "解释什么是机器学习"},
    {"role": "assistant", "content": "机器学习是人工智能的一个分支..."}
  ]
}
```

## SFT 训练配置

| 参数 | 与预训练的区别 |
|------|--------------|
| 学习率 | 更小（通常 1e-5 ~ 5e-5） |
| Epochs | 1-3 epochs（数据量小，不能过拟合） |
| 数据量 | 几千到几十万条对话 |
| 优化器 | 通常 AdamW |

## 关键问题

- [ ] SFT 数据的质量 vs 数量哪个更重要？
- [ ] 过拟合的风险（数据少时容易过拟合）
- [ ] SFT 后模型会"忘记"预训练知识吗？
- [ ] 为什么 loss 只在 assistant 部分计算？

## SFT 之后模型的状态

- ✅ 能遵循指令
- ✅ 能进行多轮对话
- ✅ 输出格式规范
- ⚠️ 可能还有有害输出（需要 RL 进一步对齐）
- ⚠️ 可能过于"听话"（sycophantic）

## 代码笔记

```python
# TODO: 贴入 SFT 相关代码并注释
```

## 我的理解/疑问

<!-- 在这里记录你的理解和疑问 -->

