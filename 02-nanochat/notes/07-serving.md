# Serving（部署推理）

> nanochat Stage 6：评估模型 + Web 聊天界面

## 目标

将训练好的模型部署为一个可交互的 ChatGPT 风格 Web 应用。

## 推理流程

```
用户输入 "你好"
    ↓
Tokenize: [user_token, 你, 好, assistant_token]
    ↓
模型自回归生成：
    for each step:
        - 输入当前序列 → 模型 → logits
        - 采样下一个 token
        - 直到生成 <|endoftext|> 或达到 max_len
    ↓
Decode tokens → 文本
    ↓
返回给用户 "你好！有什么可以帮助你的？"
```

## 推理优化：KV Cache

### 为什么需要 KV Cache

```
不用 cache（每步重新计算所有 attention）：
Step 1: 计算 [t1] 的 attention          → 生成 t2
Step 2: 计算 [t1, t2] 的 attention       → 生成 t3
Step 3: 计算 [t1, t2, t3] 的 attention   → 生成 t4
                                          重复计算！

用 cache（缓存之前的 K, V）：
Step 1: 计算 [t1] 的 K,V → 缓存        → 生成 t2
Step 2: 只算 [t2] 的 Q，与缓存的 K,V 做 attention → 生成 t3
Step 3: 只算 [t3] 的 Q，与缓存的 K,V 做 attention → 生成 t4
                                          增量计算！
```

- [ ] KV Cache 的内存占用计算
- [ ] MQA 如何减少 KV Cache（这就是为什么 nanochat 用 MQA）
- [ ] batch 推理时 KV Cache 的管理

## 采样策略

| 策略 | 说明 | 参数 |
|------|------|------|
| Temperature | 控制随机性 | T ∈ (0, 2] |
| Top-k | 只从前 k 个 token 采样 | k |
| Top-p | 从累积概率 ≥ p 的 token 中采样 | p ∈ (0, 1] |
| Repetition Penalty | 惩罚重复 token | — |

## Web UI

- [ ] 前端界面实现
- [ ] WebSocket / SSE 流式输出
- [ ] 对话历史管理
- [ ] 多轮对话的 context 拼接

## 评估

### 常见评估指标

| 指标 | 说明 |
|------|------|
| Perplexity | 模型困惑度（越低越好） |
| MMLU | 知识广度评估 |
| HumanEval | 代码能力 |
| MT-Bench | 对话质量 |

### nanochat 的评估结果

| 配置 | 性能 |
|------|------|
| 4小时 ($100) | 能进行基本对话 |
| 12小时 ($300) | 略超 GPT-2 |

## 部署要点

- [ ] GPU 内存需求计算
- [ ] batch size 对吞吐量的影响
- [ ] 延迟优化（首 token 延迟 vs 生成速度）
- [ ] 量化（int8/int4）是否适用

## 代码笔记

```python
# TODO: 贴入 serving 相关代码并注释
```

## 我的理解/疑问

<!-- 在这里记录你的理解和疑问 -->

