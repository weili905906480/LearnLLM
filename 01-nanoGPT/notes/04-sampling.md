# sample.py 解析

> 文件：nanoGPT/sample.py
> 作用：从训练好的模型生成文本

## 采样流程

```
1. 加载模型 checkpoint
2. 设置 prompt（起始文本）
3. 自回归生成：
   for i in range(num_tokens):
       - 输入当前序列 → 模型 → logits
       - 取最后一个位置的 logits
       - 除以 temperature
       - Top-k 过滤
       - Softmax → 概率分布
       - 采样下一个 token
       - 拼接到序列中
4. 解码 token → 文本
```

## 关键概念

### 1. Temperature

- [ ] temperature = 1.0 → 正常采样
- [ ] temperature < 1.0 → 更确定（趋向 greedy）
- [ ] temperature > 1.0 → 更随机（更有创意）
- [ ] 数学原理：logits / T 后再 softmax

### 2. Top-k Sampling

- [ ] 只保留概率最高的 k 个 token
- [ ] 其余 token 概率设为 0
- [ ] 防止低概率的"垃圾 token"被选中

### 3. Top-p (Nucleus) Sampling

- [ ] 保留累积概率 ≥ p 的最小 token 集合
- [ ] 比 top-k 更动态（不同位置保留数量不同）

### 4. Greedy vs Sampling

- [ ] Greedy：每次选概率最大的（确定性）
- [ ] Sampling：按概率随机抽取（多样性）

---

## 实验：不同参数对生成效果的影响

| temperature | top_k | 生成效果 |
|-------------|-------|----------|
| 0.8 | 200 | — |
| 1.0 | 200 | — |
| 1.2 | 200 | — |
| 0.8 | 40 | — |

## 我的理解/疑问

<!-- 在这里记录你的理解和疑问 -->

