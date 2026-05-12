# model.py 解析

> 文件：nanoGPT/model.py
> 作用：定义 GPT 模型架构

## 类结构

```python
class CausalSelfAttention(nn.Module):
    """多头因果自注意力"""
    pass

class MLP(nn.Module):
    """前馈网络（2层线性 + GELU激活）"""
    pass

class Block(nn.Module):
    """一个 Transformer Block = Attention + MLP"""
    pass

class GPT(nn.Module):
    """完整 GPT 模型"""
    pass
```

## 逐模块笔记

### 1. CausalSelfAttention

**作用：** 计算 token 之间的注意力权重，决定"当前 token 应该关注哪些之前的 token"

**关键点：**
- [ ] Q, K, V 的计算方式
- [ ] 注意力分数 = Q @ K^T / sqrt(d_k)
- [ ] Causal mask 的实现
- [ ] 多头的 reshape 操作
- [ ] Flash Attention 的使用

**代码片段 & 注释：**
```python
# TODO: 贴入关键代码并添加自己的注释
```

---

### 2. MLP (Feed Forward)

**作用：** 对每个 token 的表示做非线性变换，增加模型表达能力

**关键点：**
- [ ] 先升维（×4），再降维
- [ ] GELU 激活函数
- [ ] 为什么需要 FFN（注意力只做线性组合）

**代码片段 & 注释：**
```python
# TODO: 贴入关键代码并添加自己的注释
```

---

### 3. Block

**作用：** 组合 Attention + MLP，加上残差连接和 LayerNorm

**关键点：**
- [ ] Pre-Norm vs Post-Norm（nanoGPT 用的是 Pre-Norm）
- [ ] 残差连接的作用（梯度流动）

---

### 4. GPT (主模型)

**作用：** 组装所有组件，提供 forward / generate 接口

**关键点：**
- [ ] Token Embedding + Position Embedding
- [ ] N 个 Block 堆叠
- [ ] 最后的 LM Head（Linear 层，权重与 Embedding 共享）
- [ ] `generate()` 方法的实现（自回归采样）

---

## 超参数

| 参数 | GPT-2 Small | GPT-2 Medium | GPT-2 Large |
|------|-------------|--------------|-------------|
| n_layer | 12 | 24 | 36 |
| n_head | 12 | 16 | 20 |
| n_embd | 768 | 1024 | 1280 |
| vocab_size | 50257 | 50257 | 50257 |
| block_size | 1024 | 1024 | 1024 |

## 我的理解/疑问

<!-- 在这里记录你的理解和疑问 -->

