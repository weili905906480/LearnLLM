# MiniMind 模型实现详解（结合具体矩阵示例）

> 对应源码：`model/model_minimind.py`

这是一个 **MiniMind**（极简 GPT 风格的因果语言模型，支持 **MoE** 稀疏专家和 **GQA** 分组注意力）的完整实现，包含从配置、RoPE、注意力、SwiGLU FFN、MoE 路由，到 `generate` 采样的全链路。

先约定一个用于演示的小配置（真实默认值是 `hidden_size=768`，这里为了手算缩小）：

```
hidden_size D = 8      # 词向量维度
num_attention_heads H = 2
num_key_value_heads   = 1   # GQA：1 个 KV 头服务 2 个 Q 头
head_dim = 4                # 8 / 2
seq_len = 3                 # 假设输入 3 个 token
vocab_size = 6400
```

---

## 1. `MiniMindConfig`（配置类，[model_minimind.py:10-45](../model/model_minimind.py#L10-L45)）

只是把模型超参集中存起来。几个值得注意的点：

- **`intermediate_size`（L26）**：`ceil(hidden_size * π / 64) * 64`。以 768 为例：`768*π ≈ 2412.7 → /64 ≈ 37.7 → ceil=38 → 2432`。本质就是把 FFN 宽度向上取整到 64 的倍数，没有特殊含义。
- **`rope_scaling`（L32-39）**：只有 `inference_rope_scaling=True` 时才构造 YaRN 配置，否则为 `None`（默认关闭）。
- **MoE 配置（L41-45）**：`num_experts=4`、`num_experts_per_tok=1`（top-1 路由）等，`use_moe=False` 时全部忽略。

---

## 2. `RMSNorm`（[model_minimind.py:50-60](../model/model_minimind.py#L50-L60)）

RMSNorm 是 LLaMA 系常用的归一化，**不做均值中心化**，只按均方根缩放。

```
RMS(x) = x / sqrt(mean(x²) + eps)   # 然后再乘可学习的 weight
```

代码 L57 里 `torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)` 就是 `1/sqrt(mean(x²)+eps)`。

**矩阵示例**（dim=4，eps 忽略）：

```
x = [1, 3, 2, 4]
x² = [1, 9, 4, 16]
mean(x²) = 30/4 = 7.5
sqrt = 2.739
x / 2.739 = [0.365, 1.095, 0.730, 1.461]
再逐元素乘 weight（初始为 1，训练中学习）
```

`forward` 里 `.float()` 提升精度计算、`.type_as(x)` 转回原 dtype，是为了混合精度训练时数值稳定（L59-60）。

---

## 3. RoPE 旋转位置编码（[model_minimind.py:62-84](../model/model_minimind.py#L62-L84)）

### 3.1 `precompute_freqs_cis`（L62-78）——预计算频率表

核心公式（L63）：

```
freqs[i] = 1 / rope_base^(2i/dim),   i = 0, 1, ..., dim/2-1
```

**矩阵示例**（head_dim=4，rope_base=1e6）：

```
dim//2 = 2 个频率
arange(0,4,2) = [0, 2]  → /4 → [0, 0.5]
base^[0, 0.5] = [1, 1000]
freqs = [1/1, 1/1000] = [1.0, 0.001]
```

然后 L74-75 计算所有位置的角度：`torch.outer(t, freqs)`，得到形状 `[end, 2]` 的角度矩阵（`end=32768` 是最大位置）。

最后 L76-78 拼出 cos/sin 表，形状都是 `[end, dim]`：

```
freqs_cos = [cos(θ0), cos(θ1), cos(θ0), cos(θ1)]   # cat 两次
freqs_sin = [sin(θ0), sin(θ1), sin(θ0), sin(θ1)]
```

注意这里是 **`cos(freqs)` 复制两份**，对应的是「前半/后半配对」的旋转方式（LLaMA 风格，不是 GPT-NeoX 的相邻配对）。

### 3.2 `apply_rotary_pos_emb`（L80-84）——旋转

```
rotate_half(x) = [-x[后半], x[前半]]
q_embed = q * cos + rotate_half(q) * sin
```

**矩阵示例**（head_dim=4，某个头某位置的 q = `[1,2,3,4]`，θ0、θ1 对应上一步）：

```
rotate_half(q) = [-3, -4, 1, 2]
设 c0=cos(θ0), s0=sin(θ0), c1=cos(θ1), s1=sin(θ1)

q_embed[0] = 1*c0 + (-3)*s0 = c0 - 3*s0
q_embed[1] = 2*c1 + (-4)*s1 = 2*c1 - 4*s1
q_embed[2] = 3*c0 + 1*s0 = 3*c0 + s0
q_embed[3] = 4*c1 + 2*s1 = 4*c1 + 2*s1
```

可以看到**配对是 (维度0, 维度2) 用 θ0、(维度1, 维度3) 用 θ1**——这与上面 cos 表「θ0 出现两次、θ1 出现两次」正好一致。RoPE 的核心思想：位置信息通过「按位置角度旋转向量」编码进去，且**只依赖于相对位置差**，所以能外推到更长序列。

---

## 4. `repeat_kv`（[model_minimind.py:86-89](../model/model_minimind.py#L86-L89)）——GQA 的 KV 头复制

当 `num_key_value_heads < num_attention_heads` 时，需要把 KV 头复制多份去匹配 Q 头数量。

**矩阵示例**（xk 形状 `[1, 3, 1, 4]`，n_rep=2）：

```
x[:, :, :, None, :]     # [1,3,1,4] → [1,3,1,1,4]
.expand(...)            # → [1,3,1,2,4]   （广播，不复制内存）
.reshape(1,3,2,4)       # → [1,3,2,4]     （1 个 KV 头复制成 2 个）
```

这样 2 个 Q 头就能各自和同一个 KV 头做注意力，节省 KV cache 显存。

---

## 5. `Attention`（[model_minimind.py:91-134](../model/model_minimind.py#L91-L134)）——注意力核心

### 5.1 初始化（L92-109）

- `q_proj`：`D → H×head_dim`（8→8）
- `k_proj` / `v_proj`：`D → kv_heads×head_dim`（8→4，因为 kv_heads=1）
- `o_proj`：`H×head_dim → D`
- `q_norm` / `k_norm`：对每个头的向量再做一次 RMSNorm（QK-Norm，稳定训练）

### 5.2 `forward` 完整流程（L111-134）

**矩阵示例**（x 形状 `[1, 3, 8]`，输入 3 个 token）：

```python
# 1. 投影 (L113)
xq = x @ W_q^T   # [1, 3, 8]  (2头 × 4维)
xk = x @ W_k^T   # [1, 3, 4]  (1头 × 4维)
xv = x @ W_v^T   # [1, 3, 4]

# 2. 切头 (L114-116)
xq → [1, 3, 2, 4]      # (batch, seq, 2头, 4维)
xk → [1, 3, 1, 4]
xv → [1, 3, 1, 4]

# 3. QK-Norm (L117) + RoPE (L119)
xq, xk = q_norm(xq), k_norm(xk)      # 每个 4 维向量 RMSNorm
xq, xk = apply_rotary_pos_emb(xq, xk, cos, sin)   # 加位置

# 4. 拼接历史 KV（KV cache，L120-123）
# 生成时 past_key_value 存了之前所有 token 的 K/V，拼到前面

# 5. 转置成 (batch, heads, seq, head_dim) (L124)
xq  → [1, 2, 3, 4]
xk  → repeat_kv 后 → [1, 2, 3, 4]
xv  → repeat_kv 后 → [1, 2, 3, 4]

# 6. 计算注意力 (L125-131)
```

**注意力分数计算**（走非 flash 分支，L128-131）：

```python
scores = xq @ xk^T / sqrt(head_dim)   # [1, 2, 3, 3]，每个头一个 3×3 矩阵
```

比如第 0 个头，3 个 token 两两打分得到 3×3 矩阵 `S`：

```
S = [ s(我,我)  s(我,爱)  s(我,你)
      s(爱,我)  s(爱,爱)  s(爱,你)
      s(你,我)  s(你,爱)  s(你,你) ] / sqrt(4)
```

**因果掩码**（L129）：只允许看到当前及之前的 token：

```python
# triu(1) 上三角全 -inf，加到 S 上
S = S + [  0, -∞, -∞
           0,  0, -∞
           0,  0,  0 ]
```

这样第 0 行只看「我」，第 1 行看「我 爱」，第 2 行看「我 爱 你」。`-seq_len:` 这个切片是为了处理「有历史 KV cache」时只对新 token 做掩码。

**softmax + 加权求和**（L131）：

```
output = softmax(S) @ xv     # [1, 2, 3, 4]
```

**flash 分支**（L125-126）：当 `seq_len>1`、无缓存、mask 全 1 时用 `F.scaled_dot_product_attention` 走融合内核，省显存更快。

最后 L132-133 把 `[1,2,3,4]` 转回 `[1,3,8]` 再过 `o_proj` 和残差 dropout。

---

## 6. `FeedForward`（[model_minimind.py:136-146](../model/model_minimind.py#L136-L146)）——SwiGLU

标准 SwiGLU 前馈层，三块权重：

```
FFN(x) = down_proj( silu(gate_proj(x)) ⊙ up_proj(x) )
```

**矩阵示例**（D=8，intermediate=16）：

```
x              → [1, 3, 8]
gate_proj(x)   → [1, 3, 16]  → silu 激活
up_proj(x)     → [1, 3, 16]
两者逐元素相乘   → [1, 3, 16]   # "门控"：gate 控制 up 哪些维度通过
down_proj(...) → [1, 3, 8]
```

其中 `silu(x) = x * sigmoid(x)`。`gate_proj` 先激活再乘 `up_proj`，这是区别于普通 FFN 的「门控线性单元」结构。

---

## 7. `MOEFeedForward`（[model_minimind.py:148-237](../model/model_minimind.py#L148-L237)）——MoE 稀疏专家

这是文件的重点，注释已经非常详细。这里把核心流程用**具体数字**再走一遍。

**设定**：num_experts=3，num_experts_per_tok=1（top-1），4 个 token（N=4），hidden=4。

### 步骤 1-2：展平 + 门控打分（L172-176）

```
x [1,4,4] → x_flat [4,4]           # 每行一个 token
logits = x_flat @ W_gate^T          # [4,3]
scores = softmax(logits, dim=-1)    # [4,3]，每行和为 1
```

假设 scores 为：

```
scores = [ [0.6, 0.3, 0.1]   # t0
           [0.2, 0.1, 0.7]   # t1
           [0.1, 0.8, 0.1]   # t2
           [0.7, 0.2, 0.1] ] # t3
```

### 步骤 3：Top-1 选择（L181）

```
topk_weight, topk_idx = torch.topk(scores, k=1)
topk_idx = [[0], [2], [1], [0]]   # t0→专家0, t1→专家2, t2→专家1, t3→专家0
```

### 步骤 4：归一化（L183）

k=1 时 `weight / weight.sum = 1`，所以这里实际不起作用（注释里也说了）。

### 步骤 5：逐专家分桶计算（L187-207）

遍历 3 个专家，每个专家只算命中自己的 token：

```
专家0：mask = (topk_idx==0) = [[T],[F],[F],[T]] → 命中 token_idx=[0,3]
       y[0] += expert0(x_flat[0]) * w0
       y[3] += expert0(x_flat[3]) * w3

专家1：命中 token_idx=[2] → y[2] += expert1(x_flat[2])
专家2：命中 token_idx=[1] → y[1] += expert2(x_flat[1])
```

`index_add_(0, idx, src)` 语义是 `y[idx[j]] += src[j]`，即**按行号把结果累加回输出**。

关键点 **L204-207**：如果某个专家在本批次一个 token 都没命中（`mask.any()==False`），且处于训练态，就执行 `y[0,0] += 0 * sum(p.sum()...)`。这个 0 乘技巧把该专家的参数强行纳入 autograd 图，**防止「专家饿死」时梯度丢失**，数值上无影响。

### 步骤 6：负载均衡辅助损失（L229-235）

防止所有 token 都挤到少数专家。公式（Switch Transformer / DeepSeek 风格）：

```
aux_loss = num_experts × coef × Σ( load ⊙ scores.mean(0) )
```

用注释里的数字：

```
load = one_hot(topk_idx).mean(0) = [0.50, 0.25, 0.25]   # 各专家被选中的实际频率
scores.mean(0) = [0.4375, 0.324, 0.2385]                 # 门控想选各专家的平均概率
load ⊙ scores.mean(0) = [0.21875, 0.081, 0.05963]
Σ = 0.35938
× num_experts(3) = 1.0781
× coef(5e-4) = 0.000539                                  # 最终 aux_loss
```

`load` 是「实际负载」，`scores.mean(0)` 是「期望偏好」，两者点乘：既想让负载更均匀，又不违背门控本身的选择倾向。

最后 L237 把 `[N, D]` 还原成 `[B, S, D]` 输出。

---

## 8. `MiniMindBlock`（[model_minimind.py:239-255](../model/model_minimind.py#L239-L255)）——单层 Transformer

一个标准 decoder 层，两个残差连接：

```
residual = hidden_states
# 1. 注意力子层（pre-norm）
hidden_states, present_kv = self_attn(input_layernorm(hidden_states), ...)
hidden_states += residual                    # 残差连接

# 2. FFN/MoE 子层（pre-norm）
hidden_states = hidden_states + mlp(post_attention_layernorm(hidden_states))
```

用的是 **pre-LN（pre-norm）** 结构：归一化放在子层**之前**，这是现代 LLM 的标准做法，训练更稳定。`use_moe` 决定 mlp 是普通 FFN 还是 MoE（L245）。

---

## 9. `MiniMindModel`（[model_minimind.py:257-293](../model/model_minimind.py#L257-L293)）——模型主干

- **embed_tokens**（L262）：`[vocab_size, hidden_size]` 词嵌入表。
- **layers**（L264）：N 个 `MiniMindBlock`。
- **RoPE 表预计算**（L266-268）：用 `register_buffer(..., persistent=False)` 存 cos/sin，不参与梯度、不存进 checkpoint。
- **meta-device 修复**（L277-279）：transformers ≥5.x 用 meta 设备初始化时 buffer 会丢，这里检测到 `freqs_cos[0,0]==0` 就重算一遍。

**forward 流程（L270-293）**：

```python
hidden_states = embed_tokens(input_ids)        # [B, S] → [B, S, 8]
start_pos = past_key_values[0][0].shape[1] if ... else 0   # KV cache 已生成的 token 数
position_embeddings = (cos[start_pos:...], sin[start_pos:...])  # 只取当前需要的 RoPE 位置段

# 逐层前向，收集每层的 (K, V) 用于 KV cache
for layer in layers:
    hidden_states, present = layer(hidden_states, position_embeddings, ...)

hidden_states = norm(hidden_states)            # 最后归一化
aux_loss = Σ 各 MoE 层的 aux_loss               # L292 汇总所有 MoE 层的负载均衡损失
```

`start_pos`（L274）是 KV cache 的关键：增量生成时，第 2 次只输入 1 个新 token，但 RoPE 需要知道它处于序列的第几个位置，就从 cache 长度算出来。

---

## 10. `MiniMindForCausalLM`（[model_minimind.py:295-349](../model/model_minimind.py#L295-L349)）——带 LM head 的因果模型

### 10.1 初始化与权重绑定（L298-304）

```
lm_head = Linear(hidden_size, vocab_size, bias=False)     # [8, 6400]
tie_word_embeddings=True 时：lm_head.weight 与 embed_tokens.weight 共享
```

共享权重能省大量参数（vocab_size 通常很大），且训练时输入/输出嵌入语义对齐。

### 10.2 forward + loss（L306-314）

```python
hidden_states → lm_head → logits   # [B, S, 6400]

# 交叉熵：下一 token 预测
x = logits[..., :-1, :]      # 用前 S-1 个位置预测
y = labels[..., 1:]          # 目标是后 S-1 个位置（整体右移一格）
loss = cross_entropy(x.reshape(-1, 6400), y.reshape(-1), ignore_index=-100)
```

**矩阵示例**（S=3，序列「我 爱 你」）：

```
logits = [logits(我), logits(爱), logits(你)]
预测：logits(我)→爱,  logits(爱)→你
目标：labels = [爱, 你]        # 左移，与预测对齐
```

`ignore_index=-100` 跳过 padding token 的 loss。`logits_to_keep` 参数（L308）用于投机解码，只保留最后几个位置的 logits 省显存。

### 10.3 `generate` 采样（L317-349）

手写的一个自回归采样循环，`@torch.inference_mode()` 关闭梯度加速。每步流程：

```python
# 1. 只输入最后一个新 token，配合 KV cache (L326)
outputs = self.forward(input_ids[:, past_len:], ...)

# 2. 温度缩放 (L328)
logits = logits / temperature        # 越高越随机

# 3. 重复惩罚 (L329-331)：对已出现的 token，正分÷penalty、负分×penalty，抑制重复

# 4. top-k 过滤 (L332-333)：只保留概率最高的 k 个，其余置 -inf

# 5. top-p（nucleus）过滤 (L334-338)：累计概率和 ≤ p 的 token 保留

# 6. 采样 (L339)：do_sample 用 multinomial 按概率抽，否则 argmax 贪心

# 7. 拼 token、更新 KV cache、检测 eos 提前终止 (L341-346)
```

几个细节：
- `logits = outputs.logits[:, -1, :]`（L328）：只取最后一个位置的 logits，因为前面已经生成完了，只需预测下一个 token。
- `mask.scatter(1, sorted_indices, mask)`（L338）：把「累积概率超过 top_p 的 token」标记回原索引位置，再置 -inf。
- `finished |= next_token.eq(eos_token_id)`（L345）：batch 内所有序列都生成 eos 就提前 break。

---

## 总结：一张数据流全景图

```
input_ids [B, S]
   │  embed_tokens  → [B, S, D]
   │  dropout
   ▼
┌─ × num_hidden_layers ─────────────────────────────┐
│ MiniMindBlock:                                    │
│   x ──RMSNorm──► Attention(QKV+RoPE+GQA) ──+──► x │
│                                            │       │
│   x ──RMSNorm──► FFN/MoE(SwiGLU) ──────────+──► x │
└────────────────────────────────────────────────────┘
   ▼
RMSNorm → [B, S, D]
   ▼
lm_head (与 embedding 共享权重) → logits [B, S, vocab]
   ▼
cross_entropy(左移对齐) → loss
```

这个实现的精髓在于：**GQA（KV 头复制省显存）、RoPE（相对位置外推）、SwiGLU（门控 FFN）、MoE 稀疏激活（总参数大但激活参数小）+ 负载均衡损失、以及手写的 KV cache + top-k/top-p 采样**，麻雀虽小五脏俱全，是一个很好的 LLM 结构学习范本。

---

## 附：相关文档

- [MOEFeedForward详解.md](./MOEFeedForward详解.md) —— MoE 稀疏专家层单独详解
