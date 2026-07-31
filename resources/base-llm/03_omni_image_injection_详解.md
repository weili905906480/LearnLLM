# Seeker-Omni 图像特征注入与多模态训练详解

> 本文是 `03_simplified_omni.md` 的补充笔记，详细拆解图像特征如何注入 LLM、训练三阶段如何划分、以及多模态对齐阶段的具体梯度流向。

## 一、三阶段训练管线全景

```
Stage 1               Stage 2               Stage 3
纯文本预训练     →    纯文本 SFT       →    端到端多模态对齐
(pre-train)          (instruction tuning)    (e2e alignment)
    │                     │                      │
    │                     │                      │
  只有文本              只有文本            🔴 这里才有图片！
```

| 阶段 | 数据 | 加载方式 | 图片 |
|------|------|---------|------|
| Stage 1 预训练 | minimind pretrain_hq.jsonl | MemmapDataset 离线 .bin | ❌ 无 |
| Stage 2 SFT | minimind sft_mini_512.jsonl | MemmapDataset 离线 .bin | ❌ 无 |
| Stage 3 多模态对齐 | flickr8k train_imgonly.jsonl | RawSample 流式解析 + SigLIP 在线提取 | ✅ 实时加载 JPG |

Stage 3 必须用在线加载（而非离线 `.bin`）的原因是：该阶段可能解冻 SigLIP 的最后 N 层，同一张图片在每个 epoch 的特征会随权重更新而变化，离线存盘的特征会过时。

## 二、图像特征提取完整管线

```
原始图片 (.jpg)
  │
  ▼
PIL.Image.open() → 转 RGB 三通道
  │
  ▼
SigLIP Processor: Resize → 384×384 → Normalize
  │
  ▼
SiglipVisionModel (ViT):  图片 → [1, 729, 1152]
                            729 = (384/16)² + 1个CLS token
  │
  ▼
PerceiverResampler:  64个可学习 latent queries × 交叉注意力
                     729 tokens → [1, 64, 1152]
  │
  ▼
Linear Projection:  [1, 64, 1152] × W[768, 1152] → [1, 64, 768]
  │
  ▼
Gate Modulation:  × tanh(img_gate[768])
  │
  ▼
注入到 LLM 词嵌入序列中 <img> 占位符位置
```

## 三、注入过程的矩阵变化（具体数值示例）

### 假设参数

| 参数 | 值 |
|------|-----|
| 序列长度 seq_len | 20 |
| hidden_size (LLM 维度) | 768 |
| 图像 token 数 | 4（简化，实际为 64） |
| 图像特征维度 | 1152 |
| 特殊 token 数 | 6 |
| 词表大小 vocab_size | 6400 |

### 步骤 1：文本嵌入

```python
# lm.py: _embed_tokens()
is_special = input_ids < 6   # 小于6的是特殊token

# 特殊 token → special_embed (只有6行)
special_embed.shape: [6, 768]

# 普通 token → base_embed (ID要减去偏移6)
base_embed.shape: [6394, 768]
```

关键初始化：`<img_bos>`、`<img>`、`<img_eos>` 的词嵌入被**刻意初始化为全零**：

```python
# lm.py: reset_parameters()
for tid in (self.special.img_bos, self.special.img, self.special.img_eos):
    self.special_embed.weight[tid].zero_()  # ← 强行置零
```

嵌入后 `x` 矩阵：

```
x: [1, 20, 768]

位置 0:  special_embed[1]    → [ 0.02, -0.01,  0.00, ...,  0.03]   <|im_start|>
位置 1:  base_embed[429]     → [-0.05,  0.01,  0.02, ..., -0.01]   "请"
位置 2:  base_embed[1614]    → [ 0.01, -0.03,  0.00, ...,  0.02]   "描述"
...
位置 7:  special_embed[4]    → [ 0.00,  0.00,  0.00, ...,  0.00]   <img_bos> ← 全零!
位置 8:  special_embed[3]    → [ 0.00,  0.00,  0.00, ...,  0.00]   <img>     ← 全零!
位置 9:  special_embed[3]    → [ 0.00,  0.00,  0.00, ...,  0.00]   <img>     ← 全零!
位置10:  special_embed[3]    → [ 0.00,  0.00,  0.00, ...,  0.00]   <img>     ← 全零!
位置11:  special_embed[3]    → [ 0.00,  0.00,  0.00, ...,  0.00]   <img>     ← 全零!
位置12:  special_embed[5]    → [ 0.00,  0.00,  0.00, ...,  0.00]   <img_eos> ← 全零!
...
位置19:  special_embed[2]    → [ 0.01,  0.02, -0.01, ...,  0.00]   <|im_end|>
```

### 步骤 2：投影（维度对齐）

```python
# img_proj: Linear(1152 → 768, bias=False), weight: [768, 1152]
img_tokens = img_proj(image_feats)  # [1, 4, 768]
```

### 步骤 3：门控调制（核心保护机制）

```python
# img_gate: [768], 初始化为全零
img_tokens = img_tokens * torch.tanh(img_gate)
# tanh([0, 0, ..., 0]) = [0, 0, ..., 0]
# → 训练第0步: img_tokens = 全零，模型等效纯文本
```

| 训练步数 | gate 值示例 | tanh(gate) 近似值 | 视觉贡献 |
|---------|------------|-------------------|---------|
| 第 0 步 | [0, 0, ..., 0] | [0, 0, ..., 0] | 零 |
| 第 500 步 | [0.02, -0.01, ..., 0.03] | [0.020, -0.010, ..., 0.030] | 微弱 |
| 第 5000 步 | [0.15, -0.08, ..., 0.22] | [0.149, -0.079, ..., 0.217] | 显著 |

### 步骤 4：逐位置加法注入

```python
# projector.py: inject_feature_tokens()
img_mask = input_ids == 3   # 3 = <img> 的 token ID
pos = img_mask.nonzero()    # [[0, 8], [0, 9], [0, 10], [0, 11]]

# 🔴 是加法，不是替换！
x[pos[:,0], pos[:,1]] = x[pos[:,0], pos[:,1]] + flat_img_tokens
```

注入前后对比（以位置 8 为例）：

```
注入前: x[0, 8] = [ 0.00,  0.00,  0.00,  0.00, ...,  0.00]  ← <img>的全零嵌入
                 +      +      +      +             +
注入值:          [ 0.05, -0.03,  0.01,  0.12, ...,  0.08]  ← 投影门控后的视觉特征
                 =      =      =      =             =
注入后: x[0, 8] = [ 0.05, -0.03,  0.01,  0.12, ...,  0.08]  ← 视觉信息到位
```

### 步骤 5：进入 Transformer

```
x (图文混合) → Dropout → Block₀ → Block₁ → ... → Blockₙ₋₁ → RMSNorm → LM Head → Logits
     ↑
  每一层的 Self-Attention 中，所有位置的 token 互相做注意力
  视觉 token（位置 8-11）和文本 token（位置 0-7, 12-19）充分交互
```

## 四、注入位置：最早层（Early Fusion）

```
embedding 输出 → [+注入视觉特征] → Dropout → Block 0 → Block 1 → ... → Block N-1
                      ↑
              🔴 就在这一层，所有 Transformer Block 之前
```

这是**早融合（Early Fusion）**设计：视觉特征从第一层 Transformer 开始就参与全部注意力计算，每一层都能做图文交互。代价是训练初期不稳定，需要门控机制 + 冻结策略来保护。

## 五、Labels 的传入机制

图像特征注入和训练标签走的是两条独立的路径：

```
输入流（数据）:
  DataLoader → batch = {
      "input_ids":     [1, 20],       ← 给模型看的原始 token
      "labels":        [1, 20],       ← 用来算 loss 的目标
      "attention_mask":[1, 20],
      "image_feats":   [1, 4, 1152]   ← 图像特征（独立传入）
  }

模型内部:
  forward(input_ids, labels, image_feats):
      x = embed(input_ids)           ← token → 向量
      x = inject(x, image_feats)     ← 图像特征注入到 x
      logits = transformer(x)        ← 前向计算
      loss = CrossEntropy(logits, labels, ignore_index=-100)  ← 只对 assistant 内容算 loss
```

### Labels 的关键设计：`-100` 掩码

```python
# sft_builder.py
for m in conversations:
    role = m["role"]
    msg_labels = [-100] * len(msg_tokens)    # 默认全部忽略

    if role == "assistant":
        # 🔴 只有助手的回答内容才恢复为真实 token ID
        for j in range(len(header), len(msg_tokens)):
            msg_labels[j] = msg_tokens[j]
```

最终 labels 序列：

```
序列内容:    <im_start> user "请" "描述" ... <img_bos> <img>×4 <img_eos> <im_start> assistant "在" "草" "地" "上" ...
labels:      -100      -100  -100  -100     -100      -100     -100      -100       -100      78  1623 1973  32
                                                                  ↑          ↑         ↑    ↑    ↑    ↑
                                                              角色标识     角色标识     ✅    ✅   ✅   ✅
                                                                                    只在这里算 loss！
```

图像占位符虽然 label = -100（不直接产生 loss），但图像特征通过 Self-Attention 影响了对 assistant 回答内容的预测，梯度会从回答位置的 loss 回传到图像特征对应的位置 → 更新 `img_proj`。

## 六、多模态对齐阶段的参数更新明细

### 6.1 最保守模式（freeze_backbone=true, unfreeze_last_n=0）

只训练 2 个参数：

| 参数 | 形状 (d=768) | 参数量 | 作用 |
|------|-------------|--------|------|
| `img_proj.weight` | [768, 1152] | ~884K | 视觉特征 → LLM 维度 |
| `img_gate` | [768] | 768 | 逐维度门控系数 |

**可训练参数仅约 885K，不到模型总参数的 1%。**

### 6.2 部分解冻模式（freeze_backbone=true, unfreeze_last_n=2）

每个解冻的 Transformer Block 包含：

| 子组件 | 矩阵 | 形状 (d=768, ff=2048) | 参数量 |
|--------|------|----------------------|--------|
| RMSNorm ×2 | weight | [768] ×2 | 1,536 |
| Q 投影 | weight | [768, 768] | ~590K |
| K 投影 | weight | [256, 768]* | ~39K |
| V 投影 | weight | [256, 768]* | ~39K |
| O 投影 | weight | [768, 768] | ~590K |
| Gate 上采样 | weight | [2048, 768] | ~1.57M |
| Up 上采样 | weight | [2048, 768] | ~1.57M |
| Down 下采样 | weight | [768, 2048] | ~1.57M |
| **单层合计** | | | **~4.97M** |

\* 假设 GQA: num_kv_heads=4, head_dim=64

加上 `final_norm`（768 参数），解冻 2 层的总增加约 10M 参数。

### 6.3 SigLIP 侧的梯度（vision_train_last_n > 0）

SigLIP ViT 每层包含（以 siglip-base-patch16-384 为例）：

```
每个 ViT Block:
  LayerNorm ×2:   weight [1152], bias [1152]
  Self-Attention:
    q_proj:       weight [1152, 1152], bias [1152]
    k_proj:       weight [1152, 1152], bias [1152]
    v_proj:       weight [1152, 1152], bias [1152]
    o_proj:       weight [1152, 1152], bias [1152]
  MLP:
    fc1:          weight [4304, 1152], bias [4304]
    fc2:          weight [1152, 4304], bias [1152]

Post-LayerNorm:   weight [1152], bias [1152]
```

| 配置 | 解冻的 SigLIP 参数量 |
|------|-------------------|
| vision_train_last_n=0 | 0（全冻结） |
| vision_train_last_n=2 | ~14M（最后 2 层 + post_layernorm） |
| vision_train_last_n=4 | ~28M（最后 4 层 + post_layernorm） |

## 七、梯度回传完整路径

以 `unfreeze_last_n=2, vision_train_last_n=2` 为例：

```
前向传播:
  图片 → SigLIP[❄️ 前10层] → SigLIP[🔥 后2层+post_ln] → 视觉特征 [1,64,1152]
                                                              │
  文本 → embed[❄️] → x → [+inject: img_proj[🔥] × tanh(img_gate[🔥])] → x'
                        │
                        ├──→ Block[0..N-3] [❄️] → 不变
                        ├──→ Block[N-2]    [🔥] → 权重被更新
                        ├──→ Block[N-1]    [🔥] → 权重被更新
                        └──→ final_norm    [🔥] → logits → loss

反向传播:
  loss
    │
    ├──→ final_norm.weight              ✅ 更新
    │
    ├──→ Block[N-1] 全部矩阵             ✅ 更新
    │     ├── w_down, w_gate, w_up
    │     ├── q_proj, k_proj, v_proj, o_proj
    │     └── attn_norm, mlp_norm
    │
    ├──→ Block[N-2] 全部矩阵             ✅ 更新
    │
    ├──→ Block[0..N-3]                  🚫 梯度截断（冻结）
    │
    ├──→ img_proj.weight                ✅ 更新
    ├──→ img_gate                       ✅ 更新
    │
    └──→ SigLIP 解冻层                   ✅ 更新
          ├── layer[10].attention.*
          ├── layer[10].mlp.*
          ├── layer[11].attention.*
          ├── layer[11].mlp.*
          └── post_layernorm.*
```

### 蒸馏梯度与语言梯度的汇合

```python
loss = loss_lm + α × loss_distill
#                    ↑
#     mse_distill(student_hidden, teacher_hidden)
```

```
loss_lm ──────────────────────────────┐
                                      ├──→ SigLIP 解冻层 ← 两股力量拉扯
loss_distill ────→ SigLIP 解冻层 ─────┘
                         │
            ┌────────────┼────────────┐
            ▼            ▼            ▼
      "看懂图片"    "保留视觉能力"   最终更新方向
      (来自语言)   (来自蒸馏 MSE)   = 两者折中
```

| 梯度来源 | 方向 | 不加约束的后果 |
|----------|------|--------------|
| loss_lm → SigLIP | 调整特征让 LLM 更好预测下一个词 | 视觉特征被语言目标扭曲，丧失识别能力 |
| loss_distill → SigLIP | 保持和冻结教师模型一致 | 完全不动，等于白解冻 |

## 八、参数更新总览表

| 模块 | 子矩阵 | 形状示例 | 最保守模式 | 部分解冻(unfreeze_last_n=2) |
|------|--------|---------|-----------|---------------------------|
| special_embed | weight | [6, d] | ❄️ 冻结 | ❄️ 冻结 |
| base_embed | weight | [V-6, d] | ❄️ 冻结 | ❄️ 冻结 |
| **img_proj** | weight | [d, 1152] | 🔥 训练 | 🔥 训练 |
| **img_gate** | param | [d] | 🔥 训练 | 🔥 训练 |
| blocks[0:N-2] | q/k/v/o/gate/up/down + 2×RMSNorm | ~5M/层 | ❄️ 冻结 | ❄️ 冻结 |
| blocks[N-2] | 同上 | ~5M | ❄️ 冻结 | 🔥 训练 |
| blocks[N-1] | 同上 | ~5M | ❄️ 冻结 | 🔥 训练 |
| final_norm | weight | [d] | ❄️ 冻结 | 🔥 训练 |
| SigLIP 冻结部分 | attention + mlp + ln | ~80M | ❄️ 冻结 | ❄️ 冻结 |
| SigLIP 最后 N 层 | attention + mlp + ln | ~7M/层 | ❄️ 冻结 | 🔥 训练 |
| **最小可训练参数** | | **~885K** | | |

## 九、为什么离线 bin 和在线加载同时存在

```
离线 memmap (.bin 文件)                  在线流式（实时加载）
─────────────────────────               ────────────────────
图像 → SigLIP → 压缩 → 存盘              图像 → SigLIP → 压缩 → 直接用
         ↓                                       ↓
    SigLIP 完全冻结                           SigLIP 最后 N 层解冻
    特征永不改变                              权重在训练中变化
         ↓                                       ↓
    存一次，反复读                              每轮 epoch 重跑

适用: Stage 1/2 (纯文本，无图片)           适用: Stage 3 (多模态对齐)
      Stage 3 可选变体（若 freeze_vision=true）
```

两种方式不是矛盾的——`MemmapWriter` 预留了 `image_feats.bin` 的写入能力，是为了一种可选变体：如果 Stage 3 也完全冻结 SigLIP，就可以先把特征全部抽好存成 `.bin`，训练时直接 memmap 读取，省 GPU 算力。但一旦决定解冻 SigLIP 做真正的端到端联合训练，就必须切换为在线加载。

## 十、训练所需硬件建议

| 模式 | 最低显存 | 推荐配置 |
|------|---------|---------|
| 最保守（只训 projector, ~885K 参数） | 4-6 GB | GTX 1060 6GB / RTX 2060 |
| 部分解冻（+2 层 LLM, ~10M 参数） | 6-8 GB | RTX 2060 6GB / RTX 3060 12GB |
| 完整训练（全解冻） | 16+ GB | RTX 3090 / RTX 4090 |

GT 730（2GB 显存，Compute Capability 3.5）无法运行任何模式——既不能训练也不能推理。建议使用 Google Colab（免费 T4 16GB）或 AutoDL（RTX 3090 ~1-2 元/小时）作为替代方案。
