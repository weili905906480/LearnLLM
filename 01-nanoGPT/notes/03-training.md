# train.py 解析

> 文件：nanoGPT/train.py
> 作用：模型训练的完整流程

## 训练流程概览

```
1. 初始化配置
2. 设置 DDP（分布式，如果多卡）
3. 加载数据
4. 初始化模型（或从 checkpoint 恢复）
5. 设置优化器（AdamW）
6. 训练循环：
   for iter in range(max_iters):
       - 获取 batch
       - forward pass → loss
       - backward pass → gradients
       - gradient clipping
       - optimizer step
       - learning rate schedule
       - 定期 eval & save checkpoint
```

## 关键组件笔记

### 1. 数据加载

**方式：** 直接 mmap 读取 bin 文件，随机取 batch

- [ ] 数据预处理流程（prepare.py）
- [ ] token 编码方式
- [ ] batch 的构造（input = tokens[i:i+block_size], target = tokens[i+1:i+block_size+1]）

---

### 2. 优化器配置

- [ ] AdamW（带 weight decay）
- [ ] 哪些参数需要 decay（权重矩阵），哪些不需要（bias, LayerNorm）
- [ ] beta1, beta2 的选择

---

### 3. Learning Rate Schedule

```
lr
 ↑
 |   /‾‾‾‾‾‾‾‾‾‾‾‾‾‾\
 |  /                  \  cosine decay
 | / warmup             \
 |/                      \_____ min_lr
 +──────────────────────────────→ iter
```

- [ ] warmup 阶段的作用
- [ ] cosine decay 的公式
- [ ] min_lr 的设置

---

### 4. 梯度累积 (Gradient Accumulation)

- [ ] 为什么需要（GPU 内存不够大 batch）
- [ ] 实现方式（多次 forward/backward，最后才 step）
- [ ] `gradient_accumulation_steps` 参数

---

### 5. 混合精度训练

- [ ] float16 / bfloat16 的区别
- [ ] `torch.cuda.amp` 的使用
- [ ] GradScaler 的作用（仅 fp16 需要）

---

### 6. 分布式训练 (DDP)

- [ ] DataDistributedParallel 基本原理
- [ ] 多卡时梯度同步方式
- [ ] `torchrun` 启动方式

---

## 关键超参数

| 参数 | shakespeare_char | GPT-2 复现 |
|------|-----------------|------------|
| batch_size | 64 | 12 |
| block_size | 256 | 1024 |
| max_iters | 5000 | 600000 |
| learning_rate | 1e-3 | 6e-4 |
| warmup_iters | 100 | 2000 |

## 实验记录

| 实验 | 配置 | 最终 loss | 备注 |
|------|------|-----------|------|
| — | — | — | — |

## 我的理解/疑问

<!-- 在这里记录你的理解和疑问 -->

