# nanoGPT train.py 逐段解析

> 源码：https://github.com/karpathy/nanoGPT/blob/master/train.py
> 约 300 行，实现完整的 GPT 训练流程

---

## 整体结构

```
train.py
├── 1. 配置系统       超参数定义，支持命令行覆盖
├── 2. DDP 初始化     多卡分布式训练设置
├── 3. 数据加载       np.memmap 高效读取 bin 文件
├── 4. 模型初始化     scratch / resume / gpt2 三种模式
├── 5. 优化器配置     AdamW + weight decay 分组
├── 6. 学习率调度     warmup + cosine decay
├── 7. 评估函数       定期 eval，记录 train/val loss
└── 8. 训练循环       梯度累积 + 混合精度 + 参数更新
```

---

## 1. 配置系统

```python
# 数据
dataset = 'openwebtext'
gradient_accumulation_steps = 5 * 8  # 模拟大 batch
batch_size = 12
block_size = 1024                     # 上下文窗口长度

# 模型
n_layer = 12
n_head  = 12
n_embd  = 768
dropout = 0.0
bias    = False

# 优化器
learning_rate = 6e-4
max_iters     = 600000
weight_decay  = 1e-1
beta1, beta2  = 0.9, 0.95
grad_clip     = 1.0

# 学习率衰减
warmup_iters    = 2000
lr_decay_iters  = 600000
min_lr          = 6e-5   # = learning_rate / 10

# 系统
device = 'cuda'
dtype  = 'bfloat16'
compile = True
```

**关键设计：**
- `gradient_accumulation_steps = 40`：有效 batch = 12 × 1024 × 40 ≈ 0.5M tokens/step
- `vocab_size = 50304`（非 50257）：补到 64 的倍数，GPU 矩阵运算更高效
- `init_from`：支持 `scratch` / `resume` / `gpt2` 三种初始化

---

## 2. DDP 分布式训练

```python
ddp = int(os.environ.get('RANK', -1)) != -1
if ddp:
    init_process_group(backend='nccl')
    ddp_rank       = int(os.environ['RANK'])
    ddp_local_rank = int(os.environ['LOCAL_RANK'])
    ddp_world_size = int(os.environ['WORLD_SIZE'])
    device = f'cuda:{ddp_local_rank}'
    master_process = ddp_rank == 0
    # 多卡均分梯度累积步数
    gradient_accumulation_steps //= ddp_world_size
```

- 通过环境变量检测是否处于多卡模式
- `master_process`：只有主进程负责日志和保存 checkpoint
- 用 `torchrun` 启动时自动设置 RANK / LOCAL_RANK / WORLD_SIZE

---

## 3. 数据加载

```python
def get_batch(split):
    data = np.memmap(
        os.path.join(data_dir, f'{split}.bin'),
        dtype=np.uint16, mode='r'
    )
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([torch.from_numpy(data[i  :i+block_size  ].astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy(data[i+1:i+1+block_size].astype(np.int64)) for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y
```

**核心设计：**
- `np.memmap`：17GB 数据不加载到内存，按需读取
- next-token prediction 的数据格式：
  ```
  x: [t0, t1, t2, ..., t1023]   ← 输入
  y: [t1, t2, t3, ..., t1024]   ← 目标（右移一位）
  ```

---

## 4. 模型初始化

```python
if init_from == 'scratch':
    model = GPT(GPTConfig(**model_args))

elif init_from == 'resume':
    ckpt = torch.load(os.path.join(out_dir, 'ckpt.pt'))
    model = GPT(GPTConfig(**ckpt['model_args']))
    model.load_state_dict(ckpt['model'])
    iter_num = ckpt['iter_num']

elif init_from.startswith('gpt2'):
    model = GPT.from_pretrained(init_from)  # 加载 HF 权重
```

---

## 5. 优化器：Weight Decay 分组

```python
def configure_optimizers(model, weight_decay, learning_rate, betas):
    decay_params   = [p for n, p in model.named_parameters() if p.dim() >= 2]
    nodecay_params = [p for n, p in model.named_parameters() if p.dim() <  2]

    optim_groups = [
        {'params': decay_params,   'weight_decay': weight_decay},
        {'params': nodecay_params, 'weight_decay': 0.0},
    ]
    optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas)
    return optimizer
```

**规则：**
- 2D 参数（权重矩阵）→ 加 weight decay（L2 正则）
- 1D 参数（bias、LayerNorm）→ 不加 decay

---

## 6. 学习率调度

```python
def get_lr(it):
    if it < warmup_iters:                    # Warmup：线性增长
        return learning_rate * it / warmup_iters
    if it > lr_decay_iters:                  # 衰减结束：保持 min_lr
        return min_lr
    # 中间：余弦衰减
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (learning_rate - min_lr)
```

```
lr ↑
   |   /‾‾‾‾‾‾‾‾‾‾‾‾\
   |  /    cosine     \
   | / warmup          \
   |/                   \_____ min_lr
   +──────────────────────────→ iter
   0  2000             600000
```

---

## 7. 评估函数

```python
@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)       # 采样 200 次取平均
        for k in range(eval_iters):
            X, Y = get_batch(split)
            with ctx:
                logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out
```

---

## 8. 核心训练循环

```python
X, Y = get_batch('train')   # 预取第一个 batch

for iter_num in range(max_iters):

    # 1. 更新学习率
    lr = get_lr(iter_num)
    for pg in optimizer.param_groups:
        pg['lr'] = lr

    # 2. 定期评估 & 保存 checkpoint
    if iter_num % eval_interval == 0:
        losses = estimate_loss()
        if losses['val'] < best_val_loss:
            best_val_loss = losses['val']
            torch.save({'model': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'iter_num': iter_num,
                        'best_val_loss': best_val_loss}, 'ckpt.pt')

    # 3. 梯度累积循环
    for micro_step in range(gradient_accumulation_steps):
        if ddp:
            # 只在最后一步做跨卡梯度同步
            model.require_backward_compat = (micro_step == gradient_accumulation_steps - 1)
        with ctx:                          # autocast bf16
            logits, loss = model(X, Y)
            loss = loss / gradient_accumulation_steps

        X, Y = get_batch('train')          # 预取下一个 batch（重叠 IO）
        scaler.scale(loss).backward()      # 累积梯度

    # 4. 梯度裁剪
    if grad_clip != 0.0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

    # 5. 参数更新
    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad(set_to_none=True)  # 比 zero_grad() 更省内存
```

### 梯度累积机制

```
batch_size=12, gradient_accumulation_steps=40
有效 batch = 12 × 40 = 480 样本

micro_step 0~39: forward → loss/40 → backward（梯度累加）
↓
clip_grad → optimizer.step() → zero_grad
```

### 混合精度

```python
ctx    = torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16)
scaler = torch.cuda.amp.GradScaler(enabled=(dtype == 'float16'))
# bf16 不需要 GradScaler，fp16 才需要防止梯度下溢
```

---

## 关键设计决策汇总

| 设计 | 原因 |
|------|------|
| `np.memmap` | 17GB 数据不加载到内存 |
| 梯度累积 | 用小 batch 模拟大 batch，节省显存 |
| Cosine LR + Warmup | 前期稳定，后期充分收敛 |
| Weight Decay 只对 2D | Bias/LayerNorm 参数少，加 decay 有害 |
| `vocab_size` 取 64 倍数 | GPU 矩阵乘法对齐优化 |
| DDP 延迟梯度同步 | 减少梯度累积中间的通信开销 |
| `grad_clip = 1.0` | 防止梯度爆炸导致训练崩溃 |
| `zero_grad(set_to_none=True)` | 比 `zero_grad()` 更省内存 |

---

## 我的理解 / 疑问

<!-- 在这里记录学习笔记 -->
