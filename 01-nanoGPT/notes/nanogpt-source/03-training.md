# nanoGPT `train.py` 逐行解析

> 源码：https://github.com/karpathy/nanoGPT/blob/master/train.py  
> `train.py` 约 300 行，但覆盖了从配置、数据读取、分布式训练到保存 checkpoint 的完整 GPT 训练流程。

---

## 整体结构

```text
train.py
├── 1. 配置系统       定义默认超参数，并允许通过命令行配置文件覆盖
├── 2. DDP 初始化     配置多 GPU 的进程、设备与梯度同步行为
├── 3. 数据加载       用 np.memmap 按需读取 .bin token 文件
├── 4. 模型初始化     从零训练、断点恢复或加载 GPT-2 预训练权重
├── 5. 优化器配置     AdamW，并按参数维度区分 weight decay
├── 6. 学习率调度     线性 warmup 后执行余弦衰减
├── 7. 评估函数       在 train / val 数据上抽样计算平均损失
└── 8. 训练循环       梯度累积、混合精度、裁剪梯度并更新参数
```

---

## 1. 配置系统

```python
# -------------------- 数据相关配置 --------------------
dataset = 'openwebtext'                  # 数据集名称；通常对应 data/openwebtext/ 目录。
gradient_accumulation_steps = 5 * 8      # 单次参数更新前累积 40 个 micro-batch 的梯度。
batch_size = 12                          # 每个 GPU、每个 micro-step 读取的序列数量 B。
block_size = 1024                        # 每条训练序列的 token 数量 T，即模型最大上下文窗口。

# -------------------- 模型结构配置 --------------------
n_layer = 12                             # Transformer Block 的层数；GPT-2 small 使用 12 层。
n_head = 12                              # 每层多头注意力的注意力头数量。
n_embd = 768                             # token 表示、残差流的隐藏维度 C。
dropout = 0.0                            # Dropout 概率；大规模预训练通常可设为 0。
bias = False                             # 是否在线性层和 LayerNorm 中保留 bias 参数。

# -------------------- 优化器配置 --------------------
learning_rate = 6e-4                     # AdamW 训练初期达到的最大学习率。
max_iters = 600000                       # 参数更新（optimizer.step）总次数上限，而不是样本数。
weight_decay = 1e-1                      # 对部分权重施加的 AdamW 解耦权重衰减系数。
beta1, beta2 = 0.9, 0.95                 # AdamW 一阶、二阶动量的指数滑动平均系数。
grad_clip = 1.0                          # 全局梯度范数的最大值；0.0 表示关闭裁剪。

# -------------------- 学习率衰减配置 --------------------
warmup_iters = 2000                      # 前 2,000 次更新把学习率从 0 线性升至峰值。
lr_decay_iters = 600000                  # 在该迭代步结束余弦衰减；通常与 max_iters 相同。
min_lr = 6e-5                            # 衰减后的学习率下限，等于峰值学习率的 1/10。

# -------------------- 运行环境配置 --------------------
device = 'cuda'                          # 单卡时模型和 batch 所在的计算设备。
dtype = 'bfloat16'                       # autocast 的低精度类型；bf16 的动态范围接近 fp32。
compile = True                           # 是否调用 torch.compile 编译模型以提升后续迭代吞吐。
```

**关键设计：**

- `gradient_accumulation_steps = 40` 时，单卡一次更新实际处理的 token 数为 `12 × 1024 × 40 = 491,520`；多卡时还需乘以 GPU 数量。
- `vocab_size = 50304`（而不是 GPT-2 原始的 `50257`）会补齐至 64 的倍数，常能让 GPU 上的矩阵乘法更高效。
- `init_from` 支持 `scratch`、`resume`、`gpt2` 三种初始化来源。

---

## 2. DDP 分布式训练

```python
ddp = int(os.environ.get('RANK', -1)) != -1  # 若 torchrun 注入了 RANK 环境变量，则启用 DDP。

if ddp:                                      # 仅在多进程 / 多 GPU 模式下进入此分支。
    init_process_group(backend='nccl')       # 初始化通信组；NCCL 是 CUDA GPU 间通信的高效后端。
    ddp_rank = int(os.environ['RANK'])       # 全局进程编号，取值为 0 到 world_size - 1。
    ddp_local_rank = int(os.environ['LOCAL_RANK'])  # 当前机器上的本地 GPU 编号。
    ddp_world_size = int(os.environ['WORLD_SIZE'])  # 全部训练进程（通常即 GPU）数量。
    device = f'cuda:{ddp_local_rank}'        # 每个进程绑定到自己负责的一张 GPU，避免设备冲突。
    torch.cuda.set_device(device)            # 将该 GPU 设为当前进程默认 CUDA 设备。
    master_process = ddp_rank == 0           # 仅 rank 0 打印日志、写 TensorBoard/W&B、保存 checkpoint。
    gradient_accumulation_steps //= ddp_world_size  # 平分每卡累积次数，保持全局有效 batch 大小不变。
else:                                        # 单 GPU 或 CPU 运行时不需要进程间通信。
    master_process = True                     # 唯一进程自然承担日志和保存职责。
    ddp_world_size = 1                        # 统一后续计算逻辑中的 world size。
```

- 用 `torchrun --standalone --nproc_per_node=8 train.py ...` 启动时，PyTorch 会自动设置 `RANK`、`LOCAL_RANK` 和 `WORLD_SIZE`。
- `RANK` 是**全局**排名；`LOCAL_RANK` 是本机 GPU 排名。在单机训练中两者经常相同，但多机训练时不同。
- 主进程以外的进程也会做前向和反向传播，只是不应重复输出日志或覆盖同一个 checkpoint 文件。

---

## 3. 数据加载

```python
def get_batch(split):                                               # split 为 'train' 或 'val'，返回一个随机训练 batch。
    data = np.memmap(                                                # 创建磁盘文件的内存映射对象，不会把整个文件读入 RAM。
        os.path.join(data_dir, f'{split}.bin'),                      # 拼出如 data/openwebtext/train.bin 的二进制 token 文件路径。
        dtype=np.uint16,                                             # 文件中每个 token 用无符号 16 位整数存储，节省磁盘空间。
        mode='r'                                                     # 以只读模式映射，训练过程不会修改原始数据文件。
    )
    ix = torch.randint(len(data) - block_size, (batch_size,))        # 随机选 B 个起点，确保每个起点后至少有 block_size+1 个 token。
    x = torch.stack([                                                # 将 B 条独立序列堆叠为形状 [B, T] 的输入张量。
        torch.from_numpy(                                            # 将 NumPy 切片零拷贝/低开销地包装为 PyTorch Tensor。
            data[i:i + block_size].astype(np.int64)                 # 取 [i, i+T) 的输入 token，并转为 Embedding 需要的 int64。
        )
        for i in ix                                                  # 对 batch 中每一个随机起点分别截取一段序列。
    ])
    y = torch.stack([                                                # 构造与 x 对齐的监督目标，最终形状同样为 [B, T]。
        torch.from_numpy(                                            # 将对应的 NumPy 切片转换为 PyTorch Tensor。
            data[i + 1:i + 1 + block_size].astype(np.int64)         # 比 x 整体右移一位；每个位置的目标是下一个 token。
        )
        for i in ix                                                  # 使用与 x 完全相同的起点，保证输入和标签一一对应。
    ])
    x = x.pin_memory().to(device, non_blocking=True)                 # 固定 CPU 内存并异步复制 x 到 GPU，提高主机到设备传输效率。
    y = y.pin_memory().to(device, non_blocking=True)                 # 以相同方式把标签复制到 GPU。
    return x, y                                                      # 返回模型前向传播所需的输入 token 与目标 token。
```

**next-token prediction 的数据格式：**

```text
x: [t0, t1, t2, ..., t1023]  ← 已知上下文，输入模型
y: [t1, t2, t3, ..., t1024]  ← 每个位置应预测的“下一个 token”
```

例如，模型看到 `t0` 时要预测 `t1`；看到 `t0, t1` 时要预测 `t2`。训练损失会对所有 `T` 个位置的预测同时计算交叉熵。

> 原版 nanoGPT 会依据设备类型选择是否使用 `pin_memory` 与 `non_blocking`，并在 CUDA 环境中启用它们。

---

## 4. 模型初始化

```python
if init_from == 'scratch':                                    # 模式一：不加载任何已有权重，从随机参数开始训练。
    model_args['vocab_size'] = meta_vocab_size or 50304       # 优先采用数据集 meta.pkl 的词表大小，否则使用对齐后的默认值。
    gptconf = GPTConfig(**model_args)                         # 将配置字典解包为 GPTConfig，供模型构造函数使用。
    model = GPT(gptconf)                                      # 创建一个参数随机初始化的 GPT 模型。

elif init_from == 'resume':                                   # 模式二：从 out_dir 中已有的 checkpoint 继续训练。
    ckpt_path = os.path.join(out_dir, 'ckpt.pt')               # 构造 checkpoint 文件路径。
    checkpoint = torch.load(ckpt_path, map_location=device)   # 读取模型、优化器及训练进度，并映射到当前设备。
    model_args = checkpoint['model_args']                      # 恢复训练时的模型结构，避免结构与权重不匹配。
    gptconf = GPTConfig(**model_args)                          # 用保存下来的配置重建完全相同的网络结构。
    model = GPT(gptconf)                                       # 先创建空模型，随后再把保存的权重写入。
    state_dict = checkpoint['model']                           # 取出保存的参数名到 Tensor 的映射表。
    model.load_state_dict(state_dict)                          # 将 checkpoint 权重加载到新创建的模型中。
    iter_num = checkpoint['iter_num']                          # 恢复已完成的参数更新次数，供日志和 LR 调度继续使用。
    best_val_loss = checkpoint['best_val_loss']                # 恢复历史最佳验证损失，避免错误覆盖最佳模型判断。

elif init_from.startswith('gpt2'):                             # 模式三：加载 gpt2、gpt2-medium 等公开预训练权重。
    model = GPT.from_pretrained(init_from, dict(dropout=dropout))  # 下载/转换 GPT-2 权重；可覆盖当前任务的 dropout。
    model_args = model.config.to_dict()                         # 把预训练模型实际采用的结构保存到 model_args。
```

- `scratch` 适合预训练新模型。
- `resume` 不只恢复模型参数，实际训练中也应恢复优化器状态、迭代数和最佳验证损失，才能真正“接着训练”。
- `gpt2` 适合在 GPT-2 权重基础上继续预训练或微调；其结构由预训练模型决定，而非当前手写的 `n_layer` 等默认值。

---

## 5. 优化器：Weight Decay 分组

```python
def configure_optimizers(model, weight_decay, learning_rate, betas, device_type):  # 根据参数类别创建 AdamW。
    param_dict = {pn: p for pn, p in model.named_parameters()}                      # 收集“参数名 -> 参数张量”的字典。
    param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}         # 过滤被冻结、不参与反向传播的参数。
    decay_params = [p for _, p in param_dict.items() if p.dim() >= 2]                # 2D 及以上参数通常是 Embedding/Linear 权重矩阵。
    nodecay_params = [p for _, p in param_dict.items() if p.dim() < 2]               # 1D/标量参数通常是 bias 或 LayerNorm 的缩放参数。

    optim_groups = [                                                                # 为两类参数创建不同的优化器参数组。
        {'params': decay_params, 'weight_decay': weight_decay},                     # 权重矩阵使用指定 decay，抑制过大的权重。
        {'params': nodecay_params, 'weight_decay': 0.0},                             # bias、LayerNorm 不衰减，这是 Transformer 常见做法。
    ]
    fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters    # 检查当前 PyTorch 是否支持 fused AdamW。
    use_fused = fused_available and device_type == 'cuda'                            # fused 实现只在 CUDA 上启用，通常更快。
    optimizer = torch.optim.AdamW(                                                   # 创建采用解耦 weight decay 的 AdamW 优化器。
        optim_groups,                                                                # 传入上面按是否衰减划分的参数组。
        lr=learning_rate,                                                            # 设置初始学习率；训练循环还会每步更新它。
        betas=betas,                                                                 # 设置 Adam 的一阶和二阶动量系数。
        fused=use_fused                                                              # 若环境支持则调用高性能融合 CUDA 内核。
    )
    return optimizer                                                                 # 将配置完成的优化器交给训练循环。
```

**规则：**

- 2D 及以上参数（词嵌入表、位置嵌入表、线性层权重）使用 weight decay。
- 1D 参数（bias、LayerNorm 的缩放参数）不使用 weight decay。
- 这是按**张量维度**而非按参数名称分组的简洁实现；在 nanoGPT 的模型结构中，该规则恰好符合常见训练实践。

---

## 6. 学习率调度

```python
def get_lr(it):                                                       # 根据当前参数更新步数 it 计算本步应使用的学习率。
    if it < warmup_iters:                                              # 阶段一：模型刚开始训练时进行线性 warmup。
        return learning_rate * it / warmup_iters                       # 从 0 平滑增长到 learning_rate，降低初始训练不稳定性。
    if it > lr_decay_iters:                                            # 阶段三：超过设定衰减终点后不再继续减小学习率。
        return min_lr                                                  # 返回预设下限，避免学习率无限趋近于 0。
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)  # 将当前衰减区间位置归一化到 [0, 1]。
    assert 0 <= decay_ratio <= 1                                       # 防御性检查：确认只在合法的衰减区间内计算余弦。
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))              # 余弦系数从 1 平滑下降到 0。
    return min_lr + coeff * (learning_rate - min_lr)                   # 将 [0, 1] 系数映射到 [min_lr, learning_rate]。
```

```text
lr ↑
   |   /‾‾‾‾‾‾‾‾‾‾‾‾\
   |  /    cosine     \
   | / warmup          \
   |/                   \_____ min_lr
   +──────────────────────────→ iter
   0  2000             600000
```

- warmup 避免刚初始化的模型在大步长更新下立即发散。
- 余弦曲线在起点和终点处导数平滑，通常比突变式降学习率更稳定。
- 原实现使用 `if it > lr_decay_iters`，因此 `it == lr_decay_iters` 会按余弦公式计算，结果正好也是 `min_lr`。

---

## 7. 评估函数

```python
@torch.no_grad()                                                    # 评估过程不构建反向传播图，减少显存占用并提升速度。
def estimate_loss():                                                # 分别估计训练集和验证集上的平均交叉熵损失。
    out = {}                                                        # 用字典保存每个 split 的最终平均损失。
    model.eval()                                                    # 切换到评估模式，关闭 Dropout 等训练期随机行为。
    for split in ['train', 'val']:                                  # 对训练集和验证集各做一次随机抽样评估。
        losses = torch.zeros(eval_iters)                            # 预先分配长度为 eval_iters 的 CPU 张量记录每次 loss。
        for k in range(eval_iters):                                 # 重复抽取多个随机 batch，降低单 batch 的偶然性。
            X, Y = get_batch(split)                                # 从指定数据集随机取得一批输入 token 和下一个 token 标签。
            with ctx:                                               # 在与训练相同的 autocast 精度设置下执行前向计算。
                logits, loss = model(X, Y)                          # 传入标签后，模型会同时返回预测 logits 与交叉熵 loss。
            losses[k] = loss.item()                                 # 取 Python 标量并存入记录；item 会与 GPU 同步。
        out[split] = losses.mean()                                  # 对所有评估 batch 的损失求均值，作为该 split 的估计值。
    model.train()                                                   # 恢复训练模式，以便后续循环正确启用训练期行为。
    return out                                                       # 返回如 {'train': tensor(...), 'val': tensor(...)} 的结果。
```

`estimate_loss()` 不会遍历完整验证集，而是随机抽取 `eval_iters` 个 batch 估计平均损失。这在大数据集上速度更快，但结果会存在少量采样波动。

---

## 8. 核心训练循环

```python
X, Y = get_batch('train')                                           # 循环开始前预取第一个训练 batch，保证第一次前向传播有数据可用。
t0 = time.time()                                                    # 记录当前迭代开始时间，用于计算训练吞吐和耗时。
local_iter_num = 0                                                  # 记录当前进程实际运行过的迭代次数，常用于 compile 的预热判断。
raw_model = model.module if ddp else model                          # DDP 包装时通过 .module 获取原始 GPT，方便保存或访问模型方法。

while True:                                                         # nanoGPT 使用无限循环，并在末尾按 iter_num 主动退出。
    lr = get_lr(iter_num) if decay_lr else learning_rate            # 根据当前步数取得调度后的学习率；也可以关闭调度而固定 LR。
    for param_group in optimizer.param_groups:                      # AdamW 可能有多个参数组（decay / no-decay）。
        param_group['lr'] = lr                                      # 将本轮相同学习率写入每一个参数组。

    if iter_num % eval_interval == 0 and master_process:            # 仅主进程每隔固定步数进行评估、日志记录和保存。
        losses = estimate_loss()                                    # 得到随机抽样估计的 train loss 与 val loss。
        print(f"step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")  # 输出监控指标。
        if losses['val'] < best_val_loss or always_save_checkpoint: # 验证损失变好（或强制保存）才写 checkpoint。
            best_val_loss = losses['val']                           # 更新当前已知的最佳验证损失。
            if iter_num > 0:                                        # 跳过初始评估的保存，避免未训练模型被当作正式 checkpoint。
                checkpoint = {                                     # 组装可恢复训练所需的全部状态。
                    'model': raw_model.state_dict(),                # 保存原始 GPT 的全部参数和 buffer，不保存 DDP 外层包装。
                    'optimizer': optimizer.state_dict(),            # 保存 AdamW 动量等内部状态，恢复训练时非常重要。
                    'model_args': model_args,                       # 保存网络结构配置，恢复时可先正确重建模型。
                    'iter_num': iter_num,                           # 保存当前训练进度，LR 调度从该位置继续。
                    'best_val_loss': best_val_loss,                 # 保存模型选择依据，防止恢复后遗失最佳记录。
                    'config': config,                               # 保存本次运行的完整配置，方便复现实验。
                }
                torch.save(checkpoint, os.path.join(out_dir, 'ckpt.pt'))  # 将 Python 状态字典序列化为 checkpoint 文件。

    if iter_num == 0 and eval_only:                                 # eval_only 模式只评估一次，不执行任何参数更新。
        break                                                       # 跳出训练循环，随后执行资源清理代码。

    for micro_step in range(gradient_accumulation_steps):           # 一个 optimizer step 内连续累积多个 micro-batch 的梯度。
        if ddp:                                                     # DDP 默认每次 backward 都会 all-reduce 梯度，累积时需要避免重复通信。
            model.require_backward_grad_sync = (                    # 仅最后一个 micro-step 同步各卡的累计梯度。
                micro_step == gradient_accumulation_steps - 1       # 最后一次 backward 前将同步开关置为 True。
            )
        with ctx:                                                    # 在 autocast 上下文中执行前向传播，部分算子以 bf16/fp16 运行。
            logits, loss = model(X, Y)                              # 模型根据 X 预测所有位置 token，并根据 Y 计算平均交叉熵。
            loss = loss / gradient_accumulation_steps               # 把每次 loss 除以累积次数，保证总梯度与大 batch 平均 loss 一致。
        X, Y = get_batch('train')                                   # 在反向传播期间准备下一 micro-batch，尽可能与 GPU 计算重叠。
        scaler.scale(loss).backward()                               # fp16 时按比例放大 loss 防下溢；bf16/FP32 时 scaler 等价于直接 backward。

    if grad_clip != 0.0:                                            # 配置非零时启用梯度裁剪，防止偶发梯度爆炸。
        scaler.unscale_(optimizer)                                  # 先把 fp16 放大的梯度还原，否则裁剪阈值没有实际意义。
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)  # 若全局 L2 范数超阈值，按比例缩小全部梯度。

    scaler.step(optimizer)                                          # 若 fp16 梯度有限且无溢出，则调用 optimizer.step() 更新参数。
    scaler.update()                                                 # 根据是否检测到 fp16 溢出，动态调整下一步的缩放因子。
    optimizer.zero_grad(set_to_none=True)                           # 清除旧梯度；设为 None 比写满 0 通常更省内存和时间。

    t1 = time.time()                                                # 记录本次参数更新完成的时间。
    dt = t1 - t0                                                    # 计算本轮训练循环耗时（秒）。
    t0 = t1                                                         # 将终点作为下一轮的起点。
    if iter_num % log_interval == 0 and master_process:             # 仅主进程按间隔打印训练损失和性能数据。
        lossf = loss.item() * gradient_accumulation_steps           # 还原为除法前的单个累计 batch 平均 loss，便于阅读。
        print(f"iter {iter_num}: loss {lossf:.4f}, time {dt * 1000:.2f}ms")  # 输出当前训练进度与单步耗时。

    iter_num += 1                                                   # 完成一次 optimizer 更新后，推进全局迭代计数。
    local_iter_num += 1                                             # 推进当前进程的本地迭代计数。
    if iter_num > max_iters:                                        # 达到最大训练步数后停止；原代码使用严格大于号。
        break                                                       # 退出 while 循环，训练结束。
```

### 梯度累积机制

```text
batch_size = 12，gradient_accumulation_steps = 40
单卡一次参数更新的有效样本数 = 12 × 40 = 480 条序列
单卡一次参数更新的有效 token 数 = 12 × 40 × 1024 = 491,520 个 token

micro_step 0~39：forward → loss / 40 → backward（梯度相加，不更新参数）
↓
unscale → clip_grad → optimizer.step → zero_grad
```

将每个 `micro_step` 的 loss 除以累积次数是关键。因为反向传播会将梯度相加；除法确保最终梯度等于“将这些 micro-batch 合成一个大 batch 后，**平均 loss** 的梯度”，而不是其 40 倍。

### DDP 延迟梯度同步

正常 DDP 下，每次 `backward()` 都会执行一次跨 GPU 的 all-reduce。若一轮需要累积 40 次梯度，这会产生 40 次通信。通过：

```python
model.require_backward_grad_sync = (micro_step == gradient_accumulation_steps - 1)  # 仅最终反向传播触发 all-reduce。
```

前 39 次仅在各自 GPU 本地累积梯度，最后一次再同步总梯度，通信次数从 40 次降为 1 次，同时得到相同的结果。

### 混合精度

```python
ctx = torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16)  # 自动让适合的 CUDA 算子使用 bf16，其他敏感算子保留较高精度。
scaler = torch.cuda.amp.GradScaler(enabled=(dtype == 'float16'))    # 仅 fp16 时启用动态 loss scaling；bf16 通常不需要。
```

- **bf16** 与 fp32 具有相近的指数范围，不容易发生梯度下溢，因此一般无需 `GradScaler`。
- **fp16** 的可表示数值范围较小，训练时可能导致很小的梯度变为 0；`GradScaler` 通过暂时放大 loss 来缓解这个问题。
- 即使 bf16 下 `GradScaler(enabled=False)`，也可以继续统一调用 `scale`、`step`、`update`，其行为会退化为普通训练流程。

---

## 关键设计决策汇总

| 设计 | 原因 |
|------|------|
| `np.memmap` | 大型 token 数据集按需读取，不必全部加载进内存。 |
| 梯度累积 | 用显存可承受的小 micro-batch 模拟更大的有效 batch。 |
| Warmup + Cosine LR | 初期减少不稳定，后期平滑降低步长以利于收敛。 |
| Weight Decay 仅作用于 2D+ 参数 | 不对 bias / LayerNorm 参数施加不合适的正则化。 |
| `vocab_size` 取 64 的倍数 | 改善 GPU 矩阵计算的维度对齐与吞吐。 |
| DDP 延迟梯度同步 | 避免梯度累积中重复 all-reduce，显著降低通信开销。 |
| `grad_clip = 1.0` | 限制全局梯度范数，减少梯度爆炸导致训练发散的概率。 |
| `zero_grad(set_to_none=True)` | 省去把每个梯度张量写零的操作，通常更省内存和时间。 |
| 保存 optimizer state | 断点恢复后保留 AdamW 动量和方差估计，训练轨迹更连续。 |

---

## 我的理解 / 疑问

<!-- 在这里记录学习笔记 -->
