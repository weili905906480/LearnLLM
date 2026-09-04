# train_pretrain.py 详细解析

> 对应源码：`trainer/train_pretrain.py`
> 关联文件：`trainer/trainer_utils.py`、`dataset/lm_dataset.py`、`model/model_minimind.py`

---

## 一、这个脚本是干什么的

`train_pretrain.py` 是 MiniMind 的**预训练（Pretrain）**入口：在一堆纯文本上做 **next-token prediction（下一个 token 预测）**，让模型学会语言的统计规律，得到一份 base 权重（后续再拿去 SFT/RL）。

一句话流程：

> 读 jsonl 纯文本 → tokenize 成 id → 组成 batch → 前向算 loss → 反向累积梯度 → 每 N 步更新一次参数 → 定期打印/保存 → 训练完得到 `pretrain_768.pth`

它不涉及「用户提问/助手回答」的角色划分，也没有 loss mask（除了 padding），这是它和 `SFTDataset` 的本质区别。

---

## 二、文件头与导入（1–21 行）

```python
__package__ = "trainer"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
```

因为脚本是直接 `python train_pretrain.py` 跑的（而不是 `python -m trainer.train_pretrain`），`__package__` 是空，需要手动把项目根目录塞进 `sys.path`，才能 `from model.model_minimind import ...`、`from dataset.lm_dataset import ...` 成功。

```python
import datasets  # noqa: F401  # Windows pyarrow/torch DLL conflict workaround
```

这是 Windows 上的一个 hack：先 import 一次 `datasets`（间接 import pyarrow），规避 DLL 加载冲突。**纯兼容性处理，不影响逻辑。**

导入的关键模块：
- `MiniMindConfig`：模型超参配置
- `PretrainDataset`：预训练数据集
- `trainer_utils` 里的一堆工具函数（`get_lr`、`Logger`、`lm_checkpoint`、`init_distributed_mode`、`setup_seed`、`init_model`、`SkipBatchSampler`）
- `torch.cuda.amp` 相关：混合精度

---

## 三、核心函数 `train_epoch`（24–80 行）

这是整个训练循环。分块讲。

### 3.1 签名与循环入口

```python
def train_epoch(epoch, loader, iters, start_step=0, wandb=None):
    for step, (input_ids, labels) in enumerate(loader, start=start_step + 1):
```

- `epoch`：第几个 epoch（从 0 开始，用于日志）
- `loader`：`DataLoader`，每次 `yield` 一个 batch `(input_ids, labels)`
- `iters`：本 epoch 的总 step 数（用于算进度、算 lr、判断「是否最后一步」）
- `start_step`：断点续训时，从第 `start_step` 之后开始（`enumerate(..., start=start_step+1)`）

> 注意 `input_ids` 和 `labels` 的形状都是 `[batch_size, max_seq_len]`，默认 `[32, 340]`。batch 里的每个样本已经由 `PretrainDataset` 补齐到等长。

### 3.2 移动数据到设备 + 设置当前 step 的学习率

```python
input_ids = input_ids.to(args.device)
labels = labels.to(args.device)
lr = get_lr(epoch * iters + step, args.epochs * iters, args.learning_rate)
for param_group in optimizer.param_groups:
    param_group['lr'] = lr
```

`get_lr` 是**余弦退火**（带 0.1 倍下限），见 `trainer_utils.py:40`：

```python
def get_lr(current_step, total_steps, lr):
    return lr*(0.1 + 0.45*(1 + math.cos(math.pi * current_step / total_steps)))
```

**具体数字举例**（`learning_rate=5e-4`，假设总 step 数 `total_steps = epochs * iters = 2 * 1000 = 2000`）：

| 时刻 | `cos(π·step/total)` | 公式值 | 实际 lr |
|------|------|------|------|
| step=0（开始） | cos(0)=1 | 0.1+0.45×2=1.0 | **5e-4**（满学习率） |
| step=1000（一半） | cos(π/2)=0 | 0.1+0.45×1=0.55 | 2.75e-4 |
| step=2000（结束） | cos(π)=−1 | 0.1+0.45×0=0.1 | **5e-5**（降到 1/10） |

要点：
- **没有 warmup**，一上来就是满学习率，然后平滑余弦衰减到底。
- **下限是 0.1×lr**，不是衰减到 0。
- lr 是**每个 step 都动态算的**，按「全局 step = epoch×iters + step」来定位进度，跨 epoch 连续。

### 3.3 混合精度前向 + 梯度累积反向（35–40 行）

```python
with autocast_ctx:
    res = model(input_ids, labels=labels)
    loss = res.loss + res.aux_loss
    loss = loss / args.accumulation_steps
scaler.scale(loss).backward()
```

- `autocast_ctx`：在 CUDA 上是一个 `torch.cuda.amp.autocast(dtype=bf16/fp16)` 上下文，让前向在低精度下跑（省显存、提速）；CPU 上则是 `nullcontext()`（什么都不做）。
- `res.loss`：**语言建模交叉熵损失**（CE loss），即预测下一个 token 的平均损失。model forward 内部已经做了「`x[:-1]` 预测 `y[1:]`」的错位，所以数据侧不用手动错位。
- `res.aux_loss`：**MoE 路由器的负载均衡辅助损失**。只有当 `use_moe=True` 时才有值（否则是 `None` 或 0）。它鼓励每个专家被均衡使用，防止「路由坍缩到单个专家」。
- **`loss = loss / accumulation_steps`**：这是梯度累积的关键。把一个小 batch 的 loss 先缩小 `accumulation_steps` 倍再反向，这样累积 N 步后梯度等价于「一个大 batch」的梯度。

**梯度累积的直觉例子**（`accumulation_steps=8`, `batch_size=32`）：

> 不做累积时，一次 `optimizer.step()` 需要 32 条样本的梯度。
> 现在 GPU 显存放不下 32×8=256 条，就每次只放 32 条，算一次梯度但不更新，重复 8 次，最后一次性更新——**等效于 batch_size=256**。

`scaler.scale(loss).backward()`：混合精度 GradScaler 会把 loss 放大后再反传（防止 fp16 下梯度下溢到 0）。

### 3.4 梯度累积到阈值才真正更新（42–49 行）

```python
if step % args.accumulation_steps == 0:
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad(set_to_none=True)
```

只有当 step 是 8 的整数倍时才真正更新参数：

1. `scaler.unscale_(optimizer)`：把之前 `scale` 放大的梯度**还原**成真实尺度。
2. `clip_grad_norm_`：**梯度裁剪**，把所有参数的梯度范数限制在 `grad_clip=1.0` 内，防止梯度爆炸。
3. `scaler.step(optimizer)`：真正执行 `AdamW` 更新权重。
4. `scaler.update()`：动态调整 GradScaler 的 scale 因子。
5. `zero_grad(set_to_none=True)`：清空累积的梯度，开始下一轮。

**时间线举例**（step=1..8）：
- step 1–7：各做一次前向+反向，梯度在 `.grad` 里**累加**，不更新。
- step 8：前向+反向后，此时 `.grad` 已累积 8 个微 batch 的梯度 → 裁剪 → 更新 → 清空。
- step 9–16：重复……

### 3.5 日志打印（51–59 行）

```python
if step % args.log_interval == 0 or step == iters:
    spend_time = time.time() - start_time
    current_loss = loss.item() * args.accumulation_steps
    current_aux_loss = res.aux_loss.item() if res.aux_loss is not None else 0.0
    current_logits_loss = current_loss - current_aux_loss
    current_lr = optimizer.param_groups[-1]['lr']
    eta_min = spend_time / max(step - start_step, 1) * (iters - step) // 60
```

注意几个细节：
- `current_loss = loss.item() * accumulation_steps`：因为前面 loss 被除了 8，这里**乘回去**，让打印的是「未缩放的真实 loss」。
- `current_logits_loss`：CE 损失（语言建模主体）；`aux_loss` 单独拆出来，方便观察 MoE 辅助损失占比。
- `eta_min`：**预计剩余时间**（分钟）。公式 = 已经花的时间 / 已走 step 数 × 剩余 step 数，再 `//60` 转分钟。

打印一条，形如：

```
Epoch:[1/2](800/1000), loss: 4.2351, logits_loss: 4.2351, aux_loss: 0.0000, lr: 0.00044234, epoch_time: 3.2min
```

（非 MoE 时 `aux_loss=0.0000`，因为 `use_moe=0`。）

### 3.6 定期保存（61–71 行）

```python
if (step % args.save_interval == 0 or step == iters) and is_main_process():
    model.eval()
    moe_suffix = '_moe' if lm_config.use_moe else ''
    ckp = f'{args.save_dir}/{args.save_weight}_{lm_config.hidden_size}{moe_suffix}.pth'
    raw_model = model.module if isinstance(model, DistributedDataParallel) else model
    raw_model = getattr(raw_model, '_orig_mod', raw_model)
    state_dict = raw_model.state_dict()
    torch.save({k: v.half().cpu() for k, v in state_dict.items()}, ckp)
    lm_checkpoint(lm_config, weight=args.save_weight, model=model, optimizer=optimizer, scaler=scaler, epoch=epoch, step=step, wandb=wandb, save_dir='../checkpoints')
    model.train()
    del state_dict
```

要点：
- 只在**主进程**（rank 0）保存，避免多卡重复写文件。
- `model.eval()` → 存 → `model.train()`：保存时临时切到 eval 模式（关闭 dropout 等，其实预训练 dropout=0 影响不大），存完切回训练模式。
- `raw_model = model.module if DDP else model`：DDP 包裹后真实模型在 `.module` 里。
- `raw_model = getattr(raw_model, '_orig_mod', raw_model)`：如果用了 `torch.compile`，真实模型被包在 `_orig_mod` 里，这里再剥一层。
- **保存两份东西**：
  1. `../out/pretrain_768.pth`：纯模型权重（`half()` 转 fp16 省空间，`cpu()` 释放显存），供后续加载/推理用。
  2. `../checkpoints/pretrain_768_resume.pth`（由 `lm_checkpoint` 生成）：**完整训练状态**（模型+优化器+scaler+epoch+step+wandb_id），供断点续训用。

> 为什么 `lm_checkpoint` 里用 `os.replace` 写 `.tmp` 再改名？**原子写入**——防止写到一半进程崩溃留下损坏的 ckpt 文件。

### 3.7 尾部：处理不整除的残余梯度（75–80 行）

```python
if last_step > start_step and last_step % args.accumulation_steps != 0:
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad(set_to_none=True)
```

**场景**：一个 epoch 的 step 数（比如 1000）不是 8 的整数倍时，循环结束时还残留着不足 8 步的梯度没更新。这段代码在 epoch 结束时「把剩下的一点点梯度也更新掉」，不浪费。

> 等价于 PyTorch 官方示例里常见的「`if (idx+1) % accumulation_steps == 0: step()` + 结尾补一次」写法。

---

## 四、主流程 `__main__`（83–172 行）

### 阶段 0：命令行参数（84–107 行）

定义了所有可调参数，几个重要的默认值：

| 参数 | 默认 | 含义 |
|------|------|------|
| `epochs` | 2 | 训练轮数 |
| `batch_size` | 32 | 微 batch（真实 batch = 32×8=256） |
| `learning_rate` | 5e-4 | 初始（也是峰值）学习率 |
| `accumulation_steps` | 8 | 梯度累积步数 |
| `hidden_size` | 768 | 隐藏维度 |
| `num_hidden_layers` | 8 | 层数 |
| `max_seq_len` | 340 | 序列截断长度（中文 1 token ≈ 1.5~1.7 字符） |
| `use_moe` | 0 | 是否 MoE |
| `dtype` | bfloat16 | 混合精度类型 |
| `from_weight` | none | 从哪个权重继续，none=从头 |

### 阶段 1：初始化分布式 + 种子（109–112 行）

```python
local_rank = init_distributed_mode()
if dist.is_initialized(): args.device = f"cuda:{local_rank}"
setup_seed(42 + (dist.get_rank() if dist.is_initialized() else 0))
```

`init_distributed_mode`（`trainer_utils.py:44`）：
- 如果环境变量 `RANK == -1`（即没有用 `torchrun` 启动），**返回 0，走单卡**。
- 否则用 `nccl` 后端初始化进程组，`torch.cuda.set_device(local_rank)`。

`setup_seed(42 + rank)`：每个 rank 用**不同的种子**（42、43、44…），保证多卡时各卡采样到的数据不同（配合后面的 `randperm`），避免所有卡学同一批数据。

### 阶段 2：建目录 + 配置 + 检测续训 ckpt（114–117 行）

```python
os.makedirs(args.save_dir, exist_ok=True)
lm_config = MiniMindConfig(hidden_size=args.hidden_size, num_hidden_layers=args.num_hidden_layers, use_moe=bool(args.use_moe))
ckp_data = lm_checkpoint(lm_config, weight=args.save_weight, save_dir='../checkpoints') if args.from_resume==1 else None
```

- `lm_config` 里 `vocab_size=6400`（词表大小）、`tie_word_embeddings=True`（输入输出 embedding 共享权重）等。
- `from_resume==1` 时，`lm_checkpoint` 进入「加载模式」（`model=None` 分支），去 `../checkpoints/pretrain_768_resume.pth` 找续训文件；找不到返回 `None`。`from_resume==0` 则直接 `ckp_data=None`，不检测。

### 阶段 3：混合精度配置（119–122 行）

```python
device_type = "cuda" if "cuda" in args.device else "cpu"
dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
autocast_ctx = nullcontext() if device_type == "cpu" else torch.cuda.amp.autocast(dtype=dtype)
```

- bf16 是默认（`bfloat16`），它在 A100/30 系/40 系等新卡上数值更稳（指数位多、不易溢出）。
- CPU 上不做 autocast（用 `nullcontext` 占位）。
- **注意**：bf16 时 `GradScaler` 是**禁用**的（见阶段 5），因为 bf16 不需要 scale 防下溢；只有 fp16 才启用。

### 阶段 4：配置 wandb（124–131 行）

这里 `import swanlab as wandb`——**实际用的是国产实验跟踪工具 SwanLab，别名当 wandb 用**。只在主进程初始化，支持从 ckpt 恢复 `wandb_id` 续接上次的 run。

### 阶段 5：定义模型、数据、优化器（133–138 行）

```python
model, tokenizer = init_model(lm_config, args.from_weight, device=args.device)
train_ds = PretrainDataset(args.data_path, tokenizer, max_length=args.max_seq_len)
train_sampler = DistributedSampler(train_ds) if dist.is_initialized() else None
scaler = torch.cuda.amp.GradScaler(enabled=(args.dtype == 'float16'))
optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)
```

- `init_model`（`trainer_utils.py:119`）：
  - 加载 tokenizer（`../model` 目录，`AutoTokenizer`）。
  - `MiniMindForCausalLM(lm_config)` 建模型。
  - 若 `from_weight != 'none'`：从 `../out/{from_weight}_768.pth` 加载权重（`strict=False` 允许部分加载）。默认 `none` → **随机初始化，从头预训练**。
  - 打印参数量（`get_model_params`），并把模型 `.to(device)`。
- `PretrainDataset`：加载 jsonl 纯文本，见下方数据流例子。
- `DistributedSampler`：多卡时让每张卡拿到**互不重叠**的数据切片；单卡为 `None`。
- `GradScaler(enabled=(dtype=='float16'))`：**只有 fp16 才启用**，bf16 时 scale 因子恒为 1（等效普通反向）。
- `AdamW` 优化器，初始 lr=5e-4（之后每步被 `get_lr` 覆盖）。

### 阶段 6：从 ckpt 恢复（140–147 行）

```python
if ckp_data:
    model.load_state_dict(ckp_data['model'])
    optimizer.load_state_dict(ckp_data['optimizer'])
    scaler.load_state_dict(ckp_data['scaler'])
    start_epoch = ckp_data['epoch']
    start_step = ckp_data.get('step', 0)
```

恢复模型权重、优化器动量状态、scaler 状态，以及**上次训练到哪一步**（`start_epoch`/`start_step`）。

### 阶段 7：编译 + 分布式包装（149–154 行）

```python
if args.use_compile == 1:
    model = torch.compile(model)
if dist.is_initialized():
    model = DistributedDataParallel(model, device_ids=[local_rank])
```

- `torch.compile`：图优化加速（首次编译慢，之后快）。
- `DDP`：多卡数据并行，每卡一份模型副本，反向后自动 all-reduce 梯度求平均。

### 阶段 8：训练主循环（156–167 行）

```python
for epoch in range(start_epoch, args.epochs):
    train_sampler and train_sampler.set_epoch(epoch)
    setup_seed(42 + epoch); indices = torch.randperm(len(train_ds)).tolist()
    skip = start_step if (epoch == start_epoch and start_step > 0) else 0
    batch_sampler = SkipBatchSampler(train_sampler or indices, args.batch_size, skip)
    loader = DataLoader(train_ds, batch_sampler=batch_sampler, num_workers=args.num_workers, pin_memory=True)
    ...
    train_epoch(epoch, loader, len(loader) + skip, start_step, wandb)
```

这段是数据采样的关键：

1. **`set_epoch(epoch)`**：让 DDP 的 sampler 每个 epoch 用不同的随机种子打乱顺序（避免每轮顺序一样）。
2. **`setup_seed(42 + epoch); indices = randperm(...)`**：单卡时，手动生成一个「打乱后的样本索引」列表，代替 sampler。
3. **`SkipBatchSampler`**（`trainer_utils.py:134`）：把索引按 `batch_size=32` 打包成 batch，并支持**跳过前 N 个 batch**（续训用）。它的 `__len__` = `ceil(样本数/32) - skip_batches`。

**`SkipBatchSampler` 例子**（假设 100 个样本，batch_size=32，skip=0）：

```
indices = [37, 12, 88, ... 共100个打乱的索引]
batch0 = [37,12,...,(32个)]   # 第1个batch
batch1 = [下一个32个]
batch2 = [再32个]
batch3 = [最后4个]            # 100 = 32*3 + 4，余数也打包
```

`__len__` = ceil(100/32) = 4。

4. **续训的 skip 逻辑**：`skip = start_step`（只在 `epoch == start_epoch` 时生效）。`SkipBatchSampler` 里 `skipped < skip_batches` 时会**丢弃**前几个 batch，让 DataLoader 从上次断点继续。

**这里有个巧妙点**：`train_epoch` 里 `enumerate(loader, start=start_step+1)` 让 step 编号从断点接着数，而 `iters = len(loader) + skip` 补回被跳过的 batch 数，保证：
- lr 调度用的 `epoch*iters + step` 是全局连续的正确进度；
- 日志显示 `(step/iters)` 比例正确。

### 阶段 9：清理（169–172 行）

```python
if dist.is_initialized():
    dist.barrier()
    dist.destroy_process_group()
```

所有卡 `barrier()` 同步后销毁进程组，干净退出。

---

## 五、一条数据的完整旅程（端到端例子）

假设 `pretrain_t2t_mini.jsonl` 里有一行：

```json
{"text": "今天天气不错，适合出去散步。"}
```

**1. `PretrainDataset.__getitem__`**（`max_length=340`）：

```
tokenize("今天天气不错，适合出去散步。", add_special_tokens=False, max_length=338, truncation=True)
  → 假设得到 12 个 token id: [101, 234, 567, 89, ...]

tokens = [1] + [101, 234, ...] + [2]        # bos=1, eos=2
       = [1, 101, 234, ..., 2]              # 14 个

input_ids = tokens + [0] * (340 - 14)       # 补 326 个 pad(0)
labels    = input_ids.clone()
labels[input_ids == 0] = -100               # pad 位置置 -100
```

返回 `(input_ids, labels)`，形状都是 `[340]`。`labels` 里只有 pad 是 -100，其余全是真实 token id（因为预训练是「全序列 next-token 预测」）。

**2. DataLoader 组 batch**：32 条样本堆成 `input_ids [32, 340]`、`labels [32, 340]`。

**3. 前向 + loss**：模型内部对每条序列做 `x[:-1]` 预测 `y[1:]`，即位置 0 预测位置 1，位置 1 预测位置 2……CE 损失只对 `labels != -100` 的位置求和（pad 被忽略）。假设平均 CE=4.23。

**4. 反向 + 累积**：`loss = 4.23 / 8 = 0.529`，`backward()` 累积梯度。

**5. 到第 8 步**：裁剪梯度 → `optimizer.step()` 更新权重 → 清梯度。

---

## 六、几个容易被忽略的细节总结

1. **`loss / accumulation_steps` 是「梯度累积的数学正确性」核心**：不除以 N，累积 N 步就相当于学习率×N 了。
2. **`iters = len(loader) + skip`**：为了续训时 lr/进度日志不错位，是一个容易漏的补丁。
3. **`scaler` 只在 fp16 启用**：bf16 下 GradScaler 是「摆设」，但代码为了统一仍走 `scaler.scale/unscale/step` 这套 API。
4. **保存两份文件**：`../out/*.pth`（纯权重，给下游用） vs `../checkpoints/*_resume.pth`（完整训练状态，给续训用），职责分离。
5. **`setup_seed(42 + epoch)` 每 epoch 重设种子 + `randperm`**：单卡手动实现「每轮不同 shuffle」，DDP 下则由 `set_epoch` 承担。
6. **`half()` 保存**：权重存 fp16 省一半空间；加载后 `load_state_dict(strict=False)` 允许跨架构微调（比如预训练 768 → SFT 仍 768 但某些层不同）。

---

## 附：关键辅助函数速查

| 函数 | 位置 | 作用 |
|------|------|------|
| `get_lr` | trainer_utils.py:40 | 余弦退火（0.1 下限）学习率 |
| `Logger` | trainer_utils.py:35 | 仅主进程打印 |
| `is_main_process` | trainer_utils.py:31 | 判断是否 rank 0 / 单卡 |
| `init_distributed_mode` | trainer_utils.py:44 | 检测环境变量决定 DDP / 单卡 |
| `setup_seed` | trainer_utils.py:54 | 设随机种子 + 关闭 cudnn benchmark |
| `lm_checkpoint` | trainer_utils.py:63 | 保存/加载训练状态（原子写入） |
| `init_model` | trainer_utils.py:119 | 建模型 + 可选加载预训练权重 |
| `SkipBatchSampler` | trainer_utils.py:134 | 打包 batch + 跳过前 N 个 batch（续训） |
