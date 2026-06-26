# scripts/base_train.py 源码详解

> 源文件：`02-nanochat/code/scripts/base_train.py`
>
> 作用：训练 nanochat 的 Base Model，也就是还没有经过 SFT/RL 对齐的预训练语言模型。它负责把分词器、GPT 模型、分布式环境、数据加载器、优化器、学习率调度、评测、采样、checkpoint 和最终报告串成一个完整训练流程。

---

## 1. 这个脚本解决什么问题

`base_train.py` 是 nanochat Stage 2 预训练阶段的主入口。运行方式有两种：

```bash
python -m scripts.base_train
```

或使用多卡分布式：

```bash
torchrun --nproc_per_node=8 -m scripts.base_train
```

如果只在 CPU 或普通笔记本上学习流程，可以把模型和训练量大幅缩小：

```bash
python -m scripts.base_train --depth=4 --max-seq-len=512 --device-batch-size=1 --eval-tokens=512 --core-metric-every=-1 --total-batch-size=512 --num-iterations=20
```

脚本的核心目标是：

1. 根据少量参数，尤其是 `--depth`，自动推导模型大小和训练超参。
2. 从预训练文本数据中不断取 token batch，执行 next-token prediction。
3. 支持单机单卡、单机多卡 DDP、CPU/MPS/CUDA，以及可选 FP8。
4. 定期评估验证集 bpb、CORE 指标，采样文本，并保存 checkpoint。
5. 将训练配置和训练结果写入 nanochat report。

---

## 2. 顶部初始化与依赖

脚本开头先设置：

```python
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
```

这是 CUDA 显存分配器配置，目的是减少长时间训练中的显存碎片问题。它必须尽量早设置，最好在大量 PyTorch CUDA 分配发生前完成。

主要导入可以按功能分组理解：

| 模块 | 用途 |
| --- | --- |
| `torch`, `torch.distributed` | 张量计算、反向传播、DDP 同步 |
| `wandb` | 可选训练日志记录 |
| `GPT`, `GPTConfig`, `Linear` | 模型结构、配置、FP8 评测时临时替换用的线性层 |
| `dataloader.py` | 分布式 token 数据加载器 |
| `common.py` | 设备初始化、日志、dtype、FLOPS 查询、rank 0 打印等 |
| `tokenizer.py` | 获取 tokenizer 和 token 字节长度 |
| `checkpoint_manager.py` | 保存/加载 checkpoint |
| `loss_eval.py` | 验证集 bits per byte 评估 |
| `engine.py` | 采样生成文本 |
| `flash_attention.py` | 判断是否使用 Flash Attention 3 |
| `scripts.base_eval.evaluate_core` | 计算 CORE 指标 |

`print_banner()` 会打印 nanochat 的启动横幅，用于区分日志。

---

## 3. 命令行参数

脚本使用 `argparse` 定义所有训练参数，可以分为几类。

### 3.1 日志与运行环境

```python
--run
--device-type
```

- `--run`：wandb run 名称。默认是 `dummy`，表示不启用真实 wandb 日志。
- `--device-type`：可填 `cuda`、`cpu`、`mps`。留空时自动检测。

### 3.2 FP8 训练

```python
--fp8
--fp8-recipe
```

FP8 只适合 H100 这类支持 FP8 的 CUDA GPU。`--fp8-recipe` 有：

- `tensorwise`：更快，默认推荐。
- `rowwise`：可能更准确，但更慢。

### 3.3 模型结构

```python
--depth
--aspect-ratio
--head-dim
--max-seq-len
--window-pattern
```

`--depth` 是 nanochat 设计里的核心复杂度旋钮。脚本会用它推导：

```python
base_dim = depth * aspect_ratio
model_dim = ceil_to_multiple(base_dim, head_dim)
num_heads = model_dim // head_dim
```

因此 depth 越大，层数更多，隐藏维度也更大。

`--window-pattern` 控制每层注意力窗口：

- `L` 表示 full attention。
- `S` 表示 half-context sliding window attention。
- 默认 `"SSSL"` 表示三层短窗口、一层长窗口循环。

### 3.4 训练时长

```python
--num-iterations
--target-flops
--target-param-data-ratio
```

三者按优先级生效：

1. 如果 `--num-iterations > 0`，直接使用用户指定步数。
2. 否则如果 `--target-flops > 0`，根据目标 FLOPs 反推步数。
3. 否则使用 `--target-param-data-ratio` 计算训练 token 数，再除以 batch size 得到步数。

默认最常用的是第三种。

### 3.5 优化参数

包括 batch size、各类学习率、weight decay、warmup/warmdown：

```python
--device-batch-size
--total-batch-size
--embedding-lr
--unembedding-lr
--matrix-lr
--scalar-lr
--weight-decay
--warmup-steps
--warmdown-ratio
--final-lr-frac
--resume-from-step
```

nanochat 的优化器不是单一 AdamW，而是 `Muon + AdamW` 的组合：

- 矩阵参数使用 Muon。
- embedding、unembedding、标量参数等使用 AdamW。

这就是为什么脚本要分开设置 `embedding_lr`、`unembedding_lr`、`matrix_lr`、`scalar_lr`。

### 3.6 评测、采样与保存

```python
--eval-every
--eval-tokens
--core-metric-every
--core-metric-max-per-task
--sample-every
--save-every
--model-tag
```

- `eval_every`：每隔多少 step 计算验证集 bpb。
- `core_metric_every`：每隔多少 step 计算 CORE 指标。
- `sample_every`：每隔多少 step 用固定 prompts 采样。
- `save_every`：checkpoint 保存频率。默认 `-1` 表示只在最后保存。
- `model_tag`：覆盖 checkpoint 目录名。

---

## 4. 计算环境初始化

```python
device_type = autodetect_device_type() if args.device_type == "" else args.device_type
ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init(device_type)
master_process = ddp_rank == 0
```

`compute_init()` 负责初始化训练后端：

- 单进程时，`ddp_world_size = 1`。
- 使用 `torchrun` 时，会初始化 distributed process group，并为每个 rank 设置对应设备。

`master_process` 表示 rank 0。只有 rank 0 负责主要日志、wandb 和部分采样输出，避免多进程重复打印。

CUDA 下还会设置：

```python
synchronize = torch.cuda.synchronize
get_max_memory = torch.cuda.max_memory_allocated
gpu_peak_flops = get_peak_flops(gpu_device_name)
```

这些用于后面计算 step 时间、显存峰值和 MFU。

---

## 5. wandb 与 Flash Attention 状态

wandb 初始化逻辑：

```python
use_dummy_wandb = args.run == "dummy" or not master_process
wandb_run = DummyWandb() if use_dummy_wandb else wandb.init(...)
```

也就是说：

- 默认 `--run dummy` 不写真实 wandb。
- 非 rank 0 进程永远使用 `DummyWandb`。

Flash Attention 3 判断逻辑：

```python
from nanochat.flash_attention import USE_FA3
using_fa3 = USE_FA3
```

如果无法使用 FA3，脚本会打印警告。尤其当 `window_pattern != "L"` 且回退到 PyTorch SDPA 时，滑动窗口注意力不受 SDPA 高效支持，GPU 利用率会明显变差。

---

## 6. tokenizer 初始化

```python
tokenizer = get_tokenizer()
token_bytes = get_token_bytes(device=device)
vocab_size = tokenizer.get_vocab_size()
```

这里有三个用途：

1. `tokenizer` 用于数据加载、CORE 评估和采样。
2. `token_bytes` 用于把 token loss 转换为 bits per byte。
3. `vocab_size` 用于构造 GPT 模型的词表大小。

bits per byte 简称 bpb，是比 token loss 更便于跨 tokenizer 比较的压缩指标。

---

## 7. 模型构建：先 meta，再分配，再初始化

脚本定义了：

```python
def build_model_meta(depth):
    ...
    with torch.device("meta"):
        model_meta = GPT(config)
    return model_meta
```

`meta` device 只创建张量形状和 dtype，不分配真实内存。这有两个好处：

1. 可以先廉价得到模型结构、参数量、FLOPs 等信息。
2. 避免先在 CPU 创建完整权重再搬到 GPU 造成额外内存峰值。

真正训练的模型按三步创建：

```python
model = build_model_meta(args.depth)
model.to_empty(device=device)
model.init_weights()
```

含义是：

1. 在 meta device 上创建模型结构。
2. 在目标设备上为每个 tensor 分配 storage，但内容还未初始化。
3. 调用模型自己的 `init_weights()` 初始化权重。

如果传了 `--resume-from-step`，脚本会加载 checkpoint：

```python
model_data, optimizer_data, meta_data = load_checkpoint(...)
model.load_state_dict(model_data, strict=True, assign=True)
```

`assign=True` 表示尽量直接替换参数 tensor，有助于减少额外拷贝。

checkpoint 目录为：

```text
<base_dir>/base_checkpoints/<model_tag 或 d{depth}>
```

---

## 8. FP8 训练与评测时禁用 FP8

如果开启 `--fp8`，脚本会把符合条件的 `nn.Linear` 转成 FP8 训练层：

```python
convert_to_float8_training(model, config=fp8_config, module_filter_fn=fp8_module_filter)
```

过滤条件包括：

- 必须是 `nn.Linear`。
- 输入输出维度都能被 16 整除。
- 矩阵不能太小，最小维度至少 128。

这样做是因为 FP8 硬件 kernel 对维度有要求，小矩阵用 FP8 也未必划算。

脚本还定义了：

```python
@contextmanager
def disable_fp8(model):
    ...
```

它在评估时临时把 `Float8Linear` 替换为普通 `Linear`，并共享同一份权重。原因是评估希望使用 BF16 路径，结果更稳定，也避免 FP8 量化对验证指标造成额外噪声。

流程是：

1. 找到所有类型名包含 `Float8` 的模块。
2. 用 nanochat 自定义 `Linear` 临时替换。
3. `yield` 执行评估。
4. finally 中恢复原 FP8 模块。

---

## 9. torch.compile

```python
orig_model = model
model = torch.compile(model, dynamic=False)
```

训练使用 compiled model，提高执行效率。`dynamic=False` 的依据是训练输入 shape 固定：

```text
[device_batch_size, max_seq_len]
```

脚本保留 `orig_model` 有两个用途：

1. 保存 checkpoint 时保存原始模型的 `state_dict()`。
2. 采样和 CORE 评估时输入长度会变化，使用未编译模型可以避免反复触发重新编译。

---

## 10. 参数量、FLOPs 与缩放律

训练前，脚本先计算参数量：

```python
param_counts = model.num_scaling_params()
num_params = param_counts["total"]
num_flops_per_token = model.estimate_flops()
```

然后定义用于缩放律的参数量：

```python
scaling_params = transformer_matrices + lm_head
```

这里没有简单使用总参数量，而是采用作者经验上更适合缩放律分析的参数集合。

### 10.1 训练 token 数

默认按 data:param ratio 计算：

```python
target_tokens = target_param_data_ratio * num_scaling_params
```

例如 `--target-param-data-ratio=12` 表示目标训练 token 数约为参与缩放律参数量的 12 倍。

### 10.2 参考模型 d12

脚本创建一个 depth=12 的 meta model：

```python
d12_ref = build_model_meta(12)
D_REF = target_param_data_ratio * get_scaling_params(d12_ref)
B_REF = 2**19
```

d12 是 nanochat 的参考点。许多超参以 d12 的经验值为基准，再迁移到其他 depth。

### 10.3 自动 batch size

如果用户没有指定 `--total-batch-size`，脚本按 Power Lines 论文经验式估计：

```python
predicted_batch_size = B_REF * (target_tokens / D_REF) ** 0.383
total_batch_size = nearest_power_of_2(predicted_batch_size)
```

直觉是：训练数据规模越大，最优 batch size 也应变大，但增长速度小于线性。

### 10.4 学习率缩放

如果 batch size 不等于参考 batch：

```python
batch_lr_scale = (total_batch_size / B_REF) ** 0.5
```

即 AdamW 常见的 sqrt batch scaling。Muon 在这里也沿用同样规则，这是脚本注释中明确写出的经验假设。

### 10.5 weight decay 缩放

脚本使用 T_epoch 框架保持正则化强度：

```python
weight_decay_scaled = weight_decay * sqrt(B / B_REF) * (D_REF / D)
```

其中：

- `B` 是当前总 batch size。
- `D` 是当前目标 token horizon。
- `B_REF` 和 `D_REF` 来自 d12 参考模型。

---

## 11. 优化器初始化

优化器由模型方法创建：

```python
optimizer = model.setup_optimizer(
    unembedding_lr=args.unembedding_lr * batch_lr_scale,
    embedding_lr=args.embedding_lr * batch_lr_scale,
    scalar_lr=args.scalar_lr * batch_lr_scale,
    matrix_lr=args.matrix_lr * batch_lr_scale,
    weight_decay=weight_decay_scaled,
)
```

这一步会在模型内部把参数分组，并给每组设置：

- 初始学习率 `initial_lr`
- 参数类别 `kind`
- Muon 或 AdamW 所需的额外配置

训练循环里每一步会重新写入每个 param group 的 `lr`，Muon 组还会额外更新 `momentum` 和 `weight_decay`。

如果是 resume，则加载 optimizer state：

```python
optimizer.load_state_dict(optimizer_data)
```

---

## 12. GradScaler

```python
scaler = torch.amp.GradScaler() if COMPUTE_DTYPE == torch.float16 else None
```

只有 fp16 训练需要 GradScaler。bf16 和 fp32 不需要，因为 bf16 的指数范围接近 fp32，不容易出现 fp16 常见的梯度下溢。

分布式 fp16 下还有一个重要细节：如果任一 rank 发现 inf/nan 梯度，所有 rank 都必须跳过 optimizer step。脚本通过 all-reduce `found_inf` 标志实现一致跳过。

---

## 13. 数据加载器

```python
train_loader = tokenizing_distributed_data_loader_with_state_bos_bestfit(...)
build_val_loader = lambda: tokenizing_distributed_data_loader_bos_bestfit(...)
x, y, dataloader_state_dict = next(train_loader)
```

训练数据加载器有两个特点：

1. 分布式：不同 rank 读取不同数据切片，配合 DDP。
2. 带状态：保存 checkpoint 时会保存 `dataloader_state_dict`，resume 后可以尽量从相同数据位置继续。

`bos_bestfit` 表示数据会按序列长度尽量填满 `max_seq_len`，并在文档边界处插入 BOS token。

`x, y` 是 next-token prediction 的输入和目标：

```text
x = token 序列前 T 个位置
y = token 序列后移一位
```

---

## 14. 训练步数与调度器

### 14.1 num_iterations

脚本保证至少指定一种训练时长：

```python
assert args.num_iterations > 0 or args.target_param_data_ratio > 0 or args.target_flops > 0
```

最终得到：

```python
total_tokens = total_batch_size * num_iterations
```

### 14.2 学习率 schedule

```python
def get_lr_multiplier(it):
    ...
```

分三段：

1. warmup：前 `warmup_steps` 线性升到 1。
2. plateau：保持 1。
3. warmdown：最后 `warmdown_ratio * num_iterations` 步线性降到 `final_lr_frac`。

注意 warmdown 是线性，不是 cosine。

### 14.3 Muon momentum schedule

```python
def get_muon_momentum(it):
    ...
```

Muon momentum 的变化：

- 前 400 步从 0.85 升到 0.97。
- 中间保持 0.97。
- warmdown 阶段从 0.97 降到 0.90。

### 14.4 weight decay schedule

```python
def get_weight_decay(it):
    return weight_decay_scaled * 0.5 * (1 + cos(pi * it / num_iterations))
```

这是 cosine decay，从初始 `weight_decay_scaled` 逐渐降到 0。

---

## 15. resume 与 loop state

如果不是 resume，训练循环状态从头开始：

```python
step = 0
val_bpb = None
min_val_bpb = inf
smooth_train_loss = 0
total_training_time = 0
```

如果 resume，则从 checkpoint meta 中恢复：

```python
step = meta_data["step"]
loop_state = meta_data["loop_state"]
val_bpb = meta_data["val_bpb"]
min_val_bpb = loop_state["min_val_bpb"]
smooth_train_loss = loop_state["smooth_train_loss"]
total_training_time = loop_state["total_training_time"]
```

因此 checkpoint 不只保存模型和优化器，还保存日志所需状态、最优验证指标和数据加载器进度。

---

## 16. gradient accumulation

单个 rank 一次 forward/backward 处理的 token 数：

```python
tokens_per_fwdbwd = device_batch_size * max_seq_len
```

所有 rank 合计：

```python
world_tokens_per_fwdbwd = tokens_per_fwdbwd * ddp_world_size
```

为了达到全局 `total_batch_size`：

```python
grad_accum_steps = total_batch_size // world_tokens_per_fwdbwd
```

脚本要求整除：

```python
assert total_batch_size % world_tokens_per_fwdbwd == 0
```

也就是说，真实优化器 step 的 batch token 数是：

```text
device_batch_size * max_seq_len * ddp_world_size * grad_accum_steps
```

---

## 17. 主训练循环整体结构

主循环是：

```python
while True:
    last_step = step == num_iterations
    ...
    if eval condition: ...
    if core condition: ...
    if sample condition: ...
    if save condition: ...
    if last_step: break
    single training step
    logging
    step += 1
    gc management
```

一个容易忽略的点：循环会运行 `num_iterations + 1` 次。第 `num_iterations` 次不再训练，只用于最后一次 eval/save。

---

## 18. 验证集 bpb 评估

触发条件：

```python
args.eval_every > 0 and (last_step or step % args.eval_every == 0)
```

执行流程：

```python
model.eval()
val_loader = build_val_loader()
eval_steps = eval_tokens // (device_batch_size * max_seq_len * ddp_world_size)
with disable_fp8(model):
    val_bpb = evaluate_bpb(model, val_loader, eval_steps, token_bytes)
model.train()
```

所有 rank 都参与验证。`evaluate_bpb` 会统计若干 token 的 loss，并转换为 bits per byte。

日志会记录：

- 当前 step
- 已消耗训练 FLOPs
- 已统计训练时间
- `val/bpb`

---

## 19. CORE 指标评估

触发条件：

```python
args.core_metric_every > 0 and (last_step or (step > 0 and step % args.core_metric_every == 0))
```

CORE 指标使用：

```python
results = evaluate_core(orig_model, tokenizer, device, max_per_task=...)
```

这里刻意用 `orig_model`，因为 CORE 评估的输入 shape 会变化。如果用 `torch.compile` 后的训练模型，可能导致频繁重新编译。

日志记录：

- `core_metric`
- `centered_results`

---

## 20. 文本采样

采样只在 master process 执行：

```python
if args.sample_every > 0 and master_process and ...
```

脚本用固定 prompts 检查模型基本知识和补全能力，例如：

```text
The capital of France is
The chemical symbol of gold is
If 5*x + 3 = 13, then x is
```

采样过程：

```python
engine = Engine(orig_model, tokenizer)
tokens = tokenizer(prompt, prepend="<|bos|>")
sample, _ = engine.generate_batch(..., temperature=0)
```

`temperature=0` 表示贪心生成，方便观察训练过程中模型能力是否稳定提升。

---

## 21. checkpoint 保存

保存条件：

```python
last_step
or (
    step > 0
    and step != args.resume_from_step
    and args.save_every > 0
    and step % args.save_every == 0
)
```

保存内容包括：

| 内容 | 说明 |
| --- | --- |
| `orig_model.state_dict()` | 原始未 compile 模型参数 |
| `optimizer.state_dict()` | 优化器状态 |
| `step` | 当前 step |
| `val_bpb` | 最近一次验证 bpb |
| `model_config` | 模型结构配置 |
| `user_config` | 命令行参数 |
| `device_batch_size`, `max_seq_len`, `total_batch_size` | 训练 batch 配置 |
| `dataloader_state_dict` | 数据加载器状态 |
| `loop_state` | 最优 bpb、平滑 loss、累计训练时间 |

保存时传入 `rank=ddp_rank`，通常由 checkpoint manager 内部保证只有合适的 rank 写文件。

---

## 22. 单个训练 step

训练 step 从同步和计时开始：

```python
synchronize()
t0 = time.time()
```

### 22.1 forward/backward 与梯度累积

```python
for micro_step in range(grad_accum_steps):
    loss = model(x, y)
    train_loss = loss.detach()
    loss = loss / grad_accum_steps
    loss.backward()
    x, y, dataloader_state_dict = next(train_loader)
```

关键点：

1. 每个 micro step 都 forward/backward 一次。
2. loss 除以 `grad_accum_steps`，因为 PyTorch 默认累加梯度；这样累加后的梯度等价于完整大 batch 的平均 loss 梯度。
3. `next(train_loader)` 放在 backward 后面，注释说是为了在 GPU 忙时预取下一批数据。严格说 Python 执行仍是顺序的，但数据加载器内部可能有异步或预取机制。

fp16 时改用：

```python
scaler.scale(loss).backward()
```

### 22.2 更新优化器参数组

每个 optimizer step 前重新设置：

```python
lrm = get_lr_multiplier(step)
muon_momentum = get_muon_momentum(step)
muon_weight_decay = get_weight_decay(step)
```

然后遍历 param groups：

```python
group["lr"] = group["initial_lr"] * lrm
if group["kind"] == "muon":
    group["momentum"] = muon_momentum
    group["weight_decay"] = muon_weight_decay
```

只有 Muon 组使用动态 momentum 和 weight decay。

### 22.3 optimizer step

fp16 路径：

```python
scaler.unscale_(optimizer)
all_reduce found_inf if DDP
scaler.step(optimizer)
scaler.update()
```

非 fp16 路径：

```python
optimizer.step()
```

最后清空梯度：

```python
model.zero_grad(set_to_none=True)
```

`set_to_none=True` 可以减少显存写入，通常比把梯度清零更高效。

---

## 23. 训练日志

训练 loss 使用 EMA 平滑：

```python
smooth_train_loss = 0.9 * smooth_train_loss + 0.1 * train_loss_f
debiased_smooth_loss = smooth_train_loss / (1 - 0.9 ** (step + 1))
```

debiased 是为了修正 EMA 初期偏低的问题。

吞吐和 MFU：

```python
tok_per_sec = total_batch_size / dt
flops_per_sec = num_flops_per_token * total_batch_size / dt
mfu = flops_per_sec / (gpu_peak_flops * ddp_world_size)
```

MFU 是 Model FLOPS Utilization，用来估计当前训练实际利用了理论峰值算力的多少。

脚本前 10 步不计入 `total_training_time`：

```python
if step > 10:
    total_training_time += dt
```

这是因为前几步可能包含编译、缓存、预热等额外开销，不代表稳定训练速度。

每 100 step 写一次 wandb：

```python
if step % 100 == 0:
    wandb_run.log(...)
```

---

## 24. 垃圾回收管理

训练第一步结束后：

```python
gc.collect()
gc.freeze()
gc.disable()
```

注释说明：Python GC 在长训练中可能频繁扫描对象，带来约数百毫秒级别的额外开销。这里选择手动管理：

- 第一轮后收集一次垃圾。
- freeze 当前存活对象。
- 禁用自动 GC。
- 之后每 5000 step 手动 `gc.collect()` 一次。

这是偏工程性能优化的细节，不影响训练数学逻辑。

---

## 25. 训练结束统计与 report

循环结束后打印：

```python
Peak memory usage
Total training time
Minimum validation bpb
```

然后写入 report：

```python
from nanochat.report import get_report
get_report().log(section="Base model training", data=[...])
```

report 分三块：

1. 用户传入的 CLI 参数。
2. 训练设置统计：参数量、FLOPs/token、训练步数、训练 token 数、DDP world size、schedule 参数等。
3. 训练结果：最小/最终 bpb、CORE、MFU、总 FLOPs、总时间、峰值显存。

最后清理：

```python
wandb_run.finish()
compute_cleanup()
```

`compute_cleanup()` 会在 DDP 场景下销毁 process group，并做设备相关清理。

---

## 26. 从整体数据流理解

可以把整个脚本压缩成下面的数据流：

```text
CLI args
  -> compute_init()
  -> tokenizer
  -> GPTConfig / GPT model
  -> optional checkpoint resume
  -> optional FP8 conversion
  -> torch.compile
  -> scaling-law hyperparameter calculation
  -> optimizer
  -> train/val dataloaders
  -> schedulers
  -> training loop
       -> eval bpb
       -> eval CORE
       -> sample text
       -> save checkpoint
       -> forward/backward over micro-batches
       -> optimizer step
       -> log throughput/loss/MFU
  -> report
  -> cleanup
```

最核心的训练闭环是：

```text
取 batch (x, y)
  -> model(x, y) 得到 next-token loss
  -> loss / grad_accum_steps
  -> backward 累积梯度
  -> 按 schedule 更新 lr / momentum / weight_decay
  -> optimizer.step()
  -> 清梯度
  -> 记录 loss、速度、MFU
```

---

## 27. 阅读这个脚本时最重要的几个点

1. `--depth` 不只是层数，它还间接决定隐藏维度、head 数、参数规模、目标 token 数和默认 batch size。
2. 模型先在 `meta` device 上创建，再 `to_empty()` 到目标设备，最后初始化权重，这是为了降低初始化内存峰值。
3. `orig_model` 和 `model` 分别表示未 compile 和已 compile 版本；训练用 compiled，保存/动态 shape 评估用 original。
4. `total_batch_size` 是 token 级别的全局 batch size，不是样本条数。
5. gradient accumulation 的单位是 micro-batch，最终每个 optimizer step 看到的是 `total_batch_size` 个 token。
6. 学习率、Muon momentum、weight decay 都在每个 step 动态更新。
7. FP8 只影响训练线性层；评估时通过 `disable_fp8()` 临时回到 BF16/普通 Linear。
8. checkpoint 保存了 dataloader 和 loop state，因此 resume 尽量能延续原训练轨迹。
9. 验证指标 bpb 衡量语言建模压缩能力，CORE 指标更接近下游基础能力。
10. 训练循环多跑一次 final iteration，是为了在最后 step 做评估和保存，而不是多训练一步。

---

## 28. 常见修改入口

| 需求 | 主要修改位置 |
| --- | --- |
| 改模型大小策略 | `build_model_meta()` 和 scaling-law 相关代码 |
| 改训练时长计算 | `num_iterations` 计算分支 |
| 改学习率曲线 | `get_lr_multiplier()` |
| 改 Muon momentum | `get_muon_momentum()` |
| 改 weight decay 曲线 | `get_weight_decay()` |
| 改评估频率或指标 | eval/core metric 条件分支 |
| 改采样 prompt | `prompts = [...]` |
| 改 checkpoint 内容 | `save_checkpoint(..., metadata)` |
| 改日志字段 | `wandb_run.log(...)` 和 `print0(...)` |

---

## 29. 与其他文件的关系

| 文件 | 关系 |
| --- | --- |
| `nanochat/gpt.py` | 定义 GPT 模型、参数统计、FLOPs 估计、优化器分组 |
| `nanochat/optim.py` | 实现 MuonAdamW / DistMuonAdamW |
| `nanochat/dataloader.py` | 产生训练/验证 token batch，并支持分布式和 resume |
| `nanochat/tokenizer.py` | 提供 tokenizer 和 bpb 需要的 token byte 信息 |
| `nanochat/loss_eval.py` | 实现验证集 bpb |
| `scripts/base_eval.py` | 实现 CORE 指标评估 |
| `nanochat/engine.py` | 用于训练中定期文本采样 |
| `nanochat/checkpoint_manager.py` | checkpoint 保存与加载 |
| `nanochat/common.py` | 设备、DDP、dtype、日志、FLOPS 等通用工具 |
| `nanochat/report.py` | 写训练报告 |

---

## 30. 最小学习建议

如果只是学习源码，可以按这个顺序读：

1. 先读本文档，建立 `base_train.py` 的整体执行顺序。
2. 再读 `nanochat/gpt.py`，理解 `GPT.forward()`、`setup_optimizer()`、`num_scaling_params()` 和 `estimate_flops()`。
3. 再读 `nanochat/dataloader.py`，理解 `(x, y)` 如何从原始文本构造出来。
4. 最后读 `nanochat/optim.py` 和 `scripts/base_eval.py`，理解 Muon 与评估指标。

这样可以先抓住训练主干，再补齐模型结构、数据和优化器细节。
