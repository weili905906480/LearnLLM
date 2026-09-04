# train_distillation.py 详解

> 文件位置：`resources/minimind/trainer/train_distillation.py`

这份代码是 **MiniMind 的知识蒸馏（Knowledge Distillation, KD）训练脚本**：用一个（更大更强的）教师模型 `teacher_model` 指导学生模型 `model` 训练，学生同时学「真实标签（CE Loss）」和「教师输出的软标签（KL 散度 Loss）」。下面按模块逐段解释。

---

## 一、文件头部与导入（1–22 行）

```python
__package__ = "trainer"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import datasets  # noqa: F401  # Windows pyarrow/torch DLL conflict workaround
```

- `__package__ = "trainer"` + 手动 `sys.path.append(父目录)`：因为脚本是**直接 `python train_distillation.py` 运行**而非作为包导入，这两行让 `from model.xxx`、`from dataset.xxx`、`from trainer.xxx` 能正确解析。
- `import datasets`（HuggingFace）放在最前面并标注 `noqa`：这是 Windows 上 pyarrow 与 torch 的 **DLL 加载顺序冲突**的 workaround，先导入 datasets 避免后导入时崩溃（注释里指向 issue #771）。

导入了：

- `F = torch.nn.functional`（用于 `softmax`/`log_softmax`/`kl_div`/`cross_entropy`）
- `dist`、`DistributedDataParallel`、`DistributedSampler`（分布式训练）
- `nullcontext`（CPU 上不做 autocast 时用的空上下文）
- 自研工具：`get_lr, Logger, lm_checkpoint, init_distributed_mode, setup_seed, init_model, SkipBatchSampler`

---

## 二、核心：蒸馏损失函数（25–36 行）

```python
def distillation_loss(student_logits, teacher_logits, temperature=1.0, reduction='batchmean'):
    with torch.no_grad():
        teacher_probs = F.softmax(teacher_logits / temperature, dim=-1).detach()

    student_log_probs = F.log_softmax(student_logits / temperature, dim=-1)

    kl = F.kl_div(student_log_probs, teacher_probs, reduction=reduction)
    return (temperature ** 2) * kl
```

这是标准的知识蒸馏 KL 散度损失，分几步：

1. **教师软标签**：`softmax(teacher_logits / T)`。温度 `T` 越高，概率分布越「平滑」，越能暴露类别之间的相对关系（比如「猫」和「豹」都比「汽车」更像，这种暗知识在 `T=1` 时被硬标签抹掉了）。
2. **学生 log-softmax**：`log_softmax(student_logits / T)`。
3. **KL 散度**：`KL(teacher || student) = Σ teacher_probs · log(teacher_probs/student_probs)`。PyTorch 的 `F.kl_div(input, target)` 里 `input` 是 log 概率、`target` 是概率，等价于这里 `student_log_probs` 与 `teacher_probs` 的逐点乘积求和。
4. **温度平方缩放** `T²`：因为对 logits 除以 `T` 后，梯度的量级被缩小了约 `1/T²`，乘回来让蒸馏损失的梯度尺度不因温度变化而失真。

具体举例（1 个 token、词表 V=4）：

```text
教师原始 logits   z_t = [4.0, 3.0, 2.0, 1.0]
学生原始 logits   z_s = [3.0, 2.0, 1.0, 0.0]

温度 T=1（接近 one-hot）: softmax(z_t)   ≈ [0.644, 0.237, 0.087, 0.032]
温度 T=2（更平滑）      : softmax(z_t/2) ≈ [0.455, 0.276, 0.167, 0.102]
                                          ↑最高值降低、最低值升高，分布更"平"
```

关键细节：教师部分用 `no_grad()` + `.detach()`，**教师的输出只是固定目标，不反传梯度**；只有学生端可训练。

---

## 三、训练一个 epoch：`train_epoch`（39–143 行）

### 3.1 教师模型固定（43–45 行）

```python
teacher_model.eval()
teacher_model.requires_grad_(False)
```

教师始终在 eval 模式、关闭梯度，且**不会**被 `optimizer` 收录（下面 optimizer 只包装 `model.parameters()`），所以它只做前向推理。

### 3.2 主循环与数据准备（47–54 行）

```python
for step, (input_ids, labels) in enumerate(loader, start=start_step + 1):
    ...
    loss_mask = (labels[..., 1:] != -100).float()
    lr = get_lr(epoch * iters + step, args.epochs * iters, args.learning_rate)
```

- `loss_mask`：`labels` 里 `-100` 是「忽略位」（padding / prompt 部分），`!= -100` 得到参与计算的 token 掩码。注意取 `[..., 1:]` 是为了对齐「预测下一个 token」的位移（详见 3.4）。

  具体举例（batch=1, seq_len=6，最后一位是 padding）：

  ```text
  labels          = [  1,   2,   3,   4,   5, -100]
  labels[..., 1:] = [  2,   3,   4,   5, -100]     # 整体左移一位
  loss_mask       = [  1,   1,   1,   1,    0]     # 前4位参与loss，最后padding位忽略
  ```
- `get_lr`：余弦退火学习率调度（见 `trainer_utils.py:40`，`0.1 + 0.45*(1+cos(π·step/total))`），`lr` 随训练进度从初始 `1.0×lr` 单调衰减到 `0.1×lr`（`step=0` 时为 `1.0×lr`，`step=total/2` 时为 `0.55×lr`，`step=total` 时为 `0.1×lr`）。

### 3.3 学生 + 教师前向（56–66 行）

```python
with autocast_ctx:
    res = model(input_ids)
    student_logits = res.logits[..., :-1, :].contiguous()

if teacher_model is not None:
    with torch.no_grad():
        teacher_logits = teacher_model(input_ids).logits[..., :-1, :].contiguous()
        vocab_size_student = student_logits.size(-1)
        teacher_logits = teacher_logits[..., :vocab_size_student]
```

- 学生模型在 `autocast_ctx`（混合精度）里前向，返回对象含 `.logits`（还可能有 `.aux_loss`，MoE 用）。
- 教师模型在 `no_grad` 里前向，省显存。
- `[..., :-1, :]`：语言模型输出 `logits[t]` 预测的是位置 `t+1` 的 token，所以丢掉最后一个位置、与 `labels[1:]` 对齐。

  举例（seq_len=6, vocab_size=4）：

  ```text
  logits[0] 预测 token[1]，logits[1] 预测 token[2]，...，logits[4] 预测 token[5]
  logits[5]（预测不存在的"第7个token"）没有真实标签，直接丢弃
  => student_logits 形状从 [batch, 6, 4] 变为 [batch, 5, 4]
  ```
- **截断教师词表** `teacher_logits[..., :vocab_size_student]`：教师和学生可能词表大小不同，只保留前 `vocab_size_student` 维，保证两者能逐元素对齐做 KL。

### 3.4 Ground-Truth CE Loss（68–80 行）

```python
shift_labels = labels[..., 1:].contiguous()
loss_mask_flat = loss_mask.view(-1)
ce_loss = F.cross_entropy(
    student_logits.view(-1, student_logits.size(-1)),
    shift_labels.view(-1),
    ignore_index=-100,
    reduction='none'
)
ce_loss_raw = torch.sum(ce_loss * loss_mask_flat) / (loss_mask_flat.sum() + 1e-8)
if lm_config_student.use_moe: ce_loss = ce_loss_raw + res.aux_loss
else: ce_loss = ce_loss_raw
```

- `cross_entropy(..., reduction='none')` 对每个 token 算 CE，`ignore_index=-100` 让 padding 位返回 0（配合掩码）。
- 手动 `sum / (mask.sum()+1e-8)`：等价于「只在有效 token 上平均」的 masked CE。加 `1e-8` 防止全 masked 时除零。

  手算举例（batch=1，共 5 个预测位，其中 1 个 padding）：

  ```text
  ce_loss(逐token) = [0.9, 1.2, 0.7, 2.0, 0.0]   # 最后一个 padding 位为 0
  loss_mask_flat   = [  1,   1,   1,   1,   0]
  sum(ce * mask)   = 0.9 + 1.2 + 0.7 + 2.0 = 4.8
  loss_mask.sum()  = 4
  ce_loss_raw      = 4.8 / 4 = 1.2   （分母 +1e-8 防止全 padding 时除零）
  ```
- 若学生是 **MoE** 模型，叠加 `res.aux_loss`（路由负载均衡的辅助损失）。

### 3.5 Distillation Loss（82–90 行）

```python
distill_loss = distillation_loss(
    student_logits.view(-1, student_logits.size(-1))[loss_mask_flat == 1],
    teacher_logits.view(-1, teacher_logits.size(-1))[loss_mask_flat == 1],
    temperature=temperature
)
```

把 logits 展平成 `(N, V)` 后，用布尔索引只挑出**有效 token**（`loss_mask_flat == 1`）的那些行，让教师和学生在这些位置做 KL。没有教师时给 0。

  举例（`loss_mask_flat = [1, 1, 0, 1]`，即 4 个 token 中第 3 个是 padding）：

  ```text
  student_logits.view(-1, V) = 4 行（对应 4 个 token）
  [loss_mask_flat == 1]      = 保留第 1、2、4 行，丢掉第 3 行
  => 蒸馏损失只在 3 个有效 token 上计算
  ```

### 3.6 总损失与反向（92–102 行）

```python
loss = (alpha * ce_loss + (1 - alpha) * distill_loss) / args.accumulation_steps

scaler.scale(loss).backward()

if step % args.accumulation_steps == 0:
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad(set_to_none=True)
```

- **`alpha` 权衡**：`alpha=0.5`（默认）意味着真实标签和监督信号各占一半。`alpha=1` 退化为纯 SFT，`alpha=0` 退化为纯蒸馏。
- 损失除以 `accumulation_steps`，实现**梯度累积**：每步只 backward，攒满 `accumulation_steps` 步才 `step()`。
- **混合精度**（`GradScaler`）：`scale(loss).backward()` → 每累积步 `unscale_`（还原梯度，保证裁剪数值正确）→ `clip_grad_norm_`（梯度裁剪）→ `scaler.step` → `scaler.update`（动态调整 scale）→ 清零。

### 3.7 日志与保存（104–136 行）

- **日志**：每 `log_interval` 步打印 `loss / ce / aux_loss / distill / lr / 剩余时间`。注意 `current_loss = loss.item() * accumulation_steps` 是把之前除以累积步的因子乘回去，还原为真实单步 loss。`eta_min` 用「已耗时/已走步数 × 剩余步数」估算剩余分钟数。
- **保存**：每 `save_interval` 步（且是主进程）：
  1. 保存**纯权重**到 `save_dir/{save_weight}_{hidden}{_moe}.pth`（用于 `init_model` 加载做推理）；
  2. 调用 `lm_checkpoint(...)` 保存 **resume 检查点**（含 model/optimizer/epoch/step 等）到 `../checkpoints`，供断点续训。
  - 保存前先 `model.eval()`、`torch.save` 后 `model.train()`，且把权重转 `half()`（fp16）存到 CPU 省空间。
  - `raw_model = getattr(raw_model, '_orig_mod', raw_model)` 是 `torch.compile` 的兼容处理（拿回未编译的原始模块）。

### 3.8 尾部剩余梯度（138–143 行）

```python
if last_step > start_step and last_step % args.accumulation_steps != 0:
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad(set_to_none=True)
```

如果 epoch 结束时累积步数**没被整除**（最后一小批不够一组），仍要把残留梯度执行一次 `step()`，避免最后几个 step 的梯度丢失。

---

## 四、主函数 `__main__`（146–248 行）

### 4.1 参数（148–177 行）

关键参数一览：

| 参数 | 默认 | 含义 |
|------|------|------|
| `student_hidden_size` / `student_num_layers` | 768 / 8 | 学生模型结构 |
| `teacher_hidden_size` / `teacher_num_layers` | 768 / 8 | 教师模型结构（可比学生大） |
| `student_use_moe` / `teacher_use_moe` | 0 / 1 | 是否 MoE。默认**用 MoE 教师蒸馏 dense 学生** |
| `from_student_weight` / `from_teacher_weight` | `full_sft` | 各自的初始化权重 |
| `alpha` | 0.5 | CE 与 KL 的权衡 |
| `temperature` | 1.5 | 蒸馏温度 |
| `max_seq_len` | 340 | 截断长度（注释：中文 1 token ≈ 1.5–1.7 字符） |
| `use_compile` | 0 | 是否 `torch.compile` 加速 |

注释第 147 行点明了典型场景：

> 用 MoE 模型蒸馏 dense 模型，也可以更大的 teacher_hidden_size 蒸馏更小的 student_hidden_size。

### 4.2 初始化流程（179–214 行）

1. **分布式** `init_distributed_mode()`：读环境变量 `RANK`，若为 -1 则单卡；否则 `init_process_group("nccl")` 并返回 `local_rank`。
2. **随机种子** `setup_seed(42 + rank)`：不同 rank 用不同种子（分布式时每个进程数据打乱略有差异，可增加多样性）。
3. **配置**：分别构建 `lm_config_student` 和 `lm_config_teacher`。
4. **混合精度**：`autocast_ctx = nullcontext()`（CPU 时）或 `torch.cuda.amp.autocast(dtype)`；`GradScaler` 只在 **fp16**（非 bf16）时启用，因为 bf16 动态范围大，不需要 scale。
5. **wandb（实为 swanlab）**：`import swanlab as wandb`——用国产 SwanLab 替换了 wandb 接口。
6. **建模型**：`init_model` 从 `../out/{weight}_{hidden}{_moe}.pth` 加载预训练权重。教师随即 `eval()` + `requires_grad_(False)` 冻结。
7. **数据**：`SFTDataset` + `DistributedSampler`（多卡时每卡分片）。

### 4.3 续训恢复（216–223 行）

```python
if ckp_data:
    model.load_state_dict(ckp_data['model'])
    optimizer.load_state_dict(ckp_data['optimizer'])
    scaler.load_state_dict(ckp_data['scaler'])
    start_epoch = ckp_data['epoch']
    start_step = ckp_data.get('step', 0)
```

只有 `--from_resume 1` 时才通过 `lm_checkpoint(...)` 的「加载模式」拿到 `ckp_data`，恢复模型/优化器/缩放器和 epoch、step。

### 4.4 编译 + DDP（226–230 行）

```python
if args.use_compile == 1: model = torch.compile(model)
if dist.is_initialized(): model = DistributedDataParallel(model, device_ids=[local_rank])
```

先编译再 DDP 包装（`torch.compile` 先于 DDP，PyTorch 推荐顺序）。

### 4.5 训练循环（233–243 行）

```python
for epoch in range(start_epoch, args.epochs):
    train_sampler and train_sampler.set_epoch(epoch)
    setup_seed(42 + epoch); indices = torch.randperm(len(train_ds)).tolist()
    skip = start_step if (epoch == start_epoch and start_step > 0) else 0
    batch_sampler = SkipBatchSampler(train_sampler or indices, args.batch_size, skip)
    loader = DataLoader(train_ds, batch_sampler=batch_sampler, num_workers=args.num_workers, pin_memory=True)
```

- `set_epoch(epoch)`：让 `DistributedSampler` 每个 epoch 打乱顺序不同。
- 单卡时用 `torch.randperm` 得到打乱后的索引；多卡时用 sampler。
- **`SkipBatchSampler`**（`trainer_utils.py:134`）：续训时跳过前 `start_step` 个 batch，实现「从上次断点接着练」。注意第 241 行把 `iters` 传成 `len(loader) + skip`，是因为 `enumerate` 从 `start_step+1` 起，这样 `iters` 才是正确的总步数，供 `get_lr` 和进度打印用。

---

## 五、整体流程图

```
学生模型(model) ──前向──▶ student_logits ──┬─ CE Loss(真实标签) ──┐
                                            │                        ├─ loss = α·CE + (1-α)·KL
教师模型(teacher)─前向(no_grad)─▶ teacher_logits ─┴─ KL散度(软标签) ─┘
                                                                    │
                              scaler.scale(loss).backward()（混合精度）
                                                                    │
                             累积步 → unscale → clip_grad → step → update
```

**一句话总结**：这个脚本让学生模型在 SFT 数据上同时优化「对标准答案的交叉熵」和「对教师输出概率分布的 KL 散度」，通过温度 `T` 软化教师分布传递「暗知识」，用 `alpha` 控制两者的比重，从而用较小的学生模型逼近较大教师模型（尤其是 MoE 教师）的能力。

---

## 附：可深入的点

- `F.kl_div` 与标准 KL 公式的对应关系
- 温度平方缩放 `T²` 的数学推导
- MoE 蒸馏里 `aux_loss` 和词表截断的具体作用
