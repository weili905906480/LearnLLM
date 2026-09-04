# `train_full_sft.py` 详解（附具体例子）

> 对应源码：`trainer/train_full_sft.py`

## 一句话概括

这是 MiniMind 的**全参数监督微调（Full SFT）**训练脚本：加载一个预训练好的小语言模型，在「多轮对话」数据上**只对 assistant 回答部分**计算交叉熵损失、更新**全部**参数，最后把模型权重保存下来。所谓 "Full"，是相对 LoRA 等参数高效微调而言——这里不冻结任何参数，所有层都参与梯度更新。

---

## 一、整体流程

脚本从 `if __name__ == "__main__"`（第 84 行）开始，可分成 9 个阶段，代码里也自己标了序号：

```
1. 初始化分布式环境 & 随机种子
2. 配置目录、模型超参、读取续训检查点
3. 设置混合精度 autocast
4. 配置 wandb（实际上用的是 swanlab）
5. 定义模型 / 数据集 / 优化器 / GradScaler
6. 从检查点恢复 optimizer、scaler、epoch、step 状态
7. torch.compile 加速 + DDP 分布式包装
8. 逐 epoch 调用 train_epoch 训练（核心）
9. 清理分布式进程组
```

核心训练逻辑在 `train_epoch`（第 24–81 行），我们重点讲它。

---

## 二、贯穿全文的具体例子

为了后面每一步都能对上号，先定一组**具体参数**（都用脚本默认值附近）：

| 参数 | 取值 | 含义 |
|---|---|---|
| `epochs` | 2 | 训练 2 轮 |
| `batch_size` | 16 | 每批 16 条对话 |
| `learning_rate` | 1e-5 | 初始学习率 |
| `accumulation_steps` | 1 | 不累积梯度，每步都更新 |
| `hidden_size` | 768 | 隐藏维 768 |
| `num_hidden_layers` | 8 | 8 层 |
| `max_seq_len` | 768 | 序列最长 768 token |
| `use_moe` | 0 | 不用 MoE（因此 `aux_loss` 恒为 0） |
| `from_weight` | pretrain | 基于预训练权重继续训练 |

**训练数据**取一条 jsonl 样本（`sft_t2t_mini.jsonl` 里的一行）：

```json
{"conversations": [
  {"role": "user",     "content": "你好"},
  {"role": "assistant", "content": "你好！有什么可以帮你？"}
]}
```

这条样本从原始 json 到参与 loss 计算，会经历一条完整流水线，后面每一步都拿它做演示。

---

## 三、逐段详解

### 阶段 0：模块导入与路径处理（第 1–21 行）

```python
__package__ = "trainer"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import datasets  # noqa: F401  # Windows pyarrow/torch DLL 冲突 workaround
```

- `__package__ = "trainer"`：因为脚本是直接 `python train_full_sft.py` 运行的（不是 `python -m`），Python 会把 `__package__` 设为空，导致相对导入失败。手动指定包名后，`from model.model_minimind import ...` 这类导入才能正常工作。
- `sys.path.append('..')`：把上级目录加进搜索路径，让 `model/`、`dataset/`、`trainer/` 三个包可被导入。
- `import datasets`：一个 Windows 平台的坑——pyarrow 和 torch 的 DLL 加载顺序冲突，先导入 HF 的 `datasets` 可以规避（issue #771）。

---

### 阶段 1：分布式与随机种子（第 110–113 行）

```python
local_rank = init_distributed_mode()
if dist.is_initialized(): args.device = f"cuda:{local_rank}"
setup_seed(42 + (dist.get_rank() if dist.is_initialized() else 0))
```

`init_distributed_mode()`（在 `trainer_utils.py`）判断环境变量 `RANK`：

- **单卡/非 DDP**：`RANK` 不存在 → 返回 0，`args.device` 保持默认 `cuda:0`。
- **多卡 DDP**：调用 `dist.init_process_group("nccl")`，返回 `LOCAL_RANK`，并把 `device` 设为 `cuda:{local_rank}`。

种子设为 `42 + rank`，保证不同卡之间数据打乱不完全一样（配合 `DistributedSampler` 使用）。

> 我们例子按**单卡**走，所以 `local_rank=0`，`device="cuda:0"`，种子 42。

---

### 阶段 2：目录、模型配置、续训检查点（第 115–118 行）

```python
os.makedirs(args.save_dir, exist_ok=True)   # 建 ../out
lm_config = MiniMindConfig(hidden_size=768, num_hidden_layers=8, use_moe=False)
ckp_data = lm_checkpoint(lm_config, weight='full_sft', save_dir='../checkpoints') if args.from_resume==1 else None
```

- `MiniMindConfig` 打包了模型结构超参（768 维、8 层、非 MoE）。
- `lm_checkpoint(...)` 在**只传 config 不传 model** 时是「加载模式」：去 `../checkpoints/full_sft_768.pth`（或其 `_resume.pth`）找续训数据。本例 `from_resume=0`，所以 `ckp_data = None`。

---

### 阶段 3：混合精度（第 120–123 行）

```python
device_type = "cuda" if "cuda" in args.device else "cpu"
dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
autocast_ctx = nullcontext() if device_type == "cpu" else torch.cuda.amp.autocast(dtype=dtype)
```

- 默认 `dtype=bfloat16` → `autocast_ctx` 是 **bf16 的 autocast 上下文**，前向时自动把张量降到 bf16 省显存、提速。
- CPU 训练则 `nullcontext()`（什么都不做）。

---

### 阶段 4：wandb（第 125–132 行）

```python
wandb = None
if args.use_wandb and is_main_process():
    import swanlab as wandb   # 注意：这里 "wandb" 实际是国产的 swanlab
    ...
```

只有主进程（rank 0）才初始化日志记录。注释写 wandb，实际 `import swanlab as wandb`。本例不传 `--use_wandb`，跳过。

---

### 阶段 5：模型 / 数据 / 优化器（第 134–139 行）——**数据是怎么造出来的**

```python
model, tokenizer = init_model(lm_config, args.from_weight, device=args.device)
train_ds = SFTDataset(args.data_path, tokenizer, max_length=args.max_seq_len)
train_sampler = DistributedSampler(train_ds) if dist.is_initialized() else None
scaler = torch.cuda.amp.GradScaler(enabled=(args.dtype == 'float16'))
optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)
```

- `init_model`：`MiniMindForCausalLM(lm_config)` + `AutoTokenizer`；因为 `from_weight='pretrain'`，会 `torch.load('../out/pretrain_768.pth')` 并 `load_state_dict(strict=False)` 加载预训练权重。
- `scaler`：注意 `enabled=(dtype == 'float16')`。默认 bf16 时 **scaler 是禁用的**（bf16 不需要 loss 缩放，因为指数位宽和 fp32 一样，不会像 fp16 那样溢出）。
- `optimizer`：AdamW，lr=1e-5。

#### 关键：`SFTDataset.__getitem__` 如何把一条对话变成 `(input_ids, labels)`

这是整个 SFT 的**灵魂**。以我们的例子走一遍（对应 `lm_dataset.py` 第 220–241 行）：

**① `pre_processing_chat`（第 223 行）**：以 0.2 概率给对话前面加一句随机 system 提示（数据增强）。假设这次没加，conversations 原样返回。

**② `create_chat_prompt`（第 225 行）**：调用 `tokenizer.apply_chat_template(tokenize=False)`，把结构化对话渲染成带 `<|im_start|>/<|im_end|>` 的纯文本：

```
<|im_start|>user\n你好<|im_end|>\n<|im_start|>assistant\n你好！有什么可以帮你？<|im_end|>\n
```

**③ `post_processing_chat`（第 227 行）**：以 80% 概率删掉空的 `<think>\n\n</think>\n\n`（针对 DeepSeek-R1 式思考标签），本例无此标签，不变。

**④ tokenize + 截断 + 补齐（第 229–232 行）**：tokenize 后截到 768，右侧补 `pad_token_id=0` 到长度 768。

**⑤ `generate_labels`（第 234 行）——loss 掩码核心**。这是最关键的一步，直接决定「模型只学回答、不学提示」。

`generate_labels` 预先算出两个记号（第 124–126 行）：
- `bos_id` = `<|im_start|>assistant\n` 的 id 序列 = `[1, 10]`（标出「回答段开头」）
- `eos_id` = `<|im_end|>\n` 的 id 序列 = `[2]`（标出「回合结束」）

然后线性扫描 input_ids，只把夹在 `bos_id` 之后、`eos_id` 之间的那段填成真实 token id，其余全填 `-100`。

用 `lm_dataset.py` 注释里那组示意 id 举例（`<|im_start|>`=1、`<|im_end|>`=2、pad=0，中文用示意 id）：

```
位置 idx :  0   1   2   3   4   5   6   7   8   9   10  11
内容     :  <|im_start|> user 你  好 <|im_end|> <|im_start|> assistant 你 好 <|im_end|> <pad> <pad>
id       :  1   20  30  40  2   1   10  50  60  2   0   0
labels   : -100 -100 -100 -100 -100 -100 -100 50  60  2  -100 -100
                                ↑ 这里 idx5~6=[1,10] 命中 bos_id，回答从 idx7 开始
```

最终 `__getitem__` 返回 `(input_ids[768], labels[768])` 两个 `long` 张量。**只有 idx 7/8/9（即回答「你」「好」+ 结束符）参与 loss**，用户提问和 padding 全是 -100 被忽略。

---

### 阶段 6：恢复训练状态（第 141–148 行）

本例 `ckp_data=None`，所以跳过，`start_epoch=0, start_step=0`。续训时则恢复 model/optimizer/scaler 权重和 epoch/step。

---

### 阶段 7：编译 & DDP 包装（第 150–155 行）

```python
if args.use_compile == 1:
    model = torch.compile(model)     # 图优化加速
if dist.is_initialized():
    model = DistributedDataParallel(model, device_ids=[local_rank])
```

单卡时模型保持裸模型不动。

---

### 阶段 8：训练主循环（第 157–168 行）——**一个 step 里发生了什么**

```python
for epoch in range(start_epoch, args.epochs):
    train_sampler and train_sampler.set_epoch(epoch)
    setup_seed(42 + epoch)
    indices = torch.randperm(len(train_ds)).tolist()   # 每 epoch 打乱一遍
    skip = start_step if (epoch == start_epoch and start_step > 0) else 0
    batch_sampler = SkipBatchSampler(train_sampler or indices, args.batch_size, skip)
    loader = DataLoader(train_ds, batch_sampler=batch_sampler, num_workers=8, pin_memory=True)
    train_epoch(epoch, loader, len(loader), 0, wandb)
```

假设 `len(train_ds)=1600` 条 → `iters = len(loader) = 1600/16 = 100` 步/epoch，2 个 epoch 共 200 步。

`SkipBatchSampler` 把打乱后的下标按 batch_size=16 分组，续训时能跳过前面 `skip` 个 batch 精确接上进度。

#### 核心：`train_epoch` 内部（第 24–81 行）

逐行拆解一个 step（以 epoch 0、step 1 为例）：

```python
for step, (input_ids, labels) in enumerate(loader, start=1):
```

`loader` 给出一批 `input_ids: [16, 768]`、`labels: [16, 768]`（16 条对话、每条 768 token）。

```python
input_ids = input_ids.to(args.device); labels = labels.to(args.device)
```

搬到 GPU。

```python
lr = get_lr(epoch * iters + step, args.epochs * iters, args.learning_rate)
for param_group in optimizer.param_groups:
    param_group['lr'] = lr
```

**学习率调度**（`trainer_utils.py` 第 40–41 行）：

```
lr = lr * (0.1 + 0.45 * (1 + cos(π * cur_step / total_steps)))
```

- `cur_step` 是全局步数（本例 0…200），`total_steps = 2*100 = 200`。
- step=0 时：`cos(0)=1` → `0.1+0.45×2 = 1.0` → lr=1e-5（满量）
- step=200 时：`cos(π)=-1` → `0.1+0.45×0 = 0.1` → lr=1e-6

所以是**从 1e-5 余弦衰减到 1e-6** 的余弦退火（注意它没有 warmup 段，也不是衰减到 0，而是留 0.1 倍地板值）。

```python
with autocast_ctx:
    res = model(input_ids, labels=labels)
    loss = res.loss + res.aux_loss
    loss = loss / args.accumulation_steps
```

**前向 + 损失计算**（对应 `model_minimind.py` 第 664–687 行）：

1. 模型算出 `logits: [16, 768, 6400]`（6400 是词表大小）。
2. 因为传了 `labels`，做 **next-token 错位**：
   ```python
   x = logits[..., :-1, :]   # 位置 0..766 的预测，去掉最后一位（它没有下一个 token）
   y = labels[..., 1:]       # 位置 1..767 的标签，去掉第一位（它是已知输入）
   loss = cross_entropy(x.view(-1, 6400), y.view(-1), ignore_index=-100)
   ```

用前面那组 id 解释错位：模型在**位置 6**（最后一个 `assistant\n` token）产生 logits，去预测 **labels[7]=50**（第一个回答 token「你」）；位置 7 预测 labels[8]=60（「好」）；位置 8 预测 labels[9]=2（`<|im_end|>`）。其余位置因为 labels 是 -100，被 `ignore_index=-100` 跳过。这样模型就学到了「在 `<|im_start|>assistant\n` 之后续写出回答」。

3. `res.aux_loss` 是 MoE 负载均衡损失，本例非 MoE，恒为 0。
4. `loss / accumulation_steps`：为梯度累积做的归一化（累积 N 步 = 等效 batch_size × N，所以要除以 N 保持梯度量级）。本例 N=1，不变。

```python
scaler.scale(loss).backward()   # 反向传播，累积梯度
```

```python
if step % args.accumulation_steps == 0:   # 累积满 N 步才更新
    scaler.unscale_(optimizer)            # fp16 时反缩放梯度（bf16 时 no-op）
    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)  # 梯度裁剪，阈值 1.0
    scaler.step(optimizer)                # 真正更新参数
    scaler.update()                       # 更新 scaler 的内部缩放系数
    optimizer.zero_grad(set_to_none=True) # 清空梯度
```

标准训练四件套：**裁剪梯度 → 更新 → 清梯度**。梯度裁剪（`clip_grad_norm_` 阈值 1.0）防止梯度爆炸。

```python
if step % args.log_interval == 0 or step == iters:   # 每 100 步或最后一步打印
    ...   # 打印 loss / logits_loss / aux_loss / lr / 预计剩余时间(eta)
```

每 100 步打印一次日志，`eta_min` 是按当前速度估算的「跑完剩余步数还需要多少分钟」。

```python
if (step % args.save_interval == 0 or step == iters) and is_main_process():
    model.eval()
    ...   # 存两个东西：
    # (1) torch.save(半精度 state_dict) 到 ../out/full_sft_768.pth   ← 最终拿来推理/推理加载
    # (2) lm_checkpoint(...) 存 resume 检查点到 ../checkpoints，含 optimizer/scaler/epoch/step/wandb_id
    model.train()
```

保存时把权重转 `half()` 省一半磁盘，且只用主进程存（避免多卡重复写）。注意存的是 `raw_model`（剥掉 DDP 外壳、剥掉 `torch.compile` 的 `_orig_mod` 包装，拿到真正参数）。

```python
del input_ids, labels, res, loss   # 及时释放显存
```

#### 尾部梯度冲刷（第 76–81 行）

```python
if last_step > start_step and last_step % args.accumulation_steps != 0:
    ...  # 如果这个 epoch 的步数不是 accumulation_steps 的整数倍，最后残留的不足 N 步梯度也要手动更新一次
```

防止「最后一个 batch 的梯度永远不更新」被白白丢弃。

---

### 阶段 9：清理（第 170–173 行）

```python
if dist.is_initialized():
    dist.barrier()          # 所有卡同步，确保都写完了再退出
    dist.destroy_process_group()
```

多卡时同步后销毁进程组，避免进程退出不同步导致挂起。

---

## 四、把整条链路串起来的「一次完整训练」

用例子总结一遍，从数据到参数更新：

1. **数据**：`"你好 / 你好！有什么可以帮你？"` 这条对话 → 渲染成 `<|im_start|>user\n你好<|im_end|>\n<|im_start|>assistant\n你好！有什么可以帮你？<|im_end|>\n` → tokenize 成 768 个 id。
2. **打标签**：扫描出 `assistant\n` 后到 `<|im_end|>` 之间的那段，`labels` 只在「你好！有什么可以帮你？<|im_end|>」这些 token 位置填真实 id，其余全 -100。
3. **前向**：16 条这样的样本堆成 `[16,768]` 进模型，得到 `logits [16,768,6400]`，错位后只对回答段算交叉熵 → `loss`。
4. **反向 + 更新**：`loss.backward()` → 梯度裁剪(1.0) → AdamW 更新全部参数（lr 按余弦从 1e-5 衰减到 1e-6）。
5. **保存**：每隔 1000 步或最后一步，把半精度权重存到 `../out/full_sft_768.pth`，并落一份带优化器状态的 resume 检查点。

**本质**：Full SFT 就是用「只对回答段求梯度」的交叉熵，把预训练模型的「续写能力」收束成「给定 `<|im_start|>assistant\n` 就生成合理回答」的对话能力，而 -100 掩码保证了它不会去死记硬背用户/系统的提示词。
