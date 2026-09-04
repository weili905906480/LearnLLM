# train_dpo.py 详解

`train_dpo.py` 做的是 **DPO（Direct Preference Optimization，直接偏好优化）**，属于 RLHF 的对齐阶段。目标一句话概括：

> 让模型（**策略模型 policy**）输出「人类更喜欢的回答（chosen）」，而不是「被嫌弃的回答（rejected）」，同时别跑偏太远（用冻结的**参考模型 ref** 当锚点）。

## 一、DPO 与 SFT 的区别

| | SFT（`train_full_sft.py`） | DPO（本脚本） |
|---|---|---|
| 数据 | 单条「问题 + 标准答案」 | 成对「问题 + 好回答(chosen) + 坏回答(rejected)」 |
| 目标 | 最大化标准答案的似然 | 拉开 chosen 与 rejected 的**相对**概率差距 |
| 是否有参考模型 | 无 | 有（ref_model，冻结，防止遗忘） |
| 学习率 | 较大（如 4e-7） | 很小（默认 4e-8，避免灾难性遗忘） |

---

## 二、整体流程一览

```
主函数 __main__
 ├─ 1. 初始化分布式 + 随机种子
 ├─ 2. 建目录 / 构造 MiniMindConfig / 若续训则加载 ckp
 ├─ 3. 混合精度上下文（bf16 或 fp16 + GradScaler）
 ├─ 4. 可选 wandb（代码里 import swanlab as wandb）
 ├─ 5. 初始化 policy 模型 + 冻结的 ref 模型（都从同一权重 from_weight 加载）
 │     构造 DPODataset / DistributedSampler / optimizer / GradScaler
 ├─ 6. 若续训，恢复 model/optimizer/scaler/epoch/step
 ├─ 7. 可选 torch.compile + DDP 包装
 ├─ 8. 逐 epoch 训练（train_epoch）
 └─ 9. 销毁分布式进程组
```

核心计算都在两个函数里：`logits_to_log_probs`（把 logits 变成每个 token 的对数概率）和 `dpo_loss`（DPO 损失公式）。

---

## 三、`logits_to_log_probs` —— 把 logits 转成「真实 token 的对数概率」

```python
def logits_to_log_probs(logits, labels):
    log_probs = F.log_softmax(logits, dim=2)      # 对词表维度做 log_softmax
    log_probs_per_token = torch.gather(log_probs, dim=2, index=labels.unsqueeze(2)).squeeze(-1)
    return log_probs_per_token
```

- `logits` 形状 `(batch, seq_len, vocab_size)`，是模型对每个位置、每个词表 token 打的原始分数。
- `F.log_softmax(logits, dim=2)`：在最后一个维度（词表）上归一化并取对数，得到每个位置每个 token 的 **log 概率**。
- `torch.gather(...)`：只挑出 `labels` 指定的那个 token 的 log 概率，得到 `(batch, seq_len)`。

**具体例子**（假设 `vocab_size=5`，词表 id：0=pad, 1=`<|im_start|>`, 2=`<|im_end|>`, 3=`好`, 4=`你`）：

```text
某位置 t 的 logits:            [0.1, -1.0, 2.5, 1.8, -0.3]
log_softmax 后（每项都 < 0，exp 后和为 1）:
                              [-2.31, -3.41, 0.09, -0.61, -2.71]
labels[t] = 3（真实 token 是「好」）
gather 取出第 3 号 → log_probs_per_token[t] = -0.61
```

也就是说，这一步得到「模型认为每个位置该输出那个真实 token 的对数概率」。

---

## 四、`dpo_loss` —— DPO 的核心损失（重点 + 数值例子）

```python
def dpo_loss(ref_log_probs, policy_log_probs, mask, beta):
    ref_log_probs    = (ref_log_probs    * mask).sum(dim=1)   # 每个样本在「回答段」上的对数概率和
    policy_log_probs = (policy_log_probs * mask).sum(dim=1)

    batch_size = ref_log_probs.shape[0]                       # 注意：这是 chosen+rejected 拼接后的总大小
    chosen_ref_log_probs    = ref_log_probs[:batch_size // 2]
    reject_ref_log_probs    = ref_log_probs[batch_size // 2:]
    chosen_policy_log_probs = policy_log_probs[:batch_size // 2]
    reject_policy_log_probs = policy_log_probs[batch_size // 2:]

    pi_logratios  = chosen_policy_log_probs - reject_policy_log_probs
    ref_logratios = chosen_ref_log_probs    - reject_ref_log_probs
    logits = pi_logratios - ref_logratios
    loss = -F.logsigmoid(beta * logits)
    return loss.mean()
```

### 1. 掩码的作用：只统计「回答部分」

`mask` 是 `DPODataset.generate_loss_mask` 生成的，**只有 assistant 回答段（含 `<|im_end|>`）为 1，其余（system/user 提示、padding）为 0**。

`(log_probs * mask).sum(dim=1)` 就是把一条序列中「回答段每个 token 的 log 概率」加起来，得到整段回答的对数概率 `log P(回答 | 上下文)`。用户提示部分不参与，这样模型只需要学「怎么答」，不用学「用户怎么问」。

### 2. 前一半 chosen、后一半 rejected 的约定

关键点：训练循环里把 chosen 和 rejected **按行拼接**了：

```python
x = torch.cat([x_chosen, x_rejected], dim=0)   # 前 B 行是 chosen，后 B 行是 rejected
```

所以 `dpo_loss` 里 `batch_size // 2` 正好切开：`[:B]` 是 chosen，`[B:]` 是 rejected。这是 DPO 实现里最常见的「一半一半」打包方式。

### 3. 数学含义

设 `π_θ` = policy，`π_ref` = reference，定义：

```
pi_logratios  = log π_θ(chosen)   − log π_θ(rejected)
ref_logratios = log π_ref(chosen) − log π_ref(rejected)
logits        = pi_logratios − ref_logratios
             = log [ π_θ(chosen)/π_θ(rejected) ] − log [ π_ref(chosen)/π_ref(rejected) ]
```

最终损失就是标准 DPO 公式：

```
loss = −log σ( β · logits )
```

- `β`（代码里 `--beta`，默认 0.15）：控制「奖励强度 / 偏离参考模型的程度」。β 越大越强调偏好差距，β 越小越保守、越贴近 ref。
- 当模型给 chosen 的概率相对 rejected 越高（且相对 ref 涨得越多），`logits` 越大，`σ(·)→1`，loss→0，训练目标达成。

### 4. 完整数值例子

假设 `batch_size=2`（1 个 chosen + 1 个 rejected），每条序列求和后得到：

```text
log π_θ(chosen)    = -3.0      （policy 对好回答的对数概率）
log π_θ(rejected)  = -5.0      （policy 对坏回答的对数概率）
log π_ref(chosen)  = -4.0
log π_ref(rejected)= -4.5
```

逐步计算：

```text
pi_logratios  = (-3.0) - (-5.0) = 2.0     # policy 明显更偏好 chosen
ref_logratios = (-4.0) - (-4.5) = 0.5     # ref 只稍微偏好 chosen
logits = 2.0 - 0.5 = 1.5

beta = 0.15
β·logits = 0.15 × 1.5 = 0.225

σ(0.225) = 1/(1+e^-0.225) ≈ 0.556
loss = -log(0.556) ≈ 0.587
```

直觉：policy 已经在「正确方向」上跑赢了 ref（pi 的偏好差 2.0 > ref 的 0.5），所以损失不大（0.587）。如果模型反过来更偏好 rejected（比如 `π_θ(rejected) > π_θ(chosen)`，`pi_logratios` 变成负的），`β·logits` 会是很小的负数，`σ(·)` 很小，`-log` 会变得很大，狠狠惩罚模型。

---

## 五、`train_epoch` —— 训练循环逐段解析

### 1. 取数据、拼接 chosen/rejected

```python
x = torch.cat([x_chosen, x_rejected], dim=0)   # (2B, seq_len)
y = torch.cat([y_chosen, y_rejected], dim=0)
mask = torch.cat([mask_chosen, mask_rejected], dim=0)
```

回顾 `DPODataset.__getitem__` 里每个字段是怎么来的（对每条 chosen/rejected 独立做）：

```text
input_ids   = [<|im_start|>user 你好 <|im_end|> <|im_start|>assistant 回答 <|im_end|> 0 0 ...]
loss_mask   = [0 0 0 0 0  0 0 0 0  0  1 1 1 1  0 0 ...]   # 只有 assistant 回答段=1
x      = input_ids[:-1]    # 输入：去掉最后一个 token
y      = input_ids[1:]     # 标签：整体右移一位（标准 next-token 预测）
mask   = loss_mask[1:]     # 掩码同步右移，和 y 对齐
```

**为什么 `x=input_ids[:-1]`、`y=input_ids[1:]`？** 语言模型在每个位置 `t` 拿 `x[t]` 预测下一个 token `y[t]`，所以 x 和 y 天然错一位。`mask` 也跟着错位，保证 `mask[t]=1` 时 `y[t]` 是回答 token。

**为什么这里不像 SFT 那样把 labels 里非回答段填 -100？** DPO 不用 CrossEntropyLoss，而是手动用 `mask` 把非回答段的 log prob 乘 0 排除掉（`log_probs * mask`），效果等价。

### 2. 学习率调度

```python
lr = get_lr(epoch * iters + step, args.epochs * iters, args.learning_rate)
```

`get_lr` 是**余弦退火 + 预热**：`lr * (0.1 + 0.45*(1 + cos(π·t/T)))`。起 t=0 时系数 = 0.1+0.45·2 = 1.0，结尾 t=T 时 = 0.1+0.45·0 = 0.1，即从初始 lr 平滑降到 0.1×lr。DPO 的初始 lr 极小（4e-8），进一步防遗忘。

### 3. 前向：ref 冻结、policy 可训练

```python
with autocast_ctx:
    with torch.no_grad():
        ref_outputs = ref_model(x)
        ref_logits = ref_outputs.logits
    ref_log_probs = logits_to_log_probs(ref_logits, y)

    outputs = model(x)
    logits = outputs.logits
    policy_log_probs = logits_to_log_probs(logits, y)

    dpo_loss_val = dpo_loss(ref_log_probs, policy_log_probs, mask, beta=beta)
    loss = dpo_loss_val + outputs.aux_loss   # aux_loss 是 MoE 的负载均衡损失，非 MoE 时为 0
    loss = loss / args.accumulation_steps
```

要点：
- **ref 全程 `torch.no_grad()`**：参考模型只出 logits，不回传梯度，作为「不动的标尺」。
- **同一个 x 同时喂给两个模型**：因为 x 已经是 chosen+rejected 拼接好的，ref 和 policy 拿到完全相同的输入，logits 逐位置可比。
- `outputs.aux_loss`：MoE（混合专家）架构才有的辅助损失，普通 dense 模型是 0。

### 4. 反向传播 + 梯度累积 + 梯度裁剪

```python
scaler.scale(loss).backward()

if step % args.accumulation_steps == 0:
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)  # 裁剪到 1.0
    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad(set_to_none=True)
```

- `scaler`（GradScaler）只在 fp16 时才真正生效（`enabled=(args.dtype=='float16')`）；bf16 时 `loss/accumulation_steps` 已提前除，`scaler.scale` 实际不缩放。
- **梯度累积**：每 `accumulation_steps` 步才真正更新一次权重，等效于放大了 batch size（默认 1，即不累积）。`loss = loss / accumulation_steps` 是提前把累积的梯度做了平均。
- `clip_grad_norm_` 防止 DPO 里个别样本梯度爆炸。

### 5. 日志

打印/记录 epoch、step、总 loss、dpo_loss、aux_loss、lr、ETA。注意 `current_loss = loss.item() * accumulation_steps` 把之前除以累积步数的 loss 还原成「真实一步损失」。

### 6. 保存权重

保存两样东西：
1. **纯权重** `{save_dir}/{save_weight}_{hidden_size}{moe}.pth`（如 `../out/dpo_768.pth`），转成 fp16 存 CPU 省空间。
2. **完整 resume 状态** `../checkpoints/dpo_768_resume.pth`（通过 `lm_checkpoint`），包含 model/optimizer/epoch/step/wandb_id，用于续训。

这里有个细节：解 DDP 包装用了 `getattr(raw_model, '_orig_mod', raw_model)`——因为 `torch.compile` 会把模型包成 `_orig_mod` 属性，多包一层保险。

### 7. 显存清理

逐个 `del` 中间变量，DPO 一次要同时存 ref + policy 两份 logits，显存紧张，及时释放很关键。

### 8. 尾部残余梯度处理

如果 epoch 结束时还有不足 `accumulation_steps` 的未更新梯度，这里补做最后一次 step，避免丢梯度。

---

## 六、主函数里几个值得注意的设计点

### 1. `__package__ = "trainer"` + 手动加 sys.path

让脚本能作为 `trainer.train_dpo` 被导入，同时把上一级目录（`minimind/`）加进 `sys.path`，这样 `from model.model_minimind import ...`、`from dataset.lm_dataset import ...` 能正常工作。

### 2. `import datasets` 的注释

`# Windows pyarrow/torch DLL conflict workaround (issue #771)` —— 这是 Windows 上一个已知的 DLL 加载顺序坑：先 import `datasets`（会带 pyarrow）能规避与 torch 的 DLL 冲突。属于环境兼容性 hack。

### 3. 模型初始化：policy 和 ref 从同一个起点出发

```python
model, tokenizer = init_model(lm_config, args.from_weight, device=args.device)   # policy
ref_model, _ = init_model(lm_config, args.from_weight, device=args.device)       # ref
ref_model.eval()
ref_model.requires_grad_(False)
```

两者都从 `from_weight`（默认 `full_sft`，即 SFT 后的权重）加载，**初始完全相同**。随后 policy 参与训练更新，ref 永远冻结在 SFT 状态——这就是 DPO 里「参考模型」的角色，防止 policy 在讨好偏好的同时彻底忘掉 SFT 学到的能力。

### 4. 续训逻辑

`--from_resume 1` 时先 `lm_checkpoint(...)` 读 `_resume.pth`，恢复 model/optimizer/scaler/epoch/step，然后：

```python
skip = start_step if (epoch == start_epoch and start_step > 0) else 0
batch_sampler = SkipBatchSampler(train_sampler or indices, args.batch_size, skip)
```

`SkipBatchSampler` 会跳过前面 `skip` 个 batch，实现「断点续训」——从上次中断的 step 精确接上，而不是从头再训一遍（否则会重复吃数据）。

### 5. 每个 epoch 重新 shuffle

```python
setup_seed(42 + epoch)
indices = torch.randperm(len(train_ds)).tolist()
```

用「固定种子 + epoch」保证每次 run 的数据顺序可复现，同时不同 epoch 之间顺序不同。

---

## 七、一句话总结

`train_dpo.py` 把成对的偏好数据（chosen/rejected）拼接成 2 倍 batch，同时喂给**可训练的 policy** 和**冻结的 ref**，在回答段上算对数概率，套 DPO 公式 `-log σ(β·logits)` 拉大「chosen 相对 rejected 的概率优势（且相对 ref 也变好）」，配以极小学习率、梯度累积/裁剪、混合精度、断点续训，实现低成本、免 reward model 的偏好对齐。
