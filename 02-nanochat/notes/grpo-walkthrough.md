# nanochat GRPO 实现原理与代码详解

> 源文件：
> - `scripts/chat_rl.py` — RL 训练主脚本
> - `tasks/gsm8k.py` — GSM8K 数据集与 Reward 函数

---

## 一、背景：nanochat 的"GRPO"是什么

```python
"""
Reinforcement learning on GSM8K via "GRPO".

I put GRPO in quotes because we actually end up with something a lot
simpler and more similar to just REINFORCE:

1) Delete trust region, so there is no KL regularization to a reference model
2) We are on policy, so there's no need for PPO ratio+clip.
3) We use DAPO style normalization that is token-level, not sequence-level.
4) Instead of z-score normalization (r - mu)/sigma, only use (r - mu) as the advantage.
"""
```

作者在注释里坦承：代码里加了引号的 "GRPO"，本质上更接近 **REINFORCE with Baseline**。

与标准算法的差异：

| 组件 | 标准 GRPO | nanochat 实现 | 原因 |
|------|----------|--------------|------|
| 参考模型 + KL 惩罚 | ✅ 有 | ❌ 删除 | 完全在线采样，无需约束偏移 |
| PPO ratio + clip | ✅ 有 | ❌ 删除 | on-policy，old/new policy 相同 |
| z-score 归一化 `(r-μ)/σ` | ✅ 有 | ❌ 只用 `(r-μ)` | σ 接近 0 时会数值爆炸 |
| 优势估计来源 | 组内均值 | 组内均值 | ✅ 保留 GRPO 核心 |
| 归一化粒度 | 序列级 | **token 级（DAPO）** | 更公平，长短序列均衡 |


---

## 二、整体训练流程图

```
SFT Checkpoint（已会聊天的模型）
        ↓ load_model("sft", ...)
      model + tokenizer
        ↓
┌─────────────────────────────────────────────────────┐
│  for step in range(num_steps):                      │
│                                                     │
│  【采样阶段 Rollout】                                │
│    for example_idx（每步处理 examples_per_step 道题）│
│      q = GSM8K 数学题                               │
│      tokens = render_for_completion(q)              │
│      [o₁, o₂, ..., o₁₆] = model.generate × 16 次  │
│      [r₁, r₂, ..., r₁₆] = reward(q, oᵢ)  各 0/1  │
│      μ = mean(rᵢ)                                   │
│      Aᵢ = rᵢ - μ （优势估计）                       │
│                                                     │
│  【训练阶段 Update】                                 │
│    logp = -model(inputs, targets)  # log 概率        │
│    pg_obj = Σ logp × A / num_valid_tokens           │
│    loss = -pg_obj                                   │
│    loss.backward()                                  │
│    optimizer.step()                                 │
│                                                     │
│  【评估】每 eval_every 步评估 pass@k                 │
└─────────────────────────────────────────────────────┘
        ↓
  RL Checkpoint（数学能力更强的模型）
```


---

## 三、CLI 参数详解

```python
parser.add_argument("--num-epochs", type=int, default=1)
```
> 在 GSM8K 训练集上训练的轮数。默认 1 epoch，即每道题只被训练一次。
> `num_steps = (len(train_task) // examples_per_step) * num_epochs`
> GSM8K 训练集约 7473 题，`examples_per_step=16`，所以约 467 步。

```python
parser.add_argument("--examples-per-step", type=int, default=16)
```
> 每个优化步骤（一次 `optimizer.step()`）处理多少道题。
> 16 道题 × 16 个采样 = 256 条序列参与一次参数更新。

```python
parser.add_argument("--num-samples", type=int, default=16)
```
> 每道题采样多少个不同答案（即 GRPO 的"组大小 G"）。
> G 越大，优势估计越准确，但计算成本越高。

```python
parser.add_argument("--device-batch-size", type=int, default=8)
```
> 单次 forward 最多处理的序列数。
> 16 个采样分两次 forward（16/8=2），避免 OOM。

```python
parser.add_argument("--temperature", type=float, default=1.0)
```
> 采样温度。`temperature=1.0` 保证多样性（如果 temperature 太低，16 个答案几乎相同，优势方差为 0，梯度消失）。

```python
parser.add_argument("--init-lr-frac", type=float, default=0.05)
```
> RL 阶段学习率是 SFT 基础学习率的 5%。
> 原因：RL 容易过拟合或破坏 SFT 学到的能力，用极小学习率保守更新。


---

## 四、GSM8K 数据集与 Reward 函数（tasks/gsm8k.py）

### 数据格式

GSM8K 的每道题包含问题和解答，解答中用 `<<expr=result>>` 标记计算步骤：

```
Question: Weng earns $12 an hour for babysitting. She did 50 minutes. How much?
Answer:
Weng earns 12/60 = $<<12/60=0.2>>0.2 per minute.
Working 50 minutes, she earned 0.2 x 50 = $<<0.2*50=10>>10.
#### 10
```

### `get_example` — 解析为对话格式

```python
def get_example(self, index):
    row = self.ds[index]
    question = row['question']
    answer = row['answer']

    assistant_message_parts = []
    parts = re.split(r'(<<[^>]+>>)', answer)  # 按 <<...>> 切分
    for part in parts:
        if part.startswith('<<') and part.endswith('>>'):
            inner = part[2:-2]           # 去掉 << >>
            if '=' in inner:
                expr, result = inner.rsplit('=', 1)
            else:
                expr, result = inner, ""
            # 计算步骤 → python 工具调用
            assistant_message_parts.append({"type": "python", "text": expr})
            # 计算结果 → python 输出（mask=0，不训练）
            assistant_message_parts.append({"type": "python_output", "text": result})
        else:
            # 普通文字 → text 类型（mask=1，训练）
            assistant_message_parts.append({"type": "text", "text": part})
```

> `<<12/60=0.2>>` 被拆分为：
> - `{"type": "python", "text": "12/60"}` → mask=1，模型学会写表达式
> - `{"type": "python_output", "text": "0.2"}` → mask=0，不学计算结果（由 REPL 执行）
>
> 这个设计让模型学会"用 Python 计算"，而不是"死记硬背计算结果"。

### `extract_answer` — 从文本提取最终答案

```python
GSM_RE = re.compile(r"#### (\-?[0-9\.\,]+)")

def extract_answer(completion):
    match = GSM_RE.search(completion)
    if match:
        match_str = match.group(1).strip()
        match_str = match_str.replace(",", "")  # 去掉千位分隔符（如 1,000 → 1000）
        return match_str
    return None  # 没找到 #### 标记 → 答案无效
```

> 正则 `#### (\-?[0-9\.,]+)` 匹配 GSM8K 标准格式的最终答案行。
> 模型生成的回答也需要包含 `#### <数字>` 格式才能被评分。

### `reward` / `evaluate` — 奖励函数

```python
def evaluate(self, conversation, assistant_response):
    # 从 ground truth 提取标准答案
    last_text_part = conversation['messages'][-1]['content'][-1]['text']
    ref_num = extract_answer(last_text_part)

    # 从模型生成文本提取预测答案
    pred_num = extract_answer(assistant_response)

    # 精确匹配（字符串比较）
    is_correct = int(pred_num == ref_num)
    return is_correct

def reward(self, conversation, assistant_response):
    # RL 训练直接复用 evaluate，返回 float
    return float(self.evaluate(conversation, assistant_response))
```

> Reward 函数极其简单：**0 或 1 的二元奖励**。
> - 模型回答包含 `#### 10` 且 ground truth 也是 10 → reward=1.0
> - 答案错误或格式不对（没有 `####`） → reward=0.0
>
> 注意：字符串精确匹配，`10.0` ≠ `10`（但 GSM8K 答案通常是整数）。


---

## 五、核心：get_batch() — 采样与优势计算

这是整个 RL 的心脏，每次调用 `next(batch_iterator)` 就执行以下步骤。

### Step 1：初始化与数据分片

```python
@torch.no_grad()
def get_batch():
    assistant_end = tokenizer.encode_special("<|assistant_end|>")
    # 每个 rank 负责不同的题目（数据并行）
    rank_indices = range(ddp_rank, len(train_task), ddp_world_size)
    # rank 0: 题目 0, 8, 16, 24, ...
    # rank 1: 题目 1, 9, 17, 25, ...
    for example_idx in itertools.cycle(rank_indices):
        conversation = train_task[example_idx]
```

> - `@torch.no_grad()`：采样阶段不计算梯度，节省显存
> - `itertools.cycle`：无限循环遍历，支持多 epoch
> - `rank_indices`：DDP 数据分片，避免不同 GPU 处理相同题目

### Step 2：渲染 Prefix（去掉标准答案）

```python
        tokens = tokenizer.render_for_completion(conversation)
        prefix_length = len(tokens)
```

> `render_for_completion` 做两件事：
> 1. 去掉对话中最后一条 assistant 消息（标准答案）
> 2. 在末尾追加 `<|assistant_start|>`，让模型知道"该你回答了"
>
> 结果类似：
> ```
> <|bos|><|user_start|>Weng earns $12...<|user_end|><|assistant_start|>
> ```
> 模型从这里开始自由生成。

### Step 3：批量采样 G 个答案

```python
        generated_token_sequences = []
        masks = []
        # 分批采样，避免 OOM（16 个样本 / 8 batch_size = 2 次）
        num_sampling_steps = args.num_samples // args.device_batch_size
        for sampling_step in range(num_sampling_steps):
            # 每次采样用不同 seed，保证 16 个答案真的不同
            seed = hash((step, example_idx, sampling_step)) & 0x7FFFFFFF
            generated_token_sequences_batch, masks_batch = engine.generate_batch(
                tokens,
                num_samples=args.device_batch_size,   # 一次生成 8 个
                max_tokens=args.max_new_tokens,        # 最多 256 token
                temperature=args.temperature,           # 1.0，保证多样性
                top_k=args.top_k,                       # top-50 过滤
                seed=seed,
            )
            generated_token_sequences.extend(generated_token_sequences_batch)
            masks.extend(masks_batch)
```

> **seed 的设计**：`hash((step, example_idx, sampling_step))` 由三个维度决定：
> - `step`：不同训练步骤得到不同种子
> - `example_idx`：不同题目得到不同种子
> - `sampling_step`：同一道题的两批次采样（0~7 和 8~15）得到不同种子
>
> **mask 的含义**：Engine 返回的 mask，prompt 部分为 0，生成部分为 1。
> 后续这个 mask 用来确定哪些 token 参与 loss 计算。

### Step 4：计算每个答案的 Reward

```python
        rewards = []
        for sample_tokens in generated_token_sequences:
            generated_tokens = sample_tokens[prefix_length:]  # 只取生成部分
            generated_text = tokenizer.decode(generated_tokens)
            reward = train_task.reward(conversation, generated_text)  # 0.0 或 1.0
            rewards.append(reward)
```

> 通过 `prefix_length` 精确切出模型生成的部分（去掉 prompt）。
> 对 16 个生成结果分别调用 reward 函数，得到 16 个 0/1 分数。

### Step 5：对齐序列长度（Padding）

```python
        max_length = max(len(seq) for seq in generated_token_sequences)
        # 用 assistant_end token 填充短序列
        padded_generated_token_sequences = [
            seq + [assistant_end] * (max_length - len(seq))
            for seq in generated_token_sequences
        ]
        padded_masks = [
            mask + [0] * (max_length - len(mask))
            for mask in masks
        ]
```

> 16 个答案长度不同，需要对齐才能组成矩阵（batch 运算要求形状一致）。
> - 用 `<|assistant_end|>` token 填充：语义上是"回答结束"，不影响模型对内容的理解
> - 填充位置的 mask=0：这些 padding token 不参与 loss 计算

### Step 6：构建训练张量

```python
        ids = torch.tensor(padded_generated_token_sequences, dtype=torch.long, device=device)
        # (num_samples, max_length) = (16, ~200)

        mask_ids = torch.tensor(padded_masks, dtype=torch.long, device=device)

        # 自回归 LM 的标准输入/目标构造
        inputs = ids[:, :-1]    # 去掉最后一个 token
        targets = ids[:, 1:].clone()  # 去掉第一个 token（右移一位）

        # 将不参与训练的位置的 target 设为 -1（cross_entropy 的 ignore_index）
        targets[mask_ids[:, 1:] == 0] = -1
        # mask_ids[:, 1:] 对应 targets 的对齐位置
        # mask=0 的位置（prompt + padding）→ target=-1 → 不参与 loss
```

> **关键设计**：Engine 生成时 mask=0 的位置包括：
> - Prompt 部分（题目和 user/assistant 特殊 token）
> - 工具调用结果（`python_output` 部分）
> - Padding 部分
>
> 这保证了模型只在"自己生成的文字"上被训练，不会试图"学会"用户的问题或计算器的输出。

### Step 7：计算优势（GRPO 核心）

```python
        rewards = torch.tensor(rewards, dtype=torch.float, device=device)
        # shape: (16,)，例如 [1.0, 0.0, 0.0, 1.0, 1.0, 0.0, ...]

        # GRPO 优势 = 组内相对得分（简化版，不除以标准差）
        mu = rewards.mean()         # 组内平均分，例如 0.375（6/16 正确）
        advantages = rewards - mu   # 例如 [0.625, -0.375, -0.375, 0.625, ...]
```

> **直觉**：
> - 正确答案（reward=1）：优势 = 1 - 0.375 = +0.625 → 增大生成概率
> - 错误答案（reward=0）：优势 = 0 - 0.375 = -0.375 → 减小生成概率
>
> **为什么不除以标准差 σ？**
> 如果所有答案都错（rewards 全 0）或都对（rewards 全 1）：
> - σ = 0（零方差）
> - (r-μ)/σ = 0/0，数值崩溃
>
> nanochat 的做法：只减均值，σ=0 时所有优势=0，梯度自然为 0（模型不更新），安全。

```python
        # yield 给训练循环使用
        yield generated_token_sequences, inputs, targets, rewards, advantages
```


---

## 六、训练主循环详解

### 优化器初始化

```python
optimizer = model.setup_optimizer(
    unembedding_lr=args.unembedding_lr,   # 0.004
    embedding_lr=args.embedding_lr,       # 0.2
    matrix_lr=args.matrix_lr,             # 0.02
    weight_decay=args.weight_decay,       # 0.0
)

# RL 阶段用极小学习率：SFT 基础 LR 的 5%
for group in optimizer.param_groups:
    group["lr"] = group["lr"] * args.init_lr_frac   # × 0.05
    group["initial_lr"] = group["lr"]
```

> `init_lr_frac=0.05` 是 RL 训练的关键参数：
> - SFT 后的模型已经能流畅对话，RL 只需微调数学推理能力
> - 学习率太大会破坏 SFT 学到的格式和语言能力（灾难性遗忘）
> - 0.05 × (SFT LR) 非常保守，确保 RL 只做小幅度调整

### 学习率调度

```python
def get_lr_multiplier(it):
    lrm = 1.0 - it / num_steps  # 线性衰减到 0
    return lrm
```

> 从 1.0 线性降到 0，整个 RL 训练过程学习率单调递减。
> 与 SFT 的三段式（warmup + constant + warmdown）不同，RL 更简单直接。

### 主循环：每步的完整流程

```python
batch_iterator = get_batch()  # 初始化生成器

for step in range(num_steps):
```

#### 6.1 评估阶段（每 eval_every 步）

```python
    if step % args.eval_every == 0:  # 默认每 60 步评估一次
        model.eval()
        passk = torch.zeros(args.device_batch_size, device=device)  # pass@1 ~ pass@8

        records_iter = run_gsm8k_eval(
            val_task, tokenizer, engine,
            num_samples=args.device_batch_size,  # 每题生成 8 个答案
            max_examples=args.eval_examples,     # 只评估 400 道题（加速）
            temperature=1.0,                     # 温度采样（评估多样性）
        )
        records = list(records_iter)

        # 计算 pass@k（k=1~8）
        for k in range(1, args.device_batch_size + 1):
            # pass@k = 至少有 k 个答案里有 1 个正确的题目比例
            passk[k - 1] = sum(
                any(o["is_correct"] for o in r["outcomes"][:k])
                for r in records
            )
```

> **pass@k 指标解读**：
> - `pass@1`：每次只生成 1 个答案，能答对的比例（贪心解码能力）
> - `pass@8`：生成 8 个答案，至少 1 个对的比例（探索能力上限）
>
> 如果 pass@1 提高但 pass@8 不变，说明模型在"确定性"上进步（更准）。
> 如果 pass@8 也提高，说明模型覆盖到了更多样的正确推理路径。

```python
        # DDP 聚合各 rank 的评估结果
        if ddp:
            dist.all_reduce(num_records, op=dist.ReduceOp.SUM)
            dist.all_reduce(passk, op=dist.ReduceOp.SUM)
        passk = passk / num_records.item()
```

> 分布式评估：各 rank 分别评估不同题目，`all_reduce SUM` 后除以总题数，得到全局 pass@k。

#### 6.2 前向传播与梯度计算

```python
    for example_step in range(examples_per_rank):   # 每步处理多道题
        sequences_all, inputs_all, targets_all, rewards_all, advantages_all = next(batch_iterator)

        model.train()  # 切换回训练模式

        # 分批前向（避免 OOM）
        num_passes = inputs_all.size(0) // args.device_batch_size
        # inputs_all.size(0) = num_samples = 16
        # device_batch_size = 8
        # num_passes = 2（分两次 forward）

        for pass_idx in range(num_passes):
            b0 = pass_idx * args.device_batch_size
            b1 = (pass_idx + 1) * args.device_batch_size

            inputs     = inputs_all[b0:b1]      # (8, T)
            targets    = targets_all[b0:b1]     # (8, T)，-1 是 ignore
            advantages = advantages_all[b0:b1]  # (8,)
```

#### 6.3 计算 log 概率（策略梯度核心）

```python
            # model 正常返回 NLL（负对数似然），取负得到 log 概率
            logp = -model(inputs, targets, loss_reduction='none').view_as(inputs)
            # logp shape: (8, T)
            # logp[i, t] = log P(target[i,t] | context)
            # ignore_index=-1 保证 prompt/padding 位置的 loss=0（logp 对应位置也是 0）
```

> `loss_reduction='none'` 让 model 返回每个 token 位置的独立 loss，而不是整体平均。
> 取负数转为 log 概率（NLL = -log P → log P = -NLL）。

#### 6.4 策略梯度目标（REINFORCE 核心公式）

```python
            # 策略梯度目标：E[log π(a|s) × A]
            pg_obj = (logp * advantages.unsqueeze(-1)).sum()
            # logp:            (8, T)
            # advantages:      (8,)
            # advantages.unsqueeze(-1): (8, 1) 广播到 (8, T)
            # 乘积：           (8, T)  每个 token 的梯度贡献
            # .sum()：         标量，所有 token 的梯度之和
```

> **数学含义**：
> ```
> pg_obj = Σᵢ Σₜ log π(oᵢ,ₜ | q, oᵢ,<ₜ) × Aᵢ
>
> 其中：
>   i = 序列索引（0~7）
>   t = token 位置
>   Aᵢ = 第 i 个答案的优势（标量，同一序列所有 token 共享）
>   π = 当前模型的生成策略
> ```
>
> 当 Aᵢ > 0（好答案）：目标要求 log π 尽量大 → 增大该答案每个 token 的生成概率
> 当 Aᵢ < 0（坏答案）：目标要求 log π 尽量小 → 减小该答案每个 token 的生成概率

#### 6.5 DAPO 风格 token 级归一化

```python
            # 有效 token 数（排除 prompt、padding 位置的 -1）
            num_valid = (targets >= 0).sum().clamp(min=1)

            # 除以有效 token 数 + passes 数 + examples 数
            pg_obj = pg_obj / (num_valid * num_passes * examples_per_rank)
```

> **为什么要归一化？**
>
> 不归一化：长序列（200 token）的梯度贡献远大于短序列（50 token），不公平。
>
> Token 级归一化（DAPO）：每个有效 token 对梯度的贡献权重相同。
>
> `num_passes * examples_per_rank`：梯度累积的规范化，确保多次 backward 的梯度之和等效于一次完整更新。

#### 6.6 计算 loss 并反向传播

```python
            # 最大化策略梯度目标 = 最小化其负值
            loss = -pg_obj
            loss.backward()   # 累积梯度（不 zero_grad）
```

> 梯度在所有 `pass_idx` 和 `example_step` 上累积，最后统一 `optimizer.step()`。
> 这等效于对所有 `examples_per_rank × num_passes × device_batch_size` 条序列做批量梯度更新。

#### 6.7 更新参数

```python
    lrm = get_lr_multiplier(step)
    for group in optimizer.param_groups:
        group["lr"] = group["initial_lr"] * lrm  # 线性衰减

    optimizer.step()
    model.zero_grad(set_to_none=True)  # 清空梯度
```

> 注意：RL 阶段没有 Muon 动量 warmup（SFT 有）。
> 学习率线性衰减，训练结束时 LR → 0，避免最后阶段的大幅更新。

### 保存 Checkpoint

```python
    if master_process and (step % args.save_every == 0 or step == num_steps - 1):
        save_checkpoint(
            checkpoint_dir, step,
            model.state_dict(),
            None,  # 不保存优化器状态（节省空间）
            {"model_config": model_config_kwargs},
        )
```

> RL 阶段**不保存优化器状态**，因为：
> - RL 训练时间短（1 epoch），不太需要断点续训
> - 优化器状态文件很大，RL 阶段节省空间更重要


---

## 七、完整数学推导

### REINFORCE 基础

策略梯度定理：

$$\nabla_\theta J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta}\left[\sum_t \nabla_\theta \log \pi_\theta(a_t|s_t) \cdot R(\tau)\right]$$

其中 $R(\tau)$ 是轨迹的总 reward。

### GRPO 的改进：组内基线

用同组答案的平均 reward 作为基线（Baseline），减少方差：

$$\nabla_\theta J(\theta) = \mathbb{E}\left[\sum_t \nabla_\theta \log \pi_\theta(a_t|s_t) \cdot (R_i - \mu)\right]$$

其中 $\mu = \frac{1}{G}\sum_{i=1}^G R_i$

**基线的意义**：
- 无基线：每次更新方向取决于绝对 reward，方差高
- 有基线：只关注"相对于平均水平好多少"，方差低，收敛快

### nanochat 的实现（代码中逐步对应）

```
题目 q
    ↓ render_for_completion()
prefix tokens（含 <|assistant_start|>）
    ↓ engine.generate_batch(G=16, temperature=1.0)
{o₁, o₂, ..., o₁₆}           # G 个不同答案
    ↓ reward() × G
{r₁=1, r₂=0, ..., r₁₆=0}     # 二元奖励
    ↓ μ = mean(rᵢ)
μ = 0.375                      # 16 个答案中 6 个正确
    ↓ Aᵢ = rᵢ - μ
{A₁=+0.625, A₂=-0.375, ...}   # 优势估计
    ↓ logp = -model(inputs, targets)
logp[i,t] = log π(oᵢ,ₜ|q, oᵢ,<t)  # 每个 token 的对数概率
    ↓ pg_obj = Σᵢ Σₜ logpᵢ,ₜ × Aᵢ / num_valid
    ↓ loss = -pg_obj
    ↓ loss.backward() + optimizer.step()
```

### 梯度方向分析

对于正确答案（Aᵢ = +0.625）：
```
∂loss/∂θ = -Aᵢ × ∂logp/∂θ = -0.625 × ∂logp/∂θ

由于 ∂logp/∂θ 指向增大 logp 的方向，
-0.625 × 该方向 = 指向增大 logp 的方向
→ 参数更新后，模型生成该答案的概率增大 ✅
```

对于错误答案（Aᵢ = -0.375）：
```
∂loss/∂θ = -(-0.375) × ∂logp/∂θ = +0.375 × ∂logp/∂θ

→ 参数更新后，模型生成该答案的概率减小 ✅
```

当所有答案都错（全 0）或都对（全 1）：
```
μ = 0（或 1）
Aᵢ = 0 - 0 = 0（或 1 - 1 = 0）
pg_obj = Σ logp × 0 = 0
loss = 0
loss.backward() → 梯度全为 0 → 参数不更新 ✅
```
这是 nanochat 不使用 z-score 归一化的原因：当 σ=0 时自动退化为零梯度，而非 0/0 崩溃。


---

## 八、分布式训练细节（DDP）

### 数据分片

```python
rank_indices = range(ddp_rank, len(train_task), ddp_world_size)
# 8 张卡：
# rank 0: 题目 0, 8, 16, 24, ...
# rank 1: 题目 1, 9, 17, 25, ...
# ...
# rank 7: 题目 7, 15, 23, 31, ...
```

不同 GPU 处理不同题目，避免重复计算。

### 梯度同步

```python
loss.backward()
# DDP 自动 all-reduce 梯度（各 rank 梯度求平均）
# 不需要显式调用 dist.all_reduce
```

DDP 封装的模型在 `backward()` 时自动同步所有参数的梯度，保证各 GPU 的参数更新一致。

### 评估聚合

```python
num_records = torch.tensor(len(records), dtype=torch.long, device=device)
if ddp:
    dist.all_reduce(num_records, op=dist.ReduceOp.SUM)  # 统计总评估题数
    dist.all_reduce(passk, op=dist.ReduceOp.SUM)        # 各 rank 答对数求和
passk = passk / num_records.item()                      # 除以总题数得到比率
```

### last_step 同步

chat_rl.py 没有 chat_sft.py 那样的 `last_step` 分布式同步，因为 RL 用固定的 `num_steps` 步数控制训练终止，不存在数据耗尽的竞态问题。

---

## 九、run_gsm8k_eval — 评估函数

```python
def run_gsm8k_eval(task, tokenizer, engine,
    max_examples=None,    # 评估多少道题（默认 400）
    num_samples=1,        # 每题生成几个答案（默认 device_batch_size=8）
    max_completion_tokens=256,
    temperature=0.0,      # 默认贪心（训练时用 1.0 是为了测多样性）
    top_k=50
):
    max_examples = min(max_examples, len(task)) if max_examples is not None else len(task)

    for idx in range(ddp_rank, max_examples, ddp_world_size):
        conversation = task[idx]
        tokens = tokenizer.render_for_completion(conversation)
        prefix_length = len(tokens)

        generated_token_sequences, masks = engine.generate_batch(
            tokens,
            num_samples=num_samples,
            max_tokens=max_completion_tokens,
            temperature=temperature,
            top_k=top_k
        )

        outcomes = []
        for sample_tokens in generated_token_sequences:
            generated_tokens = sample_tokens[prefix_length:]
            generated_text = tokenizer.decode(generated_tokens)
            is_correct = task.evaluate(conversation, generated_text)
            outcomes.append({"is_correct": is_correct})

        yield {"idx": idx, "outcomes": outcomes}
```

> 使用 `yield` 而非返回列表：
> - 评估可能很慢（400 题 × 8 个答案）
> - 流式输出允许调用方边收集边处理
>
> 评估时 `temperature=1.0`（而非 0），目的是测 pass@k 而非贪心准确率：
> - `temperature=0` 的 pass@1 = 贪心准确率（每次结果相同）
> - `temperature=1.0` 的 pass@8 = 模型有没有能力找到正确路径（即使不是最可能路径）

---

## 十、关键超参数的设计逻辑

| 参数 | 值 | 设计原因 |
|------|-----|---------|
| `num_samples=16` | 每题采样 16 个 | G 越大优势估计越准，但 16 已是显存上限 |
| `temperature=1.0` | 高随机性 | 保证 16 个答案足够多样，避免全部相同导致 Aᵢ=0 |
| `init_lr_frac=0.05` | LR 极小 | 避免 RL 破坏 SFT 的语言能力 |
| `max_new_tokens=256` | 最多 256 token | GSM8K 答案通常较短；防止无限生成 |
| `num_epochs=1` | 只训练 1 轮 | 防止对有限 GSM8K 数据过拟合 |
| `eval_every=60` | 每 60 步评估 | 训练总步数约 467，每 13% 评估一次 |
| `weight_decay=0.0` | 无权重衰减 | RL 时间短，不需要正则化 |

---

## 十一、局限性与可改进方向

### 当前实现的局限

```
1. 只用 GSM8K（数学题）
   → 其他能力（代码、推理、对话）没有 RL 优化

2. 二元 reward（0/1）
   → 无法区分"差一点的错误"和"完全跑偏的错误"
   → 无法奖励部分正确的推理过程

3. 不保存优化器状态
   → 无法续训，每次 RL 必须从头开始

4. G=16 的固定组大小
   → 对某些题目（极难或极易），所有答案要么全对要么全错，
     优势 Aᵢ=0，白费计算资源
```

### 可能的改进方向

```
1. 过程奖励（Process Reward Model, PRM）
   → 对每个推理步骤单独打分，而不是只看最终答案

2. 更多任务
   → HumanEval（代码正确性）
   → ARC（多选题正确性）
   → 自定义拼写/逻辑任务

3. 动态组大小
   → 全对/全错时自动跳过，节省计算

4. Curriculum Learning
   → 从简单题开始，逐渐加难，提高学习效率
```

---

## 十二、整体代码执行时序

```
python -m scripts.chat_rl
    │
    ├─ 加载 SFT checkpoint (load_model)
    ├─ 初始化 Engine（KV Cache 推理引擎）
    ├─ 初始化 GSM8K 数据集
    ├─ 初始化 MuonAdamW 优化器（LR × 0.05）
    │
    ├─ step 0: 评估 pass@k（初始基线）
    │
    └─ for step in range(num_steps=467):
         │
         ├─ [采样] get_batch() → next()
         │    ├─ 取题目 q（当前 rank 负责的题）
         │    ├─ render_for_completion → prefix tokens
         │    ├─ generate_batch(G=16) → 16 个答案
         │    ├─ reward × 16 → [1,0,0,1,...]
         │    ├─ advantages = rewards - mean
         │    └─ yield (inputs, targets, advantages)
         │
         ├─ [训练] examples_per_rank=2 道题 × 2 passes
         │    ├─ logp = -model(inputs, targets, reduction='none')
         │    ├─ pg_obj = Σ(logp × A) / num_valid
         │    ├─ loss = -pg_obj
         │    └─ loss.backward()（梯度累积）
         │
         ├─ optimizer.step()（参数更新）
         ├─ model.zero_grad()
         │
         ├─ 每 60 步: 评估 pass@k，记录 wandb
         └─ 每 60 步: 保存 checkpoint
```

