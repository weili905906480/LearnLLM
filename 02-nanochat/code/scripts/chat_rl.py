"""
在 GSM8K（小学数学应用题）数据集上用 "GRPO" 进行强化学习微调。

"GRPO" 加引号的原因：本实现实际上简化成了更接近 REINFORCE 的方法，
做了以下 4 个简化：

1) 删除信任域（Trust Region）：没有对参考模型的 KL 散度正则化
2) 完全在策略（On-Policy）：不需要 PPO 的 ratio + clip 机制
3) 使用 DAPO 风格的 Token 级别归一化（而非序列级别）
4) 优势函数仅使用 (r - μ) 而非 z-score 归一化 (r - μ)/σ

核心思想：
- 对每个问题生成多个候选回答（num_samples 个）
- 根据 GSM8K 的 ground truth 答案计算每个回答的 reward（0 或 1）
- 用这些 reward 计算优势函数（advantage）
- 用策略梯度（Policy Gradient）方法强化高 reward 回答的生成概率

运行方式：
单 GPU:
python -m scripts.chat_rl

8 GPU 分布式训练:
torchrun --standalone --nproc_per_node=8 -m scripts.chat_rl -- --run=default
"""

import argparse
import os
import itertools
import wandb
import torch
import torch.distributed as dist
from nanochat.common import compute_init, compute_cleanup, print0, get_base_dir, DummyWandb, autodetect_device_type
from nanochat.checkpoint_manager import save_checkpoint, load_model
from nanochat.engine import Engine
from tasks.gsm8k import GSM8K

# =============================================================================
# 第一部分：命令行参数解析
# =============================================================================

parser = argparse.ArgumentParser(description="Reinforcement learning on GSM8K")

# --- 日志 ---
parser.add_argument("--run", type=str, default="dummy", help="wandb 运行名称（'dummy' 表示禁用 wandb 日志）")

# --- 运行时环境 ---
parser.add_argument("--device-type", type=str, default="", help="计算设备类型：cuda|cpu|mps（空字符串 = 自动检测）")

# --- 模型加载 ---
# 从 SFT（监督微调）阶段产出的检查点加载模型
parser.add_argument("--model-tag", type=str, default=None, help="要加载的模型标签")
parser.add_argument("--model-step", type=int, default=None, help="要加载的模型步数（None = 加载最终检查点）")

# --- 训练步数与数据量 ---
parser.add_argument("--num-epochs", type=int, default=1, help="遍历 GSM8K 训练集的轮数")

# --- 批次大小 / 采样配置 ---
parser.add_argument("--device-batch-size", type=int, default=8,
                    help="单次前向传播的最大批次大小（受显存限制）")
parser.add_argument("--examples-per-step", type=int, default=16,
                    help="每个优化步中所有 rank 使用的不同问题数量")
parser.add_argument("--num-samples", type=int, default=16,
                    help="每个问题/示例生成多少个候选回答（即 GRPO 中的 G 个样本）")

# --- 生成（解码）参数 ---
parser.add_argument("--max-new-tokens", type=int, default=256, help="每个样本最大生成的 token 数")
parser.add_argument("--temperature", type=float, default=1.0, help="采样温度（1.0 = 不调整概率分布）")
parser.add_argument("--top-k", type=int, default=50, help="Top-K 采样（0 = 禁用，仅保留概率最高的 K 个 token）")

# --- 优化器参数 ---
parser.add_argument("--embedding-lr", type=float, default=0.2, help="嵌入层参数的学习率（AdamW）")
parser.add_argument("--unembedding-lr", type=float, default=0.004, help="解嵌层参数的学习率（AdamW）")
parser.add_argument("--matrix-lr", type=float, default=0.02, help="矩阵参数的学习率（Muon）")
parser.add_argument("--weight-decay", type=float, default=0.0, help="嵌入/解嵌参数的权重衰减（AdamW）")
parser.add_argument("--init-lr-frac", type=float, default=0.05,
                    help="初始学习率占基准学习率的比例（小幅预热效果）")

# --- 评估 / 检查点 ---
parser.add_argument("--eval-every", type=int, default=60, help="每隔 N 步评估 pass@k 指标")
parser.add_argument("--eval-examples", type=int, default=400, help="评估 pass@k 时使用的测试样本数量")
parser.add_argument("--save-every", type=int, default=60, help="每隔 N 步保存模型检查点")

args = parser.parse_args()
user_config = vars(args).copy()  # 保存用户配置用于日志

# =============================================================================
# 第二部分：计算环境和模型初始化
# =============================================================================

# 初始化设备（CUDA/CPU/MPS）、DDP 分布式通信
device_type = autodetect_device_type() if args.device_type == "" else args.device_type
ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init(device_type)
master_process = ddp_rank == 0  # 只有主进程负责日志和检查点保存

# --- wandb 日志初始化 ---
use_dummy_wandb = args.run == "dummy" or not master_process
wandb_run = DummyWandb() if use_dummy_wandb else wandb.init(
    project="nanochat-rl", name=args.run, config=user_config
)

# --- 加载 SFT 阶段训练好的模型和分词器 ---
# load_model 返回 (model, tokenizer, meta)
# phase="eval" 表示以评估模式加载（后续训练时再切换到 train 模式）
model, tokenizer, meta = load_model(
    "sft", device, phase="eval", model_tag=args.model_tag, step=args.model_step
)
engine = Engine(model, tokenizer)  # 用于批量生成回答（rollout）

# =============================================================================
# 第三部分：Rollout / 采样生成器
#
# 这是一个生成器函数，每次调用 yield 返回一个训练批次，包含：
# - generated_token_sequences: 原始生成的 token 序列（未填充）
# - inputs: 填充后的输入 token ID（去掉最后一个 token 的自回归输入）
# - targets: 填充后的目标 token ID（去掉第一个 token 的自回归目标）
# - rewards: 每条样本的 reward（0 或 1）
# - advantages: 优势函数值 = reward - mean(reward)
#
# 关键设计：
# - 每个问题生成 num_samples 个候选回答（GRPO 的核心：组内相对比较）
# - 使用 DAPO 风格的 token 级归一化（而非序列级）
# - 优势函数只用 (r - μ)，不用除以标准差（见脚本开头的设计说明）
# =============================================================================

train_task = GSM8K(subset="main", split="train")  # GSM8K 训练集（约 7473 个问题）
val_task = GSM8K(subset="main", split="test")     # GSM8K 测试集（约 1319 个问题）
# 计算总训练步数：每个 epoch 遍历训练集，每个 step 消费 examples_per_step 个不同问题
num_steps = (len(train_task) // args.examples_per_step) * args.num_epochs
print0(f"Calculated number of steps: {num_steps}")


@torch.no_grad()  # 采样生成过程不需要梯度（梯度只在训练前向+反向时计算）
def get_batch():
    """
    训练批次的无限生成器。

    每个 epoch 内，每个 DDP rank 负责训练集中不同的问题子集
    （通过 ddp_rank 跳跃式索引实现数据分片）。

    对每个问题：
    1. 渲染对话（保留 <|assistant_start|>，删除后面的 Assistant 回复）
    2. 生成 num_samples 个候选回答
    3. 用 GSM8K 的 reward 函数评判每个回答的正确性
    4. 将所有回答填充到相同长度
    5. 计算 reward → 优势函数 → yield

    Yields:
        generated_token_sequences: List[List[int]]，每个样本的 token 序列
        inputs: Tensor (B, T-1)，自回归模型的输入
        targets: Tensor (B, T-1)，自回归模型的目标（无效 token 标记为 -1）
        rewards: Tensor (B,)，每个样本的 reward
        advantages: Tensor (B,)，每个样本的优势函数值
    """
    # <|assistant_end|> 用于填充序列，它也会被 mask 掉（不参与 loss 计算）
    assistant_end = tokenizer.encode_special("<|assistant_end|>")

    # 每个 rank 负责不同的训练样本（循环交错分配：rank 0 取 idx=0, 1*r, 2*r...）
    rank_indices = range(ddp_rank, len(train_task), ddp_world_size)

    for example_idx in itertools.cycle(rank_indices):  # 无限循环（由外部 step 计数控制退出）

        # ---------- 步骤 1：准备对话 ----------
        # train_task[example_idx] 返回 List[ Dict[str,str] ]，
        # 如 [{"role":"user","content":"..."}, {"role":"assistant","content":"..."}]
        conversation = train_task[example_idx]

        # 将对话渲染为 token 序列并准备好让模型生成 Assistant 回复
        # render_for_completion 将：
        # - 所有消息转为 token ID
        # - 删除最后一个 Assistant 消息的内容
        # - 但保留 <|assistant_start|> token（让模型从这里开始生成）
        tokens = tokenizer.render_for_completion(conversation)
        prefix_length = len(tokens)  # 记录前缀长度，后续用于提取"生成部分"

        # ---------- 步骤 2：批量生成候选回答 ----------
        model.eval()  # 确保在评估模式（dropout 等禁用）
        generated_token_sequences = []  # 存储所有生成的 token 序列
        masks = []                       # 存储每个序列的 mask（0=不参与 loss，1=参与）

        # 分多批次生成以避免超出显存（device_batch_size 限制单次前向的批次大小）
        num_sampling_steps = args.num_samples // args.device_batch_size
        for sampling_step in range(num_sampling_steps):
            # 用 step、example_idx、sampling_step 组合生成确定性的随机种子
            # & 0x7FFFFFFF 确保是 int32 的正半部分
            # 不同步数、不同问题、不同批次使用不同种子 → 生成的回答多样化
            seed = hash((step, example_idx, sampling_step)) & 0x7FFFFFFF
            generated_token_sequences_batch, masks_batch = engine.generate_batch(
                tokens,
                num_samples=args.device_batch_size,
                max_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_k=args.top_k,
                seed=seed,  # 必须每批次改变种子，确保不同批次生成不同的回答
            )
            generated_token_sequences.extend(generated_token_sequences_batch)
            masks.extend(masks_batch)

        # ---------- 步骤 3：计算每个样本的 reward ----------
        # GSM8K 的 reward 函数：提取模型输出中的最终答案并与标准答案比较
        # 正确 → 1.0，错误 → 0.0
        rewards = []
        for sample_tokens in generated_token_sequences:
            # 只取生成部分的 token（去掉前缀/提示词部分）
            generated_tokens = sample_tokens[prefix_length:]
            # 解码回文本
            generated_text = tokenizer.decode(generated_tokens)
            # 评判正确性
            reward = train_task.reward(conversation, generated_text)
            rewards.append(reward)

        # ---------- 步骤 4：填充序列到相同长度 ----------
        # 不同样本生成的 token 数不同，需要填充对齐才能堆叠为 tensor
        max_length = max(len(seq) for seq in generated_token_sequences)
        # 用 <|assistant_end|> 填充（这个 token 会被 mask 掉，不影响 loss）
        padded_generated_token_sequences = [
            seq + [assistant_end] * (max_length - len(seq))
            for seq in generated_token_sequences
        ]
        # mask 也填充 0（表示这些位置不参与 loss 计算）
        padded_masks = [
            mask + [0] * (max_length - len(mask))
            for mask in masks
        ]

        # ---------- 步骤 5：构建训练张量 ----------
        ids = torch.tensor(padded_generated_token_sequences, dtype=torch.long, device=device)
        mask_ids = torch.tensor(padded_masks, dtype=torch.long, device=device)

        # 构建自回归输入/目标对：
        # inputs: [tok0, tok1, ..., tok_{n-2}]  （去掉最后一个 token）
        # targets: [tok1, tok2, ..., tok_{n-1}] （去掉第一个 token，每个位置的预测目标）
        inputs = ids[:, :-1]
        targets = ids[:, 1:].clone()  # clone 是为了接下来做原地修改
        # 将 mask=0 位置的目标设为 -1（PyTorch CrossEntropyLoss 的 ignore_index）
        # Engine 返回的 mask=0 表示：提示词部分 + 工具调用的强制 token
        # 这些位置我们不希望模型去学习（因为提示词是给定的，工具调用 token 是确定的）
        targets[mask_ids[:, 1:] == 0] = -1  # <-- 原地修改

        # ---------- 步骤 6：计算优势函数 ----------
        # 优势函数 = reward - mean(reward)
        # 注意：这里不使用 z-score 归一化 (r - μ)/σ，仅用 (r - μ)
        # 原因：
        # - 去中心化即可消除基准偏差
        # - 不除以 σ 保留了 reward 绝对大小的信息
        # - 当所有回答都是同一质量时（σ≈0），除以 σ 会导致数值不稳定
        rewards = torch.tensor(rewards, dtype=torch.float, device=device)
        mu = rewards.mean()                     # 组内平均 reward
        advantages = rewards - mu               # 优势：高于均值 → 正值（强化），低于均值 → 负值（抑制）

        yield generated_token_sequences, inputs, targets, rewards, advantages

# =============================================================================
# 第四部分：GSM8K pass@k 评估循环
#
# pass@k 的含义：对每个问题生成 k 个回答，只要其中有一个正确就算通过。
# pass@k 越高，说明模型的"尝试多样性"和"正确率"越好。
# =============================================================================

def run_gsm8k_eval(task, tokenizer, engine,
    max_examples=None,         # 最多评估多少个问题（None = 全部）
    num_samples=1,             # 每个问题生成几个回答（即 k 值）
    max_completion_tokens=256, # 每个回答最多生成多少 token
    temperature=0.0,           # 采样温度（0.0 = 贪心解码）
    top_k=50
):
    """
    评估 GSM8K 任务的 pass@k 指标。

    在分布式环境下，所有 rank 协作评估（不同 rank 评估不同的问题子集），
    但此函数不做跨 rank 的聚合 —— 由调用方负责。

    由于评估可能耗时较长，此函数逐个 yield 结果记录，
    调用方可以实时获取中间结果。

    Args:
        task: GSM8K 任务实例（train 或 test 子集）
        tokenizer: 分词器
        engine: 文本生成引擎
        max_examples: 最多评估的问题数
        num_samples: 每个问题生成的回答数（k 值）
        max_completion_tokens: 每个回答的最大长度
        temperature: 采样温度
        top_k: Top-K 采样参数

    Yields:
        record: dict，包含 {"idx": 问题索引, "outcomes": [{"is_correct": bool}, ...]}
    """
    max_examples = min(max_examples, len(task)) if max_examples is not None else len(task)
    # 每个 rank 评估不同的问题（跳跃式索引，避免重复）
    for idx in range(ddp_rank, max_examples, ddp_world_size):
        conversation = task[idx]
        # 渲染对话并准备好生成位置
        tokens = tokenizer.render_for_completion(conversation)
        prefix_length = len(tokens)

        # 批量生成 k 个回答
        assert num_samples <= args.device_batch_size  # 通常 k 不会超过 device_batch_size
        generated_token_sequences, masks = engine.generate_batch(
            tokens,
            num_samples=num_samples,
            max_tokens=max_completion_tokens,
            temperature=temperature,
            top_k=top_k
        )

        # 逐个检查每个回答是否正确
        outcomes = []
        for sample_tokens in generated_token_sequences:
            generated_tokens = sample_tokens[prefix_length:]      # 提取生成部分
            generated_text = tokenizer.decode(generated_tokens)   # 解码为文本
            is_correct = task.evaluate(conversation, generated_text)  # 与标准答案比对
            outcomes.append({
                "is_correct": is_correct
            })

        record = {
            "idx": idx,
            "outcomes": outcomes,
        }
        yield record

# =============================================================================
# 第五部分：训练循环
#
# 每个训练步的流程：
# 1. 条件评估：pass@k 指标
# 2. 对每个问题执行 rollout → 前向传播计算 log-prob → 反向传播累积梯度
# 3. 所有问题的梯度累积完成后，更新模型参数
# 4. 条件保存检查点
#
# 策略梯度（Policy Gradient）目标：
#   loss = - mean(log_p(token) * advantage)  across all valid tokens
#
# 直觉：
# - advantage > 0（回答比平均好）→ 最大化该 token 的 log 概率 → 强化这些生成
# - advantage < 0（回答比平均差）→ 最小化该 token 的 log 概率 → 抑制这些生成
# =============================================================================

# --- 初始化优化器 ---
# 使用与预训练相同的 MuonAdamW 组合优化器
# 注意：RL 阶段通常使用较小的学习率和零权重衰减（避免破坏预训练学到的能力）
optimizer = model.setup_optimizer(
    unembedding_lr=args.unembedding_lr,
    embedding_lr=args.embedding_lr,
    matrix_lr=args.matrix_lr,
    weight_decay=args.weight_decay,  # 默认为 0，不对 RL 阶段施加额外正则化
)

# 设置初始学习率为基准学习率的一个小比例（类似于一个小的 warmup）
# 这有助于在 RL 训练初期保持模型稳定性
for group in optimizer.param_groups:
    group["lr"] = group["lr"] * args.init_lr_frac
    group["initial_lr"] = group["lr"]  # 记录初始学习率，后续调度器乘子基于此

# --- 学习率调度器 ---
# 简单的线性衰减：从初始值线性降到 0（覆盖整个训练过程）
def get_lr_multiplier(it):
    """
    计算步数 it 处的学习率乘子。

    线性衰减策略：lrm = 1.0 - it / num_steps
    训练开始时 lrm=1.0，训练结束时 lrm=0.0
    """
    lrm = 1.0 - it / num_steps
    return lrm

# --- 计算每个 rank 处理的问题数 ---
# 确保 examples_per_step 能均匀分配给所有 rank
print0(f"Total sequences per step: {args.examples_per_step * args.num_samples}")  # 每个 step 的总序列数（问题数×每题采样数）
assert args.examples_per_step % ddp_world_size == 0, \
    "Desired examples per step must be divisible by the number of ranks"
examples_per_rank = args.examples_per_step // ddp_world_size  # 每个 GPU 负责的问题数
print0(f"Calculated examples per rank: {examples_per_rank}")

# =============================================================================
# 主训练循环
# =============================================================================

batch_iterator = get_batch()  # 创建无限训练批次生成器
for step in range(num_steps):

    # ============ 评估块：pass@k 指标 ============
    # 衡量模型解决 GSM8K 问题的能力
    if step % args.eval_every == 0:
        model.eval()
        # passk[t] = 前 t+1 个回答中至少有一个正确的题目数
        passk = torch.zeros(args.device_batch_size, device=device)

        # 在测试集上评估
        records_iter = run_gsm8k_eval(
            val_task, tokenizer, engine,
            num_samples=args.device_batch_size,
            max_examples=args.eval_examples,
            temperature=1.0  # 评估时使用 temperature=1.0（不做温度缩放）
        )
        records = list(records_iter)  # 收集所有 rank 的结果

        # 计算 pass@1 到 pass@k（k = device_batch_size）
        # pass@k = 前 k 个回答中至少有一个正确的题目数
        for k in range(1, args.device_batch_size + 1):
            # any(o["is_correct"] for o in r["outcomes"][:k]): 前 k 个回答中是否至少有一个正确
            passk[k - 1] = sum(
                any(o["is_correct"] for o in r["outcomes"][:k])
                for r in records
            )

        # 聚合所有 rank 的统计（分布式环境下需要 all_reduce）
        num_records = torch.tensor(len(records), dtype=torch.long, device=device)
        if ddp:
            dist.all_reduce(num_records, op=dist.ReduceOp.SUM)   # 汇总所有 rank 评估的总题目数
            dist.all_reduce(passk, op=dist.ReduceOp.SUM)         # 汇总所有 rank 通过的题目数

        passk = passk / num_records.item()  # 归一化得到比例（范围 [0, 1]）
        print_passk = [f"Pass@{k}: {passk[k - 1].item():.4f}" for k in range(1, args.device_batch_size + 1)]
        print0(f"Step {step} | {', '.join(print_passk)}")

        # 日志记录
        log_passk = {f"pass@{k}": passk[k - 1].item() for k in range(1, args.device_batch_size + 1)}
        wandb_run.log({
            "step": step,
            **log_passk,
        })

    # ============ 训练块：策略梯度更新 ============
    # 对每个问题独立进行 rollout + 梯度累积
    rewards_list = []      # 记录每个问题的平均 reward（用于日志）
    sequence_lengths = []  # 记录每个样本的序列长度（用于日志）

    for example_step in range(examples_per_rank):
        # --- 获取一个批次（一个问题的所有采样结果）---
        sequences_all, inputs_all, targets_all, rewards_all, advantages_all = next(batch_iterator)

        model.train()  # 切换到训练模式

        # --- 分段前向传播（避免超出显存）---
        # 一个问题的 num_samples 个样本可能超过 device_batch_size，需要拆分
        assert inputs_all.size(0) % args.device_batch_size == 0, \
            f"num_samples ({inputs_all.size(0)}) 必须能被 device_batch_size ({args.device_batch_size}) 整除"
        num_passes = inputs_all.size(0) // args.device_batch_size

        for pass_idx in range(num_passes):
            # 切出当前微批次的输入
            b0, b1 = pass_idx * args.device_batch_size, (pass_idx + 1) * args.device_batch_size
            inputs = inputs_all[b0:b1]       # (device_batch_size, T-1)
            targets = targets_all[b0:b1]     # (device_batch_size, T-1)
            rewards = rewards_all[b0:b1]     # (device_batch_size,)
            advantages = advantages_all[b0:b1]  # (device_batch_size,)

            # --- 计算对数概率 ---
            # model() 返回 NLL（负对数似然，Negative Log-Likelihood）= -log_p
            # loss_reduction='none' 返回逐 token 的 loss（而不是标量均值）
            # 取反得到 log_p（对数概率）
            logp = -model(inputs, targets, loss_reduction='none').view_as(inputs)  # (B, T)

            # --- 计算策略梯度目标 ---
            # PG 目标：对高 advantage 的 token 增加其对数概率
            #   objective = sum(log_p(token) * advantage(token))  across all tokens
            #   忽略 targets=-1 的位置（这些位置 loss=0，不贡献梯度）
            #   advantages.unsqueeze(-1) 将 (B,) 广播为 (B, 1)，与 (B, T) 做逐元素乘法
            pg_obj = (logp * advantages.unsqueeze(-1)).sum()

            # --- 归一化 ---
            # 除以三部分以消除批次规模的影响：
            # 1. num_valid: 当前微批次中有效 token 的数量（排除 ignore_index=-1 的位置）
            # 2. num_passes: 这个问题被拆分成几个前向 pass
            # 3. examples_per_rank: 这个 rank 负责几个问题
            num_valid = (targets >= 0).sum().clamp(min=1)  # clamp(min=1) 防止除零
            pg_obj = pg_obj / (num_valid * num_passes * examples_per_rank)

            # --- 关键说明：为什么不需要 PPO 的 ratio + clip？---
            # 这里的 rollout 和数据采样是同一步内完成的（完全在策略），
            # 生成回答的模型和计算梯度的模型是同一个，不存在"行为策略"与"目标策略"的偏差。
            # PPO 的 ratio 矫正和 clip 是针对 off-policy 数据（用旧策略收集的经验训练新策略），
            # 在完全 on-policy 的场景下不需要。

            # 最终 loss = -objective（因为优化器是最小化 loss，而我们要最大化 objective）
            loss = -pg_obj
            loss.backward()  # 累积梯度（不清零，等所有问题和 pass 做完后一起更新）

            print0(f"Step {step}/{num_steps} | Example step {example_step} | Pass {pass_idx} | loss: {loss.item():.6f} | Average reward: {rewards.mean().item()}")

        # 记录该问题的统计信息（用于日志）
        rewards_list.append(rewards_all.mean().item())
        sequence_lengths.extend(len(seq) for seq in sequences_all)

    # --- 日志：Rollout 统计 ---
    mean_reward = sum(rewards_list) / len(rewards_list)          # 所有问题的平均 reward
    mean_sequence_length = sum(sequence_lengths) / len(sequence_lengths)  # 平均生成长度

    if ddp:  # 跨 rank 聚合（取平均）
        mean_reward_tensor = torch.tensor(mean_reward, dtype=torch.float, device=device)
        mean_sequence_length_tensor = torch.tensor(mean_sequence_length, dtype=torch.float, device=device)
        dist.all_reduce(mean_reward_tensor, op=dist.ReduceOp.AVG)
        dist.all_reduce(mean_sequence_length_tensor, op=dist.ReduceOp.AVG)
        mean_reward = mean_reward_tensor.item()
        mean_sequence_length = mean_sequence_length_tensor.item()

    print0(f"Step {step}/{num_steps} | Average reward: {mean_reward} | Average sequence length: {mean_sequence_length:.2f}")
    wandb_run.log({
        "step": step,
        "reward": mean_reward,
        "sequence_length": mean_sequence_length,
    })

    # --- 优化器更新 ---
    # 设置当前步的学习率（按调度器）
    lrm = get_lr_multiplier(step)
    for group in optimizer.param_groups:
        group["lr"] = group["initial_lr"] * lrm
    optimizer.step()                   # 用所有累积的梯度更新模型参数
    model.zero_grad(set_to_none=True)  # 清零梯度（set_to_none=True 释放内存而非清零）

    wandb_run.log({
        "step": step,
        "lrm": lrm,
    })

    # --- 保存检查点 ---
    # 跳过第 0 步，每隔 save_every 步保存，最后一步始终保存
    if master_process and ((step > 0 and step % args.save_every == 0) or step == num_steps - 1):
        base_dir = get_base_dir()
        depth = model.config.n_layer
        output_dirname = args.model_tag if args.model_tag else f"d{depth}"  # 基于基础模型深度命名
        checkpoint_dir = os.path.join(base_dir, "chatrl_checkpoints", output_dirname)
        model_config_kwargs = model.config.__dict__  # 模型配置（简单粗暴地使用 __dict__）
        save_checkpoint(
            checkpoint_dir,
            step,
            model.state_dict(),
            None,  # 注意：不保存优化器状态（RL 训练通常不需要从中间恢复优化器状态）
            {
                "model_config": model_config_kwargs,
            }
        )
        print(f"✅ Saved model checkpoint to {checkpoint_dir}")

# =============================================================================
# 第六部分：训练结束后的报告与清理
# =============================================================================

# 将训练配置写入报告（用于实验对比和分析）
from nanochat.report import get_report
get_report().log(section="Chat RL", data=[
    user_config,  # CLI 参数
])

wandb_run.finish()  # 结束 wandb 运行
compute_cleanup()   # 清理 DDP 分布式通信组和进程
