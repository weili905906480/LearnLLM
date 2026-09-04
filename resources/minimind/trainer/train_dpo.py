"""
MiniMind DPO（Direct Preference Optimization，直接偏好优化）训练脚本。

作用：在 SFT 之后的对齐阶段，让模型（策略模型 policy）输出「人类更喜欢的回答 chosen」，
而不是「被嫌弃的回答 rejected」，同时用冻结的参考模型（ref）当锚点，防止模型跑偏/遗忘。

与 SFT（train_full_sft.py）的本质区别：
    SFT：单条「问题 + 标准答案」，最大化标准答案的似然，只有一个模型。
    DPO：成对「问题 + 好回答(chosen) + 坏回答(rejected)」，拉开两者的相对概率差距，
         需要 policy + ref 两个模型（ref 冻结）。

与 GRPO / PPO 的区别（详见 doc/DPO_GRPO_PPO对比.md）：
    DPO  不需要 reward model，不需要 critic，也不需要在线采样，用闭式公式直接算 loss，最轻量。
    GRPO 需要 reward model + 在线采样，但用「组内相对比较」替代 critic。
    PPO  需要 reward model + critic（value model）+ 在线采样，最重、最完整。

DPO 损失公式（见下方 dpo_loss 函数）：
    loss = -log σ( β · [ log(π_θ(chosen)/π_θ(rejected)) - log(π_ref(chosen)/π_ref(rejected)) ] )
"""

import os
import sys

__package__ = "trainer"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import datasets  # noqa: F401  # Windows pyarrow/torch DLL conflict workaround (issue #771)
import argparse
import time
import warnings
import torch
import torch.nn.functional as F
import torch.distributed as dist
from contextlib import nullcontext
from torch import optim
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler
from model.model_minimind import MiniMindConfig
from dataset.lm_dataset import DPODataset
from trainer.trainer_utils import get_lr, Logger, is_main_process, lm_checkpoint, init_distributed_mode, setup_seed, init_model, SkipBatchSampler

warnings.filterwarnings('ignore')


def logits_to_log_probs(logits, labels):
    """
    把模型输出的 logits 转成「每个位置真实 token 的对数概率（log prob）」。

    输入：
        logits : shape (batch_size, seq_len, vocab_size)，模型对每个位置每个词表 token 打的原始分数
        labels : shape (batch_size, seq_len)，每个位置的真实 token id（已经右移一位，见 DPODataset）
    输出：
        log_probs_per_token : shape (batch_size, seq_len)，每个位置真实 token 的 log 概率

    步骤：
        1. F.log_softmax(logits, dim=2)：在词表维度(dim=2)上做归一化并取对数，
           得到每个位置每个 token 的 log 概率（都是负数，exp 后求和 = 1）。
        2. torch.gather(...)：只挑出 labels 指定的那个 token 的 log 概率。

    具体例子（假设 vocab_size=5，词表 id：0=pad, 1=<|im_start|>, 2=<|im_end|>, 3=好, 4=你）：
        某位置 t 的 logits:           [0.1, -1.0, 2.5, 1.8, -0.3]
        log_softmax 后（示意值）:     [-2.31, -3.41, 0.09, -0.61, -2.71]
        labels[t] = 3（真实 token 是「好」）
        gather 取出第 3 号 → log_probs_per_token[t] = -0.61

    这一层后面会跟 loss_mask 相乘，只保留「assistant 回答段」的 log 概率（见 dpo_loss）。
    """
    log_probs = F.log_softmax(logits, dim=2)
    log_probs_per_token = torch.gather(log_probs, dim=2, index=labels.unsqueeze(2)).squeeze(-1)
    return log_probs_per_token


def dpo_loss(ref_log_probs, policy_log_probs, mask, beta):
    """
    DPO 核心损失。输入 ref 和 policy 各自对整条序列的逐 token log 概率，输出一个标量 loss。

    输入：
        ref_log_probs    : shape (batch_size, seq_len)，参考模型(冻结)的逐 token log 概率
        policy_log_probs : shape (batch_size, seq_len)，策略模型(可训练)的逐 token log 概率
        mask             : shape (batch_size, seq_len)，只有 assistant 回答段为 1，其余为 0
        beta             : DPO 的 beta 参数，控制「偏好强度 / 偏离 ref 的程度」

    关键约定：batch 里前一半是 chosen、后一半是 rejected。
        因为 train_epoch 里 `x = torch.cat([x_chosen, x_rejected], dim=0)` 是按行拼接的，
        所以这里 `batch_size // 2` 正好切开：[:B] 是 chosen，[B:] 是 rejected。

    计算过程：
        1. 先乘 mask 再 sum(dim=1)，把每条序列「回答段所有 token 的 log 概率」加起来，
           得到整段回答的对数概率 log P(回答 | 上下文)。用户提示部分被 mask 屏蔽，不参与。
        2. 分开 chosen / rejected，算 policy 和 ref 各自的「chosen 减 rejected」差值：
               pi_logratios  = log π_θ(chosen)   - log π_θ(rejected)
               ref_logratios = log π_ref(chosen) - log π_ref(rejected)
               logits        = pi_logratios - ref_logratios
                            = log[π_θ(chosen)/π_θ(rejected)] - log[π_ref(chosen)/π_ref(rejected)]
        3. loss = -log σ(β · logits)
           - 模型越偏好 chosen（相对 rejected，且相对 ref 涨得越多），logits 越大，σ(·)→1，loss→0。
           - 模型若反过来偏好 rejected，logits 为负，σ(·) 很小，-log 很大，被狠狠惩罚。

    完整数值例子（batch_size=2，即 1 个 chosen + 1 个 rejected，每条已按 mask 求和）：
        log π_θ(chosen)    = -3.0   （policy 对好回答的对数概率）
        log π_θ(rejected)  = -5.0   （policy 对坏回答的对数概率）
        log π_ref(chosen)  = -4.0
        log π_ref(rejected)= -4.5

        逐步计算：
            pi_logratios  = (-3.0) - (-5.0) = 2.0     # policy 明显更偏好 chosen
            ref_logratios = (-4.0) - (-4.5) = 0.5     # ref 只稍微偏好 chosen
            logits = 2.0 - 0.5 = 1.5

            beta = 0.15
            β·logits = 0.15 × 1.5 = 0.225
            σ(0.225) = 1/(1+e^-0.225) ≈ 0.556
            loss = -log(0.556) ≈ 0.587

        直觉：policy 已在正确方向上跑赢 ref（偏好差 2.0 > 0.5），所以损失不大（0.587）。
    """
    # 第 1 行：把「逐 token 的 log 概率」乘以 mask，只保留回答段，再沿 seq 维度求和。
    #         因为 log 概率相加 = 相乘（log P = Σ log p_t），所以 sum 得到「整段回答的对数概率」log P(回答|上下文)。
    #         用户提示和 padding 位置 mask=0，被屏蔽掉，不参与。
    ref_log_probs = (ref_log_probs * mask).sum(dim=1)
    # 第 2 行：对策略模型做同样的 mask + 求和，得到每条样本整段回答的 log 概率。
    policy_log_probs = (policy_log_probs * mask).sum(dim=1)

    # 第 3 行：记录拼接后的总行数（= 2 × 真实 batch_size，因为 chosen 和 rejected 按行拼在一起）。
    batch_size = ref_log_probs.shape[0]
    # 第 4~7 行：按「前一半 chosen、后一半 rejected」的约定切开，取出四个标量向量（各 shape (B,)）。
    chosen_ref_log_probs = ref_log_probs[:batch_size // 2]        # ref 对 chosen 的整段 log 概率
    reject_ref_log_probs = ref_log_probs[batch_size // 2:]        # ref 对 rejected 的整段 log 概率
    chosen_policy_log_probs = policy_log_probs[:batch_size // 2]  # policy 对 chosen 的整段 log 概率
    reject_policy_log_probs = policy_log_probs[batch_size // 2:]  # policy 对 rejected 的整段 log 概率

    # 第 8 行：policy 模型对「chosen 相对 rejected」的偏好差（对数几率）
    #         = log π_θ(chosen) - log π_θ(rejected) = log[ π_θ(chosen) / π_θ(rejected) ]
    pi_logratios = chosen_policy_log_probs - reject_policy_log_probs
    # 第 9 行：ref 模型对「chosen 相对 rejected」的偏好差，作为「不动的标尺」基准锚点。
    ref_logratios = chosen_ref_log_probs - reject_ref_log_probs
    # 第 10 行：最终 logits = policy 的偏好差 − ref 的偏好差。
    #          这是 DPO 的核心量：衡量「policy 相对 ref 多涨了多少对 chosen 的偏好」。
    #          logits > 0 表示 policy 比 ref 更偏好 chosen（好），< 0 表示更偏好 rejected（坏）。
    logits = pi_logratios - ref_logratios
    # 第 11 行：DPO 损失 = -log σ(β · logits)；F.logsigmoid(x) = log σ(x)，σ 是 sigmoid。
    #          β·logits 越大 → σ→1 → log→0 → loss→0（模型表现好，几乎不惩罚）。
    #          β·logits 越小(负) → σ→0 → log→ -∞ → loss→ +∞（模型偏好反了，狠狠惩罚）。
    loss = -F.logsigmoid(beta * logits)
    # 第 12 行：对 batch 内所有偏好对取平均，得到一个标量 loss。
    return loss.mean()


def train_epoch(epoch, loader, iters, ref_model, lm_config, start_step=0, wandb=None, beta=0.1):
    start_time = time.time()
    last_step = start_step

    for step, batch in enumerate(loader, start=start_step + 1):
        last_step = step
        # 取出本 batch 的 chosen / rejected 两路数据，各自搬到训练设备
        x_chosen = batch['x_chosen'].to(args.device)
        x_rejected = batch['x_rejected'].to(args.device)
        y_chosen = batch['y_chosen'].to(args.device)
        y_rejected = batch['y_rejected'].to(args.device)
        mask_chosen = batch['mask_chosen'].to(args.device)
        mask_rejected = batch['mask_rejected'].to(args.device)

        # 关键：把 chosen 和 rejected 按行拼接成一个 batch（前 B 行 chosen，后 B 行 rejected）。
        # 这样 ref 和 policy 各前向一次即可同时算 chosen/rejected 的 logits，dpo_loss 里再切开。
        # 例：batch_size=4 时，cat 后 x.shape = (8, seq_len)，前 4 行是 chosen，后 4 行是 rejected。
        x = torch.cat([x_chosen, x_rejected], dim=0)
        y = torch.cat([y_chosen, y_rejected], dim=0)
        mask = torch.cat([mask_chosen, mask_rejected], dim=0)

        # 余弦退火学习率：lr * (0.1 + 0.45*(1 + cos(π·t/T)))，t=0 时系数 1.0，t=T 时降到 0.1。
        # DPO 初始 lr 极小（默认 4e-8），再配合退火进一步防止遗忘。
        lr = get_lr(epoch * iters + step, args.epochs * iters, args.learning_rate)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        with autocast_ctx:
            # 参考模型：只前向出 logits，torch.no_grad() 切断梯度，作为「不动的标尺」
            with torch.no_grad():
                ref_outputs = ref_model(x)
                ref_logits = ref_outputs.logits
            ref_log_probs = logits_to_log_probs(ref_logits, y)

            # 策略模型：可训练，正常前向
            outputs = model(x)
            logits = outputs.logits
            policy_log_probs = logits_to_log_probs(logits, y)

            dpo_loss_val = dpo_loss(ref_log_probs, policy_log_probs, mask, beta=beta)
            # aux_loss 是 MoE 架构的负载均衡辅助损失，普通 dense 模型时为 0
            loss = dpo_loss_val + outputs.aux_loss
            # 除以累积步数，等价于把多步累积的梯度做平均
            loss = loss / args.accumulation_steps

        scaler.scale(loss).backward()

        # 每 accumulation_steps 步才真正更新一次权重（梯度累积）
        if step % args.accumulation_steps == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)  # 防梯度爆炸
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        if step % args.log_interval == 0 or step == iters:
            spend_time = time.time() - start_time
            # 前面 loss 除了 accumulation_steps，这里乘回来，还原成「真实一步损失」用于日志
            current_loss = loss.item() * args.accumulation_steps
            current_dpo_loss = dpo_loss_val.item()
            current_aux_loss = outputs.aux_loss.item()
            current_lr = optimizer.param_groups[-1]['lr']
            eta_min = spend_time / max(step - start_step, 1) * (iters - step) // 60

            Logger(f'Epoch:[{epoch + 1}/{args.epochs}]({step}/{iters}), loss: {current_loss:.4f}, dpo_loss: {current_dpo_loss:.4f}, aux_loss: {current_aux_loss:.4f}, learning_rate: {current_lr:.8f}, epoch_time: {eta_min:.3f}min')

            if wandb: wandb.log({"loss": current_loss, "dpo_loss": current_dpo_loss, "aux_loss": current_aux_loss, "learning_rate": current_lr, "epoch_time": eta_min})

        if (step % args.save_interval == 0 or step == iters) and is_main_process():
            model.eval()
            moe_suffix = '_moe' if lm_config.use_moe else ''
            ckp = f'{args.save_dir}/{args.save_weight}_{lm_config.hidden_size}{moe_suffix}.pth'
            # 解 DDP / torch.compile 包装，拿到真正的模型再取 state_dict
            raw_model = model.module if isinstance(model, DistributedDataParallel) else model
            raw_model = getattr(raw_model, '_orig_mod', raw_model)
            state_dict = raw_model.state_dict()
            # 存纯权重（fp16 到 CPU 省空间），用于后续加载/推理
            torch.save({k: v.half().cpu() for k, v in state_dict.items()}, ckp)
            # 另存完整 resume 状态（model/optimizer/epoch/step/wandb_id），用于断点续训
            lm_checkpoint(lm_config, weight=args.save_weight, model=model, optimizer=optimizer, scaler=scaler, epoch=epoch, step=step, wandb=wandb, save_dir='../checkpoints')
            model.train()
            del state_dict

        # 及时释放显存：DPO 一次要同时存 ref + policy 两份 logits，显存紧张
        del x_chosen, x_rejected, y_chosen, y_rejected, mask_chosen, mask_rejected, x, y, mask
        del ref_outputs, ref_logits, ref_log_probs, outputs, logits, policy_log_probs, loss

    # 尾部处理：epoch 结束若还有不足 accumulation_steps 的未更新梯度，补做最后一次 step
    if last_step > start_step and last_step % args.accumulation_steps != 0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MiniMind DPO (Direct Preference Optimization)")
    parser.add_argument("--save_dir", type=str, default="../out", help="模型保存目录")
    parser.add_argument('--save_weight', default='dpo', type=str, help="保存权重的前缀名")
    parser.add_argument("--epochs", type=int, default=1, help="训练轮数")
    parser.add_argument("--batch_size", type=int, default=4, help="batch size")
    parser.add_argument("--learning_rate", type=float, default=4e-8, help="初始学习率（建议<=5e-8避免遗忘）")
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu", help="训练设备")
    parser.add_argument("--dtype", type=str, default="bfloat16", help="混合精度类型")
    parser.add_argument("--num_workers", type=int, default=8, help="数据加载线程数")
    parser.add_argument("--accumulation_steps", type=int, default=1, help="梯度累积步数")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="梯度裁剪阈值")
    parser.add_argument("--log_interval", type=int, default=100, help="日志打印间隔")
    parser.add_argument("--save_interval", type=int, default=100, help="模型保存间隔")
    parser.add_argument('--hidden_size', default=768, type=int, help="隐藏层维度")
    parser.add_argument('--num_hidden_layers', default=8, type=int, help="隐藏层数量")
    parser.add_argument('--max_seq_len', default=1024, type=int, help="训练的最大截断长度（中文1token≈1.5~1.7字符）")
    parser.add_argument('--use_moe', default=0, type=int, choices=[0, 1], help="是否使用MoE架构（0=否，1=是）")
    parser.add_argument("--data_path", type=str, default="../dataset/dpo.jsonl", help="DPO训练数据路径")
    parser.add_argument('--from_weight', default='full_sft', type=str, help="基于哪个权重训练")
    parser.add_argument('--from_resume', default=0, type=int, choices=[0, 1], help="是否自动检测&续训（0=否，1=是）")
    parser.add_argument('--beta', default=0.15, type=float, help="DPO中的beta参数")
    parser.add_argument("--use_wandb", action="store_true", help="是否使用wandb")
    parser.add_argument("--wandb_project", type=str, default="MiniMind-DPO", help="wandb项目名")
    parser.add_argument("--use_compile", default=0, type=int, choices=[0, 1], help="是否使用torch.compile加速（0=否，1=是）")
    args = parser.parse_args()

    # ========== 1. 初始化环境和随机种子 ==========
    local_rank = init_distributed_mode()
    if dist.is_initialized(): args.device = f"cuda:{local_rank}"
    setup_seed(42 + (dist.get_rank() if dist.is_initialized() else 0))

    # ========== 2. 配置目录、模型参数、检查ckp ==========
    os.makedirs(args.save_dir, exist_ok=True)
    lm_config = MiniMindConfig(hidden_size=args.hidden_size, num_hidden_layers=args.num_hidden_layers, use_moe=bool(args.use_moe))
    ckp_data = lm_checkpoint(lm_config, weight=args.save_weight, save_dir='../checkpoints') if args.from_resume==1 else None

    # ========== 3. 设置混合精度 ==========
    device_type = "cuda" if "cuda" in args.device else "cpu"
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
    autocast_ctx = nullcontext() if device_type == "cpu" else torch.cuda.amp.autocast(dtype=dtype)

    # ========== 4. 配wandb ==========
    wandb = None
    if args.use_wandb and is_main_process():
        import swanlab as wandb
        wandb_id = ckp_data.get('wandb_id') if ckp_data else None
        resume = 'must' if wandb_id else None
        wandb_run_name = f"MiniMind-DPO-Epoch-{args.epochs}-BatchSize-{args.batch_size}-LR-{args.learning_rate}"
        wandb.init(project=args.wandb_project, name=wandb_run_name, id=wandb_id, resume=resume)

    # ========== 5. 定义模型和参考模型 ==========
    # 策略模型：参与训练
    model, tokenizer = init_model(lm_config, args.from_weight, device=args.device)
    Logger(f'策略模型总参数量：{sum(p.numel() for p in model.parameters()) / 1e6:.3f} M')
    # 参考模型：从同一个 from_weight 加载，初始和 policy 完全相同，但冻结不更新。
    # ref 的角色是「锚点」——防止 policy 在讨好偏好时彻底忘掉 SFT 学到的能力。
    ref_model, _ = init_model(lm_config, args.from_weight, device=args.device)
    ref_model.eval()
    ref_model.requires_grad_(False)
    Logger(f'参考模型总参数量：{sum(p.numel() for p in ref_model.parameters()) / 1e6:.3f} M')

    train_ds = DPODataset(args.data_path, tokenizer, max_length=args.max_seq_len)
    train_sampler = DistributedSampler(train_ds) if dist.is_initialized() else None
    # GradScaler 只在 fp16 时真正生效；bf16 时 enabled=False，scaler.scale 实际不缩放
    scaler = torch.cuda.amp.GradScaler(enabled=(args.dtype == 'float16'))
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)

    # ========== 6. 从ckp恢复状态 ==========
    start_epoch, start_step = 0, 0
    if ckp_data:
        model.load_state_dict(ckp_data['model'])
        optimizer.load_state_dict(ckp_data['optimizer'])
        scaler.load_state_dict(ckp_data['scaler'])
        start_epoch = ckp_data['epoch']
        start_step = ckp_data.get('step', 0)

    # ========== 7. 编译和分布式包装 ==========
    if args.use_compile == 1:
        model = torch.compile(model)
        Logger('torch.compile enabled')
    if dist.is_initialized():
        model = DistributedDataParallel(model, device_ids=[local_rank])

    # ========== 8. 开始训练 ==========
    for epoch in range(start_epoch, args.epochs):
        train_sampler and train_sampler.set_epoch(epoch)
        # 固定种子+epoch 保证可复现，同时不同 epoch 数据顺序不同
        setup_seed(42 + epoch); indices = torch.randperm(len(train_ds)).tolist()
        # 断点续训：skip 记录需要跳过的 batch 数，从上次中断的 step 精确接上
        skip = start_step if (epoch == start_epoch and start_step > 0) else 0
        batch_sampler = SkipBatchSampler(train_sampler or indices, args.batch_size, skip)
        loader = DataLoader(train_ds, batch_sampler=batch_sampler, num_workers=args.num_workers, pin_memory=True)
        if skip > 0:
            Logger(f'Epoch [{epoch + 1}/{args.epochs}]: 跳过前{start_step}个step，从step {start_step + 1}开始')
            train_epoch(epoch, loader, len(loader) + skip, ref_model, lm_config, start_step, wandb, args.beta)
        else:
            train_epoch(epoch, loader, len(loader), ref_model, lm_config, 0, wandb, args.beta)

    # ========== 9. 清理分布进程 ==========
    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()
