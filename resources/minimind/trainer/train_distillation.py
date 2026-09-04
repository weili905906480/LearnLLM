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
from dataset.lm_dataset import SFTDataset
from trainer.trainer_utils import get_lr, Logger, is_main_process, lm_checkpoint, init_distributed_mode, setup_seed, init_model, SkipBatchSampler

warnings.filterwarnings('ignore')


def distillation_loss(student_logits, teacher_logits, temperature=1.0, reduction='batchmean'):
    """
    知识蒸馏损失：KL(教师软标签 || 学生软标签)，带温度缩放。

    原理（以 1 个 token、词表 V=4 为例）：

        教师原始 logits   z_t = [4.0, 3.0, 2.0, 1.0]
        学生原始 logits   z_s = [3.0, 2.0, 1.0, 0.0]

    (1) 先除以温度 T，再 softmax，得到"软化"后的概率分布（软标签）。
        T 越大分布越平滑，越能暴露 token 之间的相对关系（暗知识）：

            温度 T=1（接近 one-hot）: softmax(z_t)   ≈ [0.644, 0.237, 0.087, 0.032]
            温度 T=2（更平滑）      : softmax(z_t/2) ≈ [0.455, 0.276, 0.167, 0.102]
                                                       ↑最高值降低、最低值升高，分布更"平"

    (2) 学生用同样的温度做 log_softmax，与教师软标签算 KL 散度：
            KL = Σ_i 教师prob[i] * log(教师prob[i] / 学生prob[i])
        即让学生去"模仿"教师的完整概率分布，而不是只学一个硬标签 argmax。

    (3) 最后乘 T²：softmax(z/T) 对 z 的梯度量级 ∝ 1/T，
        经过 KL 后梯度 ∝ 1/T²，乘回 T² 使蒸馏损失的梯度尺度与 T 无关，便于调参。
    """
    with torch.no_grad():
        # 教师是"目标"，冻结梯度；只算一次软标签作为固定监督信号（.detach() 切断计算图）
        teacher_probs = F.softmax(teacher_logits / temperature, dim=-1).detach()

    # 学生端需要反传梯度，用 log_softmax（数值上等于 log(softmax)，但更稳定）
    student_log_probs = F.log_softmax(student_logits / temperature, dim=-1)

    # F.kl_div(input=log概率, target=概率) = Σ target * (log(target) - input)
    # 代入即 Σ teacher_probs * (log teacher_probs - log student_probs)，正是 KL(教师||学生)
    kl = F.kl_div(
        student_log_probs,
        teacher_probs,
        reduction=reduction
    )
    # 温度平方补偿（见 docstring 第 (3) 点）
    return (temperature ** 2) * kl


def train_epoch(epoch, loader, iters, teacher_model, lm_config_student, start_step=0, wandb=None, alpha=0.0, temperature=1.0):
    start_time = time.time()
    last_step = start_step

    if teacher_model is not None:
        # 教师模型全程冻结：只推理、不更新、不算梯度
        teacher_model.eval()
        teacher_model.requires_grad_(False)

    for step, (input_ids, labels) in enumerate(loader, start=start_step + 1):
        last_step = step
        input_ids = input_ids.to(args.device)
        labels = labels.to(args.device)
        # ---------- 构造损失掩码 loss_mask ----------
        # labels 中 -100 表示"忽略位"（padding，或无需监督的 prompt 部分）。
        # 语言模型用 logits[t] 预测 labels[t+1]（下一个 token），
        # 所以 labels 要整体左移一位：labels[..., 1:] 才是每个 logits 位置对应的真实标签。
        #
        # 举例（batch=1, seq_len=6，最后一位是 padding）：
        #     labels          = [  1,   2,   3,   4,   5, -100]
        #     labels[..., 1:] = [  2,   3,   4,   5, -100]
        #     loss_mask       = [  1,   1,   1,   1,    0]   # 前4位参与loss，最后padding位忽略
        loss_mask = (labels[..., 1:] != -100).float()
        # 余弦退火学习率：lr 随训练进度从初始 lr 单调衰减到 0.1*lr
        # (get_lr = lr*(0.1 + 0.45*(1+cos(pi*step/total)))：step=0 时为 1.0*lr，step=total 时为 0.1*lr)
        lr = get_lr(epoch * iters + step, args.epochs * iters, args.learning_rate)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        # ---------- 前向传播（学生模型） ----------
        # autocast_ctx：混合精度上下文，bf16/fp16 自动降精度以提速、省显存
        with autocast_ctx:
            res = model(input_ids)
            # logits 原始形状: [batch, seq_len, vocab_size]
            # [..., :-1, :] 去掉最后一个位置的 logits：
            # 位置 t 的 logits 预测的是 t+1 的 token，共 seq_len-1 个有效预测，
            # 与下方 shift_labels = labels[..., 1:] 一一对齐。
            #
            # 举例（seq_len=6, vocab_size=4）：
            #     logits[0] 预测 token[1]，logits[1] 预测 token[2]，...，logits[4] 预测 token[5]
            #     logits[5]（预测不存在的"第7个token"）没有真实标签，直接丢弃
            #     => student_logits 形状变为 [batch, 5, 4]
            student_logits = res.logits[..., :-1, :].contiguous()

        # ---------- 教师模型前向传播（eval & no_grad，只推理不反传） ----------
        if teacher_model is not None:
            with torch.no_grad():
                teacher_logits = teacher_model(input_ids).logits[..., :-1, :].contiguous()
                # 教师和学生可能词表大小不同（如教师 vocab=128256，学生 vocab=6400），
                # 只取教师 logits 的前 vocab_size_student 维，保证两者最后一维对齐，才能逐 token 算 KL。
                vocab_size_student = student_logits.size(-1)
                teacher_logits = teacher_logits[..., :vocab_size_student]

        # ========== 计算损失 ==========
        # 1) Ground-Truth CE Loss：学生直接学真实标签（硬标签）
        #    对齐方式：student_logits 已去掉末位，对应 labels[..., 1:]，逐位置预测"下一个token"
        shift_labels = labels[..., 1:].contiguous()
        loss_mask_flat = loss_mask.view(-1)          # 展平成 [batch*(seq_len-1)]，与 view(-1) 后的 logits 行一一对应
        # 对每个 token 独立算 CE（reduction='none' 不先做平均），得到逐 token 的损失向量
        ce_loss = F.cross_entropy(
            student_logits.view(-1, student_logits.size(-1)),   # [batch*(seq_len-1), vocab_size]
            shift_labels.view(-1),                              # [batch*(seq_len-1)]
            ignore_index=-100,      # -100 位（padding）返回 0，不贡献梯度
            reduction='none'
        )
        # 只对有效 token 求平均：用掩码把 padding 位的损失归零，再除以有效 token 数。
        # 举例（batch=1，共 5 个预测位，其中 1 个 padding）：
        #     ce_loss(逐token)  = [0.9, 1.2, 0.7, 2.0, 0.0]   # 最后一个 padding 位为 0
        #     loss_mask_flat    = [  1,   1,   1,   1,   0]
        #     sum(ce*mask)      = 0.9 + 1.2 + 0.7 + 2.0 = 4.8
        #     loss_mask.sum()   = 4
        #     ce_loss_raw       = 4.8 / 4 = 1.2   （分母 +1e-8 防止全 padding 时除零）
        ce_loss_raw = torch.sum(ce_loss * loss_mask_flat) / (loss_mask_flat.sum() + 1e-8)
        # MoE 模型需额外加上路由负载均衡的辅助损失 aux_loss（让各专家被均匀使用，防止少数专家"一家独大"）
        if lm_config_student.use_moe: ce_loss = ce_loss_raw + res.aux_loss
        else: ce_loss = ce_loss_raw

        # 2) Distillation Loss：学生模仿教师输出的概率分布（软标签）
        #    同样只在有效 token 上计算：用布尔索引 [loss_mask_flat == 1] 挑出非 padding 行，
        #    使教师和学生都只保留"有效 token"对应的 logits 行，再逐行算 KL 散度。
        if teacher_model is not None:
            distill_loss = distillation_loss(
                student_logits.view(-1, student_logits.size(-1))[loss_mask_flat == 1],   # [有效token数, vocab_size]
                teacher_logits.view(-1, teacher_logits.size(-1))[loss_mask_flat == 1],   # [有效token数, vocab_size]
                temperature=temperature
            )
        else:
            # 无教师时蒸馏损失为 0（此时等价于纯 SFT 训练）
            distill_loss = torch.tensor(0.0, device=args.device)

        # 3) 总损失 = alpha * CE + (1-alpha) * Distill
        #    alpha 控制两者比重：alpha=1 纯 SFT，alpha=0 纯蒸馏，默认 0.5 各占一半。
        #    除以 accumulation_steps 实现梯度累积（等价于把多个小 batch 拼成一个大 batch）。
        loss = (alpha * ce_loss + (1 - alpha) * distill_loss) / args.accumulation_steps

        # 混合精度反向：GradScaler 把 loss 放大 scale 倍再 backward，
        # 防止 fp16 下梯度下溢（数值太小被截断成 0）
        scaler.scale(loss).backward()

        # 攒满 accumulation_steps 个 step 才真正更新一次权重
        if step % args.accumulation_steps == 0:
            scaler.unscale_(optimizer)                                                # 还原梯度真实尺度（否则裁剪阈值不准确）
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)        # 梯度裁剪，防止梯度爆炸
            scaler.step(optimizer)                                                    # 更新权重
            scaler.update()                                                           # 动态调整 scale 因子
            optimizer.zero_grad(set_to_none=True)                                     # 清空梯度（置 None 比置 0 更省内存）

        if step % args.log_interval == 0 or step == iters:
            spend_time = time.time() - start_time
            # loss 之前除以了 accumulation_steps，这里乘回去还原成"单步真实 loss"
            current_loss = loss.item() * args.accumulation_steps
            current_ce_loss = ce_loss_raw.item()
            current_aux_loss = res.aux_loss.item() if lm_config_student.use_moe else 0.0
            current_lr = optimizer.param_groups[-1]['lr']
            # 估算剩余时间（分钟）：已耗时 / 已走步数 × 剩余步数
            eta_min = spend_time / max(step - start_step, 1) * (iters - step) // 60
            
            Logger(f'Epoch:[{epoch + 1}/{args.epochs}]({step}/{iters}), loss: {current_loss:.4f}, ce: {current_ce_loss:.4f}, aux_loss: {current_aux_loss:.4f}, distill: {distill_loss.item():.4f}, learning_rate: {current_lr:.8f}, epoch_time: {eta_min:.3f}min')
            
            if wandb:
                wandb.log({
                    "loss": current_loss,
                    "ce_loss": current_ce_loss,
                    "aux_loss": current_aux_loss,
                    "distill_loss": distill_loss.item() if teacher_model is not None else 0.0,
                    "learning_rate": current_lr,
                    "epoch_time": eta_min
                })

        # 只在主进程保存，避免多卡重复写同一个文件
        if (step % args.save_interval == 0 or step == iters) and is_main_process():
            model.eval()
            moe_suffix = '_moe' if lm_config_student.use_moe else ''
            ckp = f'{args.save_dir}/{args.save_weight}_{lm_config_student.hidden_size}{moe_suffix}.pth'
            # 取回"原始模型"：DDP 会包一层 module、torch.compile 会包一层 _orig_mod，逐层剥掉
            raw_model = model.module if isinstance(model, DistributedDataParallel) else model
            raw_model = getattr(raw_model, '_orig_mod', raw_model)
            state_dict = raw_model.state_dict()
            # 权重转 fp16 存 CPU，省磁盘空间（这是仅用于推理加载的纯权重文件）
            torch.save({k: v.half().cpu() for k, v in state_dict.items()}, ckp)
            # 额外保存 resume 检查点（含 optimizer/scaler/epoch/step 状态），用于断点续训
            lm_checkpoint(lm_config_student, weight=args.save_weight, model=model, optimizer=optimizer, scaler=scaler, epoch=epoch, step=step, wandb=wandb, save_dir='../checkpoints')
            model.train()   # 恢复训练模式（save 前刚切到了 eval）
            del state_dict

        del input_ids, labels, loss_mask, res, student_logits, ce_loss, distill_loss, loss

    # epoch 结束时若累积步数没被整除（最后一批不足一组），把残留梯度也执行一次更新，
    # 避免最后几个 step 的梯度被白白丢弃
    if last_step > start_step and last_step % args.accumulation_steps != 0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)


if __name__ == "__main__":
    # 模拟用moe模型蒸馏dense模型，也可以用更大teacher_hidden_size模型蒸馏更小student_hidden_size的
    parser = argparse.ArgumentParser(description="MiniMind Knowledge Distillation")
    parser.add_argument("--save_dir", type=str, default="../out", help="模型保存目录")
    parser.add_argument('--save_weight', default='full_dist', type=str, help="保存权重的前缀名")
    parser.add_argument("--epochs", type=int, default=6, help="训练轮数")
    parser.add_argument("--batch_size", type=int, default=32, help="batch size")
    parser.add_argument("--learning_rate", type=float, default=5e-6, help="初始学习率")
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu", help="训练设备")
    parser.add_argument("--dtype", type=str, default="bfloat16", help="混合精度类型")
    parser.add_argument("--num_workers", type=int, default=8, help="数据加载线程数")
    parser.add_argument("--accumulation_steps", type=int, default=1, help="梯度累积步数")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="梯度裁剪阈值")
    parser.add_argument("--log_interval", type=int, default=100, help="日志打印间隔")
    parser.add_argument("--save_interval", type=int, default=100, help="模型保存间隔")
    parser.add_argument("--max_seq_len", type=int, default=340, help="训练的最大截断长度（中文1token≈1.5~1.7字符）")
    parser.add_argument("--data_path", type=str, default="../dataset/sft_t2t_mini.jsonl", help="训练数据路径")
    parser.add_argument('--student_hidden_size', default=768, type=int, help="学生模型隐藏层维度")
    parser.add_argument('--student_num_layers', default=8, type=int, help="学生模型隐藏层数量")
    parser.add_argument('--teacher_hidden_size', default=768, type=int, help="教师模型隐藏层维度")
    parser.add_argument('--teacher_num_layers', default=8, type=int, help="教师模型隐藏层数量")
    parser.add_argument('--student_use_moe', default=0, type=int, choices=[0, 1], help="学生模型是否使用MoE（0=否，1=是）")
    parser.add_argument('--teacher_use_moe', default=1, type=int, choices=[0, 1], help="教师模型是否使用MoE（0=否，1=是）")
    parser.add_argument('--from_student_weight', default='full_sft', type=str, help="学生模型基于哪个权重")
    parser.add_argument('--from_teacher_weight', default='full_sft', type=str, help="教师模型基于哪个权重")
    parser.add_argument('--from_resume', default=0, type=int, choices=[0, 1], help="是否自动检测&续训（0=否，1=是）")
    parser.add_argument('--alpha', default=0.5, type=float, help="CE损失权重，总损失=alpha*CE+(1-alpha)*KL")
    parser.add_argument('--temperature', default=1.5, type=float, help="蒸馏温度（推荐范围1.0-2.0）")
    parser.add_argument("--use_wandb", action="store_true", help="是否使用wandb")
    parser.add_argument("--wandb_project", type=str, default="MiniMind-Distillation", help="wandb项目名")
    parser.add_argument("--use_compile", default=0, type=int, choices=[0, 1], help="是否使用torch.compile加速（0=否，1=是）")
    args = parser.parse_args()

    # ========== 1. 初始化环境和随机种子 ==========
    local_rank = init_distributed_mode()      # 检测是否 DDP 模式（读环境变量 RANK），单卡时返回 0
    if dist.is_initialized(): args.device = f"cuda:{local_rank}"
    # 每个 rank 用不同种子（42+rank）：多卡时各卡数据打乱顺序略有差异，可增加样本多样性
    setup_seed(42 + (dist.get_rank() if dist.is_initialized() else 0))
    
    # ========== 2. 配置目录、模型参数、检查ckp ==========
    os.makedirs(args.save_dir, exist_ok=True)
    lm_config_student = MiniMindConfig(hidden_size=args.student_hidden_size, num_hidden_layers=args.student_num_layers, use_moe=bool(args.student_use_moe))
    lm_config_teacher = MiniMindConfig(hidden_size=args.teacher_hidden_size, num_hidden_layers=args.teacher_num_layers, use_moe=bool(args.teacher_use_moe))
    ckp_data = lm_checkpoint(lm_config_student, weight=args.save_weight, save_dir='../checkpoints') if args.from_resume==1 else None
    
    # ========== 3. 设置混合精度 ==========
    device_type = "cuda" if "cuda" in args.device else "cpu"
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
    # CPU 上不启用 autocast（用空上下文 nullcontext 占位）；GPU 上用 bf16/fp16 自动降精度
    autocast_ctx = nullcontext() if device_type == "cpu" else torch.cuda.amp.autocast(dtype=dtype)
    
    # ========== 4. 配wandb ==========
    wandb = None
    if args.use_wandb and is_main_process():
        import swanlab as wandb
        wandb_id = ckp_data.get('wandb_id') if ckp_data else None
        resume = 'must' if wandb_id else None
        wandb_run_name = f"MiniMind-Distill-S{args.student_hidden_size}T{args.teacher_hidden_size}-Epoch-{args.epochs}-BS-{args.batch_size}-LR-{args.learning_rate}"
        wandb.init(project=args.wandb_project, name=wandb_run_name, id=wandb_id, resume=resume)
    
    # ========== 5. 定义学生和教师模型 ==========
    # init_model 会从 ../out/{weight}_{hidden}{_moe}.pth 加载预训练权重（如 full_sft_768.pth）
    model, tokenizer = init_model(lm_config_student, args.from_student_weight, device=args.device)
    Logger(f'学生模型总参数量：{sum(p.numel() for p in model.parameters()) / 1e6:.3f} M')
    teacher_model, _ = init_model(lm_config_teacher, args.from_teacher_weight, device=args.device)
    teacher_model.eval()                # 教师切 eval（关闭 dropout 等随机性，输出更稳定）
    teacher_model.requires_grad_(False) # 教师冻结，不参与梯度计算
    Logger(f'教师模型总参数量：{sum(p.numel() for p in teacher_model.parameters()) / 1e6:.3f} M')
    train_ds = SFTDataset(args.data_path, tokenizer, max_length=args.max_seq_len)
    train_sampler = DistributedSampler(train_ds) if dist.is_initialized() else None  # 多卡时每卡分片
    # GradScaler 只在 fp16 时启用：bf16 的动态范围与 fp32 相同，无需 scale 放大梯度
    scaler = torch.cuda.amp.GradScaler(enabled=(args.dtype == 'float16'))
    # 注意 optimizer 只收录 model（学生）的参数，teacher 不会被更新
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
        train_sampler and train_sampler.set_epoch(epoch)   # 每个 epoch 重新打乱数据顺序（DDP 采样器需要）
        setup_seed(42 + epoch); indices = torch.randperm(len(train_ds)).tolist()  # 单卡时手动打乱索引
        # 续训场景：只在首个 epoch 跳过已训练过的前 start_step 个 batch
        skip = start_step if (epoch == start_epoch and start_step > 0) else 0
        # SkipBatchSampler：从打乱后的索引里，跳过前 skip 个 batch 再喂给 DataLoader
        batch_sampler = SkipBatchSampler(train_sampler or indices, args.batch_size, skip)
        loader = DataLoader(train_ds, batch_sampler=batch_sampler, num_workers=args.num_workers, pin_memory=True)
        if skip > 0:
            Logger(f'Epoch [{epoch + 1}/{args.epochs}]: 跳过前{start_step}个step，从step {start_step + 1}开始')
            # 注意 iters 传 len(loader)+skip：因为 enumerate 从 start_step+1 起，
            # 需要补上跳过的步数，iters 才等于"完整总步数"，供 get_lr 和进度打印正确计算
            train_epoch(epoch, loader, len(loader) + skip, teacher_model, lm_config_student, start_step, wandb, args.alpha, args.temperature)
        else:
            train_epoch(epoch, loader, len(loader), teacher_model, lm_config_student, 0, wandb, args.alpha, args.temperature)
    
    # ========== 9. 清理分布进程 ==========
    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()