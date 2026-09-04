"""
MiniMind LoRA 微调训练脚本
=============================

一句话理解 LoRA 训练的本质：**冻结预训练大模型的全部权重，只训练额外注入的「低秩增量」矩阵**。

与全量微调（train_full.py）的区别，从「训练什么参数」这个角度看：

    全量微调：更新模型所有参数 W（MiniMind-768 约 16~17M 个参数）。
    LoRA 微调：W 全部 requires_grad=False 冻结；只更新 ΔW = B·A。
                其中 A 形状 [rank, in]、B 形状 [out, rank]，参数量从 in×out 降到 rank×(in+out)。

本脚本的 LoRA 注入策略（见 model/model_lora.py 的 apply_lora）：
    只给「方阵线性层」（in_features == out_features）注入 LoRA 分支。
    在 MiniMind-768 里，满足条件的正好是每层注意力里的：
        q_proj：768 -> 8*96 = 768（方阵 ✓）
        o_proj：768 -> 768          （方阵 ✓）
        k_proj/v_proj：768 -> 4*96 = 384（非方阵 ✗，跳过）
    共 8 层，每层 2 个方阵层 → 16 个 LoRA 模块。

参数量具体算一笔账（hidden_size=768, rank=16）：
    单个 LoRA 模块：A [16,768] + B [768,16] = 12288 + 12288 = 24576 参数
    每层 2 个模块（q_proj + o_proj）= 49152 参数
    8 层合计 = 49152 × 8 = 393216 ≈ 0.39M 参数
    相比 16M 全量参数，占比约 2%~4%（脚本第 137 行会动态打印精确占比）。

训练主流程（main 函数）：
    1. 初始化分布式环境 & 随机种子
    2. 构造 MiniMindConfig、检测 checkpoint（是否续训）
    3. 配置混合精度（bf16/fp16 autocast + GradScaler）
    4. 配置 wandb（本项目用 swanlab 替代）
    5. 加载基础权重 -> apply_lora 注入 LoRA -> 冻结非 LoRA 参数
    6. 构造 SFT 数据集、优化器（只优化 LoRA 参数）
    7. 若续训，恢复 epoch/step/optimizer/scaler 状态
    8. 分布式包装（DDP）
    9. 逐 epoch 训练
    10. 清理进程组
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
import torch.distributed as dist
from contextlib import nullcontext
from torch import optim, nn
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler
from model.model_minimind import MiniMindConfig
from dataset.lm_dataset import SFTDataset
from model.model_lora import save_lora, apply_lora
from trainer.trainer_utils import get_lr, Logger, is_main_process, lm_checkpoint, init_distributed_mode, setup_seed, init_model, SkipBatchSampler

warnings.filterwarnings('ignore')


def train_epoch(epoch, loader, iters, lora_params, start_step=0, wandb=None):
    """
    训练一个 epoch。

    参数说明：
        epoch         : 当前是第几个 epoch（从 0 开始，用于日志显示和 LR 调度）
        loader        : DataLoader，每次 yield 一个 (input_ids, labels) batch
        iters         : 本 epoch 的总 step 数（续训时 = 实际 step + 跳过的 step，见 main 第 179 行）
        lora_params   : 可训练的 LoRA 参数列表（传给 clip_grad_norm_ 做梯度裁剪）
        start_step    : 续训时的起始 step，从 start_step+1 继续枚举（默认 0 表示从头开始）
        wandb         : swanlab 句柄，主进程用于记录指标

    核心流程（每步）：
        前向 -> 取 loss -> 除以累积步数 -> 反向（梯度累积）-> 每 accumulation_steps 步更新一次参数
        -> 周期性打印日志 / 保存 checkpoint。
    """
    start_time = time.time()
    last_step = start_step  # 记录本 epoch 实际走到过的最晚 step，用于末尾判断是否还有未 flush 的累积梯度
    # enumerate 从 start_step+1 开始编号：续训时 step 编号连续，日志能接上上次停下的位置
    for step, (input_ids, labels) in enumerate(loader, start=start_step + 1):
        # ===== 1. 数据搬到设备 =====
        # input_ids/labels 形状都是 [B, S]，元素为 token id（长整型）。
        # 例：B=32, S=340 -> [32, 340] 的 int64 张量，约 32*340*8B ≈ 87KB
        input_ids = input_ids.to(args.device)
        labels = labels.to(args.device)
        last_step = step

        # ===== 2. 余弦退火学习率调度（cosine annealing） =====
        # get_lr(step, total, lr) = lr * (0.1 + 0.45 * (1 + cos(π * step / total)))
        # 学习率从 lr 平滑衰减到 0.1*lr（带 10% 的 warmup 下限，避免末尾 LR 归零导致训练停滞）。
        # 具体数值举例（lr=1e-4, total=1000 步）：
        #   step=0    : cos(0)      = 1  -> lr = 1e-4 * (0.1 + 0.45*2) = 1e-4 * 1.0 = 1e-4  （初始 LR）
        #   step=500  : cos(π/2)    = 0  -> lr = 1e-4 * (0.1 + 0.45*1) = 1e-4 * 0.55 = 5.5e-5 （半程）
        #   step=1000 : cos(π)      = -1 -> lr = 1e-4 * (0.1 + 0.45*0) = 1e-4 * 0.1 = 1e-5  （末尾，降到 1/10）
        lr = get_lr(epoch * iters + step, args.epochs * iters, args.learning_rate)
        # 把新 LR 写进优化器的每个参数组（这里只有一个 param_group，即 lora_params）
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        # ===== 3. 前向 + 计算损失（混合精度 autocast 下进行） =====
        with autocast_ctx:
            # model(input_ids, labels=labels) 走 MiniMindForCausalLM.forward：
            #   hidden_states -> lm_head -> logits [B, S, 6400]
            #   loss = 交叉熵（next-token 预测，ignore_index=-100 忽略 padding/非回答位置）
            #   aux_loss = MoE 负载均衡辅助损失（非 MoE 时为标量 0）
            res = model(input_ids, labels=labels)
            # 总损失 = 主语言建模损失 + MoE 辅助损失。
            # 例：res.loss=2.31（logits_loss），res.aux_loss=0.0005（只有 use_moe=1 时才非 0）
            #     总 loss = 2.31 + 0.0005 = 2.3105
            loss = res.loss + res.aux_loss
            # ===== 梯度累积归一化 =====
            # 除以 accumulation_steps 是梯度累积的关键：让 loss 按「平均」而非「求和」缩放，
            # 这样累积 N 个 micro-batch 后的梯度等价于一个 batch_size×N 的大 batch 的平均梯度。
            # 例：accumulation_steps=4，4 个 micro-batch 的 loss 各除以 4 再累加，
            #     最终梯度 = 1/4 * Σ(grad_i)，等价于「一个 batch_size×4 的 batch 反向一次」。
            loss = loss / args.accumulation_steps

        # ===== 4. 反向传播（梯度累积：不清零，梯度逐次累加） =====
        # scaler.scale 用于 fp16 混合精度：把 loss 放大（如 ×2^N）再反向，避免 fp16 梯度下溢到 0。
        # bf16 时 GradScaler 被禁用（enabled=False），scale 相当于恒等变换。
        scaler.scale(loss).backward()

        # ===== 5. 每 accumulation_steps 步做一次参数更新 =====
        # 例：accumulation_steps=4 -> 前 3 步只累积梯度，第 4 步才 unscale/clip/step/zero_grad
        if step % args.accumulation_steps == 0:
            # 5a. unscale：把放大过的梯度还原回真实尺度，供 clip 和 step 使用
            scaler.unscale_(optimizer)
            # 5b. 梯度裁剪（只对 LoRA 参数）：防止某个 batch 产生爆炸梯度把训练带偏。
            #     具体计算（以 2 个参数的梯度向量 g = [3.0, 4.0] 为例）：
            #       总范数 ||g|| = sqrt(3² + 4²) = 5.0
            #       设 grad_clip=1.0，则缩放系数 = clip / max(1.0, 5.0) = 0.2
            #       裁剪后 g = [0.6, 0.8]，范数恰好 = 1.0（方向不变，长度截断到阈值）
            torch.nn.utils.clip_grad_norm_(lora_params, args.grad_clip)
            # 5c. 用裁剪后的梯度更新 LoRA 参数
            scaler.step(optimizer)
            # 5d. 更新 loss scale（fp16 下动态调整；bf16 下为 no-op）
            scaler.update()
            # 5e. 清空累积的梯度，准备下一轮累积。set_to_none=True 把梯度置 None 而非 0，更省显存
            optimizer.zero_grad(set_to_none=True)

        # ===== 6. 周期打印日志 =====
        if step % args.log_interval == 0 or step == iters:
            spend_time = time.time() - start_time
            # 之前 loss 除了 accumulation_steps，这里乘回来恢复「真实单 batch 损失」用于可读的日志
            current_loss = loss.item() * args.accumulation_steps
            current_aux_loss = res.aux_loss.item() if res.aux_loss is not None else 0.0
            # 主 LM 损失 = 总损失 - MoE 辅助损失（拆开看更方便观察收敛）
            current_logits_loss = current_loss - current_aux_loss
            current_lr = optimizer.param_groups[-1]['lr']
            # 估算剩余时间：已用时间 / 已完成 step 数 × 剩余 step 数，单位分钟
            eta_min = spend_time / max(step - start_step, 1) * (iters - step) // 60
            Logger(f'Epoch:[{epoch + 1}/{args.epochs}]({step}/{iters}), loss: {current_loss:.4f}, logits_loss: {current_logits_loss:.4f}, aux_loss: {current_aux_loss:.4f}, lr: {current_lr:.8f}, epoch_time: {eta_min:.1f}min')
            if wandb: wandb.log({"loss": current_loss, "logits_loss": current_logits_loss, "aux_loss": current_aux_loss, "learning_rate": current_lr, "epoch_time": eta_min})

        # ===== 7. 周期保存 LoRA 权重 & 训练状态 =====
        if (step % args.save_interval == 0 or step == iters) and is_main_process():
            model.eval()  # 切到 eval 模式（关 dropout），保存前避免记录训练态的随机性
            moe_suffix = '_moe' if lm_config.use_moe else ''
            lora_save_path = f'{args.save_dir}/{args.lora_name}_{lm_config.hidden_size}{moe_suffix}.pth'
            # LoRA只保存LoRA权重：只存每个方阵线性层注入的 A/B 两个小矩阵，
            # 不存冻结的基础权重，文件小（本例约 0.39M 参数 × 2 bytes(fp16) ≈ 0.8MB），便于分发/合并
            save_lora(model, lora_save_path)
            # 同时存完整训练状态（模型全量权重 + optimizer + scaler + epoch/step）到 ../checkpoints，用于断点续训
            lm_checkpoint(lm_config, weight=args.lora_name, model=model, optimizer=optimizer, scaler=scaler, epoch=epoch, step=step, wandb=wandb, save_dir='../checkpoints')
            model.train()  # 恢复训练模式

        # 及时释放引用，降低峰值显存（PyTorch 张量在引用计数归零后由缓存分配器回收）
        del input_ids, labels, res, loss

    # ===== 8. 处理 epoch 末尾「不满一个累积步」的残余梯度 =====
    # 例：本 epoch 共 25 个 step，accumulation_steps=4，则 25 % 4 = 1，最后 1 个 micro-batch 的梯度
    #     没触发过 step，需要在这里手动 flush 一次，否则它的梯度会被丢弃。
    if last_step > start_step and last_step % args.accumulation_steps != 0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(lora_params, args.grad_clip)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MiniMind LoRA Fine-tuning")
    parser.add_argument("--save_dir", type=str, default="../out", help="模型保存目录")
    parser.add_argument("--lora_name", type=str, default="lora_medical", help="LoRA权重名称(如lora_identity/lora_medical等)")
    parser.add_argument("--epochs", type=int, default=10, help="训练轮数")
    parser.add_argument("--batch_size", type=int, default=32, help="batch size")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="初始学习率")
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu", help="训练设备")
    parser.add_argument("--dtype", type=str, default="bfloat16", help="混合精度类型")
    parser.add_argument("--num_workers", type=int, default=8, help="数据加载线程数")
    parser.add_argument("--accumulation_steps", type=int, default=1, help="梯度累积步数")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="梯度裁剪阈值")
    parser.add_argument("--log_interval", type=int, default=10, help="日志打印间隔")
    parser.add_argument("--save_interval", type=int, default=1000, help="模型保存间隔")
    parser.add_argument('--hidden_size', default=768, type=int, help="隐藏层维度")
    parser.add_argument('--num_hidden_layers', default=8, type=int, help="隐藏层数量")
    parser.add_argument('--max_seq_len', default=340, type=int, help="训练的最大截断长度（中文1token≈1.5~1.7字符）")
    parser.add_argument('--use_moe', default=0, type=int, choices=[0, 1], help="是否使用MoE架构（0=否，1=是）")
    parser.add_argument("--data_path", type=str, default="../dataset/lora_medical.jsonl", help="LoRA训练数据路径")
    parser.add_argument('--from_weight', default='full_sft', type=str, help="基于哪个权重训练，默认full_sft")
    parser.add_argument('--from_resume', default=0, type=int, choices=[0, 1], help="是否自动检测&续训（0=否，1=是）")
    parser.add_argument("--use_wandb", action="store_true", help="是否使用wandb")
    parser.add_argument("--wandb_project", type=str, default="MiniMind-LoRA", help="wandb项目名")
    parser.add_argument("--use_compile", default=0, type=int, choices=[0, 1], help="是否使用torch.compile加速（0=否，1=是）")
    args = parser.parse_args()

    # ========== 1. 初始化环境和随机种子 ==========
    # 通过环境变量 RANK 判断是否处于 DDP 模式；非 DDP 返回 0。
    # DDP 模式下会 init_process_group(nccl) 并 set_device(local_rank)
    local_rank = init_distributed_mode()
    if dist.is_initialized(): args.device = f"cuda:{local_rank}"
    # 种子 = 42 + rank，保证多卡时各进程种子不同（避免各卡生成完全相同的 dropout/采样序列）
    setup_seed(42 + (dist.get_rank() if dist.is_initialized() else 0))

    # ========== 2. 配置目录、模型参数、检查ckp ==========
    os.makedirs(args.save_dir, exist_ok=True)
    # 用命令行参数构造模型配置（hidden_size=768, num_hidden_layers=8, use_moe=0/1）
    lm_config = MiniMindConfig(hidden_size=args.hidden_size, num_hidden_layers=args.num_hidden_layers, use_moe=bool(args.use_moe))
    # from_resume=1 时自动检测 ../checkpoints 下的续训文件；否则 ckp_data=None 表示从头训练
    ckp_data = lm_checkpoint(lm_config, weight=args.lora_name, save_dir='../checkpoints') if args.from_resume==1 else None

    # ========== 3. 设置混合精度 ==========
    # device_type 决定是否启用 autocast（CPU 不支持 AMP，用 nullcontext 绕过）
    device_type = "cuda" if "cuda" in args.device else "cpu"
    # bf16 无下溢风险（指数位宽），GradScaler 禁用；fp16 需要 GradScaler 防梯度下溢
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
    autocast_ctx = nullcontext() if device_type == "cpu" else torch.cuda.amp.autocast(dtype=dtype)

    # ========== 4. 配wandb ==========
    # 本项目用 swanlab 替代 wandb（import swanlab as wandb），接口兼容
    wandb = None
    if args.use_wandb and is_main_process():
        import swanlab as wandb
        wandb_id = ckp_data.get('wandb_id') if ckp_data else None
        resume = 'must' if wandb_id else None  # 有旧 run id 则续接同一个 run，否则新建
        wandb_run_name = f"MiniMind-LoRA-{args.lora_name}-Epoch-{args.epochs}-BatchSize-{args.batch_size}-LR-{args.learning_rate}"
        wandb.init(project=args.wandb_project, name=wandb_run_name, id=wandb_id, resume=resume)

    # ========== 5. 定义模型、应用LoRA、冻结非LoRA参数 ==========
    # init_model：加载 AutoTokenizer + 构造 MiniMindForCausalLM，并从 ../out/{from_weight}_768.pth 加载基础权重
    model, tokenizer = init_model(lm_config, args.from_weight, device=args.device)
    # apply_lora：给所有「方阵线性层」注入 LoRA 分支（此处只注入 q_proj/o_proj，共 16 个模块）
    apply_lora(model)

    # 统计参数
    total_params = sum(p.numel() for p in model.parameters())
    # 统计名字含 'lora' 的参数（即注入的 A/B 两个小矩阵的所有权重）
    lora_params_count = sum(p.numel() for name, p in model.named_parameters() if 'lora' in name)
    Logger(f"LLM 总参数量: {total_params / 1e6:.3f} M")
    Logger(f"LoRA 参数量: {lora_params_count / 1e6:.3f} M")
    Logger(f"LoRA 参数占比: {lora_params_count / total_params * 100:.2f}%")
    # 具体数值举例（hidden=768, rank=16）：
    #   总参数量 ≈ 16.0 M；LoRA 参数量 ≈ 0.393 M；占比 ≈ 2.4%

    # 冻结非LoRA参数，收集LoRA参数
    # 关键：基础权重 requires_grad=False（不参与反向），只有 LoRA 的 A/B 参与梯度更新。
    # 这样：
    #   1) 显存中只存 LoRA 参数的梯度与 Adam 状态，大幅省显存；
    #   2) 保存 checkpoint 时只需存 LoRA 增量，基础权重保持原样不动。
    lora_params = []
    for name, param in model.named_parameters():
        if 'lora' in name:
            param.requires_grad = True
            lora_params.append(param)
        else:
            param.requires_grad = False

    # ========== 6. 定义数据和优化器 ==========
    # SFTDataset：把对话模板化后 tokenize，返回 (input_ids [S], labels [S])，labels 只在「assistant 回答段」有值，其余为 -100
    train_ds = SFTDataset(args.data_path, tokenizer, max_length=args.max_seq_len)
    # DDP 模式下用 DistributedSampler 让各卡分到不重叠的数据分片；单卡为 None
    train_sampler = DistributedSampler(train_ds) if dist.is_initialized() else None
    # GradScaler 仅在 fp16 时启用（enabled=True）；bf16 时禁用（enabled=False），避免无谓的放大/缩小
    scaler = torch.cuda.amp.GradScaler(enabled=(args.dtype == 'float16'))
    # 优化器只优化 LoRA 参数（lora_params 列表），基础权重已被冻结、不传入
    optimizer = optim.AdamW(lora_params, lr=args.learning_rate)

    # ========== 7. 从ckp恢复状态 ==========
    start_epoch, start_step = 0, 0
    if ckp_data:
        # strict=False：允许 checkpoint 里多/少一些 key（如 LoRA 权重不匹配时仍可加载基础权重）
        model.load_state_dict(ckp_data['model'], strict=False)
        optimizer.load_state_dict(ckp_data['optimizer'])
        scaler.load_state_dict(ckp_data['scaler'])
        start_epoch = ckp_data['epoch']
        start_step = ckp_data.get('step', 0)

    # ========== 8. 编译和分布式包装 ==========
    # LoRA 用 monkey-patch 替换了 nn.Linear 的 forward，与 torch.compile 的图捕获不兼容，故自动关闭
    if args.use_compile == 1:
        args.use_compile = 0
        Logger('[LoRA] monkey-patch forward 与 torch.compile 不兼容，use_compile 已自动关闭')
    # DDP 包装：多卡训练时同步各卡梯度（all-reduce）并广播参数
    if dist.is_initialized():
        model = DistributedDataParallel(model, device_ids=[local_rank])

    # ========== 9. 开始训练 ==========
    for epoch in range(start_epoch, args.epochs):
        # DDP 下每个 epoch 重新 shuffle（DistributedSampler.set_epoch 保证各 epoch 分片顺序不同）
        train_sampler and train_sampler.set_epoch(epoch)
        # 重新设置种子 + 生成打乱后的样本索引（单卡模式用 randperm 实现 shuffle）
        setup_seed(42 + epoch); indices = torch.randperm(len(train_ds)).tolist()
        # 续训时：首个 epoch 跳过前 start_step 个 step，从上次停下的位置继续
        skip = start_step if (epoch == start_epoch and start_step > 0) else 0
        # SkipBatchSampler：把索引按 batch_size 分组，并跳过前 skip 个 batch（实现断点续训的「跳过已训练数据」）
        batch_sampler = SkipBatchSampler(train_sampler or indices, args.batch_size, skip)
        loader = DataLoader(train_ds, batch_sampler=batch_sampler, num_workers=args.num_workers, pin_memory=True)
        if skip > 0:
            Logger(f'Epoch [{epoch + 1}/{args.epochs}]: 跳过前{start_step}个step，从step {start_step + 1}开始')
            # iters 传 len(loader)+skip，使 step 编号连续、LR 调度按「全量 step」计算
            train_epoch(epoch, loader, len(loader) + skip, lora_params, start_step, wandb)
        else:
            train_epoch(epoch, loader, len(loader), lora_params, 0, wandb)

    # ========== 10. 清理分布进程 ==========
    if dist.is_initialized():
        dist.barrier()  # 所有卡同步到同一进度后再退出，避免某卡提前销毁通信组
        dist.destroy_process_group()
