import os
import sys

# ========== 模块导入与路径处理 ==========
# 脚本是直接 `python train_full_sft.py` 运行的（不是 `python -m trainer.train_full_sft`），
# 此时 Python 会把 __package__ 设为空字符串，导致 `from model.model_minimind import ...`
# 这类「包内绝对导入」找不到包。手动把 __package__ 置为 "trainer" 即可解决。
__package__ = "trainer"
# 把上级目录（minimind 根目录）加进模块搜索路径，这样 `model/`、`dataset/`、`trainer/`
# 三个包才能被 import。__file__ 是当前文件路径，'..' 即 minimind 根目录。
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import datasets  # noqa: F401  # Windows pyarrow/torch DLL 冲突 workaround (issue #771)
# ↑ 一个 Windows 平台的坑：pyarrow（HuggingFace datasets 的依赖）和 torch 的 DLL
#   加载顺序冲突会报错。在 import torch 之前先 import datasets 可规避该冲突。
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
from trainer.trainer_utils import get_lr, Logger, is_main_process, lm_checkpoint, init_distributed_mode, setup_seed, init_model, SkipBatchSampler

warnings.filterwarnings('ignore')


def train_epoch(epoch, loader, iters, start_step=0, wandb=None):
    """
    训练一个 epoch。

    参数：
        epoch       : 当前是第几个 epoch（从 0 开始），只用于打印进度。
        loader      : DataLoader，每次 yield 一批 (input_ids [B, S], labels [B, S])。
        iters       : 本 epoch 的总步数（用于算学习率和进度打印）。
        start_step  : 续训时从第几步开始（>0 表示跳过了前 start_step 步，日志里显示得连贯）。
        wandb       : 日志记录器（实际是 swanlab），None 表示不记录。

    一个 step 的完整流程（以 batch_size=16、max_seq_len=768、accumulation_steps=1 为例）：
        data -> GPU -> 算学习率 -> autocast 前向算 loss -> 反传 -> （累积满 N 步后）梯度裁剪+更新+清梯度
        -> 打日志 -> 存权重 -> 释放显存

    关键点：
        1. 混合精度：前向用 autocast_ctx（默认 bf16），反向用 GradScaler（仅 fp16 时启用）。
        2. 梯度累积：loss 先除以 accumulation_steps，累积满 N 步才真正 step 一次，
           等效于 batch_size 放大 N 倍，同时保持梯度量级不变。
        3. 梯度裁剪：clip_grad_norm_ 把梯度范数压到 grad_clip 内，防梯度爆炸。
        4. 结尾冲刷：若本 epoch 步数不是 N 的整数倍，末尾不足 N 步的残留梯度也要手动更新一次。
    """
    start_time = time.time()
    last_step = start_step
    # enumerate(loader, start=start_step+1)：step 编号从 1 开始（续训时从 start_step+1 开始），
    # 保证日志里的步号全局连续，不会每 epoch 都从 1 重新数。
    for step, (input_ids, labels) in enumerate(loader, start=start_step + 1):
        # ---- 1. 数据搬到训练设备 ----
        # 例：input_ids/labels 形状都是 [16, 768]（16 条对话、每条固定 768 token）。
        input_ids = input_ids.to(args.device)
        labels = labels.to(args.device)
        last_step = step

        # ---- 2. 更新学习率（余弦退火，逐 step 衰减）----
        # get_lr 的入参是「全局步数」：epoch*iters+step，总步数 = epochs*iters。
        # 返回 lr*(0.1 + 0.45*(1+cos(π*cur/total)))，从 1.0*lr 余弦衰减到 0.1*lr。
        lr = get_lr(epoch * iters + step, args.epochs * iters, args.learning_rate)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        # ---- 3. 前向 + 算损失（混合精度上下文内）----
        with autocast_ctx:
            # model(input_ids, labels=labels) 内部：算出 logits [B,S,6400] 后，
            # 做 next-token 错位（x=logits[:-1] 预测 y=labels[1:]），
            # 再用 CrossEntropyLoss(ignore_index=-100) 只对 labels 非 -100 的位置（assistant 回答段）求 loss。
            # 返回的 res 是 MoeCausalLMOutputWithPast 对象，含 .loss 和 .aux_loss。
            res = model(input_ids, labels=labels)
            # 总损失 = 语言建模损失（交叉熵）+ MoE 负载均衡辅助损失（非 MoE 时为 0）。
            # 例：use_moe=0 时 aux_loss 恒为 0，loss 就只剩交叉熵。
            loss = res.loss + res.aux_loss
            # 除以累积步数：梯度累积 N 步等于把 batch_size 放大 N 倍，
            # 每一步梯度只贡献 1/N，N 步加起来才等于一个大 batch 的梯度量级。
            loss = loss / args.accumulation_steps

        # ---- 4. 反向传播（梯度累积：先不更新，只累加梯度）----
        # scaler.scale：fp16 时把 loss 放大（防止下溢），bf16 时 scaler 被禁用、scale 是恒等映射。
        # .backward() 累加梯度到各参数 .grad，不清空——这正是「梯度累积」的机制。
        scaler.scale(loss).backward()

        # ---- 5. 累积满 accumulation_steps 步才真正更新一次参数 ----
        if step % args.accumulation_steps == 0:
            # unscale_：fp16 时把梯度按之前的 scale 系数还原回来；bf16 时 no-op。
            scaler.unscale_(optimizer)
            # 梯度裁剪：把整个模型所有参数的梯度范数限制在 grad_clip（默认 1.0）内，
            # 防止某一步梯度爆炸把权重冲飞。
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

            # 真正更新参数（AdamW）。fp16 时 scaler 会先检测梯度是否溢出，溢出则跳过本次更新。
            scaler.step(optimizer)
            # 更新 scaler 内部的缩放系数（动态调整 loss 放大倍数）。
            scaler.update()

            # 清空梯度，set_to_none=True 直接把 .grad 置 None（比 .zero_() 更省显存）。
            optimizer.zero_grad(set_to_none=True)

        # ---- 6. 打印日志 ----
        # 每 log_interval 步（默认 100）或最后一步打印一次。
        if step % args.log_interval == 0 or step == iters:
            spend_time = time.time() - start_time
            # 注意：当前 loss 是被除以了 accumulation_steps 的，所以要乘回去还原真实量级。
            current_loss = loss.item() * args.accumulation_steps
            # aux_loss 仅在 MoE 模式下非零；非 MoE 时 res.aux_loss 是标量 0（非 None），
            # 这里三元判断只是防御性写法。
            current_aux_loss = res.aux_loss.item() if res.aux_loss is not None else 0.0
            # logits_loss = 纯语言建模损失（总 loss 减掉 MoE 辅助损失）。
            current_logits_loss = current_loss - current_aux_loss
            # 取最后一个参数组的 lr（所有参数组 lr 相同，取哪个都行）。
            current_lr = optimizer.param_groups[-1]['lr']
            # eta_min：按当前平均速度估算「跑完剩余步数还要多少分钟」。
            # spend_time/(step-start_step) = 平均每步耗时，× 剩余步数(iters-step)，//60 转分钟。
            eta_min = spend_time / max(step - start_step, 1) * (iters - step) // 60
            Logger(f'Epoch:[{epoch + 1}/{args.epochs}]({step}/{iters}), loss: {current_loss:.4f}, logits_loss: {current_logits_loss:.4f}, aux_loss: {current_aux_loss:.4f}, lr: {current_lr:.8f}, epoch_time: {eta_min:.1f}min')
            if wandb: wandb.log({"loss": current_loss, "logits_loss": current_logits_loss, "aux_loss": current_aux_loss, "learning_rate": current_lr, "epoch_time": eta_min})

        # ---- 7. 保存模型 ----
        # 每 save_interval 步（默认 1000）或最后一步保存，且只主进程保存（避免多卡重复写盘）。
        if (step % args.save_interval == 0 or step == iters) and is_main_process():
            model.eval()   # 切 eval 模式（关 dropout 等），保存的权重更干净
            moe_suffix = '_moe' if lm_config.use_moe else ''
            # 权重文件命名：full_sft_768.pth（MoE 则是 full_sft_768_moe.pth）
            ckp = f'{args.save_dir}/{args.save_weight}_{lm_config.hidden_size}{moe_suffix}.pth'
            # 剥掉包装拿到「真正的模型」：DDP 外壳（model.module）-> torch.compile 的 _orig_mod 包装。
            raw_model = model.module if isinstance(model, DistributedDataParallel) else model
            raw_model = getattr(raw_model, '_orig_mod', raw_model)
            state_dict = raw_model.state_dict()
            # 转 half()（fp16）再搬到 CPU 存盘，省一半磁盘空间；推理时也是用这份半精度权重。
            torch.save({k: v.half().cpu() for k, v in state_dict.items()}, ckp)
            # 另存一份「resume 检查点」到 ../checkpoints，额外带 optimizer/scaler/epoch/step/wandb_id，
            # 供 --from_resume 续训时恢复完整训练状态（光有模型权重不够续训）。
            lm_checkpoint(lm_config, weight=args.save_weight, model=model, optimizer=optimizer,
                         epoch=epoch, step=step, wandb=wandb, save_dir='../checkpoints', scaler=scaler)
            model.train()   # 存完切回 train 模式继续训练
            del state_dict  # 及时释放 state_dict 占的显存/内存

        # ---- 8. 及时释放本步的中间变量，省显存 ----
        del input_ids, labels, res, loss

    # ---- 9. 尾部梯度冲刷 ----
    # 若本 epoch 实际跑了步数不是 accumulation_steps 的整数倍（例如累积 4 步、却剩 3 步），
    # 循环里最后一个 if 分支没触发，这 3 步的梯度会永远不被更新而白算。
    # 这里补一次：把残留的不足 N 步梯度也裁剪 + 更新 + 清空。
    if last_step > start_step and last_step % args.accumulation_steps != 0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)


if __name__ == "__main__":
    # ========== 命令行参数定义 ==========
    parser = argparse.ArgumentParser(description="MiniMind Full SFT")
    parser.add_argument("--save_dir", type=str, default="../out", help="模型保存目录")
    parser.add_argument('--save_weight', default='full_sft', type=str, help="保存权重的前缀名")
    parser.add_argument("--epochs", type=int, default=2, help="训练轮数")
    parser.add_argument("--batch_size", type=int, default=16, help="batch size")
    parser.add_argument("--learning_rate", type=float, default=1e-5, help="初始学习率")
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu", help="训练设备")
    parser.add_argument("--dtype", type=str, default="bfloat16", help="混合精度类型")
    parser.add_argument("--num_workers", type=int, default=8, help="数据加载线程数")
    parser.add_argument("--accumulation_steps", type=int, default=1, help="梯度累积步数")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="梯度裁剪阈值")
    parser.add_argument("--log_interval", type=int, default=100, help="日志打印间隔")
    parser.add_argument("--save_interval", type=int, default=1000, help="模型保存间隔")
    parser.add_argument('--hidden_size', default=768, type=int, help="隐藏层维度")
    parser.add_argument('--num_hidden_layers', default=8, type=int, help="隐藏层数量")
    parser.add_argument('--max_seq_len', default=768, type=int, help="训练的最大截断长度（中文1token≈1.5~1.7字符）")
    parser.add_argument('--use_moe', default=0, type=int, choices=[0, 1], help="是否使用MoE架构（0=否，1=是）")
    parser.add_argument("--data_path", type=str, default="../dataset/sft_t2t_mini.jsonl", help="训练数据路径")
    parser.add_argument('--from_weight', default='pretrain', type=str, help="基于哪个权重训练，为none则不基于任何权重训练")
    parser.add_argument('--from_resume', default=0, type=int, choices=[0, 1], help="是否自动检测&续训（0=否，1=是）")
    parser.add_argument("--use_wandb", action="store_true", help="是否使用wandb")
    parser.add_argument("--wandb_project", type=str, default="MiniMind-Full-SFT", help="wandb项目名")
    parser.add_argument("--use_compile", default=0, type=int, choices=[0, 1], help="是否使用torch.compile加速（0=否，1=是）")
    args = parser.parse_args()

    # ========== 1. 初始化环境和随机种子 ==========
    # 判断是否多卡：有 RANK 环境变量则走 DDP 初始化，返回 local_rank；单卡则返回 0。
    local_rank = init_distributed_mode()
    # DDP 模式下，device 由 local_rank 决定（每个进程绑一张卡）。
    if dist.is_initialized(): args.device = f"cuda:{local_rank}"
    # 种子 = 42 + rank：不同卡之间种子错开，配合 DistributedSampler 保证各卡数据切分不重复。
    setup_seed(42 + (dist.get_rank() if dist.is_initialized() else 0))

    # ========== 2. 配置目录、模型参数、检查ckp ==========
    # 建输出目录（../out），存最终模型权重。
    os.makedirs(args.save_dir, exist_ok=True)
    # 用命令行超参构造模型配置对象：hidden_size=768、num_hidden_layers=8、use_moe=False。
    lm_config = MiniMindConfig(hidden_size=args.hidden_size, num_hidden_layers=args.num_hidden_layers, use_moe=bool(args.use_moe))
    # from_resume=1 时，从 ../checkpoints/full_sft_768_resume.pth 读取续训数据（含模型/优化器/步数等）；
    # from_resume=0 时 ckp_data=None，从头训练。
    ckp_data = lm_checkpoint(lm_config, weight=args.save_weight, save_dir='../checkpoints') if args.from_resume==1 else None

    # ========== 3. 设置混合精度 ==========
    # 判断设备类型（cuda / cpu），决定用 autocast 还是什么都不做。
    device_type = "cuda" if "cuda" in args.device else "cpu"
    # bf16 和 fp16 两种混合精度；CPU 不支持 amp，autocast_ctx 用 nullcontext（即关掉混合精度）。
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
    autocast_ctx = nullcontext() if device_type == "cpu" else torch.cuda.amp.autocast(dtype=dtype)

    # ========== 4. 配wandb ==========
    # 注释写 wandb，实际 `import swanlab as wandb`——用的是国产的 swanlab（接口兼容 wandb）。
    wandb = None
    if args.use_wandb and is_main_process():
        import swanlab as wandb
        # 续训时带上旧的 wandb_id，用 resume='must' 强制续上同一条日志曲线。
        wandb_id = ckp_data.get('wandb_id') if ckp_data else None
        resume = 'must' if wandb_id else None
        # 运行名把关键超参拼进去，便于在面板上区分不同实验。
        wandb_run_name = f"MiniMind-Full-SFT-Epoch-{args.epochs}-BatchSize-{args.batch_size}-LearningRate-{args.learning_rate}"
        wandb.init(project=args.wandb_project, name=wandb_run_name, id=wandb_id, resume=resume)

    # ========== 5. 定义模型、数据、优化器 ==========
    # init_model：加载 tokenizer + 构建 MiniMindForCausalLM；from_weight='pretrain' 时
    # 还会 torch.load('../out/pretrain_768.pth') 用 strict=False 加载预训练权重，最后 .to(device)。
    model, tokenizer = init_model(lm_config, args.from_weight, device=args.device)
    # SFTDataset：把每条 jsonl 对话渲染成 chat_template 文本 -> tokenize -> 只对 assistant 回答段
    # 打真实 label、其余 -100，返回 (input_ids, labels)。
    train_ds = SFTDataset(args.data_path, tokenizer, max_length=args.max_seq_len)
    # 多卡时用 DistributedSampler 把数据按卡切分，且每 epoch 打乱；单卡时为 None（后面用 randperm 打乱）。
    train_sampler = DistributedSampler(train_ds) if dist.is_initialized() else None
    # GradScaler 仅 fp16 时启用（bf16 指数位宽和 fp32 一样，不会像 fp16 那样下溢，无需缩放）。
    # 默认 dtype=bf16，所以 scaler 实际是禁用状态（scale/unscale 都是恒等操作）。
    scaler = torch.cuda.amp.GradScaler(enabled=(args.dtype == 'float16'))
    # AdamW 优化器：训练全部参数（Full SFT 不冻结任何层）。
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)

    # ========== 6. 从ckp恢复状态 ==========
    # 续训时恢复：模型权重、优化器状态、scaler 状态，以及已训到的 epoch/step（精确接着上次跑）。
    start_epoch, start_step = 0, 0
    if ckp_data:
        model.load_state_dict(ckp_data['model'])
        optimizer.load_state_dict(ckp_data['optimizer'])
        scaler.load_state_dict(ckp_data['scaler'])
        start_epoch = ckp_data['epoch']
        start_step = ckp_data.get('step', 0)

    # ========== 7. 编译和分布式包装 ==========
    # torch.compile 把模型编译成优化后的图，通常能提速（首次编译有额外开销）。
    if args.use_compile == 1:
        model = torch.compile(model)
        Logger('torch.compile enabled')
    # 多卡时用 DistributedDataParallel 包装：自动做梯度 all-reduce，各卡平均梯度。
    if dist.is_initialized():
        model = DistributedDataParallel(model, device_ids=[local_rank])

    # ========== 8. 开始训练 ==========
    for epoch in range(start_epoch, args.epochs):
        # DDP 下每 epoch 重新设置 sampler 的 epoch，保证各 epoch 数据划分不同。
        train_sampler and train_sampler.set_epoch(epoch)
        # 单卡模式：手动用 randperm 打乱整个数据集的下标（seed 与 epoch 相关，每轮打乱不同）。
        setup_seed(42 + epoch); indices = torch.randperm(len(train_ds)).tolist()
        # 续训时：只在「续训起始的那个 epoch」跳过前 start_step 步；后续 epoch 从 0 开始不跳。
        skip = start_step if (epoch == start_epoch and start_step > 0) else 0
        # SkipBatchSampler：把打乱后的下标按 batch_size 分组，并跳过前 skip 个 batch，
        # 实现「从第 start_step 步接着训」的精确对齐。
        batch_sampler = SkipBatchSampler(train_sampler or indices, args.batch_size, skip)
        # pin_memory=True：数据先放进锁页内存，加速 CPU->GPU 的异步拷贝（H2D）。
        loader = DataLoader(train_ds, batch_sampler=batch_sampler, num_workers=args.num_workers, pin_memory=True)
        if skip > 0:
            # 续训场景：提示跳过多少步，并把 iters 传成 len(loader)+skip 使步号/学习率全局连续。
            Logger(f'Epoch [{epoch + 1}/{args.epochs}]: 跳过前{start_step}个step，从step {start_step + 1}开始')
            train_epoch(epoch, loader, len(loader) + skip, start_step, wandb)
        else:
            # 正常场景：iters = 本 epoch 的 batch 数。
            train_epoch(epoch, loader, len(loader), 0, wandb)

    # ========== 9. 清理分布进程 ==========
    # barrier：所有卡同步（确保都保存完再退出）；destroy：销毁进程组，避免进程退出不同步而挂起。
    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()
