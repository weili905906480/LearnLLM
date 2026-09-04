"""
MiniMind 预训练（Pretrain）脚本。

================================ 这个脚本是干什么的 ================================
预训练是训练一个 LLM 的**第一步**：在一堆「原始纯文本」上做 next-token prediction
（下一个 token 预测），让模型学到语言的统计规律（语法、常识、世界知识……），
得到一份 **base 权重**（如 pretrain_768.pth），之后再拿去 SFT 微调、RL 对齐。

一句话流程：
    读 jsonl 纯文本 -> tokenize 成 id -> 组成 batch -> 前向算 loss -> 反向累积梯度
    -> 每 N 步更新一次参数 -> 定期打印/保存 -> 训练完得到 pretrain_768.pth

================================ 预训练 vs SFT 的本质区别 ================================
预训练没有「用户提问 / 助手回答」的角色之分，也没有 loss mask（掩码）：
    预训练：labels = 整句所有 token 都参与预测（只有 <pad> 是 -100）
    SFT   ：labels 只在 assistant 回答段填真实 id，system/user 提示段全是 -100

================================ 脚本结构 ================================
  1. train_epoch()：单个 epoch 的训练循环（前向、反向、梯度累积、日志、保存）
  2. __main__    ：解析参数 -> 初始化环境 -> 建模型/数据/优化器 -> 循环调用 train_epoch
"""
import os
import sys

# 直接 `python train_pretrain.py` 运行时，__package__ 为空、项目根目录不在 sys.path 里，
# 所以手动设置包名并把上级目录（minimind/）塞进 sys.path，
# 这样后面的 `from model.xxx import ...`、`from dataset.xxx import ...` 才能成功导入。
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
from dataset.lm_dataset import PretrainDataset
from trainer.trainer_utils import get_lr, Logger, is_main_process, lm_checkpoint, init_distributed_mode, setup_seed, init_model, SkipBatchSampler

warnings.filterwarnings('ignore')


def train_epoch(epoch, loader, iters, start_step=0, wandb=None):
    """
    跑完一个 epoch 的训练循环。

    参数：
        epoch      : 第几个 epoch（从 0 开始，仅用于日志/显示）
        loader     : DataLoader，每次 yield 一个 batch (input_ids, labels)，形状都是 [batch_size, max_seq_len]
        iters      : 本 epoch 的总 step 数（用于算进度、算学习率、判断「是否最后一步」）
        start_step : 断点续训时，从第 start_step 之后开始（enumerate(start=start_step+1)）
        wandb      : 实验跟踪对象（这里是 swanlab 的别名），None 表示不记录

    每个 step 内部做的事：
        1. 数据搬到 GPU
        2. 按「全局 step」动态计算余弦退火学习率
        3. 混合精度前向，取 loss（含 MoE 辅助损失），除以 accumulation_steps 后反向
        4. 每 accumulation_steps 步：梯度裁剪 -> 更新参数 -> 清空梯度
        5. 定期打印日志（loss / lr / 预计剩余时间）
        6. 定期保存模型（纯权重 + 续训状态两份）
    """
    start_time = time.time()          # 本 epoch 起点，用于估算剩余时间
    last_step = start_step            # 记录「最后实际走过的 step」，用于末尾判断是否有残余梯度
    # enumerate(loader, start=start_step+1)：让 step 编号从断点接着数（正常从头训练时 start_step=0，即从 1 开始）
    for step, (input_ids, labels) in enumerate(loader, start=start_step + 1):
        # ---- 1. 把当前 batch 搬到训练设备（GPU）----
        input_ids = input_ids.to(args.device)   # [batch_size, max_seq_len]，如 [32, 340]
        labels = labels.to(args.device)         # 同上
        last_step = step

        # ---- 2. 计算并设置当前 step 的学习率（余弦退火）----
        # get_lr(当前全局step, 总step数, 初始lr) = lr*(0.1 + 0.45*(1 + cos(pi * step/total)))
        # 例（lr=5e-4，total=2000）：
        #   step=0    -> cos(0)=1   -> 0.1+0.45*2=1.0 -> lr=5e-4（满学习率，无 warmup）
        #   step=1000 -> cos(π/2)=0 -> 0.1+0.45*1=0.55 -> lr=2.75e-4
        #   step=2000 -> cos(π)=-1 -> 0.1+0.45*0=0.1 -> lr=5e-5（下限 0.1×，不衰减到 0）
        # 用「全局 step = epoch*iters + step」定位进度，保证跨 epoch 连续衰减。
        lr = get_lr(epoch * iters + step, args.epochs * iters, args.learning_rate)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        # ---- 3. 混合精度前向 + 梯度累积反向 ----
        # autocast_ctx：CUDA 上是 torch.cuda.amp.autocast(dtype=bf16/fp16)，前向在低精度跑（省显存、提速）；
        #               CPU 上则是 nullcontext()（什么都不做）。
        with autocast_ctx:
            res = model(input_ids, labels=labels)
            # res.loss    : 语言建模交叉熵损失（CE），即「预测下一个 token」的平均损失。
            #               注意错位（x[:-1] 预测 y[1:]）已在 model.forward 内部完成，数据侧无需手动错位。
            # res.aux_loss: MoE 路由器的负载均衡辅助损失；非 MoE 时为标量 0，MoE 时为各层负载均衡损失之和。
            loss = res.loss + res.aux_loss
            # 关键：loss 先除以 accumulation_steps 再反向。
            #   这样累积 N 步后的梯度，等价于「一个 N 倍大 batch」的梯度，
            #   数学上保证梯度累积不改变等效学习率（否则等于学习率放大 N 倍）。
            loss = loss / args.accumulation_steps

        # scaler.scale(loss)：混合精度下先把 loss 放大（防止 fp16 梯度下溢到 0）再反向；
        # bf16 下 GradScaler 被禁用（enabled=False），scale 因子恒为 1，等价普通反向。
        scaler.scale(loss).backward()

        # ---- 4. 每 accumulation_steps 步才真正更新一次参数 ----
        # 例（accumulation_steps=8, batch_size=32）：
        #   step 1~7 : 各做一次前向+反向，梯度在 .grad 里「累加」，不更新
        #   step 8   : 前向+反向后 .grad 已累积 8 个微 batch 的梯度 -> 裁剪 -> 更新 -> 清空
        #   等效 batch_size = 32 * 8 = 256（显存放不下大 batch 时的常用技巧）
        if step % args.accumulation_steps == 0:
            scaler.unscale_(optimizer)                                  # 1) 把 scale 放大的梯度还原成真实尺度
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)  # 2) 梯度裁剪，防止梯度爆炸（范数限制在 grad_clip=1.0）
            scaler.step(optimizer)                                      # 3) 真正执行 AdamW 更新权重
            scaler.update()                                             # 4) 动态调整 GradScaler 的 scale 因子
            optimizer.zero_grad(set_to_none=True)                       # 5) 清空累积梯度，开始下一轮

        # ---- 5. 日志打印（每 log_interval 步，或最后一个 step）----
        if step % args.log_interval == 0 or step == iters:
            spend_time = time.time() - start_time
            current_loss = loss.item() * args.accumulation_steps       # 前面除过 accumulation_steps，这里乘回真实 loss
            current_aux_loss = res.aux_loss.item() if res.aux_loss is not None else 0.0
            current_logits_loss = current_loss - current_aux_loss      # CE 主体损失（扣除 MoE 辅助损失）
            current_lr = optimizer.param_groups[-1]['lr']
            # 预计剩余时间（分钟）= 已花时间 / 已走 step 数 * 剩余 step 数，再 //60 转分钟
            eta_min = spend_time / max(step - start_step, 1) * (iters - step) // 60
            Logger(f'Epoch:[{epoch + 1}/{args.epochs}]({step}/{iters}), loss: {current_loss:.4f}, logits_loss: {current_logits_loss:.4f}, aux_loss: {current_aux_loss:.4f}, lr: {current_lr:.8f}, epoch_time: {eta_min:.1f}min')
            if wandb: wandb.log({"loss": current_loss, "logits_loss": current_logits_loss, "aux_loss": current_aux_loss, "learning_rate": current_lr, "epoch_time": eta_min})

        # ---- 6. 定期保存（每 save_interval 步，或最后一个 step），且只在主进程执行 ----
        if (step % args.save_interval == 0 or step == iters) and is_main_process():
            model.eval()                                                       # 临时切 eval（关 dropout 等）
            moe_suffix = '_moe' if lm_config.use_moe else ''
            ckp = f'{args.save_dir}/{args.save_weight}_{lm_config.hidden_size}{moe_suffix}.pth'
            # 剥掉 DDP / torch.compile 的外壳，拿到真实模型
            raw_model = model.module if isinstance(model, DistributedDataParallel) else model
            raw_model = getattr(raw_model, '_orig_mod', raw_model)
            state_dict = raw_model.state_dict()
            # 保存第 1 份：纯模型权重（half() 转 fp16 省空间，cpu() 释放显存），供后续加载/推理用
            torch.save({k: v.half().cpu() for k, v in state_dict.items()}, ckp)
            # 保存第 2 份：完整训练状态（模型+优化器+scaler+epoch+step+wandb_id），供断点续训用
            lm_checkpoint(lm_config, weight=args.save_weight, model=model, optimizer=optimizer, scaler=scaler, epoch=epoch, step=step, wandb=wandb, save_dir='../checkpoints')
            model.train()                                                      # 切回训练模式
            del state_dict

        # 显式释放本 step 的中间变量，避免累积占用显存
        del input_ids, labels, res, loss

    # ---- 7. epoch 结束时处理「不整除」的残余梯度 ----
    # 场景：一个 epoch 的 step 数（如 1000）不是 accumulation_steps（8）的整数倍时，
    #       循环结束时还残留不足 8 步的梯度没更新；这里把剩下的梯度也更新掉，不浪费。
    if last_step > start_step and last_step % args.accumulation_steps != 0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)


if __name__ == "__main__":
    # ================================================================
    # 0. 解析命令行参数（所有可调项都集中在这里）
    # ================================================================
    parser = argparse.ArgumentParser(description="MiniMind Pretraining")
    parser.add_argument("--save_dir", type=str, default="../out", help="模型保存目录")
    parser.add_argument('--save_weight', default='pretrain', type=str, help="保存权重的前缀名")
    parser.add_argument("--epochs", type=int, default=2, help="训练轮数")
    parser.add_argument("--batch_size", type=int, default=32, help="batch size")
    parser.add_argument("--learning_rate", type=float, default=5e-4, help="初始学习率")
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu", help="训练设备")
    parser.add_argument("--dtype", type=str, default="bfloat16", help="混合精度类型")
    parser.add_argument("--num_workers", type=int, default=8, help="数据加载线程数")
    parser.add_argument("--accumulation_steps", type=int, default=8, help="梯度累积步数")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="梯度裁剪阈值")
    parser.add_argument("--log_interval", type=int, default=100, help="日志打印间隔")
    parser.add_argument("--save_interval", type=int, default=1000, help="模型保存间隔")
    parser.add_argument('--hidden_size', default=768, type=int, help="隐藏层维度")
    parser.add_argument('--num_hidden_layers', default=8, type=int, help="隐藏层数量")
    parser.add_argument('--max_seq_len', default=340, type=int, help="训练的最大截断长度（中文1token≈1.5~1.7字符）")
    parser.add_argument('--use_moe', default=0, type=int, choices=[0, 1], help="是否使用MoE架构（0=否，1=是）")
    parser.add_argument("--data_path", type=str, default="../dataset/pretrain_t2t_mini.jsonl", help="预训练数据路径")
    parser.add_argument('--from_weight', default='none', type=str, help="基于哪个权重训练，为none则从头开始")
    parser.add_argument('--from_resume', default=0, type=int, choices=[0, 1], help="是否自动检测&续训（0=否，1=是）")
    parser.add_argument("--use_wandb", action="store_true", help="是否使用wandb")
    parser.add_argument("--wandb_project", type=str, default="MiniMind-Pretrain", help="wandb项目名")
    parser.add_argument("--use_compile", default=0, type=int, choices=[0, 1], help="是否使用torch.compile加速（0=否，1=是）")
    args = parser.parse_args()

    # ========== 1. 初始化环境和随机种子 ==========
    # init_distributed_mode：环境变量 RANK==-1（未用 torchrun 启动）时返回 0 走单卡；
    #                       否则用 nccl 后端初始化进程组，并返回 local_rank。
    local_rank = init_distributed_mode()
    if dist.is_initialized(): args.device = f"cuda:{local_rank}"   # DDP 下每张卡绑定自己的 local_rank 卡
    # 每个 rank 用不同种子（42, 43, 44...），配合后面的 randperm 让各卡采到不同数据，避免多卡学同一批样本
    setup_seed(42 + (dist.get_rank() if dist.is_initialized() else 0))

    # ========== 2. 配置目录、模型参数、检查ckp ==========
    os.makedirs(args.save_dir, exist_ok=True)
    lm_config = MiniMindConfig(hidden_size=args.hidden_size, num_hidden_layers=args.num_hidden_layers, use_moe=bool(args.use_moe))
    # from_resume==1 时去 ../checkpoints/pretrain_768_resume.pth 找续训文件；找不到返回 None。
    # from_resume==0 时直接 None，不检测。
    ckp_data = lm_checkpoint(lm_config, weight=args.save_weight, save_dir='../checkpoints') if args.from_resume==1 else None

    # ========== 3. 设置混合精度 ==========
    device_type = "cuda" if "cuda" in args.device else "cpu"
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
    # CUDA 上启用 autocast（前向低精度）；CPU 上 nullcontext 占位，不做混合精度
    autocast_ctx = nullcontext() if device_type == "cpu" else torch.cuda.amp.autocast(dtype=dtype)

    # ========== 4. 配wandb ==========
    # 注意：这里 `import swanlab as wandb`，实际用的是国产实验跟踪工具 SwanLab，别名当 wandb 用
    wandb = None
    if args.use_wandb and is_main_process():
        import swanlab as wandb
        wandb_id = ckp_data.get('wandb_id') if ckp_data else None   # 有续训则拿上次的 run id
        resume = 'must' if wandb_id else None                       # 有 id 就强制续接上次的 run
        wandb_run_name = f"MiniMind-Pretrain-Epoch-{args.epochs}-BatchSize-{args.batch_size}-LearningRate-{args.learning_rate}"
        wandb.init(project=args.wandb_project, name=wandb_run_name, id=wandb_id, resume=resume)

    # ========== 5. 定义模型、数据、优化器 ==========
    # init_model：加载 tokenizer + 建模型；from_weight!='none' 时从 ../out/{from_weight}_768.pth 加载预训练权重，
    #             默认 'none' 即随机初始化、从头预训练。
    model, tokenizer = init_model(lm_config, args.from_weight, device=args.device)
    # PretrainDataset：读 jsonl 纯文本，tokenize 后返回 (input_ids, labels)，详见 dataset/lm_dataset.py
    train_ds = PretrainDataset(args.data_path, tokenizer, max_length=args.max_seq_len)
    # DistributedSampler：多卡时让每张卡拿到「互不重叠」的数据切片；单卡为 None
    train_sampler = DistributedSampler(train_ds) if dist.is_initialized() else None
    # GradScaler 只在 fp16 时启用（需要 scale 防下溢）；bf16 无需 scale，enabled=False 时 scale 因子恒为 1
    scaler = torch.cuda.amp.GradScaler(enabled=(args.dtype == 'float16'))
    # AdamW 优化器，初始 lr=5e-4（之后每步被 get_lr 动态覆盖）
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)

    # ========== 6. 从ckp恢复状态 ==========
    # 恢复模型权重、优化器动量状态、scaler 状态，以及上次训练到哪一步（start_epoch/start_step）
    start_epoch, start_step = 0, 0
    if ckp_data:
        model.load_state_dict(ckp_data['model'])
        optimizer.load_state_dict(ckp_data['optimizer'])
        scaler.load_state_dict(ckp_data['scaler'])
        start_epoch = ckp_data['epoch']
        start_step = ckp_data.get('step', 0)

    # ========== 7. 编译和分布式包装 ==========
    if args.use_compile == 1:
        model = torch.compile(model)   # 图优化加速（首次编译慢，之后快）
        Logger('torch.compile enabled')
    if dist.is_initialized():
        # DDP 数据并行：每卡一份模型副本，反向后自动 all-reduce 梯度求平均
        model = DistributedDataParallel(model, device_ids=[local_rank])

    # ========== 8. 开始训练 ==========
    for epoch in range(start_epoch, args.epochs):
        # 多卡：set_epoch 让 DistributedSampler 每轮用不同种子打乱顺序（避免每轮数据顺序一样）
        train_sampler and train_sampler.set_epoch(epoch)
        # 单卡：手动生成「打乱后的样本索引」列表，代替 sampler（每轮用不同种子 42+epoch）
        setup_seed(42 + epoch); indices = torch.randperm(len(train_ds)).tolist()
        # 断点续训：只在「断点所在的那个 epoch」跳过前 start_step 个 step，之后的 epoch 从 0 开始
        skip = start_step if (epoch == start_epoch and start_step > 0) else 0
        # SkipBatchSampler：把索引按 batch_size 打包，并支持「跳过前 skip 个 batch」（续训用）。
        #   例：100 个样本、batch_size=32 -> batch0=[32个], batch1=[32个], batch2=[32个], batch3=[最后4个]，共 4 个 batch。
        batch_sampler = SkipBatchSampler(train_sampler or indices, args.batch_size, skip)
        loader = DataLoader(train_ds, batch_sampler=batch_sampler, num_workers=args.num_workers, pin_memory=True)
        if skip > 0:
            Logger(f'Epoch [{epoch + 1}/{args.epochs}]: 跳过前{start_step}个step，从step {start_step + 1}开始')
            # iters 传 len(loader)+skip：补回被跳过的 batch 数，
            # 这样 train_epoch 里 enumerate(start=start_step+1) 的 step 编号、
            # 以及 get_lr 用的 epoch*iters+step 全局进度、日志 (step/iters) 比例都保持正确连续。
            train_epoch(epoch, loader, len(loader) + skip, start_step, wandb)
        else:
            train_epoch(epoch, loader, len(loader), 0, wandb)

    # ========== 9. 清理分布进程 ==========
    if dist.is_initialized():
        dist.barrier()                # 所有卡同步，确保都训练完再退出
        dist.destroy_process_group()  # 销毁进程组，干净退出
