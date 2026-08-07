"""
对基础模型进行监督微调（Supervised Fine-Tuning, SFT），使其从"文本补全模型"
转变为"对话助手模型"。

运行方式：
单 GPU:
python -m scripts.chat_sft

8 GPU 分布式训练:
torchrun --standalone --nproc_per_node=8 -m scripts.chat_sft -- --device-batch-size=16

整体流程：
1. 解析命令行参数
2. 初始化计算环境（设备、DDP、wandb）
3. 从 base 预训练检查点加载模型和超参（支持继承或覆盖）
4. 编译模型（torch.compile）
5. 初始化优化器（可选：从预训练检查点热启动动量缓冲区）
6. 构建 SFT 数据混合（SmolTalk + MMLU + GSM8K + 拼写 + 身份对话）
7. 进入训练循环（BOS 对齐 + BestFit 打包 + 评估 + 保存）
8. 清理资源

与 base_train.py 的关键区别：
- 加载预训练权重和超参（继承机制）
- 数据加载器使用 BOS 对齐 + BestFit 打包（而非连续序列）
- 训练步数由数据集大小驱动（而非固定步数）
- 学习率调度器基于进度比例（而非绝对步数）
- 评估使用 ChatCORE（对话/推理能力）而非 CORE 指标
"""

import gc
import argparse
import os
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"  # 启用 PyTorch 内存分配器的可扩展段，减少显存碎片
import time
import wandb
import torch
from nanochat.common import compute_init, compute_cleanup, print0, DummyWandb, get_base_dir, autodetect_device_type, get_peak_flops, COMPUTE_DTYPE, COMPUTE_DTYPE_REASON, is_ddp_initialized
from nanochat.tokenizer import get_token_bytes
from nanochat.checkpoint_manager import save_checkpoint, load_model, load_optimizer_state
from nanochat.loss_eval import evaluate_bpb
import torch.distributed as dist
from nanochat.flash_attention import HAS_FA3
from nanochat.engine import Engine
from scripts.chat_eval import run_chat_eval

from tasks.common import TaskMixture
from tasks.gsm8k import GSM8K
from tasks.mmlu import MMLU
from tasks.smoltalk import SmolTalk
from tasks.customjson import CustomJSON
from tasks.spellingbee import SimpleSpelling, SpellingBee

# =============================================================================
# 第一部分：命令行参数解析
# 大部分训练超参默认值为 None，表示从预训练检查点继承
# =============================================================================

parser = argparse.ArgumentParser(description="Supervised fine-tuning (SFT) the model")

# --- 日志 ---
parser.add_argument("--run", type=str, default="dummy", help="wandb 运行名称（'dummy' = 禁用 wandb 日志）")

# --- 运行时环境 ---
parser.add_argument("--device-type", type=str, default="", help="计算设备类型：cuda|cpu|mps（空字符串 = 自动检测）")

# --- 模型加载 ---
parser.add_argument("--model-tag", type=str, default=None, help="要加载的预训练模型标签")
parser.add_argument("--model-step", type=int, default=None, help="要加载的预训练模型步数（None = 最终检查点）")
parser.add_argument("--load-optimizer", type=int, default=1,
                    help="是否从预训练检查点热启动优化器状态（0=否, 1=是）")

# --- 训练步数 ---
parser.add_argument("--num-iterations", type=int, default=-1,
                    help="优化步数（-1 = 完整遍历训练集一个 epoch 后停止）")

# --- 批次大小（默认从预训练检查点继承）---
parser.add_argument("--max-seq-len", type=int, default=None,
                    help="最大上下文长度（默认：从预训练检查点继承）")
parser.add_argument("--device-batch-size", type=int, default=None,
                    help="每张卡的单步批次大小（默认：从预训练检查点继承）")
parser.add_argument("--total-batch-size", type=int, default=None,
                    help="全局总批次大小（以 token 计，默认：从预训练检查点继承）")

# --- 优化器参数（默认从预训练检查点继承）---
parser.add_argument("--embedding-lr", type=float, default=None,
                    help="嵌入层学习率（默认：从预训练检查点继承）")
parser.add_argument("--unembedding-lr", type=float, default=None,
                    help="解嵌层学习率（默认：从预训练检查点继承）")
parser.add_argument("--matrix-lr", type=float, default=None,
                    help="矩阵参数学习率（默认：从预训练检查点继承）")
parser.add_argument("--init-lr-frac", type=float, default=0.8,
                    help="初始学习率占基准学习率的比例（通常小于 1.0，避免 SFT 初期震荡）")
parser.add_argument("--warmup-ratio", type=float, default=0.0,
                    help="学习率预热占总进度的比例")
parser.add_argument("--warmdown-ratio", type=float, default=0.5,
                    help="学习率衰减占总进度的比例")
parser.add_argument("--final-lr-frac", type=float, default=0.0,
                    help="最终学习率相对于初始学习率的比例（0 = 衰减到零）")

# --- 评估配置 ---
parser.add_argument("--eval-every", type=int, default=200,
                    help="每隔 N 步评估验证集 BPB（-1 = 禁用）")
parser.add_argument("--eval-tokens", type=int, default=40 * 524288,
                    help="验证集评估时使用的 token 总数")
parser.add_argument("--chatcore-every", type=int, default=200,
                    help="每隔 N 步计算 ChatCORE 指标（-1 = 禁用）")
parser.add_argument("--chatcore-max-cat", type=int, default=-1,
                    help="分类任务（ARC/MMLU 等）每个任务最多评估的问题数（-1 = 无限制）")
parser.add_argument("--chatcore-max-sample", type=int, default=24,
                    help="生成式任务（GSM8K/HumanEval/SpellingBee）每个任务最多评估的问题数")

# --- 数据混合比例 ---
# 控制不同数据集的重复轮数（epoch），确保各能力均衡发展
parser.add_argument("--mmlu-epochs", type=int, default=3,
                    help="MMLU 在训练混合中的 epoch 数（训练多选题解答能力）")
parser.add_argument("--gsm8k-epochs", type=int, default=4,
                    help="GSM8K 在训练混合中的 epoch 数（训练数学推理和工具使用）")

args = parser.parse_args()
user_config = vars(args).copy()  # 保存用户原始配置供日志

# =============================================================================
# 第二部分：计算环境初始化
# =============================================================================

# 初始化设备（CUDA/CPU/MPS）和 DDP 分布式通信
device_type = autodetect_device_type() if args.device_type == "" else args.device_type
ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init(device_type)
master_process = ddp_rank == 0  # 主进程负责日志和检查点
print0(f"COMPUTE_DTYPE: {COMPUTE_DTYPE} ({COMPUTE_DTYPE_REASON})")

# 根据设备类型选择同步和内存统计函数
synchronize = torch.cuda.synchronize if device_type == "cuda" else lambda: None
get_max_memory = torch.cuda.max_memory_allocated if device_type == "cuda" else lambda: 0

if device_type == "cuda":
    gpu_device_name = torch.cuda.get_device_name(0)
    gpu_peak_flops = get_peak_flops(gpu_device_name)  # GPU 理论峰值 BF16 FLOPS
    print0(f"GPU: {gpu_device_name} | Peak FLOPS (BF16): {gpu_peak_flops:.2e}")
else:
    gpu_peak_flops = float('inf')  # CPU/MPS 上 MFU 无意义

# --- wandb 日志初始化 ---
use_dummy_wandb = args.run == "dummy" or not master_process
wandb_run = DummyWandb() if use_dummy_wandb else wandb.init(
    project="nanochat-sft", name=args.run, config=user_config
)

# --- Flash Attention 状态检查 ---
if not HAS_FA3:
    print0("WARNING: Flash Attention 3 not available, using PyTorch SDPA fallback. Training will be less efficient.")

# =============================================================================
# 第三部分：加载预训练基础模型并继承训练超参
#
# 核心理念：SFT 应该从预训练成功的超参出发，只对需要调整的部分做覆盖。
# 所有默认值为 None 的参数会从预训练检查点中自动继承。
# =============================================================================

# 加载基础模型和分词器（phase="train" 表示用于训练）
model, tokenizer, meta = load_model(
    "base", device, phase="train", model_tag=args.model_tag, step=args.model_step
)

# --- 从预训练检查点继承训练超参 ---
# 继承优先级：CLI 显式指定 > 预训练检查点中记录的值 > 硬编码回退值
# 对每个参数：
#   - arg_val = None（用户未显式指定）→ 使用预训练检查点中的值
#   - arg_val 显式指定且与预训练值不同 → 使用用户指定值（覆盖）
#   - arg_val 显式指定且与预训练值相同 → 使用该值
pretrain_user_config = meta.get("user_config", {})
for name, fallback, source in [
    ("max_seq_len",       2048,  meta),               # 从检查点 meta 中取
    ("device_batch_size", 32,    meta),               # 从检查点 meta 中取
    ("total_batch_size",  524288, meta),              # 从检查点 meta 中取
    ("embedding_lr",      0.3,   pretrain_user_config),  # 从预训练 CLI 参数中取
    ("unembedding_lr",    0.004, pretrain_user_config),  # 从预训练 CLI 参数中取
    ("matrix_lr",         0.02,  pretrain_user_config),  # 从预训练 CLI 参数中取
]:
    arg_val = getattr(args, name)
    pretrain_val = source.get(name)
    if arg_val is None:
        # 用户未指定 → 从预训练检查点继承
        resolved = pretrain_val if pretrain_val is not None else fallback
        setattr(args, name, resolved)
        print0(f"Inherited {name}={resolved} from pretrained checkpoint")
    elif pretrain_val is not None and arg_val != pretrain_val:
        # 用户显式覆盖 → 提示
        print0(f"NOTE: --{name.replace('_', '-')}={arg_val} overrides pretrained value of {pretrain_val}")
    else:
        print0(f"Using {name}={arg_val}")

# =============================================================================
# 第四部分：编译模型并计算批次配置
# =============================================================================

# 保留原始模型引用（评估和保存检查点时使用）
orig_model = model
# 编译模型（dynamic=False：输入形状不变，编译器可做更激进优化）
model = torch.compile(model, dynamic=False)

depth = model.config.n_layer                           # 模型层数
num_flops_per_token = model.estimate_flops()           # 每个 token 的 FLOPs 估算
tokens_per_fwdbwd = args.device_batch_size * args.max_seq_len           # 单卡单次前向 token 数
world_tokens_per_fwdbwd = tokens_per_fwdbwd * ddp_world_size           # 所有卡单次前向 token 数

# 确保全局批次可以被单次前向整除（梯度累积需要整数步数）
assert args.total_batch_size % world_tokens_per_fwdbwd == 0, \
    f"total_batch_size ({args.total_batch_size}) 必须能被 world_tokens_per_fwdbwd ({world_tokens_per_fwdbwd}) 整除"
grad_accum_steps = args.total_batch_size // world_tokens_per_fwdbwd   # 梯度累积步数

print0(f"Tokens / micro-batch / rank: {args.device_batch_size} x {args.max_seq_len} = {tokens_per_fwdbwd:,}")
print0(f"Tokens / micro-batch: {world_tokens_per_fwdbwd:,}")
print0(f"Total batch size {args.total_batch_size:,} => gradient accumulation steps: {grad_accum_steps}")

# 获取每个 token 对应的 UTF-8 字节数（用于 BPB 计算）
token_bytes = get_token_bytes(device=device)

# =============================================================================
# 第五部分：初始化优化器（MuonAdamW）
#
# SFT 的关键设计决策：
# - 权重衰减设为 0.0：预训练阶段已将权重衰减余弦降至零，
#   SFT 从零继续（少量微调步数下，权重衰减会不合理地压小参数）
# - 可选支持从预训练优化器热启动（沿用动量缓冲区，加速收敛）
# =============================================================================

# 创建优化器：Muon 优化矩阵参数，AdamW 优化嵌入/解嵌层
optimizer = model.setup_optimizer(
    unembedding_lr=args.unembedding_lr,
    embedding_lr=args.embedding_lr,
    matrix_lr=args.matrix_lr,
    weight_decay=0.0  # SFT 阶段不需要权重衰减
)

# --- 可选：从预训练优化器热启动 ---
# 热启动的好处：保留预训练阶段积累的动量缓冲区（Muon 的 Newton-Schulz 迭代状态等），
# 使 SFT 初期的优化更平滑。但需要特别注意以下问题：
#
# 问题：load_state_dict 会覆盖整个 param_group 的所有字段（包括 LR、betas 等）。
# 预训练结束时学习率已衰减到接近零，如果直接加载，SFT 的学习率会被覆盖成零。
#
# 解决：在 load 之前保存当前 param_group 的学习率，load 之后恢复。
base_dir = get_base_dir()
if args.load_optimizer:
    optimizer_data = load_optimizer_state(
        "base", device, rank=ddp_rank, model_tag=args.model_tag, step=args.model_step
    )
    if optimizer_data is not None:
        # 保存 SFT 设定的学习率（即将被覆盖）
        base_lrs = [group["lr"] for group in optimizer.param_groups]
        # 加载预训练优化器状态（含动量缓冲区，但会覆盖 LR 等元数据）
        optimizer.load_state_dict(optimizer_data)
        del optimizer_data  # 立即释放原始数据
        # 恢复 SFT 的学习率
        for group, base_lr in zip(optimizer.param_groups, base_lrs):
            group["lr"] = base_lr
        print0("Loaded optimizer state from pretrained checkpoint (momentum buffers only, LRs reset)")
    else:
        print0("WARNING: optimizer checkpoint not found, starting with fresh optimizer (slightly worse)")

# =============================================================================
# 第六部分：混合精度梯度缩放器
# =============================================================================

scaler = torch.amp.GradScaler() if COMPUTE_DTYPE == torch.float16 else None
if scaler is not None:
    print0("GradScaler enabled for fp16 training")

# --- 设置初始学习率为基准学习率的比例 ---
# init_lr_frac < 1.0 意味着初始学习率比基准低（如 0.8 = 从 80% 开始衰减）
# 这比从零预热更简单，在 SFT 的小步数场景下效果类似
for group in optimizer.param_groups:
    group["lr"] = group["lr"] * args.init_lr_frac
    group["initial_lr"] = group["lr"]  # 记录调度器的"基准"学习率

# =============================================================================
# 第七部分：SFT 数据混合与数据加载器
#
# 数据混合策略（6 个数据源）：
# 1. SmolTalk（46 万行）：通用对话数据，SFT 的主力数据源
# 2. CustomJSON × 2（每份 1000 行）：合成身份对话（"你是谁？"等）
# 3. MMLU × mmlu_epochs（每 epoch 10 万行）：多选答题 → 训练选择题解答能力
# 4. GSM8K × gsm8k_epochs（每 epoch 8000 行）：数学题 → 训练推理和工具使用
# 5. SimpleSpelling（20 万行）：简单拼写（如 "spell the word apple"）
# 6. SpellingBee（8 万行）：字母计数（如 "how many 'r' in strawberry?"）
#
# 数据加载器的核心设计：BOS 对齐 + BestFit 打包
# - 每一行以 BOS（对话开始）开头
# - 使用 BestFit 算法将多个对话拼接填满一行（不裁剪！）
# - 当没有对话能填入剩余空间时，用 BOS token 填充
# - 填充位置在 loss 计算时被 mask 掉（targets = -1）
# =============================================================================

identity_conversations_filepath = os.path.join(base_dir, "identity_conversations.jsonl")
train_tasks = [
    SmolTalk(split="train"),                                    # 46 万行通用对话
    CustomJSON(filepath=identity_conversations_filepath),       # 1000 行身份对话（第 1 epoch）
    CustomJSON(filepath=identity_conversations_filepath),       # 1000 行身份对话（第 2 epoch）
    *[MMLU(subset="all", split="auxiliary_train") for _ in range(args.mmlu_epochs)],    # 每 epoch 10 万行
    *[GSM8K(subset="main", split="train") for _ in range(args.gsm8k_epochs)],          # 每 epoch 8000 行
    SimpleSpelling(size=200000, split="train"),                 # 20 万行简单拼写
    SpellingBee(size=80000, split="train"),                     # 8 万行拼写游戏
]
train_dataset = TaskMixture(train_tasks)  # TaskMixture 将所有任务合并为一个统一数据集
print0(f"Training mixture: {len(train_dataset):,} rows (MMLU x{args.mmlu_epochs}, GSM8K x{args.gsm8k_epochs})")

# 验证集：按训练混合的比例粗略匹配
val_dataset = TaskMixture([
    SmolTalk(split="test"),                                     # 2.4 万行
    MMLU(subset="all", split="test", stop=5200),                # 只用 5200 行（匹配训练比例）
    GSM8K(subset="main", split="test", stop=420),               # 只用 420 行（匹配训练比例）
])  # 总计约 2.96 万行

# --- BOS 对齐 + BestFit 打包的数据加载器 ---
# SFT 的数据加载器与预训练有几个关键区别：
# 1. 预训练用的是"连续序列"（跨文档的数据流）
# 2. SFT 用的是"对话片段"（每个 BOS 开始一个新的对话）
# 3. BestFit 算法在打包时尽量不裁剪对话（保证数据完整性）
#
# 全局变量（训练中动态更新）：
last_step = False          # 是否到达训练结束（由数据消耗或步数限制触发）
approx_progress = 0.0      # 当前训练进度（0→1）
current_epoch = 1          # 当前数据 epoch

def sft_data_generator_bos_bestfit(split, buffer_size=100):
    """
    BOS 对齐的 SFT 数据加载器，使用 BestFit-Pad 打包策略。

    设计理念：
    - 每行的第一个 token 是 BOS（对话开始标志）
    - 多个对话被拼接到同一行中（用 BestFit 算法最大化空间利用率）
    - 当没有对话能填入剩余空间时，用 BOS token 填充（不裁剪对话！）
    - 填充位置的目标值设为 -1（CrossEntropy 的 ignore_index，不参与 loss 计算）
    - 如果某条对话的长度超过了整行容量，直接丢弃（无法在不裁剪的情况下打包）

    参数：
        split: "train" 或 "val"
        buffer_size: 预取缓冲区大小（管理对话队列）

    产出：
        inputs:  Tensor (device_batch_size, max_seq_len)，自回归输入
        targets: Tensor (device_batch_size, max_seq_len)，自回归目标（填充位置 = -1）
    """
    global last_step, approx_progress, current_epoch

    assert split in {"train", "val"}, "split must be 'train' or 'val'"
    dataset = train_dataset if split == "train" else val_dataset
    dataset_size = len(dataset)
    assert dataset_size > 0

    row_capacity = args.max_seq_len + 1  # +1 是因为需要为最后一个位置的 target 留空
    bos_token = tokenizer.get_bos_token_id()

    # 对话缓冲区：存储 (token_ids, loss_mask) 元组
    conv_buffer = []
    cursor = ddp_rank     # 每个 rank 处理不同的对话子集（分布式数据分片）
    consumed = ddp_rank    # 跟踪实际消费量（与 buf 中的游标分离，用于进度计算）
    epoch = 1
    it = 0                 # 迭代计数（用于 num_iterations 限制）

    def refill_buffer():
        """填充对话预取缓冲区（确保缓冲区中有足够的候选对话）"""
        nonlocal cursor, epoch
        while len(conv_buffer) < buffer_size:
            conversation = dataset[cursor]
            # 渲染对话：返回 (token_ids, loss_mask)
            # loss_mask: 1=assistant 回复（参与 loss），0=用户提示/BOS/特殊 token/工具输出
            ids, mask = tokenizer.render_conversation(conversation)
            # 丢弃无法放入单行的过长对话（否则会导致批次中没有有效训练目标）
            if len(ids) <= row_capacity:
                conv_buffer.append((ids, mask))
            cursor += ddp_world_size
            if cursor >= dataset_size:
                # 回绕到数据集开头（新的 epoch）
                cursor = cursor % dataset_size
                epoch += 1

    while True:
        rows = []           # 存放完整的行（device_batch_size 行）
        mask_rows = []      # 对应的 loss mask
        row_lengths = []    # 每行的实际内容长度（不含填充）

        # 为每个 batch 位置构建一行
        for _ in range(args.device_batch_size):
            row = []         # 当前行的 token 列表
            mask_row = []    # 对应的 mask 列表
            padded = False
            content_len = 0

            while len(row) < row_capacity:
                # 确保缓冲区有足够的候选对话
                while len(conv_buffer) < buffer_size:
                    refill_buffer()

                remaining = row_capacity - len(row)

                # --- BestFit 算法：寻找能完全填入的最大对话 ---
                # 遍历所有候选对话，找长度 ≤ remaining 且最大的那个
                best_idx = -1
                best_len = 0
                for i, (conv, _) in enumerate(conv_buffer):
                    conv_len = len(conv)
                    if conv_len <= remaining and conv_len > best_len:
                        best_idx = i
                        best_len = conv_len

                if best_idx >= 0:
                    # 找到可以完整填入的对话 → 使用整条对话
                    conv, conv_mask = conv_buffer.pop(best_idx)
                    row.extend(conv)
                    mask_row.extend(conv_mask)
                    consumed += ddp_world_size  # 跟踪消费进度
                else:
                    # 没有对话能填入剩余空间 → 用 BOS token 填充（不裁剪对话！）
                    content_len = len(row)
                    row.extend([bos_token] * remaining)
                    mask_row.extend([0] * remaining)      # mask=0 表示不参与 loss
                    padded = True
                    break  # 行已填满

            # 记录每行的实际内容长度
            if padded:
                row_lengths.append(content_len)
            else:
                row_lengths.append(row_capacity)
            rows.append(row[:row_capacity])
            mask_rows.append(mask_row[:row_capacity])

        # --- 停止条件检查 ---
        it += 1
        if 0 < args.num_iterations <= it and split == "train":
            last_step = True

        # --- 更新训练进度（基于消费量，而非游标位置，考虑缓冲的影响）---
        if split == "train":
            current_epoch = epoch
            if args.num_iterations > 0:
                approx_progress = it / args.num_iterations           # 基于步数
            else:
                approx_progress = consumed / dataset_size            # 基于消费的数据量
            # 当已消费完整个数据集时触发停止
            # 注意：是 consumed >= dataset_size（游标走过一圈以上），而非 cursor 循环一次
            if consumed >= dataset_size:
                last_step = True

        # --- 构建 PyTorch 张量 ---
        use_cuda = device_type == "cuda"
        # 先创建 CPU 张量（pin_memory 加速 CPU→GPU 传输）
        batch_tensor = torch.tensor(rows, dtype=torch.long, pin_memory=use_cuda)
        # inputs: 去掉最后一列（位置 t 的输入是 tok[t]）
        inputs = batch_tensor[:, :-1].to(device=device, dtype=torch.int32, non_blocking=use_cuda).contiguous()
        # targets: 去掉第一列（位置 t 的预测目标是 tok[t+1]）
        targets = batch_tensor[:, 1:].to(device=device, dtype=torch.int64, non_blocking=use_cuda).contiguous()

        # --- 应用 loss mask ---
        # mask 的含义（来自 render_conversation）：
        #   mask=1: assistant 的回复 → 参与 loss 计算
        #   mask=0: 用户提示词、BOS token、特殊 token、工具调用输出 → 不参与 loss
        mask_tensor = torch.tensor(mask_rows, dtype=torch.int8)
        mask_targets = mask_tensor[:, 1:].to(device=device)  # 与 targets 对齐（偏移一位）
        targets[mask_targets == 0] = -1                     # -1 是 CrossEntropy 的 ignore_index

        # --- 屏蔽填充位置 ---
        # 对于每行，content_len 之后的位置都是填充（应被忽略）
        for i, content_len in enumerate(row_lengths):
            if content_len < row_capacity:
                targets[i, content_len-1:] = -1             # 填充位置的 target 设为 ignore_index

        yield inputs, targets

# 创建训练和验证数据加载器
train_loader = sft_data_generator_bos_bestfit("train")
build_val_loader = lambda: sft_data_generator_bos_bestfit("val")  # 每次评估时重新创建（从头遍历验证集）

# =============================================================================
# 第八部分：学习率和动量调度器
#
# 与 base_train.py 的关键区别：
# - 使用 progress（0→1 的比例）而非绝对步数来驱动调度器
# - 原因：SFT 的步数可能不预先确定（由数据量驱动），用比例更灵活
# =============================================================================

def get_lr_multiplier(progress):
    """
    基于训练进度的学习率乘子。

    三个阶段（与 base_train 相同的形状，但用 progress 而非 step）：
    1. 线性预热：progress < warmup_ratio，从 0 线性增加到 1.0
    2. 恒定：progress 在 [warmup_ratio, 1-warmdown_ratio]，保持 1.0
    3. 线性衰减：progress > 1-warmdown_ratio，从 1.0 线性降到 final_lr_frac

    +1e-8 防止 progress=0 时除零（在某些边界情况下）
    """
    if progress < args.warmup_ratio:
        # 阶段 1：线性预热
        return (progress + 1e-8) / args.warmup_ratio
    elif progress <= 1.0 - args.warmdown_ratio:
        # 阶段 2：恒定
        return 1.0
    else:
        # 阶段 3：线性衰减
        decay = (progress - (1.0 - args.warmdown_ratio)) / args.warmdown_ratio
        return (1 - decay) * 1.0 + decay * args.final_lr_frac


def get_muon_momentum(it):
    """
    基于步数的 Muon 动量调度器。

    SFT 的动量范围比预训练小一些（0.85→0.95 vs 0.85→0.97），
    因为微调阶段不需要那么强的动量来稳定优化。
    """
    frac = min(it / 300, 1)                                # 前 300 步完成预热
    momentum = (1 - frac) * 0.85 + frac * 0.95             # 从 0.85 线性增加到 0.95
    return momentum

# =============================================================================
# 第九部分：训练循环
#
# 流程：验证评估 → ChatCORE 评估 → 保存检查点 → 梯度累积 → 优化器更新 → 日志
# =============================================================================

x, y = next(train_loader)  # 预取第一批数据（流水线：在 GPU 计算时并行加载下一批）

# --- 初始化循环状态变量 ---
min_val_bpb = float("inf")           # 最佳验证 BPB
smooth_train_loss = 0                # 训练 loss 的指数移动平均
ema_beta = 0.9                       # EMA 衰减因子
total_training_time = 0              # 累积训练时间（排除前 10 步）
step = 0

while True:
    flops_so_far = num_flops_per_token * args.total_batch_size * step

    # --- DDP 同步 last_step ---
    # 关键：在分布式环境下，各 rank 的 last_step 可能不同（消费速度不完全一致）
    # 对所有 rank 的 last_step 做 all_reduce(MAX)，确保要么所有人都停，要么都不停
    # 如果不做这个同步，某些 rank 提前退出会导致通信死锁（NCCL 超时）
    if ddp:
        last_step_tensor = torch.tensor(last_step, dtype=torch.int32, device=device)
        dist.all_reduce(last_step_tensor, op=dist.ReduceOp.MAX)  # 任一 rank 触发 last_step → 全体停止
        last_step = bool(last_step_tensor.item())

    # ============ 评估块 1：验证集 BPB ============
    # 与预训练相同的评估方式，确保可比性
    if last_step or (args.eval_every > 0 and step % args.eval_every == 0):
        model.eval()
        val_loader = build_val_loader()
        eval_steps = args.eval_tokens // (args.device_batch_size * args.max_seq_len * ddp_world_size)
        val_bpb = evaluate_bpb(model, val_loader, eval_steps, token_bytes)
        print0(f"Step {step:05d} | Validation bpb: {val_bpb:.4f}")
        if val_bpb < min_val_bpb:
            min_val_bpb = val_bpb
        wandb_run.log({
            "step": step,
            "total_training_flops": flops_so_far,
            "total_training_time": total_training_time,
            "val/bpb": val_bpb,
        })
        model.train()

    # ============ 评估块 2：ChatCORE 指标 ============
    # ChatCORE 是专门衡量"对话助手能力"的评估指标，涵盖 6 个任务：
    # - 分类任务（每选项概率比较）：ARC-Easy, ARC-Challenge, MMLU
    # - 生成式任务（解码文本后判断正确性）：GSM8K, HumanEval, SpellingBee
    #
    # 指标计算方式：centered accuracy
    #   centered_acc = (acc - random_baseline) / (1.0 - random_baseline)
    #   范围：0 = 随机猜测水平，1.0 = 完美
    #   例如：MMLU 有 4 个选项（random=0.25），模型得 70%
    #     centered = (0.70 - 0.25) / (1.0 - 0.25) = 0.45 / 0.75 = 0.60
    chatcore_results = {}
    if args.chatcore_every > 0 and (last_step or (step > 0 and step % args.chatcore_every == 0)):
        model.eval()
        engine = Engine(orig_model, tokenizer)  # 使用未编译模型（输入形状变化）

        # 6 个 ChatCORE 任务的配置
        all_tasks = ['ARC-Easy', 'ARC-Challenge', 'MMLU', 'GSM8K', 'HumanEval', 'SpellingBee']
        categorical_tasks = {'ARC-Easy', 'ARC-Challenge', 'MMLU'}      # 分类任务（比较选项概率）
        baseline_accuracies = {                                        # 各任务的随机基线准确率
            'ARC-Easy': 0.25, 'ARC-Challenge': 0.25, 'MMLU': 0.25,   # 4 选 1 → 25%
            'GSM8K': 0.0, 'HumanEval': 0.0, 'SpellingBee': 0.0,     # 生成式任务 → 0%
        }

        task_results = {}
        for task_name in all_tasks:
            # 分类任务用 chatcore_max_cat，生成式任务用 chatcore_max_sample
            limit = args.chatcore_max_cat if task_name in categorical_tasks else args.chatcore_max_sample
            max_problems = None if limit < 0 else limit  # -1 表示不做数量限制
            acc = run_chat_eval(task_name, orig_model, tokenizer, engine,
                                batch_size=args.device_batch_size, max_problems=max_problems)
            task_results[task_name] = acc
            print0(f"  {task_name}: {100 * acc:.2f}%")

        # 计算 centered mean（去基线后的平均准确率）
        def centered_mean(tasks):
            """
            计算一组任务的 centered accuracy 均值。

            centered_acc = (acc - baseline) / (1.0 - baseline)
            直观理解：有多大比例从随机猜测提升到了完美水平。
            """
            return sum(
                (task_results[t] - baseline_accuracies[t]) / (1.0 - baseline_accuracies[t])
                for t in tasks
            ) / len(tasks)

        chatcore = centered_mean(all_tasks)                          # 总体 ChatCORE
        chatcore_cat = centered_mean(categorical_tasks)              # 仅分类任务的 ChatCORE
        print0(f"Step {step:05d} | ChatCORE: {chatcore:.4f} | ChatCORE_cat: {chatcore_cat:.4f}")

        wandb_run.log({
            "step": step,
            "total_training_flops": flops_so_far,
            "chatcore_metric": chatcore,
            "chatcore_cat": chatcore_cat,
            **{f"chatcore/{task_name}": acc for task_name, acc in task_results.items()},
        })
        model.train()

    # ============ 保存检查点 ============
    # 仅在训练结束时保存（所有 rank 参与，各自保存优化器的分片状态）
    if last_step:
        output_dirname = args.model_tag if args.model_tag else f"d{depth}"  # 如 d12
        checkpoint_dir = os.path.join(base_dir, "chatsft_checkpoints", output_dirname)
        save_checkpoint(
            checkpoint_dir,
            step,
            orig_model.state_dict(),         # 模型参数
            optimizer.state_dict(),          # 优化器状态
            {
                "step": step,
                "val_bpb": val_bpb,          # 最终验证 loss
                "model_config": {            # 模型架构配置
                    "sequence_len": args.max_seq_len,
                    "vocab_size": tokenizer.get_vocab_size(),
                    "n_layer": depth,
                    "n_head": model.config.n_head,
                    "n_kv_head": model.config.n_kv_head,
                    "n_embd": model.config.n_embd,
                    "window_pattern": model.config.window_pattern,
                },
                "user_config": user_config,  # 训练脚本的 CLI 参数
            },
            rank=ddp_rank,
        )

    # --- 终止条件 ---
    if last_step:
        break

    # =========================================================================
    # 单个训练步（与 base_train.py 结构相同）
    # 流程：梯度累积（多次前向+反向）→ 调度器更新 → 优化器步进 → 清零梯度
    # =========================================================================

    synchronize()
    t0 = time.time()
    for micro_step in range(grad_accum_steps):
        # 前向传播 + 损失计算
        loss = model(x, y)
        train_loss = loss.detach()              # 分离用于日志
        loss = loss / grad_accum_steps          # 归一化：累积梯度 = 梯度和
        if scaler is not None:                   # FP16 模式：放大 loss 防止梯度下溢
            scaler.scale(loss).backward()
        else:
            loss.backward()                      # BF16/FP32：直接反向传播
        # 流水线预取下一批数据（GPU 忙于反向传播时并行加载）
        x, y = next(train_loader)
        # 更新进度（单调递增：只有当新进度更大时才更新）
        progress = max(progress, approx_progress)

    # --- 优化器更新 ---
    # 设置调度器参数
    lrm = get_lr_multiplier(progress)           # 学习率乘子（按训练进度）
    muon_momentum = get_muon_momentum(step)     # Muon 动量（按训练步数）

    for group in optimizer.param_groups:
        group["lr"] = group["initial_lr"] * lrm
        if group['kind'] == 'muon':              # Muon 优化器需要动量设置
            group["momentum"] = muon_momentum

    if scaler is not None:
        scaler.unscale_(optimizer)               # 撤消梯度缩放
        # DDP 环境下：各 rank 的 found_inf 状态需要同步（任一 rank 有 inf → 全部跳过）
        if is_ddp_initialized():
            for v in scaler._found_inf_per_device(optimizer).values():
                dist.all_reduce(v, op=dist.ReduceOp.MAX)
        scaler.step(optimizer)
        scaler.update()                          # 更新 GradScaler 的缩放因子
    else:
        optimizer.step()                         # BF16/FP32：直接更新参数

    model.zero_grad(set_to_none=True)           # 清零梯度（set_to_none 直接释放内存）
    synchronize()
    t1 = time.time()
    dt = t1 - t0                                # 当前步实际计算时间（秒）
    # =========================================================================

    # --- 状态更新 ---
    step += 1

    # --- 日志记录（仅 CPU 操作）---
    smooth_train_loss = ema_beta * smooth_train_loss + (1 - ema_beta) * train_loss.item()  # EMA 训练 loss
    debiased_smooth_loss = smooth_train_loss / (1 - ema_beta ** (step + 1))               # 消除 EMA 冷启动偏差

    pct_done = 100 * progress                               # 训练进度百分比
    tok_per_sec = int(args.total_batch_size / dt)            # 吞吐量（token/秒）
    flops_per_sec = num_flops_per_token * args.total_batch_size / dt  # 每秒实际 FLOPs
    mfu = 100 * flops_per_sec / (gpu_peak_flops * ddp_world_size)   # 模型算力利用率（MFU）

    if step > 10:
        total_training_time += dt  # 排除前 10 步（编译/预热开销）

    print0(f"step {step:05d} ({pct_done:.2f}%) | loss: {debiased_smooth_loss:.6f} | lrm: {lrm:.2f} | dt: {dt * 1000:.2f}ms | tok/sec: {tok_per_sec:,} | mfu: {mfu:.2f} | epoch: {current_epoch} | total time: {total_training_time/60:.2f}m")

    # 每 10 步记录一次 wandb（比预训练的 100 步更频繁，SFT 的步数通常更少）
    if step % 10 == 0:
        wandb_run.log({
            "step": step,
            "total_training_flops": flops_so_far,
            "total_training_time": total_training_time,
            "train/loss": debiased_smooth_loss,
            "train/lrm": lrm,
            "train/dt": dt,
            "train/tok_per_sec": tok_per_sec,
            "train/mfu": mfu,
            "train/epoch": current_epoch,
        })

    # --- 垃圾回收管理 ---
    # 与预训练相同的策略：手动管理 GC 以避免训练过程中的卡顿
    if step == 1:
        gc.collect()   # 清理初始化阶段的垃圾
        gc.freeze()    # 冻结当前存活对象（排除在自动 GC 之外）
        gc.disable()   # 完全禁用自动 GC
    elif step % 5000 == 0:  # 每 5000 步...
        gc.collect()          # 手动安全回收一次

# =============================================================================
# 第十部分：训练结束后的统计输出与报告
# =============================================================================

print0(f"Peak memory usage: {get_max_memory() / 1024 / 1024:.2f}MiB")
print0(f"Total training time: {total_training_time/60:.2f}m")
print0(f"Minimum validation bpb: {min_val_bpb:.4f}")

# 写入报告
from nanochat.report import get_report
get_report().log(section="SFT", data=[
    user_config,  # CLI 参数（方便复现）
    {  # 训练配置统计
        "Number of iterations": step,
        "DDP world size": ddp_world_size,
    },
    {  # 训练结果统计
        "Minimum validation bpb": min_val_bpb,
    }
])

# --- 清理资源 ---
wandb_run.finish()  # 结束 wandb 运行
compute_cleanup()   # 清理 DDP 通信组和进程
