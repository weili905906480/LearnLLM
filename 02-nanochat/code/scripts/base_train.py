"""
训练 GPT 基础模型。从项目根目录运行：

单机单卡:
python -m scripts.base_train

分布式训练（例如 8 卡）:
torchrun --nproc_per_node=8 -m scripts.base_train

如果只有 CPU/Macbook，建议训练一个小得多的模型：
python -m scripts.base_train --depth=4 --max-seq-len=512 --device-batch-size=1 --eval-tokens=512 --core-metric-every=-1 --total-batch-size=512 --num-iterations=20

训练脚本的整体流程：
1. 解析命令行参数（模型架构、训练超参、评估频率等）
2. 初始化计算环境（设备检测、DDP 分布式通信、混合精度配置）
3. 初始化 wandb 日志记录
4. 检测 Flash Attention 3 是否可用
5. 加载分词器（tokenizer）
6. 构建模型（meta 设备上创建结构 → 目标设备上分配存储 → 初始化权重）
7. 可选：转换为 FP8 混合精度训练（减少显存、加速计算）
8. 编译模型（torch.compile）
9. 根据缩放定律（Scaling Laws）自动计算最优训练步数、批次大小、学习率等
10. 初始化优化器（Muon 优化矩阵参数 + AdamW 优化其他参数）
11. 初始化数据加载器
12. 进入训练循环：评估 → 保存检查点 → 前向反向传播 → 优化器更新 → 日志记录
13. 训练结束后打印统计信息并清理资源
"""

import os
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"  # 启用 PyTorch 内存分配器的可扩展段，减少显存碎片
import gc
import json
import time
import math
import argparse
from dataclasses import asdict
from contextlib import contextmanager

import wandb
import torch
import torch.distributed as dist

from nanochat.gpt import GPT, GPTConfig, Linear
from nanochat.dataloader import tokenizing_distributed_data_loader_bos_bestfit, tokenizing_distributed_data_loader_with_state_bos_bestfit
from nanochat.common import compute_init, compute_cleanup, print0, DummyWandb, print_banner, get_base_dir, autodetect_device_type, get_peak_flops, COMPUTE_DTYPE, COMPUTE_DTYPE_REASON, is_ddp_initialized
from nanochat.tokenizer import get_tokenizer, get_token_bytes
from nanochat.checkpoint_manager import save_checkpoint, load_checkpoint
from nanochat.loss_eval import evaluate_bpb
from nanochat.engine import Engine
from nanochat.flash_attention import HAS_FA3
from scripts.base_eval import evaluate_core
print_banner()

# =============================================================================
# 第一部分：命令行参数解析
# 定义模型架构、训练策略、评估频率等所有可配置项
# =============================================================================

parser = argparse.ArgumentParser(description="Pretrain base model")

# --- 日志与运行管理 ---
parser.add_argument("--run", type=str, default="dummy", help="wandb 运行名称（'dummy' 表示禁用 wandb 日志）")

# --- 运行时环境 ---
parser.add_argument("--device-type", type=str, default="", help="计算设备类型：cuda|cpu|mps（空字符串 = 自动检测）")

# --- FP8 训练（混合精度）---
# FP8 是 NVIDIA H100+ GPU 上支持的 8 位浮点格式，相比 BF16 可以进一步节省显存和加速计算
parser.add_argument("--fp8", action="store_true", help="启用 FP8 训练（需要 H100+ GPU 和 torchao 库）")
parser.add_argument("--fp8-recipe", type=str, default="tensorwise", choices=["rowwise", "tensorwise"],
                    help="FP8 缩放策略：tensorwise（更快，推荐）/ rowwise（更精确但较慢）")

# --- 模型架构 ---
# 模型总参数量 ≈ depth * (depth * aspect_ratio)^2，depth 是核心缩放参数
parser.add_argument("--depth", type=int, default=20, help="Transformer 模型的层数（深度）")
parser.add_argument("--aspect-ratio", type=int, default=64, help="模型维度 = depth * aspect_ratio")
parser.add_argument("--head-dim", type=int, default=128, help="每个注意力头的目标维度")
parser.add_argument("--max-seq-len", type=int, default=2048, help="最大上下文长度（序列长度）")
parser.add_argument("--window-pattern", type=str, default="SSSL",
                    help="滑动窗口模式（按层平铺）：L=全上下文注意力, S=半上下文注意力（如 'SSL'）")

# --- 训练步数与数据量 ---
# 以下三项按优先级决定训练步数：--num-iterations > --target-flops > --target-param-data-ratio
parser.add_argument("--num-iterations", type=int, default=-1, help="显式指定优化步数（-1 = 禁用）")
parser.add_argument("--target-flops", type=float, default=-1.0, help="根据目标 FLOPs 自动计算训练步数（-1 = 禁用）")
parser.add_argument("--target-param-data-ratio", type=float, default=12,
                    help="根据数据:参数比自动计算训练步数（Chinchilla 最优比 = 20, -1 = 禁用）")

# --- 优化器参数 ---
parser.add_argument("--device-batch-size", type=int, default=32,
                    help="每张卡的单步批次大小。如果显存不足，建议逐步降低：16→8→4...")
parser.add_argument("--total-batch-size", type=int, default=-1,
                    help="全局总批次大小（以 token 计）。常用值如 524288（-1 = 自动计算最优值）")
parser.add_argument("--embedding-lr", type=float, default=0.3, help="嵌入层参数的学习率（使用 AdamW 优化器）")
parser.add_argument("--unembedding-lr", type=float, default=0.008, help="解嵌层参数的学习率（使用 AdamW 优化器）")
parser.add_argument("--weight-decay", type=float, default=0.28, help="Muon 优化器的权重衰减系数（作用于权重矩阵）")
parser.add_argument("--matrix-lr", type=float, default=0.02, help="矩阵参数的学习率（使用 Muon 优化器）")
parser.add_argument("--scalar-lr", type=float, default=0.5, help="标量参数的学习率（如 resid_lambdas、x0_lambdas）")
parser.add_argument("--warmup-steps", type=int, default=40, help="学习率预热步数（从 0 线性增加到目标值）")
parser.add_argument("--warmdown-ratio", type=float, default=0.65, help="学习率衰减占总步数的比例")
parser.add_argument("--final-lr-frac", type=float, default=0.05, help="最终学习率相对于初始学习率的比例")
parser.add_argument("--resume-from-step", type=int, default=-1, help="从指定步数恢复训练（-1 = 不恢复）")

# --- 评估配置 ---
parser.add_argument("--eval-every", type=int, default=250, help="每隔 N 步评估验证集 loss（-1 = 禁用）")
parser.add_argument("--eval-tokens", type=int, default=80 * 524288, help="验证集评估时使用的 token 总数")
parser.add_argument("--core-metric-every", type=int, default=2000, help="每隔 N 步计算 CORE 评估指标（-1 = 禁用）")
parser.add_argument("--core-metric-max-per-task", type=int, default=500, help="CORE 指标中每个任务的最大样本数")
parser.add_argument("--sample-every", type=int, default=2000, help="每隔 N 步从模型采样生成文本（-1 = 禁用）")
parser.add_argument("--save-every", type=int, default=-1, help="每隔 N 步保存检查点（-1 = 仅在训练结束时保存）")

# --- 输出配置 ---
parser.add_argument("--model-tag", type=str, default=None, help="自定义模型标签，用于检查点目录名称")

args = parser.parse_args()
user_config = vars(args).copy()  # 保存用户的原始配置，后续用于日志记录和报告

# =============================================================================
# 第二部分：计算环境初始化
# 1. 检测设备类型（CUDA GPU / CPU / Apple MPS）
# 2. 初始化 DDP 分布式通信（单机多卡或多机多卡）
# 3. 获取 GPU 的峰值算力（用于后续计算 MFU）
# =============================================================================

device_type = autodetect_device_type() if args.device_type == "" else args.device_type

# compute_init 返回 DDP 相关的通信组信息
# ddp: 是否为分布式训练
# ddp_rank: 当前进程的全局编号（0 到 world_size-1）
# ddp_local_rank: 当前进程在单机上的本地编号
# ddp_world_size: 分布式训练的进程总数（通常等于 GPU 数量）
# device: 当前进程绑定的 torch 设备对象
ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init(device_type)

# 只有 rank=0（主进程）负责日志记录、检查点保存等非训练任务
master_process = ddp_rank == 0

# 根据设备类型选择同步和内存统计函数
synchronize = torch.cuda.synchronize if device_type == "cuda" else lambda: None  # CUDA 同步（用于准确计时）
get_max_memory = torch.cuda.max_memory_allocated if device_type == "cuda" else lambda: 0  # 峰值显存使用量

if device_type == "cuda":
    gpu_device_name = torch.cuda.get_device_name(0)
    gpu_peak_flops = get_peak_flops(gpu_device_name)  # GPU 的理论峰值 BF16 FLOPS
    print0(f"GPU: {gpu_device_name} | Peak FLOPS (BF16): {gpu_peak_flops:.2e}")
else:
    gpu_peak_flops = float('inf')  # CPU/MPS 上 MFU（模型算力利用率）没有意义，设为无穷大
print0(f"COMPUTE_DTYPE: {COMPUTE_DTYPE} ({COMPUTE_DTYPE_REASON})")

# --- wandb 日志初始化 ---
# 如果 run 名为 "dummy" 或不是主进程，则使用 DummyWandb（不实际记录日志）
use_dummy_wandb = args.run == "dummy" or not master_process
wandb_run = DummyWandb() if use_dummy_wandb else wandb.init(project="nanochat", name=args.run, config=user_config)

# =============================================================================
# 第三部分：Flash Attention 3 检测与配置
# FA3 是 NVIDIA Hopper 架构（H100+）上的高效注意力实现，大幅提升训练效率
# 如果不可用，则回退到 PyTorch 的 SDPA（Scaled Dot-Product Attention）
# 注意：SDPA 不支持滑动窗口注意力模式！
# =============================================================================

from nanochat.flash_attention import USE_FA3
using_fa3 = USE_FA3
if using_fa3:
    print0("✓ Using Flash Attention 3 (Hopper GPU detected), efficient, new and awesome.")
else:
    print0("!" * 80)
    if HAS_FA3 and COMPUTE_DTYPE != torch.bfloat16:
        # FA3 仅支持 bf16，如果 COMPUTE_DTYPE 不是 bf16 则无法使用
        print0(f"WARNING: Flash Attention 3 only supports bf16, but COMPUTE_DTYPE={COMPUTE_DTYPE}. Using PyTorch SDPA fallback")
    else:
        print0("WARNING: Flash Attention 3 not available, using PyTorch SDPA fallback")
    print0("WARNING: Training will be less efficient without FA3")
    if args.window_pattern != "L":
        # 关键警告：SDPA 不支持滑动窗口，使用 SSL/SSSL 等模式会严重影响 GPU 利用率
        print0(f"WARNING: SDPA has no support for sliding window attention (window_pattern='{args.window_pattern}'). Your GPU utilization will be terrible.")
        print0("WARNING: Recommend using --window-pattern L for full context attention without alternating sliding window patterns.")
    print0("!" * 80)

# =============================================================================
# 第四部分：加载分词器
# 分词器用于将文本转换为 token ID 序列，是模型输入输出的桥梁
# token_bytes 用于计算 BPB（Bits Per Byte）指标
# =============================================================================

tokenizer = get_tokenizer()
token_bytes = get_token_bytes(device=device)  # 每个 token 对应的 UTF-8 字节数张量（用于 loss 归一化）
vocab_size = tokenizer.get_vocab_size()
print0(f"Vocab size: {vocab_size:,}")

# =============================================================================
# 第五部分：构建模型
# 分三步完成模型初始化：
# 1. 在 meta 设备上构建（只记录张量形状和数据类型，不分配实际内存）
# 2. 在目标设备上分配未初始化的存储空间（to_empty）
# 3. 按设计好的分布初始化所有权重（init_weights）
# =============================================================================

def build_model_meta(depth):
    """
    在 meta 设备上构建模型（仅记录形状/数据类型，不分配实际数据）。

    设计原则：
    - model_dim 会被向上取整到 head_dim 的倍数，确保整除
    - 这保证了 head_dim == args.head_dim 始终成立
    - FA3 要求 head_dim 能被 8 整除，这个设计自动满足该要求

    参数:
        depth: Transformer 的层数

    返回:
        model_meta: 在 meta 设备上的 GPT 模型实例（只有结构信息）
    """
    base_dim = depth * args.aspect_ratio  # 基础模型维度 = 深度 × 宽高比
    # 将 model_dim 向上取整到 head_dim 的整数倍
    model_dim = ((base_dim + args.head_dim - 1) // args.head_dim) * args.head_dim
    num_heads = model_dim // args.head_dim  # 注意力头的数量
    config = GPTConfig(
        sequence_len=args.max_seq_len,
        vocab_size=vocab_size,
        n_layer=depth,
        n_head=num_heads,
        n_kv_head=num_heads,  # GQA（分组查询注意力）中 KV 头数 = 查询头数（即不使用 GQA）
        n_embd=model_dim,
        window_pattern=args.window_pattern,
    )
    # 在 meta 设备上创建模型 —— 不分配任何实际 GPU/CPU 内存
    with torch.device("meta"):
        model_meta = GPT(config)
    return model_meta


# 构建模型并移到目标设备
model = build_model_meta(args.depth)                # 1) 在 meta 设备上构建（仅形状和数据类型，不分配数据）
model_config = model.config                           # 保存模型配置供后续使用
model_config_kwargs = asdict(model_config)             # 转为字典便于 JSON 序列化
print0(f"Model config:\n{json.dumps(model_config_kwargs, indent=2)}")
model.to_empty(device=device)                        # 2) 在目标设备上分配内存（张量数据未初始化，为随机值）
model.init_weights()                                 # 3) 按照 GPT 架构设计初始化所有参数

# --- 如果是从检查点恢复训练，则加载之前保存的模型参数 ---
base_dir = get_base_dir()
output_dirname = args.model_tag if args.model_tag else f"d{args.depth}"  # 检查点目录名：如 d12、d20 等
checkpoint_dir = os.path.join(base_dir, "base_checkpoints", output_dirname)
resuming = args.resume_from_step != -1
if resuming:
    print0(f"Resuming optimization from step {args.resume_from_step}")
    # 加载模型参数、优化器状态、元数据（含数据加载器状态）
    model_data, optimizer_data, meta_data = load_checkpoint(
        checkpoint_dir, args.resume_from_step, device, load_optimizer=True, rank=ddp_rank
    )
    model.load_state_dict(model_data, strict=True, assign=True)  # assign=True 直接复制张量数据（不创建新张量）
    del model_data  # 复制完成后立即释放原始检查点数据占用的内存

# =============================================================================
# 第六部分：FP8 混合精度训练初始化
# FP8 是 NVIDIA H100+ 的硬件特性，相比 BF16 能进一步降低显存和带宽需求
# 工作原理：将符合条件的 Linear 层替换为 Float8Linear 层
# 限制条件：输入/输出维度必须能被 16 整除（FP8 硬件要求），且维度不能太小
#
# 注意：必须在 torch.compile 之前完成 FP8 转换！
# =============================================================================

if args.fp8:
    if device_type != "cuda":
        print0("Warning: FP8 training requires CUDA, ignoring --fp8 flag")
    else:
        # 使用项目自定义的 FP8 实现（比 torchao 更简洁，API 完全兼容）
        from nanochat.fp8 import Float8LinearConfig, convert_to_float8_training

        import torch.nn as nn

        # FP8 模块过滤器：判断哪些 nn.Linear 层应该被转换为 Float8Linear
        def fp8_module_filter(mod: nn.Module, fqn: str) -> bool:
            """
            判断模块是否适合转换为 FP8。

            过滤条件（全部满足才转换）：
            1. 必须是 nn.Linear 层
            2. 输入和输出特征维度必须能被 16 整除（FP8 硬件对齐要求）
            3. 输入和输出维度的最小值必须 >= 128（小矩阵 FP8 收益低，保持 BF16）
            """
            if not isinstance(mod, nn.Linear):
                return False
            if mod.in_features % 16 != 0 or mod.out_features % 16 != 0:
                return False
            if min(mod.in_features, mod.out_features) < 128:
                return False
            return True

        fp8_config = Float8LinearConfig.from_recipe_name(args.fp8_recipe)
        num_linear = sum(1 for m in model.modules() if isinstance(m, nn.Linear))
        convert_to_float8_training(model, config=fp8_config, module_filter_fn=fp8_module_filter)
        num_fp8 = sum(1 for m in model.modules() if 'Float8' in type(m).__name__)
        num_skipped = num_linear - num_fp8
        print0(f"✓ FP8 training enabled ({args.fp8_recipe} scaling) - converted {num_fp8}/{num_linear} linear layers, skipped {num_skipped} (too small)")


# --- 用于临时禁用 FP8 的上下文管理器 ---
# 评估和推理时应该在 BF16 精度下进行，以获得更一致和准确的结果
@contextmanager
def disable_fp8(model):
    """
    临时将 Float8Linear 模块替换为普通的 nn.Linear，在上下文内以 BF16 精度运行。

    实现方式：
    由于 CastConfig 是不可变的数据类（frozen dataclass），无法直接修改 scaling_type，
    所以采用模块级别的替换策略：记录所有 Float8Linear 的位置 → 替换为 Linear →
    退出上下文时恢复。

    使用 device="meta" 创建 Linear 层可以避免 VRAM 峰值 —— 权重张量会在设置后直接共用。
    """
    import torch.nn as nn

    # 第一步：找到所有 Float8Linear 模块及其在模型树中的位置
    fp8_locations = []  # 列表元素: (父模块, 属性名, Float8Linear 模块实例)
    for name, module in model.named_modules():
        if 'Float8' in type(module).__name__:
            if '.' in name:
                parent_name, attr_name = name.rsplit('.', 1)
                parent = model.get_submodule(parent_name)
            else:
                parent = model
                attr_name = name
            fp8_locations.append((parent, attr_name, module))

    if not fp8_locations:
        yield  # 没有 FP8 模块，无需任何操作
        return

    # 第二步：将所有 Float8Linear → Linear（共用权重张量，不复制数据）
    for parent, attr_name, fp8_module in fp8_locations:
        linear = Linear(
            fp8_module.in_features,
            fp8_module.out_features,
            bias=fp8_module.bias is not None,
            device="meta",                    # 在 meta 设备创建壳子，避免额外 VRAM 分配
            dtype=fp8_module.weight.dtype,    # 匹配原始权重的数据类型
        )
        linear.weight = fp8_module.weight     # 直接引用原模块的权重张量（不复制！）
        if fp8_module.bias is not None:
            linear.bias = fp8_module.bias
        setattr(parent, attr_name, linear)

    try:
        yield  # 在此上下文内，模型以 BF16 精度运行
    finally:
        # 第三步：恢复所有 Float8Linear 模块
        for parent, attr_name, fp8_module in fp8_locations:
            setattr(parent, attr_name, fp8_module)

# =============================================================================
# 第七部分：编译模型
# torch.compile 使用 JIT 编译优化计算图，减少 Python 开销和 CUDA kernel 启动次数
# dynamic=False：模型输入形状不会变化，编译器可以做更激进的优化
# =============================================================================

# 保留原始未编译模型的引用：
# - 用于保存检查点（state_dict 是原始参数名）
# - 用于评估和推理（输入形状可能变化，编译模型无法处理）
orig_model = model
model = torch.compile(model, dynamic=False)  # 编译后训练循环的 Python 开销显著降低

# =============================================================================
# 第八部分：缩放定律（Scaling Laws）与超参自动推导
#
# 根据目标模型规模和已验证的缩放规律，自动计算：
# 1. 最优训练 token 数（基于数据:参数比）
# 2. 最优全局批次大小（基于 Power Lines 论文的 B_opt ∝ D^0.383）
# 3. 学习率缩放修正（批次越大，学习率可以越高）
# 4. 权重衰减缩放修正（基于 T_epoch 框架）
#
# 这些计算使得从 d12 参考模型到任意深度的模型的超参迁移变得自动化（类似 muP）
# =============================================================================

# 获取模型的参数计数，用于缩放定律计算
param_counts = model.num_scaling_params()
print0(f"Parameter counts:")
for key, value in param_counts.items():
    print0(f"{key:24s}: {value:,}")
num_params = param_counts['total']
num_flops_per_token = model.estimate_flops()  # 估计每个 token 的前向+反向 FLOPs
print0(f"Estimated FLOPs per token: {num_flops_per_token:e}")

# --- 1) 根据缩放定律计算最优训练 token 数量 ---
# 计算最优模型的 Token:Param 比例由 --target-param-data-ratio 决定（通过缩放定律实验确定）
# 模型已初始化，所以参数数量已知。最优 Token 数 = target-param-data-ratio × 参数量

def get_scaling_params(m):
    """
    获取用于缩放定律计算的参数数量。

    缩放定律研究表明，transformer 矩阵参数 + lm_head 参数
    给出最干净的缩放关系（详见 dev/LOG.md 2026年1月27日）。
    """
    params_counts = m.num_scaling_params()
    scaling_params = params_counts['transformer_matrices'] + params_counts['lm_head']
    return scaling_params

num_scaling_params = get_scaling_params(model)
# 对该模型而言的最优 token 数
target_tokens = int(args.target_param_data_ratio * num_scaling_params)

# d12 是我们的参考模型 —— 很多超参都在 d12 上调优，然后通过缩放定律迁移到更深/浅的模型
d12_ref = build_model_meta(12)  # 在 meta 设备上创建 d12 参考模型
D_REF = args.target_param_data_ratio * get_scaling_params(d12_ref)  # d12 的计算最优训练 token 数（实验测定）
B_REF = 2 ** 19  # d12 的最优批次大小 ≈ 524,288 tokens（实验测定）

# --- 2) 计算最优全局批次大小 ---
# 参考 Power Lines 论文 (https://arxiv.org/abs/2505.13738)：
# B_opt ∝ D^0.383，即最优批次大小随数据量以约 0.383 次方增长
# 例如：如果 D 从 d12 到 d24 翻倍，B 应增长 2^0.383 ≈ 1.3x
total_batch_size = args.total_batch_size  # 用户可能手动指定
if total_batch_size == -1:
    batch_size_ratio = target_tokens / D_REF
    predicted_batch_size = B_REF * batch_size_ratio ** 0.383
    # 取最接近的 2 的幂，GPU 硬件上 2 的幂的批次大小效率最高
    total_batch_size = 2 ** round(math.log2(predicted_batch_size))
    print0(f"Auto-computed optimal batch size: {total_batch_size:,} tokens")

# --- 3) 学习率缩放修正 ---
# 更大的批次 ➔ 梯度估计更准确 ➔ 可以使用更大的学习率
# SGD: η ∝ B/B_ref（线性缩放，nanochat 不使用）
# AdamW: η ∝ √(B/B_ref)（平方根缩放，标准做法）
# Muon: η ∝ √(B/B_ref)（假设与 AdamW 类似，尚未仔细研究）
batch_lr_scale = 1.0
batch_ratio = total_batch_size / B_REF
if batch_ratio != 1.0:
    batch_lr_scale = batch_ratio ** 0.5  # η ∝ √(B/B_ref)
    print0(f"Scaling LRs by {batch_lr_scale:.4f} for batch size {total_batch_size:,} (reference: {B_REF:,})")

# --- 4) 权重衰减缩放修正 ---
# 采用 T_epoch 框架 (https://arxiv.org/abs/2405.13698)
# 核心思想：T_epoch = B/(η·λ·D) 应保持恒定
# 结合上面的 η ∝ √(B/B_ref) 学习率缩放，推导可得：
# λ = λ_ref · √(B/B_ref) · (D_ref/D)
# 注意：该论文研究的是 AdamW，不是 Muon。此处直接借用 AdamW 理论
weight_decay_scaled = args.weight_decay * math.sqrt(total_batch_size / B_REF) * (D_REF / target_tokens)
if weight_decay_scaled != args.weight_decay:
    print0(f"Scaling weight decay from {args.weight_decay:.6f} to {weight_decay_scaled:.6f} for depth {args.depth}")

# =============================================================================
# 第九部分：初始化优化器（MuonAdamW 组合优化器）
#
# 参数分为三类，使用不同的优化器和超参：
# 1. 矩阵参数（Linear 权重）：使用 Muon 优化器（基于 Newton-Schulz 迭代的矩阵正交化）
# 2. 嵌入/解嵌层、标量参数：使用 AdamW 优化器
# 3. 标量参数（如 residual lambdas）：更高的学习率
# =============================================================================

optimizer = model.setup_optimizer(
    # --- AdamW 参数（用于嵌入层、解嵌层、标量等 1D 参数）---
    unembedding_lr=args.unembedding_lr * batch_lr_scale,  # 解嵌层学习率通常很小
    embedding_lr=args.embedding_lr * batch_lr_scale,      # 嵌入层学习率
    scalar_lr=args.scalar_lr * batch_lr_scale,            # 标量参数学习率最高
    # --- Muon 参数（用于矩阵权重）---
    matrix_lr=args.matrix_lr * batch_lr_scale,
    weight_decay=weight_decay_scaled,
)

if resuming:
    optimizer.load_state_dict(optimizer_data)
    del optimizer_data

# =============================================================================
# 第十部分：混合精度的梯度缩放器
# - BF16: 不需要 GradScaler（指数位与 FP32 相同，梯度不会溢出）
# - FP32: 不需要 GradScaler（完全精度）
# - FP16: 需要 GradScaler（指数位只有 5 位，梯度容易下溢）
# =============================================================================

scaler = torch.amp.GradScaler() if COMPUTE_DTYPE == torch.float16 else None
if scaler is not None:
    print0("GradScaler enabled for fp16 training")

# =============================================================================
# 第十一部分：初始化数据加载器
# 使用分布式 token 化数据加载器，自动处理：
# - 数据分片（按 DDP rank 分配不同的数据子集）
# - BOS token 添加
# - 批次拼接（bestfit 策略：尽可能填满每个批次）
# =============================================================================

# 如果恢复训练，传入之前保存的数据加载器状态（恢复数据读取位置）
dataloader_resume_state_dict = None if not resuming else meta_data["dataloader_state_dict"]
train_loader = tokenizing_distributed_data_loader_with_state_bos_bestfit(
    tokenizer, args.device_batch_size, args.max_seq_len, split="train", device=device,
    resume_state_dict=dataloader_resume_state_dict
)
# 验证集数据加载器使用工厂函数（每次评估时重建，确保从头遍历验证集）
build_val_loader = lambda: tokenizing_distributed_data_loader_bos_bestfit(
    tokenizer, args.device_batch_size, args.max_seq_len, split="val", device=device
)
# 预取第一批训练数据（输入 x 和标签 y，以及数据加载器的状态字典）
x, y, dataloader_state_dict = next(train_loader)

# =============================================================================
# 第十二部分：确定训练步数和初始化调度器
#
# 训练步数由以下三项决定（按优先级）：
#   1. --num-iterations（显式指定）
#   2. --target-flops（根据目标计算量推算）
#   3. --target-param-data-ratio（根据数据:参数比推算，最常用）
#
# 三个调度器：
#   1. 学习率调度器：线性预热 → 恒定 → 线性衰减
#   2. Muon 动量调度器：从 0.85 预热到 0.97 → 恒定 → 衰减到 0.90
#   3. 权重衰减调度器：余弦衰减到零
# =============================================================================

# 确定训练步数（三种方式的优先级递减）
assert args.num_iterations > 0 or args.target_param_data_ratio > 0 or args.target_flops > 0
if args.num_iterations > 0:
    num_iterations = args.num_iterations
    print0(f"Using user-provided number of iterations: {num_iterations:,}")
elif args.target_flops > 0:
    # 根据目标 FLOPs 反推训练步数（用于缩放定律分析，如 runs/scaling_laws.sh）
    num_iterations = round(args.target_flops / (num_flops_per_token * total_batch_size))
    print0(f"Calculated number of iterations from target FLOPs: {num_iterations:,}")
elif args.target_param_data_ratio > 0:
    # 根据数据:参数比计算训练步数（最常见的用法）
    num_iterations = target_tokens // total_batch_size
    print0(f"Calculated number of iterations from target data:param ratio: {num_iterations:,}")
else:
    raise ValueError("No training horizon specified")

total_tokens = total_batch_size * num_iterations  # 整个训练过程中的实际总 token 数
print0(f"Total number of training tokens: {total_tokens:,}")
print0(f"Tokens : Scaling params ratio: {total_batch_size * num_iterations / num_scaling_params:.2f}")  # 例：Chinchilla 最优≈20
print0(f"Total training FLOPs estimate: {num_flops_per_token * total_tokens:e}")

# --- 学习率调度器：三阶段（预热 → 恒定 → 衰减）---
def get_lr_multiplier(it):
    """
    计算步数 it 处的学习率乘子（相对于初始学习率）。

    阶段：
    1. 线性预热：从 it=0 开始，学习率从 0 线性增加到 1.0（持续 warmup_steps 步）
    2. 恒定阶段：学习率保持 1.0
    3. 线性衰减：学习率从 1.0 线性降到 final_lr_frac（持续 warmdown_ratio 步）
    """
    warmup_iters = args.warmup_steps
    warmdown_iters = round(args.warmdown_ratio * num_iterations)
    if it < warmup_iters:
        # 阶段 1：线性预热
        return (it + 1) / warmup_iters
    elif it <= num_iterations - warmdown_iters:
        # 阶段 2：恒定
        return 1.0
    else:
        # 阶段 3：线性衰减
        progress = (num_iterations - it) / warmdown_iters
        return progress * 1.0 + (1 - progress) * args.final_lr_frac


# --- Muon 动量调度器 ---
# 动量值影响 Muon 优化器的更新方向和幅度的连续性
def get_muon_momentum(it):
    """
    计算步数 it 处的 Muon 动量值。

    阶段：
    1. 前 400 步：从 0.85 线性增加到 0.97（快速预热）
    2. 中间阶段：保持 0.97
    3. 学习率衰减期间：从 0.97 线性降到 0.90（让模型在训练末期更稳定地收敛）
    """
    warmdown_iters = round(args.warmdown_ratio * num_iterations)
    warmdown_start = num_iterations - warmdown_iters
    if it < 400:
        # 阶段 1：快速预热到 0.97
        frac = it / 400
        return (1 - frac) * 0.85 + frac * 0.97
    elif it >= warmdown_start:
        # 阶段 3：学习率衰减时动量同步降低
        progress = (it - warmdown_start) / warmdown_iters
        return 0.97 * (1 - progress) + 0.90 * progress
    else:
        # 阶段 2：恒定
        return 0.97


# --- 权重衰减调度器 ---
# 所有权重的权重衰减在整个训练过程中按余弦曲线衰减至零
def get_weight_decay(it):
    """
    计算步数 it 处的权重衰减系数。

    采用余弦调度：从 weight_decay_scaled 开始，在整个训练过程中
    平滑衰减到 0。余弦衰减在初期变化平缓，末期快速趋零。
    """
    return weight_decay_scaled * 0.5 * (1 + math.cos(math.pi * it / num_iterations))

# =============================================================================
# 第十三部分：训练循环
#
# 每次迭代包含：
# 1. 条件评估：验证 loss → CORE 指标 → 采样生成 → 保存检查点
# 2. 梯度累积：将全局批次拆分为多个微批次，累积梯度后一次性更新
# 3. 优化器更新：设置调度器值 → 反缩放梯度 → 更新参数 → 清零梯度
# 4. 日志记录：训练 loss、学习率、吞吐量、MFU、ETA
# =============================================================================

# --- 初始化循环状态变量 ---
if not resuming:
    # 全新训练：从零开始
    step = 0
    val_bpb = None                                          # 验证集 BPB（Bits Per Byte），loss 的替代指标
    min_val_bpb = float("inf")                              # 最佳验证 BPB（用于追踪模型最优状态）
    smooth_train_loss = 0                                    # 训练 loss 的指数移动平均（EMA）
    total_training_time = 0                                  # 累积训练时间（排除前 10 步的热身步数）
else:
    # 恢复训练：从检查点恢复状态
    step = meta_data["step"]
    loop_state = meta_data["loop_state"]
    val_bpb = meta_data["val_bpb"]
    min_val_bpb = loop_state["min_val_bpb"]
    smooth_train_loss = loop_state["smooth_train_loss"]
    total_training_time = loop_state["total_training_time"]

# --- 计算梯度累积步数 ---
# 全局批次大小可能远大于单张卡的单次前向能容纳的 Token 数
# 通过梯度累积：将大批次拆分为多个小批次，每个小批次独立前向+反向，累积梯度，最后一次性更新
tokens_per_fwdbwd = args.device_batch_size * args.max_seq_len                        # 单张卡的单次前向 Token 数
world_tokens_per_fwdbwd = tokens_per_fwdbwd * ddp_world_size                         # 所有卡的单次前向总 Token 数
assert total_batch_size % world_tokens_per_fwdbwd == 0, \
    f"total_batch_size ({total_batch_size}) 必须能被 world_tokens_per_fwdbwd ({world_tokens_per_fwdbwd}) 整除"
grad_accum_steps = total_batch_size // world_tokens_per_fwdbwd                        # 梯度累积步数
print0(f"Tokens / micro-batch / rank: {args.device_batch_size} x {args.max_seq_len} = {tokens_per_fwdbwd:,}")
print0(f"Tokens / micro-batch: {world_tokens_per_fwdbwd:,}")
print0(f"Total batch size {total_batch_size:,} => gradient accumulation steps: {grad_accum_steps}")

# =============================================================================
# 主训练循环
# 循环运行 num_iterations + 1 次（最后一步仅用于最终的评估和检查点保存）
# =============================================================================
while True:
    last_step = step == num_iterations  # 最后一步：只做评估和保存，不做参数更新
    flops_so_far = num_flops_per_token * total_batch_size * step  # 从训练开始到目前为止的总 FLOPs

    # ============ 评估块 1：验证集 BPB 评估 ============
    # BPB（Bits Per Byte）：比 loss 更可解释的指标，衡量每个字节需要的比特数
    # 所有 DDP rank 都参与评估（分布式计算 loss 的平均值）
    if args.eval_every > 0 and (last_step or step % args.eval_every == 0):
        model.eval()                                                    # 切换到评估模式（禁用 dropout 等）
        val_loader = build_val_loader()                                 # 创建新的验证集数据加载器
        eval_steps = args.eval_tokens // (args.device_batch_size * args.max_seq_len * ddp_world_size)
        with disable_fp8(model):                                        # 确保在 BF16 精度下评估
            val_bpb = evaluate_bpb(model, val_loader, eval_steps, token_bytes)
        print0(f"Step {step:05d} | Validation bpb: {val_bpb:.6f}")
        if val_bpb < min_val_bpb:
            min_val_bpb = val_bpb
        wandb_run.log({
            "step": step,
            "total_training_flops": flops_so_far,
            "total_training_time": total_training_time,
            "val/bpb": val_bpb,
        })
        model.train()                                                   # 恢复训练模式

    # ============ 评估块 2：CORE 指标评估 ============
    # CORE 是一组下游任务的综合评估指标（类似 MMLU-lite）
    # 使用未编译的原始模型（因为输入形状随任务变化）
    # 在 BF16 精度下评估（禁用 FP8）
    results = {}
    if args.core_metric_every > 0 and (last_step or (step > 0 and step % args.core_metric_every == 0)):
        model.eval()
        with disable_fp8(orig_model):
            results = evaluate_core(orig_model, tokenizer, device, max_per_task=args.core_metric_max_per_task)
        print0(f"Step {step:05d} | CORE metric: {results['core_metric']:.4f}")
        wandb_run.log({
            "step": step,
            "total_training_flops": flops_so_far,
            "core_metric": results["core_metric"],
            "centered_results": results["centered_results"],
        })
        model.train()

    # ============ 评估块 3：模型采样生成 ============
    # 用固定的提示词观察模型生成能力的变化（仅主进程执行）
    # 使用未编译的原始模型（输入形状不断变化）
    if args.sample_every > 0 and master_process and (last_step or (step > 0 and step % args.sample_every == 0)):
        model.eval()
        prompts = [
            "The capital of France is",
            "The chemical symbol of gold is",
            "If yesterday was Friday, then tomorrow will be",
            "The opposite of hot is",
            "The planets of the solar system are:",
            "My favorite color is",
            "If 5*x + 3 = 13, then x is",
        ]
        engine = Engine(orig_model, tokenizer)                         # 使用原始模型避免重编译
        for prompt in prompts:
            tokens = tokenizer(prompt, prepend="<|bos|>")
            with disable_fp8(orig_model):                              # BF16 推理
                sample, _ = engine.generate_batch(tokens, num_samples=1, max_tokens=16, temperature=0)
            print0(tokenizer.decode(sample[0]))                        # temperature=0 = 贪心解码（确定性输出）
        model.train()

    # ============ 保存检查点 ============
    # 保存条件：训练结束时、或每隔 save_every 步（排除第 0 步和恢复训练时的重复保存）
    if last_step or (step > 0 and step != args.resume_from_step and args.save_every > 0 and step % args.save_every == 0):
        save_checkpoint(
            checkpoint_dir,
            step,
            orig_model.state_dict(),           # 模型参数（使用未编译模型以确保兼容性）
            optimizer.state_dict(),            # 优化器状态（含动量缓冲区等）
            {                                   # 元数据（保存为 JSON）
                "step": step,
                "val_bpb": val_bpb,            # 当前验证 loss
                "model_config": model_config_kwargs,
                "user_config": user_config,     # 训练脚本的输入参数
                "device_batch_size": args.device_batch_size,
                "max_seq_len": args.max_seq_len,
                "total_batch_size": total_batch_size,
                "dataloader_state_dict": dataloader_state_dict,  # 数据加载器状态（含 epoch、位置等）
                "loop_state": {                # 训练循环中所有可变状态（除 step 外）
                    "min_val_bpb": min_val_bpb,
                    "smooth_train_loss": smooth_train_loss,
                    "total_training_time": total_training_time,
                },
            },
            rank=ddp_rank,
        )

    # ============ 终止条件 ============
    # 当 step == num_iterations 时，已完成所有评估和检查点保存，退出循环
    # TODO: 未来应添加 loss 爆炸等异常情况的终止条件
    if last_step:
        break

    # =========================================================================
    # 单个训练步（核心）
    # 流程：梯度累积（多次前向+反向）→ 设置调度器参数 → 更新优化器 → 清零梯度
    # =========================================================================

    # --- 梯度累积循环 ---
    # 每一次微迭代执行一次前向传播和反向传播，梯度自动累积在参数的 .grad 属性中
    synchronize()                              # CUDA 同步（确保计时准确）
    t0 = time.time()                           # 记录当前步的开始时间
    for micro_step in range(grad_accum_steps):
        loss = model(x, y)                     # 前向传播（使用当前微批次的输入）
        train_loss = loss.detach()             # 保存分离后的 loss 用于日志（不参与梯度计算）
        loss = loss / grad_accum_steps         # 关键：归一化损失！归一化后 .backward() 等效于梯度的求和
        if scaler is not None:                 # FP16 模式：使用 GradScaler 防止梯度下溢
            scaler.scale(loss).backward()      # loss 放大 → 反向传播 → 梯度保持有效范围
        else:
            loss.backward()                    # BF16/FP32：直接反向传播，不需要缩放
        # 流水线优化：在当前 GPU 忙于前向/反向时，预取下一批数据
        x, y, dataloader_state_dict = next(train_loader)

    # --- 优化器更新 ---
    # 1. 设置各参数组的学习率乘子（按调度器）
    lrm = get_lr_multiplier(step)
    muon_momentum = get_muon_momentum(step)
    muon_weight_decay = get_weight_decay(step)
    for group in optimizer.param_groups:
        group["lr"] = group["initial_lr"] * lrm    # 实际学习率 = 初始学习率 × 调度乘子
        if group['kind'] == 'muon':                 # Muon 优化器需要额外的动量和权重衰减设置
            group["momentum"] = muon_momentum
            group["weight_decay"] = muon_weight_decay

    # 2. 执行优化器步进
    if scaler is not None:
        scaler.unscale_(optimizer)                                      # FP16：先将梯度的缩放撤消
        # 分布式训练中，所有 rank 必须一致决定是否跳过某一步
        # 每个 rank 可能独立检测到 inf/nan 梯度，因此对 found_inf 做 all_reduce
        # 使用 MAX 归约：只要任一 rank 发现 inf，所有 rank 都跳过
        if is_ddp_initialized():
            for v in scaler._found_inf_per_device(optimizer).values():
                dist.all_reduce(v, op=dist.ReduceOp.MAX)
        scaler.step(optimizer)                                          # 更新参数（可能因 inf 而跳过）
        scaler.update()                                                 # 更新 GradScaler 的缩放因子
    else:
        optimizer.step()                                                # BF16/FP32：直接更新参数

    # 3. 清零梯度（set_to_none=True 比 zero_grad() 更高效：直接释放梯度内存而不写入零值）
    model.zero_grad(set_to_none=True)

    train_loss_f = train_loss.item()  # .item() 触发 CPU-GPU 同步（将 GPU 张量值复制到 Python float）
    synchronize()                     # 再次同步以确保计时准确
    t1 = time.time()
    dt = t1 - t0                      # 当前训练步的纯计算时间（秒）
    # =========================================================================

    # --- 日志记录与监控（仅 CPU 操作）---
    ema_beta = 0.9                    # EMA 衰减因子（用于平滑训练 loss 曲线）
    smooth_train_loss = ema_beta * smooth_train_loss + (1 - ema_beta) * train_loss_f  # 更新训练 loss 的指数移动平均
    debiased_smooth_loss = smooth_train_loss / (1 - ema_beta ** (step + 1))          # 消除 EMA 的冷启动偏差

    pct_done = 100 * step / num_iterations                         # 训练进度百分比
    tok_per_sec = int(total_batch_size / dt)                        # 每秒处理的 Token 数（吞吐量）
    flops_per_sec = num_flops_per_token * total_batch_size / dt     # 每秒的实际 FLOPs
    mfu = 100 * flops_per_sec / (gpu_peak_flops * ddp_world_size)  # MFU（Model FLOPs Utilization）：模型算力利用率
    #         ↑ 实际每秒 FLOPs / (理论峰值 FLOPs × GPU 数量)

    if step > 10:
        total_training_time += dt  # 仅统计第 10 步之后的时间（排除前 10 步的编译/预热开销）

    # 计算 ETA（预计完成时间）
    steps_done = step - 10
    if steps_done > 0:
        avg_time_per_step = total_training_time / steps_done          # 基于历史步骤的平均每步耗时
        remaining_steps = num_iterations - step
        eta_seconds = remaining_steps * avg_time_per_step
        eta_str = f" | eta: {eta_seconds/60:.1f}m"                   # ETA 以分钟为单位
    else:
        eta_str = ""

    # epoch 信息：演示数据集的遍历进度
    epoch = f"{dataloader_state_dict['epoch']} pq: {dataloader_state_dict['pq_idx']} rg: {dataloader_state_dict['rg_idx']}"
    print0(f"step {step:05d}/{num_iterations:05d} ({pct_done:.2f}%) | loss: {debiased_smooth_loss:.6f} | lrm: {lrm:.2f} | dt: {dt * 1000:.2f}ms | tok/sec: {tok_per_sec:,} | bf16_mfu: {mfu:.2f} | epoch: {epoch} | total time: {total_training_time/60:.2f}m{eta_str}")

    # 每 100 步记录一次到 wandb（减少日志写入频率）
    if step % 100 == 0:
        log_data = {
            "step": step,
            "total_training_flops": flops_so_far,
            "total_training_time": total_training_time,
            "train/loss": debiased_smooth_loss,
            "train/lrm": lrm,
            "train/dt": dt,
            "train/tok_per_sec": tok_per_sec,
            "train/mfu": mfu,
            "train/epoch": epoch,
        }
        wandb_run.log(log_data)

    # --- 状态更新 ---
    first_step_of_run = (step == 0) or (resuming and step == args.resume_from_step)
    step += 1  # 步数递增

    # --- 垃圾回收管理 ---
    # Python 的 GC（垃圾回收器）有时会过于活跃，在训练过程中花费 ~500ms 扫描循环引用，
    # 但实际清理的对象很少。因此采取人工干预策略：
    if first_step_of_run:
        gc.collect()   # 手动触发垃圾回收（清理初始化阶段产生的大量临时对象）
        gc.freeze()    # 将当前所有存活对象标记为 "不可被 GC 回收"（永久排除在 GC 扫描之外）
        gc.disable()   # 完全禁用自动 GC——所有后续分配的对象将不会被自动扫描
    elif step % 5000 == 0:  # 每 5000 步...
        gc.collect()          # 手动触发一次垃圾回收（安全网：防止长时间训练中真正的内存泄漏）

# =============================================================================
# 第十四部分：训练结束后的统计输出与报告
# =============================================================================

# 打印最终统计
print0(f"Peak memory usage: {get_max_memory() / 1024 / 1024:.2f}MiB")
print0(f"Total training time: {total_training_time/60:.2f}m")
if val_bpb is not None:
    print0(f"Minimum validation bpb: {min_val_bpb:.6f}")

# 将训练结果写入报告（用于后续分析和比较不同实验）
from nanochat.report import get_report
get_report().log(section="Base model training", data=[
    user_config,  # 命令行参数（方便复现实验）
    {  # 训练配置统计
        "Number of parameters": num_params,
        "Number of FLOPs per token": f"{num_flops_per_token:e}",
        "Calculated number of iterations": num_iterations,
        "Number of training tokens": total_tokens,
        "Tokens : Scaling params ratio": total_batch_size * num_iterations / num_scaling_params,
        "DDP world size": ddp_world_size,
        "warmup_steps": args.warmup_steps,
        "warmdown_ratio": args.warmdown_ratio,
        "final_lr_frac": args.final_lr_frac,
    },
    {  # 训练结果统计
        "Minimum validation bpb": min_val_bpb if val_bpb is not None else None,
        "Final validation bpb": val_bpb,
        "CORE metric estimate": results.get("core_metric", None),
        "MFU %": f"{mfu:.2f}%",
        "Total training flops": f"{flops_so_far:e}",
        "Total training time": f"{total_training_time/60:.2f}m",
        "Peak memory usage": f"{get_max_memory() / 1024 / 1024:.2f}MiB",
    }
])

# --- 清理资源 ---
wandb_run.finish()  # 结束 wandb 运行
compute_cleanup()   # 清理 DDP 分布式通信组和进程
