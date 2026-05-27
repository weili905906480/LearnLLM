# scripts/chat_sft.py 源码逐行详解

> 源文件：https://github.com/karpathy/nanochat/blob/master/scripts/chat_sft.py
>
> **SFT（Supervised Fine-Tuning，监督微调）** 是在预训练 base model 基础上，用对话数据进行微调，让模型学会"如何聊天"的阶段。

---

## 文件头注释

```python
"""
Supervised fine-tuning (SFT) the model.
Run as:

python -m scripts.chat_sft

Or torchrun for training:

torchrun --standalone --nproc_per_node=8 -m scripts.chat_sft -- --device-batch-size=16
"""
```

> 说明脚本用途和两种启动方式：
> - `python -m scripts.chat_sft`：单 GPU 运行
> - `torchrun --nproc_per_node=8`：8 卡分布式训练（DDP）

---

## 第一部分：导入依赖

```python
import gc
```
> 导入 Python 垃圾回收模块。训练时会手动控制 GC，避免其在训练步骤中触发造成卡顿（~500ms 的 GC 暂停会严重影响吞吐）。

```python
import argparse
```
> 命令行参数解析库，用于处理所有 `--xxx` 参数。

```python
import os
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
```
> 设置 PyTorch CUDA 内存分配器使用"可扩展内存段"策略。
> 默认分配器会产生大量内存碎片，导致 OOM。该设置让分配器动态扩展已有内存段，显著减少碎片。

```python
import time
```
> 用于计时每个训练步骤的耗时（`dt`）。

```python
import wandb
```
> Weights & Biases 实验追踪库，用于记录 loss、学习率、评估指标等训练曲线。

```python
import torch
```
> PyTorch 主库，所有张量操作和神经网络的基础。


```python
from nanochat.common import compute_init, compute_cleanup, print0, DummyWandb, get_base_dir,
    autodetect_device_type, get_peak_flops, COMPUTE_DTYPE, COMPUTE_DTYPE_REASON, is_ddp_initialized
```
> 从 nanochat 公共工具模块导入：
> - `compute_init`：初始化 DDP（分布式训练）和设备
> - `compute_cleanup`：训练结束后清理分布式进程组
> - `print0`：只在 rank 0（主进程）打印，避免多卡时重复输出
> - `DummyWandb`：不需要 wandb 时的空对象替代，保持接口统一
> - `get_base_dir`：获取模型 checkpoint 的根目录
> - `autodetect_device_type`：自动检测 cuda/mps/cpu
> - `get_peak_flops`：获取 GPU 理论峰值算力（用于计算 MFU）
> - `COMPUTE_DTYPE`：全局计算精度（bf16/fp32/fp16）
> - `COMPUTE_DTYPE_REASON`：说明为何选择该精度
> - `is_ddp_initialized`：判断是否在 DDP 模式下运行

```python
from nanochat.tokenizer import get_token_bytes
```
> 获取每个 token 对应的字节数，用于计算 bits-per-byte（bpb）这个与词表大小无关的损失指标。

```python
from nanochat.checkpoint_manager import save_checkpoint, load_model, load_optimizer_state
```
> - `save_checkpoint`：保存模型权重 + 优化器状态 + 元数据
> - `load_model`：加载预训练 base model
> - `load_optimizer_state`：单独加载优化器状态（动量缓冲区等）

```python
from nanochat.loss_eval import evaluate_bpb
```
> 在验证集上评估 bits-per-byte 损失。bpb 比 cross-entropy loss 更好，因为它与词表大小无关，可以跨模型横向比较。

```python
import torch.distributed as dist
```
> PyTorch 分布式通信库，用于多卡之间同步数据（all_reduce 等操作）。

```python
from nanochat.flash_attention import HAS_FA3
```
> 检测当前环境是否支持 Flash Attention 3（需要 Hopper 架构 GPU，即 H100）。

```python
from nanochat.engine import Engine
```
> 带 KV Cache 的高效推理引擎，用于 ChatCORE 评估时生成模型回复。

```python
from scripts.chat_eval import run_chat_eval
```
> 运行 ChatCORE 评估的函数，测试 ARC、MMLU、GSM8K 等多个任务的准确率。

```python
from tasks.common import TaskMixture
from tasks.gsm8k import GSM8K
from tasks.mmlu import MMLU
from tasks.smoltalk import SmolTalk
from tasks.customjson import CustomJSON
from tasks.spellingbee import SimpleSpelling, SpellingBee
```
> 导入各个训练数据集/任务：
> - `TaskMixture`：将多个 task 按顺序拼接成一个大数据集
> - `GSM8K`：8K 条小学数学题，教模型数学推理
> - `MMLU`：多选题，覆盖广泛知识领域
> - `SmolTalk`：HuggingFace 的通用对话数据集（460K 条）
> - `CustomJSON`：从自定义 jsonl 文件加载对话（用于注入身份信息）
> - `SimpleSpelling`/`SpellingBee`：拼写和字母计数任务

---

## 第二部分：CLI 参数定义

```python
parser = argparse.ArgumentParser(description="Supervised fine-tuning (SFT) the model")
```
> 创建参数解析器，描述为 SFT 脚本。

```python
parser.add_argument("--run", type=str, default="dummy",
    help="wandb run name ('dummy' disables wandb logging)")
```
> wandb 实验名称。默认 `"dummy"` 表示不记录 wandb，方便本地调试。

```python
parser.add_argument("--device-type", type=str, default="",
    help="cuda|cpu|mps (empty = autodetect)")
```
> 指定计算设备。空字符串时自动检测。

```python
parser.add_argument("--model-tag", type=str, default=None,
    help="model tag to load from")
parser.add_argument("--model-step", type=int, default=None,
    help="model step to load from")
```
> 指定要加载的预训练 checkpoint：
> - `model-tag`：对应 checkpoint 目录名（如 `d26`）
> - `model-step`：具体的训练步数（None 表示加载最新的）

```python
parser.add_argument("--load-optimizer", type=int, default=1,
    help="warm-start optimizer from pretrained checkpoint (0=no, 1=yes)")
```
> 是否从预训练 checkpoint 加载优化器状态（动量缓冲区）。
> 热启动优化器可以让 SFT 开始时梯度方向更准确，收敛更快。


```python
parser.add_argument("--num-iterations", type=int, default=-1,
    help="number of optimization steps (-1 = full epoch)")
```
> 训练步数上限。`-1` 表示跑完整个数据集一个 epoch 为止（由数据集大小决定）。

```python
parser.add_argument("--max-seq-len", type=int, default=None, ...)
parser.add_argument("--device-batch-size", type=int, default=None, ...)
parser.add_argument("--total-batch-size", type=int, default=None, ...)
```
> batch 相关参数，默认全为 `None`，表示从预训练 checkpoint 的元数据中继承，保持与预训练一致。

```python
parser.add_argument("--embedding-lr", type=float, default=None, ...)
parser.add_argument("--unembedding-lr", type=float, default=None, ...)
parser.add_argument("--matrix-lr", type=float, default=None, ...)
```
> 三类参数的学习率，默认继承自预训练：
> - `embedding-lr`：词嵌入层（Adam 优化）
> - `unembedding-lr`：输出投影层（Adam 优化）
> - `matrix-lr`：Transformer 内部矩阵权重（Muon 优化）

```python
parser.add_argument("--init-lr-frac", type=float, default=0.8, ...)
```
> SFT 开始时学习率是基础 LR 的 0.8 倍。
> 原因：预训练结束时 LR warmdown 到接近 0，SFT 需要从一个较低但非零的 LR 重新开始，避免破坏预训练习得的知识。

```python
parser.add_argument("--warmup-ratio", type=float, default=0.0, ...)
parser.add_argument("--warmdown-ratio", type=float, default=0.5, ...)
parser.add_argument("--final-lr-frac", type=float, default=0.0, ...)
```
> 学习率调度参数：
> - `warmup-ratio=0.0`：SFT 默认不做 warmup（直接从 init_lr_frac 开始）
> - `warmdown-ratio=0.5`：后 50% 的训练做线性衰减
> - `final-lr-frac=0.0`：最终 LR 衰减到 0

```python
parser.add_argument("--eval-every", type=int, default=200, ...)
parser.add_argument("--eval-tokens", type=int, default=40*524288, ...)
```
> 每 200 步评估一次验证集 bpb，评估时消耗约 40×524288 ≈ 2000 万个 token。

```python
parser.add_argument("--chatcore-every", type=int, default=200, ...)
parser.add_argument("--chatcore-max-cat", type=int, default=-1, ...)
parser.add_argument("--chatcore-max-sample", type=int, default=24, ...)
```
> ChatCORE 评估频率和规模：
> - `chatcore-every=200`：每 200 步评估一次
> - `chatcore-max-cat=-1`：多选题任务不限制题目数量
> - `chatcore-max-sample=24`：生成式任务（GSM8K 等）每次最多测 24 道题（控制评估时间）

```python
parser.add_argument("--mmlu-epochs", type=int, default=3, ...)
parser.add_argument("--gsm8k-epochs", type=int, default=4, ...)
```
> 训练数据混合中 MMLU 和 GSM8K 各重复几轮：
> - MMLU × 3：100K × 3 = 300K 条多选题
> - GSM8K × 4：8K × 4 = 32K 条数学题

```python
args = parser.parse_args()
user_config = vars(args).copy()
```
> 解析命令行参数，并将参数字典备份为 `user_config`，后续会写入 checkpoint 元数据和 wandb，方便复现实验。

---

## 第三部分：计算环境初始化

```python
device_type = autodetect_device_type() if args.device_type == "" else args.device_type
```
> 如果未指定设备类型，自动检测（优先 cuda，其次 mps，最后 cpu）。

```python
ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init(device_type)
```
> 初始化分布式训练环境，返回：
> - `ddp`：是否在 DDP 模式下（True/False）
> - `ddp_rank`：当前进程的全局编号（0 ~ world_size-1）
> - `ddp_local_rank`：当前进程在本机的编号（用于绑定 GPU）
> - `ddp_world_size`：总进程数（即 GPU 总数）
> - `device`：当前进程使用的设备（如 `cuda:0`）

```python
master_process = ddp_rank == 0
```
> 标记当前进程是否为主进程（rank 0）。只有主进程负责打印日志、保存 checkpoint 等。

```python
print0(f"COMPUTE_DTYPE: {COMPUTE_DTYPE} ({COMPUTE_DTYPE_REASON})")
```
> 打印当前使用的计算精度，例如 `COMPUTE_DTYPE: bfloat16 (H100 detected)`。

```python
synchronize = torch.cuda.synchronize if device_type == "cuda" else lambda: None
```
> 计时前的 GPU 同步函数。CUDA 操作是异步的，计时必须先同步才准确。CPU/MPS 不需要同步，用空函数代替。

```python
get_max_memory = torch.cuda.max_memory_allocated if device_type == "cuda" else lambda: 0
```
> 获取峰值显存占用函数，用于训练结束后报告显存使用情况。

```python
if device_type == "cuda":
    gpu_device_name = torch.cuda.get_device_name(0)
    gpu_peak_flops = get_peak_flops(gpu_device_name)
    print0(f"GPU: {gpu_device_name} | Peak FLOPS (BF16): {gpu_peak_flops:.2e}")
else:
    gpu_peak_flops = float('inf')
```
> 获取 GPU 型号和理论峰值算力（如 H100 的 BF16 峰值约 989 TFLOPS）。
> `gpu_peak_flops` 用于计算 MFU（模型算力利用率 = 实际算力 / 理论峰值）。
> 非 CUDA 设备设为无穷大，使 MFU 为 0，避免无意义的计算。


```python
use_dummy_wandb = args.run == "dummy" or not master_process
wandb_run = DummyWandb() if use_dummy_wandb else wandb.init(project="nanochat-sft", name=args.run, config=user_config)
```
> 两种情况使用空的 DummyWandb 对象：
> 1. `--run dummy`（用户指定不记录）
> 2. 非主进程（只有 rank 0 记录，避免多卡重复上传）
>
> 否则初始化真实 wandb，记录到项目 `nanochat-sft`。

```python
if not HAS_FA3:
    print0("WARNING: Flash Attention 3 not available, using PyTorch SDPA fallback. Training will be less efficient.")
```
> 检查 Flash Attention 3 可用性。FA3 在 H100 上比 PyTorch 内置的 SDPA（Scaled Dot-Product Attention）快约 2 倍。没有 FA3 时会 fallback 到 SDPA，速度较慢。

---

## 第四部分：加载预训练模型

```python
model, tokenizer, meta = load_model("base", device, phase="train",
    model_tag=args.model_tag, step=args.model_step)
```
> 加载预训练的 base model（即预训练阶段产出的权重）：
> - `"base"`：加载 base model 的 checkpoint（区别于 sft/rl 阶段的）
> - `phase="train"`：将模型设为训练模式（启用 Dropout 等）
> - `meta`：checkpoint 中存储的元数据字典，包含训练超参、模型配置等

---

## 第五部分：从预训练继承超参数

```python
pretrain_user_config = meta.get("user_config", {})
for name, fallback, source in [
    ("max_seq_len",       2048,  meta),
    ("device_batch_size", 32,    meta),
    ("total_batch_size",  524288, meta),
    ("embedding_lr",      0.3,   pretrain_user_config),
    ("unembedding_lr",    0.004, pretrain_user_config),
    ("matrix_lr",         0.02,  pretrain_user_config),
]:
```
> 定义需要从预训练继承的参数列表，每项包含：
> - 参数名
> - 兜底默认值（如果 checkpoint 里也没有的话）
> - 查找来源（`meta` 或 `pretrain_user_config`）

```python
    arg_val = getattr(args, name)
    pretrain_val = source.get(name)
    if arg_val is None:
        resolved = pretrain_val if pretrain_val is not None else fallback
        setattr(args, name, resolved)
        print0(f"Inherited {name}={resolved} from pretrained checkpoint")
    elif pretrain_val is not None and arg_val != pretrain_val:
        print0(f"NOTE: --{name.replace('_', '-')}={arg_val} overrides pretrained value of {pretrain_val}")
    else:
        print0(f"Using {name}={arg_val}")
```
> 三条逻辑：
> 1. 用户没有指定（`None`）→ 从 checkpoint 继承，找不到则用兜底默认值
> 2. 用户指定了但与预训练值不同 → 打印提示说明用户正在覆盖
> 3. 用户指定了且与预训练一致 → 正常使用

---

## 第六部分：编译模型与计算 batch 参数

```python
orig_model = model
model = torch.compile(model, dynamic=False)
```
> - `orig_model`：保留未编译的原始模型，用于推理评估（因为评估时输入 shape 会变化）
> - `torch.compile`：将模型编译为优化的计算图，训练速度约提升 20-40%
> - `dynamic=False`：假设输入形状固定（训练时 batch shape 确实固定），编译更彻底

```python
depth = model.config.n_layer
num_flops_per_token = model.estimate_flops()
```
> - `depth`：模型层数，用于命名 checkpoint 目录
> - `num_flops_per_token`：每个 token 的浮点运算量估算，用于计算 MFU 和总训练算力

```python
tokens_per_fwdbwd = args.device_batch_size * args.max_seq_len
```
> 单个 GPU 每次前向+反向传播处理的 token 数。例如 batch_size=32，seq_len=2048 → 65536 tokens。

```python
world_tokens_per_fwdbwd = tokens_per_fwdbwd * ddp_world_size
```
> 所有 GPU 每步合计处理的 token 数。8 卡时为 65536 × 8 = 524288 tokens。

```python
assert args.total_batch_size % world_tokens_per_fwdbwd == 0
grad_accum_steps = args.total_batch_size // world_tokens_per_fwdbwd
```
> 计算梯度累积步数。若 `total_batch_size=524288`，8 卡时无需累积（步数=1）。
> 若单卡运行，步数=8，即累积 8 次微批次再更新参数，等效于 8 卡的 batch size。

```python
token_bytes = get_token_bytes(device=device)
```
> 预加载每个 token 的字节数张量，形状为 `[vocab_size]`，用于 bpb 计算。


---

## 第七部分：初始化优化器

```python
optimizer = model.setup_optimizer(
    unembedding_lr=args.unembedding_lr,
    embedding_lr=args.embedding_lr,
    matrix_lr=args.matrix_lr,
    weight_decay=0.0
)
```
> 初始化 MuonAdamW 组合优化器：
> - Transformer 矩阵权重（QKV、FFN 等）使用 **Muon** 优化器
> - 嵌入层和输出层使用 **AdamW** 优化器
> - `weight_decay=0.0`：SFT 阶段不做权重衰减（预训练末期 weight_decay 已衰减到 0，SFT 继续保持）

```python
base_dir = get_base_dir()
if args.load_optimizer:
    optimizer_data = load_optimizer_state("base", device, rank=ddp_rank,
        model_tag=args.model_tag, step=args.model_step)
```
> 如果开启热启动优化器，从预训练 checkpoint 加载优化器状态（主要是 Adam 的一阶/二阶动量缓冲区，以及 Muon 的动量）。
> 热启动让 SFT 初期梯度更新方向更准确，有助于更快收敛。

```python
    if optimizer_data is not None:
        base_lrs = [group["lr"] for group in optimizer.param_groups]
        optimizer.load_state_dict(optimizer_data)
        del optimizer_data
        for group, base_lr in zip(optimizer.param_groups, base_lrs):
            group["lr"] = base_lr
```
> 加载优化器状态有一个陷阱：`load_state_dict` 会把预训练的 LR 也一并恢复，而预训练末期 LR warmdown 到接近 0。
> 解决方案：先记录 SFT 应该使用的学习率（`base_lrs`），加载后再强制恢复。

```python
scaler = torch.amp.GradScaler() if COMPUTE_DTYPE == torch.float16 else None
```
> fp16 训练需要 GradScaler 防止梯度下溢（fp16 动态范围小，梯度容易变成 0）。
> bf16 和 fp32 不需要，设为 None。

```python
for group in optimizer.param_groups:
    group["lr"] = group["lr"] * args.init_lr_frac
    group["initial_lr"] = group["lr"]
```
> 将所有参数组的学习率乘以 `init_lr_frac`（默认 0.8），并保存为 `initial_lr`。
> `initial_lr` 是学习率调度器的基准值，后续调度器乘以 multiplier 时以此为基础。

---

## 第八部分：构建训练数据混合

```python
identity_conversations_filepath = os.path.join(base_dir, "identity_conversations.jsonl")
```
> 合成身份对话数据的路径，约 1000 条，内容是让模型了解自己是谁（如"你是 nanochat"）。

```python
train_tasks = [
    SmolTalk(split="train"),                              # 460K 条通用对话
    CustomJSON(filepath=identity_conversations_filepath), # 1000 条身份对话（第1轮）
    CustomJSON(filepath=identity_conversations_filepath), # 1000 条身份对话（第2轮，重复以加强）
    *[MMLU(subset="all", split="auxiliary_train") for _ in range(args.mmlu_epochs)],
                                                          # 100K × 3 = 300K 条多选题
    *[GSM8K(subset="main", split="train") for _ in range(args.gsm8k_epochs)],
                                                          # 8K × 4 = 32K 条数学题
    SimpleSpelling(size=200000, split="train"),            # 200K 条简单拼写题
    SpellingBee(size=80000, split="train"),                # 80K 条字母计数题
]
```
> 训练数据混合策略的设计逻辑：
> - **SmolTalk**：通用对话能力的基础，数据量最大
> - **Identity**：反复 2 次强化模型的自我认知
> - **MMLU × 3**：多选题格式训练，提升选择题能力
> - **GSM8K × 4**：数学推理训练，重复多轮强化
> - **Spelling**：基础语言能力（拼写/字母计数），防止模型退化

```python
train_dataset = TaskMixture(train_tasks)
```
> 将所有 task 拼接成一个顺序数据集，总行数 = 各 task 行数之和。

```python
val_dataset = TaskMixture([
    SmolTalk(split="test"),
    MMLU(subset="all", split="test", stop=5200),
    GSM8K(subset="main", split="test", stop=420),
])
```
> 验证集：从每个训练数据源取对应的 test split，并限制 MMLU/GSM8K 的数量以匹配训练数据比例。

---

## 第九部分：数据加载器（BOS-BestFit Packing）

```python
last_step = False
approx_progress = 0.0
current_epoch = 1
```
> 三个全局变量，由数据生成器内部更新：
> - `last_step`：是否应该停止训练（数据集跑完或达到 num_iterations）
> - `approx_progress`：当前训练进度（0→1），用于 LR 调度
> - `current_epoch`：当前 epoch 数，用于日志

```python
def sft_data_generator_bos_bestfit(split, buffer_size=100):
```
> 定义核心数据加载生成器。`buffer_size=100` 表示预先缓冲 100 条对话，用于 Best-Fit 装箱算法。

```python
    global last_step, approx_progress, current_epoch
```
> 声明修改全局变量，生成器内部会更新这三个状态。

```python
    row_capacity = args.max_seq_len + 1
```
> 每行实际分配 `max_seq_len + 1` 个 token 位置。
> 原因：语言模型训练时，输入是 `tokens[:-1]`，目标是 `tokens[1:]`，需要多 1 个位置。

```python
    bos_token = tokenizer.get_bos_token_id()
```
> 获取 BOS（Beginning Of Sequence）特殊 token 的 ID，用于填充。


```python
    conv_buffer = []
    cursor = ddp_rank
    consumed = ddp_rank
    epoch = 1
    it = 0
```
> - `conv_buffer`：已分词但未装箱的对话缓冲区
> - `cursor`：数据集读取指针，不同 rank 从不同位置开始（rank 0 读第 0、8、16... 条，rank 1 读第 1、9、17... 条），实现数据并行
> - `consumed`：实际已消耗的对话数（用于进度计算，区别于 cursor 的预读位置）
> - `epoch`/`it`：epoch 和 iteration 计数

```python
    def refill_buffer():
        nonlocal cursor, epoch
        while len(conv_buffer) < buffer_size:
            conversation = dataset[cursor]
            ids, mask = tokenizer.render_conversation(conversation)
            conv_buffer.append((ids, mask))
            cursor += ddp_world_size
            if cursor >= dataset_size:
                cursor = cursor % dataset_size
                epoch += 1
```
> 补充对话缓冲区：
> - 取出一条对话，用 `render_conversation` 将其分词并生成 loss mask
> - `loss mask`：1 表示 assistant 回复（参与 loss 计算），0 表示用户输入（不参与）
> - `cursor += ddp_world_size`：每次跳过其他 rank 负责的数据，保证各 rank 数据不重叠

```python
    while True:
        rows = []
        mask_rows = []
        row_lengths = []
        for _ in range(args.device_batch_size):
            row = []
            mask_row = []
            padded = False
            while len(row) < row_capacity:
```
> 外层 while 永远循环（由 `yield` 控制），每次生成一个完整 batch。
> 内层 for 循环构造 batch 中的每一行（共 `device_batch_size` 行）。
> 最内层 while 向当前行填充对话，直到填满 `row_capacity`。

```python
                remaining = row_capacity - len(row)
                best_idx = -1
                best_len = 0
                for i, (conv, _) in enumerate(conv_buffer):
                    conv_len = len(conv)
                    if conv_len <= remaining and conv_len > best_len:
                        best_idx = i
                        best_len = conv_len
```
> **Best-Fit 装箱算法**：
> 在缓冲区中找到能放入当前剩余空间的最长对话。
> 这样可以最大化每行的填充率，减少 padding 浪费。

```python
                if best_idx >= 0:
                    conv, conv_mask = conv_buffer.pop(best_idx)
                    row.extend(conv)
                    mask_row.extend(conv_mask)
                    consumed += ddp_world_size
                else:
                    content_len = len(row)
                    row.extend([bos_token] * remaining)
                    mask_row.extend([0] * remaining)
                    padded = True
                    break
```
> - 找到合适的对话：取出装入当前行
> - 没有合适的：用 BOS token 填充剩余空间，并标记 `padded=True`（padding 位置的 mask=0，不参与 loss）
> - 注意：填充而非截断，保证不丢弃任何 token

```python
        batch_tensor = torch.tensor(rows, dtype=torch.long, pin_memory=use_cuda)
        inputs = batch_tensor[:, :-1].to(device=device, dtype=torch.int32, non_blocking=use_cuda).contiguous()
        targets = batch_tensor[:, 1:].to(device=device, dtype=torch.int64, non_blocking=use_cuda).contiguous()
```
> 构建输入/目标张量（自回归语言模型的标准处理）：
> - `inputs = tokens[:-1]`：去掉最后一个 token 作为输入
> - `targets = tokens[1:]`：去掉第一个 token 作为预测目标
> - `non_blocking=True`：异步传输到 GPU，不阻塞 CPU，提高吞吐
> - `pin_memory=True`（CUDA）：锁页内存，加速 CPU→GPU 传输

```python
        mask_tensor = torch.tensor(mask_rows, dtype=torch.int8)
        mask_targets = mask_tensor[:, 1:].to(device=device)
        targets[mask_targets == 0] = -1
```
> **关键的 Loss Masking**：
> - 将 loss mask 对齐到 targets（同样偏移 1 位）
> - 用户输入部分（mask=0）的 targets 设为 -1
> - PyTorch cross-entropy 的 `ignore_index=-1` 会自动跳过这些位置
> - 效果：模型只学习如何生成 assistant 的回复，不学习如何生成用户输入

```python
        for i, content_len in enumerate(row_lengths):
            if content_len < row_capacity:
                targets[i, content_len-1:] = -1
```
> 对 padding 位置同样设 targets=-1，确保 padding 不参与 loss 计算。

```python
        yield inputs, targets
```
> 生成器 yield 当前 batch，暂停执行直到外部调用 `next()`。

---

## 第十部分：学习率和动量调度器

```python
def get_lr_multiplier(progress):
    if progress < args.warmup_ratio:
        return (progress + 1e-8) / args.warmup_ratio
    elif progress <= 1.0 - args.warmdown_ratio:
        return 1.0
    else:
        decay = (progress - (1.0 - args.warmdown_ratio)) / args.warmdown_ratio
        return (1 - decay) * 1.0 + decay * args.final_lr_frac
```
> 三段式学习率调度（基于 progress 0→1，而不是绝对步数）：
> ```
> LR
> ↑
> 1.0|    _______________
>    |   /               \
> 0.0|__/                 \____
>    0  warmup  constant  warmdown  1
> ```
> - **warmup 阶段**（0 ~ warmup_ratio）：线性从 0 升到 1
> - **constant 阶段**：保持 1.0
> - **warmdown 阶段**（后 warmdown_ratio）：线性降到 final_lr_frac
> - `+1e-8`：防止 warmup_ratio=0 时除零

```python
def get_muon_momentum(it):
    frac = min(it / 300, 1)
    momentum = (1 - frac) * 0.85 + frac * 0.95
    return momentum
```
> Muon 优化器的动量热身：
> - 前 300 步从 0.85 线性增长到 0.95
> - 原因：训练初期动量过大会导致不稳定，逐渐增大更安全


---

## 第十一部分：训练主循环

```python
x, y = next(train_loader)
```
> 预取第一个 batch。GPU 在处理当前 batch 的前向/反向时，CPU 可以同时准备下一个 batch，形成流水线。

```python
min_val_bpb = float("inf")
smooth_train_loss = 0
ema_beta = 0.9
total_training_time = 0
step = 0
```
> 训练循环状态变量：
> - `min_val_bpb`：记录历史最优验证 bpb（用于日志）
> - `smooth_train_loss`：训练 loss 的 EMA（指数移动平均），减少噪声
> - `ema_beta=0.9`：EMA 衰减系数，越大越平滑
> - `total_training_time`：累计训练时间（排除前 10 步的启动开销）

```python
while True:
    flops_so_far = num_flops_per_token * args.total_batch_size * step
```
> 估算到当前步骤为止总共使用的浮点运算量（FLOPs），用于 wandb 记录和横向比较不同模型的训练效率。

```python
    if ddp:
        last_step_tensor = torch.tensor(last_step, dtype=torch.int32, device=device)
        dist.all_reduce(last_step_tensor, op=dist.ReduceOp.MAX)
        last_step = bool(last_step_tensor.item())
```
> **分布式同步停止信号**：
> 不同 rank 可能在不同时刻触发 `last_step=True`（因为各 rank 消耗的数据量不同）。
> 用 `all_reduce MAX` 操作：任意一个 rank 触发停止，所有 rank 都停止，避免死锁。

### 验证集评估

```python
    if last_step or (args.eval_every > 0 and step % args.eval_every == 0):
        model.eval()
        val_loader = build_val_loader()
        eval_steps = args.eval_tokens // (args.device_batch_size * args.max_seq_len * ddp_world_size)
        val_bpb = evaluate_bpb(model, val_loader, eval_steps, token_bytes)
        print0(f"Step {step:05d} | Validation bpb: {val_bpb:.4f}")
        if val_bpb < min_val_bpb:
            min_val_bpb = val_bpb
        wandb_run.log({...})
        model.train()
```
> 每 `eval_every` 步（默认 200 步）或最后一步时：
> 1. 切换模型到评估模式（关闭 Dropout 等）
> 2. 创建新的验证集加载器（保证每次评估从头开始）
> 3. 计算验证集 bpb（bits-per-byte）
> 4. 记录到 wandb
> 5. 切换回训练模式

### ChatCORE 评估

```python
    chatcore_results = {}
    if args.chatcore_every > 0 and (last_step or (step > 0 and step % args.chatcore_every == 0)):
        model.eval()
        engine = Engine(orig_model, tokenizer)
```
> ChatCORE 评估使用**未编译的原始模型**（`orig_model`），因为评估时输入形状会变化（不同问题长度不同），`dynamic=False` 编译的模型无法处理。

```python
        all_tasks = ['ARC-Easy', 'ARC-Challenge', 'MMLU', 'GSM8K', 'HumanEval', 'SpellingBee']
        categorical_tasks = {'ARC-Easy', 'ARC-Challenge', 'MMLU'}
        baseline_accuracies = {
            'ARC-Easy': 0.25, 'ARC-Challenge': 0.25, 'MMLU': 0.25,
            'GSM8K': 0.0, 'HumanEval': 0.0, 'SpellingBee': 0.0,
        }
```
> - `categorical_tasks`：多选题任务（4 选 1），随机基线准确率 = 25%
> - `baseline_accuracies`：各任务的随机猜测基线，用于"中心化"准确率

```python
        def centered_mean(tasks):
            return sum((task_results[t] - baseline_accuracies[t]) /
                       (1.0 - baseline_accuracies[t]) for t in tasks) / len(tasks)
        chatcore = centered_mean(all_tasks)
```
> **ChatCORE 计算公式**：
> ```
> centered_acc = (acc - baseline) / (1.0 - baseline)
> ```
> 中心化后：0 = 随机水平，1 = 满分。
> ChatCORE 是所有任务中心化准确率的均值，范围 [0, 1]。

### 保存 Checkpoint

```python
    if last_step:
        output_dirname = args.model_tag if args.model_tag else f"d{depth}"
        checkpoint_dir = os.path.join(base_dir, "chatsft_checkpoints", output_dirname)
        save_checkpoint(
            checkpoint_dir, step,
            orig_model.state_dict(),    # 模型权重
            optimizer.state_dict(),     # 优化器状态
            {
                "step": step,
                "val_bpb": val_bpb,
                "model_config": {...},  # 模型结构参数
                "user_config": user_config,  # 训练超参
            },
            rank=ddp_rank,
        )
```
> 仅在训练结束时保存 checkpoint（`last_step=True`）。
> - 保存路径格式：`{base_dir}/chatsft_checkpoints/d26/step_XXXXX.pt`
> - 所有 rank 都参与保存（因为分布式训练时每个 rank 有独立的优化器分片）
> - `orig_model.state_dict()`：使用未编译模型的权重（与编译版本等价，但可移植）

---

## 第十二部分：单步训练

```python
    synchronize()
    t0 = time.time()
    for micro_step in range(grad_accum_steps):
        loss = model(x, y)
```
> `synchronize()` 确保 GPU 操作完成后再开始计时。
> 梯度累积循环：每次用一个微批次前向传播计算 loss。

```python
        train_loss = loss.detach()
        loss = loss / grad_accum_steps
```
> - `.detach()`：从计算图中分离出 loss 值用于日志，不影响反向传播
> - `/ grad_accum_steps`：梯度累积时每次 backward 会累加梯度，需要除以步数归一化，等效于对整个 total_batch_size 求平均

```python
        if scaler is not None:
            scaler.scale(loss).backward()
        else:
            loss.backward()
```
> - fp16 模式：用 GradScaler 缩放 loss 后再 backward，防止梯度下溢
> - bf16/fp32 模式：直接 backward

```python
        x, y = next(train_loader)
        progress = max(progress, approx_progress)
```
> **关键的流水线优化**：在 GPU 计算 backward 的同时，CPU 已经开始准备下一个 batch。
> `max` 确保 progress 单调递增（避免数据生成器的缓冲造成 progress 短暂回退）。


```python
    lrm = get_lr_multiplier(progress)
    muon_momentum = get_muon_momentum(step)
    for group in optimizer.param_groups:
        group["lr"] = group["initial_lr"] * lrm
        if group['kind'] == 'muon':
            group["momentum"] = muon_momentum
```
> 每步动态更新学习率和 Muon 动量：
> - 用 `initial_lr × multiplier` 计算实际 LR，实现调度
> - 只对 `kind='muon'` 的参数组更新动量（AdamW 组不需要）

```python
    if scaler is not None:
        scaler.unscale_(optimizer)
        if is_ddp_initialized():
            for v in scaler._found_inf_per_device(optimizer).values():
                dist.all_reduce(v, op=dist.ReduceOp.MAX)
        scaler.step(optimizer)
        scaler.update()
    else:
        optimizer.step()
```
> fp16 模式的优化器步骤：
> 1. `unscale_`：将梯度从缩放空间恢复到真实空间
> 2. 分布式同步 inf/nan 检测（任意 rank 发现无效梯度，所有 rank 跳过此步）
> 3. `scaler.step`：更新参数（若梯度含 inf/nan 则自动跳过）
> 4. `scaler.update`：自动调整缩放因子（连续正常则增大，遇到 inf 则缩小）
>
> bf16/fp32 模式直接 `optimizer.step()`。

```python
    model.zero_grad(set_to_none=True)
```
> 清空梯度，`set_to_none=True` 比 `fill_(0)` 更高效，直接释放梯度张量的内存。

```python
    synchronize()
    t1 = time.time()
    dt = t1 - t0
```
> 再次同步确保 GPU 完成所有计算后记录结束时间，计算本步耗时 `dt`。

---

## 第十三部分：日志记录

```python
    step += 1
    smooth_train_loss = ema_beta * smooth_train_loss + (1 - ema_beta) * train_loss.item()
    debiased_smooth_loss = smooth_train_loss / (1 - ema_beta**(step + 1))
```
> EMA（指数移动平均）平滑训练 loss：
> - EMA 公式：`S_t = β·S_{t-1} + (1-β)·x_t`
> - 早期步骤 EMA 偏小（因为初始值为 0），用偏差修正：`S_t / (1 - β^t)`
> - 效果：loss 曲线更平滑，易于观察趋势

```python
    tok_per_sec = int(args.total_batch_size / dt)
    flops_per_sec = num_flops_per_token * args.total_batch_size / dt
    mfu = 100 * flops_per_sec / (gpu_peak_flops * ddp_world_size)
```
> 计算训练效率指标：
> - `tok_per_sec`：每秒处理的 token 数，直观反映吞吐
> - `mfu`（Model FLOPS Utilization）：实际算力 / 理论峰值算力 × 100%
>   - H100 上好的 MFU 约 40-60%，说明硬件被有效利用
>   - MFU 低说明存在瓶颈（显存带宽、数据加载、通信开销等）

```python
    if step > 10:
        total_training_time += dt
```
> 跳过前 10 步（JIT 编译、数据预热等），确保计时更准确。

```python
    print0(f"step {step:05d} ({pct_done:.2f}%) | loss: {debiased_smooth_loss:.6f} | ...")
```
> 每步打印训练状态，只在 rank 0 打印（避免 8 卡重复输出 8 次）。

```python
    if step % 10 == 0:
        wandb_run.log({...})
```
> 每 10 步上传一次 wandb（不是每步都传，减少网络开销）。

---

## 第十四部分：垃圾回收管理

```python
    if step == 1:
        gc.collect()
        gc.freeze()
        gc.disable()
    elif step % 5000 == 0:
        gc.collect()
```
> Python GC 在训练过程中会不定期触发，每次耗时约 500ms，严重影响训练节奏。
> 三步手动管理策略：
> 1. **第 1 步后 `gc.collect()`**：手动清理模型加载期间产生的大量临时对象
> 2. **`gc.freeze()`**：将当前存活对象标记为"冻结"，GC 不再扫描它们（它们会一直存活到训练结束）
> 3. **`gc.disable()`**：完全关闭自动 GC，只在极少情况下手动触发
> 4. **每 5000 步手动 collect()**：防止极长训练中的内存泄漏

---

## 第十五部分：训练结束

```python
print0(f"Peak memory usage: {get_max_memory() / 1024 / 1024:.2f}MiB")
print0(f"Total training time: {total_training_time/60:.2f}m")
print0(f"Minimum validation bpb: {min_val_bpb:.4f}")
```
> 训练结束后汇报：峰值显存、总训练时间、最优验证 bpb。

```python
from nanochat.report import get_report
get_report().log(section="SFT", data=[
    user_config,
    {"Number of iterations": step, "DDP world size": ddp_world_size},
    {"Minimum validation bpb": min_val_bpb},
])
```
> 将训练摘要写入 nanochat Report（一个结构化的训练报告文件），记录配置和结果。

```python
wandb_run.finish()
compute_cleanup()
```
> - `wandb_run.finish()`：正常结束 wandb run，上传所有未同步的数据
> - `compute_cleanup()`：销毁 DDP 进程组，释放分布式通信资源

---

## 整体数据流总结

```
预训练 Checkpoint
        ↓ load_model("base")
      model + tokenizer + meta
        ↓ 继承超参数
      batch_size / seq_len / lr
        ↓ torch.compile
      编译优化的模型
        ↓
训练数据混合 (SmolTalk + MMLU + GSM8K + Spelling)
        ↓ BOS-BestFit Packing
      inputs [B, T]  targets [B, T]（-1 mask 掉用户输入）
        ↓ 前向传播
      loss（只对 assistant 回复计算）
        ↓ 反向传播 + 梯度累积
      梯度
        ↓ Muon/AdamW 更新参数
      更新后的模型
        ↓ 每 200 步评估
      val_bpb + ChatCORE
        ↓ 训练结束
      SFT Checkpoint → chat_web.py / chat_rl.py
```
