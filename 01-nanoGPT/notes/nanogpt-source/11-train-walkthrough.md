# train.py 逐行深度解析

> nanoGPT 的训练引擎：**~300 行实现完整 GPT 训练流程**
>
> 源码：[github.com/karpathy/nanoGPT/blob/master/train.py](https://github.com/karpathy/nanoGPT/blob/master/train.py)
>
> 配合阅读：[02-model.md](./02-model.md)（模型结构）/ [03-training.md](./03-training.md)（快速概览）

---

## 文件总览

```
train.py 的执行顺序：

 ① 超参数配置     — 定义所有可调旋钮
 ② configurator  — 命令行/配置文件覆盖
 ③ DDP 初始化    — 多卡分布式训练环境
 ④ 随机种子       — 可复现性
 ⑤ 数据加载       — np.memmap 极简方案
 ⑥ 模型初始化    — scratch / resume / gpt2 三种
 ⑦ torch.compile — 编译加速
 ⑧ DDP 包装      — 多卡梯度同步
 ⑨ 优化器配置    — AdamW + 参数分组
 ⑩ 混合精度准备  — autocast + GradScaler
 ⑪ 学习率调度    — warmup + cosine decay
 ⑫ 评估函数      — estimate_loss()
 ⑬ 训练循环      — 梯度累积 + 更新 + checkpoint
```



---

## 一、超参数配置

### 源码

```python
# ─── I/O ───────────────────────────────────────────────────────
out_dir   = 'out'
eval_interval   = 2000    # 每多少步评估一次
log_interval    = 1       # 每多少步打印一次 loss
eval_iters      = 200     # 评估时采样多少个 batch 取平均
eval_only       = False   # 只评估不训练

always_save_checkpoint = True   # 即使 val loss 没降也保存

init_from = 'scratch'   # 'scratch' / 'resume' / 'gpt2*'

# ─── 数据 ──────────────────────────────────────────────────────
dataset    = 'openwebtext'
gradient_accumulation_steps = 5 * 8   # 模拟大 batch，节省显存
batch_size = 12
block_size = 1024

# ─── 模型 ──────────────────────────────────────────────────────
n_layer  = 12
n_head   = 12
n_embd   = 768
dropout  = 0.0
bias     = False

# ─── AdamW 优化器 ──────────────────────────────────────────────
learning_rate = 6e-4
max_iters     = 600000
weight_decay  = 1e-1
beta1, beta2  = 0.9, 0.95
grad_clip     = 1.0

# ─── 学习率衰减 ────────────────────────────────────────────────
decay_lr      = True
warmup_iters  = 2000
lr_decay_iters = 600000
min_lr        = 6e-5

# ─── DDP ───────────────────────────────────────────────────────
backend = 'nccl'

# ─── 系统 ──────────────────────────────────────────────────────
device   = 'cuda'
dtype    = 'bfloat16'
compile  = True
```

### 参数详解

#### 有效 batch 大小的计算

```
batch_size                  = 12      样本数/步（每张卡）
block_size                  = 1024    每个样本的 token 数
gradient_accumulation_steps = 40      累积多少步再更新一次

每步有效 tokens = 12 × 1024 × 40 = 491,520 ≈ 0.5M tokens/step
600,000 步总共训练 tokens = 0.5M × 600K = 300B tokens

GPT-3 论文建议：对于 125M 模型，训练 ~300B tokens 是最优点
nanoGPT 的默认配置正好对齐了这个 Chinchilla 最优点
```

#### beta1=0.9, beta2=0.95 vs Adam 默认 (0.9, 0.999)

```
Adam 默认 beta2=0.999：对二阶矩的历史加权非常长（记住约1000步前的梯度）
GPT 训练 beta2=0.95：   历史更短（约20步），对梯度变化响应更快

为什么 LLM 训练用更小的 beta2？
  LLM 训练数据分布广、梯度变化剧烈
  更小的 beta2 = 对最近梯度更敏感 = 更快适应新数据
  → Karpathy 实验发现 0.95 比 0.999 效果更好
```

#### min_lr = learning_rate / 10

```
min_lr = 6e-5 = 6e-4 / 10

这是 Chinchilla 论文中的经验法则：
  最小学习率 ≈ 最大学习率 / 10
  允许模型在训练后期仍在缓慢优化，而不是完全停止学习
```



---

## 二、configurator.py — 配置覆盖机制

```python
# train.py 开头的这两行（实际用 exec 实现）：
config_keys = [k for k,v in globals().items()
               if not k.startswith('_') and isinstance(v, (int, float, bool, str))]
exec(open('configurator.py').read())
config = {k: globals()[k] for k in config_keys}
```

### 工作原理

```
configurator.py 做的事情：
  1. 读取命令行参数（sys.argv）
  2. 如果参数是一个 .py 文件路径 → exec() 执行它（覆盖全局变量）
  3. 如果参数是 --key=value 格式 → 直接修改对应全局变量

等价效果：
  python train.py config/train_gpt2.py --batch_size=32

  → 先执行 config/train_gpt2.py 里的赋值
  → 再把 batch_size 从文件里的值覆盖为 32

优先级：命令行 --key=value > 配置文件 > 代码默认值
```

### 为什么不用 argparse / hydra？

```
argparse：
  - 每新增一个参数都要注册，繁琐
  - 不支持配置文件继承

hydra：
  - 强大但复杂，有学习成本

configurator.py 的方式（~30行）：
  - 配置文件就是 Python 文件，可以任意逻辑
  - 命令行覆盖任意变量，无需提前声明
  - 读代码就知道有哪些参数（都在 train.py 顶部）
  - 极简，透明，没有魔法

Karpathy 的设计哲学：够用就好，不引入复杂性
```



---

## 三、DDP 初始化 — 分布式数据并行

### 源码

```python
ddp = int(os.environ.get('RANK', -1)) != -1
if ddp:
    init_process_group(backend=backend)      # 'nccl'（GPU间通信）
    ddp_rank       = int(os.environ['RANK'])           # 全局进程编号
    ddp_local_rank = int(os.environ['LOCAL_RANK'])     # 本机卡号
    ddp_world_size = int(os.environ['WORLD_SIZE'])     # 总进程数
    device = f'cuda:{ddp_local_rank}'
    torch.cuda.set_device(device)
    master_process = (ddp_rank == 0)         # 只有 rank 0 做日志/保存
    seed_offset = ddp_rank                   # 每卡不同随机种子
    assert gradient_accumulation_steps % ddp_world_size == 0
    gradient_accumulation_steps //= ddp_world_size  # 均分给每张卡
else:
    master_process = True
    seed_offset = 0
    gradient_accumulation_steps = gradient_accumulation_steps
```

### DDP 是什么？

```
DDP = DistributedDataParallel（分布式数据并行）

单卡训练：
  一张 GPU → 跑一份模型 → 一个 batch → 更新参数

DDP 多卡训练（以 8 卡为例）：
  8 张 GPU → 每张卡都有完整的模型副本
            → 每张卡处理不同的 mini-batch
            → 反向传播后，所有卡的梯度自动求平均
            → 每张卡用同样的平均梯度更新参数
  
  结果：8 卡 = 有效 batch_size × 8，速度约快 8 倍（线性扩展）
```

### RANK / LOCAL_RANK / WORLD_SIZE 的区别

```
假设：2 台服务器，每台 4 张 GPU，共 8 个进程

  WORLD_SIZE = 8       （总进程数）

  服务器 0：
    进程0: RANK=0, LOCAL_RANK=0  （第0台机器第0张卡）
    进程1: RANK=1, LOCAL_RANK=1
    进程2: RANK=2, LOCAL_RANK=2
    进程3: RANK=3, LOCAL_RANK=3

  服务器 1：
    进程4: RANK=4, LOCAL_RANK=0  （第1台机器第0张卡）
    进程5: RANK=5, LOCAL_RANK=1
    进程6: RANK=6, LOCAL_RANK=2
    进程7: RANK=7, LOCAL_RANK=3

LOCAL_RANK 用于选 GPU：
  device = f'cuda:{ddp_local_rank}'
  → 保证每台机器上的进程用不同的 GPU，不冲突
```

### 启动命令

```bash
# 单机 8 卡
torchrun --nproc_per_node=8 train.py config/train_gpt2.py

# 2 机器各 8 卡（需要配置 master 节点地址）
torchrun --nproc_per_node=8 --nnodes=2 \
         --node_rank=0 --master_addr=10.0.0.1 train.py

# 单卡（不走 DDP）
python train.py config/train_shakespeare_char.py
```

### gradient_accumulation_steps 为什么要除以 world_size？

```
设计目标：无论用几张卡，有效 batch 大小保持不变

单卡时：
  gradient_accumulation_steps = 40
  effective_batch = 12 × 40 = 480 样本/步

8 卡 DDP 时：
  每张卡自己跑 gradient_accumulation_steps = 40/8 = 5 步
  每步处理 12 个样本
  8 张卡同时 → 每步有效样本 = 12 × 5 × 8 = 480 ✓

  如果不除：
  每张卡跑 40 步，8 张卡同时 → 有效样本 = 12 × 40 × 8 = 3840 ✗
  等于把 batch 放大了 8 倍，需要调整学习率
```



---

## 四、数据加载 — get_batch()

### 源码

```python
data_dir = os.path.join('data', dataset)

def get_batch(split):
    # np.memmap：不把整个文件加载到 RAM，按需读取
    if split == 'train':
        data = np.memmap(os.path.join(data_dir, 'train.bin'), dtype=np.uint16, mode='r')
    else:
        data = np.memmap(os.path.join(data_dir, 'val.bin'),   dtype=np.uint16, mode='r')

    ix = torch.randint(len(data) - block_size, (batch_size,))   # 随机起始位置
    x = torch.stack([torch.from_numpy((data[i  :i+block_size  ]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i+1:i+1+block_size]).astype(np.int64)) for i in ix])

    if device_type == 'cuda':
        x = x.pin_memory().to(device, non_blocking=True)
        y = y.pin_memory().to(device, non_blocking=True)
    else:
        x, y = x.to(device), y.to(device)
    return x, y
```

### np.memmap 是什么？

```
普通 np.load：
  把整个文件读入 RAM
  train.bin ≈ 17GB → 需要 17GB RAM → 绝大多数机器装不下

np.memmap（内存映射文件）：
  不读入 RAM
  只建立"虚拟地址 → 文件位置"的映射
  访问某个位置时，OS 按需从磁盘加载到内存（透明的）
  只需要 block_size × batch_size 的工作集内存

类比：
  np.load    = 把整本书复印一遍放桌子上
  np.memmap  = 把书放书架上，想看哪页翻哪页
```

### next-token prediction 的数据格式

```python
# 假设 data = [t0, t1, t2, t3, t4, t5, t6, ...]，block_size=4，某次 ix[0]=1
x[0] = data[1:5] = [t1, t2, t3, t4]   ← 输入序列
y[0] = data[2:6] = [t2, t3, t4, t5]   ← 目标序列（右移一位）
```

```
x: [t1,  t2,  t3,  t4 ]
y: [t2,  t3,  t4,  t5 ]
    ↑    ↑    ↑    ↑
    给定 t1，预测 t2
         给定 t1,t2，预测 t3
              ...
              给定 t1~t4，预测 t5

每个样本包含 block_size=1024 个"预测任务"
每个 batch 的 token 总数 = 12 × 1024 = 12,288
```

### pin_memory + non_blocking 加速

```
CPU → GPU 数据传输（PCIe 总线）是训练瓶颈之一

普通方式：
  x.to('cuda')
  → CPU RAM → PCIe → GPU VRAM（同步，等待完成才继续）

加速方式：
  x.pin_memory()          → 把 CPU tensor 放到"页锁定内存"（不被换出到磁盘）
  .to(device, non_blocking=True)
                          → 异步传输（不等待完成，CPU 继续做别的事）

效果：
  在 GPU 跑上一步 forward/backward 的同时
  CPU 异步准备下一步的数据并传到 GPU
  → IO 和计算重叠 → 消除等待时间
```

### 为什么不用 DataLoader？

```
PyTorch DataLoader 的功能：
  - 多进程并行加载
  - 自动 shuffle
  - 自动批处理
  - pin_memory

get_batch() 为什么不用？
  - 数据就是一个巨大的 flat token 数组
  - 没有样本边界（不是一条一条的）
  - 随机取起始点 = 天然 shuffle
  - np.memmap 已经很快了

DataLoader 的开销（进程管理、数据序列化）
在这个场景下反而是负担

结论：简单场景用简单方法，不要过度工程化
```



---

## 五、模型初始化 — 三种模式

### 源码

```python
model_args = dict(n_layer=n_layer, n_head=n_head, n_embd=n_embd,
                  block_size=block_size, bias=bias, vocab_size=None,
                  dropout=dropout)

if init_from == 'scratch':
    # ─── 从零开始 ─────────────────────────────────────────────
    print("Initializing a new model from scratch")
    model_args['vocab_size'] = meta_vocab_size if meta_vocab_size is not None else 50304
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)

elif init_from == 'resume':
    # ─── 从 checkpoint 恢复 ───────────────────────────────────
    print(f"Resuming training from {out_dir}")
    ckpt_path = os.path.join(out_dir, 'ckpt.pt')
    checkpoint = torch.load(ckpt_path, map_location=device)
    checkpoint_model_args = checkpoint['model_args']
    # 强制恢复关键参数（确保模型结构和 ckpt 一致）
    for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size']:
        model_args[k] = checkpoint_model_args[k]
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)
    state_dict = checkpoint['model']
    # 修复 torch.compile 保存的 key 前缀问题
    unwanted_prefix = '_orig_mod.'
    for k, v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
    iter_num      = checkpoint['iter_num']
    best_val_loss = checkpoint['best_val_loss']

elif init_from.startswith('gpt2'):
    # ─── 加载 OpenAI 预训练权重 ───────────────────────────────
    print(f"Initializing from OpenAI GPT-2 weights: {init_from}")
    override_args = dict(dropout=dropout)
    model = GPT.from_pretrained(init_from, override_args)
    for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size']:
        model_args[k] = getattr(model.config, k)
```

### 三种模式详解

```
init_from = 'scratch'
  ├── 用途：全量预训练（从零开始）
  ├── 初始权重：随机（N(0, 0.02)）
  ├── vocab_size：尝试从 data/meta.pkl 读，否则 50304
  └── 典型场景：训练 OpenWebText，需要 A100×8 + 几天

init_from = 'resume'
  ├── 用途：断点续训（中途崩了/想继续训）
  ├── 恢复：模型权重 + 优化器状态 + iter_num + best_val_loss
  ├── 关键：模型结构必须和 ckpt 完全一致（强制复制 6 个关键参数）
  └── 典型场景：长时间训练任务保护

init_from = 'gpt2' / 'gpt2-medium' / ...
  ├── 用途：微调 GPT-2 预训练模型
  ├── 权重来源：从 HuggingFace 下载 OpenAI 官方权重
  ├── 只覆盖 dropout（其他结构参数固定）
  └── 典型场景：finetune_shakespeare.py，只需 20 步
```

### '_orig_mod.' 前缀问题

```
问题：用 torch.compile() 编译后的模型保存 checkpoint 时
      state_dict 的 key 会多一个 '_orig_mod.' 前缀

  编译前：'transformer.wte.weight'
  编译后：'_orig_mod.transformer.wte.weight'

解决：resume 时去掉这个前缀，才能正确 load_state_dict

  for k, v in list(state_dict.items()):
      if k.startswith('_orig_mod.'):
          state_dict[k[10:]] = state_dict.pop(k)
```



---

## 六、torch.compile() — 编译加速

### 源码

```python
if compile:
    print("compiling the model... (takes a ~minute)")
    uncompiled_model = model
    model = torch.compile(model)   # PyTorch 2.0+
```

### 原理

```
普通 PyTorch（Eager Mode）：
  每个 Python 操作（加法、矩阵乘等）逐条执行
  每次都要调 Python → C++ → CUDA kernel
  Python 解释器的调度开销巨大

torch.compile()：
  ① 追踪模型的计算图（TorchDynamo：用字节码拦截）
  ② 优化计算图（算子融合、消除冗余、内存规划）
  ③ 生成高效的 CUDA 代码（Triton kernel）
  ④ 后续调用直接运行优化后的代码

效果：
  A100 上 GPT-2 训练速度提升约 20-30%
  首次调用需要 ~1 分钟编译（之后就快了）
```

### 算子融合示例

```
未融合：
  y = x * 2          → 启动 CUDA kernel 1（乘法），写 GPU VRAM
  z = y + 3          → 启动 CUDA kernel 2（加法），读 GPU VRAM，再写
  w = torch.relu(z)  → 启动 CUDA kernel 3（relu），读 VRAM，再写

  3次 kernel 启动开销 + 3次 VRAM 读写

融合后（compile 自动完成）：
  一个 kernel: w = relu(x * 2 + 3)
  1次 kernel 启动 + 1次 VRAM 写

  VRAM 带宽往往是瓶颈，减少 VRAM 访问 = 大幅提速
```

---

## 七、DDP 包装

### 源码

```python
model.to(device)

if ddp:
    model = DDP(model, device_ids=[ddp_local_rank])

# 获取原始模型（用于 configure_optimizers 等需要访问模型属性的地方）
raw_model = model.module if ddp else model
```

### DDP 的工作机制

```
DDP 包装后，反向传播时自动做：
  AllReduce（所有卡的梯度求平均）
  → 每张卡收到平均后的梯度
  → 用同样的梯度做 optimizer.step()
  → 所有卡的模型参数始终保持同步

时序示意（4 卡）：
  card0: forward → backward → [等待AllReduce] → 收到平均梯度 → step()
  card1: forward → backward → [等待AllReduce] → 收到平均梯度 → step()
  card2: forward → backward → [等待AllReduce] → 收到平均梯度 → step()
  card3: forward → backward → [等待AllReduce] → 收到平均梯度 → step()
                               ↑
                        这里是唯一的通信点
```

### model.module 是什么？

```
DDP 把原模型包了一层：
  model = DDP(raw_model, ...)
  model.forward()       → DDP 的 forward（内部调 raw_model.forward）
  model.module          → raw_model（原始 GPT 对象）

为什么需要 raw_model？
  configure_optimizers() 是 GPT 类的方法
  model.configure_optimizers()  → 报错（DDP 没有这个方法）
  raw_model.configure_optimizers()  → 正常工作

所以：
  raw_model = model.module if ddp else model
  optimizer = raw_model.configure_optimizers(...)
```



---

## 八、优化器配置 — configure_optimizers()

### 源码（在 model.py 的 GPT 类中）

```python
def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
    # ① 找出所有需要梯度的参数
    param_dict = {pn: p for pn, p in self.named_parameters() if p.requires_grad}

    # ② 分两组：有 weight decay 和没有 weight decay
    decay_params   = [p for n, p in param_dict.items() if p.dim() >= 2]
    nodecay_params = [p for n, p in param_dict.items() if p.dim() <  2]

    optim_groups = [
        {'params': decay_params,   'weight_decay': weight_decay},
        {'params': nodecay_params, 'weight_decay': 0.0},
    ]

    # ③ 尝试用 fused AdamW（PyTorch 2.0+，更快）
    use_fused = (device_type == 'cuda') and \
                ('fused' in inspect.signature(torch.optim.AdamW).parameters)
    extra_args = dict(fused=True) if use_fused else dict()

    optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
    return optimizer
```

### weight decay 分组规则详解

```
decay_params（p.dim() >= 2）：
  所有矩阵类参数（Linear.weight, Embedding.weight）
  shape 举例：(768, 768)、(50304, 768)
  → 加 weight_decay = 0.1

nodecay_params（p.dim() < 2）：
  bias 向量：(768,) or scalar
  LayerNorm 的 weight/bias：(768,)
  → weight_decay = 0.0
```

#### 为什么只对矩阵加 weight decay？

```
Weight Decay 的作用：让参数倾向于小值，防止过拟合
  等价于 L2 正则化：loss += λ × Σw²

对 bias 不加 decay 的原因：
  bias 是每个神经元的"基准偏移"，绝对值大小无所谓
  强制缩小 bias 会损害模型表达能力，无正则化价值

对 LayerNorm 参数不加 decay 的原因：
  LayerNorm 的 weight(γ) 是缩放因子，bias(β) 是偏移
  它们只有 n_embd 个参数（相对少），正则化收益小
  强制缩小 γ 会削弱归一化后的表示能力

对 Embedding 的特殊性：
  理论上 Embedding 也是矩阵，应该 decay
  但由于 weight tying（wte 和 lm_head 共享），它已经在 decay_params 里了
  不会被双重处理
```

#### 参数量统计（GPT-2 124M）

```python
# nanoGPT 会打印：
# num decayed parameter tensors: 50, with 124,318,464 parameters
# num non-decayed parameter tensors: 25, with 294,912 parameters
# using fused AdamW: True

decay:   ~124M 参数（主要是 Linear 权重，12 层 × 4 个矩阵）
nodecay: ~0.3M 参数（bias + LayerNorm）
```

### fused AdamW 是什么？

```
普通 AdamW（for 循环实现）：
  for each parameter p:
      compute gradient estimate
      update m (first moment)
      update v (second moment)
      p -= lr * m / (sqrt(v) + eps) + wd * p

fused AdamW（CUDA kernel 实现）：
  所有参数的更新用一个 CUDA kernel 完成
  → 大量减少 kernel 启动开销
  → 约快 15-20%

requires PyTorch 2.0+ 和 CUDA
inspect.signature 检查函数签名是否有 'fused' 参数
（优雅地向前兼容旧版 PyTorch）
```



---

## 九、学习率调度 — get_lr()

### 源码

```python
def get_lr(it):
    # 阶段 1：线性 Warmup
    if it < warmup_iters:
        return learning_rate * it / warmup_iters

    # 阶段 3：保持最低学习率
    if it > lr_decay_iters:
        return min_lr

    # 阶段 2：余弦衰减
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))  # 从 1.0 衰减到 0.0
    return min_lr + coeff * (learning_rate - min_lr)
```

### 三个阶段可视化

```
learning_rate = 6e-4
min_lr        = 6e-5
warmup_iters  = 2,000
lr_decay_iters= 600,000

lr ↑
6e-4 │      *
     │    *   *
     │  *       *
     │*           *
     │              *
     │                 *
     │                     *
     │                           *
     │                                  *
6e-5 │                                         *────────
     └─┬───┬────────────────────────────────────────────→ iter
       0  2000                               600000
       │← warmup →│←─────── cosine decay ─────────→│

阶段1（0~2000）：  线性上升，避免初始梯度爆炸
阶段2（2000~600K）：余弦衰减，平滑降低学习率
阶段3（600K~）：   保持 min_lr = 6e-5，避免学习率为0
```

### coeff 的计算

```python
decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
# decay_ratio: 从 0.0（刚过warmup）到 1.0（到达lr_decay_iters）

coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
# 当 decay_ratio=0.0: coeff = 0.5 * (1 + cos(0))   = 0.5 * 2 = 1.0
# 当 decay_ratio=0.5: coeff = 0.5 * (1 + cos(π/2)) = 0.5 * 1 = 0.5
# 当 decay_ratio=1.0: coeff = 0.5 * (1 + cos(π))   = 0.5 * 0 = 0.0

lr = min_lr + coeff * (learning_rate - min_lr)
# decay_ratio=0.0: lr = 6e-5 + 1.0 * (6e-4 - 6e-5) = 6e-4  ← 最大
# decay_ratio=0.5: lr = 6e-5 + 0.5 * (6e-4 - 6e-5) = 3.3e-4
# decay_ratio=1.0: lr = 6e-5 + 0.0 * (6e-4 - 6e-5) = 6e-5  ← 最小
```

### 为什么用 Cosine 而不是线性衰减？

```
线性衰减：
  lr 按固定速率下降
  优点：简单直观
  缺点：前期下降太快（此时模型还没充分探索），后期下降太慢

余弦衰减：
  前期（decay_ratio 小）：曲线平缓，lr 变化慢 → 充分探索
  中期（decay_ratio 0.5）：下降最快
  后期（接近1.0）：曲线平缓，lr 变化慢 → 精细收敛
  
  自然契合了训练动态：
  开始时需要大步探索，后期需要小步精调

  实验结论（来自 GPT-3 / Chinchilla 论文）：
  Cosine decay 比线性衰减好约 0.05-0.1 loss 点
```

### 学习率调度在训练循环中的应用

```python
for iter_num in range(max_iters):
    # 每步动态更新 lr
    if decay_lr:
        lr = get_lr(iter_num)
    else:
        lr = learning_rate

    # 直接修改 optimizer 的 param_groups
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
```

```
注意：不用 scheduler 对象（如 CosineAnnealingLR）
而是手动在每步计算并赋值

优点：
  - 完全透明，逻辑一目了然
  - 方便 resume（iter_num 传入 get_lr 即可自动恢复到正确 lr）
  - 可以任意自定义 lr 曲线

缺点：
  - 需要自己维护（PyTorch scheduler 会自动处理 resume 等细节）
  - 如果自己实现有 bug，不容易发现
```



---

## 十、混合精度训练 — autocast + GradScaler

### 源码

```python
# 配置混合精度上下文
device_type = 'cuda' if 'cuda' in device else 'cpu'
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]

ctx = nullcontext() if device_type == 'cpu' else \
      torch.amp.autocast(device_type=device_type, dtype=ptdtype)

# GradScaler：只有 float16 才需要
scaler = torch.cuda.amp.GradScaler(enabled=(dtype == 'float16'))
```

### float32 / float16 / bfloat16 的区别

```
浮点数格式（IEEE 754）：

  float32 (FP32)：  1位符号 + 8位指数  + 23位尾数 = 32 bits
    精度高，范围大，GPU 计算慢
    范围: ≈ ±3.4e38

  float16 (FP16)：  1位符号 + 5位指数  + 10位尾数 = 16 bits
    精度低，范围小，GPU 计算快（A100: 312 TFLOPS vs 77 TFLOPS for FP32）
    范围: ≈ ±65504    ← 容易溢出/下溢！

  bfloat16 (BF16)： 1位符号 + 8位指数  + 7位尾数  = 16 bits
    精度低，但范围大（和 FP32 一样），GPU 计算快
    范围: ≈ ±3.4e38   ← 不容易溢出，训练更稳定

          符号  指数    尾数
  FP32:   [1]  [8位]  [23位]   → 精度高，速度慢
  FP16:   [1]  [5位]  [10位]   → 精度低，范围小，容易出 NaN
  BF16:   [1]  [8位]  [ 7位]   → 精度低，范围大，训练稳定

推荐：
  A100/H100 → bfloat16（最稳定）
  V100/T4   → float16（没有 BF16 硬件支持）
  CPU/调试  → float32
```

### autocast 做了什么？

```python
with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
    logits, loss = model(X, Y)
```

```
autocast 上下文管理器：
  自动决定每个算子用什么精度

  用 BF16（快）的算子：
    矩阵乘法（Linear, Attention）→ 占 95% 的 FLOPs
    卷积

  保持 FP32（慢但精度关键）的算子：
    softmax, layer_norm, loss 计算
    → 这些精度敏感，用 FP32 保证数值稳定

  效果：速度接近纯 BF16，精度接近纯 FP32
  实际加速：约 2-3x（矩阵乘法占主要时间）
```

### GradScaler 解决 FP16 的梯度下溢

```
FP16 的问题：
  梯度值通常很小（如 1e-8）
  FP16 最小正数 ≈ 6e-5
  → 梯度被截断为 0 → 参数不更新 → 训练停滞

GradScaler 的解决方案（Loss Scaling）：

  前向：
    loss_scaled = loss * scale_factor    scale_factor 初始 = 65536
  
  反向：
    backward() 计算的是 scaled 后的梯度
    梯度值 × scale_factor → 不下溢了

  更新前：
    scaler.unscale_(optimizer)          梯度除以 scale_factor，还原真实梯度
    clip_grad_norm_(...)                梯度裁剪（需要在 unscale 后）
    scaler.step(optimizer)             检查是否有 inf/nan，没有则 step()
    scaler.update()                    动态调整 scale_factor

  动态调整：
    如果连续 N 步没有 inf/nan → scale_factor 翻倍（更激进）
    如果出现 inf/nan → scale_factor 减半（更保守），这步跳过

BF16 不需要 GradScaler 的原因：
  BF16 的指数位和 FP32 一样（8位）
  → 表示范围和 FP32 一样
  → 不会下溢
  → scaler = GradScaler(enabled=False) 等于啥也不做
```



---

## 十一、评估函数 — estimate_loss()

### 源码

```python
@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):         # eval_iters = 200
            X, Y = get_batch(split)
            with ctx:
                logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out
```

### 为什么需要 estimate_loss 而不是直接用训练 loss？

```
训练 loss 的问题：
  每步 loss 都是单个 mini-batch 的结果，方差很大
  一个特别难/容易的 batch 会让 loss 看起来很好或很差
  不能反映模型的真实能力

estimate_loss 的解决方案：
  采样 200 个 batch，取平均 → 稳定的 loss 估计
  同时评估 train split 和 val split
  
  train loss：评估模型在训练数据上的拟合程度
  val loss：评估模型的泛化能力（没见过的数据）

  val loss - train loss = 过拟合程度
  → 如果差距大，说明模型记住了训练数据，泛化差
```

### model.eval() 和 model.train() 的区别

```
model.eval()：
  - Dropout 关闭（所有神经元都激活）
  - BatchNorm 使用全局统计量（非 mini-batch）
  - 梯度不计算（配合 @torch.no_grad()）

model.train()：
  - Dropout 正常工作（随机 mask 神经元）
  - BatchNorm 使用当前 mini-batch 统计量

nanoGPT 里：
  dropout=0.0（预训练不用 dropout）
  没用 BatchNorm
  → 两个模式实际上没有功能区别
  → 但保留是好习惯（未来可能改配置）
```

### @torch.no_grad() 装饰器

```python
@torch.no_grad()
def estimate_loss():
    ...
```

```
等价于：
def estimate_loss():
    with torch.no_grad():
        ...

效果：
  函数内部的所有操作不构建计算图
  → 不记录前向传播的中间激活值
  → 显存占用大幅减少（不需要保存反向传播所需的激活）
  → 速度约快 2-4x（少了 autograd 的追踪开销）

何时用：
  推理、评估、生成 → 不需要梯度，用 no_grad
  训练前向传播 → 需要梯度，不用 no_grad
```

---

## 十二、Checkpoint 保存与加载

### 源码

```python
if iter_num % eval_interval == 0 and master_process:
    losses = estimate_loss()
    print(f"step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    if losses['val'] < best_val_loss or always_save_checkpoint:
        best_val_loss = losses['val']
        if iter_num > 0:
            checkpoint = {
                'model':       raw_model.state_dict(),
                'optimizer':   optimizer.state_dict(),
                'model_args':  model_args,
                'iter_num':    iter_num,
                'best_val_loss': best_val_loss,
                'config':      config,
            }
            print(f"saving checkpoint to {out_dir}")
            torch.save(checkpoint, os.path.join(out_dir, 'ckpt.pt'))
```

### checkpoint 保存了什么？

```
model:       模型参数（所有 layer 的 weight/bias）
             ← 最重要，这是"成果"

optimizer:   优化器状态（m, v 动量，step 计数）
             ← 用于精确 resume，没有它 resume 时前几百步会不稳定

model_args:  模型结构配置（n_layer, n_head, n_embd 等）
             ← 确保 resume 时能重建完全相同的模型结构

iter_num:    训练到第几步了
             ← resume 时恢复 lr schedule 的位置

best_val_loss: 历史最好的 val loss
             ← 用于判断是否保存新的 checkpoint

config:      所有超参数（记录实验设置）
             ← 方便事后查看这个 checkpoint 是怎么训练的
```

### always_save_checkpoint vs 只在 val loss 改善时保存

```
always_save_checkpoint = True（默认）：
  每次评估都覆盖保存
  优点：永远有最新的 checkpoint，方便断点续训
  缺点：可能保存一个比历史最好结果更差的模型

always_save_checkpoint = False：
  只有 val loss < best_val_loss 时才保存
  优点：保留最优模型
  缺点：如果最后几万步 loss 没有改善，就没有 checkpoint 可以 resume

实践建议：长时间训练时，两个策略都要，保留两个文件：
  ckpt.pt       → 最新（用于 resume）
  ckpt_best.pt  → 历史最优（用于推理）
```



---

## 十三、核心训练循环 — 逐行解析

### 源码（完整）

```python
X, Y = get_batch('train')   # 在循环外预取第一个 batch
local_iter_num = 0

while True:
    # ── 1. 设置学习率 ──────────────────────────────────────────
    lr = get_lr(iter_num) if decay_lr else learning_rate
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    # ── 2. 定期评估 ────────────────────────────────────────────
    if iter_num % eval_interval == 0 and master_process:
        losses = estimate_loss()
        # ... 打印、保存 checkpoint ...

    if iter_num == 0 and eval_only:
        break

    # ── 3. 梯度累积前向/反向 ───────────────────────────────────
    for micro_step in range(gradient_accumulation_steps):
        if ddp:
            # 只在最后一个 micro_step 触发 AllReduce 梯度同步
            model.require_backward_grad_sync = (micro_step == gradient_accumulation_steps - 1)

        with ctx:
            logits, loss = model(X, Y)
            loss = loss / gradient_accumulation_steps    # ← 关键：缩放 loss

        X, Y = get_batch('train')    # 立即预取下一个 batch（异步 IO）
        scaler.scale(loss).backward()

    # ── 4. 梯度裁剪 ────────────────────────────────────────────
    if grad_clip != 0.0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

    # ── 5. 参数更新 ────────────────────────────────────────────
    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad(set_to_none=True)

    # ── 6. 计时和日志 ──────────────────────────────────────────
    t1 = time.time()
    dt = t1 - t0
    t0 = t1
    if iter_num % log_interval == 0 and master_process:
        lossf = loss.item() * gradient_accumulation_steps  # 还原真实 loss 值
        if local_iter_num >= 5:   # 让 mfu 估计稳定后再打印
            mfu = raw_model.estimate_mfu(batch_size * gradient_accumulation_steps, dt)
            running_mfu = mfu if running_mfu == -1.0 else 0.9*running_mfu + 0.1*mfu
        print(f"iter {iter_num}: loss {lossf:.4f}, time {dt*1000:.2f}ms, mfu {running_mfu*100:.2f}%")

    iter_num += 1
    local_iter_num += 1
    if iter_num > max_iters:
        break
```

### 关键细节深入解析

#### 为什么 loss 要除以 gradient_accumulation_steps？

```python
loss = loss / gradient_accumulation_steps
```

```
目标：梯度累积 N 步后，等效于用 N 倍大的 batch 做一次 step

如果不除：
  每步 loss = cross_entropy(logits, targets)   ← 约在 2~5 之间
  backward() 计算的梯度是基于这个 loss 的
  累积 40 步 → 梯度是单步的 40 倍
  optimizer.step() 会走 40 倍大的步长
  → 等效于学习率 × 40，训练不稳定

除以 gradient_accumulation_steps 后：
  每步贡献的梯度 = 真实梯度 / 40
  累积 40 步 → 梯度恢复到正常大小
  → 和真正的大 batch 等效（每个样本贡献 1/total_batch 的梯度）

数学验证：
  大 batch (N×B) 的梯度 = (1/NB) Σᵢ ∂lᵢ/∂θ
  梯度累积: Σⱼ [ (1/B) Σᵢ ∂lᵢⱼ/∂θ / N ] = (1/NB) Σᵢⱼ ∂lᵢⱼ/∂θ  ✓
```

#### DDP 的延迟梯度同步

```python
if ddp:
    model.require_backward_grad_sync = (micro_step == gradient_accumulation_steps - 1)
```

```
DDP 默认行为：
  每次 backward() 都触发 AllReduce（梯度跨卡求平均）
  梯度累积 40 步 → 40 次 AllReduce → 大量通信开销！

优化：在梯度累积期间关闭同步
  micro_step 0~38：require_backward_grad_sync = False → backward 不通信
  micro_step 39：  require_backward_grad_sync = True  → backward 触发 AllReduce

  只做 1 次 AllReduce，而不是 40 次
  通信量减少 40 倍
  
  注意：这等价于每张卡在本地累积 40 步的梯度，
        最后一步 AllReduce 交换各卡的"40步累积梯度"
        结果和逐步同步完全等价（梯度累积是线性操作）
```

#### IO 和计算重叠

```python
with ctx:
    logits, loss = model(X, Y)        # GPU 计算当前 batch
    loss = loss / gradient_accumulation_steps

X, Y = get_batch('train')            # ← 立即开始取下一个 batch！
scaler.scale(loss).backward()        # GPU 继续做反向传播
```

```
时序：
  GPU:  [forward(X,Y)] [backward] [forward(X',Y')] [backward] ...
  CPU:                [get_batch(X',Y')]           [get_batch(X'',Y'')] ...

CPU 取数据 和 GPU 跑反向传播 同时进行
→ 消除了 IO 等待时间

前提：
  get_batch() 使用了 pin_memory + non_blocking=True
  CPU 和 GPU 之间的数据传输也是异步的
```

#### 梯度裁剪

```python
if grad_clip != 0.0:
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
```

```
unscale_ 必须在 clip 之前：
  如果使用 FP16 + GradScaler，梯度是 scale 过的（×65536）
  clip_grad_norm_ 会把 scaled 梯度当真实梯度，阈值判断错误
  unscale_ 先把梯度还原，再 clip

clip_grad_norm_(model.parameters(), max_norm=1.0)：
  计算所有参数梯度向量的 L2 范数（global norm）
  如果 global norm > 1.0，所有梯度等比例缩小

  global_norm = sqrt( Σᵢ ||∇wᵢ||² )
  if global_norm > max_norm:
      factor = max_norm / global_norm
      for each grad: grad *= factor

为什么需要梯度裁剪？
  偶发的"坏 batch"（极端 token 分布）会产生非常大的梯度
  → 一步跨越太远 → 模型参数飞走 → loss 飙升为 NaN → 训练崩溃

grad_clip=1.0：
  只要梯度范数超过 1.0，就缩小
  LLM 训练的标准做法，几乎所有大模型都用
```

#### zero_grad(set_to_none=True)

```python
optimizer.zero_grad(set_to_none=True)
```

```
默认 zero_grad()：
  将所有参数的 .grad 张量填为 0
  → 这些 tensor 依然存在于显存中

set_to_none=True：
  将所有参数的 .grad 设为 None
  → tensor 的存储被释放
  → 节省显存（124M 模型 × 4 bytes × 2 = ~1GB）
  → 下一次 backward() 会重新分配 grad tensor

注意：如果某参数在当前 batch 没有梯度（如 sparse embedding）
  None 语义正确（"没计算过"）
  0 语义错误（"计算了但是0"）→ 可能触发不必要的计算
```



---

## 十四、MFU — 模型浮点利用率

### 源码（在 model.py 的 GPT 类中）

```python
def estimate_mfu(self, fwdbwd_per_iter, dt):
    """
    估算模型浮点利用率（MFU）
    对标 A100 bfloat16 峰值：312 TFLOPS
    参考：PaLM 论文 (https://arxiv.org/abs/2204.02311) 附录 B
    """
    N = self.get_num_params()                    # 模型参数量
    cfg = self.config
    L, H, Q, T = cfg.n_layer, cfg.n_head, cfg.n_embd//cfg.n_head, cfg.block_size

    # 每个 token 的 FLOPs 估算：
    #   前向传播：6N + 12LHQ×T（注意力的额外贡献）
    #   前向+反向：×2（反向约是前向 2 倍）
    flops_per_token = 6*N + 12*L*H*Q*T
    flops_per_fwdbwd = flops_per_token * T
    flops_per_iter = flops_per_fwdbwd * fwdbwd_per_iter    # fwdbwd_per_iter = batch_size × accum_steps

    # 实际达到的 TFLOPS
    flops_achieved = flops_per_iter * (1.0/dt)
    flops_promised = 312e12   # A100 BF16 峰值
    mfu = flops_achieved / flops_promised
    return mfu
```

### MFU 的含义

```
MFU = 实际 FLOPS / 硬件峰值 FLOPS

例：
  A100 峰值 BF16 = 312 TFLOPS
  实际达到 156 TFLOPS → MFU = 0.5 = 50%

好的 MFU 是多少？
  30-40%：正常
  40-60%：优秀
  60%+：  极优（接近 FlashAttention + 完全优化）
  <20%：  有问题（IO 瓶颈、batch 太小、没用 compile 等）

为什么达不到 100%？
  - 内存带宽瓶颈（VRAM 读写）
  - kernel 启动开销
  - 注意力的不规则访问模式
  - 数值检查（overflow/nan 检测）
  - Softmax、LayerNorm 等无法完全并行的操作
```

### FLOPs 估算推导

```
6N 的来源（参考 Kaplan et al. 2020）：
  前向传播：1 次矩阵乘法 ≈ 2 × 参数量 FLOPs（每参数 1 乘 + 1 加）
  前向+反向：≈ 3 × 前向（反向约 2 倍前向）
  → 总: 6N

12LHQT 的来源（注意力的 O(T²) 部分）：
  Q×K 矩阵乘: 每层每头 2T² × Q FLOPs
  softmax(QK)×V:  同上
  L 层，H 头
  → 总: 4 × L × H × T² × Q
  前向+反向 ×3 = 12LHQT²
  → 每 token: 12LHQT

GPT-2 124M 举例：
  N = 124M
  6N = 744M FLOPs/token
  12LHQ = 12 × 12 × 12 × 64 = 110,592
  12LHQ×T = 110,592 × 1024 ≈ 113M FLOPs/token
  
  总计 ≈ 857M FLOPs/token
  前向+反向 × 约2 = 1714M FLOPs/token
```

---

## 十五、完整训练流程数据流图

```
磁盘文件
  train.bin (17GB, 9B tokens, uint16)
  val.bin   (8.5MB, 4M tokens, uint16)
       ↓
  np.memmap（内存映射，不加载到 RAM）
       ↓
  get_batch()：随机取 12 × 1024 连续 token
       ↓ pin_memory + non_blocking
  GPU VRAM：x,y = (12,1024) int64

  ┌─────────────────────────────────────────────────────────┐
  │  训练循环（600,000 步）                                   │
  │                                                          │
  │  for iter_num in range(600000):                         │
  │                                                          │
  │    lr = get_lr(iter_num)      # cosine warmup           │
  │                                                          │
  │    for micro_step in range(40):   # 梯度累积             │
  │      with autocast(bfloat16):                           │
  │        logits, loss = model(x, y)                       │
  │           ↓                                              │
  │        Token Embed + Pos Embed                          │
  │           ↓                                              │
  │        12 × (LayerNorm → Attention → MLP)               │
  │           ↓                                              │
  │        LayerNorm → lm_head → CrossEntropy(loss)         │
  │                                                          │
  │      loss /= 40                                          │
  │      x, y = get_batch()       # 立即预取（IO 重叠）     │
  │      scaler.scale(loss).backward()   # 累积梯度          │
  │                                                          │
  │    unscale_() → clip_grad_norm_(1.0)                    │
  │    scaler.step(optimizer)     # AdamW 参数更新           │
  │    scaler.update()                                       │
  │    optimizer.zero_grad(set_to_none=True)                 │
  │                                                          │
  │    每 2000 步：estimate_loss() + save ckpt.pt           │
  └─────────────────────────────────────────────────────────┘

  ckpt.pt（最终产物）
       ↓
  sample.py → GPT.generate() → 输出文本
```

---

## 十六、关键设计决策汇总

| 设计 | 实现 | 为什么 |
|------|------|--------|
| 极简数据加载 | `np.memmap` + 随机采样 | 17GB 数据无法放入 RAM，DataLoader 反而有开销 |
| 梯度累积 | `loss /= N` + 累积 backward | 用小 batch 模拟 0.5M tokens 的有效 batch，节省显存 |
| IO 与计算重叠 | backward 前预取下一批 | 消除 GPU 等待 CPU 的时间 |
| Cosine LR + Warmup | 手动 `get_lr()` | 对齐 Chinchilla 推荐，resume 自动恢复 |
| 分组 Weight Decay | 2D 参数 decay，1D 不 decay | Bias/LayerNorm 加 decay 有害 |
| fused AdamW | `inspect.signature` 检测 | PyTorch 2.0+ 快 15-20% |
| BF16 混合精度 | `autocast` + 无 GradScaler | A100 加速 2-3x，BF16 范围大不需要 scaling |
| DDP 延迟同步 | `require_backward_grad_sync` | 40 次 AllReduce → 1 次，通信减少 40x |
| torch.compile | PyTorch 2.0 编译 | 算子融合，A100 上加速 ~25% |
| `set_to_none=True` | `zero_grad` 参数 | 比 zero_grad() 节省 ~1GB 显存 |
| 动态 GradScaler | FP16 专用，BF16 禁用 | FP16 防梯度下溢，BF16 不需要 |
| Checkpoint 每步都存 | `always_save_checkpoint` | 长时间训练需要随时 resume |
| MFU 监控 | `estimate_mfu()` | 快速发现 IO 瓶颈、配置问题 |

---

## 十七、如何运行

```bash
# 1. 准备数据（字符级 Shakespeare，最快，适合入门）
python data/shakespeare_char/prepare.py

# 2. 单卡训练（macbook 也能跑）
python train.py config/train_shakespeare_char.py

# 3. 8 卡 A100 训练 GPT-2 124M
torchrun --nproc_per_node=8 train.py config/train_gpt2.py

# 4. 从 checkpoint 恢复训练
python train.py config/train_gpt2.py --init_from=resume

# 5. 在 GPT-2 预训练权重上微调
python train.py config/finetune_shakespeare.py

# 6. 生成文本
python sample.py --out_dir=out-shakespeare-char \
                 --start="ROMEO:" \
                 --num_samples=3 \
                 --max_new_tokens=200

# 7. 只评估（不训练）
python train.py config/eval_gpt2.py
```

---

## 我的理解 / 疑问

<!-- 在这里记录学习笔记 -->
