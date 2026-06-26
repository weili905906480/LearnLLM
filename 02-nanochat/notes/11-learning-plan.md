# 从零到 nanochat：LLM 学习完整计划

---

## 学习路线总览

```
阶段1：数学与编程基础（2-3周）
    ↓
阶段2：深度学习基础（2-3周）
    ↓
阶段3：PyTorch 实战（2-3周）
    ↓
阶段4：Transformer 架构（2-3周）
    ↓
阶段5：LLM 预训练（2-3周）
    ↓
阶段6：LLM 微调与对齐（2-3周）
    ↓
阶段7：nanochat 项目深入（3-4周）
```

---

## 阶段1：数学与编程基础（2-3周）

### 1.1 线性代数

**核心概念：**
- 向量、矩阵、张量
- 矩阵乘法、转置、逆
- 特征值与特征向量
- 矩阵分解（SVD、PCA）

**学习资源：**
- [3Blue1Brown: Essence of Linear Algebra](https://www.youtube.com/playlist?list=PLZHQObOWTQDPD3MizzM2xVFitgF8hE_ab)
- [MIT 18.06 Linear Algebra](https://ocw.mit.edu/courses/18-06-linear-algebra-spring-2010/)

**与 LLM 的关联：**
- Embedding 层：向量表示
- Attention 机制：矩阵乘法
- 模型参数：高维张量

### 1.2 概率论与统计

**核心概念：**
- 概率分布（均匀、正态、softmax）
- 条件概率、贝叶斯定理
- 期望、方差、协方差
- 最大似然估计

**与 LLM 的关联：**
- Softmax：概率分布
- 交叉熵损失：最大似然
- 采样策略：温度、top-k、top-p

### 1.3 微积分

**核心概念：**
- 导数、偏导数、梯度
- 链式法则
- 梯度下降
- 梯度消失/爆炸

**与 LLM 的关联：**
- 反向传播：链式法则
- 优化器：梯度下降变体
- 梯度裁剪：防止爆炸

### 1.4 Python 编程

**核心技能：**
- Python 基础语法
- NumPy 数组操作
- 函数式编程（map、filter、reduce）
- 类与继承
- 装饰器、生成器

**练习：**
- 用 NumPy 实现矩阵乘法
- 实现简单的神经网络（纯 Python）

---

## 阶段2：深度学习基础（2-3周）

### 2.1 神经网络基础

**核心概念：**
- 感知机、多层感知机
- 激活函数（ReLU、GELU、Sigmoid）
- 前向传播、反向传播
- 损失函数（MSE、交叉熵）

**学习资源：**
- [3Blue1Brown: Neural Networks](https://www.youtube.com/playlist?list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi)
- [CS231n: Convolutional Neural Networks](http://cs231n.stanford.edu/)

### 2.2 优化算法

**核心概念：**
- 随机梯度下降（SGD）
- 动量（Momentum）
- Adam 优化器
- 学习率调度

**与 nanochat 的关联：**
- nanochat 使用 MuonAdamW 优化器
- 学习率 warmup + warmdown 调度

### 2.3 正则化技术

**核心概念：**
- Dropout
- Batch Normalization / Layer Normalization / RMS Normalization
- 权重衰减（Weight Decay）
- 梯度裁剪（Gradient Clipping）

**与 nanochat 的关联：**
- nanochat 使用 RMS Normalization
- 使用权重衰减作为正则化

### 2.4 序列模型基础

**核心概念：**
- 循环神经网络（RNN）
- 长短期记忆网络（LSTM）
- 门控循环单元（GRU）
- 序列到序列模型

**目的：** 理解为什么 Transformer 被发明

---

## 阶段3：PyTorch 实战（2-3周）

### 3.1 PyTorch 基础

**学习内容：**
- Tensor 操作（创建、索引、变形）
- 自动微分（autograd）
- GPU 加速（.to(device)）
- 数据加载（Dataset、DataLoader）

**学习资源：**
- [PyTorch 官方教程](https://pytorch.org/tutorials/)
- [60分钟闪电战](https://pytorch.org/tutorials/beginner/deep_learning_60min_blitz.html)

**练习：**
```python
# 练习1：实现线性回归
import torch
import torch.nn as nn

# 创建数据
X = torch.randn(100, 1)
y = 2 * X + 1 + 0.1 * torch.randn(100, 1)

# 定义模型
model = nn.Linear(1, 1)

# 训练循环
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

for epoch in range(100):
    y_pred = model(X)
    loss = criterion(y_pred, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

### 3.2 PyTorch 模型定义

**学习内容：**
- `nn.Module` 继承
- `__init__` 和 `forward` 方法
- 参数注册（`nn.Parameter`）
- 模型保存与加载（`state_dict`）

**nanochat 代码对应：**
- `nanochat/gpt.py:154` — `GPT(nn.Module)`
- `nanochat/gpt.py:65` — `CausalSelfAttention(nn.Module)`
- `nanochat/gpt.py:100` — `MLP(nn.Module)`

### 3.3 PyTorch 训练循环

**学习内容：**
- 标准训练循环模板
- 梯度累积
- 混合精度训练（GradScaler）
- 分布式训练（DDP）

**nanochat 代码对应：**
- `scripts/base_train.py` — 预训练循环
- `scripts/chat_sft.py` — SFT 训练循环

**练习：**
```python
# 练习2：实现 MNIST 分类
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms

# 数据加载
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)

# 模型定义
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 128)
        self.fc2 = nn.Linear(128, 10)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = x.view(-1, 784)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# 训练
model = Net()
optimizer = optim.Adam(model.parameters())
criterion = nn.CrossEntropyLoss()

for epoch in range(5):
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
```

### 3.4 PyTorch 高级特性

**学习内容：**
- `torch.compile`（JIT 编译）
- 自定义 autograd Function
- Hook 机制
- Profiler 性能分析

**nanochat 代码对应：**
- `scripts/chat_sft.py:120` — `torch.compile(model)`

---

## 阶段4：Transformer 架构（2-3周）

### 4.1 Attention 机制

**核心概念：**
- 自注意力（Self-Attention）
- 缩放点积注意力（Scaled Dot-Product Attention）
- 多头注意力（Multi-Head Attention）
- 因果注意力（Causal Attention）

**论文：**
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762)

**nanochat 代码对应：**
- `nanochat/gpt.py:65` — `CausalSelfAttention` 类

**关键公式：**
```
Attention(Q, K, V) = softmax(QK^T / √d_k) V
```

### 4.2 Transformer 块

**核心概念：**
- 位置编码（Positional Encoding）/ 旋转位置编码（RoPE）
- 前馈网络（Feed-Forward Network）
- 残差连接（Residual Connection）
- 层归一化（Layer Normalization）/ RMS 归一化

**nanochat 代码对应：**
- `nanochat/gpt.py:100` — `Block` 类
- 使用 RoPE（旋转位置编码）
- 使用 RMS Normalization

### 4.3 GPT 架构

**核心概念：**
- Decoder-only 架构
- 自回归生成
- 下一个 Token 预测
- KV Cache

**nanochat 代码对应：**
- `nanochat/gpt.py:154` — `GPT` 类
- `nanochat/gpt.py:416` — `forward` 方法
- `nanochat/gpt.py:483` — `generate` 方法

**练习：**
```python
# 练习3：实现简化版 Transformer
import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleAttention(nn.Module):
    def __init__(self, d_model, n_head):
        super().__init__()
        self.n_head = n_head
        self.d_k = d_model // n_head
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.proj = nn.Linear(d_model, d_model)
    
    def forward(self, x):
        B, T, C = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, self.n_head, self.d_k).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        att = (q @ k.transpose(-2, -1)) * (1.0 / self.d_k ** 0.5)
        att = F.softmax(att, dim=-1)
        out = (att @ v).transpose(1, 2).reshape(B, T, C)
        return self.proj(out)
```

### 4.4 现代 Transformer 变体

**核心概念：**
- Grouped Query Attention（GQA）
- Sliding Window Attention
- Mixture of Experts（MoE）
- Flash Attention

**nanochat 代码对应：**
- 使用 GQA（`n_kv_head` < `n_head`）
- 使用 Sliding Window（`window_pattern`）
- 支持 Flash Attention 3

---

## 阶段5：LLM 预训练（2-3周）

### 5.1 预训练数据

**核心概念：**
- 数据收集与清洗
- 分词器（BPE、WordPiece）
- 数据混合策略
- 数据并行

**nanochat 代码对应：**
- `nanochat/dataset.py` — 数据加载
- `nanochat/tokenizer.py` — 分词器
- `scripts/tok_train.py` — 分词器训练

**练习：**
```python
# 练习4：实现简单 BPE 分词器
import re
from collections import Counter

def train_bpe(text, vocab_size):
    # 初始词表：单个字符
    tokens = list(text.encode('utf-8'))
    vocab = {bytes([i]): i for i in range(256)}
    
    while len(vocab) < vocab_size:
        # 统计相邻 token 对的频率
        pairs = Counter(zip(tokens[:-1], tokens[1:]))
        if not pairs:
            break
        
        # 合并最频繁的 pair
        best_pair = max(pairs, key=pairs.get)
        new_token = best_pair[0] + best_pair[1]
        vocab[new_token] = len(vocab)
        
        # 更新 token 序列
        new_tokens = []
        i = 0
        while i < len(tokens):
            if i < len(tokens) - 1 and (tokens[i], tokens[i+1]) == best_pair:
                new_tokens.append(new_token)
                i += 2
            else:
                new_tokens.append(tokens[i])
                i += 1
        tokens = new_tokens
    
    return vocab
```

### 5.2 预训练目标

**核心概念：**
- 下一个 Token 预测（Next Token Prediction）
- 因果语言模型（Causal LM）
- 损失函数（交叉熵）
- Bits Per Byte（BPB）指标

**nanochat 代码对应：**
- `nanochat/gpt.py:474` — 损失计算
- `nanochat/loss_eval.py` — BPB 评估

### 5.3 预训练优化

**核心概念：**
- 学习率调度（Warmup + Cosine/Linear Decay）
- 权重初始化
- 梯度累积
- 混合精度训练

**nanochat 代码对应：**
- `scripts/base_train.py:314` — `get_lr_multiplier` 学习率调度
- `nanochat/gpt.py:202` — `init_weights` 权重初始化
- `nanochat/optim.py` — MuonAdamW 优化器

### 5.4 缩放定律

**核心概念：**
- Chinchilla 缩放定律
- 计算最优模型大小
- 训练 Token 数 vs 模型大小

**nanochat 代码对应：**
- `runs/scaling_laws.sh` — 缩放定律实验
- `--depth` 参数控制模型大小

---

## 阶段6：LLM 微调与对齐（2-3周）

### 6.1 监督微调（SFT）

**核心概念：**
- 对话格式（Chat Template）
- 指令微调（Instruction Tuning）
- 多任务混合训练
- 学习率调整

**nanochat 代码对应：**
- `scripts/chat_sft.py` — SFT 训练脚本
- `tasks/` 目录 — 各种训练任务

**关键知识点：**
- SFT 学习率通常比预训练低 10-100 倍
- 需要过滤无效训练目标（NaN loss 问题）
- 数据混合比例影响模型能力

### 6.2 RLHF（基于人类反馈的强化学习）

**核心概念：**
- 奖励模型（Reward Model）
- PPO 算法
- DPO（Direct Preference Optimization）
- GRPO（Group Relative Policy Optimization）

**nanochat 代码对应：**
- `scripts/chat_rl.py` — RL 训练脚本
- 使用 GRPO 算法

### 6.3 推理能力训练

**核心概念：**
- Chain-of-Thought（CoT）
- 工具使用（Tool Use）
- 自我验证（Self-Verification）
- 搜索与规划

**nanochat 代码对应：**
- GSM8K：数学推理 + Python 工具调用
- SpellingBee：手动推理 + Python 验证

### 6.4 模型评估

**核心概念：**
- 困惑度（Perplexity）
- Bits Per Byte（BPB）
- 下游任务评估（MMLU、GSM8K 等）
- 人类评估

**nanochat 代码对应：**
- `nanochat/core_eval.py` — CORE 评分
- `scripts/chat_eval.py` — Chat 评估
- `scripts/base_eval.py` — Base 模型评估

---

## 阶段7：nanochat 项目深入（3-4周）

### 7.1 项目结构理解

**学习目标：**
- 理解整体架构
- 掌握代码组织方式
- 熟悉各模块职责

**学习路径：**
1. 阅读 `README.md` 和 `README-ch.md`
2. 阅读 `AGENTS.md` 了解项目规范
3. 按文件结构逐模块学习

**关键文件：**
```
nanochat/
├── gpt.py              # 核心模型
├── tokenizer.py        # 分词器
├── dataset.py          # 数据加载
├── optim.py            # 优化器
├── engine.py           # 推理引擎
└── checkpoint_manager.py # 检查点管理

scripts/
├── base_train.py       # 预训练
├── chat_sft.py         # SFT
├── chat_rl.py          # RL
└── chat_eval.py        # 评估

tasks/
├── common.py           # 任务基类
├── smoltalk.py         # 对话数据
├── mmlu.py             # 多选题
├── gsm8k.py            # 数学题
└── spellingbee.py      # 拼写任务
```

### 7.2 核心代码精读

**精读顺序：**

**Week 1：模型架构**
1. `nanochat/gpt.py` — GPT 模型
   - `GPTConfig` — 配置类
   - `CausalSelfAttention` — 注意力机制
   - `Block` — Transformer 块
   - `GPT` — 完整模型
   - `forward` — 前向传播
   - `generate` — 文本生成

2. `nanochat/tokenizer.py` — 分词器
   - `RustBPETokenizer` — 分词器实现
   - `render_conversation` — 对话渲染
   - `render_for_completion` — 推理渲染

**Week 2：训练流程**
1. `scripts/base_train.py` — 预训练
   - 数据加载
   - 训练循环
   - 评估与保存

2. `scripts/chat_sft.py` — SFT 训练
   - 数据混合（`TaskMixture`）
   - 数据打包（`sft_data_generator_bos_bestfit`）
   - 训练循环

3. `tasks/` 目录 — 各种任务
   - `Task` 基类
   - 各任务实现

**Week 3：推理与部署**
1. `nanochat/engine.py` — 推理引擎
   - KV Cache
   - 工具调用
   - 流式生成

2. `scripts/chat_web.py` — Web UI
3. `scripts/chat_cli.py` — CLI 界面

### 7.3 动手实践

**实践项目：**

**项目1：添加新任务**
- 创建 `tasks/my_task.py`
- 实现 `Task` 基类
- 添加到 SFT 训练混合

**项目2：修改模型架构**
- 尝试不同的注意力机制
- 实验不同的位置编码
- 调整模型大小

**项目3：优化训练**
- 尝试不同的学习率调度
- 实验不同的数据混合比例
- 优化训练速度

**项目4：改进推理**
- 实现新的采样策略
- 添加新的工具支持
- 优化生成速度

### 7.4 进阶学习

**论文阅读：**
1. [Attention Is All You Need](https://arxiv.org/abs/1706.03762) — Transformer 原始论文
2. [Language Models are Few-Shot Learners](https://arxiv.org/abs/2005.14165) — GPT-3
3. [Training Compute-Optimal Large Language Models](https://arxiv.org/abs/2203.15556) — Chinchilla
4. [Scaling Laws for Neural Language Models](https://arxiv.org/abs/2001.08361) — 缩放定律

**Karpathy 其他项目：**
1. [nanoGPT](https://github.com/karpathy/nanoGPT) — 预训练
2. [minGPT](https://github.com/karpathy/minGPT) — 教学用
3. [llm.c](https://github.com/karpathy/llm.c) — C 语言实现

---

## 学习检查点

### 阶段1 完成标志
- [ ] 能手写矩阵乘法
- [ ] 理解梯度下降原理
- [ ] 能用 NumPy 实现简单神经网络

### 阶段2 完成标志
- [ ] 理解前向/反向传播
- [ ] 知道常见激活函数的区别
- [ ] 理解梯度消失/爆炸问题

### 阶段3 完成标志
- [ ] 能用 PyTorch 定义模型
- [ ] 能编写训练循环
- [ ] 理解 Dataset/DataLoader

### 阶段4 完成标志
- [ ] 能手写 Self-Attention
- [ ] 理解 Multi-Head Attention
- [ ] 知道 RoPE 的原理

### 阶段5 完成标志
- [ ] 理解 BPE 分词器
- [ ] 理解因果语言模型
- [ ] 知道缩放定律

### 阶段6 完成标志
- [ ] 理解 SFT 的作用
- [ ] 知道 RLHF 的流程
- [ ] 理解工具调用的实现

### 阶段7 完成标志
- [ ] 能读懂 nanochat 核心代码
- [ ] 能修改和扩展功能
- [ ] 能独立训练和评估模型

---

## 常见问题

### Q: 需要 GPU 吗？
A: 学习阶段不需要。CPU 可以运行小模型和理解代码。但要训练有效果的模型需要 GPU。

### Q: 需要多少时间？
A: 全职学习约 4-6 个月。兼职学习（每天 2-3 小时）约 8-12 个月。

### Q: 数学不好能学吗？
A: 可以。先从直觉理解开始，遇到具体问题再深入数学细节。

### Q: Python 不熟能学吗？
A: 建议先花 1-2 周学 Python 基础。nanochat 代码相对简洁，适合学习。

---

## 参考资源汇总

**书籍：**
- [Dive into Deep Learning](https://d2l.ai/) — 动手学深度学习
- [Neural Networks and Deep Learning](http://neuralnetworksanddeeplearning.com/) — 神经网络基础

**课程：**
- [CS231n](http://cs231n.stanford.edu/) — 计算机视觉
- [CS224n](http://web.stanford.edu/class/cs224n/) — 自然语言处理
- [Fast.ai](https://www.fast.ai/) — 实战导向

**博客：**
- [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/)
- [The Annotated Transformer](https://nlp.seas.harvard.edu/annotated-transformer/)

**代码：**
- [nanoGPT](https://github.com/karpathy/nanoGPT) — Karpathy 的预训练实现
- [minGPT](https://github.com/karpathy/minGPT) — 教学用 GPT
- [nanochat](https://github.com/karpathy/nanochat) — 完整 LLM 训练框架

---

## 学习笔记模板

每完成一个阶段，记录以下内容：

```markdown
# 阶段N：[主题]

## 学习日期
YYYY-MM-DD ~ YYYY-MM-DD

## 核心概念
- 概念1：一句话解释
- 概念2：一句话解释

## 代码实践
- 实践1：做了什么，学到了什么
- 实践2：做了什么，学到了什么

## 遇到的问题
- 问题1：如何解决的
- 问题2：如何解决的

## 下一步计划
- 计划1
- 计划2
```

---

## 附录：数学基础学习模块

**位置：** `02-nanochat/code/learn/math/`

**模块结构：**
```
learn/
├── __init__.py
└── math/
    ├── README.md              # 说明文档
    ├── __init__.py            # 包初始化
    ├── linear_algebra.py      # 线性代数基础
    ├── probability.py         # 概率论与统计
    ├── calculus.py            # 微积分与梯度
    ├── activations.py         # 激活函数
    └── losses.py              # 损失函数
```

**运行方式：**
```powershell
# 激活虚拟环境
.venv\Scripts\activate

# 运行单个模块
python -m learn.math.linear_algebra
python -m learn.math.probability
python -m learn.math.calculus
python -m learn.math.activations
python -m learn.math.losses

# 运行所有模块
python -m learn.math
```

**各模块内容：**

| 模块 | 内容 | 与 nanochat 的关联 |
|------|------|-------------------|
| linear_algebra.py | 向量、矩阵、矩阵乘法、Softmax、Attention 计算 | Attention 机制的核心 |
| probability.py | 概率分布、Softmax、交叉熵、采样策略 | 输出层概率分布、文本生成 |
| calculus.py | 导数、梯度、链式法则、反向传播、梯度下降 | 模型训练、优化器 |
| activations.py | Sigmoid、ReLU、GELU、Softmax 等激活函数 | MLP 层的激活函数 |
| losses.py | MSE、交叉熵、KL 散度、Label Smoothing | 训练损失函数 |

**关键练习：**
1. 手写矩阵乘法，理解 Attention 计算
2. 实现 Softmax，理解概率分布
3. 实现交叉熵损失，理解训练目标
4. 实现简单反向传播，理解梯度计算

---

*最后更新：2026-06-26*
