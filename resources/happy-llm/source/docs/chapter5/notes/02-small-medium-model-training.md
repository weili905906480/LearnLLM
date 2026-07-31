# Chapter 5 小模型与中等模型训练记录

> 本文记录在当前 Windows + CPU 环境中，基于 happy-llm Chapter 5 代码完成“小模型 + 更多预训练数据”和“中等小模型”训练、SFT、推理测试的完整过程。

---

## 一、实验背景

前面直接使用 Chapter 5 默认的 `215M` 模型做了最小预训练和 SFT，流程可以跑通，但推理效果较差：模型容易生成重复词和无意义文本。

主要原因：

1. 默认模型约 `215M` 参数，对当前 CPU 环境来说偏大。
2. 最初预训练数据只有很少样本，基础语言建模能力不足。
3. SFT 不能从零教会模型稳定说话，只能在已有语言能力基础上调整回答风格。
4. 数据太少时，增加 epoch 容易过拟合，而不是提升真实对话能力。

因此改用更适合当前电脑的方案：

```text
小模型 / 中等小模型 + 更多主题预训练数据 + 适量 SFT + 短输出推理
```

---

## 二、新增文件

本次实验在 `chapter5/code/` 目录中新增了以下文件：

```text
chapter5/code/
├── seq_monkey_learnllm_pretrain.jsonl   # LearnLLM 主题预训练数据，920 条
├── BelleGroup_sft_learnllm.jsonl        # LearnLLM 主题 SFT 问答数据，111 条
├── ddp_pretrain_small.py                # 小模型 / 中等小模型预训练脚本
├── ddp_sft_small.py                     # 小模型 / 中等小模型 SFT 脚本
├── model_sample_small.py                # 小模型 / 中等小模型推理脚本
├── base_model_small/                    # 1.18M 小模型预训练权重
├── sft_model_small/                     # 1.18M 小模型 SFT 权重
├── base_model_medium/                   # 4.52M 中等小模型预训练权重
└── sft_model_medium/                    # 4.52M 中等小模型 SFT 权重
```

---

## 三、数据构造方法

### 3.1 预训练数据

预训练数据文件：

```text
seq_monkey_learnllm_pretrain.jsonl
```

规模：

```text
920 条
```

数据主题围绕 LLM 学习概念构造，包括：

- 大语言模型
- 预训练
- SFT
- tokenizer
- embedding
- Transformer
- self-attention
- loss
- batch size
- 学习率
- epoch
- checkpoint
- 过拟合
- 推理采样
- CPU 训练
- 小模型
- 数据质量
- 聊天模板
- SFT loss mask
- 验证集

数据格式为 JSONL，每行一个对象：

```json
{"text": "主题：SFT\n问题：SFT是什么？\n回答：SFT是监督微调..."}
```

这种格式适配 `PretrainDataset`，脚本会读取每行的 `text` 字段进行语言建模训练。

### 3.2 SFT 数据

SFT 数据文件：

```text
BelleGroup_sft_learnllm.jsonl
```

规模：

```text
111 条
```

数据格式为消息列表，每行类似：

```json
[
  {"role": "system", "content": "你是一个耐心、准确的LLM学习助手，回答要简洁清楚。"},
  {"role": "user", "content": "SFT是什么意思？"},
  {"role": "assistant", "content": "SFT是监督微调，使用人工整理的指令和回答数据继续训练模型，让模型更会按指令回答。"}
]
```

这种格式适配 `SFTDataset`，代码会使用 tokenizer 的 chat template 将消息列表转换为训练文本，并主要计算 assistant 回答部分的 loss。

---

## 四、模型配置

### 4.1 小模型配置

```text
dim=128
n_layers=2
n_heads=8
n_kv_heads=4
max_seq_len=256
vocab_size=6144
参数量约：1.180M
```

保存位置：

```text
base_model_small/pretrain_128_2_6144.pth
sft_model_small/sft_dim128_layers2_vocab_size6144.pth
```

### 4.2 中等小模型配置

```text
dim=256
n_layers=4
n_heads=8
n_kv_heads=4
max_seq_len=256
vocab_size=6144
参数量约：4.524M
```

保存位置：

```text
base_model_medium/pretrain_256_4_6144.pth
sft_model_medium/sft_dim256_layers4_vocab_size6144.pth
```

---

## 五、小模型训练过程

### 5.1 小模型预训练命令

PowerShell 单行命令：

```powershell
cd E:\Project\LLM\LearnLLM\resources\happy-llm\source\docs\chapter5\code; python ddp_pretrain_small.py --data_path ./seq_monkey_learnllm_pretrain.jsonl --out_dir base_model_small --batch_size 16 --epochs 20 --num_workers 0 --log_interval 50 --save_interval 500 --learning_rate 3e-4 --max_seq_len 256
```

Bash / Claude Code 命令：

```bash
cd resources/happy-llm/source/docs/chapter5/code
python ddp_pretrain_small.py \
  --data_path ./seq_monkey_learnllm_pretrain.jsonl \
  --out_dir base_model_small \
  --batch_size 16 \
  --epochs 20 \
  --num_workers 0 \
  --log_interval 50 \
  --save_interval 500 \
  --learning_rate 3e-4 \
  --max_seq_len 256
```

训练结果：

```text
Small LLM parameters: 1.180 M
Epoch 1:  loss 8.706 -> 5.403
Epoch 5:  loss 0.866
Epoch 10: loss 0.131
Epoch 15: loss 0.086
Epoch 20: loss 0.084
Saved pretrain checkpoint to base_model_small/pretrain_128_2_6144.pth
```

说明：小模型在 920 条 LearnLLM 主题预训练文本上收敛明显，已经学到一些主题词和句式。

### 5.2 小模型 SFT 命令

PowerShell 单行命令：

```powershell
cd E:\Project\LLM\LearnLLM\resources\happy-llm\source\docs\chapter5\code; python ddp_sft_small.py --data_path ./BelleGroup_sft_learnllm.jsonl --pretrain_ckpt ./base_model_small/pretrain_128_2_6144.pth --out_dir sft_model_small --batch_size 16 --epochs 12 --num_workers 0 --log_interval 10 --save_interval 100 --learning_rate 2e-4 --max_seq_len 256
```

Bash / Claude Code 命令：

```bash
python ddp_sft_small.py \
  --data_path ./BelleGroup_sft_learnllm.jsonl \
  --pretrain_ckpt ./base_model_small/pretrain_128_2_6144.pth \
  --out_dir sft_model_small \
  --batch_size 16 \
  --epochs 12 \
  --num_workers 0 \
  --log_interval 10 \
  --save_interval 100 \
  --learning_rate 2e-4 \
  --max_seq_len 256
```

训练结果：

```text
Loaded pretrain checkpoint from ./base_model_small/pretrain_128_2_6144.pth
Small LLM parameters: 1.180 M
Epoch 1:  loss 8.071
Epoch 6:  loss 4.760
Epoch 12: loss 4.671
Saved SFT checkpoint to sft_model_small/sft_dim128_layers2_vocab_size6144.pth
```

### 5.3 小模型推理效果

测试问题：

```text
SFT是什么意思？
embedding层的作用是什么？
为什么当前模型输出乱码？
什么是验证集？
请简单解释一下学习率。
CPU可以训练大语言模型吗？
```

实际效果：

- 能生成 `模型`、`数据`、`token`、`学习率`、`batch size` 等相关词。
- 但容易重复，例如不断输出 `数据。数据。数据...`。
- 回答不够稳定，不适合作为最终演示模型。

结论：

```text
1.18M 小模型训练速度快，适合验证流程。
但容量太小，回答容易重复，简单对话效果有限。
```

---

## 六、中等小模型训练过程

### 6.1 中等小模型预训练命令

PowerShell 单行命令：

```powershell
cd E:\Project\LLM\LearnLLM\resources\happy-llm\source\docs\chapter5\code; python ddp_pretrain_small.py --data_path ./seq_monkey_learnllm_pretrain.jsonl --out_dir base_model_medium --batch_size 8 --epochs 20 --num_workers 0 --log_interval 50 --save_interval 500 --learning_rate 3e-4 --dim 256 --n_layers 4 --n_heads 8 --n_kv_heads 4 --max_seq_len 256
```

Bash / Claude Code 命令：

```bash
python ddp_pretrain_small.py \
  --data_path ./seq_monkey_learnllm_pretrain.jsonl \
  --out_dir base_model_medium \
  --batch_size 8 \
  --epochs 20 \
  --num_workers 0 \
  --log_interval 50 \
  --save_interval 500 \
  --learning_rate 3e-4 \
  --dim 256 \
  --n_layers 4 \
  --n_heads 8 \
  --n_kv_heads 4 \
  --max_seq_len 256
```

训练结果：

```text
Small LLM parameters: 4.524 M
Epoch 1:  8.785 -> 0.598
Epoch 3:  0.101 -> 0.079
Epoch 10: 约 0.055
Epoch 20: 约 0.050
Saved pretrain checkpoint to base_model_medium/pretrain_256_4_6144.pth
```

说明：中等小模型训练仍然很快，约十几分钟内完成，并且预训练 loss 明显下降。

### 6.2 中等小模型 SFT 命令

PowerShell 单行命令：

```powershell
cd E:\Project\LLM\LearnLLM\resources\happy-llm\source\docs\chapter5\code; python ddp_sft_small.py --data_path ./BelleGroup_sft_learnllm.jsonl --pretrain_ckpt ./base_model_medium/pretrain_256_4_6144.pth --out_dir sft_model_medium --batch_size 8 --epochs 20 --num_workers 0 --log_interval 10 --save_interval 100 --learning_rate 2e-4 --dim 256 --n_layers 4 --n_heads 8 --n_kv_heads 4 --max_seq_len 256
```

Bash / Claude Code 命令：

```bash
python ddp_sft_small.py \
  --data_path ./BelleGroup_sft_learnllm.jsonl \
  --pretrain_ckpt ./base_model_medium/pretrain_256_4_6144.pth \
  --out_dir sft_model_medium \
  --batch_size 8 \
  --epochs 20 \
  --num_workers 0 \
  --log_interval 10 \
  --save_interval 100 \
  --learning_rate 2e-4 \
  --dim 256 \
  --n_layers 4 \
  --n_heads 8 \
  --n_kv_heads 4 \
  --max_seq_len 256
```

训练结果：

```text
Loaded pretrain checkpoint from ./base_model_medium/pretrain_256_4_6144.pth
Small LLM parameters: 4.524 M
Epoch 1:  loss 8.847 -> 6.254
Epoch 6:  loss 2.595 -> 3.034
Epoch 10: loss 1.712
Epoch 15: loss 0.729
Epoch 20: loss 1.090
Saved SFT checkpoint to sft_model_medium/sft_dim256_layers4_vocab_size6144.pth
```

---

## 七、中等小模型推理测试

### 7.1 推理命令

`model_sample_small.py` 已支持通过命令行传入模型结构参数。

PowerShell 单行命令：

```powershell
cd E:\Project\LLM\LearnLLM\resources\happy-llm\source\docs\chapter5\code; python model_sample_small.py --checkpoint ./sft_model_medium/sft_dim256_layers4_vocab_size6144.pth --dim 256 --n_layers 4 --n_heads 8 --n_kv_heads 4 --prompt "SFT是什么意思？" --max_new_tokens 40 --temperature 0.1 --top_k 10
```

Bash / Claude Code 命令：

```bash
python model_sample_small.py \
  --checkpoint ./sft_model_medium/sft_dim256_layers4_vocab_size6144.pth \
  --dim 256 \
  --n_layers 4 \
  --n_heads 8 \
  --n_kv_heads 4 \
  --prompt "SFT是什么意思？" \
  --max_new_tokens 40 \
  --temperature 0.1 \
  --top_k 10
```

### 7.2 实际推理结果

问题：

```text
SFT是什么意思？
```

输出：

```text
SFT是监督微调,使用人工整理的指令和回答数据继续训练模型,让模型更会按指令回答。。
```

问题：

```text
embedding层的作用是什么？
```

输出：

```text
embedding层把离散的token id映射成连续向量,供Transformer后续计算。。
```

问题：

```text
为什么当前模型输出乱码？
```

输出：

```text
主要因为训练数据和训练步数太少,模型还没有学到稳定的语言生成能力。
```

问题：

```text
CPU可以训练大语言模型吗？
```

输出：

```text
CPU可以跑小规模教学训练,但速度很慢,正式训练通常需要GPU。
```

结论：中等小模型已经可以回答部分简单 LearnLLM 问题，前半句通常是正确且可读的。

---

## 八、模型效果对比

| 模型 | 参数量 | 数据 | 结果 |
|---|---:|---|---|
| 默认 215M 模型 + 极少数据 | 215M | 15 条预训练 + 少量 SFT | 流程跑通，但输出多为乱码和重复 |
| small 模型 | 1.18M | 920 条预训练 + 111 条 SFT | 学到主题词，但重复严重 |
| medium 模型 | 4.52M | 920 条预训练 + 111 条 SFT | 能回答部分简单问题，效果明显最好 |

当前最推荐使用：

```text
sft_model_medium/sft_dim256_layers4_vocab_size6144.pth
```

---

## 九、经验总结

### 9.1 为什么中等小模型比 215M 更好？

虽然 215M 参数更多，但它在当前实验中只有极少预训练数据，基础语言能力不足。大模型需要更多数据才能学好。

中等小模型参数更少，数据量相对更充足，更容易在 CPU 环境下训练到可见效果。

### 9.2 为什么不能只增加 SFT？

SFT 主要学习“如何回答”，不是从零学习语言本身。如果 base model 不会稳定生成中文，SFT 很难让它变成可用聊天模型。

因此需要：

```text
先预训练学语言模式，再 SFT 学问答格式
```

### 9.3 为什么推理要短输出？

当前模型仍然很小，长文本生成容易发散或重复。推荐使用：

```text
max_new_tokens=30-50
temperature=0.0-0.2
top_k=5-10
```

### 9.4 当前模型适合什么？

适合：

- LLM 训练流程演示
- tokenizer -> pretrain -> SFT -> inference 全链路学习
- 固定主题 LearnLLM 问答
- 小模型训练实验

不适合：

- 通用聊天
- 开放域问答
- 复杂推理
- 生产环境使用

---

## 十、后续优化方向

如果继续提升效果，可以尝试：

1. 扩充 SFT 数据到 `300-500` 条。
2. 增加更多不同问法，减少模板化重复。
3. 答案尽量短，控制在 1-2 句话。
4. 加入验证集，观察是否过拟合。
5. 尝试 `dim=384, n_layers=6`，但 CPU 训练会更慢。
6. 在推理中加入 repetition penalty，减少重复输出。
7. 增加预训练数据多样性，不只使用合成文本。

当前阶段的最佳实用命令：

```powershell
cd E:\Project\LLM\LearnLLM\resources\happy-llm\source\docs\chapter5\code; python model_sample_small.py --checkpoint ./sft_model_medium/sft_dim256_layers4_vocab_size6144.pth --dim 256 --n_layers 4 --n_heads 8 --n_kv_heads 4 --prompt "SFT是什么意思？" --max_new_tokens 40 --temperature 0.1 --top_k 10
```
