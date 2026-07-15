# 第五章代码运行指南与 Embedding 原理

> 本文整理自 happy-llm 第五章代码的实际运行经验，包含环境搭建、训练流程、以及 Embedding 层和语义学习的核心原理。

---

## 一、代码结构总览

```
chapter5/code/
├── train_tokenizer.py      # 训练自定义 BPE tokenizer
├── deal_dataset.py         # 处理预训练和 SFT 数据集
├── download_dataset.sh     # Linux 数据下载脚本
├── windows_download_dataset.sh  # Windows 数据下载脚本
├── k_model.py              # 模型定义（Transformer，215M 参数）
├── dataset.py              # 数据集类（PretrainDataset / SFTDataset）
├── ddp_pretrain.py         # 预训练脚本
├── ddp_sft_full.py         # SFT 微调脚本
├── model_sample.py         # 推理采样脚本
├── export_model.py         # 导出模型到 HuggingFace 格式
└── tokenizer_k/            # 训练好的 tokenizer（需先生成）
```

## 二、环境搭建

### 2.1 依赖安装

```bash
cd resources/happy-llm/source/docs/chapter5/code
pip install -r requirements.txt
```

实际安装中遇到的问题及解决方案：

| 问题 | 解决方案 |
|------|---------|
| Python 3.14 兼容性 | torch 自动安装了 2.12.1+cpu 版本，兼容 |
| tokenizers 版本冲突 | transformers 要求 tokenizers>=0.22.0,<=0.23.0 |
| datasets 安装超时 | 使用 `pip install --no-deps datasets` 先装包 |

实际安装的核心包版本：

```
torch          2.12.1+cpu
transformers   5.12.1
tokenizers     0.22.2
pandas         3.0.3
numpy          2.4.4
tiktoken       0.13.0
swanlab        0.8.4
```

### 2.2 GPU 环境

- GT 730（2GB）：只能用 CPU 训练
- 8GB 显存：batch_size=4-8，增大 accumulation_steps
- 12-16GB 显存：默认参数可用
- 24GB+ 显存：无压力

## 三、运行流程

### 步骤 1：下载数据集

```powershell
# Windows PowerShell
$env:HF_ENDPOINT = "https://hf-mirror.com"
$dataset_dir = "你的数据目录"

# 预训练数据
modelscope download --dataset ddzhu123/seq-monkey mobvoi_seq_monkey_general_open_corpus.jsonl.tar.bz2 --local_dir "$dataset_dir"
tar -xvf "$dataset_dir\mobvoi_seq_monkey_general_open_corpus.jsonl.tar.bz2" -C "$dataset_dir"

# SFT 数据
huggingface-cli download --repo-type dataset --resume-download BelleGroup/train_3.5M_CN --local-dir "$dataset_dir\BelleGroup"
```

### 步骤 2：处理数据集

编辑 `deal_dataset.py` 中的路径后运行：

```bash
python deal_dataset.py
# 产出: seq_monkey_datawhale.jsonl, BelleGroup_sft.jsonl
```

### 步骤 3：训练 Tokenizer

```bash
python train_tokenizer.py
# 产出: tokenizer_k/ 目录
```

### 步骤 4：预训练

```bash
python ddp_pretrain.py \
  --data_path ./seq_monkey_datawhale.jsonl \
  --batch_size 8 \
  --gpus 0 \
  --num_workers 2
# 产出: base_model_215M/pretrain_1024_18_6144.pth
```

### 步骤 5：SFT 微调

```bash
python ddp_sft_full.py \
  --data_path ./BelleGroup_sft.jsonl \
  --batch_size 8 \
  --gpus 0 \
  --num_workers 2
# 产出: sft_model_215M/sft_dim1024_layers18_vocab_size6144.pth
```

### 步骤 6：推理测试

```bash
python model_sample.py
```

## 四、小批量快速验证

用 15 条样本 + 缩小模型（dim=128, n_layers=2, 1.18M 参数）在 CPU 上验证：

```
Epoch:[1/3](0/8) loss:8.736
Epoch:[2/3](0/8) loss:8.636
Epoch:[3/3](6/8) loss:8.196  ← loss 正常收敛
```

## 五、nn.Embedding 原理

### 5.1 本质：按索引查表

`nn.Embedding(vocab_size, dim)` 本质是一个权重矩阵，形状为 `[vocab_size, dim]`。

```python
import torch
import torch.nn as nn

emb = nn.Embedding(6, 4)

# 权重矩阵（随机初始化）：
# 索引:  0       1       2       3       4       5
# [[-0.12,  0.34,  0.56, -0.78],   ← token 0
#  [ 0.91, -0.23,  0.45,  0.67],   ← token 1
#  [-0.34,  0.12,  0.89, -0.56],   ← token 2
#  [ 0.78, -0.45,  0.23,  0.91],   ← token 3
#  [-0.67,  0.56, -0.12,  0.34],   ← token 4
#  [ 0.45, -0.89,  0.67, -0.23]]   ← token 5
```

### 5.2 查找过程

```python
input_ids = torch.tensor([3, 1, 4])
output = emb(input_ids)

# 等价于 emb.weight[input_ids]，直接按索引取行：
# token 3 → [0.78, -0.45, 0.23, 0.91]
# token 1 → [0.91, -0.23, 0.45, 0.67]
# token 4 → [-0.67, 0.56, -0.12, 0.34]
#
# output shape: [3, 4]  (3个token, 每个4维)
```

### 5.3 批量输入

```python
batch = torch.tensor([[3, 1, 4],    # 样本1
                      [2, 0, 5]])   # 样本2
output = emb(batch)
# shape: [2, 3, 4]  (2个样本, 每个3个token, 每个4维)
```

### 5.4 在模型中的角色

```
输入 token IDs          Embedding 层           输出向量
[3, 1, 4]      →    查表 emb.weight     →   [[0.78, -0.45, 0.23, 0.91],
                                              [0.91, -0.23, 0.45, 0.67],
                                              [-0.67, 0.56, -0.12, 0.34]]
```

`emb.weight` 是可学习参数，训练过程中通过反向传播不断更新。

## 六、语义相近的 token 如何学到相近的向量

### 6.1 核心机制：共享上下文 → 共享梯度方向

模型不会直接"理解语义"，而是通过预测任务间接学习。

训练数据中有大量这样的句子：

```
"我 喜欢 吃 苹果"
"我 喜欢 吃 香蕉"
"我 喜欢 吃 西瓜"
"我 喜欢 吃 橘子"
```

模型要学的是：给定 `"我 喜欢 吃 ___"`，预测下一个词。

### 6.2 反向传播过程

```
输入: [我, 喜欢, 吃]  →  预测: 苹果
输入: [我, 喜欢, 吃]  →  预测: 香蕉
输入: [我, 喜欢, 吃]  →  预测: 西瓜
```

梯度更新：

```
预测 "苹果" 时:
  loss = -log P(苹果 | 上下文)
  梯度 ∂loss/∂e_苹果 → 让 e_苹果 更接近 "食物" 的模式

预测 "香蕉" 时:
  loss = -log P(香蕉 | 上下文)
  梯度 ∂loss/∂e_香蕉 → 让 e_香蕉 也更接近 "食物" 的模式
```

大量训练后，embedding 空间自动涌现语义聚类：

```
苹果 ≈ [0.8, 0.2, -0.1, 0.5]   ← 都在 "食物" 区域
香蕉 ≈ [0.7, 0.3, -0.2, 0.6]   ← 都在 "食物" 区域
桌子 ≈ [-0.5, 0.9, 0.4, -0.3]  ← 在 "家具" 区域，远离食物
```

### 6.3 共享上下文导致向量趋同的示意

```
           相同上下文
              ↓
"苹果" ← [我, 喜欢, 吃, __] → "香蕉"
              ↓
         相似的梯度方向
              ↓
       embedding 趋向相近
```

这就是 Word2Vec 的经典发现：**"You shall know a word by the company it keeps"**。

### 6.4 Transformer 中的额外机制

embedding 不仅受预测任务影响，还受 self-attention 影响：

```
输入: "猫 坐在 垫子 上"
       ↓
  self-attention 让 "猫" 和 "垫子" 交互
       ↓
  梯度回传到 embedding
       ↓
  "猫" 和 "垫子" 的 embedding 被间接拉近
  （因为它们经常在相似语境中共现）
```

### 6.5 总结

| 机制 | 作用 |
|------|------|
| 相同上下文 | 语义相近的词出现在相似位置 |
| 共享梯度方向 | 相同上下文产生相似的梯度更新 |
| 预测任务 | 迫使模型区分不同语义的词 |
| Self-attention | 让经常共现的词互相影响 |

**没有显式的"语义相似度损失函数"**，完全靠预测任务 + 大量数据，自动涌现出语义聚类。

---

## 七、时间与资源估算

| 硬件 | 预训练 1 epoch | SFT 1 epoch |
|------|---------------|-------------|
| 单卡 RTX 3090/4090 | ~10-20 小时 | ~10-20 小时 |
| 单卡 V100/A100 | ~6-15 小时 | ~6-15 小时 |
| CPU | 数天 | 数天 |

显存需求（默认 batch_size=64, seq_len=512, bf16）：约 **8-10 GB**。
