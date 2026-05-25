# C3 PyTorch: Advanced Architectures and Deployment — 课程总览

> **DeepLearning.AI — PyTorch for Deep Learning Professional Certificate**  
> 第三门课：高级架构设计、模型可解释性与生产部署

---

## 课程目标

```
学完本课你能做到：

  ✅ 设计超越 Sequential 的自定义复杂架构（Siamese / ResNet / DenseNet）
  ✅ 从零实现 Transformer 架构（Self-Attention → 多头 → 完整 Block）
  ✅ 理解扩散模型的前向加噪与反向去噪过程
  ✅ 用 Grad-CAM / Saliency Map 可视化模型决策依据
  ✅ 用 ONNX 导出跨框架模型
  ✅ 用 MLflow 管理实验与模型版本
  ✅ 用量化和剪枝压缩模型，准备部署
```

---

## 章节导航

| 章节 | 核心内容 | 关键 API / 工具 |
|------|---------|---------------|
| 第1章 | 自定义高级架构 | `nn.ModuleList` / 多输入输出 / 参数共享 |
| 第2章 | 经典深度架构 | Siamese Network / ResNet / DenseNet |
| 第3章 | 计算机视觉进阶 | CNN 感受野 / 特征提取 |
| 第4章 | 模型可解释性 | Saliency Maps / Grad-CAM |
| 第5章 | Transformer 架构 | Self-Attention / 多头注意力 / 位置编码 |
| 第6章 | 扩散模型 | 前向加噪 / U-Net 去噪 |
| 第7章 | 模型部署 | ONNX / MLflow / 量化 / 剪枝 |

---

## 各章核心知识点

---

### 第1章：自定义高级架构

```python
# 动态层数网络（用 nn.ModuleList）
class DynamicNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super().__init__()
        # 用 ModuleList 存储可变数量的层
        self.layers = nn.ModuleList(
            [nn.Linear(input_size if i == 0 else hidden_size, hidden_size)
             for i in range(num_layers)]
        )
        self.output = nn.Linear(hidden_size, output_size)
        self.relu   = nn.ReLU()

    def forward(self, x):
        for layer in self.layers:
            x = self.relu(layer(x))
        return self.output(x)

# 参数共享（Siamese 网络的基础）
class SharedEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(   # 只定义一次
            nn.Linear(784, 256), nn.ReLU(),
            nn.Linear(256, 128)
        )

    def forward(self, x1, x2):
        # 两个输入共享同一个 encoder 的参数
        return self.encoder(x1), self.encoder(x2)
```

```
nn.ModuleList vs nn.Sequential vs Python list：

  Python list [layer1, layer2]：
    ❌ PyTorch 不知道这些层的存在
    ❌ 参数不会被优化器追踪

  nn.ModuleList([layer1, layer2])：
    ✅ PyTorch 追踪所有子层参数
    ✅ 但不定义前向传播顺序（需在 forward 手动遍历）

  nn.Sequential(layer1, layer2)：
    ✅ 自动按顺序执行
    ❌ 不灵活（只能线性堆叠）
```

---

### 第2章：经典深度架构

#### Siamese Network（孪生网络）

```python
class SiameseNet(nn.Module):
    """判断两个输入是否相似（如人脸验证）"""
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(784, 256), nn.ReLU(),
            nn.Linear(256, 128)
        )

    def forward(self, x1, x2):
        z1 = self.encoder(x1)   # 编码为向量
        z2 = self.encoder(x2)
        # 计算 L2 距离（越小越相似）
        dist = torch.norm(z1 - z2, dim=1)
        return dist

# 训练用 Contrastive Loss
# 相似对：拉近距离；不相似对：推远距离
```

```
Siamese 网络的用途：
  人脸验证（是否同一个人）
  签名验证（是否真实签名）
  图片相似度搜索
  少样本学习（few-shot learning）

架构特点：
  x1 ──→ [Encoder] ──→ z1 ──┐
          ↑ 共享参数          ├──→ 距离/相似度
  x2 ──→ [Encoder] ──→ z2 ──┘
```

---

#### ResNet（残差网络）

```python
class ResidualBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, 1, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(out_ch)
        self.relu  = nn.ReLU()

        # shortcut：shape 不匹配时用 1×1 卷积对齐
        self.shortcut = nn.Sequential()
        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride, bias=False),
                nn.BatchNorm2d(out_ch)
            )

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + self.shortcut(x)    # ← 残差相加：F(x) + x
        return self.relu(out)
```

```
残差连接为什么有效：

  普通网络：  x → [Block] → y = F(x)
  ResNet：    x → [Block] → F(x)
              x ──────────→ +  → y = F(x) + x

  反向传播时梯度：
    ∂Loss/∂x = ∂Loss/∂y × (∂F(x)/∂x + 1)
                                         ↑
                              即使 ∂F(x)/∂x ≈ 0（梯度消失）
                              梯度仍能通过 +1 完整传回

  → 解决了深层网络的梯度消失问题
  → 网络可以堆到 100+ 层

shape 变化（stride=2 降采样）：
  输入 [B, 64, 56, 56]
  主路径 conv(stride=2) → [B, 128, 28, 28]
  shortcut 1×1 conv(stride=2) → [B, 128, 28, 28]
  相加 → [B, 128, 28, 28]
```

---

#### DenseNet（密集连接网络）

```python
class DenseBlock(nn.Module):
    """每一层接收前面所有层的输出（特征复用）"""
    def __init__(self, in_channels, growth_rate, num_layers):
        super().__init__()
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            self.layers.append(
                nn.Sequential(
                    nn.BatchNorm2d(in_channels + i * growth_rate),
                    nn.ReLU(),
                    nn.Conv2d(in_channels + i * growth_rate,
                              growth_rate, 3, padding=1, bias=False)
                )
            )

    def forward(self, x):
        features = [x]
        for layer in self.layers:
            new_feat = layer(torch.cat(features, dim=1))  # 拼接所有历史特征
            features.append(new_feat)
        return torch.cat(features, dim=1)
```

```
DenseNet vs ResNet：

  ResNet：  x → Block → F(x) + x       （加法，shape 不变）
  DenseNet：x → Block → cat([x, F(x)])  （拼接，通道数增加）

  DenseNet 每层都能看到之前所有层的特征：
    Layer0 输入: [x]            通道数: C
    Layer1 输入: [x, F0(x)]     通道数: C + k
    Layer2 输入: [x, F0, F1]    通道数: C + 2k
    ...
    k = growth_rate（每层增长的通道数）

  优点：特征复用，参数效率更高
  缺点：内存占用大（需要存储所有中间特征）
```

---

### 第4章：模型可解释性

#### Saliency Map（显著图）

```python
def compute_saliency(model, x, target_class):
    """计算输入图像哪些像素对预测最重要"""
    x = x.unsqueeze(0).requires_grad_(True)

    output = model(x)
    loss = output[0, target_class]
    loss.backward()

    # 梯度的绝对值即为显著性
    saliency = x.grad.abs().squeeze()
    saliency = saliency.max(dim=0).values   # 取 RGB 三通道最大值
    return saliency
```

```
Saliency Map 原理：

  对输入像素求梯度：∂output / ∂input_pixel

  梯度大的像素 → 对预测影响大 → 显示为亮色
  梯度小的像素 → 对预测影响小 → 显示为暗色

  结果：一张和原图同尺寸的热力图
        告诉你"模型在看哪里"
```

---

#### Grad-CAM（梯度类激活图）

```python
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.gradients = None
        self.activations = None

        # 注册钩子：记录前向激活和反向梯度
        target_layer.register_forward_hook(
            lambda m, i, o: setattr(self, 'activations', o)
        )
        target_layer.register_backward_hook(
            lambda m, gi, go: setattr(self, 'gradients', go[0])
        )

    def generate(self, x, target_class):
        output = self.model(x)
        self.model.zero_grad()
        output[0, target_class].backward()

        # 梯度全局平均池化 → 权重
        weights = self.gradients.mean(dim=[2, 3], keepdim=True)  # [B, C, 1, 1]

        # 加权激活图
        cam = (weights * self.activations).sum(dim=1)  # [B, H, W]
        cam = torch.relu(cam)                          # 只看正激活
        return cam
```

```
Saliency Map vs Grad-CAM：

  Saliency Map：              Grad-CAM：
    对输入像素求梯度              对中间特征图求梯度
    分辨率 = 原图分辨率            分辨率 = 特征图分辨率（较低）
    噪点多，边界清晰               平滑，突出高层语义区域
    像素级定位                    区域级定位

  用途：
    Saliency: 看哪些像素触发了预测
    Grad-CAM: 看图像哪个区域是模型关注的（更常用）
```

---

### 第5章：Transformer 架构

```python
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.d_model  = d_model
        self.n_heads  = n_heads
        self.d_k      = d_model // n_heads  # 每个头的维度

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

    def forward(self, x, mask=None):
        B, T, C = x.shape

        # 投影并拆分多头 [B, T, C] → [B, n_heads, T, d_k]
        Q = self.W_q(x).view(B, T, self.n_heads, self.d_k).transpose(1, 2)
        K = self.W_k(x).view(B, T, self.n_heads, self.d_k).transpose(1, 2)
        V = self.W_v(x).view(B, T, self.n_heads, self.d_k).transpose(1, 2)

        # 注意力分数
        scores = Q @ K.transpose(-2, -1) / (self.d_k ** 0.5)  # [B, h, T, T]
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        attn = scores.softmax(dim=-1)

        # 加权求和 + 合并多头
        out = (attn @ V).transpose(1, 2).contiguous().view(B, T, C)
        return self.W_o(out)


class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        self.attn  = MultiHeadAttention(d_model, n_heads)
        self.ff    = nn.Sequential(
            nn.Linear(d_model, d_ff), nn.GELU(),
            nn.Linear(d_ff, d_model)
        )
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # Pre-Norm + 残差连接
        x = x + self.drop(self.attn(self.ln1(x), mask))
        x = x + self.drop(self.ff(self.ln2(x)))
        return x
```

```
Transformer Block 数据流：

  输入 x: [B, T, d_model]
     ↓ LayerNorm
     ↓ MultiHeadAttention（Q K V 计算）
     ↓ Dropout
     + x（残差）
  中间 x: [B, T, d_model]
     ↓ LayerNorm
     ↓ FFN（Linear → GELU → Linear，维度先扩4倍再缩回）
     ↓ Dropout
     + x（残差）
  输出 x: [B, T, d_model]   ← shape 不变

Self-Attention 中 Q/K/V 的含义：
  Q（Query）：当前 token 想找什么信息？
  K（Key）：  每个 token 有什么信息可被找到？
  V（Value）：实际传递出去的内容是什么？

  注意力分数 = Q @ K^T / √d_k
  → Softmax → 对 V 加权求和
```

---

### 第7章：模型部署

#### ONNX 导出

```python
import torch.onnx

model.eval()
dummy_input = torch.rand(1, 3, 224, 224)  # 示例输入

torch.onnx.export(
    model,
    dummy_input,
    'model.onnx',
    input_names=['input'],
    output_names=['output'],
    dynamic_axes={'input': {0: 'batch_size'},  # 支持动态 batch
                  'output': {0: 'batch_size'}},
    opset_version=17
)
```

```
ONNX 的作用：

  PyTorch 模型  →  ONNX 格式  →  任意框架/设备推理
                                   ├── TensorRT（NVIDIA GPU）
                                   ├── OpenVINO（Intel CPU）
                                   ├── ONNX Runtime（通用）
                                   └── 移动端（CoreML/TFLite转换）
```

---

#### 量化（Quantization）

```python
# 动态量化（最简单）：将 Linear 层权重从 float32 → int8
quantized_model = torch.quantization.quantize_dynamic(
    model,
    {nn.Linear},       # 量化哪些层
    dtype=torch.qint8  # 目标精度
)

# 效果：
#   模型大小减少 75%（float32→int8，4倍压缩）
#   推理速度提升 2~4倍
#   精度损失通常 < 1%
```

```
精度 vs 速度 权衡：

  float32：32位，精度最高，速度最慢，显存最大
  float16：16位，速度提升约2倍，精度轻微损失
  int8：    8位，速度提升约4倍，精度损失<1%（通常可接受）
  int4：    4位，速度提升约8倍，精度损失较明显

常用场景：
  训练阶段：float32 / bfloat16
  推理部署：int8（服务器）/ int4（移动端）
```

---

#### 剪枝（Pruning）

```python
import torch.nn.utils.prune as prune

# 对 Linear 层按 L1 范数剪枝 30% 的权重
prune.l1_unstructured(model.fc1, name='weight', amount=0.3)

# 永久化剪枝（移除 mask，真正删除权重）
prune.remove(model.fc1, 'weight')

# 查看稀疏度
total  = model.fc1.weight.numel()
zeros  = (model.fc1.weight == 0).sum().item()
print(f"稀疏度: {zeros/total:.1%}")
```

```
剪枝的直觉：

  神经网络中很多权重接近 0，对输出贡献极小
  剪掉这些权重 → 模型更小更快

  结构化剪枝 vs 非结构化剪枝：
    非结构化：随机置零单个权重（稀疏矩阵，硬件加速有限）
    结构化：  整行/列/通道置零（真正减少计算量，硬件友好）
```

---

## 三门课知识体系总览

```
C1 — 基础
  Tensor / Autograd / nn.Module / Loss / Optimizer / DataLoader
  → 能搭、能跑

C2 — 进阶
  评估指标 / LR调度 / 灵活架构 / Optuna / 数据增强 / 迁移学习 / TensorBoard
  → 能跑好

C3 — 高级
  Siamese / ResNet / DenseNet / Transformer / 扩散模型
  可解释性 / ONNX / 量化 / 剪枝
  → 能上线
```

---

## 常见问题速查

| 问题 | 解决方案 |
|------|---------|
| 深层网络 loss 不降（退化问题） | 加残差连接（ResNet Block） |
| 模型太大，部署慢 | 量化（int8）+ 剪枝（prune 30~50%） |
| 不知道模型在"看"哪里 | Grad-CAM 可视化 |
| 需要跨框架部署 | 导出 ONNX |
| 训练数据少，需要做相似度任务 | Siamese Network |
| 需要理解序列/文本 | Transformer Block |
| 实验太多，管不住 | MLflow 记录所有实验 |

---

## 完整学习路径回顾

```
C1 PyTorch Fundamentals
  Tensor → Autograd → nn.Module → Loss → DataLoader → 训练循环
      ↓
C2 Techniques & Ecosystem Tools
  评估 → LR调度 → 灵活架构 → 超参搜索 → 数据增强 → 迁移学习 → 监控
      ↓
C3 Advanced Architectures & Deployment
  Siamese → ResNet → DenseNet → Transformer → 扩散模型
  Grad-CAM → ONNX → MLflow → 量化 → 剪枝
      ↓
可以开始学习：
  LLM（nanochat / nanoGPT）
  更复杂的CV/NLP任务
  生产级 ML 系统设计
```
