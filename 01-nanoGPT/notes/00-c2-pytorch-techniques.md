# C2 PyTorch: Techniques and Ecosystem Tools — 课程总览

> **DeepLearning.AI — PyTorch for Deep Learning Professional Certificate**  
> 第二门课：进阶技术与生态工具，让模型更好、更快、更可靠

---

## 课程目标

```
学完本课你能做到：

  ✅ 用评估指标（Precision/Recall/F1）全面衡量模型性能
  ✅ 使用学习率调度策略让训练更稳定收敛
  ✅ 设计灵活的多输入/多输出网络架构
  ✅ 用 Optuna 自动搜索最优超参数
  ✅ 用数据增强提升模型泛化能力
  ✅ 用迁移学习快速解决新任务
  ✅ 用 TensorBoard 可视化训练过程
```

---

## 章节导航

| 章节 | 文件 | 核心内容 | 关键工具/API |
|------|------|---------|------------|
| 第1章 | [c2-01-evaluation.md](./c2-01-evaluation.md) | 模型评估指标 | Accuracy / Precision / Recall / F1 |
| 第2章 | [c2-02-lr-schedule.md](./c2-02-lr-schedule.md) | 学习率调度 | `StepLR` / `CosineAnnealingLR` / `OneCycleLR` |
| 第3章 | [c2-03-flexible-arch.md](./c2-03-flexible-arch.md) | 灵活架构设计 | 多输入/多输出 / 残差连接 / 分支网络 |
| 第4章 | [c2-04-optuna.md](./c2-04-optuna.md) | 超参数调优 | `optuna` / `trial.suggest_*` |
| 第5章 | [c2-05-augmentation.md](./c2-05-augmentation.md) | 数据增强 | `transforms` / `RandomCrop` / `ColorJitter` |
| 第6章 | [c2-06-transfer-learning.md](./c2-06-transfer-learning.md) | 迁移学习 | `torchvision.models` / 冻结参数 / 微调 |
| 第7章 | [c2-07-tensorboard.md](./c2-07-tensorboard.md) | 训练监控 | `SummaryWriter` / loss/acc 可视化 |

---

## 各章核心知识点

---

### 第1章：模型评估指标

```python
from sklearn.metrics import classification_report
import torch

# 在评估循环中收集预测结果
all_preds, all_labels = [], []
model.eval()
with torch.no_grad():
    for X, y in val_loader:
        preds = model(X).argmax(dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(y.cpu().numpy())

print(classification_report(all_labels, all_preds))
```

```
四个核心指标（以二分类为例）：

  真实\预测    正(1)    负(0)
  ─────────────────────────
  正(1)        TP        FN
  负(0)        FP        TN

  Accuracy  = (TP+TN) / 全部样本     总体正确率
  Precision = TP / (TP+FP)          预测为正中真正为正的比例（查准率）
  Recall    = TP / (TP+FN)          真正为正中被预测到的比例（查全率）
  F1        = 2 × P×R / (P+R)       P和R的调和平均

什么时候用哪个：
  垃圾邮件检测 → 高 Precision（不能误判正常邮件）
  癌症筛查     → 高 Recall（不能漏诊）
  通用场景     → F1（P和R的均衡）
  类别不均衡   → 不用 Accuracy，用 F1
```

---

### 第2章：学习率调度（LR Scheduler）

```python
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

# 方式1：StepLR — 每隔 step_size 个 epoch，lr × gamma
scheduler = torch.optim.lr_scheduler.StepLR(
    optimizer, step_size=10, gamma=0.5
)

# 方式2：CosineAnnealingLR — 余弦衰减到 eta_min
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=50, eta_min=1e-6
)

# 方式3：OneCycleLR — 先升后降（最常用）
scheduler = torch.optim.lr_scheduler.OneCycleLR(
    optimizer, max_lr=1e-2,
    steps_per_epoch=len(train_loader), epochs=30
)

# 训练循环中调用
for epoch in range(num_epochs):
    train(...)
    scheduler.step()       # 每个 epoch 结束后调用
    # OneCycleLR 在每个 batch 后调用
```

```
各调度策略的 lr 曲线：

lr ↑
   │  StepLR:             CosineAnnealingLR:      OneCycleLR:
   │  ───────             ─────────────────        ──────────
   │  1e-3 ──┐            1e-3 ╲                   1e-3    ╱╲
   │         └──┐               ╲___╱‾╲___          ╱    ╲╱
   │            └── ...               ╲___         ╱
   │  每10epoch降半         余弦波动衰减   先升后降（最佳）
   └──────────────── epoch →
```

---

### 第3章：灵活架构设计

```python
# 多输入网络（图像 + 文本特征）
class MultiInputNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.img_branch  = nn.Sequential(nn.Linear(512, 256), nn.ReLU())
        self.text_branch = nn.Sequential(nn.Linear(128, 256), nn.ReLU())
        self.classifier  = nn.Linear(512, 10)

    def forward(self, img_feat, text_feat):
        img_out  = self.img_branch(img_feat)    # [B, 256]
        text_out = self.text_branch(text_feat)  # [B, 256]
        combined = torch.cat([img_out, text_out], dim=1)  # [B, 512]
        return self.classifier(combined)        # [B, 10]

# 多输出网络（同时预测类别和边界框）
class MultiOutputNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone   = nn.Linear(512, 256)
        self.cls_head   = nn.Linear(256, 10)    # 分类头
        self.bbox_head  = nn.Linear(256, 4)     # 回归头

    def forward(self, x):
        feat = torch.relu(self.backbone(x))     # [B, 256]
        return self.cls_head(feat), self.bbox_head(feat)  # ([B,10], [B,4])
```

```
架构设计模式：

  单输入单输出（基础）：  x ──→ [网络] ──→ y

  多输入：              img ──┐
                              ├──→ [concat] ──→ [网络] ──→ y
                       text ──┘

  多输出：              x ──→ [backbone] ──→ feat ──┬──→ [cls_head]  ──→ 类别
                                                    └──→ [bbox_head] ──→ 坐标

  残差连接：            x ──┬──→ [F(x)] ──→ + ──→ out
                            └────────────────┘  (skip connection)
```

---

### 第4章：超参数调优（Optuna）

```python
import optuna

def objective(trial):
    # 定义搜索空间
    lr          = trial.suggest_float('lr', 1e-5, 1e-2, log=True)
    hidden_size = trial.suggest_categorical('hidden', [64, 128, 256, 512])
    dropout     = trial.suggest_float('dropout', 0.1, 0.5)
    num_layers  = trial.suggest_int('num_layers', 1, 4)

    # 构建并训练模型
    model = build_model(hidden_size, dropout, num_layers)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    val_acc = train_and_evaluate(model, optimizer)

    return val_acc   # Optuna 会最大化这个值

# 运行搜索
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=100)

print(study.best_params)   # 最优超参数
```

```
Optuna 的搜索方法对比：

  Grid Search（网格搜索）：
    枚举所有组合
    3个参数各4个值 → 4³=64次实验
    太慢，不实用

  Random Search（随机搜索）：
    随机采样组合
    比 Grid 快，但不利用历史信息

  Optuna（贝叶斯优化）：
    根据之前的结果推断哪里可能更好
    越搜越聪明，通常 50~100 次就够

suggest 方法速查：
  suggest_float(name, low, high)           连续浮点数
  suggest_float(name, low, high, log=True) 对数尺度（适合 lr）
  suggest_int(name, low, high)             整数
  suggest_categorical(name, choices)       离散选项
```

---

### 第5章：数据增强（Data Augmentation）

```python
from torchvision import transforms

# 训练集：加入随机变换（扩展数据多样性）
train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),  # 随机水平翻转
    transforms.RandomCrop(32, padding=4),    # 随机裁剪
    transforms.ColorJitter(                  # 颜色抖动
        brightness=0.2, contrast=0.2,
        saturation=0.2, hue=0.1
    ),
    transforms.RandomRotation(15),           # 随机旋转 ±15°
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# 验证集：只做必要的标准化，不做随机变换！
val_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])
```

```
常用增强方法速查：

  几何变换：
    RandomHorizontalFlip(p)   水平翻转（适合：自然图像）
    RandomVerticalFlip(p)     垂直翻转（适合：医学图像、卫星图）
    RandomRotation(degrees)   随机旋转
    RandomCrop(size, padding) 随机裁剪
    RandomResizedCrop(size)   随机缩放裁剪

  颜色变换：
    ColorJitter(...)           亮度/对比度/饱和度/色调
    Grayscale(p)               随机转灰度
    GaussianBlur(kernel_size)  高斯模糊

  原则：
    ✅ 训练集才做随机增强
    ❌ 验证/测试集只做 Resize + Normalize
    ✅ 增强要符合真实场景（文字识别不能翻转）
```

---

### 第6章：迁移学习（Transfer Learning）

```python
import torchvision.models as models

# 加载预训练模型（ImageNet 权重）
backbone = models.resnet50(weights='IMAGENET1K_V1')

# 策略1：特征提取（冻结所有层，只训练新头）
for param in backbone.parameters():
    param.requires_grad = False

# 替换最后的分类头（原来是1000类，改为10类）
backbone.fc = nn.Linear(backbone.fc.in_features, 10)
# 只有 backbone.fc 的参数 requires_grad=True

# 策略2：微调（解冻部分层）
for param in backbone.layer4.parameters():   # 解冻最后一个 stage
    param.requires_grad = True

# 策略2 配合分层学习率
optimizer = torch.optim.AdamW([
    {'params': backbone.layer4.parameters(), 'lr': 1e-4},  # 小lr
    {'params': backbone.fc.parameters(),     'lr': 1e-3},  # 大lr
])
```

```
迁移学习策略选择：

  数据量少 + 新任务和预训练任务相似：
    → 特征提取（冻结所有，只训新头）
    → 几百张图也能有好效果

  数据量中等 + 任务有差异：
    → 微调后几层 + 训新头
    → 解冻 layer3/layer4 + 新分类头

  数据量大 + 任务差异大：
    → 全量微调（所有层都训练）
    → 预训练权重只作为初始化

  常用预训练模型：
    ResNet50/101   通用图像分类
    EfficientNet   轻量高效
    ViT            Transformer 视觉模型
    BERT/GPT       文本任务
```

---

### 第7章：TensorBoard 训练监控

```python
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter(log_dir='runs/exp1')

for epoch in range(num_epochs):
    train_loss = train(...)
    val_acc    = evaluate(...)

    # 记录标量
    writer.add_scalar('Loss/train', train_loss, epoch)
    writer.add_scalar('Accuracy/val', val_acc, epoch)

    # 记录学习率
    writer.add_scalar('LR', optimizer.param_groups[0]['lr'], epoch)

    # 记录权重分布（调试用）
    for name, param in model.named_parameters():
        writer.add_histogram(name, param, epoch)

writer.close()

# 终端启动 TensorBoard：
# tensorboard --logdir=runs
# 浏览器打开 http://localhost:6006
```

```
SummaryWriter 常用方法：

  add_scalar(tag, value, step)       记录单个数值（loss/acc/lr）
  add_scalars(tag, {k:v}, step)      多条曲线同图对比
  add_histogram(tag, values, step)   参数/梯度分布
  add_image(tag, img_tensor, step)   可视化图片
  add_graph(model, input)            网络结构图
  add_hparams(hparam_dict, metric)   超参数对比实验
```

---

## 知识点关系图

```
C1 基础（Tensor/Module/Loss/Optimizer）
              ↓
         C2 进阶

  如何评估 ──→ Precision/Recall/F1
                      ↓
  如何训练更好 ──→ LR Scheduler + 数据增强
                      ↓
  如何设计更好 ──→ 灵活架构 + 迁移学习
                      ↓
  如何找最优 ──→ Optuna 超参搜索
                      ↓
  如何看过程 ──→ TensorBoard 监控
```

---

## C1 vs C2 对比

```
C1（基础）：                     C2（进阶）：
  能让模型跑起来                   让模型跑得更好
  ┌─────────────────────┐         ┌─────────────────────────────┐
  │ Tensor              │         │ 评估：不只看 Accuracy        │
  │ Autograd            │         │ 调度：lr 动态变化            │
  │ nn.Module           │   →     │ 架构：突破 Sequential 限制   │
  │ Loss + Optimizer    │         │ 搜索：自动找最优超参          │
  │ DataLoader          │         │ 增强：数据少也能训好          │
  └─────────────────────┘         │ 迁移：站在巨人肩膀上          │
                                  │ 监控：可视化训练全过程        │
                                  └─────────────────────────────┘
```

---

## 常见问题速查

| 问题 | 解决方案 |
|------|---------|
| 类别不平衡，Accuracy 高但效果差 | 用 F1/Precision/Recall，或 `CrossEntropyLoss(weight=...)` |
| 训练后期 loss 不再下降 | 使用 LR Scheduler（CosineAnnealing/OneCycleLR） |
| 数据量少，过拟合 | 数据增强 + Dropout + 迁移学习 |
| 不知道用什么超参 | Optuna 自动搜索（50~100 trials） |
| 微调预训练模型效果差 | 检查是否使用了分层学习率（新层大lr，旧层小lr） |
| 训练过程看不清楚发生了什么 | 接入 TensorBoard，记录 loss/acc/lr 曲线 |

---

## 下一步：C3 高级架构与部署

```
C2 完成后，C3 将学习：
  ✦ 自定义复杂架构（Siamese / ResNet / DenseNet）
  ✦ Transformer 架构从零实现
  ✦ 扩散模型（Diffusion Models）
  ✦ 模型可解释性（Grad-CAM / Saliency Maps）
  ✦ 模型部署（ONNX / MLflow）
  ✦ 模型压缩（量化 / 剪枝）
```
