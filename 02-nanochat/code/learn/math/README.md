# 数学基础学习模块

本模块包含 LLM/Transformer 所需的数学基础知识，通过代码实现帮助理解。

## 模块结构

```
math/
├── README.md              # 本文件
├── __init__.py            # 包初始化
├── linear_algebra.py      # 线性代数基础
├── probability.py         # 概率论与统计
├── calculus.py            # 微积分与梯度
├── activations.py         # 激活函数（连接数学与神经网络）
└── losses.py              # 损失函数（连接数学与训练）
```

## 学习顺序

1. **linear_algebra.py** - 向量、矩阵、矩阵乘法、转置
2. **probability.py** - 概率分布、softmax、交叉熵
3. **calculus.py** - 导数、偏导数、链式法则、梯度
4. **activations.py** - ReLU、GELU、Sigmoid 等激活函数
5. **losses.py** - MSE、交叉熵等损失函数

## 运行方式

```powershell
# 激活虚拟环境
.venv\Scripts\activate

# 运行单个模块
python -m learn.math.linear_algebra
python -m learn.math.probability
python -m learn.math.calculus

# 运行所有模块
python -m learn.math
```

## 与 nanochat 的关联

每个模块都包含与 nanochat 代码的对应关系，帮助理解：
- 矩阵乘法 → Attention 机制
- Softmax → 注意力权重计算
- 交叉熵损失 → 下一个 Token 预测
- 梯度下降 → 模型训练
