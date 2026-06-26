"""
损失函数模块

包含神经网络中常用的损失函数：
- MSE（均方误差）
- 交叉熵损失
- KL 散度
- 对比损失

与 nanochat 的关联：
- nanochat/gpt.py:477 - F.cross_entropy(logits, targets)
- nanochat/loss_eval.py - Bits Per Byte 评估
"""

import numpy as np
from typing import Tuple


# =============================================================================
# 均方误差（MSE）
# =============================================================================

def mse_loss(predicted: np.ndarray, target: np.ndarray) -> float:
    """均方误差损失: L = (1/n) * Σ (predicted - target)²
    
    常用于回归任务
    """
    return np.mean((predicted - target) ** 2)


def mse_derivative(predicted: np.ndarray, target: np.ndarray) -> np.ndarray:
    """MSE 梯度: dL/dpred = 2 * (predicted - target) / n"""
    n = len(predicted)
    return 2 * (predicted - target) / n


def mse_demo():
    """MSE 损失演示"""
    print("【均方误差（MSE）】")
    
    predicted = np.array([1.0, 2.0, 3.0])
    target = np.array([1.1, 2.2, 2.8])
    
    loss = mse_loss(predicted, target)
    grad = mse_derivative(predicted, target)
    
    print(f"预测值: {predicted}")
    print(f"目标值: {target}")
    print(f"MSE 损失: {loss:.4f}")
    print(f"梯度: {grad}")
    
    print("\n特点:")
    print("  - 用于回归任务")
    print("  - 对异常值敏感")
    print("  - 梯度与误差成正比")


# =============================================================================
# 二元交叉熵
# =============================================================================

def binary_cross_entropy(predicted: np.ndarray, target: np.ndarray) -> float:
    """二元交叉熵损失: L = -[target * log(pred) + (1-target) * log(1-pred)]
    
    常用于二分类任务
    """
    epsilon = 1e-10
    predicted = np.clip(predicted, epsilon, 1 - epsilon)
    return -np.mean(target * np.log(predicted) + (1 - target) * np.log(1 - predicted))


def binary_cross_entropy_demo():
    """二元交叉熵演示"""
    print("\n【二元交叉熵】")
    
    # 好的预测
    pred_good = np.array([0.9, 0.1, 0.8])
    target = np.array([1, 0, 1])
    loss_good = binary_cross_entropy(pred_good, target)
    
    # 差的预测
    pred_bad = np.array([0.3, 0.7, 0.4])
    loss_bad = binary_cross_entropy(pred_bad, target)
    
    print(f"目标: {target}")
    print(f"\n好的预测: {pred_good}")
    print(f"  损失: {loss_good:.4f}")
    
    print(f"\n差的预测: {pred_bad}")
    print(f"  损失: {loss_bad:.4f}")
    
    print("\n特点:")
    print("  - 用于二分类任务")
    print("  - 输出经过 Sigmoid")
    print("  - 损失越小，预测越准确")


# =============================================================================
# 多类交叉熵 - LLM 的核心损失函数
# =============================================================================

def cross_entropy_loss(predicted: np.ndarray, target: np.ndarray) -> float:
    """多类交叉熵损失: L = -Σ target * log(predicted)
    
    与 nanochat 的关联：
    - nanochat/gpt.py:477 - F.cross_entropy(logits, targets)
    - LLM 的下一个 Token 预测损失
    
    参数:
        predicted: 预测的概率分布 (vocab_size,)
        target: 真实标签 (one-hot 或类别索引)
    """
    epsilon = 1e-10
    predicted = np.clip(predicted, epsilon, 1.0)
    
    # 如果 target 是 one-hot
    if target.ndim == 1 and len(target) == len(predicted):
        return -np.sum(target * np.log(predicted))
    # 如果 target 是类别索引
    else:
        return -np.log(predicted[int(target)])


def cross_entropy_with_logits(logits: np.ndarray, target: int) -> float:
    """带 Logits 的交叉熵（数值更稳定）
    
    L = -logits[target] + log(Σ exp(logits))
    
    与 nanochat 的关联：
    - 这是 PyTorch F.cross_entropy 的实现方式
    - 避免了显式计算 softmax
    """
    # 数值稳定的 log-sum-exp
    max_logit = np.max(logits)
    log_sum_exp = max_logit + np.log(np.sum(np.exp(logits - max_logit)))
    return -logits[target] + log_sum_exp


def cross_entropy_demo():
    """交叉熵损失演示"""
    print("\n【多类交叉熵 - LLM 核心】")
    
    # 模拟模型输出 logits
    logits = np.array([2.0, 1.0, 0.5, 0.1, -0.5])
    vocab = ['the', 'a', 'cat', 'dog', 'bird']
    target_idx = 2  # 正确答案是 'cat'
    
    print(f"Logits: {logits}")
    print(f"词表: {vocab}")
    print(f"目标: {vocab[target_idx]} (index={target_idx})")
    
    # 计算 softmax
    probs = np.exp(logits) / np.sum(np.exp(logits))
    print(f"\nSoftmax 概率: {probs.round(4)}")
    
    # 交叉熵损失
    loss = cross_entropy_loss(probs, target_idx)
    loss_from_logits = cross_entropy_with_logits(logits, target_idx)
    
    print(f"交叉熵损失 (from probs): {loss:.4f}")
    print(f"交叉熵损失 (from logits): {loss_from_logits:.4f}")
    
    # Bits Per Byte (BPB)
    # nanochat 使用 BPB 作为评估指标
    bpb = loss / np.log(2)
    print(f"\nBits Per Byte (BPB): {bpb:.4f}")
    print("  BPB = loss / ln(2)")
    print("  BPB 越小越好")


# =============================================================================
# KL 散度
# =============================================================================

def kl_divergence(p: np.ndarray, q: np.ndarray) -> float:
    """KL 散度: KL(p || q) = Σ p * log(p/q)
    
    衡量两个分布的差异
    """
    epsilon = 1e-10
    p = np.clip(p, epsilon, 1.0)
    q = np.clip(q, epsilon, 1.0)
    return np.sum(p * np.log(p / q))


def kl_divergence_demo():
    """KL 散度演示"""
    print("\n【KL 散度】")
    
    # 相似的分布
    p = np.array([0.4, 0.3, 0.2, 0.1])
    q1 = np.array([0.35, 0.35, 0.2, 0.1])
    
    # 不同的分布
    q2 = np.array([0.1, 0.1, 0.4, 0.4])
    
    kl1 = kl_divergence(p, q1)
    kl2 = kl_divergence(p, q2)
    
    print(f"分布 p: {p}")
    print(f"分布 q1 (相似): {q1}")
    print(f"分布 q2 (不同): {q2}")
    print(f"\nKL(p || q1) = {kl1:.4f}")
    print(f"KL(p || q2) = {kl2:.4f}")
    print(f"KL 散度越小，分布越相似")


# =============================================================================
# 对比损失
# =============================================================================

def contrastive_loss(embedding1: np.ndarray, embedding2: np.ndarray, 
                     label: int, margin: float = 1.0) -> float:
    """对比损失
    
    用于学习相似性：
    - label=1: 相似样本，距离应该小
    - label=0: 不相似样本，距离应该大
    
    L = label * d² + (1-label) * max(0, margin - d)²
    """
    d = np.linalg.norm(embedding1 - embedding2)
    loss = label * d**2 + (1 - label) * max(0, margin - d)**2
    return loss


def contrastive_loss_demo():
    """对比损失演示"""
    print("\n【对比损失】")
    
    # 相似样本
    emb1 = np.array([1.0, 0.5, 0.3])
    emb2 = np.array([0.9, 0.6, 0.35])  # 接近 emb1
    
    # 不相似样本
    emb3 = np.array([0.1, 0.8, 0.9])  # 远离 emb1
    
    loss_similar = contrastive_loss(emb1, emb2, label=1)
    loss_different = contrastive_loss(emb1, emb3, label=0)
    
    print(f"Embedding 1: {emb1}")
    print(f"Embedding 2 (相似): {emb2}")
    print(f"Embedding 3 (不同): {emb3}")
    
    print(f"\n相似样本损失: {loss_similar:.4f}")
    print(f"不同样本损失: {loss_different:.4f}")
    
    print("\n特点:")
    print("  - 用于学习相似性/距离")
    print("  - 相似样本：最小化距离")
    print("  - 不相似样本：最大化距离（直到 margin）")


# =============================================================================
# Label Smoothing
# =============================================================================

def label_smoothing_loss(logits: np.ndarray, target: int, 
                         smoothing: float = 0.1) -> float:
    """Label Smoothing 交叉熵
    
    防止模型过于自信
    
    L = (1 - smoothing) * CE(pred, target) + smoothing * CE(pred, uniform)
    """
    n_classes = len(logits)
    
    # 计算 softmax
    probs = np.exp(logits) / np.sum(np.exp(logits))
    
    # 创建平滑标签
    smooth_target = np.full(n_classes, smoothing / (n_classes - 1))
    smooth_target[target] = 1 - smoothing
    
    # 交叉熵
    epsilon = 1e-10
    probs = np.clip(probs, epsilon, 1.0)
    loss = -np.sum(smooth_target * np.log(probs))
    
    return loss


def label_smoothing_demo():
    """Label Smoothing 演示"""
    print("\n【Label Smoothing】")
    
    logits = np.array([5.0, 1.0, 0.5, 0.1])
    target = 0
    
    loss_normal = cross_entropy_with_logits(logits, target)
    loss_smooth = label_smoothing_loss(logits, target, smoothing=0.1)
    
    print(f"Logits: {logits}")
    print(f"目标: {target}")
    print(f"\n普通交叉熵: {loss_normal:.4f}")
    print(f"Label Smoothing (0.1): {loss_smooth:.4f}")
    
    print("\n特点:")
    print("  - 防止模型过于自信")
    print("  - 提高泛化能力")
    print("  - smoothing 越大，标签越平滑")


# =============================================================================
# Focal Loss
# =============================================================================

def focal_loss(predicted: np.ndarray, target: int, 
               gamma: float = 2.0) -> float:
    """Focal Loss
    
    解决类别不平衡问题
    L = -(1-p)^γ * log(p)
    
    对容易分类的样本降低权重
    """
    p = predicted[target]
    epsilon = 1e-10
    p = np.clip(p, epsilon, 1.0)
    return -(1 - p) ** gamma * np.log(p)


def focal_loss_demo():
    """Focal Loss 演示"""
    print("\n【Focal Loss】")
    
    # 容易分类的样本
    pred_easy = np.array([0.9, 0.05, 0.03, 0.02])
    
    # 难分类的样本
    pred_hard = np.array([0.5, 0.3, 0.15, 0.05])
    
    target = 0
    
    loss_easy = focal_loss(pred_easy, target, gamma=2.0)
    loss_hard = focal_loss(pred_hard, target, gamma=2.0)
    
    # 对比普通交叉熵
    ce_easy = -np.log(pred_easy[target])
    ce_hard = -np.log(pred_hard[target])
    
    print(f"容易分类: {pred_easy}, CE={ce_easy:.4f}, Focal={loss_easy:.4f}")
    print(f"难分类:   {pred_hard}, CE={ce_hard:.4f}, Focal={loss_hard:.4f}")
    
    print("\n特点:")
    print("  - 对容易样本降低权重")
    print("  - 关注难分类样本")
    print("  - γ 越大，关注难样本越多")


# =============================================================================
# 损失函数对比
# =============================================================================

def compare_losses():
    """损失函数对比"""
    print("\n【损失函数对比】")
    
    print("损失函数          | 应用场景           | 特点")
    print("-" * 60)
    print("MSE               | 回归任务           | 对异常值敏感")
    print("Binary CE         | 二分类任务         | 输出 Sigmoid")
    print("Multi-class CE    | 多分类/LLM         | 输出 Softmax")
    print("KL 散度           | 分布匹配           | 不对称")
    print("Contrastive Loss  | 相似性学习         | 度量学习")
    print("Label Smoothing   | 分类任务           | 防止过拟合")
    print("Focal Loss        | 类别不平衡         | 关注难样本")
    
    print("\nnanochat 使用的损失函数:")
    print("  - 预训练: 交叉熵损失 (下一个 Token 预测)")
    print("  - SFT: 交叉熵损失 (只在 assistant 回复上计算)")
    print("  - 评估: Bits Per Byte (BPB)")


# =============================================================================
# 综合演示
# =============================================================================

def demo():
    """运行所有损失函数演示"""
    mse_demo()
    binary_cross_entropy_demo()
    cross_entropy_demo()
    kl_divergence_demo()
    contrastive_loss_demo()
    label_smoothing_demo()
    focal_loss_demo()
    compare_losses()
    
    print("\n" + "=" * 60)
    print("损失函数演示完成！")
    print("=" * 60)
    print("\n关键概念:")
    print("  1. 交叉熵是 LLM 的核心损失函数")
    print("  2. Label Smoothing 可以提高泛化能力")
    print("  3. BPB 是 nanochat 的评估指标")
    print("  4. 选择合适的损失函数很重要")


if __name__ == "__main__":
    demo()
