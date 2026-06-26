"""
概率论与统计基础模块

包含 LLM/Transformer 所需的概率论知识：
- 概率分布
- 条件概率
- 期望与方差
- Softmax 与概率
- 交叉熵（训练目标）

与 nanochat 的关联：
- Softmax：nanochat/gpt.py:472
- 交叉熵损失：nanochat/gpt.py:477
- 采样策略：温度、top-k、top-p
"""

import numpy as np
from typing import List, Tuple


# =============================================================================
# 概率基础
# =============================================================================

def probability_basics():
    """概率基础概念演示"""
    print("【概率基础】")
    
    # 离散概率分布
    outcomes = ['A', 'B', 'C', 'D']
    probabilities = [0.1, 0.3, 0.4, 0.2]
    
    print("离散概率分布:")
    for outcome, prob in zip(outcomes, probabilities):
        print(f"  P({outcome}) = {prob}")
    
    # 验证概率和为 1
    print(f"概率总和: {sum(probabilities)}")
    
    # 期望（Expected Value）
    values = [1, 2, 3, 4]
    expected_value = sum(v * p for v, p in zip(values, probabilities))
    print(f"\n期望 E[X] = {expected_value}")
    # 手动计算: 1*0.1 + 2*0.3 + 3*0.4 + 4*0.2 = 0.1 + 0.6 + 1.2 + 0.8 = 2.7
    
    # 方差（Variance）
    variance = sum((v - expected_value)**2 * p for v, p in zip(values, probabilities))
    print(f"方差 Var(X) = {variance:.4f}")
    print(f"标准差 σ = {np.sqrt(variance):.4f}")
    
    return probabilities


# =============================================================================
# 条件概率与贝叶斯定理
# =============================================================================

def conditional_probability():
    """条件概率与贝叶斯定理"""
    print("\n【条件概率与贝叶斯定理】")
    
    # P(A|B) = P(A ∩ B) / P(B)
    print("条件概率: P(A|B) = P(A ∩ B) / P(B)")
    
    # 贝叶斯定理
    # P(A|B) = P(B|A) * P(A) / P(B)
    print("\n贝叶斯定理: P(A|B) = P(B|A) * P(A) / P(B)")
    
    # 示例：垃圾邮件检测
    # P(垃圾) = 0.3
    # P(包含'免费'|垃圾) = 0.8
    # P(包含'免费'|正常) = 0.1
    
    p_spam = 0.3
    p_free_given_spam = 0.8
    p_free_given_normal = 0.1
    
    # P(包含'免费') = P(免费|垃圾)*P(垃圾) + P(免费|正常)*P(正常)
    p_free = p_free_given_spam * p_spam + p_free_given_normal * (1 - p_spam)
    
    # P(垃圾|包含'免费')
    p_spam_given_free = (p_free_given_spam * p_spam) / p_free
    
    print(f"\n垃圾邮件示例:")
    print(f"  P(垃圾) = {p_spam}")
    print(f"  P('免费'|垃圾) = {p_free_given_spam}")
    print(f"  P('免费'|正常) = {p_free_given_normal}")
    print(f"  P('免费') = {p_free:.4f}")
    print(f"  P(垃圾|'免费') = {p_spam_given_free:.4f}")


# =============================================================================
# Softmax 函数
# =============================================================================

def softmax(x: np.ndarray, axis: int = -1, temperature: float = 1.0) -> np.ndarray:
    """Softmax 函数 - 将 logits 转换为概率分布
    
    与 nanochat 的关联：
    - nanochat/gpt.py:472 - logits 的 softmax
    - 注意力权重计算
    - 文本生成时的概率分布
    
    参数:
        x: 输入 logits
        axis: 计算 softmax 的轴
        temperature: 温度参数，控制分布的尖锐程度
    """
    # 应用温度
    x = x / temperature
    
    # 减去最大值防止数值溢出
    exp_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)


def softmax_demo():
    """Softmax 函数演示"""
    print("\n【Softmax 函数】")
    
    # 基本 softmax
    logits = np.array([2.0, 1.0, 0.1])
    probs = softmax(logits)
    
    print(f"输入 logits: {logits}")
    print(f"Softmax 输出: {probs}")
    print(f"概率和: {probs.sum():.4f}")
    
    # 温度参数的影响
    print("\n温度参数的影响:")
    temperatures = [0.1, 0.5, 1.0, 2.0, 10.0]
    
    for temp in temperatures:
        probs = softmax(logits, temperature=temp)
        print(f"  T={temp:4.1f}: {probs} (熵={-np.sum(probs * np.log(probs + 1e-10)):.4f})")
    
    print("\n温度的作用:")
    print("  - T < 1: 分布更尖锐（更确定）")
    print("  - T = 1: 标准 softmax")
    print("  - T > 1: 分布更平坦（更随机）")


# =============================================================================
# 交叉熵损失
# =============================================================================

def cross_entropy_loss(predicted: np.ndarray, target: np.ndarray) -> float:
    """交叉熵损失 - LLM 的训练目标
    
    公式: L = -sum(target * log(predicted))
    
    与 nanochat 的关联：
    - nanochat/gpt.py:477 - F.cross_entropy(logits, targets)
    - 下一个 Token 预测的损失函数
    
    参数:
        predicted: 预测的概率分布
        target: 真实的标签（one-hot 或概率分布）
    """
    # 防止 log(0)
    epsilon = 1e-10
    predicted = np.clip(predicted, epsilon, 1.0)
    
    # 交叉熵计算
    loss = -np.sum(target * np.log(predicted))
    return loss


def cross_entropy_demo():
    """交叉熵损失演示"""
    print("\n【交叉熵损失】")
    
    # 真实标签（one-hot）
    target = np.array([0, 0, 1, 0])  # 第3个类别是正确的
    
    # 预测1：好的预测
    pred_good = np.array([0.1, 0.1, 0.7, 0.1])
    loss_good = cross_entropy_loss(pred_good, target)
    
    # 预测2：差的预测
    pred_bad = np.array([0.4, 0.3, 0.2, 0.1])
    loss_bad = cross_entropy_loss(pred_bad, target)
    
    print(f"真实标签: {target}")
    print(f"\n好的预测: {pred_good}")
    print(f"  交叉熵损失: {loss_good:.4f}")
    
    print(f"\n差的预测: {pred_bad}")
    print(f"  交叉熵损失: {loss_bad:.4f}")
    
    print(f"\n损失越小越好: {loss_good < loss_bad}")
    
    # 与准确率的关系
    print("\n损失与准确率的关系:")
    print("  - 损失小 → 模型预测准确")
    print("  - 损失大 → 模型预测不准")
    print("  - 训练目标：最小化交叉熵损失")


# =============================================================================
# 采样策略
# =============================================================================

def sampling_strategies():
    """文本生成的采样策略"""
    print("\n【采样策略】")
    
    # 假设模型输出的 logits
    logits = np.array([2.0, 1.0, 0.5, 0.1, -0.5])
    vocab = ['the', 'a', 'an', 'this', 'that']
    
    print(f"原始 logits: {logits}")
    print(f"词表: {vocab}")
    
    # 1. 贪心采样（Greedy）
    greedy_idx = np.argmax(logits)
    print(f"\n1. 贪心采样: '{vocab[greedy_idx]}' (概率最高)")
    
    # 2. 温度采样
    print("\n2. 温度采样:")
    for temp in [0.5, 1.0, 2.0]:
        probs = softmax(logits, temperature=temp)
        # 按概率采样
        sampled_idx = np.random.choice(len(vocab), p=probs)
        print(f"   T={temp}: 分布={probs.round(3)}, 采样='{vocab[sampled_idx]}'")
    
    # 3. Top-k 采样
    k = 3
    print(f"\n3. Top-k 采样 (k={k}):")
    top_k_indices = np.argsort(logits)[-k:]
    top_k_logits = logits[top_k_indices]
    top_k_probs = softmax(top_k_logits)
    print(f"   Top-k 词: {[vocab[i] for i in top_k_indices]}")
    print(f"   概率分布: {top_k_probs.round(3)}")
    
    # 4. Top-p (Nucleus) 采样
    p_threshold = 0.8
    print(f"\n4. Top-p 采样 (p={p_threshold}):")
    probs = softmax(logits)
    sorted_indices = np.argsort(probs)[::-1]
    cumulative_probs = np.cumsum(probs[sorted_indices])
    
    # 找到累积概率超过 p 的位置
    cutoff_idx = np.searchsorted(cumulative_probs, p_threshold) + 1
    top_p_indices = sorted_indices[:cutoff_idx]
    
    print(f"   按概率排序: {[vocab[i] for i in sorted_indices]}")
    print(f"   累积概率: {cumulative_probs.round(3)}")
    print(f"   Top-p 词: {[vocab[i] for i in top_p_indices]}")


# =============================================================================
# 信息论基础
# =============================================================================

def information_theory():
    """信息论基础 - 熵、KL 散度"""
    print("\n【信息论基础】")
    
    # 熵（Entropy）- 分布的不确定性
    probs = np.array([0.25, 0.25, 0.25, 0.25])  # 均匀分布
    entropy = -np.sum(probs * np.log2(probs))
    print(f"均匀分布: {probs}")
    print(f"熵 H = {entropy:.4f} bits")
    
    # 非均匀分布
    probs_skewed = np.array([0.7, 0.1, 0.1, 0.1])
    entropy_skewed = -np.sum(probs_skewed * np.log2(probs_skewed))
    print(f"\n偏斜分布: {probs_skewed}")
    print(f"熵 H = {entropy_skewed:.4f} bits")
    print(f"熵越小，分布越确定")
    
    # KL 散度 - 两个分布的差异
    p = np.array([0.7, 0.2, 0.1])
    q = np.array([0.5, 0.3, 0.2])
    
    # KL(p || q) = sum(p * log(p/q))
    epsilon = 1e-10
    kl_divergence = np.sum(p * np.log((p + epsilon) / (q + epsilon)))
    print(f"\nKL 散度:")
    print(f"  p = {p}")
    print(f"  q = {q}")
    print(f"  KL(p || q) = {kl_divergence:.4f}")
    print(f"  KL 散度越小，两个分布越相似")


# =============================================================================
# 综合演示
# =============================================================================

def demo():
    """运行所有概率论演示"""
    probability_basics()
    conditional_probability()
    softmax_demo()
    cross_entropy_demo()
    sampling_strategies()
    information_theory()
    
    print("\n" + "=" * 60)
    print("概率论与统计演示完成！")
    print("=" * 60)
    print("\n关键概念:")
    print("  1. Softmax 将 logits 转换为概率分布")
    print("  2. 交叉熵是 LLM 训练的损失函数")
    print("  3. 温度参数控制生成的随机性")
    print("  4. Top-k/Top-p 是常用的采样策略")


if __name__ == "__main__":
    demo()
