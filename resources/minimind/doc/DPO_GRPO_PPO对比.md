# DPO / GRPO / PPO 对比

三个脚本对应三种对齐（alignment）技术，可放在同一条谱系上看：`train_dpo.py`（DPO）、`train_grpo.py`（GRPO）、`train_ppo.py`（PPO）。

## 一、总览对比表

| 维度 | DPO（train_dpo.py） | GRPO（train_grpo.py） | PPO（train_ppo.py） |
|---|---|---|---|
| **数据** | 静态偏好对：chosen/rejected（`DPODataset`） | 只有 prompt（`RLAIFDataset`），在线生成 6 个回答 | 只有 prompt（`RLAIFDataset`），在线生成 1 个回答 |
| **要不要采样生成** | ❌ 不用，直接算 logits | ✅ rollout_engine 采样 | ✅ rollout_engine 采样 |
| **reward model** | ❌ 无 | ✅ `LMForRewardModel` + 规则奖励 | ✅ `LMForRewardModel` + 规则奖励 |
| **critic / value model** | ❌ 无 | ❌ 无（这是 GRPO 的关键创新） | ✅ `CriticModel`（自建 value head） |
| **reference model** | ✅ 有（冻结） | ✅ 有（冻结） | ✅ 有（冻结） |
| **模型总数** | 2（policy + ref） | 3（policy + ref + reward） | 4（actor + ref + critic + reward） |
| **优化信号** | chosen/rejected 概率差 | 组内标准化的 advantage（`(r-mean)/std`） | GAE 估计的 advantage |
| **核心损失** | `-logσ(β·logits)` | PPO-clip / cispo + KL 惩罚 | PPO-clip + value loss + KL |
| **默认学习率** | 4e-8 | 3e-7 | 3e-7 |
| **复杂度 / 显存** | 最低 | 中 | 最高 |

---

## 二、最本质的区别：优化信号从哪来

三种方法都是在回答「怎么让模型偏好好的回答」，区别在于「好」由谁来打分、怎么打分。

### DPO：把「好/坏」直接写死在数据里，用概率差当信号

`train_dpo.py` 的数据是**已经标注好的偏好对**（chosen vs rejected），不需要 reward model，也不需要 critic。它的信号是：

```python
logits = pi_logratios - ref_logratios          # 相对 ref 拉大 chosen 对 rejected 的概率优势
loss = -F.logsigmoid(beta * logits)            # train_dpo.py:48-49
```

**代价**：必须在训练前就准备好「同一问题的好回答 + 坏回答」这种成对数据。数据本身编码了偏好，模型学的是「模仿数据里的偏好」。

### GRPO：在线生成一组回答，组内互相比

`train_grpo.py` 每个 prompt 会 `rollout` 生成 `num_generations=6` 个回答，用 reward model 打分，然后**组内标准化**：

```python
grouped_rewards = rewards.view(-1, args.num_generations)         # [B, num_gen]
mean_r = grouped_rewards.mean(dim=1).repeat_interleave(...)
std_r  = grouped_rewards.std(dim=1, unbiased=False).repeat_interleave(...)
advantages = (rewards - mean_r) / (std_r + 1e-4)                 # train_grpo.py:121-124
```

GRPO 的核心思想（Group Relative Policy Optimization）：**不用 critic 去估计 baseline，直接用同一 prompt 下这一组回答的平均分当 baseline**。比平均分高的回答 advantage 为正（鼓励），低的为负（抑制）。这就是为什么它能甩掉 PPO 里的 value model——组内相对比较天然就是 baseline。

### PPO：用 critic 估计「这个状态价值多少」，再用 GAE 算 advantage

`train_ppo.py` 多了一个 `CriticModel`（`train_ppo.py:36-48`），它把 `lm_head` 换成输出单一标量的 `value_head`，用来估计每一步的「状态价值」：

```python
class CriticModel(MiniMindForCausalLM):
    def __init__(self, params):
        super().__init__(params)
        self.value_head = nn.Linear(params.hidden_size, 1)   # 输出一个 value 标量
```

然后 advantage 用 **GAE（广义优势估计）** 算（`train_ppo.py:139-146`）：

```python
for t in reversed(range(gen_len)):
    nv = old_resp_values[:, t + 1] if t < gen_len - 1 else 0.0
    delta = token_rewards[:, t] + args.gamma * nv - old_resp_values[:, t]
    lastgaelam = delta + args.gamma * args.lam * lastgaelam
    advs_rev.append(lastgaelam)
```

`token_rewards` 只在回答末尾加了整段的外部 reward（`train_ppo.py:137`），其余位置是 0；critic 负责把「稀疏的末尾奖励」通过 `value` 传播到每个 token，算出逐 token 的 advantage。

---

## 三、损失函数的差异

### DPO：一个闭式公式，简单

```python
loss = -logsigmoid(beta * ((chosen_pi - reject_pi) - (chosen_ref - reject_ref)))
```

没有 clip、没有 value loss、没有多轮更新。一条样本前向一次就出 loss。

### GRPO：PPO 风格的 ratio + clip，外加 KL 惩罚

```python
kl_div = ref_per_token_logps - per_token_logps
per_token_kl = torch.exp(kl_div) - kl_div - 1        # train_grpo.py:133，近似 KL 的泰勒展开式
ratio = torch.exp(per_token_logps - old_per_token_logps)  # 新旧策略的概率比

if args.loss_type == "cispo":
    clamped_ratio = torch.clamp(ratio, max=args.epsilon_high).detach()
    per_token_loss = -(clamped_ratio * advantages.unsqueeze(1) * per_token_logps - args.beta * per_token_kl)
else:  # grpo（标准 PPO-clip 式）
    clipped_ratio = torch.clamp(ratio, 1 - args.epsilon, 1 + args.epsilon)
    per_token_loss = -(torch.min(ratio*adv, clipped_ratio*adv) - args.beta * per_token_kl)
```

关键点：
- **`ratio`**：新策略概率 / 采样时的旧策略概率，`clip` 防止一次更新步子太大。
- **`per_token_kl`**：`exp(x)-x-1` 是 KL 散度在 0 附近的二阶近似，作用等价于 DPO 里的 ref 锚定——别离 ref 太远。
- 好处是不用 value loss（没有 critic）。

### PPO：最完整，三部分损失

```python
policy_loss = -min(adv * ratio, adv * clip(ratio, 1±clip_epsilon)) + kl_coef * kl_ref_penalty
value_loss  = 0.5 * max((V - returns)^2, (clip(V) - returns)^2)     # train_ppo.py:213-216
loss = policy_loss + vf_coef * value_loss + aux_loss
```

比 GRPO 多了 **value loss**（训练 critic 去拟合 `returns = advantages + old_values`）。这是经典 RL 的 Actor-Critic 双模型结构：actor 学策略，critic 学价值。

---

## 四、训练流程结构的差异

### DPO：纯监督式，一遍过

```
取 batch → policy/ref 各前向一次 → 算 dpo_loss → backward
```

和 SFT 结构几乎一样，只是 loss 换成了 DPO。**没有「采样 → 打分 → 更新」的循环**。

### GRPO / PPO：多了「在线采样（rollout）」环节

```
取 prompt → rollout 生成回答 → reward 打分 → 算 advantage → 更新 policy
```

二者都用 `rollout_engine`（`create_rollout_engine`，可切换 torch / sglang 两种引擎），这是 DPO 完全没有的组件。每次训练迭代都要**现场生成**回答，所以：
- 更慢（生成是自回归的，串行 decode）
- 能探索出数据里没有的新回答（online learning 的核心价值）
- 需要把采样时的旧概率 `old_per_token_logps` 存下来，用来算 ratio

### PPO 独有：同一批数据反复更新多次（inner loop）

```python
for ppo_epoch in range(args.ppo_update_iters):   # train_ppo.py:163，默认 2 次
    ...
    for i in range(0, B, mb_size):               # minibatch 更新
        ...
        if approx_kl_val > args.early_stop_kl:   # KL 太大就提前停
            stop_ppo = True
```

PPO 会拿同一批 rollout 数据更新 `ppo_update_iters` 次（还带 `early_stop_kl` 提前停止），GRPO 和 DPO 都是一遍过。这也是 PPO 更慢、更复杂的原因之一。

---

## 五、为什么会有这三个层次（演进逻辑）

1. **PPO（最老，最重）**：经典 RLHF。要 reward model + critic，用 GAE。训练难调、显存大、要维护 4 个模型。
2. **GRPO（DeepSeek 提出）**：发现「组内相对比较」可以替代 critic 的 baseline，砍掉 value model，显存和时间都省了。保留 reward model 和在线采样。
3. **DPO（最轻）**：连 reward model 和在线采样都省了，直接把「偏好」沉淀到离线数据里，用闭式解对齐。但**上限受限于你提供的偏好数据质量**——模型无法探索出比标注数据里「更好的回答」。

一个形象的类比：

| 方法 | 类比 |
|---|---|
| PPO | 请一位裁判（critic）给每一步打分，再结合外部评委（reward model） |
| GRPO | 不要裁判，让同题目的几个答案互相比高低 |
| DPO | 直接给模型看「标准答案 vs 错误答案」的对照卷，背下来 |

---

## 六、选择建议

- **数据只有成对偏好（chosen/rejected），算力紧张** → **DPO**，最简单最稳，几小时就能跑。
- **想在线探索、数据只有 prompt，但不想维护 critic** → **GRPO**，性价比最高，是当前主流（DeepSeek-R1 同款）。
- **要最强可控性、能接受复杂度和显存开销** → **PPO**，学术/研究场景完整实现。
