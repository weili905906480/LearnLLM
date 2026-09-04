# train_grpo.py 详解

`train_grpo.py` 做的是 **GRPO（Group Relative Policy Optimization，分组相对策略优化）**，属于 RLAIF 的对齐阶段。一句话概括：

> 对每个 prompt 用 policy 模型在线采样 G=6 条回答 → 用「规则 + Reward Model」打分 → 组内做均值/标准差归一化得到优势 advantage → 用带 KL 约束和裁剪的 loss 更新 policy，全程**不需要 Critic/Value 网络**。

这是 DeepSeek-R1 那套 RL 训练算法，特点是省掉了 PPO 里的价值网络。

---

## 一、GRPO 与 PPO 的核心区别

| | PPO | GRPO |
|---|---|---|
| 优势估计 | 训练 Value 网络，用 GAE 算优势 | 组内 G 条回答的 reward 标准化即可 |
| 需要几个模型 | policy + value + ref + reward（4 个） | policy + ref + reward（3 个） |
| 优势怎么算 | `A = r + γV - V` | `A = (r - mean(r)) / std(r)`（组内） |

脚本里因此初始化了 **3 个模型**（`train_grpo.py:274-279`）：

```python
model          # policy 模型（要训练，就是 full_sft 之后的模型）
ref_model      # 参考模型（冻结，算 KL 用，防止 policy 跑飞）
reward_model   # internlm2-1_8b-reward 奖励模型（打分用）
```

更完整的 DPO / GRPO / PPO 横向对比见 [`DPO_GRPO_PPO对比.md`](DPO_GRPO_PPO对比.md)。

---

## 二、数据流全景图（含张量形状）

设 `B = batch_size = 2`，`num_generations = 6`，则 `N = B×G = 12` 条回答。

```
prompt: list[str] 长度 B=2
   │ tokenize（左填充）
   ▼
prompt_inputs["input_ids"]  shape [B, P]        P = prompt 长度(含左 pad)
   │ rollout_engine.rollout(每个 prompt 采样 G 次)
   ▼
output_ids        [N, P+R]   完整序列（prompt + 生成）
completion_ids    [N, R]     只含生成部分
per_token_logps   [N, R]     采样时每个生成 token 的 log 概率（old policy）
completions       list[str]  解码后的文本（算 reward 用）
prompt_lens       [N]        每个样本的 prompt 长度
completion_mask   [N, R]     哪些位置是真实 token（非 pad）
   │ calculate_rewards
   ▼
rewards           [N]        标量奖励
   │ view(B, G) → 组内标准化
   ▼
advantages        [N]        (r - mean)/std
   │ 当前 policy 前向 + ref 前向 → 算 ratio、KL
   ▼
per_token_loss    [N, R]  → mask 后求均值 → loss → backward
```

---

## 三、GRPO 的训练数据（rlaif.jsonl）

### 1. 文件与格式

默认路径 `../dataset/rlaif.jsonl`（约 24MB，需按 README 下载），由 `RLAIFDataset` 加载：

```python
# train_grpo.py:225
parser.add_argument("--data_path", type=str, default="../dataset/rlaif.jsonl", ...)
# train_grpo.py:292
train_ds = RLAIFDataset(args.data_path, tokenizer, max_length=lm_config.max_seq_len, thinking_ratio=args.thinking_ratio)
```

`rlaif.jsonl` 是 JSONL（每行一条 JSON），格式和 SFT 一致，但 **assistant 字段不需要真实答案**。官方示例（README）：

```json
{
    "conversations": [
        {"role": "user", "content": "请解释一下什么是光合作用？"},
        {"role": "assistant", "content": "无"}
    ]
}
```

### 2. 包含哪些字段

每条样本只有一个顶层字段 `conversations`（消息列表）：

| 字段 | 说明 | GRPO 里是否用到 |
|---|---|---|
| `role` | `system` / `user` / `assistant` | ✅ 用于套 chat 模板 |
| `content` | 该角色的文本 | ✅ 用户问题是核心 |
| `reasoning_content` | 思考内容（SFT 数据里有） | ❌ `RLAIFDataset` 不读 |
| `tools` / `tool_calls` | 工具调用（SFT/Agent 数据里有） | ❌ `RLAIFDataset` 不读 |

`RLAIFDataset`（`lm_dataset.py:317-346`）比 `SFTDataset` 简单得多，**不声明 schema、也不读 `reasoning_content`/`tools`/`tool_calls`**，只通过 `apply_chat_template` 处理 `role` 和 `content`。

### 3. 关键设计：assistant 内容为什么是「无」

RLAIF = AI Feedback，答案由模型自己生成、再由 AI 打分，所以数据里不需要黄金答案。最后一轮 `assistant` 只是占位符，在 `create_chat_prompt` 里被**直接丢掉**：

```python
def create_chat_prompt(self, conversations):
    conversations = pre_processing_chat(conversations)   # 20% 概率补 system
    use_thinking = random.random() < self.thinking_ratio # 90% 开思考
    return self.tokenizer.apply_chat_template(
        conversations[:-1],                              # ← 丢掉最后一条（占位答案）
        tokenize=False,
        open_thinking=use_thinking,
        add_generation_prompt=True                       # 末尾补 <|im_start|>assistant\n
    )
```

最终喂给模型的只是一条 **prompt**（到用户提问为止 + 空 assistant 头），模型自己在 rollout 阶段采样生成回答。`__getitem__` 返回 `{'prompt': prompt, 'answer': ""}`，其中 `answer` 恒为空，**训练器根本不读它**（只取 `batch['prompt']`，见 `train_grpo.py:73`）。

### 4. 和其他 RL 数据的重要区分（容易搞混）

| 文件 | 大小 | 用途 | 数据集类 | 格式差异 |
|---|---|---|---|---|
| `rlaif.jsonl` ✨ | 24MB | **GRPO/PPO/CISPO**（`train_grpo.py`/`train_ppo.py`） | `RLAIFDataset` | 只有 `conversations`，无 `gt` |
| `agent_rl.jsonl` | 86MB | Agentic RL（`train_agent.py`，多轮 Tool-Use） | `AgentRLDataset` | 带 `tools` + 顶层 `gt` 字段 |
| `agent_rl_math.jsonl` | 18MB | Agentic RL 纯数学（RLVR，带最终答案校验） | `AgentRLDataset` | 同上 |

`agent_rl.jsonl` 长这样（对比即可看出多了 `tools` 和 `gt`）：

```json
{"conversations": [
  {"role": "system", "content": "", "tools": "[{...calculate_math...}]"},
  {"role": "user", "content": "算算7109*2920"},
  {"role": "assistant", "content": ""}
], "gt": ["20758280"]}
```

`gt` 是 Ground Truth，给 Agent RL 做「答案对不对」的规则校验（RLVR）。**GRPO 的 `rlaif.jsonl` 没有 `gt`**，纯靠 Reward Model 打分。

---

## 四、逐函数详解

### 1. `rep_penalty(text, n=3, cap=0.5)`（第 31-34 行）

一个**基于 n-gram 的重复惩罚**（不依赖 tokenizer，纯字符串级别），抑制生成内容反复复制同一句话：

```python
toks = re.findall(r"\w+|[^\w\s]", text.lower())          # 分词（字母数字 / 单个标点）
grams = [tuple(toks[i:i+3]) for i in range(len(toks)-2)] # 滑窗取三元组
return min(0.5, (len(grams)-len(set(grams))) * 0.5*2 / len(grams))
```

**具体例子**：`"The cat sat the cat sat the cat sat"`（n=3）

- 分词：`[the, cat, sat, the, cat, sat, the, cat, sat]`（9 个 token）
- 三元组共 7 个，去重后只有 3 种：`(the,cat,sat)`、`(cat,sat,the)`、`(sat,the,cat)`
- 重复率 = `(7-3)/7 ≈ 0.571`
- 惩罚 = `0.571 × 1.0 ≈ 0.571` → 超过 cap 0.5 → 返回 **0.5**

完全没重复时 `len(grams)==len(set)` → 分子为 0 → 返回 0。

### 2. `calculate_rewards(prompts, responses, reward_model)`（第 37-68 行）

对每条回答算标量 reward，累加成 `[N]` 张量。奖励由四部分组成：

**① 长度奖励**（第 54 行）
```python
rewards[i] += 0.5 if 20 <= len(response.strip()) <= 800 else -0.5
```
太短（<20 字符）或太长（>800）扣分，鼓励中等长度。

**② 思考（thinking）奖励**（第 55-58 行）——只有回答里含 `</think>` 才触发：
```python
thinking_content, answer_content = response.split('</think>', 1)
rewards += 1.0 if 20 <= len(thinking_content.strip()) <= 300 else -0.5   # 思考长度合适 +1
rewards += 0.25 if response.count('</think>') == 1 else -0.25            # 恰好一个 </think> +0.25
answer = answer_content.strip()   # 之后评估的"答案"只取 </think> 之后的部分
```

**具体例子**，一条回答是 `<think>先算 3×7=21，再加 5。</think>最终答案是 26。`：
- 思考内容 `先算 3×7=21，再加 5。` 约 12 字符 → 不在 [20,300] → **-0.5**
- `</think>` 出现 1 次 → **+0.25**
- `answer` 变成 `最终答案是 26。`

**③ 重复惩罚**（第 60 行）`rewards -= rep_penalty(answer)`

**④ Reward Model 打分**（第 50-66 行）：先用正则 `r"<\|im_start\|>(system|user|assistant)\s+(.*?)<\|im_end\|>"` 把 prompt 还原成 `messages`，再调 `reward_model.get_score(messages, answer)`，最后被 clamp 到 **[-3, 3]**（`trainer_utils.py:177`）。

> 注意：reward model 收到的不是原始生成文本，而是 **`</think>` 之后的部分**（`answer`），思考过程本身不参与打分，但通过 ② 的规则奖励间接被鼓励。

### 3. `grpo_train_epoch(...)`（第 71-203 行）—— 核心训练循环

#### 3.1 编码 prompt（第 74-78 行）
```python
prompt_inputs = tokenizer(prompts, ..., padding_side="left", add_special_tokens=False)
```
**左填充**：pad 在左边，保证真实内容在右侧、生成位置对齐；超长则截断最后 `max_seq_len` 个 token。

#### 3.2 Rollout 采样（第 80-93 行）
```python
rollout_result = rollout_engine.rollout(...)  # temperature=0.8，采样 G 次
```
以 `temperature=0.8` 采样出 G 条回答，同时记录采样时的 `per_token_logps`（old policy 的 log 概率），后面算 ratio 用。

关键行（第 93 行）——算出「生成 token 在完整序列中的位置」：
```python
logp_pos = prompt_lens.unsqueeze(1) - 1 + torch.arange(R).unsqueeze(0)   # [N, R]
```
**为什么从 `prompt_len - 1` 起步？** 模型位置 `t` 的 logits 预测的是位置 `t+1` 的 token。生成部分第一个 token 在完整序列第 `P` 位，由位置 `P-1` 的 logits 预测，所以索引从 `P-1` 开始。

#### 3.3 算当前 policy 与 ref 的逐 token log 概率（第 204-217 行）
```python
per_token_logps = F.log_softmax(res.logits[:, :-1, :], dim=-1) \
    .gather(2, outputs[:, 1:].unsqueeze(-1)).squeeze(-1) \
    .gather(1, logp_pos)      # [N, R] —— 当前 policy 的 log 概率
```

这一行把 `[N, P+R, V]` 的 logits 变成 `[N, R]` 的逐 token log 概率，分三步：

```
res.logits                               [N, P+R, V]     完整序列每位置的词表 logits
   ↓ [:, :-1, :]   丢掉最后一位
                                          [N, P+R-1, V]   位置 t 的 logits 预测 t+1 的 token
   ↓ log_softmax(dim=-1)  词表维归一化
                                          [N, P+R-1, V]   变成 log 概率分布
   ↓ gather(2, outputs[:, 1:])  抠真实 next-token 的 log 概率
                                          [N, P+R-1]      每位置只剩一个数
   ↓ gather(1, logp_pos)  只挑生成段
                                          [N, R]          per_token_logps
```

- **错位对齐**：`logits[:, :-1]` 预测 `outputs[:, 1:]`（位置 t 的 logits 预测 t+1 的 token），所以前者少最后一位、后者少第一位，两者一一对应。
- **第一次 gather**：`gather(2, outputs[:, 1:])` 在词表维抠出「真实下一个 token」的 log 概率，得到 `[N, P+R-1]`，此时还没区分 prompt 段和生成段。
- **第二次 gather**：`gather(1, logp_pos)` 只挑 `logp_pos = prompt_lens-1+arange(R)` 指向的生成段位置，得到 `[N, R]`。

`ref_per_token_logps` 完全一样，但用冻结的 `ref_model` 前向、且包在 `torch.no_grad()` 里（不计算梯度）。

> 从 logits → log_softmax → 两次 gather 的完整数值追踪（含 vocab=8 的逐位查表）见 [[grpo_train_epoch_loss算例]] 阶段 4。

#### 3.4 组内标准化算 advantage（第 121-124 行）
```python
grouped_rewards = rewards.view(-1, G)                     # [B, G]
mean_r = grouped_rewards.mean(dim=1).repeat_interleave(G) # [N]
std_r  = grouped_rewards.std(dim=1, unbiased=False).repeat_interleave(G)
advantages = (rewards - mean_r) / (std_r + 1e-4)         # [N]
```
**这是 GRPO 的精髓**：优势是**同一 prompt 的 6 条回答内部比较**，而非全局比较。

**具体数值例子**（某 prompt 的 6 条回答）：
```
rewards = [1.5, 2.0, 0.5, -0.5, 1.0, 0.0]
mean    = 0.75
std     = 0.854
advantages = [0.878, 1.464, -0.293, -1.464, 0.293, -0.878]
```
第 2 条（2.0 分）优势 +1.46（加大概率），第 4 条（-0.5 分）优势 -1.46（压低概率）。

#### 3.5 构造 completion_mask（第 126-130 行）
生成可能提前遇 EOS，后面全是 pad，这些 pad 位置不参与 loss。逻辑：每行找**第一个 EOS** 的位置 `eos_idx`（没有 EOS 就取 `R-1`），保留 `≤ eos_idx` 且非 pad 的位置。

```
completion_ids:  [ 12, 34, 56, 2, 0, 0 ]   (2=EOS, 0=pad)
completion_mask: [  1,  1,  1, 1, 0, 0 ]   ← EOS 及之前保留，pad 丢弃
```

#### 3.6 算 KL 散度和 ratio（第 260-266 行）
```python
kl_div        = ref_per_token_logps - per_token_logps
per_token_kl  = torch.exp(kl_div) - kl_div - 1   # k3 估计器
ratio         = torch.exp(per_token_logps - old_per_token_logps)
```

- **`per_token_kl`** 用 **k3 估计器**：`e^d - d - 1`，是 KL(π‖π_ref) 的始终 ≥0 下界近似。例：`d=0.5` → `e^0.5 - 0.5 - 1 = 0.149`。
- **`ratio`** 是重要性采样比率 `π_current / π_old`，衡量当前 policy 相对采样时对每个 token 的偏好变化。计算分两步：先算对数差 `per_token_logps - old_per_token_logps`，再取指数。例：old logp=-2.3，current logp=-2.0 → 对数差 `+0.3` → `ratio = e^0.3 = 1.35`（当前更看好该 token）。
  - `ratio > 1`：policy 比采样时更可能生成该 token（正 advantage 会把它继续拉高）。
  - `ratio < 1`：policy 比采样时更不可能生成该 token（负 advantage 会把它继续压低）。
  - 完整的「逐元素减法 → 取指数」数值追踪见 [[grpo_train_epoch_loss算例]] 阶段 5.2。

#### 3.7 计算 loss（第 135-143 行）—— 两种策略

**CISPO（默认，`loss_type="cispo"`）**：
```python
clamped_ratio = torch.clamp(ratio, max=epsilon_high).detach()   # 截断到 ≤5.0，并 detach
per_token_loss = -(clamped_ratio * adv * per_token_logps - beta * per_token_kl)
```
`clamped_ratio.detach()`：ratio 只作**停止梯度的权重**，梯度实际只流经 `per_token_logps`（当前 policy），只限制 ratio 上限防梯度过大。

**GRPO（`loss_type="grpo"`）**：
```python
clipped_ratio  = torch.clamp(ratio, 1-epsilon, 1+epsilon)      # PPO 式双向裁剪
per_token_loss = -(torch.min(ratio*adv, clipped_ratio*adv) - beta*per_token_kl)
```
标准 PPO clip + KL 惩罚：ratio 偏离 1 太远被裁剪，`min` 防止过度更新。两者都用 `-beta * per_token_kl`（`beta=0.1`）拉小 KL。

#### 3.8 求均值 + 反向传播（第 143-152 行）
```python
policy_loss = ((per_token_loss * completion_mask).sum(1) / completion_mask.sum(1).clamp(min=1)).mean()
loss = (policy_loss + aux_loss) / args.accumulation_steps
loss.backward()
```
按有效 token 数归一化（每条回答独立平均，再对 batch 平均）；`aux_loss` 是 MoE 负载均衡损失（非 MoE 为 0）；支持梯度累积 + 梯度裁剪。

#### 3.9 日志 / 保存 / 同步权重（第 154-196 行）
- 每隔 `log_interval` 打印 reward、KL、advantage 均值/方差、actor loss、平均回答长度、学习率。
- 每隔 `save_interval` 保存权重（带 `_moe` 后缀），并调 `lm_checkpoint` 存 optimizer/scheduler 状态支持续训。
- **第 193 行** `rollout_engine.update_policy(model)`：sglang 引擎时把最新权重推给推理服务器；torch 引擎仅更新内部引用。

---

## 五、`open_thinking` 模板如何触发 thinking

思考的开关由 `RLAIFDataset.create_chat_prompt`（`lm_dataset.py:330-338`）和 tokenizer 的 `chat_template` 配合实现。

### 1. 触发点

```python
use_thinking = random.random() < self.thinking_ratio   # thinking_ratio 默认 0.9
return self.tokenizer.apply_chat_template(
    conversations[:-1],                                # 丢掉最后一轮（黄金答案）
    tokenize=False,
    open_thinking=use_thinking,                        # ← 传给 Jinja 模板
    add_generation_prompt=True
)
```

`apply_chat_template` 会把 `open_thinking` 这个 kwarg 原样变成模板变量。

### 2. 模板里如何决定 prompt 结尾

`tokenizer_config.json` 里 `chat_template` 的最后一段：

```jinja
{%- if add_generation_prompt %}
    {{- '<|im_start|>assistant\n' }}
    {%- if open_thinking is defined and open_thinking is true %}
        {{- '<think>\n' }}                  ← 情况 A：开放思考标签
    {%- else %}
        {{- '<think>\n\n</think>\n\n' }}     ← 情况 B：空思考标签（已闭合）
    {%- endif %}
{%- endif %}
```

**情况 A：`open_thinking=True`（约 90%）**，prompt 结尾是**开着的** `<think>`：
```
<|im_start|>user
小明有3个苹果，又买了5个，一共有几个？<|im_end|>
<|im_start|>assistant
<think>
```
模型 rollout 采样时从 `<think>` 续写思考，最后用 `</think>` 收尾再出答案：
```
先算 3+5=8。</think>

小明一共有 8 个苹果。
```

**情况 B：`open_thinking=False`（约 10%）**，prompt 结尾是**已闭合的空思考标签**，模型直接出答案、不思考：
```
<|im_start|>assistant
<think>

</think>

小明一共有 8 个苹果。
```

混 10% 不思考，是为了让模型学会「简单问题直接答、复杂问题才思考」，而非对所有 prompt 都强行写思考。

### 3. 回到奖励函数的闭环

`calculate_rewards` 靠判断 `</think>` 是否出现在**生成段**里来区分两种情况（第 55-59 行）：
- **情况 A**：生成段含 `</think>` → 触发思考奖励，`split('</think>',1)` 拆出思考与答案。
- **情况 B**：生成段不含 `</think>` → 跳过思考奖励，整段直接当 `answer` 打分。

关键细节：**`<think>` 在 prompt 里，`</think>` 在生成段里**。思考不是靠 reward model 打分，而是靠**规则奖励**（长度 + 标签格式）引导，reward model 只对 `</think>` 之后的最终答案打分——这正是 DeepSeek-R1 的「格式奖励 + 答案正确性奖励」分离设计。

---

## 六、ref_model 如何来的

### 1. 代码位置（`train_grpo.py:272-277`）

```python
base_weight = args.from_weight          # 默认 'full_sft'（见第 231 行）
model, tokenizer = init_model(lm_config, base_weight, device=args.device)   # Policy
ref_model, _ = init_model(lm_config, base_weight, device=args.device)       # Reference
ref_model = ref_model.eval().requires_grad_(False)
```

`ref_model` 和 `model` 调用**同一个函数、同一个 `base_weight`**，唯一区别是后面多了 `.eval().requires_grad_(False)` 冻结。

### 2. `init_model` 内部（`trainer_utils.py:119-131`）

```python
def init_model(lm_config, from_weight='pretrain', tokenizer_path='../model', save_dir='../out', device='cuda'):
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    model = MiniMindForCausalLM(lm_config)                    # 1. 新建随机初始化模型
    if from_weight != 'none':
        moe_suffix = '_moe' if lm_config.use_moe else ''
        weight_path = f'{save_dir}/{from_weight}_{lm_config.hidden_size}{moe_suffix}.pth'
        weights = torch.load(weight_path, map_location=device)  # 2. 读磁盘权重
        model.load_state_dict(weights, strict=False)            # 3. 加载
    return model.to(device), tokenizer
```

默认参数代入路径：`../out/full_sft_768.pth`。

### 3. 三个关键结论

1. **`model` 和 `ref_model` 初始完全同权**：两次 `init_model` 从同一 checkpoint 各加载独立一份权重，初始 `KL(π‖π_ref)=0`。
2. **训练中只有 `model` 更新，`ref_model` 永远不动**：`eval()` 关 Dropout、`requires_grad_(False)` 不参与 backward，optimizer 也只包了 `model.parameters()`（第 294 行）。
3. **`ref_model` 的用途是算 KL 散度、防 policy 跑飞**（第 104/132-133 行）：`kl_div = ref_logps - per_token_logps`，`per_token_kl = exp(kl_div) - kl_div - 1`。reward 打分是「代理」目标，容易被 reward hacking，KL 约束保证 policy 不偏离 SFT 模型太远。

### 4. 显存代价

`ref_model` 是一份完整权重，额外占一份显存，但因不存梯度、以 bf16 前向，开销可控。GRPO 显存 ≈ policy（含梯度+optimizer 状态）+ ref（仅权重）+ reward（仅权重，float16）。

---

## 七、主函数流程（第 246-334 行）

按注释的 9 步：

1. **初始化分布式 + 种子**（`init_distributed_mode` / `setup_seed(42+rank)`）
2. **构造配置**：`lm_config.max_seq_len = max_seq_len + max_gen_len`（=768+1024，因为完整序列是 prompt+生成）
3. **混合精度**：bf16/fp16 的 `autocast_ctx`
4. **wandb**（实际 import 的是 `swanlab`，国产替代）
5. **初始化三个模型**（policy/ref/reward）+ rollout 引擎 + 数据集 + AdamW + 余弦退火调度
6. **可选续训**：`--from_resume 1` 时从 ckp 恢复 model/optimizer/scheduler/epoch/step
7. **编译/分布式包装**：`torch.compile` + `DDP`
8. **训练循环**：每 epoch 用 `SkipBatchSampler`（支持续训跳过已完成 step）打乱数据，调 `grpo_train_epoch`
9. **清理进程组**

---

## 八、值得注意的设计点 / 易错点

1. **`logp_pos` 的 `-1` 偏移**（第 93 行）：最容易算错的一处。模型位置 `t` 的 logits 预测 `t+1`，第一个生成 token 的 logit 位置是 `P-1` 而非 `P`。
2. **`full_mask` 用 `outputs != pad_token_id`**（第 92 行）：左填充的 prompt 和右填充的生成都要 mask 掉 pad。
3. **CISPO 与 GRPO 的差异**（第 135-142 行）：CISPO 只裁剪 ratio 上限并 detach，梯度直接走 logp；GRPO 是标准双向 PPO clip。默认 `cispo`。
4. **completion_mask 只看第一个 EOS**（第 126-130 行）：用 `argmax` 找第一个 EOS，避免把 EOS 之后因 padding 对齐出现的 token 也算进 loss。
5. **reward model 的 clamp**（`trainer_utils.py:177`）：`max(min(score, 3.0), -3.0)`，防止极端分值主导梯度。
6. **组内 std 用 `unbiased=False`**（第 123 行）：明确是「组内全体 6 条的总体标准差」，避免歧义。

---

## 九、一句话总结

`train_grpo.py` 用「在线采样 G 个回答 → 规则 + Reward Model 打分 → 组内标准化得 advantage → PPO-clip / CISPO 加 KL 惩罚」的方式更新 policy，省掉 PPO 的 critic，靠冻结的 `ref_model`（源自 `full_sft` 权重）锚定 KL、靠 `open_thinking` + `chat_template` 控制思考开关，数据只需要 `rlaif.jsonl` 里的 prompt（assistant 答案留空由模型自己生成）。
