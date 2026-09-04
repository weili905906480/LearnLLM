# train_agent.py 详解

> 文件位置：`resources/minimind/trainer/train_agent.py`

这份代码是 **MiniMind 的 Agent 强化学习（RL）训练脚本**：训练模型学会「调用工具」（function calling）。与 SFT 不同，它**不给标准答案**，而是让模型自己采样若干条回答 → 按规则/reward 打分 → 用 **GRPO / CISPO** 目标（带 KL 约束）更新策略，是「试错学习」。

---

## 一、文件定位

整个脚本分四大块：

| 区块 | 作用 |
|---|---|
| 工具定义 + 模拟执行 | 定义工具、模拟数据、解析/执行工具调用、算 reward |
| 多轮 rollout | 让策略模型带工具多轮自回归生成 |
| 训练循环 `rl_train_epoch` | GRPO/CISPO 的核心优化逻辑 |
| `__main__` 入口 | 参数解析、模型初始化、调度训练 |

三个模型各司其职：

| 模型 | 作用 |
|---|---|
| `model` | 策略模型 π_θ，被训练的对象 |
| `ref_model` | 参考模型 π_ref（冻结），用于算 KL 惩罚，防止策略跑太远 |
| `reward_model` | 外部奖励模型（internlm2-1.8b），只在「无工具调用」分支给回答打分 |

---

## 二、导入与头部

```python
__package__ = "trainer"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
```

这是 MiniMind 项目的惯例：脚本被 `python -m trainer.train_agent` 或直接运行都能正确 import 到 `model`/`dataset` 包。

```python
import datasets  # noqa: F401  # Windows pyarrow/torch DLL conflict workaround (issue #771)
```

这行比较特殊——在 Windows 上 `torch` 和 `pyarrow` 的 DLL 有冲突，先 import `datasets` 可规避该问题（所以有 `noqa` 忽略 unused import 告警）。

关键依赖：

- `from model.model_minimind import MiniMindConfig, MiniMindForCausalLM` —— 策略模型
- `from dataset.lm_dataset import AgentRLDataset` —— 数据
- `from trainer.rollout_engine import create_rollout_engine, compute_per_token_logps` —— 采样引擎（torch / sglang 两种）
- `from trainer.trainer_utils import ...` —— 训练工具集

---

## 三、工具与 Reward

### 3.1 `rep_penalty`（重复惩罚）

```python
def rep_penalty(text, n=3, cap=0.5):
    toks = re.findall(r"\w+|[^\w\s]", text.lower())
    grams = [tuple(toks[i:i + n]) for i in range(len(toks) - n + 1)]
    return min(cap, (len(grams) - len(set(grams))) * cap * 2 / len(grams)) if grams else 0.0
```

把文本切词后统计 **n-gram 重复率**：`重复的 gram 数 / 总 gram 数`，再乘以 `2*cap`，上限 `cap=0.5`。目的是惩罚模型"车轱辘话"式输出。越重复惩罚越重，最高扣 0.5。

### 3.2 `TOOLS`（工具定义）

6 个 OpenAI 风格的 function 定义：`calculate_math`（算数）、`unit_converter`（单位换算）、`get_current_weather`（天气）、`get_current_time`（时间）、`get_exchange_rate`（汇率）、`translate_text`（翻译）。每个带 JSON Schema 的参数描述，会通过 `apply_chat_template(tools=...)` 渲染成 system 里的工具声明。

### 3.3 模拟数据

因为没有真实后端，用写死的字典模拟外部服务：`WEATHER_DATA`（城市→温度/天气）、`TIME_DATA`（时区→时间）、`EXCHANGE_DATA`（货币对→汇率）、`TRANSLATE_DATA`、`UNIT_DATA`（单位换算系数）。

### 3.4 `MOCK_RESULTS`（模拟执行器）

每个工具名对应一个 lambda，根据 `args` 查表返回结果：

- `calculate_math` 用 `eval` 算表达式，先做符号归一化（`^`→`**`、`×`→`*`、`÷`→`/` 等）。注意 `{"__builtins__": {}, "math": math}` 限制了 eval 的命名空间，只允许 `math` 模块，安全性相对可控。
- `unit_converter` 用 `from_unit_to_unit` 组合键查 `UNIT_DATA`，乘系数后 `round` 到 4 位。
- `get_current_weather` 查不到城市时给默认 `("22°C", "晴")`。
- 其余类似，查不到就给兜底值。

### 3.5 `CHECK_ARGS`（参数校验）

每个工具一个校验 lambda，判断参数是否齐全（比如天气必须有 `location`）。**这是训练 reward 里的"参数合法性"判据**，见下文。

### 3.6 `parse_tool_calls`（解析工具调用）

```python
re.findall(r'<tool_call>(.*?)</tool_call>', text, re.DOTALL)
```

从模型生成的文本里抓 `<tool_call>...</tool_call>` 块，每个块 `json.loads` 成一个 dict。MiniMind 的工具调用约定就是 `<tool_call>{json}</tool_call>` 这种格式。

### 3.7 `execute_tool`（执行工具）

```python
signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(TimeoutError()))
signal.alarm(1)
```

用 **SIGALRM 信号做 1 秒超时**保护（防止 eval 死循环之类），超时抛 `TimeoutError` 返回 `None`。`finally` 里 `signal.alarm(0)` 取消定时器。

> 注意：`SIGALRM` 只在 Unix 上可用，Windows 上会 `except` 掉，这是代码的一个平台局限性（不影响训练主流程）。

### 3.8 `rollout_single`（单条多轮 rollout）—— 核心采样逻辑

输入一条 `messages`（对话历史）+ `tools`，让模型**最多 `max_turns=3` 轮**自回归，每轮可能调用工具。

**变量初始化**：

- `prompt_ids`：第一轮的完整 prompt（含工具声明 + 用户问题），只记一次。
- `response_ids` / `response_mask` / `response_old_logps`：累积所有"策略自己生成的 token"（1 表示模型生成、要算 loss；0 表示工具观察、不算 loss）。
- `open_thinking = random.random() < thinking_ratio`：按概率开启 thinking 模式（决定模板是否带 `<think>`）。

**每轮循环**：

1. **渲染上下文**：`apply_chat_template(messages, tokenize=False, add_generation_prompt=True, tools=tools, open_thinking=...)` 得到文本，再 tokenize。`add_generation_prompt=True` 会在末尾加 `assistant` 头，引导模型开始生成。

2. **采样**：`rollout_engine.rollout(...)`，`temperature=0.8`，`max_new_tokens`。得到：
   - `completion_ids`：本轮新生成的 token id
   - `per_token_logps`：每个生成 token 的 log 概率（后续算 importance ratio 用）
   - `completions`：解码文本

3. **过滤 pad/eos**：把 `pad_token_id` 和 `eos_token_id` 从生成序列里剔除（这些 token 不参与 loss 计算）。用 `zip` 对齐过滤，保持 id 和 logprob 同步。

4. **累积**：`response_ids`/`response_mask`（全 1）/`response_old_logps` 追加本轮内容。

5. **解析工具调用**：如果本轮文本里没有 `<tool_call>`，说明模型直接回答了，`break` 结束循环。

6. **执行工具**：如果有工具调用：
   - 把 assistant 的生成文本 append 进 `messages`。
   - 对每个 call：解析 `name` 和 `arguments`（可能是字符串，先 `json.loads`），`execute_tool` 执行，结果转 JSON 字符串（截断到 2048 防止"天文数字"撑爆 tokenizer），以 `{"role": "tool", "content": result_str}` 形式 append 进 `messages`。
   - **重新渲染观察上下文**：`add_generation_prompt=not unfinished`（最后一轮后不再加生成头）。计算 `observe_ids`，取 `current_len`（已有 prompt+response 长度）之后的增量部分 `obs_delta`，作为"工具观察 token"：`response_mask` 扩展为 **0**（环境返回的，不参与 loss），`response_old_logps` 扩展为 **0.0**。

**返回值**：最终文本、最终上下文、`prompt_ids`、`response_ids`、`response_mask`、`response_old_logps`、每轮输出列表、是否"未完成"（`unfinished`，即第 3 轮还在调用工具）。

### 3.9 `rollout_batch`

简单封装：对每个 `(messages, tools)` 重复 `num_gen` 次 `rollout_single`（`num_gen` = 每个 prompt 生成的样本数，用于 GRPO 的组内对比），把结果按样本摊平到 8 个 list 返回。

### 3.10 `validate_gt_in_text`（Ground-Truth 校验）

```python
nums = [float(x) for x in re.findall(r'(?<![\w.])[-+]?\d+(?:\.\d+)?(?![\w.])', text_num)]
```

从模型回答里提取所有数字（去掉逗号后），返回"哪些 GT 值出现在了回答中"的集合。匹配分两种：

- GT 字符串（`s.lower()`）出现在文本里（字符串包含）。
- GT 是纯数字且与回答里的某个数字误差 `< 1e-6`（数值匹配）。

正则 `(?<![\w.])` 和 `(?![\w.])` 是负向后行/前行断言，避免把 `3.14` 里的 `3` 和 `14` 拆开。

### 3.11 `calculate_rewards`（Reward 计算）—— 奖惩的核心

遍历每条 completion，算 `reward`，最后 clip 到 `[-3, 3]`。

**预处理**：

- `sample_idx = idx // num_gen`：把第 idx 条生成映射回它属于哪个 prompt（组内编号）。
- `turn_answers`：每轮输出取 `</think>` 之后的部分（去掉思考段）。
- `valid_names`：该样本可用工具名的集合。
- `tool_calls`：所有轮里解析出的工具调用列表。
- **标签扣分**：`<tool_call>` 和 `</tool_call>` 数量不匹配（标签没闭合）每多/少一个扣 0.5。

**分支一：无工具调用**（模型直接回答，用格式 + RM 打分）：

| 项 | 分数 | 说明 |
|---|---|---|
| 长度 | +0.5 / -0.5 | 回答长度在 `[5, 800]` 加分，否则扣 |
| 思考长度 | +1.0 / -0.5 | `<think>` 内容长度在 `[20, 300]` 加分 |
| 思考闭合 | +0.25 / -0.25 | `</think>` 恰好出现 1 次（正常闭合） |
| RM 分 | `reward_model.get_score(...)` | 从 prompt 用正则 `r"<\|im_start\|>(system\|user\|assistant)...<\|im_end\|>"` 抽出 messages，交给外部 Reward 模型打分（clip 到 `[-3,3]`） |
| 重复惩罚 | `-rep_penalty(answer)` | |

**分支二：有工具调用**（用工具执行结果 + GT 打分）：

- **工具对齐分**：

  ```python
  valid_call_count = 校验通过的工具调用数
  tool_gap = abs(valid_call_count - len(gt)) + max(0, len(tool_calls) - valid_call_count)
  reward += 0.5 if tool_gap == 0 else -0.5 * tool_gap
  ```

  `valid_call_count` 是"参数合法"的调用数（用 `CHECK_ARGS` 校验）。`tool_gap` 综合了"合法调用数 vs GT 数量的差"和"非法调用数"。gap 为 0 加分，否则按 gap 扣分。

- **GT 分**：`final_text` 取最后一个 `</tool_call>` 之后的内容（即工具执行完后的最终回答）；`validate_gt_in_text` 校验 GT 命中率，`reward += 2.5 * len(verified) / len(gt)`——按比例给分。

- **未完成扣分**：最后一轮还在调工具（`unfinished`）扣 0.5。

- **重复惩罚**。

### 3.12 `InternLM2-1.8B-Reward`（奖励模型）—— 无工具分支的"打分器"

`reward_model` 是 **InternLM2-1.8B-Reward**：一个专门"打分"的奖励模型（Reward Model, RM）。它和普通语言模型"输入文字 → 输出文字"不同，它是 **"输入（问题 + 回答）→ 输出一个分数"**：

```
输入:  "北京今天适合跑步吗？"  +  "北京今天28度，晴，适合跑步。"
输出:  2.8   ← 分数，越高代表回答质量越好
```

它由上海 AI Lab 训练，用人类偏好数据（哪个回答更好）训练出来，1.8B 是参数量。

**为什么需要它**：规则函数只能检查"表面"——长度、`<think>` 标签闭合、工具调用格式、GT 命中；但回答的**语义质量**（内容对不对、有没有帮助、流不流畅）规则查不了，需要另一个模型当"评委"。

**代码加载**（`trainer_utils.py` 的 `LMForRewardModel` 包装类）：

```python
class LMForRewardModel:
    def __init__(self, model_path, device="cuda", dtype=torch.float16):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(model_path, torch_dtype=dtype, trust_remote_code=True)
        self.model = self.model.to(device).eval()

    @torch.no_grad()
    def get_score(self, messages, response):
        history_text = "\n".join([f"{m['role']}: {m['content']}" for m in messages[:-1]])
        last_query = messages[-1]['content'] if messages else ""
        message_context = f"{history_text}\n以上是对话历史。我的新问题是：\n{last_query}"
        eval_messages = [
            {"role": "user", "content": message_context},
            {"role": "assistant", "content": response}
        ]
        score = self.model.get_score(self.tokenizer, eval_messages)
        return max(min(score, 3.0), -3.0)   # clip 到 [-3, 3]
```

注意 `trust_remote_code=True`：InternLM2-Reward 自带自定义代码，提供 `get_score(tokenizer, messages)` 方法（普通 `AutoModel` 没有）。

**为什么只在"无工具调用"分支用**：

- 有工具调用时：模型拿到了工具返回的确定结果，最终回答可以和 `gt` 精确比对（`validate_gt_in_text`），用不着 RM。
- 无工具调用时：没有工具结果可校验，回答质量只能靠 RM 主观打分。

所以 RM 是**兜底**：只有当规则和 GT 校验覆盖不到（模型没走工具链路）时，才请 RM 出马。

> ⚠️ 显存：1.8B 参数 fp16 权重约 3.6GB，加载它本身就需要一块像样的 GPU（2GB 显存的卡放不下）。

---

## 四、`rl_train_epoch`（核心训练循环）

### 4.1 Rollout 采样

```python
with torch.no_grad():
    completions, contexts, prompt_ids_batch, ... = rollout_batch(...)
```

一个 batch 的 `messages_batch` 全被采样，得到每个样本的完整生成结果。

### 4.2 Packing（打包）

把每个样本的 `prompt_ids + response_ids` 拼成完整序列，构造训练张量：

```python
ids = p + r                          # 完整 token 序列
mask = [0] * len(p) + m              # prompt 部分为 0，response 部分为 m(模型生成=1，工具观察=0)
old_logps = [0.0] * max(len(p) - 1, 0) + old_lp   # prompt 位置旧 logp 补 0
```

- **截断**：若超 `max_total_len`，从尾部保留（`ids[-max_total_len:]`），`old_logps` 也要同步截到 `len(ids)-1`（因为 logprob 是对应"预测下一个 token"的，长度比序列少 1）。
- `prompt_len`：第一个 `mask==1` 的位置，即"第一个模型生成 token"的索引。
- 最后把不同长度的样本 **pad 到 batch 内最长**，构造 `input_ids`、`prompt_lens`、`full_response_masks`（float）、`old_per_token_logps`、`full_mask`（非 pad 的位置为 1）。

### 4.3 计算 reward

调用 `calculate_rewards`，传入 `turn_outputs_batch` 和 `unfinished_batch`。

### 4.4 策略模型前向 + logprob

```python
res = model_unwrapped(input_ids, attention_mask=full_mask)
aux_loss = res.aux_loss if lm_config.use_moe else torch.tensor(0.0, ...)
logits = res.logits[:, :-1, :]
per_token_logps = F.log_softmax(logits, dim=-1).gather(2, input_ids[:, 1:].unsqueeze(-1)).squeeze(-1)
```

- `aux_loss`：MoE 的辅助损失（load balancing 等），非 MoE 则为 0。
- `logits[:, :-1, :]` 与 `input_ids[:, 1:]` 错位：用位置 t 的 logits 预测位置 t+1 的 token，得到每个"预测 token"的 log 概率。这就是当前策略的 `π_θ(token|context)`。

### 4.5 参考模型 logprob

```python
ref_per_token_logps = compute_per_token_logps(ref_model, input_ids, input_ids.size(1) - 1, ...)
```

用冻结的参考模型算同样位置的 log 概率，用于 KL 惩罚。

### 4.6 Completion mask 与 EOS 处理

```python
completion_mask = full_response_masks[:, 1:]      # 错位后对齐"预测 token"
is_eos = (input_ids[:, 1:] == eos_token_id) & completion_mask.bool()
eos_idx = ...  # 每个样本第一个 EOS 的位置（默认最后一位）
completion_mask = completion_mask * (pos <= eos_idx.unsqueeze(1)).float()
```

关键点：**只对模型生成段（mask=1）且 EOS 之前的 token 算 loss**。EOS 之后的内容（以及工具观察的 0 段）都不参与。`valid_rows = token_counts > 0` 过滤掉完全没有有效生成 token 的样本。

### 4.7 优势函数（GRPO 组内归一化）

```python
grouped_rewards = rewards.view(-1, args.num_generations)   # [num_prompts, num_gen]
mean_r = grouped_rewards.mean(dim=1).repeat_interleave(args.num_generations)
std_r  = grouped_rewards.std(dim=1, unbiased=False).repeat_interleave(args.num_generations)
advantages = (rewards - mean_r) / (std_r + 1e-4)
```

这是 **GRPO（Group Relative Policy Optimization）** 的核心：每个 prompt 生成 `num_gen` 个回答，**组内**用均值方差做标准化得到 advantage。这样就不需要 Critic 网络（省掉一半模型），只用组内相对好坏来更新。

### 4.8 Loss 计算（GRPO vs CISPO 两种目标）

先算 KL 散度和 importance ratio：

```python
kl_div = ref_per_token_logps - per_token_logps          # log(π_ref/π_θ)
per_token_kl = torch.exp(kl_div) - kl_div - 1           # 反向 KL 的泰勒展开估计（k3 估计器）
ratio = torch.exp(per_token_logps - old_per_token_logps) # π_θ / π_old（重要性采样比）
```

- `per_token_kl` 用的是 **k3 估计器**：`exp(x) - x - 1`，其中 `x = log(π_ref) - log(π_θ)`，是一个恒非负、低方差的无偏 KL 估计量。

**分支 CISPO**：

```python
clamped_ratio = torch.clamp(ratio, max=args.epsilon_high).detach()  # 上限裁剪并 detach
per_token_loss = -(clamped_ratio * advantages * per_token_logps - beta * per_token_kl)
```

CISPO（Clamped Surrogate Policy Optimization / Critic-free Importance-Sampled PO）用 `ratio` 作为 token 级权重去**加权 logprob**（而非加权 ratio 本身），且 ratio 只做上界裁剪。与 GRPO 最大的区别：**它直接优化 log 概率的加权和**，在 off-policy 度更大时更稳定。

**具体数字例子**（单个 token，advantage=+1.5 的"好样本"，`epsilon_high=5.0`、`beta=0.1`）：

| 量 | 值 | 说明 |
|---|---|---|
| `log π_old`（采样时） | -2.0 | 采样时该 token 概率 ≈ e⁻² ≈ 0.135 |
| `log π_θ`（当前策略） | -1.0 | 当前概率 ≈ e⁻¹ ≈ 0.368，更"喜欢"这个 token |
| `ratio` | e^(-1.0-(-2.0)) = e¹ ≈ **2.72** | 概率涨了 2.72 倍 |
| `advantage` | +1.5 | |
| `log π_ref`（参考） | -1.5 | |

计算：

```python
clamped_ratio = clamp(2.72, max=5.0) = 2.72          # 没超上界，不裁剪
kl_div = -1.5 - (-1.0) = -0.5
per_token_kl = exp(-0.5) - (-0.5) - 1 = 0.1065       # k3 估计器

per_token_loss = -(2.72 * 1.5 * (-1.0)  -  0.1 * 0.1065)
               = -(-4.08            -  0.0107)
               = 4.09    # 正的大数 → 梯度推高 log π_θ（好 token 更被鼓励）
```

三个关键设计点：

1. **`ratio` 是权重，不是被优化项**：它衡量"当前策略和采样时差了多少"，作为加权系数；`per_token_logps` 才是真正被优化的量。
2. **`.detach()` 让梯度纯粹**：把 `clamped_ratio` 摘出计算图变成常数，梯度 = `-(clamped_ratio * A) + β·(KL 梯度项)`，方向完全由 `advantage` 的符号决定，不会被 clip 干扰。
3. **只裁上界、不裁下界**：`ratio` 太大（如 e⁵ ≈ 148）会让单步更新爆炸，裁到 5 控制更新幅度；`ratio` 很小只说明"采样时概率高、现在几乎不生成"，作为权重变小是合理的，不必人为抬高。

**分支 GRPO**：

```python
clipped_ratio = torch.clamp(ratio, 1 - args.epsilon, 1 + args.epsilon)
per_token_loss1 = ratio * advantages
per_token_loss2 = clipped_ratio * advantages
per_token_loss = -(torch.min(per_token_loss1, per_token_loss2) - beta * per_token_kl)
```

这是标准 PPO 的 clip 目标（带 KL 惩罚的 GRPO 变体）：对 ratio 做 `[1-ε, 1+ε]` 裁剪取 min，防止单步更新过大。

**CISPO vs GRPO 的本质区别**（两者共享 95% 代码，只差这一行 loss）：

| 维度 | GRPO | CISPO |
|---|---|---|
| 被优化的项 | `ratio`（策略比） | `log π_θ`（log 概率） |
| ratio 的角色 | 优化主体 | 权重系数（detach，不产生梯度） |
| clip 方式 | 对称 `[1-ε, 1+ε]`（默认 0.2） | 只裁上界 `max=ε_high`（默认 5.0） |
| 梯度路径 | 会被 clip 截断 | 永远畅通 |

关键直觉——**GRPO 的 clip 会截断梯度**：当 `ratio` 偏离 1 很远时，`min(ratio*A, clip(ratio)*A)` 会取到"被裁成常数的 clip 项"，这一项对 `log π_θ` 的梯度 = 0，等于**停止更新**。例：`ratio=6`、`A=+1`、`ε=0.2` 时，`min(6, 1.2)=1.2`（常数，梯度=0）。

而 CISPO 把 ratio 变成"不产生梯度、只裁上界的权重"，例：`ratio=6` 裁成 5 后，`loss = -(5 * log π_θ)`，梯度 = `-5`，**仍然在更新**。

> 一句话：GRPO 用 clip 限制更新幅度，代价是 off-policy 大时梯度被切断；CISPO 让梯度永远通过 `log π_θ` 流动，更新永不中断。

**最终 loss**：

```python
policy_loss = (per_token_loss * completion_mask).sum(dim=1)[valid_rows] / token_counts[valid_rows].clamp(min=1)
loss = (policy_loss + aux_loss) / args.accumulation_steps
```

每个样本的 token 级 loss 求和后**除以有效 token 数**（平均），再对 batch 求均值；加上 MoE 辅助损失；除以梯度累积步数后 `backward()`。

### 4.9 梯度累积 + 优化器

每 `accumulation_steps` 步才真正 `clip_grad_norm_` + `optimizer.step()` + `scheduler.step()` + `zero_grad()`。

### 4.10 日志

周期性打印：Reward、KL、组内 reward 标准差（`GrpStd`）、advantage 标准差（`AdvStd`）、loss、平均长度、学习率等，并可选写 wandb（swanlab）。

### 4.11 保存

- 存权重 `.pth`（half 精度）到 `args.save_dir`。
- 调 `lm_checkpoint(...)` 存 resume checkpoint（含 optimizer/scheduler/epoch/step/wandb_id），用 `.tmp` + `os.replace` 保证原子写。

### 4.12 更新 rollout 引擎

```python
if step % args.save_interval == 0 or step == iters: rollout_engine.update_policy(model)
```

关键：rollout 引擎持有策略模型的引用，训练若干步后要把**新权重同步回引擎**（尤其 SGLang 引擎要上传权重到服务器），否则采样一直用旧模型。

### 4.13 显存释放

`del` 掉大张量，防止显存累积。末尾处理"最后一个不完整 accumulation"的余数更新。

---

## 五、`__main__` 入口

### 5.1 参数

关键超参：

- `--learning_rate` 默认 `3e-7`（RL 阶段学习率极低）
- `--num_generations` 默认 `4`（每个 prompt 采 4 个，GRPO 组大小）
- `--beta` 默认 `0.1`（KL 惩罚系数）
- `--loss_type` 默认 `cispo`（可选 `grpo`）
- `--epsilon` 0.2 / `--epsilon_high` 5.0（两种目标的裁剪参数）
- `--max_total_len` 2500（训练侧总长上界）
- `--thinking_ratio` 0.1（10% 概率开启 thinking）
- `--rollout_engine` torch / sglang
- `--from_weight` 默认 `full_sft`（从 SFT 权重出发）

### 5.2 初始化

1. `init_distributed_mode()`：检测 DDP，返回 local_rank，单机返回 0。
2. `setup_seed(42 + rank)`：随机种子按 rank 错开。
3. 构造 `MiniMindConfig`，`max_seq_len = args.max_seq_len + args.max_gen_len`（prompt 空间 + 生成长度）。
4. 若 `--from_resume`，`lm_checkpoint` 加载 resume 数据。
5. `autocast_ctx`：CUDA 用 `torch.cuda.amp.autocast(bfloat16)`，CPU 用 `nullcontext()`。
6. wandb（用 `swanlab` 库，国产替代）——resume 时 `resume='must'` 恢复原 run。

### 5.3 加载三个模型

```python
model, tokenizer = init_model(lm_config, args.from_weight, ...)   # 策略模型（要训练）
ref_model, _ = init_model(...); ref_model.eval().requires_grad_(False)  # 参考模型（冻结，算 KL）
reward_model = LMForRewardModel(...)   # 外部 Reward 模型（internlm2-1.8b，打分用）
```

三个模型各司其职：策略模型被优化；参考模型固定算 KL 惩罚；RM 打分（仅无工具调用分支）。

**关键澄清：`ref_model` 不是"已经会工具"的专家模型。** 它和 `model` 用同一个 `args.from_weight`（默认 `full_sft`）加载，是**完全相同的初始权重**，只是加载后立刻 `.eval().requires_grad_(False)` 冻结：

- `ref_model` 的"会工具程度" = SFT 阶段（`full_sft`）的水平——SFT 数据已混入 Tool Call 样本，所以它**有基础但不熟练**，它自己就是待优化的起点，不是标杆；
- 它的真正作用是 **KL 惩罚的锚点**：`per_token_kl` 衡量训练后的策略 π_θ 偏离 π_ref 多少，loss 里 `+β·KL` 惩罚"跑太远"，从而**防止 reward hacking**（钻规则空子）、**防止遗忘 SFT 学到的通用能力**；
- 类比：`ref_model` 是"训练前拍照冻结的你"，KL 约束 = "可以进步，但别变得连自己都不认识"。

所以三个模型里**没有任何一个"本来就擅长工具"**——`model` 和 `ref_model` 起点相同（SFT 水平），`reward_model` 只负责打分。工具能力是 RL 训练里从 SFT 基线一步步"逼"出来的。

### 5.4 数据与优化器

- `AgentRLDataset`：读 `agent_rl.jsonl`，每条含 `conversations`（对话）、`tools`、`gt`（标准答案）。
- `collate_fn`：batch 里三字段各自打包成 list（不做 pad，因为 RL 是逐个 rollout，不需要等长 batch）。
- `CosineAnnealingLR`：`T_max = 总优化步数`，`eta_min = lr/10`。

### 5.5 恢复状态

从 resume checkpoint 加载 model/optimizer/scheduler/epoch/step，支持断点续训。

### 5.6 编译 + DDP 包装

- `torch.compile`（可选）后要 `rollout_engine.update_policy(model)` 更新引擎引用。
- DDP 包装 model，再次 `update_policy`。

### 5.7 训练主循环

每个 epoch：

- `train_sampler.set_epoch(epoch)` 保证 DDP 下每个 epoch 数据 shuffle 不同。
- `setup_seed(42 + epoch); torch.randperm(len(train_ds))`：每 epoch 用固定种子打乱数据（可复现）。
- `SkipBatchSampler`：支持从 `start_step` 跳过已训练的 batch（断点续训用）。
- 调 `rl_train_epoch(...)`。

### 5.8 收尾

`dist.barrier()` + `dist.destroy_process_group()`。

---

## 六、整体数据流总结

```
agent_rl.jsonl ──> AgentRLDataset ──> messages + tools + gt
                                          │
                              rollout_batch（策略模型多轮生成+工具执行）
                                          │
                     prompt_ids + response_ids + mask + old_logps
                                          │
                    packing → input_ids / full_response_masks / old_logps
                                          │
        ┌─────────────┼──────────────────────┐
        │             │                      │
   calculate_rewards  policy forward   ref_model forward
   (工具执行+格式+RM)   (π_θ logprob)     (π_ref logprob)
        │             │                      │
        └──► advantages (GRPO组内标准化) ◄────┘
                     │
        GRPO / CISPO loss + KL惩罚 + MoE aux
                     │
              backward / 梯度累积 / 优化器
                     │
           rollout_engine.update_policy（同步新权重）
```

**一句话概括**：这是"离线采样 + 在线优化"的 Agent RL——用策略模型自己采样轨迹、执行模拟工具、按格式和 GT 打 reward，再用 GRPO/CISPO 目标（带 KL 约束）更新策略，从而让 MiniMind 学会"何时调用工具、参数怎么填、工具结果怎么用"。
