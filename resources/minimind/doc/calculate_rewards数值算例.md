# calculate_rewards 数值算例：一条 GRPO 奖励的完整追踪（batch=1, num_generations=2）

本文用 `batch=1`、`num_generations=2` 的具体小例子，把 `train_grpo.py` 里的 `calculate_rewards` 函数从头到尾走一遍：prompt 还原 messages → 四条规则奖励逐项累加 → Reward Model 打分 → 组内标准化出 advantage。

> 相关文档：函数整体讲解见 [[train_grpo详解]]，重复惩罚 `rep_penalty` 的独立算例见 [[train_grpo详解]] 第四节。

---

## 一、设定

```python
B = 1   # batch_size
G = 2   # num_generations（本算例改成 2，官方默认是 6）
N = B * G = 2   # rewards 张量的长度
```

输入参数：

```python
prompts   = [ "<|im_start|>user\n1+1=几<|im_end|>\n<|im_start|>assistant\n" ]   # 长度 B = 1
responses = [ r0, r1 ]                                                          # 长度 B*G = 2
```

其中两条采样回答：

```python
# r0（答对 + 认真思考）
r0 = "<think>用户问1加1等于几，这是最基本的加法，答案很明确。</think>1加1等于2。"

# r1（思考太短 + 复读机 + 答错）
r1 = "<think>让我想想。</think>不知道 不知道 不知道 不知道 不知道。"
```

---

## 二、函数签名与返回格式（先给结论）

```python
def calculate_rewards(prompts, responses, reward_model) -> torch.Tensor:
```

| 项 | 值 |
|---|---|
| 返回类型 | `torch.Tensor` |
| 形状 | `[B * num_generations]` = `[2]` |
| dtype | `torch.float32`（`torch.zeros` 默认） |
| device | `args.device`（如 `cuda:0`） |
| 元素语义 | `rewards[k]` 对应 `responses[k]`，是「规则奖励(①②③) + RM 分数(④)」之和 |
| 布局 | 第 i 个 prompt 的第 j 条回答在 `index = i*G + j`，即 `[prompt0_gen0, prompt0_gen1]` |

本算例最终返回：`tensor([4.25, -2.25])`。

---

## 三、逐行追踪

### 第 0 步：初始化（第 93 行）

```python
rewards = torch.zeros(2, device=args.device)   # shape [2]，全 0.0
reward_model_scores = []                        # 空列表，先暂存 RM 分数
batch_size = len(prompts)                       # = 1
```

整个打分包在 `with torch.no_grad():` 里（第 95 行）——reward model 不产生梯度、不参与训练。

### 第 1 步：还原 messages（第 109–111 行）

```python
pattern = r"<\|im_start\|>(system|user|assistant)\s+(.*?)<\|im_end\|>"
matches  = re.findall(pattern, prompt, re.DOTALL)
messages = [{"role": role, "content": content.strip()} for role, content in matches]
```

prompt 里只有 `user` 段是完整的（有 `<|im_end|>` 闭合），末尾的 `<|im_start|>assistant\n` 是**开着的**（答案留给模型生成、没有闭合标签），所以匹配不到。结果：

```python
messages = [{"role": "user", "content": "1+1=几"}]
```

> 注意：这一步在**双重循环里每次迭代都重复执行**，因为 `i` 恒为 0、prompt 恒为 `prompts[0]`，所以两次得到的 `messages` 完全相同。这是代码的小冗余，不影响结果。

---

### 第 2 步：迭代 ①（`j=0`，处理 `r0`，`response_idx=0`）

**逐项累加 `rewards[0]`**（对应第 115 / 123 / 125 / 130 / 133 行）：

| 行 | 项目 | 计算 | 结果 | rewards[0] 累计 |
|---|---|---|---|---|
| 115 | ① 长度奖励 | `len(r0.strip())` = **47** ∈ [20, 800] | +0.5 | 0.5 |
| 123 | ②-1 思考长度 | `thinking_content` = `<think>…明确。` = **32** 字符 ∈ [20, 300] | +1.0 | 1.5 |
| 125 | ②-2 标签格式 | `r0.count('</think>') == 1` | +0.25 | 1.75 |
| 130 | ③ 重复惩罚 | `answer = "1加1等于2。"` 分词仅 2 token < 3 → `rep_penalty = 0.0` | -0.0 | 1.75 |
| 133 | ④ RM 打分 | `get_score(messages, "1加1等于2。")`（答对，假设） | +2.5 | *(记入列表)* |

关键细节——第 120–127 行的拆分逻辑：

```python
if '</think>' in r0:                                   # True
    thinking_content, answer_content = r0.split('</think>', 1)
    # thinking_content = "<think>用户问1加1等于几，这是最基本的加法，答案很明确。"  (32 字符)
    # answer_content   = "1加1等于2。"                                            (7 字符)
    answer = answer_content.strip()                    # answer 被重写为 </think> 之后的部分
```

第 134 行把 RM 分数记入列表（**不直接加到 rewards**）：

```python
reward_model_scores.append(2.5)
```

所以迭代 ① 结束时：`rewards[0] = 1.75`（只含规则部分），`reward_model_scores = [2.5]`。

---

### 第 3 步：迭代 ②（`j=1`，处理 `r1`，`response_idx=1`）

messages 同上 → `[{"role":"user","content":"1+1=几"}]`。

**逐项累加 `rewards[1]`**：

| 行 | 项目 | 计算 | 结果 | rewards[1] 累计 |
|---|---|---|---|---|
| 115 | ① 长度奖励 | `len(r1.strip())` = **40** ∈ [20, 800] | +0.5 | 0.5 |
| 123 | ②-1 思考长度 | `thinking_content` = `<think>让我想想。` = **12** 字符 < 20 | -0.5 | 0.0 |
| 125 | ②-2 标签格式 | `r1.count('</think>') == 1` | +0.25 | 0.25 |
| 130 | ③ 重复惩罚 | `answer` 分词 6 个 token、4 个三元组去重后 2 种 → `(4-2)*0.5*2/4 = 0.5` | -0.5 | -0.25 |
| 133 | ④ RM 打分 | `get_score(messages, answer)`（答错，假设） | -2.0 | *(记入列表)* |

③ 的重复惩罚展开（`rep_penalty` 对最终答案 `answer` 操作）：

```python
answer = "不知道 不知道 不知道 不知道 不知道。"
toks   = ["不知道","不知道","不知道","不知道","不知道","。"]   # \w+ 连续匹配、标点单列，共 6 个
grams  = [不知道,不知道,不知道] × 3 + [不知道,不知道,。] × 1      # 滑窗 n=3，共 4 个三元组
set(grams) 去重后 2 种
rep_penalty = min(0.5, (4 - 2) * 0.5 * 2 / 4) = min(0.5, 0.5) = 0.5
```

第 134 行：`reward_model_scores.append(-2.0)`。

迭代 ② 结束时：`rewards[1] = -0.25`，`reward_model_scores = [2.5, -2.0]`。

---

### 第 4 步：循环结束后一次性加回 RM 分数（第 137–138 行）

```python
reward_model_scores = torch.tensor([2.5, -2.0], device=args.device)   # shape [2]
rewards += reward_model_scores
```

```
规则部分   = [1.75, -0.25]
+ RM 部分  = [2.5 , -2.0 ]
─────────────────────────
rewards    = [4.25, -2.25]
```

为什么 RM 分数要「先收集、循环外再一次性加」？因为 `get_score` 是逐个串行调用（每条回答一次前向），先把标量攒在 Python list 里，最后一把转成张量，比在循环里反复 `rewards[i] += score` 少一次设备间搬运。

---

## 四、返回数据格式（完整）

```python
return rewards
# 类型   : torch.Tensor
# 形状   : [B * num_generations] = [2]
# dtype  : torch.float32
# device : args.device
# 值     : tensor([4.25, -2.25])
# 布局   : rewards[0]=prompt0 的第 0 条(r0)，rewards[1]=prompt0 的第 1 条(r1)
```

---

## 五、返回后如何被消费（`grpo_train_epoch` 第 241–245 行）

`calculate_rewards` 返回后，训练循环立即做**组内标准化**算出 advantage：

```python
grouped = rewards.view(-1, 2)                  # [1, 2] = [[4.25, -2.25]]
mean_r  = grouped.mean(dim=1)                  # (4.25 + (-2.25)) / 2 = 1.0
std_r   = grouped.std(dim=1, unbiased=False)   # sqrt(((4.25-1)² + (-2.25-1)²)/2) = 3.25
advantages = (rewards - mean_r) / (std_r + 1e-4)
          = [3.25, -3.25] / 3.25
          = [1.0, -1.0]
```

即 `r0` 优势 **+1.0**（抬升生成概率）、`r1` 优势 **-1.0**（压低概率）——**同一 prompt 的两条回答互相对比**，而非全局比较。这正是 GRPO 不需要 Critic 网络的核心：优势来自组内相对排序。

> 注意：`unbiased=False` 是「组内全体 G 条的总体标准差」。当 `G=2` 时两条分数的差距恰好被 std 完全归一化，所以 advantage 恰好是 `±1.0`；`G=6` 时则各条按 `(r - mean)/std` 分布在不同值上（见 [[train_grpo详解]] 3.4 节的 6 条例子）。

---

## 六、四部分奖励速查表

| # | 奖励项 | 触发条件 | 分值范围 |
|---|---|---|---|
| ① | 长度奖励 | 整条回答（含思考）20~800 字符 | +0.5 / -0.5 |
| ②-1 | 思考长度 | 回答含 `</think>` 且思考段 20~300 字符 | +1.0 / -0.5 |
| ②-2 | 标签格式 | 恰好一个 `</think>` | +0.25 / -0.25 |
| ③ | 重复惩罚 | 最终答案 n-gram 重复率（`rep_penalty`） | [0, 0.5]，从总分**减去** |
| ④ | RM 打分 | 最终答案喂给 `internlm2-1_8b-reward` | [-3, 3]，内部 clamp |

关键设计：**④ 只对 `</think>` 之后的最终答案打分**，思考过程不打分，而是靠 ② 的规则奖励间接鼓励——这是 DeepSeek-R1 的「格式奖励 + 答案正确性奖励」分离设计。

---

## 七、一句话总结

`calculate_rewards` 对每条采样回答算一个标量 `reward = ①长度 + ②思考 + ③(-重复惩罚) + ④RM分数`，返回形状 `[B*G]` 的张量；`G=2` 的本例中 `r0`（认真思考答对）得 **+4.25**、`r1`（短思考复读答错）得 **-2.25**，组内标准化后 advantage 为 `[+1.0, -1.0]`，分别抬升/压低这两条回答的生成概率。
