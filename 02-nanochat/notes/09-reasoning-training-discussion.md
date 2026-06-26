# 大模型推理能力的训练来源

---

## 核心问题：推理能力是训练出来的吗？

**是的**，推理能力是训练出来的，但来源有两层：

### 1. 预训练阶段（无监督）

- 从海量文本中学习到 chain-of-thought 模式的统计规律（如数学推导、代码逻辑）
- 这是"隐式"推理能力的基础，来自数据中已有的推理示例

### 2. 后训练阶段（有监督 + 强化学习）

- **SFT**：用带推理过程的数据微调（如 GSM8K 中 step-by-step 解题），让模型学会"展示"推理
- **RL**：通过奖励信号（如答案正确性）强化正确的推理路径，淘汰错误路径

---

## 关键认知

- 预训练学到的是"推理的模式"，后训练学到的是"何时/如何使用推理"
- 所谓"涌现"能力（如 o1 的深度思考），本质是足够规模的预训练 + 精心设计的 RL 训练
- 模型并没有真正的逻辑推理，而是在学习"什么样的 token 序列能导向正确答案"的统计模式

---

## nanochat 中的实践印证

nanochat 的 SFT 训练数据（`scripts/chat_sft.py:165-173`）包含 6 种任务，其中 GSM8K 和 SpellingBee 的设计直接体现了"教模型推理"的思想：

### GSM8K（数学推理）

assistant 回复包含手动推导过程 + Python 验证：

```
User: Weng earns $12 an hour for babysitting. Yesterday, she just did 50 minutes of babysitting. How much did she earn?

Assistant:
  Weng earns 12/60 = $ [python: 12/60 → 0.2] per minute.
  Working 50 minutes, she earned 0.2 x 50 = $ [python: 0.2*50 → 10] 10.
  #### 10
```

### SpellingBee（字母计数推理）

assistant 回复包含手动逐字母计数 + Python 双重验证：

```
User: How many r are in strawberry?

Assistant:
  First spell the word out: strawberry:s,t,r,a,w,b,e,r,r,y
  Then count the occurrences of 'r':
    3:r hit! count=1
    8:r hit! count=2
    9:r hit! count=3
  This gives us 3.
  Let me double check this using Python: [python: 'strawberry'.count('r') → 3]
  Python gives us 3.
  #### 3
```

### 训练目标的 mask 设计

通过 `tokenizer.render_conversation()`（`tokenizer.py:266`），不同内容的 mask 值不同：

| 内容类型 | mask | 说明 |
|---------|------|------|
| user 消息 | 0 | 不训练 |
| assistant 文本 | 1 | 训练目标 |
| assistant python 调用 | 1 | 训练目标 |
| python_output（工具返回） | 0 | 不训练，运行时由环境产生 |

这确保模型学习"如何推理"而非"记忆工具输出"。

---

## nanochat SFT 训练数据详解

定义在 `scripts/chat_sft.py:165-173`，共 6 种任务，都继承自 `tasks/common.py:Task`，统一返回 `{"messages": [...]}` 格式的 conversation dict。

### 1. SmolTalk（`tasks/smoltalk.py`）— 460K 通用对话

```json
{
  "messages": [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "What is Python?"},
    {"role": "assistant", "content": "Python is a programming language..."}
  ]
}
```

- 来源：HuggingFaceTB/smol-smoltalk
- assistant content 为纯字符串

### 2. CustomJSON（`tasks/customjson.py`）— 1K 合成身份对话

```json
[
  {"role": "user", "content": "Who are you?"},
  {"role": "assistant", "content": "I am nanochat, a language model..."}
]
```

- 来源：identity_conversations.jsonl，每行一个 JSON 数组
- assistant content 为纯字符串
- 在 train_tasks 中加载了 2 遍（2 个 epoch）

### 3. MMLU（`tasks/mmlu.py`）— 100K 多选题

```json
{
  "messages": [
    {"role": "user", "content": "Multiple Choice question: What is 2+2?\n- 3=A\n- 4=B\n- 5=C\n- 6=D\n\nRespond only with the letter of the correct answer."},
    {"role": "assistant", "content": "B"}
  ],
  "subject": "elementary_mathematics",
  "letters": ["A","B","C","D"]
}
```

- 来源：cais/mmlu，格式由 `render_mc()` 渲染
- 选项放在字母前面（`- choice=letter`），方便小模型绑定
- assistant content 为纯字符串（单个字母）

### 4. GSM8K（`tasks/gsm8k.py`）— 8K 数学题（含 tool call）

```json
{
  "messages": [
    {"role": "user", "content": "Weng earns $12 an hour..."},
    {"role": "assistant", "content": [
      {"type": "text", "text": "Weng earns 12/60 = $"},
      {"type": "python", "text": "12/60"},
      {"type": "python_output", "text": "0.2"},
      {"type": "text", "text": "0.2 per minute. Working 50 minutes, she earned 0.2 x 50 = $"},
      {"type": "python", "text": "0.2*50"},
      {"type": "python_output", "text": "10"},
      {"type": "text", "text": "10.\n#### 10"}
    ]}
  ]
}
```

- 来源：openai/gsm8k
- assistant content 为 parts 列表，包含 text/python/python_output 三种 type
- 原始数据中 `<<expr=result>>` 格式的工具调用被解析为结构化 parts

### 5. SimpleSpelling（`tasks/spellingbee.py:233`）— 200K 单词拼写

```json
{
  "messages": [
    {"role": "user", "content": "Spell the word: apple"},
    {"role": "assistant", "content": "apple:a,p,p,l,e"}
  ]
}
```

- 简化任务，专门练习 token → 字符序列的映射
- assistant content 为纯字符串

### 6. SpellingBee（`tasks/spellingbee.py:115`）— 80K 字母计数（含 tool call）

```json
{
  "messages": [
    {"role": "user", "content": "How many r are in strawberry?"},
    {"role": "assistant", "content": [
      {"type": "text", "text": "We are asked to find the number 'r'..."},
      {"type": "python", "text": "'strawberry'.count('r')"},
      {"type": "python_output", "text": "3"},
      {"type": "text", "text": "\n\nPython gives us 3.\n\nMy final answer is:\n\n#### 3"}
    ]}
  ]
}
```

- 包含 30+ 种用户提问模板（含中日韩法德西等语言），用于数据增强
- assistant 先手动逐字母计数，再用 Python 验证
- 90% 情况下从单词中选字母，10% 随机字母

### content 类型总结

| 类型 | 格式 | 示例任务 |
|-----|------|---------|
| 纯字符串 | `"content": "text"` | SmolTalk, CustomJSON, MMLU, SimpleSpelling |
| parts 列表 | `"content": [{"type": ..., "text": ...}]` | GSM8K, SpellingBee |

parts 中的 type 有三种：
- `text`：普通文本
- `python`：Python 表达式（tool call）
- `python_output`：Python 执行结果

---

## 关键函数调用链

### 模型构建与加载

```
chat_sft.py:96  load_model("base", device, phase="train", ...)
    ↓
checkpoint_manager.py:164  load_model()
    ↓
checkpoint_manager.py:77   build_model()
    ↓
checkpoint_manager.py:101  model = GPT(model_config)   # GPT 定义在 nanochat/gpt.py:154
```

### model(x, y)（`chat_sft.py:433`）

调用 `GPT.forward()`（`nanochat/gpt.py:416`）：

```python
def forward(self, idx, targets=None, kv_cache=None, loss_reduction='mean'):
```

- `x` → `idx`：输入 token ids，shape `(B, T)`
- `y` → `targets`：目标 token ids，shape `(B, T)`
- 当 `targets` 不为 `None` 时，forward 内部计算 cross-entropy loss 并返回标量
- forward 内部流程：token embedding → smear → transformer blocks → unembedding → loss

### model.train()（`chat_sft.py:361`）

- `GPT` 类**没有重写** `train()`，直接调用 PyTorch 的 `nn.Module.train()`
- 将所有子模块的 `training` 属性设为 `True`
- 影响 Dropout 等模块的行为（启用训练模式）

### model.eval()（`chat_sft.py:348`）

- 同样调用 PyTorch 的 `nn.Module.eval()`
- 等价于 `model.train(False)`
- 将所有子模块的 `training` 属性设为 `False`，切换到推理模式

### tokenizer.render_conversation()（`tokenizer.py:266`）

```python
def render_conversation(self, conversation, max_tokens=2048):
    # 返回: ids (token id 序列), mask (训练目标标记)
```

将 conversation dict 转换为模型训练所需的 token 序列：

1. **system 消息**：合并到紧跟的 user 消息中
2. **user 消息**：包裹 `<|user_start|>...<|user_end|>`，mask = 0（不训练）
3. **assistant 消息**：包裹 `<|assistant_start|>...<|assistant_end|>`，mask = 1（训练目标）
   - 纯字符串：直接编码
   - parts 列表：
     - `type="text"` → mask = 1
     - `type="python"` → mask = 1（包裹在 `<|python_start|>...<|python_end|>`）
     - `type="python_output"` → mask = 0（包裹在 `<|output_start|>...<|output_end|>`，不训练）
4. 截断到 `max_tokens`（默认 2048）

效果：模型只在 assistant 的回复内容（不含 python_output）上计算 loss。

---

## 总结

推理能力 = 预训练的模式学习 + SFT 的推理过程示范 + RL 的正确路径强化。nanochat 的 GSM8K 和 SpellingBee 任务就是典型的"用 SFT 教模型先想再算再验证"的实践案例。
