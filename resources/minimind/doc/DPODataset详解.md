# DPODataset 详解

`DPODataset` 定义在 `dataset/lm_dataset.py:244-314`，是 **DPO（Direct Preference Optimization，直接偏好优化）** 阶段的数据集，负责把偏好对数据渲染、编码成训练所需的 `(x, y, mask)` 张量。本文逐段拆解它的构造与工作流程。

> 相关文档：训练脚本见 [[train_dpo详解]]，DPO/GRPO/PPO 对比见 [[DPO_GRPO_PPO对比]]。

## 一、DPO 在干嘛？为什么数据结构不一样

- **SFT**：一条样本是「用户问 → 助手答」，模型学「怎么生成好答案」。
- **DPO**：一条样本是「用户问 → 一个好答案（chosen）+ 一个差答案（rejected）」，模型学的是「**相对偏好**」——要抬高 chosen 的概率、压低 rejected 的概率。

所以 DPO 数据里每行有 `chosen` 和 `rejected` 两个字段，各是一段完整的对话（`list` of `{role, content}`）。DPO 的 loss 形如 `log(πθ(chosen)/πθ(rejected))`，它需要 **两条序列各自的 log-prob**，因此 `__getitem__` 要同时产出 chosen 和 rejected 两套张量。

## 二、`__init__`（244-252 行）

```python
self.padding = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
self.bos_id = tokenizer(f'{tokenizer.bos_token}assistant\n', add_special_tokens=False).input_ids
self.eos_id = tokenizer(f'{tokenizer.eos_token}\n', add_special_tokens=False).input_ids
```

- `padding`：pad token id，做了一次容错兜底（没有 pad token 就用 0）。注意下面 `generate_loss_mask` **并没有直接用到** `self.padding`，只在构造时存下备用。
- `bos_id`：`'<|im_start|>assistant\n'` 的 token 序列，是「**assistant 回合开始**」的精确记号，与 `SFTDataset.__init__` 里完全一致。因为模板里只有 assistant 回合后面紧跟 `assistant`，所以它只命中回答段开头，不会误中 system/user 回合。
- `eos_id`：`'<|im_end|>\n'` 的 token 序列，标记一个回合结束。

三者语义与 `SFTDataset` 完全一致——都是为「定位 assistant 回答段」服务。

## 三、`__getitem__`（257-296 行）——核心流水线

### 第 1 步：取样本、渲染成文本（259-269）

```python
chosen   = sample['chosen']      # 好答案的完整对话
rejected = sample['rejected']    # 差答案的完整对话
chosen_prompt = self.tokenizer.apply_chat_template(
    chosen, tokenize=False, add_generation_prompt=False)
chosen_prompt = post_processing_chat(chosen_prompt)
```

- `apply_chat_template(..., tokenize=False)`：把结构化对话套模板渲染成纯文本字符串。`add_generation_prompt=False` 因为训练数据里答案已存在，不需要末尾再留空 assistant 头。
- `post_processing_chat`：以 80% 概率移除空的 `<think>\n\n</think>\n\n` 思考标签。

### 第 2 步：tokenize + 补齐（270-275）

```python
chosen_encoding = self.tokenizer(
    chosen_prompt, truncation=True, max_length=self.max_length, padding='max_length')
```

这里和 SFT 有**关键区别**：SFT 是「手动截断 + 手动补齐」，DPO 直接把 `truncation=True` + `padding='max_length'` 交给 tokenizer 一步做完——截断到 4096 并右补齐到 4096。

### 第 3 步：生成 loss mask（278-281）

```python
chosen_loss_mask = self.generate_loss_mask(chosen_input_ids)
```

对两条序列分别生成 mask，标记「哪些位置是 assistant 回答段」。

### 第 4 步：错位构造 x / y / mask（282-287）

```python
x_chosen    = torch.tensor(chosen_input_ids[:-1])   # 输入：去掉最后一个 token
y_chosen    = torch.tensor(chosen_input_ids[1:])    # 目标：去掉第一个 token
mask_chosen = torch.tensor(chosen_loss_mask[1:])    # mask 同步错位
```

这是 next-token 预测的标准错位：用位置 `i` 的 token 预测位置 `i+1` 的 token。所以 `x` 是 `[0 : n-1]`，`y` 是 `[1 : n]`，mask 也和 `y` 对齐、切成 `[1:]`。

> 注意：`SFTDataset` 返回 `(input_ids, labels)`，错位在模型 `forward` 里做；而 `DPODataset` 直接在数据层就把错位做好了，因为 DPO loss 要分别算 chosen/rejected 的序列 log-prob，需要在数据侧就给出对齐好的 `(x, y, mask)`。

### 第 5 步：返回 dict（289-296）

最终返回 6 个张量：`x_chosen / y_chosen / mask_chosen / x_rejected / y_rejected / mask_rejected`，每个都是长度 `max_length - 1` 的 long 张量。

## 四、`generate_loss_mask`（298-314 行）——回答段定位算法

这是整个类的核心逻辑，和 `SFTDataset.generate_labels` 是**同一套扫描算法**，只是输出从「填真实 id」变成了「填 0/1 掩码」：

```python
loss_mask = [0] * len(input_ids)   # 初始全 0（默认不参与 loss）
i = 0
while i < len(input_ids):
    if input_ids[i:i + len(self.bos_id)] == self.bos_id:   # 命中 '<|im_start|>assistant\n'
        start = i + len(self.bos_id)                        # 回答内容从 bos_id 之后开始
        end = start
        while end < len(input_ids):                          # 向后找 '<|im_end|>\n'
            if input_ids[end:end + len(self.eos_id)] == self.eos_id:
                break
            end += 1
        for j in range(start, min(end + len(self.eos_id), self.max_length)):
            loss_mask[j] = 1                                 # 回答段（含结束标记）标 1
        i = end + len(self.eos_id) if end < len(input_ids) else len(input_ids)
    else:
        i += 1
return loss_mask
```

**逐步解释：**

1. `loss_mask` 初始化为全 0。0 = 不参与 loss，1 = 参与 loss。
2. 线性扫描 `input_ids`，找每个 `bos_id`（assistant 回合开头）：
   - 命中后，`start` 指向回答内容开头（bos_id 之后）。
   - 从 `start` 继续往后找 `eos_id`（回合结束标记），找到就 `break`。
   - 把 `[start, end + len(eos_id))` 这段全部标 `1`——**结束标记本身也算进 loss**，与 SFT 的 `generate_labels` 一致。
3. `i` 直接跳到 `end + len(eos_id)`，跳过整个回答段，继续找下一段 assistant 回答（多轮对话里会有多段）。
4. 没命中 `bos_id` 就 `i += 1` 继续扫。

### 和 `generate_labels` 的对比

| | `SFTDataset.generate_labels` | `DPODataset.generate_loss_mask` |
|---|---|---|
| 初始值 | `-100`（CrossEntropyLoss 的 ignore_index） | `0` |
| 回答段标记 | 填真实 token id | 填 `1` |
| 非回答段 | `-100` | `0` |
| 用途 | 直接当 `labels` 喂 CrossEntropyLoss | 独立的 mask，配合 DPO 的 log-prob 计算 |

DPO 为什么用 0/1 mask 而不是 -100：DPO loss 不直接调用 CrossEntropyLoss，而是自己算序列的对数概率 `log p(y|x)`，需要显式的 `mask` 张量来「只累加回答段的 log-prob、屏蔽 system/user/padding 段」。

## 五、一个具体例子

假设一条 chosen 样本，`max_length=4096`，渲染 + tokenize 后（示意）：

```
idx:  0         1    2    3    4    5         6    7    8
id :  <im_start>user 你  好  <im_end> <im_start>assistant 你好 <im_end> ... 0 0 (pad)
```

`generate_loss_mask` 扫到 `bos_id = <im_start>assistant` 后，把 assistant 内容 `你好<im_end>` 标 1，其余（user 段、pad）全 0。再错位切成 `x/y/mask`，最终喂给 DPO 训练循环。

## 六、小结

| 组件 | 作用 |
|---|---|
| `bos_id` / `eos_id` | 定位 assistant 回答段的「起止锚点」 |
| `generate_loss_mask` | 把「回答段」标成 1 的 0/1 掩码，屏蔽 system/user/padding |
| `__getitem__` 的错位 | 在数据侧完成 next-token 的 x→y 对齐 |
| 双序列输出 | 同时产出 chosen/rejected 两套 `(x, y, mask)`，供 DPO loss 比较相对好坏 |

它本质上就是把 SFT 的「labels 掩码」改造成「0/1 mask」，再把「单序列输出」扩展成「chosen + rejected 双序列输出」，从而适配 DPO 的偏好学习目标。

---

## 附：DPO 数据下载

DPO 训练数据为 `dataset/dpo.jsonl`，抽样自 [DPO-En-Zh-20k](https://huggingface.co/datasets/llamafactory/DPO-En-Zh-20k)，数据集托管在 ModelScope / HuggingFace：

- ModelScope：`gongjy/minimind_dataset`
- HuggingFace：`jingyaogong/minimind_dataset`

数据格式（每条一行 JSON）：

```json
{
  "chosen": [
    {"content": "Q", "role": "user"},
    {"content": "good answer", "role": "assistant"}
  ],
  "rejected": [
    {"content": "Q", "role": "user"},
    {"content": "bad answer", "role": "assistant"}
  ]
}
```

用 ModelScope 单独下载 `dpo.jsonl`（约 53MB，17,166 条）：

```bash
cd dataset/
modelscope download --dataset gongjy/minimind_dataset dpo.jsonl --local_dir .
```

文件放置路径：`dataset/dpo.jsonl`，对应 `train_dpo.py` 的默认参数 `--data_path ../dataset/dpo.jsonl`。
