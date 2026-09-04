"""
MiniMind Agent 强化学习（RL）训练脚本：训练模型学会「调用工具」（function calling）。

与 SFT（train_sft.py 等）的本质区别：
    SFT     ：直接拿 (输入, 标准答案) 做 next-token 预测，是「有监督」的模仿学习；
    Agent RL：不给标准答案，而是让模型**自己采样**若干条回答 → 按规则/reward 打分 →
              用 GRPO/CISPO 目标（带 KL 约束）更新策略，是「试错学习」。

整体流水线：
    agent_rl.jsonl ─► AgentRLDataset(messages/tools/gt)
                     ─► rollout 引擎：模型多轮自回归 + 模拟执行工具
                     ─► 打包 prompt+response → 算 π_θ/π_ref 的 logprob
                     ─► 按规则算 reward → GRPO 组内标准化得 advantage
                     ─► GRPO/CISPO loss + KL 惩罚 → 反向传播更新策略
                     ─► rollout 引擎同步新权重，进入下一轮

三个模型各司其职：
    model        : 策略模型 π_θ，被训练的对象
    ref_model    : 参考模型 π_ref（冻结），用于算 KL 惩罚，防止策略跑太远
    reward_model : 外部奖励模型（internlm2-1.8b），只在「无工具调用」分支给回答打分
"""

import os
import sys

__package__ = "trainer"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import datasets  # noqa: F401  # Windows pyarrow/torch DLL conflict workaround (issue #771)
import re
import gc
import json
import math
import random
import signal
import argparse
import warnings
import torch
import torch.nn.functional as F
import torch.distributed as dist
from contextlib import nullcontext
from torch import optim
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler
from torch.optim.lr_scheduler import CosineAnnealingLR
from transformers import AutoTokenizer
from model.model_minimind import MiniMindConfig, MiniMindForCausalLM
from dataset.lm_dataset import AgentRLDataset
from trainer.trainer_utils import Logger, is_main_process, lm_checkpoint, init_distributed_mode, setup_seed, SkipBatchSampler, init_model, LMForRewardModel
from trainer.rollout_engine import create_rollout_engine, compute_per_token_logps

warnings.filterwarnings('ignore')

# ================================ 工具与 Reward = Start ================================

def rep_penalty(text, n=3, cap=0.5):
    """
    重复惩罚（repetition penalty）：惩罚「车轱辘话」式输出，越重复扣分越多。

    算法：
        1) 分词：`\w+`（连续字母/数字）或 `[^\w\s]`（单个标点符号），统一转小写；
        2) 滑窗取 n-gram（默认三元组 n=3）；
        3) 重复率 = (总 gram 数 - 去重后的 gram 数) / 总 gram 数；
        4) 惩罚 = 重复率 * (2*cap)，再截断到 cap 上限（默认 0.5）。

    例（n=2，便于示意）：text = "天气很好天气很好"
        分词 toks = ["天气", "很好", "天气", "很好"]
        2-gram   = [("天气","很好"), ("很好","天气"), ("天气","很好")]   共 3 个
        去重后   = {("天气","很好"), ("很好","天气")}                    共 2 个
        重复率    = (3 - 2) / 3 ≈ 0.333
        惩罚      = min(0.5, 0.333 * 2 * 0.5) ≈ 0.333

    边界：完全无重复 → 惩罚 0；完全重复 → 趋近 cap=0.5；空文本（无 gram）→ 0.0。
    """
    toks = re.findall(r"\w+|[^\w\s]", text.lower())
    grams = [tuple(toks[i:i + n]) for i in range(len(toks) - n + 1)]
    return min(cap, (len(grams) - len(set(grams))) * cap * 2 / len(grams)) if grams else 0.0

# ======== 工具定义 ========
# 6 个 OpenAI 风格的 function 定义。模型通过生成 <tool_call>{json}</tool_call> 来「调用」它们。
# 每个工具的 parameters 是 JSON Schema，会被 apply_chat_template(tools=...) 渲染成 system
# 消息里的「工具声明」，告诉模型：可以调哪些工具、每个工具需要什么参数、参数是什么类型。
TOOLS = [
    {"type": "function", "function": {"name": "calculate_math", "description": "计算数学表达式", "parameters": {"type": "object", "properties": {"expression": {"type": "string"}}, "required": ["expression"]}}},
    {"type": "function", "function": {"name": "unit_converter", "description": "单位换算", "parameters": {"type": "object", "properties": {"value": {"type": "number"}, "from_unit": {"type": "string"}, "to_unit": {"type": "string"}}, "required": ["value", "from_unit", "to_unit"]}}},
    {"type": "function", "function": {"name": "get_current_weather", "description": "获取天气", "parameters": {"type": "object", "properties": {"location": {"type": "string"}}, "required": ["location"]}}},
    {"type": "function", "function": {"name": "get_current_time", "description": "获取时间", "parameters": {"type": "object", "properties": {"timezone": {"type": "string", "default": "Asia/Shanghai"}}, "required": []}}},
    {"type": "function", "function": {"name": "get_exchange_rate", "description": "查询汇率", "parameters": {"type": "object", "properties": {"from_currency": {"type": "string"}, "to_currency": {"type": "string"}}, "required": ["from_currency", "to_currency"]}}},
    {"type": "function", "function": {"name": "translate_text", "description": "翻译文本", "parameters": {"type": "object", "properties": {"text": {"type": "string"}, "target_language": {"type": "string"}}, "required": ["text", "target_language"]}}},
]

# ======== 模拟数据 ========
# 没有真实后端，用写死的字典模拟外部服务返回值，execute_tool 时按 key 查表即可。
# 例：WEATHER_DATA["北京"] = ("28°C", "晴") → 温度 28°C、天气晴。
WEATHER_DATA = {"北京": ("28°C", "晴"), "上海": ("15°C", "多云"), "广州": ("32°C", "闷热"), "深圳": ("30°C", "晴"), "杭州": ("22°C", "阴"), "成都": ("18°C", "小雨"), "武汉": ("25°C", "多云"), "南京": ("20°C", "晴"), "西安": ("16°C", "大风"), "重庆": ("26°C", "阴"), "Tokyo": ("12°C", "晴"), "New York": ("8°C", "多云"), "London": ("5°C", "小雨"), "Paris": ("10°C", "阴"), "Sydney": ("25°C", "晴朗")}
TIME_DATA = {"Asia/Shanghai": "2025-03-07 14:30:00", "America/New_York": "2025-03-07 01:30:00", "Europe/London": "2025-03-07 06:30:00", "Asia/Tokyo": "2025-03-07 15:30:00", "Europe/Paris": "2025-03-07 07:30:00", "Australia/Sydney": "2025-03-07 17:30:00"}
EXCHANGE_DATA = {("USD", "CNY"): 7.21, ("EUR", "CNY"): 7.85, ("GBP", "CNY"): 9.12, ("JPY", "CNY"): 0.048, ("USD", "EUR"): 0.92, ("USD", "GBP"): 0.79, ("CNY", "JPY"): 20.83, ("AUD", "CNY"): 4.72}
TRANSLATE_DATA = {("你好世界", "english"): "Hello World", ("Good morning", "chinese"): "早上好", ("今天天气真好", "english"): "The weather is nice today", ("I love programming", "chinese"): "我喜欢编程", ("机器学习很有趣", "english"): "Machine learning is interesting", ("Happy birthday", "chinese"): "生日快乐"}
UNIT_DATA = {"km_miles": 0.621371, "miles_km": 1.60934, "kg_pounds": 2.20462, "pounds_kg": 0.453592, "meters_feet": 3.28084, "feet_meters": 0.3048, "celsius_fahrenheit": 1.8, "fahrenheit_celsius": 0.5556}

# ======== 模拟执行 ========
# 每个「工具名」→ 一个 lambda(args)：根据参数查上面的模拟数据，返回 dict 结果。
# 例：execute_tool("get_current_weather", {"location": "北京"})
#      → 查 WEATHER_DATA["北京"] → {"city":"北京","temperature":"28°C","humidity":"65%","condition":"晴"}
#
# 注意 calculate_math 用受限命名空间的 eval（{"__builtins__": {}, "math": math}）算数学表达式，
# 即只暴露 math 模块、屏蔽内置函数，安全性相对可控；且 execute_tool 里还有 1 秒超时兜底。
MOCK_RESULTS = {
    "calculate_math": lambda args: {"result": str(eval(str(args.get("expression", "0")).replace("^", "**").replace("×", "*").replace("÷", "/").replace("−", "-").replace("（", "(").replace("）", ")"), {"__builtins__": {}, "math": math}))},
    "unit_converter": lambda args: {"result": round(float(args.get("value", 0)) * UNIT_DATA.get(f"{args.get('from_unit', '').lower()}_{args.get('to_unit', '').lower()}", 1), 4)},
    "get_current_weather": lambda args: (lambda w: {"city": args.get("location"), "temperature": w[0], "humidity": "65%", "condition": w[1]})(WEATHER_DATA.get(args.get("location"), ("22°C", "晴"))),
    "get_current_time": lambda args: {"datetime": TIME_DATA.get(args.get("timezone", "Asia/Shanghai"), "2025-03-07 14:30:00"), "timezone": args.get("timezone", "Asia/Shanghai")},
    "get_exchange_rate": lambda args: {"from": args.get("from_currency"), "to": args.get("to_currency"), "rate": EXCHANGE_DATA.get((args.get("from_currency"), args.get("to_currency")), 1.0)},
    "translate_text": lambda args: {"translated_text": TRANSLATE_DATA.get((args.get("text"), args.get("target_language")), args.get("text", ""))},
}

# ======== 参数校验 ========
# 每个「工具名」→ 一个校验 lambda，判断模型给出的参数是否「齐全且合法」。
# 这是 reward 里「参数合法性」判据：合法调用才计入 valid_call_count（见 calculate_rewards）。
# 例：CHECK_ARGS["get_current_weather"]({"location":"北京"}) → True
#     CHECK_ARGS["get_current_weather"]({})                        → False（缺 location）
#     CHECK_ARGS["get_current_time"]({})                           → True （timezone 有默认值，允许缺省）
CHECK_ARGS = {
    "calculate_math": lambda a: bool(a.get("expression")),
    "unit_converter": lambda a: a.get("value") is not None and a.get("from_unit") and a.get("to_unit"),
    "get_current_weather": lambda a: bool(a.get("location")),
    "get_current_time": lambda a: True,
    "get_exchange_rate": lambda a: bool(a.get("from_currency")) and bool(a.get("to_currency")),
    "translate_text": lambda a: bool(a.get("text")) and bool(a.get("target_language")),
}

# ======== 工具调用解析与执行 ========
def parse_tool_calls(text):
    """
    从模型生成文本中解析所有 <tool_call>...</tool_call> 块，每个块内容是一个 JSON 对象。

    例：text = "我来算一下\n<tool_call>{"name":"calculate_math","arguments":{"expression":"1+2*3"}}</tool_call>"
        返回 [{"name":"calculate_math", "arguments":{"expression":"1+2*3"}}]

    说明：
        - `re.DOTALL`：让 `.` 匹配换行，因为 JSON 内容可能跨行；
        - 一段文本可含多个 <tool_call> 块（一次调用多个工具），逐个解析；
        - JSON 解析失败（模型输出脏格式）时 `except: pass` 静默跳过该块，不做异常处理。
    """
    calls = []
    for m in re.findall(r'<tool_call>(.*?)</tool_call>', text, re.DOTALL):
        try: calls.append(json.loads(m.strip()))
        except: pass
    return calls

def execute_tool(name, args):
    """
    执行工具：查 MOCK_RESULTS 拿到对应实现并调用，用 SIGALRM 信号做 1 秒超时保护。

    流程：
        1) 查不到该工具名 → 直接返回 None（上层兜底成 '{"error": "tool not found"}'）；
        2) signal.alarm(1)：1 秒后触发 SIGALRM，抛 TimeoutError（防止 eval 死循环等）；
        3) finally 里 signal.alarm(0) 取消定时器。

    超时/执行异常 → 返回 None。
    注意：SIGALRM 仅 Unix 可用，Windows 下会静默失败（不影响训练主流程，只是少了超时保护）。
    """
    fn = MOCK_RESULTS.get(name)
    if not fn: return None
    try:
        signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(TimeoutError()))
        signal.alarm(1)
        return fn(args)
    except:
        return None
    finally:
        try: signal.alarm(0)
        except: pass

# ======== 多轮 Rollout ========
def rollout_single(rollout_engine, tokenizer, messages, tools, max_turns=3, max_new_tokens=256, thinking_ratio=0.5, device="cuda"):
    """
    单条样本的「多轮」自回归采样：模型可能多轮调用工具，直到给出最终答案或达到 max_turns。

    核心约定：只有「模型自己生成的 token」才算 response（response_mask=1，参与 loss）；
             工具返回的「观察 token」不算（response_mask=0，不参与 loss）。

    完整多轮例子（max_turns=3）：
        messages 初始 = [
            {"role":"system", "content":"你是助手..."},   # tools 会被模板渲染进 system
            {"role":"user",   "content":"北京今天天气怎么样？适合跑步吗"},
        ]

        第 1 轮：apply_chat_template(..., add_generation_prompt=True) 渲染成
            "<|im_start|>system ...<|im_end|>\n<|im_start|>user 北京今天...<|im_end|>\n<|im_start|>assistant\n"
            模型生成：<tool_call>{"name":"get_current_weather","arguments":{"location":"北京"}}</tool_call>
            解析到 1 个 tool call → execute_tool 执行 → result={"city":"北京","temperature":"28°C",...}
            messages 追加：{"role":"assistant", content:"<tool_call>..."} + {"role":"tool", content:result}

        第 2 轮：重新渲染（此时已含上轮的工具结果），再生成
            模型生成："北京今天28°C，晴，湿度65%，很适合跑步。"

        第 3 轮：这一轮生成里没有再出现 <tool_call> → 命中 `if not calls: break`，循环结束。

    最终累积：
        response_ids       = 第1轮生成 token + 工具观察 token + 第2轮生成 token
        response_mask      = 1 1 1 ... 0 0 0 ... 1 1 1 ...（模型生成=1，工具观察=0）
        response_old_logps = 对应位置 logprob（观察位置补 0.0）

    返回值（见函数末尾 return）：
        final_output     : 最后一轮的生成文本
        final_context    : 最终完整上下文（含工具结果）
        prompt_ids       : 第一轮的 prompt token（只记一次，后面轮次的增量算进 response）
        response_ids/mask/old_logps : 模型生成+工具观察的累积
        all_outputs      : 每一轮的生成文本列表
        unfinished       : 最后一轮是否仍在调工具（用于 reward 扣分）
    """
    all_outputs = []      # 记录每一轮的生成文本
    prompt_ids = None     # 第一轮的 prompt token，只记一次
    response_ids = []     # 累积：模型生成的 token + 工具观察 token
    response_mask = []    # 累积：1=模型生成(算loss)，0=工具观察(不算loss)
    response_old_logps = []  # 累积：每个 token 的旧 logprob（观察位置补 0.0）
    final_context = ""    # 最终完整上下文文本
    unfinished = False    # 是否「最后一轮仍在调工具」而没给出最终答案
    open_thinking = random.random() < thinking_ratio  # 按 thinking_ratio 概率开启 thinking 模板
    for turn in range(max_turns):
        # 1. 渲染上下文：add_generation_prompt=True 会在末尾补 "<|im_start|>assistant\n"，
        #    引导模型从 assistant 头开始生成（而不是续写 user 的话）。
        context = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, tools=tools, open_thinking=open_thinking)
        inputs = tokenizer(context, return_tensors="pt", add_special_tokens=False).to(device)
        context_ids = inputs["input_ids"][0].tolist()
        if prompt_ids is None:
            prompt_ids = context_ids  # 只有第一轮记作 prompt，后续轮次的增量算 response
        # 2. 采样生成：让策略模型生成一段新文本，并记录每个 token 的 logprob（后续算 ratio 用）
        rollout_result = rollout_engine.rollout(
            prompt_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            num_generations=1,
            max_new_tokens=max_new_tokens,
            temperature=0.8,
        )
        new_ids = rollout_result.completion_ids[0].tolist()
        new_logps = rollout_result.per_token_logps[0].tolist()
        if len(new_ids) != len(new_logps): Logger(f"rollout token/logprob length mismatch: {len(new_ids)} vs {len(new_logps)}")
        # 3. 过滤 pad/eos：这两个 token 不参与 loss，用 zip 对齐同步过滤 id 和 logprob，
        #    保证两者长度始终一致（否则后面 per-token 计算会错位）。
        pairs = [(t, lp) for t, lp in zip(new_ids, new_logps) if t != tokenizer.pad_token_id and t != tokenizer.eos_token_id]
        new_ids = [t for t, _ in pairs]
        new_logps = [lp for _, lp in pairs]
        new_text = rollout_result.completions[0]
        all_outputs.append(new_text)
        # 4. 累积本轮「模型生成的 token」：mask 全 1，表示这些是要参与 loss 的响应段
        response_ids.extend(new_ids)
        response_mask.extend([1] * len(new_ids))
        response_old_logps.extend(new_logps)
        final_context = context + new_text
        # 5. 解析工具调用：本轮没出现 <tool_call> 说明模型已给出最终答案 → 结束循环
        calls = parse_tool_calls(new_text)
        if not calls:
            break
        # 6. 有工具调用：标记「是否还没完成」，把 assistant 生成 + 每个工具结果追加进 messages
        unfinished = turn == max_turns - 1
        messages.append({"role": "assistant", "content": new_text})
        for call in calls:
            name, raw = call.get("name", ""), call.get("arguments", {})
            if isinstance(raw, str):          # arguments 可能是字符串形式，先解析成 dict
                try: raw = json.loads(raw)
                except: raw = {}
            result = execute_tool(name, raw)
            # 工具结果转 JSON 字符串；截断到 2048 防止「天文数字」撑爆 tokenizer
            result_str = (json.dumps(result, ensure_ascii=False) if result else '{"error": "tool not found"}')[:2048]  # 防止天文数字撑爆tokenizer
            messages.append({"role": "tool", "content": result_str})

        # 7. 重新渲染「观察上下文」（含刚得到的工具结果）。add_generation_prompt=not unfinished：
        #    若还没到最后一轮，末尾继续补 assistant 头；若已是最后一轮，则不再补（因为不会再生成）。
        observe_context = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=not unfinished, tools=tools, open_thinking=open_thinking)
        observe_ids = tokenizer(observe_context, return_tensors="pt", add_special_tokens=False)["input_ids"][0].tolist()
        # 8. 只取「新增」部分（工具结果那一段）作为观察 token：mask 全 0，不参与 loss
        current_len = len(prompt_ids) + len(response_ids)
        obs_delta = observe_ids[current_len:]
        response_ids.extend(obs_delta)
        response_mask.extend([0] * len(obs_delta))
        response_old_logps.extend([0.0] * len(obs_delta))
        final_context = observe_context

    final_output = all_outputs[-1] if all_outputs else ""
    prompt_ids = prompt_ids or []
    return final_output, final_context, prompt_ids, response_ids, response_mask, response_old_logps, list(all_outputs), unfinished

def rollout_batch(rollout_engine, tokenizer, messages_batch, tools_batch, num_gen, max_turns=3, max_new_tokens=256, thinking_ratio=0.5, device="cuda"):
    """
    批量 rollout：对每个 (messages, tools) 重复 num_gen 次 rollout_single，结果摊平成一维列表。

    num_gen 是「每个 prompt 生成几条回答」，用于 GRPO 组内对比（见 rl_train_epoch 的 advantages）。

    例：messages_batch 有 3 条，num_gen=4 → 共 3*4=12 条 rollout，
        索引映射：idx 0~3 属于 prompt0，4~7 属于 prompt1，8~11 属于 prompt2（idx // num_gen 可反查）。
    """
    all_completions = []
    all_contexts = []
    all_prompt_ids = []
    all_response_ids = []
    all_response_masks = []
    all_response_old_logps = []
    all_turn_outputs = []
    all_unfinished = []
    for messages, tools in zip(messages_batch, tools_batch):
        for _ in range(num_gen):
            msgs_copy = [dict(m) for m in messages]
            completion, context, prompt_ids, response_ids, response_mask, response_old_logps, turn_outputs, unfinished = rollout_single(rollout_engine, tokenizer, msgs_copy, tools, max_turns, max_new_tokens, thinking_ratio, device)
            all_completions.append(completion)
            all_contexts.append(context)
            all_prompt_ids.append(prompt_ids)
            all_response_ids.append(response_ids)
            all_response_masks.append(response_mask)
            all_response_old_logps.append(response_old_logps)
            all_turn_outputs.append(turn_outputs)
            all_unfinished.append(unfinished)
    return all_completions, all_contexts, all_prompt_ids, all_response_ids, all_response_masks, all_response_old_logps, all_turn_outputs, all_unfinished

# ======== Reward 计算 ========
def validate_gt_in_text(text, gt_list):
    """
    校验 Ground-Truth（标准答案）是否出现在模型回答里，返回「命中的 GT」集合。

    命中规则（满足其一即可）：
        1) 字符串包含：GT 的字符串形式（去空格、转小写）出现在回答文本里；
        2) 数值匹配：GT 是纯数字，且与回答里提取到的某个数字误差 < 1e-6。

    例：text = "答案是 42，天气 28 度"，gt_list = ["42", 28]
        提取数字（去掉逗号后）nums = [42.0, 28.0]
        "42" → 字符串包含 ✓（"42" 在文本里）
        28    → 数值匹配 ✓（|28 - 28.0| < 1e-6）
        返回 {"42", 28}

    正则 `(?<![\w.])` 负向后行断言 + `(?![\w.])` 负向前行断言：保证提取「独立」的数字，
    避免把 3.14 里的 3 和 14 拆成两个数。
    """
    text, text_num = str(text), str(text).replace(',', '')
    nums = [float(x) for x in re.findall(r'(?<![\w.])[-+]?\d+(?:\.\d+)?(?![\w.])', text_num)]
    return {g for g in gt_list if ((s := str(g).strip()) and s.lower() in text.lower()) or (re.fullmatch(r'[-+]?\d+(?:\.\d+)?', str(g).strip().replace(',', '')) and any(abs(float(str(g).strip().replace(',', '')) - n) < 1e-6 for n in nums))}

def calculate_rewards(prompts, completions, gt_batch, tools_batch, num_gen, reward_model=None, device="cuda", turn_outputs_batch=None, unfinished_batch=None):
    """
    为每条 rollout 打分。这是整个 RL 的「奖励函数」——不是神经网络，而是规则 + 外部 RM 的组合。

    奖励构成（全部加总后 clip 到 [-3, 3]）：

    ┌─────────────────────────────────────────────────────────────────────────┐
    │ 通用（两条分支都算）                                                      │
    │   标签扣分：每轮 <tool_call> 与 </tool_call> 数量不匹配，差一个扣 0.5     │
    ├─────────────────────────────────────────────────────────────────────────┤
    │ 分支一：无工具调用（模型直接回答）—— 用「格式 + RM」打分                   │
    │   长度分    ：回答长度 ∈ [5, 800] → +0.5，否则 -0.5                        │
    │   思考长度分：<think> 内容长度 ∈ [20, 300] → +1.0，否则 -0.5              │
    │   思考闭合分：</think> 恰好出现 1 次 → +0.25，否则 -0.25                   │
    │   RM 分     ：外部 reward_model.get_score(messages, answer)，clip[-3,3]    │
    │   重复惩罚  ：-rep_penalty(answer)                                        │
    ├─────────────────────────────────────────────────────────────────────────┤
    │ 分支二：有工具调用 —— 用「工具执行结果 + GT」打分                           │
    │   工具对齐分：tool_gap=0 → +0.5，否则 -0.5 * tool_gap                     │
    │              tool_gap = |合法调用数 - GT 数| + 非法调用数                  │
    │   GT 分    ：最终回答里命中 GT 的比例 → +2.5 * 命中数 / GT总数             │
    │   未完成扣分：最后一轮仍在调工具 → -0.5                                    │
    │   重复惩罚  ：-rep_penalty(最终回答)                                       │
    └─────────────────────────────────────────────────────────────────────────┘

    例（有工具调用，gt=["28°C"]，num_gen=4，即 idx=0 是 prompt0 的第 0 条生成）：
        模型调用 get_current_weather(北京) → 工具返回 28°C → 模型回答"北京28°C，晴"
        - 合法调用数 valid_call_count=1，len(gt)=1 → tool_gap=0 → +0.5
        - 最终回答 "北京28°C，晴" 命中 GT "28°C" → +2.5 * (1/1) = +2.5
        - 未完成=False，无重复 → 最终 reward ≈ +3.0（clip 到上限）
    """
    rewards = torch.zeros(len(completions), device=device)
    for idx, response in enumerate(completions):
        reward, answer = 0.0, response
        sample_idx = idx // num_gen          # 反查该条生成属于哪个 prompt（组内编号）
        tools = tools_batch[sample_idx]
        turn_outputs = turn_outputs_batch[idx] if turn_outputs_batch is not None else [response]
        unfinished = unfinished_batch[idx] if unfinished_batch is not None else False
        # 每轮输出取 </think> 之后的部分（去掉思考段，只留下真正的回答/工具调用）
        turn_answers = [turn.split('</think>', 1)[-1].strip() if '</think>' in turn else turn.strip() for turn in turn_outputs]
        answer = turn_answers[-1] if turn_answers else response.strip()  # 最终答案 = 最后一轮的回答
        valid_names = {t['function']['name'] for t in tools} if tools else set()  # 本样本可用的工具名集合
        tool_calls = []
        for turn_answer in turn_answers: tool_calls.extend(parse_tool_calls(turn_answer))  # 解析tool调用
        # 通用标签扣分：<tool_call> 和 </tool_call> 不成对（标签没闭合），每多/少一个扣 0.5
        reward -= 0.5 * sum(abs(turn.count('<tool_call>') - turn.count('</tool_call>')) for turn in turn_answers)  # 标签扣分
        # -------- 无工具调用：格式+reward奖励 --------
        if not tool_calls:
            reward += 0.5 if 5 <= len(response.strip()) <= 800 else -0.5  # 长度分
            if '</think>' in response:
                think, answer = response.split('</think>', 1)
                reward += 1.0 if 20 <= len(think.strip()) <= 300 else -0.5  # 思考长度分
                reward += 0.25 if response.count('</think>') == 1 else -0.25  # 思考闭合分
                answer = answer.strip()
            if reward_model is not None:
                # 从 prompt 文本里用正则抽出 (role, content) 消息对，交给 RM 对 answer 打分
                prompt = prompts[sample_idx]
                pattern = r"<\|im_start\|>(system|user|assistant)\s+(.*?)<\|im_end\|>"
                matches = re.findall(pattern, prompt, re.DOTALL)
                messages = [{"role": role, "content": content.strip()} for role, content in matches]
                score = reward_model.get_score(messages, answer)
                reward += score  # RM分
            reward -= rep_penalty(answer)
            rewards[idx] = max(min(reward, 3.0), -3.0)  # 总分Clip
        # -------- 有工具调用：执行结果奖励 --------
        else:
            gt = gt_batch[sample_idx]
            valid_call_count = 0
            # 统计「参数合法」的工具调用数：工具名在本样本可用，且参数通过 CHECK_ARGS 校验
            for tool_call in tool_calls:
                name, raw = tool_call.get("name", ""), tool_call.get("arguments", {})
                if isinstance(raw, str):
                    try: raw = json.loads(raw)
                    except: raw = {}
                check = CHECK_ARGS.get(name)
                valid_call_count += int(bool(name in valid_names and check and check(raw)))
            # tool_gap 综合两个维度：合法调用数与 GT 数的差 + 非法调用数（惩罚乱调/参数错误）
            tool_gap = abs(valid_call_count - len(gt)) + max(0, len(tool_calls) - valid_call_count)  # tool数差值
            reward += 0.5 if tool_gap == 0 else -0.5 * tool_gap  # tool对齐分

            # final_text：工具执行完之后的「最终回答」，取最后一个 </tool_call> 之后的内容
            final_text = "" if unfinished else (answer.split('</tool_call>')[-1] if '</tool_call>' in answer else answer)
            verified = validate_gt_in_text(final_text, gt) if gt else set()
            if gt: reward += 2.5 * len(verified) / len(gt)  # GT分
            if unfinished: reward -= 0.5  # 未完成扣分
            reward -= rep_penalty(final_text if final_text else answer)
            rewards[idx] = max(min(reward, 3.0), -3.0)  # 总分Clip
    return rewards

# ================================ 工具与 Reward = End ================================
def rl_train_epoch(epoch, loader, iters, rollout_engine, ref_model, reward_model=None, start_step=0, wandb=None, use_sglang=False):
    """
    一个 epoch 的训练主循环。每个 step 的流程（GRPO/CISPO）：

        rollout（采样） → packing（打包） → reward（打分） → advantage（组内标准化）
        → policy/ref 前向算 logprob → KL + ratio → GRPO/CISPO loss → backward → 优化器

    注意：整个函数依赖全局变量 model / tokenizer / optimizer / scheduler / args / lm_config，
    这些都在 __main__ 里初始化（脚本式写法，不是传参进来）。
    """
    last_step = start_step
    for step, batch in enumerate(loader, start=start_step + 1):
        # batch 来自 AgentRLDataset + collate_fn，每个字段是一个 list（batch_size 条）
        messages_batch = batch['messages']
        tools_batch = batch['tools']
        gt_batch = batch['gt']
        last_step = step

        # ---------- 1. Rollout：让策略模型采样（无需梯度） ----------
        # 每个 prompt 采 args.num_generations 条，返回摊平后的一维列表（长度 = batch_size * num_gen）
        with torch.no_grad():
            completions, contexts, prompt_ids_batch, response_ids_batch, response_masks_batch, response_old_logps_batch, turn_outputs_batch, unfinished_batch = rollout_batch(rollout_engine, tokenizer, messages_batch, tools_batch, args.num_generations, max_turns=3, max_new_tokens=args.max_gen_len, thinking_ratio=args.thinking_ratio, device=args.device)

        # ---------- 2. Packing：把 prompt + response 拼成完整序列，并构造 loss mask ----------
        # prompts 是渲染成文本的 prompt（供 reward 里 RM 抽 messages 用）
        prompts = [tokenizer.apply_chat_template(m, tokenize=False, add_generation_prompt=True, tools=t) for m, t in zip(messages_batch, tools_batch)]
        packed_samples = []
        for p, r, m, old_lp in zip(prompt_ids_batch, response_ids_batch, response_masks_batch, response_old_logps_batch):
            # ids    : prompt 部分(全 0 mask) + response 部分(模型生成=1 / 工具观察=0)
            # old_lp : 只有 response 位置有旧 logprob，prompt 位置补 0.0（且长度比序列少 1，因为 logprob 是"预测下一个 token"）
            ids = p + r
            mask = [0] * len(p) + m
            old_logps = [0.0] * max(len(p) - 1, 0) + old_lp
            # 超长截断：从尾部保留最后 max_total_len 个 token；old_logps 同步截到 len(ids)-1
            if len(ids) > args.max_total_len:
                ids = ids[-args.max_total_len:]
                mask = mask[-args.max_total_len:]
                old_logps = old_logps[-(len(ids) - 1):]
            # prompt_len = 第一个 response 生成 token 的索引（第一个 mask==1 的位置）
            prompt_len = next((i for i, v in enumerate(mask) if v == 1), len(mask))
            packed_samples.append((ids, mask, prompt_len, old_logps))
        # 把不同长度的样本 pad 到 batch 内最长长度 max_len，构造成可堆叠的张量
        # 例（batch 内两条，prompt 分别长 3/2，response 分别长 3/4）：
        #   ids[0] = [p0 p1 p2 | r0 r1 r2]                      -> pad 到 8: [p0 p1 p2 r0 r1 r2 pad pad]
        #   ids[1] = [p0 p1    | r0 r1 r2 r3]                   -> pad 到 8: [p0 p1 r0 r1 r2 r3 pad pad]
        #   mask[0] = [0 0 0 1 1 1 0 0]，mask[1] = [0 0 1 1 1 1 0 0]
        seq_lens = torch.tensor([len(ids) for ids, _, _, _ in packed_samples], device=args.device)
        max_len = seq_lens.max().item()
        input_ids = torch.tensor([ids + [tokenizer.pad_token_id] * (max_len - len(ids)) for ids, _, _, _ in packed_samples], device=args.device)
        prompt_lens = torch.tensor([prompt_len for _, _, prompt_len, _ in packed_samples], device=args.device)
        full_response_masks = torch.tensor([mask + [0] * (max_len - len(mask)) for _, mask, _, _ in packed_samples], device=args.device, dtype=torch.float32)
        # old_per_token_logps 长度是 max_len-1（对应"预测位置"），不足补 0.0
        old_per_token_logps = torch.tensor([old_logps + [0.0] * ((max_len - 1) - len(old_logps)) for _, _, _, old_logps in packed_samples], device=args.device, dtype=torch.float32)
        full_mask = (input_ids != tokenizer.pad_token_id).long()  # 非 pad 位置为 1，作 attention mask

        # ---------- 3. Reward：为每条 rollout 打分（规则 + 外部 RM） ----------
        rewards = calculate_rewards(prompts, completions, gt_batch, tools_batch, args.num_generations, reward_model, device=args.device, turn_outputs_batch=turn_outputs_batch, unfinished_batch=unfinished_batch)

        # ---------- 4. 策略模型前向，算每个「预测 token」的 log 概率 π_θ ----------
        # 关键：错位预测。位置 t 的 logits 预测的是位置 t+1 的 token：
        #   logits   = model(input_ids)[:, :-1, :]          形状 [B, L-1, vocab]（去掉最后一个位置的 logits）
        #   targets  = input_ids[:, 1:]                     形状 [B, L-1]（去掉第一个位置的 token）
        # 于是 per_token_logps[i, t] = log π_θ( input_ids[i, t+1] | input_ids[i, :t+1] )
        model_unwrapped = model.module if isinstance(model, DistributedDataParallel) else model
        with autocast_ctx:
            res = model_unwrapped(input_ids, attention_mask=full_mask)
            aux_loss = res.aux_loss if lm_config.use_moe else torch.tensor(0.0, device=args.device)  # MoE 辅助损失
            logits = res.logits[:, :-1, :]
            per_token_logps = F.log_softmax(logits, dim=-1).gather(2, input_ids[:, 1:].unsqueeze(-1)).squeeze(-1)

        # ---------- 5. 参考模型前向，算冻结策略 π_ref 的 log 概率（用于 KL 惩罚） ----------
        with torch.no_grad():
            ref_per_token_logps = compute_per_token_logps(ref_model, input_ids, input_ids.size(1) - 1, attention_mask=full_mask)

        # ---------- 6. Completion mask：只对「模型生成且 EOS 之前」的 token 算 loss ----------
        # full_response_masks[:, 1:] 错位对齐到「预测位置」（与 logps 的 [B, L-1] 对齐）
        completion_mask = full_response_masks[:, 1:]
        # is_eos：在「模型生成的 token」里标记出 EOS 的位置
        is_eos = (input_ids[:, 1:] == tokenizer.eos_token_id) & completion_mask.bool()
        # eos_idx 默认取最后一个位置；若出现了 EOS 则取第一个 EOS 的位置
        eos_idx = torch.full((completion_mask.size(0),), completion_mask.size(1) - 1, device=args.device, dtype=torch.long)
        has_eos = is_eos.any(dim=1)
        eos_idx[has_eos] = is_eos.int().argmax(dim=1)[has_eos]
        # 把 EOS 之后的位置 mask 掉（EOS 之后不该再算 loss）
        pos = torch.arange(completion_mask.size(1), device=args.device).unsqueeze(0)
        completion_mask = completion_mask * (pos <= eos_idx.unsqueeze(1)).float()
        # token_counts = 每个样本参与 loss 的有效 token 数；valid_rows 过滤「全为 0」的无效样本
        token_counts = completion_mask.sum(dim=1)
        valid_rows = token_counts > 0

        if args.debug_mode and is_main_process() and step % args.debug_interval == 0:
            for i in range(len(messages_batch)):
                Logger(f"[DEBUG] step={step}, gt[{i}]: {repr(gt_batch[i])}")
                Logger('-'*100)
                for j in range(args.num_generations):
                    idx = i * args.num_generations + j
                    plen, slen = prompt_lens[idx].item(), seq_lens[idx].item()
                    Logger(f"{'=' * 30} [DEBUG] gen[{i}][{j}] CONTEXT_BEGIN {'=' * 30}")
                    Logger(contexts[idx])
                    Logger(f"{'=' * 31} [DEBUG] gen[{i}][{j}] CONTEXT_END {'=' * 31}")
                    Logger(f"[DEBUG] gen[{i}][{j}] prompt_len={plen}, seq_len={slen}")
                    tokens = input_ids[idx, plen:slen].tolist()
                    text = tokenizer.decode(tokens, skip_special_tokens=False)
                    Logger(f"{'=' * 28} [DEBUG] gen[{i}][{j}] COMPLETION_BEGIN [{plen}:{slen}] {'=' * 28}")
                    Logger(text)
                    Logger(f"{'=' * 29} [DEBUG] gen[{i}][{j}] COMPLETION_END {'=' * 29}")
                    Logger(f"[DEBUG] gen[{i}][{j}] reward={rewards[idx].item():.4f}")
                    Logger('='*100)

        # ---------- 7. GRPO 优势函数：组内（同一 prompt 的 num_gen 条生成）标准化 ----------
        # 不用 Critic 网络，而是用「同一 prompt 多条回答的相对好坏」做 baseline。
        # advantage = (reward - 组均值) / (组标准差 + 1e-4)
        # 例（num_gen=4，某 prompt 的 4 条生成 reward = [1.0, 0.5, -0.5, 0.0]）：
        #   组均值 mean_r = 0.25，组标准差 std_r ≈ 0.559
        #   advantages = [(1.0-0.25)/0.559, (0.5-0.25)/0.559, (-0.5-0.25)/0.559, (0.0-0.25)/0.559]
        #              ≈ [1.34, 0.45, -1.34, -0.45]
        grouped_rewards = rewards.view(-1, args.num_generations)          # [num_prompts, num_gen]
        mean_r = grouped_rewards.mean(dim=1).repeat_interleave(args.num_generations)  # 每个样本的组均值
        std_r = grouped_rewards.std(dim=1, unbiased=False).repeat_interleave(args.num_generations)
        advantages = (rewards - mean_r) / (std_r + 1e-4)

        # ---------- 8. 计算 KL 惩罚与重要性采样比 ratio ----------
        # kl_div = log(π_ref) - log(π_θ) = log(π_ref/π_θ)
        # per_token_kl 用「k3 估计器」exp(x) - x - 1（恒非负、低方差的无偏 KL 估计），其中 x = kl_div
        kl_div = ref_per_token_logps - per_token_logps
        per_token_kl = torch.exp(kl_div) - kl_div - 1
        # ratio = exp(log π_θ - log π_old) = π_θ / π_old：当前策略相对采样时策略的重要性采样比
        # 用于校正「rollout 是用旧策略采的、现在用新策略更新」带来的分布偏移
        ratio = torch.exp(per_token_logps - old_per_token_logps)
        if args.loss_type == "cispo":
            # CISPO：直接用 ratio 加权「log 概率」本身（而非加权 ratio），且 ratio 只做上界裁剪。
            # per_token_loss = -(clamped_ratio * A * log π_θ  -  β * KL)
            # 优化方向：对 advantage>0 的 token 提升其 log π_θ，advantage<0 则降低。
            clamped_ratio = torch.clamp(ratio, max=args.epsilon_high).detach()
            per_token_loss = -(clamped_ratio * advantages.unsqueeze(1) * per_token_logps - args.beta * per_token_kl)
        else:
            # GRPO：标准 PPO 式 clip 目标（带 KL 惩罚）。
            # per_token_loss = -(min(ratio*A, clip(ratio)*A) - β*KL)，对 ratio 做 [1-ε, 1+ε] 裁剪取 min
            clipped_ratio = torch.clamp(ratio, 1 - args.epsilon, 1 + args.epsilon)
            per_token_loss1 = ratio * advantages.unsqueeze(1)
            per_token_loss2 = clipped_ratio * advantages.unsqueeze(1)
            per_token_loss = -(torch.min(per_token_loss1, per_token_loss2) - args.beta * per_token_kl)
        # ---------- 9. 聚合 loss：每个样本的 token 级 loss 求和 / 有效 token 数（平均），再对 batch 求均值 ----------
        # 注意：只用 completion_mask 覆盖的「模型生成且 EOS 前」的 token 参与 loss
        policy_loss = (((per_token_loss * completion_mask).sum(dim=1)[valid_rows] / token_counts[valid_rows].clamp(min=1)).mean()
                       if valid_rows.any() else per_token_loss.sum() * 0.0)
        # 加上 MoE 辅助损失，再除以梯度累积步数（配合后面的累积更新）
        loss = (policy_loss + aux_loss) / args.accumulation_steps
        loss.backward()

        # ---------- 10. 梯度累积更新：每 accumulation_steps 步才真正更新一次 ----------
        if step % args.accumulation_steps == 0:
            if args.grad_clip > 0: torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)  # 梯度裁剪防爆炸
            optimizer.step(); scheduler.step(); optimizer.zero_grad()

        # ---------- 11. 日志：打印/记录训练指标 ----------
        if step % args.log_interval == 0 or step == iters:
            pl = loss.item() * args.accumulation_steps   # 还原回「未除以累积步数」的真实 loss
            ar = rewards.mean().item()                   # 平均 reward
            al = token_counts.float().mean().item()      # 平均有效响应长度
            kl = ((ref_per_token_logps - per_token_logps) * completion_mask).sum().item() / max(token_counts.sum().item(), 1)  # 平均 KL
            gs = grouped_rewards.std(dim=1, unbiased=False).mean().item()  # 组内 reward 标准差（越大说明组内区分度越好）
            am, ast = advantages.mean().item(), advantages.std().item()
            lr = optimizer.param_groups[0]['lr']
            Logger(f'Epoch:[{epoch+1}/{args.epochs}]({step}/{iters}), Reward:{ar:.4f}, KL:{kl:.4f}, GrpStd:{gs:.4f}, AdvStd:{ast:.4f}, Loss:{pl:.4f}, AvgLen:{al:.2f}, AdvMean:{am:.4f}, LR:{lr:.8f}')
            if wandb and is_main_process():
                wandb.log({"reward":ar,"kl_ref":kl,"group_reward_std":gs,"advantages_std":ast,"policy_loss":pl,"avg_response_len":al,"advantages_mean":am,"learning_rate":lr})

        # ---------- 12. 保存模型：权重 .pth + resume checkpoint ----------
        if (step % args.save_interval == 0 or step == iters) and is_main_process():
            model.eval()
            moe_suffix = '_moe' if lm_config.use_moe else ''
            ckp = f'{args.save_dir}/{args.save_weight}_{lm_config.hidden_size}{moe_suffix}.pth'
            raw_model = model.module if isinstance(model, DistributedDataParallel) else model
            raw_model = getattr(raw_model, '_orig_mod', raw_model)  # 去掉 torch.compile 的包装层
            state_dict = raw_model.state_dict()
            torch.save({k: v.half().cpu() for k, v in state_dict.items()}, ckp)  # 权重存 half 精度省空间
            # resume checkpoint 含 optimizer/scheduler/epoch/step/wandb_id，支持断点续训
            lm_checkpoint(lm_config, weight=args.save_weight, model=model, optimizer=optimizer,
                         epoch=epoch, step=step, wandb=wandb, save_dir='../checkpoints', scheduler=scheduler)
            model.train()
            del state_dict

        # ---------- 13. 同步新权重到 rollout 引擎（关键！否则引擎一直用旧模型采样） ----------
        if step % args.save_interval == 0 or step == iters: rollout_engine.update_policy(model)

        # ---------- 14. 释放显存，避免累积 ----------
        del per_token_logps, ref_per_token_logps
        del completions, rewards, grouped_rewards, mean_r, std_r, advantages, completion_mask

    # 处理「最后一个不完整的 accumulation」的余数更新（last_step 未对齐 accumulation_steps 时补一次 step）
    if last_step > start_step and last_step % args.accumulation_steps != 0:
        if args.grad_clip > 0: torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step(); scheduler.step(); optimizer.zero_grad()


if __name__ == "__main__":
    # ============================================================
    # 入口：解析参数 → 初始化(分布式/模型/RM/引擎/数据/优化器) → 逐 epoch 训练
    # ============================================================
    parser = argparse.ArgumentParser(description="MiniMind Agent RL")
    # ---- 基础/保存 ----
    parser.add_argument("--save_dir", type=str, default="../out", help="模型保存目录")
    parser.add_argument('--save_weight', default='agent', type=str, help="保存权重名称")
    parser.add_argument("--epochs", type=int, default=1, help="训练轮数")
    parser.add_argument("--batch_size", type=int, default=2, help="批次大小")
    parser.add_argument("--learning_rate", type=float, default=3e-7, help="学习率")
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu", help="训练设备")  # RL 阶段学习率极低（3e-7）
    parser.add_argument("--dtype", type=str, default="bfloat16", help="数据类型 bfloat16/float16")
    parser.add_argument("--num_workers", type=int, default=8, help="数据加载线程数")
    parser.add_argument("--accumulation_steps", type=int, default=1, help="梯度累积步数")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="梯度裁剪阈值")
    parser.add_argument("--log_interval", type=int, default=1, help="日志打印间隔")
    parser.add_argument("--save_interval", type=int, default=10, help="模型保存间隔")
    parser.add_argument('--hidden_size', default=768, type=int, help="模型隐藏层维度")
    parser.add_argument('--num_hidden_layers', default=8, type=int, help="模型层数")
    parser.add_argument('--use_moe', default=0, type=int, choices=[0, 1], help="是否使用MoE")
    parser.add_argument('--max_seq_len', default=1024, type=int, help="最大序列长度")
    parser.add_argument("--max_gen_len", type=int, default=768, help="单次最大生成长度")
    parser.add_argument("--max_total_len", type=int, default=2500, help="训练侧最终总长度上界")
    parser.add_argument("--data_path", type=str, default="../dataset/agent_rl.jsonl", help="训练数据路径")
    # ---- RL 核心超参 ----
    parser.add_argument("--num_generations", type=int, default=4, help="每个prompt生成数量")
    parser.add_argument("--beta", type=float, default=0.1, help="KL散度惩罚系数")
    parser.add_argument("--loss_type", type=str, default="cispo", choices=["grpo", "cispo"], help="loss类型")
    parser.add_argument("--epsilon", type=float, default=0.2, help="GRPO的PPO clip epsilon")
    parser.add_argument("--epsilon_high", type=float, default=5.0, help="epsilon上界")
    parser.add_argument('--from_weight', default='full_sft', type=str, help="加载预训练权重名称")
    parser.add_argument('--from_resume', default=0, type=int, choices=[0, 1], help="是否从checkpoint恢复")
    parser.add_argument("--use_wandb", action="store_true", help="是否使用wandb记录")
    parser.add_argument("--wandb_project", type=str, default="MiniMind-Agent-RL", help="wandb项目名称")
    parser.add_argument("--use_compile", default=0, type=int, choices=[0, 1], help="是否使用torch.compile")
    parser.add_argument("--debug_mode", action="store_true", help="调试模式")
    parser.add_argument("--debug_interval", type=int, default=20, help="调试日志间隔")
    parser.add_argument("--thinking_ratio", type=float, default=0.1, help="按概率开启thinking（0.0~1.0）")
    parser.add_argument("--reward_model_path", type=str, default="../../internlm2-1_8b-reward", help="Reward模型路径")
    parser.add_argument("--rollout_engine", type=str, default="torch", choices=["torch", "sglang"], help="rollout引擎类型")
    parser.add_argument("--sglang_base_url", type=str, default="http://localhost:8998", help="SGLang服务器URL")
    parser.add_argument("--sglang_model_path", type=str, default="../model", help="SGLang tokenizer路径")
    parser.add_argument("--sglang_shared_path", type=str, default="./sglang_ckpt_agent", help="SGLang共享存储路径")
    args = parser.parse_args()

    # ---------- 分布式初始化 + 随机种子 ----------
    # init_distributed_mode：单机（无 RANK 环境变量）返回 0；DDP 下返回 local_rank 并初始化进程组
    local_rank = init_distributed_mode()
    if dist.is_initialized(): args.device = f"cuda:{local_rank}"
    setup_seed(42 + (dist.get_rank() if dist.is_initialized() else 0))  # 各 rank 用不同种子，保证 shuffle 不同

    # ---------- 模型配置 ----------
    # max_seq_len = 固定 prompt 空间 + 生成长度，为「prompt + 最长生成」预留位置
    os.makedirs(args.save_dir, exist_ok=True)
    lm_config = MiniMindConfig(hidden_size=args.hidden_size, num_hidden_layers=args.num_hidden_layers,
                               max_seq_len=args.max_seq_len + args.max_gen_len, use_moe=bool(args.use_moe))
    # 若 --from_resume，加载 resume checkpoint（含 model/optimizer/scheduler/epoch/step）
    ckp_data = lm_checkpoint(lm_config, weight=args.save_weight, save_dir='../checkpoints') if args.from_resume == 1 else None

    # ---------- 精度与 autocast 上下文 ----------
    device_type = "cuda" if "cuda" in args.device else "cpu"
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
    autocast_ctx = nullcontext() if device_type == "cpu" else torch.cuda.amp.autocast(dtype=dtype)

    # ---------- wandb（用国产 swanlab，接口兼容）----------
    wandb = None
    if args.use_wandb and is_main_process():
        import swanlab as wandb
        wandb_id = ckp_data.get('wandb_id') if ckp_data else None
        resume = 'must' if wandb_id else None  # 断点续训时恢复原 run
        wandb.init(project=args.wandb_project, name=f"Agent-RL-E{args.epochs}-B{args.batch_size}-LR{args.learning_rate}", id=wandb_id, resume=resume)

    # ---------- 三个模型：策略(训练) / 参考(冻结算KL) / 奖励模型(打分) ----------
    model, tokenizer = init_model(lm_config, args.from_weight, device=args.device)

    ref_model, _ = init_model(lm_config, args.from_weight, device=args.device)
    ref_model = ref_model.eval().requires_grad_(False)  # 参考模型只推理、不更新

    reward_model = LMForRewardModel(args.reward_model_path, device=args.device, dtype=torch.float16)
    Logger(f'Loaded reward model from {args.reward_model_path}')
    # Rollout引擎：torch（本地生成）或 sglang（远程推理服务器）
    rollout_engine = create_rollout_engine(
        engine_type=args.rollout_engine,
        policy_model=model,
        tokenizer=tokenizer,
        device=args.device,
        autocast_ctx=autocast_ctx,
        sglang_base_url=args.sglang_base_url,
        sglang_model_path=args.sglang_model_path,
        sglang_shared_path=args.sglang_shared_path,
    )
    # ---------- 数据与优化器 ----------
    train_ds = AgentRLDataset(args.data_path, tokenizer, max_length=lm_config.max_seq_len)
    train_sampler = DistributedSampler(train_ds) if dist.is_initialized() else None
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)
    # collate_fn 只把三字段各自打包成 list，不做 pad（RL 是逐个 rollout，batch 内不需要等长张量）
    def collate_fn(batch): return {'messages': [b['messages'] for b in batch], 'tools': [b['tools'] for b in batch], 'gt': [b['gt'] for b in batch]}
    loader_for_count = DataLoader(train_ds, batch_size=args.batch_size, sampler=train_sampler, collate_fn=collate_fn)
    iters = len(loader_for_count)  # 每 epoch 的 step 数
    total_optimizer_steps = math.ceil(iters / args.accumulation_steps) * args.epochs
    # 余弦退火：学习率从 lr 退火到 lr/10
    scheduler = CosineAnnealingLR(optimizer, T_max=total_optimizer_steps, eta_min=args.learning_rate / 10)

    # ---------- 断点续训：恢复 model/optimizer/scheduler/epoch/step ----------
    start_epoch, start_step = 0, 0
    if ckp_data:
        model.load_state_dict(ckp_data['model'])
        optimizer.load_state_dict(ckp_data['optimizer'])
        scheduler.load_state_dict(ckp_data['scheduler'])
        start_epoch = ckp_data['epoch']
        start_step = ckp_data.get('step', 0)

    # ---------- 可选 torch.compile + DDP 包装（每次改变 model 后都要 update_policy 同步引擎） ----------
    if args.use_compile == 1:
        model = torch.compile(model)
        Logger('torch.compile enabled')
        rollout_engine.update_policy(model)
    if dist.is_initialized():
        model = DistributedDataParallel(model, device_ids=[local_rank])
    rollout_engine.update_policy(model)

    # ---------- 训练主循环：逐 epoch 调用 rl_train_epoch ----------
    for epoch in range(start_epoch, args.epochs):
        train_sampler and train_sampler.set_epoch(epoch)  # DDP 下每个 epoch 用不同 shuffle，保证各 rank 数据不重复
        setup_seed(42 + epoch); indices = torch.randperm(len(train_ds)).tolist()  # 固定种子打乱（可复现）
        # SkipBatchSampler：断点续训时跳过已训练的 batch
        skip = start_step if (epoch == start_epoch and start_step > 0) else 0
        batch_sampler = SkipBatchSampler(train_sampler or indices, args.batch_size, skip)
        loader = DataLoader(train_ds, batch_sampler=batch_sampler, num_workers=args.num_workers, pin_memory=True, collate_fn=collate_fn)
        if skip > 0:
            Logger(f'Epoch [{epoch+1}/{args.epochs}]: skip {start_step} steps')
            rl_train_epoch(epoch, loader, len(loader) + skip, rollout_engine, ref_model, reward_model, start_step, wandb, use_sglang = (args.rollout_engine == "sglang"))
        else:
            rl_train_epoch(epoch, loader, len(loader), rollout_engine, ref_model, reward_model, 0, wandb, use_sglang = (args.rollout_engine == "sglang"))

    # ---------- DDP 收尾：等待所有进程完成后销毁进程组 ----------
    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()
