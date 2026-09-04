"""
train_grpo.py —— GRPO（Group Relative Policy Optimization，分组相对策略优化）训练脚本。

一句话概括：
    对每个 prompt 用 policy 模型在线采样 G=6 条回答 → 用「规则 + Reward Model」打分
    → 组内做均值/标准差归一化得到优势 advantage → 用带 KL 约束和裁剪的 loss 更新 policy。

    这是 DeepSeek-R1 那套 RL 算法，核心特点是**不需要 Critic/Value 网络**（PPO 需要）。

GRPO 需要的 3 个模型（见 __main__ 第 5 步）：
    model        —— policy 模型（要训练，从 full_sft 权重出发）
    ref_model    —— 参考模型（冻结，算 KL 散度用，防止 policy 跑飞）
    reward_model —— internlm2-1_8b-reward 奖励模型（给回答打分）

训练数据 rlaif.jsonl 只需要 prompt（assistant 答案留空，由模型自己采样生成），
详见 RLAIFDataset（dataset/lm_dataset.py）。

更完整的图文讲解见 doc/train_grpo详解.md。
"""
import os
import sys

__package__ = "trainer"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import datasets  # noqa: F401  # Windows pyarrow/torch DLL conflict workaround (issue #771)
import argparse
import math
import re
import gc
import warnings
import torch
import torch.nn.functional as F
import torch.distributed as dist
from transformers import AutoTokenizer
from contextlib import nullcontext
from torch import optim
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler
from torch.optim.lr_scheduler import CosineAnnealingLR
from transformers import AutoModel
from model.model_minimind import MiniMindConfig, MiniMindForCausalLM
from dataset.lm_dataset import RLAIFDataset
from trainer.trainer_utils import Logger, is_main_process, lm_checkpoint, init_distributed_mode, setup_seed, SkipBatchSampler, init_model, LMForRewardModel
from trainer.rollout_engine import create_rollout_engine

warnings.filterwarnings('ignore')


def rep_penalty(text, n=3, cap=0.5):
    """
    重复惩罚（n-gram 版本）：衡量文本里「重复三元组」的占比，返回 [0, cap] 之间的惩罚值。
    这个值会在 calculate_rewards 里从总奖励中扣除，用来抑制模型生成复读机式回答。

    原理：把文本切分成 token 后，滑窗取长度为 n 的三元组，统计「重复的三元组」占「总三元组」的比例。
    重复得越多，惩罚越接近上限 cap（默认 0.5）。

    具体例子（n=3，文本 = "The cat sat the cat sat the cat sat"）：
        分词   = [the, cat, sat, the, cat, sat, the, cat, sat]   # 9 个 token
        三元组 = 共 7 个：the-cat-sat / cat-sat-the / sat-the-cat 各出现 2~3 次
        去重后只有 3 种 → 重复率 = (7 - 3) / 7 ≈ 0.571
        惩罚   = 0.571 * (0.5 * 2) = 0.571 → 超过 cap 0.5 → 截断返回 0.5

    边界：文本为空 / 不足 n 个 token 时 grams 为空，返回 0.0（不惩罚）。
    """
    toks = re.findall(r"\w+|[^\w\s]", text.lower())          # 分词：连续的字母数字，或单个非空白标点
    grams = [tuple(toks[i:i + n]) for i in range(len(toks) - n + 1)]  # 滑窗取 n 元组
    # (len(grams) - len(set(grams))) 是「重复了多少个三元组」；除以 len(grams) 得重复比例；
    # 再乘 cap*2 放大到 [0, 2cap]，最后用 min(cap, ...) 截断到上限 cap。
    return min(cap, (len(grams) - len(set(grams))) * cap * 2 / len(grams)) if grams else 0.0


def calculate_rewards(prompts, responses, reward_model):
    """
    奖励函数：对每条生成回答算一个标量 reward，累加成形状 [B*num_generations] 的张量返回。

    奖励由四部分组成（按代码顺序）：
        ① 长度奖励：回答 20~800 字符给 +0.5，否则 -0.5（鼓励中等长度）
        ② 思考奖励：若回答里含 </think>（说明模型真的思考了），给「思考长度 + 标签格式」两项奖励
        ③ 重复惩罚：减去 rep_penalty(answer)（抑制复读机式回答）
        ④ Reward Model 打分：把「prompt 还原成 messages + 最终答案」喂给 reward_model，得到 [-3,3] 分数

    关键点：reward model 只对 </think> 之后的「最终答案」打分，思考过程本身不打分，
    而是通过 ② 的规则奖励间接被鼓励——这正是 DeepSeek-R1 的「格式奖励 + 答案正确性奖励」分离设计。

    参数：
        prompts      : list[str]，长度 B（每个 prompt 采样 num_generations 条回答）
        responses    : list[str]，长度 B*num_generations（第 i 个 prompt 的第 j 条回答在 index=i*G+j）
        reward_model : LMForRewardModel（封装 internlm2-1_8b-reward 的 get_score 接口）

    返回：rewards，形状 [B*num_generations]，每个元素是上述四部分之和。
    """
    rewards = torch.zeros(len(responses), device=args.device)   # [B*num_generations]，先置 0 再逐项累加

    with torch.no_grad():   # 打分全程不计算梯度（reward model 不参与训练）
        reward_model_scores = []
        batch_size = len(prompts)   # = B

        for i in range(batch_size):                 # 遍历每个 prompt
            for j in range(args.num_generations):   # 遍历该 prompt 采样的 G 条回答
                response_idx = i * args.num_generations + j   # 扁平化索引：第 i 组第 j 条
                response = responses[response_idx]
                prompt = prompts[i]

                # 用正则把 prompt 文本还原成 messages 列表（供 reward model 使用）。
                # MiniMind 模板形如 <|im_start|>role\n内容<|im_end|>\n，re.DOTALL 让 . 能跨行匹配内容。
                # 例：prompt = "<|im_start|>user\n1+1=几<|im_end|>\n<|im_start|>assistant\n"
                #     匹配得 [("user","1+1=几")] → messages=[{"role":"user","content":"1+1=几"}]
                pattern = r"<\|im_start\|>(system|user|assistant)\s+(.*?)<\|im_end\|>"
                matches = re.findall(pattern, prompt, re.DOTALL)
                messages = [{"role": role, "content": content.strip()} for role, content in matches]

                # ① 长度奖励：统计整条回答的字符数（含思考段），20~800 之间 +0.5，否则 -0.5
                answer = response
                rewards[response_idx] += 0.5 if 20 <= len(response.strip()) <= 800 else -0.5

                # ② 思考奖励：只有回答里出现 </think> 才触发。
                #    注意：<think> 是 prompt 里就有的（open_thinking 打开时），
                #    而 </think> 是模型自己生成的，所以用 </think> 来判断「是否真的思考了」。
                if '</think>' in response:
                    thinking_content, answer_content = response.split('</think>', 1)
                    # ②-1 思考长度奖励：思考 20~300 字符 +1.0，否则 -0.5
                    rewards[response_idx] += 1.0 if 20 <= len(thinking_content.strip()) <= 300 else -0.5
                    # ②-2 标签格式奖励：恰好一个 </think> +0.25（出现多个说明格式崩了，-0.25）
                    rewards[response_idx] += 0.25 if response.count('</think>') == 1 else -0.25
                    # 拆出「最终答案」——只把 </think> 之后的部分拿去给 reward model 打分
                    answer = answer_content.strip()

                # ③ 重复惩罚：只惩罚最终答案部分（思考过程即使重复也无妨）
                rewards[response_idx] -= rep_penalty(answer)

                # ④ Reward Model 打分（在 LMForRewardModel.get_score 内部被 clamp 到 [-3, 3]）
                score = reward_model.get_score(messages, answer)
                reward_model_scores.append(score)

        # 所有 reward model 分数转成张量后一次性加到 rewards 上
        reward_model_scores = torch.tensor(reward_model_scores, device=args.device)
        rewards += reward_model_scores

    return rewards


def grpo_train_epoch(epoch, loader, iters, rollout_engine, ref_model, reward_model, start_step=0, wandb=None, use_sglang=False):
    """
    一个 epoch 的 GRPO 训练循环（按 step 迭代）。

    形状约定（贯穿整个函数，设 batch_size=B、num_generations=G、prompt 长度=P、生成长度=R）：
        prompts          : list[str]，长度 B
        output_ids       : [B*G, P+R]   完整序列（prompt + 生成），左侧是左填充的 prompt
        completion_ids   : [B*G, R]     只含生成部分
        per_token_logps  : [B*G, R]     当前 policy 对每个生成 token 的 log 概率
        old_per_token_logps : [B*G, R]  采样时（旧 policy）的 log 概率，算 ratio 的分母
        rewards          : [B*G]        每条回答的标量奖励
        advantages       : [B*G]        组内标准化的优势 (r - mean)/std

    每个 step 的流程：
        编码 prompt → rollout 采样 G 条 → 算奖励 → 算 advantage → 算 KL/ratio → 算 loss → backward/step。
    """
    for step, batch in enumerate(loader, start=start_step + 1):
        prompts = batch['prompt']  # list[str], length B

        # ===== 1. 编码 prompt（左填充）=====
        # padding_side="left"：pad 补在左边。因为要接着 prompt 生成，真实内容必须靠右、
        # 生成位置才能对齐；若右填充，生成会从 pad 处开始，完全错位。
        # 例：两条 prompt token 化后 [你好, 1+1] 与 [今天天气如何]，pad=0：
        #     input_ids = [[0, 你好, 1+1], [今天, 天气, 如何]]  ← 短的左边补 0
        prompt_inputs = tokenizer(prompts, return_tensors="pt", padding=True, return_token_type_ids=False,
                                  padding_side="left", add_special_tokens=False).to(args.device)
        if args.max_seq_len:
            # 超长截断：只保留最后 max_seq_len 个 token（左填充下，靠右的才是有效内容）
            prompt_inputs["input_ids"] = prompt_inputs["input_ids"][:, -args.max_seq_len:]
            prompt_inputs["attention_mask"] = prompt_inputs["attention_mask"][:, -args.max_seq_len:]

        # ===== 2. Rollout 采样：每个 prompt 采样 G 条回答 =====
        # rollout 引擎内部把 prompt repeat_interleave(G) 后，用 temperature=0.8 自回归采样，
        # 同时记录采样时每个生成 token 的 log 概率（old_per_token_logps）。
        rollout_result = rollout_engine.rollout(
            prompt_ids=prompt_inputs["input_ids"],
            attention_mask=prompt_inputs["attention_mask"],
            num_generations=args.num_generations,   # G
            max_new_tokens=args.max_gen_len,        # R 的上限
            temperature=0.8,
        )
        outputs = rollout_result.output_ids            # [B*G, P+R] 完整序列
        completion_ids = rollout_result.completion_ids # [B*G, R]   只含生成部分
        completions = rollout_result.completions       # list[str]   解码文本（算奖励用）
        old_per_token_logps = rollout_result.per_token_logps.to(args.device).detach()  # [B*G, R] 采样时 log 概率
        prompt_lens = rollout_result.prompt_lens.to(args.device)                       # [B*G] 每条 prompt 长度

        # full_mask：完整序列的 attention mask（= 非 pad 位置为 1）。左填充的 prompt 和右填充的生成都要 mask。
        full_mask = (outputs != tokenizer.pad_token_id).long()

        # ===== 3. 算出「生成 token 在完整序列中的位置」=====
        # 模型在位置 t 的 logits 预测的是位置 t+1 的 token。生成部分的第一个 token 位于完整序列第 P 位，
        # 它由位置 P-1 的 logits 预测，所以索引从 P-1 起步，往后数 R 个。
        # 例：P=3, R=2 → logp_pos = [3-1+0, 3-1+1] = [2, 3]，即用 logits[2]、logits[3] 预测第 3、4 位的 token。
        logp_pos = prompt_lens.unsqueeze(1) - 1 + torch.arange(completion_ids.size(1), device=args.device).unsqueeze(0)  # [B*G, R]

        # ===== 4. 计算奖励 =====
        rewards = calculate_rewards(prompts, completions, reward_model).to(args.device)  # [B*num_gen]

        # ===== 5. 前向：当前 policy 的逐 token log 概率 =====
        model_unwrapped = model.module if isinstance(model, DistributedDataParallel) else model
        with autocast_ctx:
            # 用完整序列 outputs 前向一次，拿到所有位置的 logits
            res = model_unwrapped(outputs, attention_mask=full_mask)
            # MoE 负载均衡辅助损失（非 MoE 时为 0），会加到最终 loss 上
            aux_loss = res.aux_loss if lm_config.use_moe else torch.tensor(0.0, device=args.device)
            # logits[:, :-1, :] 预测 outputs[:, 1:]（标准 next-token 错位）：
            #   先 log_softmax 得每个位置的 log 概率分布，
            #   再 gather(2, outputs[:, 1:]) 取出「真实下一个 token」的 log 概率，得 [B*G, P+R-1]，
            #   最后 gather(1, logp_pos) 只挑出「生成段」的位置，得 [B*G, R]。
            per_token_logps = F.log_softmax(res.logits[:, :-1, :], dim=-1).gather(2, outputs[:, 1:].unsqueeze(-1)).squeeze(-1).gather(1, logp_pos)

        # ===== 6. 前向：冻结的 ref_model 的逐 token log 概率（算 KL 用）=====
        with torch.no_grad():   # ref 不计算梯度，只当「不动的标尺」
            ref_per_token_logps = F.log_softmax(ref_model(outputs, attention_mask=full_mask).logits[:, :-1, :], dim=-1).gather(2, outputs[:, 1:].unsqueeze(-1)).squeeze(-1).gather(1, logp_pos)

        # ===== 7. 调试打印：打印每个 prompt、每条采样回答和它的奖励 =====
        if args.debug_mode and is_main_process() and step % args.debug_interval == 0:
            for i in range(len(prompts)):
                Logger(f"[DEBUG] step={step}, sample[{i}]")
                Logger('-'*100)
                Logger(f"{'=' * 30} [DEBUG] sample[{i}] CONTEXT_BEGIN {'=' * 30}")
                Logger(prompts[i])
                Logger(f"{'=' * 31} [DEBUG] sample[{i}] CONTEXT_END {'=' * 31}")
                for j in range(args.num_generations):
                    idx = i * args.num_generations + j
                    Logger(f"{'=' * 28} [DEBUG] gen[{j}] RESPONSE_BEGIN {'=' * 28}")
                    Logger(completions[idx])
                    Logger(f"{'=' * 29} [DEBUG] gen[{j}] RESPONSE_END {'=' * 29}")
                    Logger(f"[DEBUG] gen[{j}] reward={rewards[idx].item():.4f}")
                Logger('='*100)

        # ===== 8. 组内标准化算 advantage（GRPO 的核心）=====
        # 把 B*G 条奖励按 (B, G) 分组，组内求均值/标准差，再标准化。
        # 这样 advantage 是「同一条 prompt 的 G 条回答互相比较」的结果，而不是全局比较。
        # 例：某 prompt 的 6 条回答 rewards=[1.5, 2.0, 0.5, -0.5, 1.0, 0.0]：
        #     mean=0.75, std≈0.854 → advantages=[0.878, 1.464, -0.293, -1.464, 0.293, -0.878]
        #     第 2 条(2.0分)优势 +1.46 要加大概率，第 4 条(-0.5分)优势 -1.46 要压低概率。
        grouped_rewards = rewards.view(-1, args.num_generations)  # [B, num_gen]
        mean_r = grouped_rewards.mean(dim=1).repeat_interleave(args.num_generations)  # [B*num_gen] 组均值广播回 B*G
        std_r = grouped_rewards.std(dim=1, unbiased=False).repeat_interleave(args.num_generations)  # [B*num_gen] 组标准差
        # +1e-4 防止 std=0（某组 G 条回答分数完全相同时会除零）
        advantages = (rewards - mean_r) / (std_r + 1e-4)  # [B*num_gen]

        # ===== 9. 构造 completion_mask（决定哪些生成 token 参与 loss）=====
        # 生成时可能提前遇到 EOS，之后全是 pad，这些 pad 位置不能参与 loss。
        completion_pad_mask = rollout_result.completion_mask.to(args.device).bool()   # 来自 rollout：标记哪些是真实 token
        # is_eos：每个位置是否是「真实存在的 EOS」
        is_eos = (completion_ids == tokenizer.eos_token_id) & completion_pad_mask  # [B*num_gen, R]
        # eos_idx：每行第一个 EOS 的位置；没有 EOS 则取 R-1（保留到末尾）
        eos_idx = torch.full((is_eos.size(0),), is_eos.size(1) - 1, dtype=torch.long, device=args.device)
        eos_idx[is_eos.any(dim=1)] = is_eos.int().argmax(dim=1)[is_eos.any(dim=1)]
        # 最终 mask：位置 <= 第一个 EOS 且非 pad 才保留
        # 例：completion_ids=[12,34,56,2,0,0]（2=EOS, 0=pad）→ eos_idx=3 → mask=[1,1,1,1,0,0]
        completion_mask = ((torch.arange(is_eos.size(1), device=args.device).expand(is_eos.size(0), -1) <= eos_idx.unsqueeze(1)) & completion_pad_mask).int()  # [B*num_gen, R]

        # ===== 10. 算 KL 散度（k3 估计器）和 importance ratio =====
        kl_div = ref_per_token_logps - per_token_logps               # log π_ref - log π
        # k3 估计器：e^d - d - 1，是 KL(π‖π_ref) 的恒 ≥0 下界近似。
        # 例：d=0.5 → e^0.5 - 0.5 - 1 ≈ 0.149。d=0 时为 0（policy 与 ref 完全一致）。
        per_token_kl = torch.exp(kl_div) - kl_div - 1  # [B*num_gen, R]
        # importance ratio = π_current / π_old（采样时是旧策略）。
        # 例：old logp=-2.3, current logp=-2.0 → ratio=e^0.3≈1.35（当前更看好该 token）。
        ratio = torch.exp(per_token_logps - old_per_token_logps)  # [B*num_gen, R]

        # ===== 11. 算逐 token 的 policy loss =====
        # 两种 loss 策略（--loss_type 切换，默认 cispo）：
        #   cispo：ratio 截断到上限并 detach（只当「停止梯度的权重」），梯度只流经 per_token_logps。
        #   grpo ：标准 PPO-clip，ratio 偏离 1 太远时双向裁剪，min 防止一次更新步子太大。
        # 二者都用 -beta * per_token_kl 把 KL 拉小（防止 policy 偏离 ref 太远，beta 默认 0.1）。
        if args.loss_type == "cispo":
            clamped_ratio = torch.clamp(ratio, max=args.epsilon_high).detach()  # 只裁上限，且 detach 阻断梯度
            per_token_loss = -(clamped_ratio * advantages.unsqueeze(1) * per_token_logps - args.beta * per_token_kl)
        else:
            clipped_ratio = torch.clamp(ratio, 1 - args.epsilon, 1 + args.epsilon)
            per_token_loss1 = ratio * advantages.unsqueeze(1)
            per_token_loss2 = clipped_ratio * advantages.unsqueeze(1)
            per_token_loss = -(torch.min(per_token_loss1, per_token_loss2) - args.beta * per_token_kl)

        # ===== 12. 归一化 + 反向传播 =====
        # 每条回答独立平均（用 completion_mask 只统计有效 token，除以有效 token 数），再对 batch 平均。
        policy_loss = ((per_token_loss * completion_mask).sum(dim=1) / completion_mask.sum(dim=1).clamp(min=1)).mean()
        loss = (policy_loss + aux_loss) / args.accumulation_steps  # 提前除以累积步数（梯度累积时取平均）
        loss.backward()

        # ===== 13. 梯度累积：每 accumulation_steps 步更新一次 =====
        if step % args.accumulation_steps == 0:
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)  # 梯度裁剪防爆炸
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        # ===== 14. 日志：每隔 log_interval 打印训练指标 =====
        if step % args.log_interval == 0 or step == iters:
            policy_loss_val = loss.item() * args.accumulation_steps  # 把之前除以累积步数的 loss 还原成「真实一步损失」
            current_aux_loss = aux_loss.item()
            avg_reward_val = rewards.mean().item()                          # 平均奖励
            avg_len_val = completion_mask.sum(dim=1).float().mean().item()  # 平均有效回答长度
            kl_ref_val = ((ref_per_token_logps - per_token_logps) * completion_mask).sum().item() / max(completion_mask.sum().item(), 1)  # 有效 token 上的平均 KL
            advantages_mean_val = advantages.mean().item()
            advantages_std_val = advantages.std().item()                    # advantage 方差：组内区分度指标
            current_lr = optimizer.param_groups[0]['lr']

            Logger(f'Epoch:[{epoch + 1}/{args.epochs}]({step}/{iters}), '
                   f'Reward: {avg_reward_val:.4f}, KL_ref: {kl_ref_val:.4f}, '
                   f'Adv Std: {advantages_std_val:.4f}, Adv Mean: {advantages_mean_val:.4f}, '
                   f'Actor Loss: {policy_loss_val:.4f}, Avg Response Len: {avg_len_val:.2f}, Learning Rate: {current_lr:.8f}')

            if wandb and is_main_process():
                wandb.log({
                    "reward": avg_reward_val,
                    "kl_ref": kl_ref_val,
                    "advantages_std": advantages_std_val,
                    "advantages_mean": advantages_mean_val,
                    "policy_loss": policy_loss_val,
                    "avg_response_len": avg_len_val,
                    "learning_rate": current_lr
                })

        # ===== 15. 保存权重（每隔 save_interval 或最后一步）=====
        if (step % args.save_interval == 0 or step == iters) and is_main_process():
            model.eval()
            moe_suffix = '_moe' if lm_config.use_moe else ''               # MoE 权重加后缀区分架构
            ckp = f'{args.save_dir}/{args.save_weight}_{lm_config.hidden_size}{moe_suffix}.pth'
            raw_model = model.module if isinstance(model, DistributedDataParallel) else model
            raw_model = getattr(raw_model, '_orig_mod', raw_model)         # torch.compile 会包成 _orig_mod，多解一层
            state_dict = raw_model.state_dict()
            torch.save({k: v.half().cpu() for k, v in state_dict.items()}, ckp)  # 纯权重，转半精度存 CPU 省空间
            # 另存完整 resume 状态（model+optimizer+scheduler+step），用于断点续训
            lm_checkpoint(lm_config, weight=args.save_weight, model=model, optimizer=optimizer,
                         epoch=epoch, step=step, wandb=wandb, save_dir='../checkpoints', scheduler=scheduler)
            model.train()
            del state_dict

        # ===== 16. 同步 policy 权重到 rollout 引擎 =====
        # sglang 引擎：把最新权重推给推理服务器（热更新）；torch 引擎：仅更新内部引用，无开销。
        if step % args.save_interval == 0 or step == iters: rollout_engine.update_policy(model)

        # ===== 17. 释放中间变量（GRPO 显存紧张：同时存 policy/ref 两份 logits + 完整 rollout）=====
        del prompt_inputs, outputs, completion_ids, per_token_logps, ref_per_token_logps
        del completions, rewards, grouped_rewards, mean_r, std_r, advantages, completion_mask, completion_pad_mask, prompt_lens, logp_pos

    # ===== 18. epoch 尾部：若还有不足 accumulation_steps 的未更新梯度，补做最后一次 step =====
    if step > start_step and step % args.accumulation_steps != 0:
        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MiniMind GRPO (Group Relative Policy Optimization)")
    parser.add_argument("--save_dir", type=str, default="../out", help="模型保存目录")
    parser.add_argument('--save_weight', default='grpo', type=str, help="保存权重的前缀名")
    parser.add_argument("--epochs", type=int, default=1, help="训练轮数")
    parser.add_argument("--batch_size", type=int, default=2, help="batch size")
    parser.add_argument("--learning_rate", type=float, default=3e-7, help="初始学习率")
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu", help="训练设备")
    parser.add_argument("--dtype", type=str, default="bfloat16", help="混合精度类型")
    parser.add_argument("--num_workers", type=int, default=8, help="数据加载线程数")
    parser.add_argument("--accumulation_steps", type=int, default=1, help="梯度累积步数")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="梯度裁剪阈值")
    parser.add_argument("--log_interval", type=int, default=1, help="日志打印间隔")
    parser.add_argument("--save_interval", type=int, default=10, help="模型保存间隔")
    parser.add_argument('--hidden_size', default=768, type=int, help="隐藏层维度")
    parser.add_argument('--num_hidden_layers', default=8, type=int, help="隐藏层数量")
    parser.add_argument('--use_moe', default=0, type=int, choices=[0, 1], help="是否使用MoE架构（0=否，1=是）")
    parser.add_argument('--max_seq_len', default=768, type=int, help="Prompt最大长度")
    parser.add_argument("--max_gen_len", type=int, default=1024, help="生成的最大长度")
    parser.add_argument("--data_path", type=str, default="../dataset/rlaif.jsonl", help="RLAIF数据路径")
    parser.add_argument("--num_generations", type=int, default=6, help="每个prompt生成的样本数")
    parser.add_argument("--beta", type=float, default=0.1, help="KL惩罚系数")
    parser.add_argument("--loss_type", type=str, default="cispo", choices=["grpo", "cispo"], help="loss类型")
    parser.add_argument("--epsilon", type=float, default=0.2, help="GRPO的PPO clip epsilon")
    parser.add_argument("--epsilon_high", type=float, default=5.0, help="epsilon上界")
    parser.add_argument('--from_weight', default='full_sft', type=str, help="基于哪个权重训练")
    parser.add_argument("--reward_model_path", type=str, default="../../internlm2-1_8b-reward", help="Reward模型路径")
    parser.add_argument('--from_resume', default=0, type=int, choices=[0, 1], help="是否自动检测&续训（0=否，1=是）")
    parser.add_argument("--use_wandb", action="store_true", help="是否使用wandb")
    parser.add_argument("--wandb_project", type=str, default="MiniMind-GRPO", help="wandb项目名")
    parser.add_argument("--use_compile", default=0, type=int, choices=[0, 1], help="是否使用torch.compile加速（0=否，1=是）")
    parser.add_argument("--debug_mode", action="store_true", help="是否打印训练调试采样")
    parser.add_argument("--debug_interval", type=int, default=20, help="debug模式下每隔多少step打印一次采样")
    parser.add_argument("--thinking_ratio", type=float, default=0.9, help="按概率开启thinking（0.0~1.0）")
    parser.add_argument("--rollout_engine", type=str, default="torch", choices=["torch", "sglang"], help="rollout引擎类型")
    parser.add_argument("--sglang_base_url", type=str, default="http://localhost:8998", help="SGLang服务器URL")
    parser.add_argument("--sglang_model_path", type=str, default="../model", help="SGLang tokenizer路径")
    parser.add_argument("--sglang_shared_path", type=str, default="./sglang_ckpt_grpo", help="SGLang共享存储路径")
    args = parser.parse_args()

    # ========== 1. 初始化环境和随机种子 ==========
    local_rank = init_distributed_mode()
    if dist.is_initialized(): args.device = f"cuda:{local_rank}"
    setup_seed(42 + (dist.get_rank() if dist.is_initialized() else 0))
    
    # ========== 2. 配置目录、模型参数、检查ckp ==========
    os.makedirs(args.save_dir, exist_ok=True)
    # 注意：config 的 max_seq_len = prompt 上限 + 生成长度上限。
    # 因为 rollout 后完整序列是 [prompt, 生成]，前向时要一次性装下 prompt+生成（默认 768+1024=1792）。
    lm_config = MiniMindConfig(hidden_size=args.hidden_size, num_hidden_layers=args.num_hidden_layers,
                               max_seq_len=args.max_seq_len + args.max_gen_len, use_moe=bool(args.use_moe))
    ckp_data = lm_checkpoint(lm_config, weight=args.save_weight, save_dir='../checkpoints') if args.from_resume==1 else None
    
    # ========== 3. 设置混合精度 ==========
    device_type = "cuda" if "cuda" in args.device else "cpu"
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
    autocast_ctx = nullcontext() if device_type == "cpu" else torch.cuda.amp.autocast(dtype=dtype)
    
    # ========== 4. 配wandb ==========
    wandb = None
    if args.use_wandb and is_main_process():
        import swanlab as wandb
        wandb_id = ckp_data.get('wandb_id') if ckp_data else None
        resume = 'must' if wandb_id else None
        wandb_run_name = f"MiniMind-GRPO-Epoch-{args.epochs}-BS-{args.batch_size}-LR-{args.learning_rate}"
        wandb.init(project=args.wandb_project, name=wandb_run_name, id=wandb_id, resume=resume)
    
    # ========== 5. 初始化模型和数据 ==========
    base_weight = args.from_weight   # 默认 'full_sft'，即从 SFT 后的权重出发做 RL
    # Policy 模型（要训练）
    model, tokenizer = init_model(lm_config, base_weight, device=args.device)
    # Reference 模型（冻结）：从**同一个 base_weight** 再加载一份独立权重，初始与 policy 完全同权。
    #   eval()          —— 关掉 Dropout 等训练行为，保证前向确定；
    #   requires_grad_(False) —— 不参与 backward、不被 optimizer 更新（optimizer 只包了 model.parameters()）。
    #   用途：算 KL(π_current ‖ π_ref)，当「不动的锚点」防止 policy 被 reward hacking 带跑偏。
    ref_model, _ = init_model(lm_config, base_weight, device=args.device)
    ref_model = ref_model.eval().requires_grad_(False)
    # Reward 模型（冻结）：internlm2-1_8b-reward，float16 加载，只前向打分、不训练
    reward_model = LMForRewardModel(args.reward_model_path, device=args.device, dtype=torch.float16)
    # Rollout引擎（可插拔替换，只负责 policy 推理）
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
    # 数据和优化器
    train_ds = RLAIFDataset(args.data_path, tokenizer, max_length=lm_config.max_seq_len, thinking_ratio=args.thinking_ratio)
    train_sampler = DistributedSampler(train_ds) if dist.is_initialized() else None
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)
    loader_for_count = DataLoader(train_ds, batch_size=args.batch_size, sampler=train_sampler)
    iters = len(loader_for_count)
    total_optimizer_steps = math.ceil(iters / args.accumulation_steps) * args.epochs
    scheduler = CosineAnnealingLR(optimizer, T_max=total_optimizer_steps, eta_min=args.learning_rate / 10)
    
    # ========== 6. 从ckp恢复状态 ==========
    start_epoch, start_step = 0, 0
    if ckp_data:
        model.load_state_dict(ckp_data['model'])
        optimizer.load_state_dict(ckp_data['optimizer'])
        scheduler.load_state_dict(ckp_data['scheduler'])
        start_epoch = ckp_data['epoch']
        start_step = ckp_data.get('step', 0)
    
    # ========== 7. 编译和分布式包装 ==========
    if args.use_compile == 1:
        model = torch.compile(model)
        Logger('torch.compile enabled')
        rollout_engine.update_policy(model)
    if dist.is_initialized():
        model = DistributedDataParallel(model, device_ids=[local_rank])
    rollout_engine.update_policy(model)
    
    # ========== 8. 开始训练 ==========
    for epoch in range(start_epoch, args.epochs):
        train_sampler and train_sampler.set_epoch(epoch)
        setup_seed(42 + epoch); indices = torch.randperm(len(train_ds)).tolist()
        skip = start_step if (epoch == start_epoch and start_step > 0) else 0
        batch_sampler = SkipBatchSampler(train_sampler or indices, args.batch_size, skip)
        loader = DataLoader(train_ds, batch_sampler=batch_sampler, num_workers=args.num_workers, pin_memory=True)
        if skip > 0: 
            Logger(f'Epoch [{epoch + 1}/{args.epochs}]: 跳过前{start_step}个step，从step {start_step + 1}开始')
            grpo_train_epoch(epoch, loader, len(loader) + skip, rollout_engine, ref_model, reward_model, start_step, wandb, use_sglang = (args.rollout_engine == "sglang"))
        else:
            grpo_train_epoch(epoch, loader, len(loader), rollout_engine, ref_model, reward_model, 0, wandb, use_sglang = (args.rollout_engine == "sglang"))
    
    # ========== 9. 清理分布进程 ==========
    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()