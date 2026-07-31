from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from ..config import ModelConfig
from ..special_tokens import build_special_token_ids, get_token_scheme_spec
from .attention import PastKeyValue
from .block import SeekerBlock
from .norm import RMSNorm
from .projector import inject_feature_tokens
from .rope import RotaryEmbedding


KVCache = list[PastKeyValue]


@dataclass
class SeekerOmniOutput:
    logits: torch.Tensor
    loss: torch.Tensor | None
    kv_cache: KVCache | None = None


class SeekerOmniLM(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.special_spec = get_token_scheme_spec(cfg.special_tokens_scheme)
        self.special = build_special_token_ids(self.special_spec)
        self.n_special = int(len(self.special_spec.special_tokens))
        self._special_token_to_id = {t: i for i, t in enumerate(self.special_spec.special_tokens)}

        if cfg.vocab_size < self.n_special:
            raise ValueError(f"vocab_size must be >= {self.n_special}, got {cfg.vocab_size}")

        self.special_embed = nn.Embedding(self.n_special, cfg.hidden_size)
        self.base_embed = nn.Embedding(cfg.vocab_size - self.n_special, cfg.hidden_size)
        self.drop = nn.Dropout(cfg.dropout)

        # 模态适配器（特征 token）。
        self.img_proj = nn.Linear(cfg.image_feat_dim, cfg.hidden_size, bias=False)

        # 门控参数初始为 0：训练初期等价于纯文本模型。
        self.img_gate = nn.Parameter(torch.zeros(cfg.hidden_size))

        head_dim = cfg.hidden_size // cfg.num_heads
        self.rope = RotaryEmbedding(dim=head_dim, max_seq_len=cfg.max_seq_len, theta=cfg.rope_theta)

        self.blocks = nn.ModuleList(
            [
                SeekerBlock(
                    cfg.hidden_size,
                    cfg.num_heads,
                    cfg.num_kv_heads,
                    dropout=cfg.dropout,
                    intermediate_size=cfg.mlp_intermediate_size,
                )
                for _ in range(cfg.num_layers)
            ]
        )
        self.norm = RMSNorm(cfg.hidden_size)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.special_embed.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.base_embed.weight, mean=0.0, std=0.02)

        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)

        # 将模态占位符/标记的词嵌入初始化为 0，避免引入噪声。
        with torch.no_grad():
            for tid in (
                self.special.img_bos,
                self.special.img,
                self.special.img_eos,
            ):
                self.special_embed.weight[tid].zero_()

    def _embed_tokens(self, input_ids: torch.Tensor) -> torch.Tensor:
        # input_ids: [B,S]
        n_special = int(self.n_special)
        is_special = input_ids < n_special

        x = torch.empty(
            (*input_ids.shape, self.cfg.hidden_size),
            device=input_ids.device,
            dtype=self.special_embed.weight.dtype,
        )

        if is_special.any():
            x[is_special] = self.special_embed(input_ids[is_special])
        if (~is_special).any():
            base_ids = (input_ids[~is_special] - n_special).clamp(min=0)
            x[~is_special] = self.base_embed(base_ids)
        return x

    def _lm_head_weight(self) -> torch.Tensor:
        # 与词嵌入权重共享。
        return torch.cat([self.special_embed.weight, self.base_embed.weight], dim=0)

    def _inject_modality_tokens(
        self,
        x: torch.Tensor,
        *,
        input_ids: torch.Tensor,
        image_feats: torch.Tensor | None,
    ) -> torch.Tensor:
        return inject_feature_tokens(
            x,
            input_ids=input_ids,
            image_feats=image_feats,
            img_token_id=int(self.special.img),
            img_proj=self.img_proj,
            img_gate=self.img_gate,
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        *,
        attention_mask: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        image_feats: torch.Tensor | None = None,
        past_kv: KVCache | None = None,
        use_cache: bool = False,
        position_offset: int = 0,
    ) -> SeekerOmniOutput:
        if past_kv is not None and not use_cache:
            raise ValueError("past_kv requires use_cache=True")
        if past_kv is not None and len(past_kv) != len(self.blocks):
            raise ValueError(f"past_kv length mismatch: got {len(past_kv)} expected {len(self.blocks)}")

        x = self._embed_tokens(input_ids)
        x = self._inject_modality_tokens(x, input_ids=input_ids, image_feats=image_feats)
        x = self.drop(x)

        seq_len = int(input_ids.shape[1])
        cos, sin = self.rope.get_cos_sin(seq_len, device=x.device, dtype=x.dtype, offset=int(position_offset))

        present_kv: KVCache | None = [] if use_cache else None

        for i, block in enumerate(self.blocks):
            if use_cache:
                pkv = past_kv[i] if past_kv is not None else None
                x, kv = block(x, cos=cos, sin=sin, attention_mask=attention_mask, past_kv=pkv, use_cache=True)
                present_kv.append(kv)
            else:
                x = block(x, cos=cos, sin=sin, attention_mask=attention_mask)

        x = self.norm(x)
        logits = F.linear(x, self._lm_head_weight())

        loss = None
        if labels is not None:
            shift_logits = logits[:, :-1, :]
            shift_labels = labels[:, 1:]

            if image_feats is not None:
                # 加权损失：强调监督区间的前若干 token（有助于描述任务在贪心解码下正确起步）。
                token_loss = F.cross_entropy(
                    shift_logits.reshape(-1, shift_logits.size(-1)),
                    shift_labels.reshape(-1),
                    ignore_index=-100,
                    reduction="none",
                ).view(shift_labels.shape[0], shift_labels.shape[1])

                mask = (shift_labels != -100).to(dtype=token_loss.dtype)
                weights = torch.ones_like(token_loss)

                has = mask.sum(dim=1) > 0
                if bool(has.any()):
                    first = torch.argmax(mask, dim=1)
                    k = 16
                    alpha = 3.0
                    for b in range(int(shift_labels.shape[0])):
                        if not bool(has[b]):
                            continue
                        s = int(first[b].item())
                        e = min(s + k, int(weights.shape[1]))
                        weights[b, s:e] = alpha

                denom = (weights * mask).sum().clamp_min(1.0)
                loss = (token_loss * weights * mask).sum() / denom
            else:
                loss = F.cross_entropy(
                    shift_logits.reshape(-1, shift_logits.size(-1)),
                    shift_labels.reshape(-1),
                    ignore_index=-100,
                )


        return SeekerOmniOutput(logits=logits, loss=loss, kv_cache=present_kv)

    def _protected_intervals(self, input_ids: torch.Tensor) -> list[list[tuple[int, int]]]:
        """返回每个样本需要保护的 [start,end) 区间，避免 streaming prefill 时把区间切开。"""
        bsz, seq_len = input_ids.shape
        out: list[list[tuple[int, int]]] = [[] for _ in range(int(bsz))]

        def find_interval(row: torch.Tensor, bos_id: int, eos_id: int) -> tuple[int, int] | None:
            bos = torch.where(row == int(bos_id))[0]
            if bos.numel() == 0:
                return None
            s = int(bos[0].item())
            eos = torch.where((row == int(eos_id)) & (torch.arange(int(row.numel()), device=row.device) > s))[0]
            if eos.numel() == 0:
                return None
            e = int(eos[0].item()) + 1
            if not (0 <= s < e <= int(seq_len)):
                return None
            return s, e

        for b in range(int(bsz)):
            row = input_ids[b]
            for bos_id, eos_id in (
                (self.special.img_bos, self.special.img_eos),
            ):
                interval = find_interval(row, bos_id, eos_id)
                if interval is not None:
                    out[b].append(interval)

        return out

    @torch.no_grad()
    def generate_text(
        self,
        input_ids: torch.Tensor,
        *,
        attention_mask: torch.Tensor | None = None,
        image_feats: torch.Tensor | None = None,
        max_new_tokens: int = 64,
        eos_id: int | None = None,
        min_id: int = 0,
        max_id: int | None = None,
        prefill_chunk_size: int | None = None,
        temperature: float = 0.0,
        top_p: float = 1.0,
        top_k: int = 0,
        repetition_penalty: float = 1.0,
        repetition_window: int = 256,
        no_repeat_ngram_size: int = 0,
    ) -> torch.Tensor:
        """带 KV 缓存的文本生成。

        - 默认使用贪心解码（`temperature<=0`）。
        - 可选采样：temperature/top_p/top_k/repetition_penalty。
        - 可选防循环：`no_repeat_ngram_size`。
        - `prefill_chunk_size` 启用 streaming prefill（分块处理 prompt），同时避免在模态占位符区间内切分。
        """
        self.eval()

        if input_ids.ndim != 2:
            raise ValueError(f"input_ids must be [B,S], got {tuple(input_ids.shape)}")

        out = input_ids
        bsz, prompt_len = out.shape

        if int(prompt_len) > int(self.cfg.max_seq_len):
            raise ValueError(f"prompt too long: {prompt_len} > cfg.max_seq_len={self.cfg.max_seq_len}")

        max_total = int(self.cfg.max_seq_len)
        remaining = max_total - int(prompt_len)
        if remaining <= 0:
            return out

        max_new_tokens = int(max_new_tokens)
        max_new_tokens = min(max_new_tokens, remaining)

        max_id = int(max_id) if max_id is not None else int(self.cfg.vocab_size)

        # attention_mask 作为 key mask（1=有效 key），本代码库只做 key 侧 mask。
        if attention_mask is None:
            key_mask = torch.ones((int(bsz), int(prompt_len)), device=out.device, dtype=torch.float32)
        else:
            key_mask = attention_mask.to(device=out.device, dtype=torch.float32)
            if int(key_mask.shape[1]) < int(prompt_len):
                pad = int(prompt_len) - int(key_mask.shape[1])
                key_mask = torch.cat(
                    [key_mask, torch.ones((int(bsz), pad), device=out.device, dtype=key_mask.dtype)],
                    dim=1,
                )
            elif int(key_mask.shape[1]) > int(prompt_len):
                key_mask = key_mask[:, : int(prompt_len)]

        # 流式预填充：为提示词构建 KV 缓存。
        protected = self._protected_intervals(out)

        chunk = int(prefill_chunk_size) if prefill_chunk_size is not None and int(prefill_chunk_size) > 0 else int(prompt_len)
        chunk = max(1, chunk)

        past_kv: KVCache | None = None
        logits_prev: torch.Tensor | None = None

        start = 0
        while start < int(prompt_len):
            end = min(start + chunk, int(prompt_len))

            if end < int(prompt_len):
                changed = True
                while changed:
                    changed = False
                    for b in range(int(bsz)):
                        for s, e in protected[b]:
                            if s < end < e:
                                end = max(end, e)
                                changed = True
                end = min(end, int(prompt_len))

            chunk_ids = out[:, start:end]
            chunk_key_mask = key_mask[:, :end]

            out_prefill = self(
                chunk_ids,
                attention_mask=chunk_key_mask,
                labels=None,
                image_feats=image_feats,
                past_kv=past_kv,
                use_cache=True,
                position_offset=int(start),
            )
            past_kv = out_prefill.kv_cache
            logits_prev = out_prefill.logits[:, -1, :]

            start = end

        if logits_prev is None or past_kv is None:
            raise RuntimeError('prefill failed to produce logits/cache')

        forbid_ids = [
            int(self.special.pad),
            int(self.special.img_bos),
            int(self.special.img),
            int(self.special.img_eos),
        ]
        # ChatML 方案：禁止在答案中生成 <|im_start|>。
        if self.special_spec.name == "minimind2_chatml":
            forbid_ids.append(int(self.special.bos))

        forbid = torch.tensor(sorted(set(forbid_ids)), device=out.device)

        rep = float(repetition_penalty)
        rep_window = int(repetition_window)
        no_repeat = int(no_repeat_ngram_size)
        do_sample = float(temperature) > 0.0
        top_p = float(top_p)
        top_k = int(top_k)

        neg_inf = torch.finfo(logits_prev.dtype).min

        def apply_repetition_penalty(logits: torch.Tensor, ids: torch.Tensor) -> None:
            if rep <= 1.0:
                return
            if ids.numel() == 0:
                return
            if rep_window > 0 and int(ids.shape[1]) > rep_window:
                ids = ids[:, -rep_window:]
            for b in range(int(ids.shape[0])):
                uniq = torch.unique(ids[b])
                if uniq.numel() == 0:
                    continue
                vals = logits[b, uniq]
                pos = vals > 0
                vals = torch.where(pos, vals / rep, vals * rep)
                logits[b, uniq] = vals

        def apply_no_repeat_ngram(logits: torch.Tensor, ids: torch.Tensor) -> None:
            if no_repeat <= 1:
                return
            for b in range(int(ids.shape[0])):
                row = ids[b].tolist()
                if len(row) < no_repeat:
                    continue

                table: dict[tuple[int, ...], set[int]] = {}
                for i in range(len(row) - no_repeat + 1):
                    key = tuple(row[i : i + no_repeat - 1])
                    nxt = int(row[i + no_repeat - 1])
                    s = table.get(key)
                    if s is None:
                        table[key] = {nxt}
                    else:
                        s.add(nxt)

                cur = tuple(row[-(no_repeat - 1) :])
                banned = table.get(cur)
                if not banned:
                    continue

                ban_ids = torch.tensor(list(banned), device=logits.device, dtype=torch.long)
                logits[b].index_fill_(0, ban_ids, neg_inf)

        def sample_from_logits(logits: torch.Tensor) -> torch.Tensor:
            if top_k > 0:
                k = min(int(top_k), int(logits.shape[-1]))
                v, _ = torch.topk(logits, k, dim=-1)
                kth = v[:, -1].unsqueeze(-1)
                logits = torch.where(logits < kth, torch.full_like(logits, neg_inf), logits)

            if 0.0 < top_p < 1.0:
                sorted_logits, sorted_idx = torch.sort(logits, descending=True, dim=-1)
                probs = torch.softmax(sorted_logits, dim=-1)
                cum = probs.cumsum(dim=-1)
                cut = cum > float(top_p)
                cut[:, 0] = False
                sorted_logits = torch.where(cut, torch.full_like(sorted_logits, neg_inf), sorted_logits)
                logits = logits.scatter(1, sorted_idx, sorted_logits)

            probs = torch.softmax(logits, dim=-1)
            return torch.multinomial(probs, num_samples=1)

        for _ in range(int(max_new_tokens)):
            next_logits = logits_prev.clone()

            # 鼓励“新颖性”（重复惩罚）。
            apply_repetition_penalty(next_logits, out)

            # 强制只输出文本（禁止控制/模态 token）。
            next_logits.index_fill_(1, forbid, neg_inf)

            if min_id > 0:
                next_logits[:, : int(min_id)] = neg_inf
            if max_id < int(next_logits.shape[-1]):
                next_logits[:, int(max_id) :] = neg_inf

            # 避免精确的 n-gram 循环。
            apply_no_repeat_ngram(next_logits, out)

            if do_sample:
                scaled = next_logits / float(temperature)
                next_id = sample_from_logits(scaled)
            else:
                next_id = torch.argmax(next_logits, dim=-1, keepdim=True)  # [B,1]

            out = torch.cat([out, next_id], dim=1)
            key_mask = torch.cat([key_mask, torch.ones((int(bsz), 1), device=out.device, dtype=key_mask.dtype)], dim=1)

            if eos_id is not None and torch.all(next_id.squeeze(-1) == int(eos_id)):
                break

            past_len = int(past_kv[0][0].shape[2])

            out_step = self(
                next_id,
                attention_mask=key_mask,
                labels=None,
                image_feats=image_feats,
                past_kv=past_kv,
                use_cache=True,
                position_offset=past_len,
            )
            past_kv = out_step.kv_cache
            logits_prev = out_step.logits[:, -1, :]

        return out
