import math, torch, torch.nn.functional as F
from torch import nn
from transformers.activations import ACT2FN
from transformers import PreTrainedModel, GenerationMixin, PretrainedConfig
from transformers.modeling_outputs import MoeCausalLMOutputWithPast

# 🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏
#                                     MiniMind Config
# 🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏
class MiniMindConfig(PretrainedConfig):
    model_type = "minimind"
    def __init__(self, hidden_size=768, num_hidden_layers=8, use_moe=False, **kwargs):
        super().__init__(**kwargs)
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.use_moe = use_moe
        self.dropout = kwargs.get("dropout", 0.0)
        self.vocab_size = kwargs.get("vocab_size", 6400)
        self.bos_token_id = kwargs.get("bos_token_id", 1)
        self.eos_token_id = kwargs.get("eos_token_id", 2)
        self.flash_attn = kwargs.get("flash_attn", True)
        self.num_attention_heads = kwargs.get("num_attention_heads", 8)
        self.num_key_value_heads = kwargs.get("num_key_value_heads", 4)
        self.head_dim = kwargs.get("head_dim", self.hidden_size // self.num_attention_heads)
        self.hidden_act = kwargs.get("hidden_act", 'silu')
        self.intermediate_size = kwargs.get("intermediate_size", math.ceil(hidden_size * math.pi / 64) * 64)
        self.max_position_embeddings = kwargs.get("max_position_embeddings", 32768)
        self.rms_norm_eps = kwargs.get("rms_norm_eps", 1e-6)
        self.rope_theta = kwargs.get("rope_theta", 1e6)
        self.tie_word_embeddings = kwargs.get("tie_word_embeddings", True)
        self.inference_rope_scaling = kwargs.get("inference_rope_scaling", False)
        self.rope_scaling = {
            "beta_fast": 32,
            "beta_slow": 1,
            "factor": 16,
            "original_max_position_embeddings": 2048,
            "attention_factor": 1.0,
            "type": "yarn"
        } if self.inference_rope_scaling else None
        ### MoE specific configs (ignored if use_moe = False)
        self.num_experts = kwargs.get("num_experts", 4)
        self.num_experts_per_tok = kwargs.get("num_experts_per_tok", 1)
        self.moe_intermediate_size = kwargs.get("moe_intermediate_size", self.intermediate_size)
        self.norm_topk_prob = kwargs.get("norm_topk_prob", True)
        self.router_aux_loss_coef = kwargs.get("router_aux_loss_coef", 5e-4)

# 🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏
#                                     MiniMind Model
# 🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏
class RMSNorm(torch.nn.Module):
    """
    RMSNorm（Root Mean Square Layer Normalization，均方根归一化）。

    相比传统 LayerNorm，它去掉了「减均值」这一步（均值中心化）和偏置 b，
    只保留「除以均方根」的缩放，并额外引入一个可学习的逐维度缩放系数 weight（γ）。
    计算量更小，实践上效果相当，是 LLaMA / DeepSeek / Qwen 等现代 LLM 的标准配置。

    数学定义（对单个向量 x ∈ R^D，D 即 dim）：

        RMS(x)            = sqrt( mean(x²) )              # 均方根
        RMSNorm(x)_i      = ( x_i / RMS(x) ) * γ_i        # 逐元素缩放，γ_i 初始为 1

    代码里等价写成 x * rsqrt(mean(x²) + eps)，其中 rsqrt(z) = 1/sqrt(z)，
    即「除以均方根」用「乘以均方根的倒数」实现，少做一次除法。
    eps 是一个极小正数，防止 mean(x²) 恰好为 0 时除 0。

    具体矩阵举例（dim=4，eps 忽略不计，x 形状 [2,4]，即 2 个 token、每个 4 维）：

        x = [[1, 2, 3, 4],
             [2, 4, 6, 8]]

        逐行（最后一个维度 dim=-1）独立归一化：

        第 1 行 [1, 2, 3, 4]：
            x.pow(2)       = [1, 4, 9, 16]
            mean(-1)       = (1 + 4 + 9 + 16) / 4 = 7.5
            rsqrt(7.5)     = 1/sqrt(7.5) ≈ 0.3651
            norm 结果       = [1,2,3,4] * 0.3651 = [0.365, 0.730, 1.095, 1.461]

        第 2 行 [2, 4, 6, 8]：
            mean(x²)       = (4+16+36+64)/4 = 30
            rsqrt(30)      ≈ 0.1826
            norm 结果       = [2,4,6,8] * 0.1826 = [0.365, 0.730, 1.095, 1.461]

        注意到第 2 行恰好是第 1 行的 2 倍，归一化后两行结果完全相同——
        这正体现了 RMSNorm 的「尺度不变性」：向量整体放大 k 倍后，归一化输出不变。

    再乘以可学习权重 weight（γ，初始全 1）得到最终输出；训练时 γ 会学到逐维度的增益。
    """
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        # 可学习缩放系数 γ，形状 [dim]，初始化为全 1（即一开始不做缩放，恒等变换）
        self.weight = nn.Parameter(torch.ones(dim))

    def norm(self, x):
        # x.pow(2)                    : 逐元素平方，形状不变
        # .mean(-1, keepdim=True)     : 沿最后一个维度（特征维）求均值，keepdim 保留该维，
        #                               使结果形状与 x 广播兼容（如 [2,4] -> [2,1]）
        # torch.rsqrt(z)              : 1/sqrt(z)
        # 最终等价于 x / sqrt(mean(x²) + eps)
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        # 1. x.float()：先把输入转成 float32 再归一化。
        #    因为训练/推理时 x 常是 bf16/fp16，直接平方、求均值容易因精度不足导致误差，
        #    float32 计算更稳定。
        # 2. self.weight * ...：广播乘上逐维度增益 γ（形状 [dim] 广播到 [..., dim]）。
        # 3. .type_as(x)：最后把结果转回 x 的原始 dtype（如 bf16），保持输出类型与输入一致。
        return (self.weight * self.norm(x.float())).type_as(x)

def precompute_freqs_cis(dim: int, end: int = int(32 * 1024), rope_base: float = 1e6, rope_scaling: dict = None):
    freqs, attn_factor = 1.0 / (rope_base ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim)), 1.0
    if rope_scaling is not None: # YaRN: f'(i) = f(i)((1-γ) + γ/s), where γ∈[0,1] is linear ramp
        orig_max, factor, beta_fast, beta_slow, attn_factor = (
            rope_scaling.get("original_max_position_embeddings", 2048), rope_scaling.get("factor", 16),
            rope_scaling.get("beta_fast", 32.0), rope_scaling.get("beta_slow", 1.0), rope_scaling.get("attention_factor", 1.0)
        )
        if end / orig_max > 1.0:
            inv_dim = lambda b: (dim * math.log(orig_max / (b * 2 * math.pi))) / (2 * math.log(rope_base))
            low, high = max(math.floor(inv_dim(beta_fast)), 0), min(math.ceil(inv_dim(beta_slow)), dim // 2 - 1)
            ramp = torch.clamp((torch.arange(dim // 2, device=freqs.device).float() - low) / max(high - low, 0.001), 0, 1)
            freqs = freqs * (1 - ramp + ramp / factor)
    t = torch.arange(end, device=freqs.device)
    freqs = torch.outer(t, freqs).float()
    freqs_cos = torch.cat([torch.cos(freqs), torch.cos(freqs)], dim=-1) * attn_factor
    freqs_sin = torch.cat([torch.sin(freqs), torch.sin(freqs)], dim=-1) * attn_factor
    return freqs_cos, freqs_sin

def apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1):
    def rotate_half(x): return torch.cat((-x[..., x.shape[-1] // 2:], x[..., : x.shape[-1] // 2]), dim=-1)
    q_embed = ((q * cos.unsqueeze(unsqueeze_dim)) + (rotate_half(q) * sin.unsqueeze(unsqueeze_dim))).to(q.dtype)
    k_embed = ((k * cos.unsqueeze(unsqueeze_dim)) + (rotate_half(k) * sin.unsqueeze(unsqueeze_dim))).to(k.dtype)
    return q_embed, k_embed

def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    bs, slen, num_key_value_heads, head_dim = x.shape
    if n_rep == 1: return x
    return (x[:, :, :, None, :].expand(bs, slen, num_key_value_heads, n_rep, head_dim).reshape(bs, slen, num_key_value_heads * n_rep, head_dim))

class Attention(nn.Module):
    """
    多头自注意力（Multi-Head Self-Attention），采用 GQA（Grouped Query Attention，分组查询注意力）。

    GQA 思想：查询头（Q）数量多于键/值头（K/V）数量，多个 Q 头共享同一个 K/V 头，
    从而在「注意力质量」与「KV cache 显存/带宽」之间取平衡。
    这里的 n_rep = n_local_heads / n_local_kv_heads 表示「每个 K/V 头被多少个 Q 头复用」。

    以默认配置为例：hidden=768、n_heads=8、kv_heads=4、head_dim=96。
        n_rep = 8 / 4 = 2，即每 2 个 Q 头共享 1 个 K/V 头。

    完整数据流（默认配置，单层，seq_len=S，KV 无 cache 分支）：

        x  [B, S, 768]
          --q_proj--> xq [B, S, 8*96]  --view--> [B, S, 8, 96]   (8 个 Q 头)
          --k_proj--> xk [B, S, 4*96]  --view--> [B, S, 4, 96]   (4 个 K 头)
          --v_proj--> xv [B, S, 4*96]  --view--> [B, S, 4, 96]   (4 个 V 头)
          --q_norm/k_norm（对 head_dim 做 RMSNorm）--> 保持形状
          --RoPE 旋转位置编码--> 保持形状
          --repeat_kv(xk, n_rep=2)--> [B, S, 8, 96]（每个 K/V 头复制 2 份对齐 Q 头）
          --transpose(1,2)--> Q/K/V 变成 [B, 8, S, 96]（头维提到第 2 维，方便批量矩阵乘）
          --scores = Q·Kᵀ/√d--> [B, 8, S, S]  --softmax(最后维)--> 注意力权重
          --output = attn·V--> [B, 8, S, 96]
          --transpose(1,2)+reshape--> [B, S, 768]  --o_proj--> [B, S, 768]

    注意力打分（Q·Kᵀ）的具体矩阵举例（为便于手算，取 1 个 head、head_dim=2、S=3）：

        Q = [[1, 0],      K = [[1, 0],
             [0, 1],           [1, 1],
             [1, 1]]           [0, 1]]

        Kᵀ = [[1, 1, 0],
              [0, 1, 1]]

        Q·Kᵀ（第 i 行第 j 列 = 第 i 个 token 的 query 与第 j 个 token 的 key 的内积）：
            = [[1, 1, 0],
               [0, 1, 1],
               [1, 2, 1]]

        除以 √d（d=head_dim=2，√2≈1.414）：
            = [[0.707, 0.707, 0.000],
               [0.000, 0.707, 0.707],
               [0.707, 1.414, 0.707]]

        加因果 mask（下三角之外的未来位置设为 -inf）：
            = [[0.707,  -inf,  -inf],
               [0.000, 0.707,  -inf],
               [0.707, 1.414, 0.707]]

        softmax 后（-inf 位置 exp→0）：
            = [[1.000, 0.000, 0.000],      <- token0 只能看自己
               [0.330, 0.670, 0.000],      <- token1 看 token0、token1
               [0.248, 0.504, 0.248]]      <- token2 看 token0、token1、token2

        最后 output = softmax(Q·Kᵀ/√d) · V：每个 token 的输出 = 按注意力权重加权求和各 token 的 V。
    """
    def __init__(self, config: MiniMindConfig):
        super().__init__()
        # K/V 头数：若未显式配置 num_key_value_heads 则退化为普通 MHA（K/V 头数 = Q 头数）
        self.num_key_value_heads = config.num_attention_heads if config.num_key_value_heads is None else config.num_key_value_heads
        self.n_local_heads = config.num_attention_heads        # Q 头数（如 8）
        self.n_local_kv_heads = self.num_key_value_heads       # K/V 头数（如 4）
        self.n_rep = self.n_local_heads // self.n_local_kv_heads  # 每个 K/V 头被几个 Q 头复用（如 2）
        self.head_dim = config.head_dim                        # 每个头的维度（如 96）
        self.is_causal = True                                  # 因果注意力：当前 token 只能看到过去
        # Q 投影：hidden -> n_heads * head_dim（Q 头全量，如 8*96）
        self.q_proj = nn.Linear(config.hidden_size, config.num_attention_heads * self.head_dim, bias=False)
        # K/V 投影：hidden -> kv_heads * head_dim（K/V 头是「组」级别，如 4*96）
        self.k_proj = nn.Linear(config.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        # 输出投影：把拼接后的所有头结果映射回 hidden
        self.o_proj = nn.Linear(config.num_attention_heads * self.head_dim, config.hidden_size, bias=False)
        # 对 Q、K 做 RMSNorm（DeepSeek 风格）：归一化后再进 RoPE，提升数值稳定/训练稳定
        self.q_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.attn_dropout = nn.Dropout(config.dropout)          # 注意力权重上的 dropout
        self.resid_dropout = nn.Dropout(config.dropout)         # 输出（残差侧）上的 dropout
        self.dropout = config.dropout
        # 是否可用 Flash Attention（SDPA）：训练/推理更快、更省显存
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention') and config.flash_attn

    def forward(self, x, position_embeddings, past_key_value=None, use_cache=False, attention_mask=None):
        bsz, seq_len, _ = x.shape
        # 1. 线性投影得到 Q/K/V（此时还是「平铺」的最后一维，未拆出头）
        #    xq: [B, S, n_heads*head_dim]，xk/xv: [B, S, kv_heads*head_dim]
        xq, xk, xv = self.q_proj(x), self.k_proj(x), self.v_proj(x)
        # 2. 拆出头维度：把最后一维切成 (heads, head_dim)
        #    例：xq [B,S,8*96] -> [B,S,8,96]；xk/xv [B,S,4*96] -> [B,S,4,96]
        xq = xq.view(bsz, seq_len, self.n_local_heads, self.head_dim)
        xk = xk.view(bsz, seq_len, self.n_local_kv_heads, self.head_dim)
        xv = xv.view(bsz, seq_len, self.n_local_kv_heads, self.head_dim)
        # 3. 对 Q、K 做 RMSNorm（在最后一个维度 head_dim 上归一化）
        xq, xk = self.q_norm(xq), self.k_norm(xk)
        # 4. 旋转位置编码 RoPE：给 Q、K 注入相对位置信息（对 V 不作用）
        cos, sin = position_embeddings
        xq, xk = apply_rotary_pos_emb(xq, xk, cos, sin)
        # 5. KV cache：增量推理时把「历史 token 的 K/V」拼在当前 K/V 前面（dim=1 是序列维）
        #    例：past_key_value[0] [B, S_prev, 4, 96] + xk [B, S_cur, 4, 96] -> [B, S_prev+S_cur, 4, 96]
        if past_key_value is not None:
            xk = torch.cat([past_key_value[0], xk], dim=1)
            xv = torch.cat([past_key_value[1], xv], dim=1)
        # use_cache=True 时把当前 K/V 作为新的 cache 返回给下一轮
        past_kv = (xk, xv) if use_cache else None
        # 6. 组对齐 + 转置成 (B, heads, S, head_dim)，方便批量矩阵乘
        #    repeat_kv：把 K/V 从 kv_heads 复制 n_rep 倍对齐到 Q 头数
        #    例：xk [B, S_prev+S_cur, 4, 96] --repeat_kv(2)--> [B, S, 8, 96]
        #    transpose(1,2)：把「头维」从第 3 维提到第 2 维 -> [B, 8, S, 96]
        xq, xk, xv = (xq.transpose(1, 2), repeat_kv(xk, self.n_rep).transpose(1, 2), repeat_kv(xv, self.n_rep).transpose(1, 2))
        # 7. 计算注意力（优先 Flash Attention，否则手动实现）
        #    Flash 条件：可用 + 序列长>1 + 非因果（或首次无 cache）+ 无 mask（或全 1）
        if self.flash and (seq_len > 1) and (not self.is_causal or past_key_value is None) and (attention_mask is None or torch.all(attention_mask == 1)):
            output = F.scaled_dot_product_attention(xq, xk, xv, dropout_p=self.dropout if self.training else 0.0, is_causal=self.is_causal)
        else:
            # 7a. 打分：scores = Q·Kᵀ / √d，形状 [B, heads, S_q, S_k]
            #     xq [B,8,S,96] @ xk^T [B,8,96,S] -> [B,8,S,S]，除以 √head_dim 缩放防止内积过大导致 softmax 饱和
            scores = (xq @ xk.transpose(-2, -1)) / math.sqrt(self.head_dim)
            # 7b. 因果 mask：把「未来」位置（上三角）置为 -inf，使 softmax 后权重为 0
            #     triu(1) 生成上三角（不含主对角线）为 1 的矩阵 [S,S]，加到分数最后 seq_len 个 query 上
            if self.is_causal: scores[:, :, :, -seq_len:] += torch.full((seq_len, seq_len), float("-inf"), device=scores.device).triu(1)
            # 7c. padding mask：把 padding 位置（attention_mask=0）对应分数减 1e9（近似 -inf）
            if attention_mask is not None: scores += (1.0 - attention_mask.unsqueeze(1).unsqueeze(2)) * -1e9
            # 7d. softmax + 加权求和：attn = softmax(scores)，output = attn · V
            #     softmax 用 float32 算（数值稳定）再转回 xq 的 dtype；dropout 作用于注意力权重
            output = self.attn_dropout(F.softmax(scores.float(), dim=-1).type_as(xq)) @ xv
        # 8. 还原形状 + 输出投影
        #    output [B,8,S,96] --transpose(1,2)--> [B,S,8,96] --reshape--> [B,S,8*96]
        #    --o_proj--> [B,S,768]，再经 resid_dropout 返回
        output = output.transpose(1, 2).reshape(bsz, seq_len, -1)
        output = self.resid_dropout(self.o_proj(output))
        return output, past_kv

class FeedForward(nn.Module):
    """
    SwiGLU 前馈网络（Feed-Forward Network，FFN），Transformer 块里注意力之后的非线性变换层。

    FFN 先把 hidden 维「升维」到更大的 intermediate 维做非线性变换，再「降维」回 hidden 维，
    用更大的中间宽度换取更强的表达能力（约占模型 2/3 的参数）。

    这里用的是 SwiGLU 结构（LLaMA/DeepSeek/Qwen 标配），公式：

        FFN(x) = down_proj( SiLU(gate_proj(x)) ⊙ up_proj(x) )

    拆开看三个投影各自的角色：
        gate_proj: hidden -> intermediate   生成「门控」值
        up_proj:   hidden -> intermediate   生成「被门控」的值
        down_proj: intermediate -> hidden   把升维后的结果映射回原维度
    其中 ⊙ 表示逐元素相乘（哈达玛积），SiLU(z) = z·sigmoid(z)。

    为什么叫「门控」：SiLU(gate) 取值范围在 (-0.28, +∞)，可正可负，
    它按位缩放 up 的结果 —— 正数放大、负数反转、接近 0 抑制，实现软门控（soft gating），
    比早期 ReLU 那种硬开关（负值直接归零）信息流失更少。

    形状流（默认配置：hidden=768，intermediate≈2432 = ceil(768·π/64)·64）：

        x  [B, S, 768]
          --gate_proj--> gate [B, S, 2432]  --SiLU--> [B, S, 2432]
          --up_proj----> up   [B, S, 2432]
          --> 逐元素相乘 SiLU(gate) ⊙ up --> [B, S, 2432]
          --down_proj--> [B, S, 768]

    具体矩阵举例（为便于手算，取单 token、hidden=2、intermediate=4，权重为示意值）：

        x = [1, -2]

        gate = gate_proj(x) = [ 0.5, -1.0,  2.0,  0.0]
        up   = up_proj(x)   = [ 1.0,  2.0,  0.5, -0.5]

        SiLU(z) = z·sigmoid(z)，逐位计算：
            SiLU( 0.5) ≈  0.5 · 0.622 =  0.311
            SiLU(-1.0) ≈ -1.0 · 0.269 = -0.269
            SiLU( 2.0) ≈  2.0 · 0.881 =  1.761
            SiLU( 0.0) =  0.0
        => SiLU(gate) = [ 0.311, -0.269, 1.761, 0.0]

        SiLU(gate) ⊙ up（逐元素相乘）：
            [ 0.311, -0.269, 1.761, 0.0] ⊙ [1.0, 2.0, 0.5, -0.5]
          = [ 0.311, -0.538, 0.881, 0.0]

        output = down_proj( [0.311, -0.538, 0.881, 0.0] )  ->  [B, S, 2]（映射回 hidden=2）

        注意第 4 位：gate=0 -> SiLU=0 -> 乘积=0，即门控「关闭」了 up 的第 4 维；
        而第 2 位 gate=-1 -> SiLU=-0.269，把 up 的 +2.0 反转成 -0.538，即「负门控」。
    """
    def __init__(self, config: MiniMindConfig, intermediate_size: int = None):
        super().__init__()
        # 中间层宽度：默认取 config.intermediate_size（≈ceil(hidden·π/64)·64），
        # MoE 场景下会传入更小的 moe_intermediate_size 覆盖它
        intermediate_size = intermediate_size or config.intermediate_size
        # 门控投影：生成「软开关」系数，经 SiLU 后逐位缩放 up 的结果
        self.gate_proj = nn.Linear(config.hidden_size, intermediate_size, bias=False)
        # 降维投影：把中间维结果映射回 hidden
        self.down_proj = nn.Linear(intermediate_size, config.hidden_size, bias=False)
        # 升维投影：生成「被门控」的值
        self.up_proj = nn.Linear(config.hidden_size, intermediate_size, bias=False)
        # 激活函数：默认 'silu'（即 SwiGLU 中的 SiLU）
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        # SwiGLU 计算：SiLU(gate_proj(x)) ⊙ up_proj(x) 再 down_proj
        # 1. self.gate_proj(x)    : [B,S,hidden] -> [B,S,intermediate]
        # 2. self.act_fn(...)     : 逐元素 SiLU（软门控）
        # 3. self.up_proj(x)      : [B,S,hidden] -> [B,S,intermediate]
        # 4. ... * ...            : 逐元素相乘（门控作用到值上）
        # 5. self.down_proj(...)  : [B,S,intermediate] -> [B,S,hidden]
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))

class MOEFeedForward(nn.Module):
    """
    MoE（混合专家）前馈层：把标准 FFN 换成「多个专家 FFN + 一个路由器」。
    每个 token 只经过被路由选中的少量专家（稀疏激活），从而用更多总参数换取更高容量，
    但保持激活参数量不变。

    数据流示意（以 hidden=4 / 3 专家 / 4 个 token / top-1 为例）：
        x [1,4,4] --view--> x_flat [4,4] --gate--> logits [4,3] --softmax--> scores [4,3]
        --topk(1)--> topk_idx [4,1] / topk_weight [4,1]
        --> 按专家分桶 --> 各专家只算命中自己的 token --> index_add_ 按行归位 --> y [4,4]
    """
    def __init__(self, config: MiniMindConfig):
        super().__init__()
        self.config = config
        # 路由器/门控：hidden_size -> num_experts，给每个 token 的每个专家打一个分数
        self.gate = nn.Linear(config.hidden_size, config.num_experts, bias=False)
        # num_experts 个独立的 SwiGLU FFN 专家
        self.experts = nn.ModuleList([FeedForward(config, intermediate_size=config.moe_intermediate_size) for _ in range(config.num_experts)])
        self.act_fn = ACT2FN[config.hidden_act]  # 冗余遗留，专家内部已使用 act_fn

    def forward(self, x):
        batch_size, seq_len, hidden_dim = x.shape
        # 1. 展平成 token 级：路由是逐 token 决策的
        #    x: [B, S, D] -> x_flat: [B*S, D]（N = B*S 个 token，每行一个 token）
        x_flat = x.view(-1, hidden_dim)
        # 2. 门控打分 + softmax：得到每个 token 对各专家的概率分布
        #    logits = x_flat @ W_gate : [N, num_experts]
        #    scores = softmax(logits, dim=-1) : [N, num_experts]，每行和为 1
        scores = F.softmax(self.gate(x_flat), dim=-1)
        # 3. Top-k 选专家：每行取得分最高的 k 个专家（默认 k=1 即 top-1）
        #    topk_idx   : [N, k] 被选中专家的索引
        #    topk_weight: [N, k] 对应权重
        #    例（k=1）: topk_idx = [[0],[2],[1],[0]] -> t0->专家0, t1->专家2, t2->专家1, t3->专家0
        topk_weight, topk_idx = torch.topk(scores, k=self.config.num_experts_per_tok, dim=-1, sorted=False)
        # 4. 归一化 top-k 权重（DeepSeek-MoE 风格）；k=1 时除完恒为 1，实际不起作用
        if self.config.norm_topk_prob: topk_weight = topk_weight / (topk_weight.sum(dim=-1, keepdim=True) + 1e-20)
        # 5. 逐专家分桶计算：初始输出全零，按 token 行号累加回 y
        #    接上例 topk_idx = [[0],[2],[1],[0]]，分桶结果：
        #      专家0 <- 行 0,3（t0,t3）   专家1 <- 行 2（t2）   专家2 <- 行 1（t1）
        y = torch.zeros_like(x_flat)  # [N, D]，例如 4×4 全 0
        for i, expert in enumerate(self.experts):
            # 哪些 token（行）被路由到专家 i：mask 形状 [N, k]
            #   例：i=0 时 mask = (topk_idx==0) = [[T],[F],[F],[T]] -> 行 0、3 命中
            mask = (topk_idx == i)
            if mask.any():
                # 命中专家 i 的 token 行号：token_idx 形状 [M]
                #   例：i=0 时 mask.any(dim=-1)=[T,F,F,T] -> .nonzero()=[[0],[3]] -> token_idx=[0,3]
                token_idx = mask.any(dim=-1).nonzero().flatten()
                # 对应权重：weight 形状 [M, 1]（top-1 且已归一化，恒为 1）
                weight = topk_weight[mask].view(-1, 1)
                # 只把命中的 M 个 token 喂给专家 i（稀疏激活），乘权重后按行号累加回 y
                # index_add_(0, idx, src) 语义: y[idx[j]] += src[j]
                #   例：i=0 且专家0=恒等映射时，
                #       expert(x_flat[[0,3]]) = [[1,0,0,1], [0,0,1,1]]
                #       y 第 0、3 行被写入 t0、t3，第 1、2 行仍为 0
                y.index_add_(0, token_idx, (expert(x_flat[token_idx]) * weight).to(y.dtype))
            elif self.training:
                # 本批次该专家没有任何 token 命中：用 0*sum(params) 把它的参数
                # 强行纳入 autograd 图，避免「专家饿死时梯度缺失」，数值上无影响
                y[0, 0] += 0 * sum(p.sum() for p in expert.parameters())
        # 循环结束后 y 的最终结果（每行 = 对应 token 经其专家后的输出）：
        #   y = [[1,0,0,1],   <- t0 走了专家0（恒等映射）
        #        [0,0,0,0],   <- t1 走了专家2（零映射）
        #        [2,2,0,0],   <- t2 走了专家1（放大2倍）
        #        [0,0,1,1]]   <- t3 走了专家0（恒等映射）
        # 6. 负载均衡辅助损失（Switch Transformer / DeepSeek 风格）：
        #    惩罚「路由不均」，防止所有 token 都挤向少数专家、其余专家「饿死」。
        #    公式：aux_loss = num_experts * coef * Σ( load ⊙ scores.mean(0) )
        #       load            = 每个专家「实际被选中」的平均频率（one-hot 后按 token 平均）
        #       scores.mean(0)  = 门控「想选」各专家的平均概率
        #    load ⊙ scores.mean(0) 逐元素相乘：既想让负载更均匀，又不想违背门控本身的偏好。
        #
        #    具体计算举例（num_experts=3，4 个 token）：
        #      topk_idx = [[0],[2],[1],[0]]
        #      one_hot 逐行: [1,0,0] / [0,0,1] / [0,1,0] / [1,0,0]
        #      load = ([1,0,0]+[0,0,1]+[0,1,0]+[1,0,0]) / 4 = [0.50, 0.25, 0.25]
        #      设 scores.mean(0) = [0.4375, 0.324, 0.2385]
        #      load ⊙ scores.mean(0) = [0.21875, 0.081, 0.05963]
        #      Σ(...) = 0.21875 + 0.081 + 0.05963 = 0.35938
        #      × num_experts(3) = 1.0781
        #      × coef(5e-4)     = 0.000539   （最终 aux_loss）
        if self.training and self.config.router_aux_loss_coef > 0:
            # load: 每个专家被实际选中的平均频率，形状 [num_experts]
            load = F.one_hot(topk_idx, self.config.num_experts).float().mean(0)
            # (load * scores.mean(0)).sum()：逐元素相乘后求和，即 Σ(load ⊙ scores.mean(0))
            self.aux_loss = (load * scores.mean(0)).sum() * self.config.num_experts * self.config.router_aux_loss_coef
        else:
            self.aux_loss = scores.new_zeros(1).squeeze()
        # 7. 还原形状输出
        return y.view(batch_size, seq_len, hidden_dim)  # [B*S, D] -> [B, S, D]

class MiniMindBlock(nn.Module):
    """
    单个 Transformer 解码器块（Decoder Block），LLM 堆叠 N 层这种块构成主干网络。

    结构是标准「Pre-Norm + 残差」范式（LLaMA/DeepSeek/Qwen 通用），包含两个子层：

        第 1 子层：注意力（自注意力，可带 KV cache）
        第 2 子层：前馈网络 FFN（稠密 SwiGLU 或 MoE）

    两个子层各带一个残差连接，且都采用 **Pre-Norm（前置归一化）**：
    即「先归一化，再进子层，最后加回残差」。归一化不在残差路径上，
    梯度可以沿残差「高速公路」更直接地回传，缓解深层网络的梯度消失/爆炸，训练更稳定。

    数学表达（记 LN 为 RMSNorm，⊕ 为残差相加）：

        h' = h        ⊕ Attention( LN1(h) )
        h  = h'       ⊕ FFN( LN2(h') )

    数据流（默认配置 hidden=768）：

        hidden_states [B, S, 768]
          ├─ residual = hidden_states                     （保存副本，供残差相加）
          ├─ LN1(input_layernorm) → [B, S, 768]
          │    └─ self_attn(...) → (out [B,S,768], present_kv)
          ├─ hidden_states = out + residual               （第 1 次残差相加）
          ├─ LN2(post_attention_layernorm) → [B, S, 768]
          │    └─ mlp(...) → [B, S, 768]
          └─ hidden_states = hidden_states + mlp_out      （第 2 次残差相加）

    残差连接的具体矩阵举例（hidden=4，单 token，子层输出为示意值）：

        输入 hidden_states            = [1, 2, 3, 4]

        第 1 子层（注意力）：
            residual                  = [1, 2, 3, 4]          <- 保存的输入副本
            attn_out                  = [0.1, -0.2, 0.3, -0.1]（假设注意力输出）
            hidden_states = attn_out + residual
                          = [1.1, 1.8, 3.3, 3.9]

        第 2 子层（FFN）：
            residual                  = [1.1, 1.8, 3.3, 3.9]  <- 保存上一层结果的副本
            mlp_out                   = [0.5, -0.3, 0.2, 0.4]（假设 FFN 输出）
            hidden_states = mlp_out + residual
                          = [1.6, 1.5, 3.5, 4.3]

        直觉：残差连接让子层「只需学习增量（残差）」，即便子层输出为 0，
        信息也能原样穿透到下一层；这也是梯度能稳定回传的原因。
    """
    def __init__(self, layer_id: int, config: MiniMindConfig):
        super().__init__()
        # 自注意力子层（GQA 多头注意力）
        self.self_attn = Attention(config)
        # 第 1 个子层的 Pre-Norm：注意力前的 RMSNorm（沿 hidden 维归一化）
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        # 第 2 个子层的 Pre-Norm：FFN 前的 RMSNorm
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        # 前馈子层：稠密 SwiGLU FFN，或 MoE 多专家 FFN（按 use_moe 选择）
        self.mlp = FeedForward(config) if not config.use_moe else MOEFeedForward(config)
        # 注：layer_id 参数当前未被本类使用（保留作扩展/调试用）

    def forward(self, hidden_states, position_embeddings, past_key_value=None, use_cache=False, attention_mask=None):
        # ---- 第 1 子层：注意力 ----
        # 保存输入作为残差基准
        residual = hidden_states
        # Pre-Norm 后再进注意力：self_attn(input_layernorm(x), ...)
        #   返回 (attn_out [B,S,768], present_key_value) —— present_key_value 是当前层的 KV cache
        hidden_states, present_key_value = self.self_attn(
            self.input_layernorm(hidden_states), position_embeddings,
            past_key_value, use_cache, attention_mask
        )
        # 残差相加：hidden_states = attn_out + residual
        hidden_states += residual
        # ---- 第 2 子层：FFN ----
        # Pre-Norm 后进 FFN，再加残差（注意：此处的残差基准是「注意力子层之后」的结果，
        # 而非最初输入，两个子层的残差是「逐子层」累加的）
        hidden_states = hidden_states + self.mlp(self.post_attention_layernorm(hidden_states))
        # 返回本层输出 + 本层的 KV cache（供生成时增量复用）
        return hidden_states, present_key_value

class MiniMindModel(nn.Module):
    """
    MiniMind 主干模型（不含 LM head，head 在 MiniMindForCausalLM 里）。

    职责：把 token id 序列一路前向成「每个位置的隐藏向量」，流程为
    词嵌入 -> (逐层 Transformer 块) -> 最终 RMSNorm。

    数据流（默认配置：vocab=6400、hidden=768、num_layers=8）：

        input_ids [B, S]
          --embed_tokens--> hidden_states [B, S, 768]   （查表得到每个 token 的向量）
          --dropout--> [B, S, 768]
          --×8 层 MiniMindBlock--> [B, S, 768]          （逐层前向，可带 KV cache）
          --final RMSNorm--> [B, S, 768]
          --> 返回 (hidden_states, presents, aux_loss)

    三个返回值：
        hidden_states : 最后一层输出的隐藏向量 [B, S, 768]，喂给 LM head 算 logits
        presents      : 每层各自的 KV cache 列表（use_cache=True 时非 None），供增量解码复用
        aux_loss      : 所有 MoE 层的负载均衡辅助损失之和（非 MoE 时为标量 0）

    词嵌入的具体矩阵举例（vocab=6400、hidden=768）：

        嵌入表 embed_tokens.weight 形状 [6400, 768]，第 i 行 = 词表第 i 个 token 的向量。
        input_ids = [[101, 2345, 42]]            （B=1, S=3）
        embed_tokens(input_ids) 即「按行查表」：
            取第 101 行 -> [ ...768 维... ]      <- token 101 的向量
            取第 2345 行 -> [ ...768 维... ]     <- token 2345 的向量
            取第 42 行  -> [ ...768 维... ]      <- token 42 的向量
        堆叠得到 [1, 3, 768]。
        本质等价于 input_ids 的 one-hot [1,3,6400] @ 嵌入表 [6400,768] 的矩阵乘。

    位置编码（RoPE）按 start_pos 切片的举例（增量解码时）：

        freqs_cos/freqs_sin 已预先算好整张表 [max_position_embeddings=32768, head_dim=96]。
        训练 / 首次推理（无 cache）：start_pos = 0，取 [0 : S] 覆盖全部位置。
        增量解码：已生成 5 个 token（past_len=5），本轮只喂 1 个新 token：
            start_pos = 5，取 freqs_cos[5:6]，只算「第 5 个位置」的旋转编码，
            历史位置的编码已在之前轮次算过、无需重复。
    """
    def __init__(self, config: MiniMindConfig):
        super().__init__()
        self.config = config
        self.vocab_size, self.num_hidden_layers = config.vocab_size, config.num_hidden_layers
        # 词嵌入表：把 token id 映射为 hidden 维向量，形状 [vocab_size, hidden_size]
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.dropout = nn.Dropout(config.dropout)         # 嵌入后的 dropout
        # 堆叠 num_hidden_layers 个 Transformer 块（如 8 层）
        self.layers = nn.ModuleList([MiniMindBlock(l, config) for l in range(self.num_hidden_layers)])
        # 最终归一化：在所有层之后、接 LM head 之前做一次 RMSNorm
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        # 预计算 RoPE 旋转位置编码表（cos/sin 各一份）
        #   freqs_cos/freqs_sin 形状 [max_position_embeddings, head_dim]，例如 [32768, 96]
        #   register_buffer(persistent=False)：作为 buffer 注册，不参与 state_dict 保存，
        #   不随模型权重持久化（因为它能由 config 重新计算出来）
        freqs_cos, freqs_sin = precompute_freqs_cis(dim=config.head_dim, end=config.max_position_embeddings, rope_base=config.rope_theta, rope_scaling=config.rope_scaling)
        self.register_buffer("freqs_cos", freqs_cos, persistent=False)
        self.register_buffer("freqs_sin", freqs_sin, persistent=False)

    def forward(self, input_ids, attention_mask=None, past_key_values=None, use_cache=False, **kwargs):
        batch_size, seq_length = input_ids.shape
        # 兼容 transformers 的 DynamicCache 对象（它有 .layers 属性）：
        # 本实现只认「按层排列的元组列表」，遇到 DynamicCache 就丢弃，走无 cache 路径
        if hasattr(past_key_values, 'layers'): past_key_values = None
        # past_key_values 为 None 时，展开成每层一个 None 的占位列表（对齐 self.layers 长度）
        past_key_values = past_key_values or [None] * len(self.layers)
        # start_pos：增量解码时「历史 KV 已覆盖到哪个位置」。
        #   past_key_values[0][0] = 第 0 层的 K，形状 [B, past_len, kv_heads, head_dim]，
        #   取第 1 维（序列维）长度即已生成 token 数；无 cache 时为 0。
        start_pos = past_key_values[0][0].shape[1] if past_key_values[0] is not None else 0
        # 1. 词嵌入 + dropout：input_ids [B,S] -> hidden_states [B,S,768]
        hidden_states = self.dropout(self.embed_tokens(input_ids))
        # 2. 容错：若 RoPE 表在 meta-device 初始化时丢失（transformers>=5.x 会置零），
        #    按 config 重新计算并搬到当前设备上
        if self.freqs_cos[0, 0] == 0:
            freqs_cos, freqs_sin = precompute_freqs_cis(dim=self.config.head_dim, end=self.config.max_position_embeddings, rope_base=self.config.rope_theta, rope_scaling=self.config.rope_scaling)
            self.freqs_cos, self.freqs_sin = freqs_cos.to(hidden_states.device), freqs_sin.to(hidden_states.device)
        # 3. 按当前位置窗口切出本轮的 RoPE 编码：
        #    [start_pos : start_pos + seq_length]，训练时覆盖 [0, S]，增量解码时只取新 token 的位置
        position_embeddings = (self.freqs_cos[start_pos:start_pos + seq_length], self.freqs_sin[start_pos:start_pos + seq_length])
        # 4. 逐层前向：把 hidden_states 依次穿过 num_hidden_layers 个块，
        #    每层传入各自的 past_key_value 并收集返回的 present（新 KV cache）
        presents = []
        for layer, past_key_value in zip(self.layers, past_key_values):
            hidden_states, present = layer(
                hidden_states,
                position_embeddings,
                past_key_value=past_key_value,
                use_cache=use_cache,
                attention_mask=attention_mask
            )
            presents.append(present)
        # 5. 最终 RMSNorm：把最后一层输出再归一化一次，交给 LM head
        hidden_states = self.norm(hidden_states)
        # 6. 汇总所有 MoE 层的负载均衡辅助损失（aux_loss）。
        #    非 MoE 层没有 aux_loss 属性，被 isinstance 过滤掉；
        #    sum(..., 起始标量 0) 保证至少返回一个标量张量（全非 MoE 时为 0）。
        aux_loss = sum([l.mlp.aux_loss for l in self.layers if isinstance(l.mlp, MOEFeedForward)], hidden_states.new_zeros(1).squeeze())
        return hidden_states, presents, aux_loss

class MiniMindForCausalLM(PreTrainedModel, GenerationMixin):
    """
    因果语言模型（Causal LM）顶层封装：主干模型 MiniMindModel + LM head，构成完整的
    「next-token 预测」模型。继承 transformers 的 PreTrainedModel 与 GenerationMixin，
    从而获得 save_pretrained / from_pretrained / generate 等标准能力。

    结构与数据流（默认配置：hidden=768、vocab=6400）：

        input_ids [B, S]
          --MiniMindModel--> hidden_states [B, S, 768]
          --lm_head (Linear 768->6400)--> logits [B, S, 6400]   （每个位置对词表 6400 个 token 的打分）

        logits[b, s, v] = 位置 s 处「下一个 token 是词表第 v 个」的未归一化分数，
        取 softmax 即得到概率分布，据此采样/取最大得到预测 token。

    权重共享（weight tying）：
        当 tie_word_embeddings=True（默认）时，让 lm_head.weight 与 embed_tokens.weight
        指向同一个张量。因为「把隐藏向量映射回词表」和「把 token 映射成向量」本质互为逆运算，
        共享可省下 vocab×hidden（6400×768≈490 万）个参数，还能让输入/输出空间对齐、提升泛化。

    LM head 投影的具体矩阵举例（B=1, S=3, hidden=768, vocab=6400）：

        hidden_states [1, 3, 768]  @  lm_head.weight^T [768, 6400]  =  logits [1, 3, 6400]
        （lm_head 是 nn.Linear(768, 6400)，内部就是 x @ W^T，W 形状 [6400, 768]）
        例如第 2 个位置的 logits[0, 1, :] 是一个 6400 维向量，
        argmax 取到最大分量下标即模型预测的下一个 token id。

    训练时 next-token 预测的标签对齐（loss 计算见 forward）：
        labels 是「真实答案」序列，logits 的每个位置要预测「它后面一个 token」，
        所以需错开一位对齐：logits 去尾、labels 去头。
    """
    config_class = MiniMindConfig
    # 声明权重共享关系：lm_head.weight 与 model.embed_tokens.weight 绑定（save_pretrained 会正确处理）
    _tied_weights_keys = {"lm_head.weight": "model.embed_tokens.weight"}
    def __init__(self, config: MiniMindConfig = None):
        # 未传 config 时用默认 MiniMindConfig()，再调用父类 PreTrainedModel.__init__ 完成注册
        self.config = config or MiniMindConfig()
        super().__init__(self.config)
        # 主干：词嵌入 + N 层 Transformer 块 + 最终 norm
        self.model = MiniMindModel(self.config)
        # LM head：把 hidden 维线性映射回词表 vocab 维，得到每个 token 的 logits
        self.lm_head = nn.Linear(self.config.hidden_size, self.config.vocab_size, bias=False)
        # 权重共享：把 embed_tokens.weight 直接指向 lm_head.weight（同一份参数）
        if self.config.tie_word_embeddings: self.model.embed_tokens.weight = self.lm_head.weight
        # transformers 的初始化钩子：初始化权重并做权重共享等收尾工作
        self.post_init()

    def forward(self, input_ids, attention_mask=None, past_key_values=None, use_cache=False, logits_to_keep=0, labels=None, **kwargs):
        # 1. 主干前向：得到最后一层隐藏向量 hidden_states [B, S, 768] + KV cache + MoE 辅助损失
        hidden_states, past_key_values, aux_loss = self.model(input_ids, attention_mask, past_key_values, use_cache, **kwargs)
        # 2. 只保留末尾 logits_to_keep 个位置的 logits（训练时省显存）。
        #    logits_to_keep=0 时 slice(0, None) = 全保留；>0 时只留最后 k 个位置（如 slice(-k, None)）。
        #    生成时只关心最后一个位置，训练时通常全保留（传 0）。
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        # 3. LM head 投影：hidden_states [B,S,768] -> logits [B,S,6400]
        logits = self.lm_head(hidden_states[:, slice_indices, :])
        loss = None
        if labels is not None:
            # 4. 对齐「预测-目标」：next-token 预测需错开一位。
            #    x = logits[..., :-1, :]  去掉最后一个位置的预测（它没有对应的下一个 token）
            #    y = labels[..., 1:]      去掉第一个位置的标签（它是「已知输入」，不需预测）
            #    对齐后：x 的第 i 个位置预测 y 的第 i 个 token，即「用位置 i 预测位置 i+1」。
            #    举例（S=4，token 序列 [我, 爱, 学习, 语言]，labels=[我, 爱, 学习, 语言]）：
            #        x:  logits 位置 0,1,2  (分别预测 爱、学习、语言)
            #        y:  labels 位置 1,2,3  (对应 爱、学习、语言)
            x, y = logits[..., :-1, :].contiguous(), labels[..., 1:].contiguous()
            # 5. 交叉熵损失：把 [B,S-1,6400] 展平成 [(B*(S-1)), 6400]，
            #    逐 token 计算分类交叉熵；ignore_index=-100 表示忽略 padding/无需预测的位置。
            loss = F.cross_entropy(x.view(-1, x.size(-1)), y.view(-1), ignore_index=-100)
        # 6. 打包成 transformers 风格的输出对象（含 loss、MoE 辅助损失、logits、KV cache、隐藏向量）
        return MoeCausalLMOutputWithPast(loss=loss, aux_loss=aux_loss, logits=logits, past_key_values=past_key_values, hidden_states=hidden_states)
    
    # https://github.com/jingyaogong/minimind/discussions/611
    @torch.inference_mode()   # 推理模式：关闭 autograd 与梯度图，省显存、提速
    def generate(self, inputs=None, attention_mask=None, max_new_tokens=8192, temperature=0.85, top_p=0.85, top_k=50, eos_token_id=2, streamer=None, use_cache=True, num_return_sequences=1, do_sample=True, repetition_penalty=1.0, **kwargs):
        # 自回归文本生成：逐 token 采样，每轮只吃「最后一个新 token」，靠 KV cache 复用历史计算。

        # 1. 准备输入：input_ids 兼容 kwargs 里的 input_ids 或 inputs 参数；
        #    .repeat(num_return_sequences, 1) 按需复制 num_return_sequences 份（如 1 份则不变）
        input_ids = kwargs.pop("input_ids", inputs).repeat(num_return_sequences, 1)
        attention_mask = attention_mask.repeat(num_return_sequences, 1) if attention_mask is not None else None
        # 取出外部传入的历史 KV cache（若有，用于续写）
        past_key_values = kwargs.pop("past_key_values", None)
        # finished：逐样本标记「是否已生成 eos 结束符」，形状 [num_return_sequences]
        finished = torch.zeros(input_ids.shape[0], dtype=torch.bool, device=input_ids.device)
        if streamer: streamer.put(input_ids.cpu())   # 先把初始输入推给 streamer（流式输出）
        # 2. 自回归循环：最多生成 max_new_tokens 个新 token
        for _ in range(max_new_tokens):
            # 已生成的序列长度（有 cache 时取 KV 序列维长度；否则 0）
            past_len = past_key_values[0][0].shape[1] if past_key_values else 0
            # 只喂「最后新增的 token」：input_ids[:, past_len:] 就是新 token 那一段
            outputs = self.forward(input_ids[:, past_len:], attention_mask, past_key_values, use_cache=use_cache, **kwargs)
            # attention_mask 追加一位 1（新 token 是真实 token，非 padding）
            attention_mask = torch.cat([attention_mask, attention_mask.new_ones(attention_mask.shape[0], 1)], -1) if attention_mask is not None else None
            # 3. 取最后一个位置的 logits 并除以温度：temperature 控制采样随机性
            #    temperature 越大越随机（趋向均匀），越小越确定（趋向 argmax）
            logits = outputs.logits[:, -1, :] / temperature
            # 4. 重复惩罚：对「已经出现过的 token」降权，抑制重复输出
            #    score>0 的除以 penalty（压小），score<0 的乘以 penalty（压更负），二者都降低其被选中概率
            if repetition_penalty != 1.0:
                for i in range(input_ids.shape[0]):
                    seen = torch.unique(input_ids[i]); score = logits[i, seen]; logits[i, seen] = torch.where(score > 0, score / repetition_penalty, score * repetition_penalty)
            # 5. top-k 过滤：只保留分数最高的 top_k 个候选，其余置 -inf（被 softmax 后概率为 0）
            if top_k > 0:
                logits[logits < torch.topk(logits, top_k)[0][..., -1, None]] = -float('inf')
            # 6. top-p（nucleus）过滤：按概率从高到低累加，保留「累加到刚好超过 top_p」的那批候选
            #    例：softmax 后概率 [0.5, 0.3, 0.15, 0.05]，top_p=0.85 时保留前 3 个（0.5+0.3+0.15=0.95≥0.85）
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                mask = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1) > top_p
                mask[..., 1:], mask[..., 0] = mask[..., :-1].clone(), 0
                logits[mask.scatter(1, sorted_indices, mask)] = -float('inf')
            # 7. 采样/贪心：do_sample=True 用多项式分布采样；否则取 argmax（贪心）
            next_token = torch.multinomial(torch.softmax(logits, dim=-1), num_samples=1) if do_sample else torch.argmax(logits, dim=-1, keepdim=True)
            # 8. 已结束的样本强制填 eos：finished=True 的样本不再生成新内容，只反复填 eos_token_id
            if eos_token_id is not None: next_token = torch.where(finished.unsqueeze(-1), next_token.new_full((next_token.shape[0], 1), eos_token_id), next_token)
            # 9. 拼接到 input_ids，并更新 KV cache 供下一轮使用
            input_ids = torch.cat([input_ids, next_token], dim=-1)
            past_key_values = outputs.past_key_values if use_cache else None
            if streamer: streamer.put(next_token.cpu())   # 流式推送新 token
            # 10. 更新 finished：生成了 eos 的样本标记为结束；全部结束后提前 break
            if eos_token_id is not None:
                finished |= next_token.squeeze(-1).eq(eos_token_id)
                if finished.all(): break
        if streamer: streamer.end()
        # 需要 KV cache 时返回字典，否则只返回生成的 token id 序列
        if kwargs.get("return_kv"): return {'generated_ids': input_ids, 'past_kv': past_key_values}
        return input_ids