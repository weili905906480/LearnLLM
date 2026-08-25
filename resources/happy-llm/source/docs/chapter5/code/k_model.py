import math
import inspect
from dataclasses import dataclass
from typing import Any, Optional, Tuple
import torch
import torch.nn.functional as F
from torch import nn

from transformers import PreTrainedModel, AutoTokenizer
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers import PretrainedConfig


class ModelConfig(PretrainedConfig):
    model_type = "Tiny-K"
    def __init__(
            self,
            dim: int = 768,
            n_layers: int = 12,
            n_heads: int = 16,
            n_kv_heads: int = 8,
            vocab_size: int = 6144,
            hidden_dim: int = None,
            multiple_of: int = 64,
            norm_eps: float = 1e-5,
            max_seq_len: int = 512,
            dropout: float = 0.0,
            flash_attn: bool = True,
            pad_token_id: int = 0,
            **kwargs,
    ):
        self.dim = dim
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.multiple_of = multiple_of
        self.norm_eps = norm_eps
        self.max_seq_len = max_seq_len
        self.dropout = dropout
        self.flash_attn = flash_attn
        self.pad_token_id = pad_token_id
        super().__init__(**kwargs)

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float):
        super().__init__()
        # eps是为了防止除以0的情况
        self.eps = eps
        # weight是一个可学习的参数，全部初始化为1
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        # 计算RMSNorm的核心部分
        # x.pow(2).mean(-1, keepdim=True)计算了输入x的平方的均值
        # torch.rsqrt是平方根的倒数，这样就得到了RMSNorm的分母部分，再加上eps防止分母为0
        # 最后乘以x，得到RMSNorm的结果
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        # forward函数是模型的前向传播
        # 首先将输入x转为float类型，然后进行RMSNorm，最后再转回原来的数据类型
        # 最后乘以weight，这是RMSNorm的一个可学习的缩放因子
        output = self._norm(x.float()).type_as(x)
        return output * self.weight

# 获得旋转嵌入的实部和虚部
# 注意：此处的dim应为 dim//n_head，因为我们是对每个head进行旋转嵌入
def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    # torch.arange(0, dim, 2)[: (dim // 2)].float()生成了一个从0开始，步长为2的序列，长度为dim的一半
    # 然后每个元素除以dim，再取theta的倒数，得到频率
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    # 生成一个从0到end的序列，长度为end
    t = torch.arange(end, device=freqs.device)
    # 计算外积，得到一个二维矩阵，每一行是t的元素乘以freqs的元素
    freqs = torch.outer(t, freqs).float()
    # 计算频率的余弦值，得到实部
    freqs_cos = torch.cos(freqs)
    # 计算频率的正弦值，得到虚部
    freqs_sin = torch.sin(freqs)
    return freqs_cos, freqs_sin

# 此函数的作用是将freqs_cis调整为与x的形状相同，以便能够与x进行广播操作
def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    # 获取x的维度数
    ndim = x.ndim
    # 断言，确保1在x的维度范围内
    assert 0 <= 1 < ndim
    # 断言，确保freqs_cis的形状与x的第二维和最后一维相同
    assert freqs_cis.shape == (x.shape[1], x.shape[-1])
    # 构造一个新的形状，除了第二维和最后一维，其他维度都为1，这样做是为了能够将freqs_cis与x进行广播操作
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    # 将freqs_cis调整为新的形状，并返回
    return freqs_cis.view(shape)

def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cos: torch.Tensor,
    freqs_sin: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:

    # 将查询和键张量转换为浮点数，并重塑形状以分离实部和虚部
    xq_r, xq_i = xq.float().reshape(xq.shape[:-1] + (-1, 2)).unbind(-1)
    xk_r, xk_i = xk.float().reshape(xk.shape[:-1] + (-1, 2)).unbind(-1)

    # 重新塑形频率张量以进行广播
    freqs_cos = reshape_for_broadcast(freqs_cos, xq_r)
    freqs_sin = reshape_for_broadcast(freqs_sin, xq_r)

    # 应用旋转，分别计算旋转后的实部和虚部
    xq_out_r = xq_r * freqs_cos - xq_i * freqs_sin
    xq_out_i = xq_r * freqs_sin + xq_i * freqs_cos
    xk_out_r = xk_r * freqs_cos - xk_i * freqs_sin
    xk_out_i = xk_r * freqs_sin + xk_i * freqs_cos

    # 将最后两个维度合并，并还原为原始张量的形状
    xq_out = torch.stack([xq_out_r, xq_out_i], dim=-1).flatten(3)
    xk_out = torch.stack([xk_out_r, xk_out_i], dim=-1).flatten(3)

    return xq_out.type_as(xq), xk_out.type_as(xk)

def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    # 获取输入张量的形状：批量大小、序列长度、键/值对头的数量、每个头的维度大小
    bs, slen, n_kv_heads, head_dim = x.shape
    
    # 如果重复次数为1，则不需要重复，直接返回原始张量
    if n_rep == 1:
        return x
    
    # 对张量进行扩展和重塑操作以重复键值对
    return (
        x[:, :, :, None, :]  # 在第四个维度（头的维度前）添加一个新的维度
        .expand(bs, slen, n_kv_heads, n_rep, head_dim)  # 将新添加的维度扩展到n_rep大小，实现重复的效果
        .reshape(bs, slen, n_kv_heads * n_rep, head_dim)  # 重新塑形，合并键/值对头的数量和重复次数的维度
    )

class Attention(nn.Module):
    def __init__(self, args: ModelConfig):
        super().__init__()
        # 根据是否指定n_kv_heads，确定用于键（key）和值（value）的头的数量。
        self.n_kv_heads = args.n_heads if args.n_kv_heads is None else args.n_kv_heads
        # 确保总头数可以被键值头数整除。
        assert args.n_heads % self.n_kv_heads == 0

        # 模型并行处理大小，默认为1。
        model_parallel_size = 1
        # 本地计算头数，等于总头数除以模型并行处理大小。
        self.n_local_heads = args.n_heads // model_parallel_size
        # 本地键值头数，等于键值头数除以模型并行处理大小。
        self.n_local_kv_heads = self.n_kv_heads // model_parallel_size
        # 重复次数，用于扩展键和值的尺寸。
        self.n_rep = self.n_local_heads // self.n_local_kv_heads
        # 每个头的维度，等于模型维度除以头的总数。
        self.head_dim = args.dim // args.n_heads

        # 定义权重矩阵。
        self.wq = nn.Linear(args.dim, args.n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(args.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(args.dim, self.n_kv_heads * self.head_dim, bias=False)
        # 输出权重矩阵。
        self.wo = nn.Linear(args.n_heads * self.head_dim, args.dim, bias=False)

        # 定义dropout。
        self.attn_dropout = nn.Dropout(args.dropout)
        self.resid_dropout = nn.Dropout(args.dropout)
        # 保存dropout概率。
        self.dropout = args.dropout

        # 检查是否使用Flash Attention（需要PyTorch >= 2.0）。
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
        if not self.flash:
            # 若不支持Flash Attention，则使用手动实现的注意力机制，并设置mask。
            print("WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0")
            # 创建一个上三角矩阵，用于遮蔽未来信息。
            mask = torch.full((1, 1, args.max_seq_len, args.max_seq_len), float("-inf"))
            mask = torch.triu(mask, diagonal=1)
            # 注册为模型的缓冲区
            self.register_buffer("mask", mask)

    def forward(self, x: torch.Tensor, freqs_cos: torch.Tensor, freqs_sin: torch.Tensor, attention_mask: Optional[torch.Tensor] = None):
        # 获取批次大小和序列长度，[batch_size, seq_len, dim]
        bsz, seqlen, _ = x.shape

        # 计算查询（Q）、键（K）、值（V）。
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)
        # 调整形状以适应头的维度。
        xq = xq.view(bsz, seqlen, self.n_local_heads, self.head_dim)
        xk = xk.view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)
        xv = xv.view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)

        # 应用旋转位置嵌入（RoPE）。
        xq, xk = apply_rotary_emb(xq, xk, freqs_cos, freqs_sin)

        # 对键和值进行扩展以适应重复次数。
        xk = repeat_kv(xk, self.n_rep)
        xv = repeat_kv(xv, self.n_rep)

        # 将头作为批次维度处理。
        xq = xq.transpose(1, 2)
        xk = xk.transpose(1, 2)
        xv = xv.transpose(1, 2)
        key_padding_mask = None
        if attention_mask is not None:
            key_padding_mask = attention_mask[:, None, None, :].to(dtype=torch.bool)

        # 根据是否支持Flash Attention，选择实现方式。
        if self.flash:
            # 使用Flash Attention。
            if key_padding_mask is not None:
                causal_mask = torch.ones((seqlen, seqlen), dtype=torch.bool, device=x.device).tril()
                full_attn_mask = causal_mask[None, None, :, :] & key_padding_mask
                output = torch.nn.functional.scaled_dot_product_attention(
                    xq,
                    xk,
                    xv,
                    attn_mask=full_attn_mask,
                    dropout_p=self.dropout if self.training else 0.0,
                    is_causal=False,
                )
            else:
                output = torch.nn.functional.scaled_dot_product_attention(
                    xq,
                    xk,
                    xv,
                    attn_mask=None,
                    dropout_p=self.dropout if self.training else 0.0,
                    is_causal=True,
                )
        else:
            # 使用手动实现的注意力机制。
            scores = torch.matmul(xq, xk.transpose(2, 3)) / math.sqrt(self.head_dim)
            assert hasattr(self, 'mask')
            scores = scores + self.mask[:, :, :seqlen, :seqlen]
            if key_padding_mask is not None:
                scores = scores.masked_fill(~key_padding_mask, float("-inf"))
            scores = F.softmax(scores.float(), dim=-1).type_as(xq)
            scores = self.attn_dropout(scores)
            output = torch.matmul(scores, xv)

        # 恢复时间维度并合并头。
        output = output.transpose(1, 2).contiguous().view(bsz, seqlen, -1)

        # 最终投影回残差流。
        output = self.wo(output)
        output = self.resid_dropout(output)
        return output

class MLP(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, multiple_of: int, dropout: float):
        super().__init__()
        # 如果没有指定隐藏层的维度，我们将其设置为输入维度的4倍
        # 然后将其减少到2/3，最后确保它是multiple_of的倍数
        if hidden_dim is None:
            hidden_dim = 4 * dim
            hidden_dim = int(2 * hidden_dim / 3)
            hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)
        # 定义第一层线性变换，从输入维度到隐藏维度
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        # 定义第二层线性变换，从隐藏维度到输入维度
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        # 定义第三层线性变换，从输入维度到隐藏维度
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)
        # 定义dropout层，用于防止过拟合
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # 前向传播函数
        # 首先，输入x通过第一层线性变换和SILU激活函数
        # 然后，结果乘以输入x通过第三层线性变换的结果
        # 最后，通过第二层线性变换和dropout层
        return self.dropout(self.w2(F.silu(self.w1(x)) * self.w3(x)))
    

class DecoderLayer(nn.Module):
    def __init__(self, layer_id: int, args: ModelConfig):
        super().__init__()
        # 定义多头注意力的头数
        self.n_heads = args.n_heads
        # 定义输入维度
        self.dim = args.dim
        # 定义每个头的维度，等于输入维度除以头数
        self.head_dim = args.dim // args.n_heads
        # 定义LLaMA2Attention对象，用于进行多头注意力计算
        self.attention = Attention(args)
        # 定义LLaMAMLP对象，用于进行前馈神经网络计算
        self.feed_forward = MLP(
            dim=args.dim,
            hidden_dim=args.hidden_dim,
            multiple_of=args.multiple_of,
            dropout=args.dropout,
        )
        # 定义层的ID
        self.layer_id = layer_id
        # 定义注意力计算的归一化层
        self.attention_norm = RMSNorm(args.dim, eps=args.norm_eps)
        # 定义前馈神经网络计算的归一化层
        self.ffn_norm = RMSNorm(args.dim, eps=args.norm_eps)

    def forward(self, x, freqs_cos, freqs_sin, attention_mask: Optional[torch.Tensor] = None):
        # 前向传播函数
        # 首先，输入x经过注意力归一化层，然后进行注意力计算，结果与输入x相加得到h
        # 然后，h经过前馈神经网络归一化层，然后进行前馈神经网络计算，结果与h相加得到输出
        h = x + self.attention.forward(self.attention_norm(x), freqs_cos, freqs_sin, attention_mask=attention_mask)
        out = h + self.feed_forward.forward(self.ffn_norm(h))
        return out

class Transformer(PreTrainedModel):
    config_class = ModelConfig  # 配置类
    last_loss: Optional[torch.Tensor] # 记录最后一次计算的损失

    def __init__(self, args: ModelConfig = None):
        super().__init__(args)
        # 初始化模型参数
        self.args = args
        # 词汇表大小
        self.vocab_size = args.vocab_size
        # 层数
        self.n_layers = args.n_layers

        # 词嵌入层
        self.tok_embeddings = nn.Embedding(args.vocab_size, args.dim)
        # Dropout层
        self.dropout = nn.Dropout(args.dropout)
        # Decoder层
        self.layers = torch.nn.ModuleList()
        for layer_id in range(args.n_layers):
            self.layers.append(DecoderLayer(layer_id, args))
        # 归一化层
        self.norm = RMSNorm(args.dim, eps=args.norm_eps)
        # 输出层
        self.output = nn.Linear(args.dim, args.vocab_size, bias=False)

        # 将词嵌入层的权重与输出层的权重共享
        self.tok_embeddings.weight = self.output.weight 

        # 预计算相对位置嵌入的频率
        freqs_cos, freqs_sin = precompute_freqs_cis(self.args.dim // self.args.n_heads, self.args.max_seq_len)
        self.register_buffer("freqs_cos", freqs_cos, persistent=False)
        self.register_buffer("freqs_sin", freqs_sin, persistent=False)

        # 初始化所有权重
        self.apply(self._init_weights)
        # 对残差投影进行特殊的缩放初始化
        for pn, p in self.named_parameters():
            if pn.endswith('w3.weight') or pn.endswith('wo.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * args.n_layers))

        # 初始化最后一次前向传播的损失属性
        self.last_loss = None
        self.OUT = CausalLMOutputWithPast()  # 输出容器
        self._no_split_modules = [name for name, _ in self.named_modules()]  # 不分割的模块列表

    def _init_weights(self, module):
        # 初始化权重的函数
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def _prepare_attention_mask(self, attention_mask: Optional[torch.Tensor], tokens: torch.Tensor) -> Optional[torch.Tensor]:
        """
        把各种格式的 attention_mask 统一规范成模型内部需要的标准格式：
            - 形状：2D (bsz, seq_len)
            - 类型：bool（True=真实 token，False=padding）
            - 设备：与 tokens 一致
        如果传入 None 则原样返回 None（后续 Attention 会走纯因果注意力分支）。

        参数：
            attention_mask: 可能为 None，或 2D/3D/4D，或 int/float/bool 类型的 mask
            tokens:         input_ids，形状 (bsz, seq_len)，用作设备与形状的参照物

        返回：
            规范化后的 (bsz, seq_len) bool 张量，或 None
        """
        # 无 mask 直接返回 None，不做处理
        if attention_mask is None:
            return None
        # 降维：把 HF 风格的高维 mask 压成 2D
        #   4D (bsz, 1, 1, seq_len) -> 取 [:,0,0,:] -> (bsz, seq_len)
        #   3D (bsz, 1, seq_len)    -> 取 [:,0,:]   -> (bsz, seq_len)
        if attention_mask.dim() == 4:
            attention_mask = attention_mask[:, 0, 0, :]
        elif attention_mask.dim() == 3:
            attention_mask = attention_mask[:, 0, :]
        # 对齐设备，否则后面跨设备做 & / masked_fill 会报错
        attention_mask = attention_mask.to(tokens.device)
        # 统一 dtype 为 bool：非零 -> True(真实)，0 -> False(padding)
        # 例：[[1,1,0]] (long)  ->  [[True,True,False]] (bool)
        if attention_mask.dtype != torch.bool:
            attention_mask = attention_mask > 0
        # 形状校验：必须和 input_ids 完全一致，都是 (bsz, seq_len)，否则尽早暴露错误
        if attention_mask.shape != tokens.shape:
            raise ValueError(f"attention_mask shape {attention_mask.shape} must match input_ids shape {tokens.shape}")
        return attention_mask

    def _left_pad_by_attention_mask(
            self,
            idx: torch.Tensor,
            attention_mask: Optional[torch.Tensor],
            pad_token_id: int
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        把带 padding 的 batch 重新整理成「左填充」的紧凑格式：
            - 真实 token 右对齐（内容贴着末尾），padding 全部在左边
            - 去掉每条序列尾部的 padding，压缩到最长的「有效长度」
        用于生成（generate/generate_super）前的预处理：自回归要基于末尾预测下一个 token，
        左填充能保证末位一定是真实 token，而不是 padding。

        参数：
            idx:            (bsz, seq_len) 的 token 张量
            attention_mask: (bsz, seq_len) 的 bool 张量，True=真实 / False=padding
            pad_token_id:   填充符 id

        返回：
            (packed_idx, packed_mask)：均为 (bsz, max_len)，max_len 为最长的有效长度
        """
        # 无 mask 或 全部是真实 token（无 padding）时无需处理，直接返回
        if attention_mask is None or attention_mask.all():
            return idx, attention_mask

        # 每条序列的真实长度：True=1/False=0，按序列维求和
        # 例：mask = [[T,T,T,F,F],[T,F,F,F,F]] -> lengths = [3, 1]
        bsz = idx.size(0)
        lengths = attention_mask.long().sum(dim=1)
        # 目标长度 = 最长的有效长度（去掉 padding 后的紧凑长度），至少为 1
        max_len = max(int(lengths.max().item()), 1)   # 例：max(3,1) = 3
        # 先全填 padding：packed_idx 全 pad_token_id，packed_mask 全 False
        packed_idx = idx.new_full((bsz, max_len), pad_token_id)
        packed_mask = attention_mask.new_zeros((bsz, max_len), dtype=torch.bool)

        for row in range(bsz):
            valid_len = int(lengths[row].item())
            # 整条都是 padding 的样本，跳过（保持全 pad、全 False）
            if valid_len <= 0:
                continue
            # 布尔索引：取出该行所有 True 位置的真实 token，按原顺序，顺便去掉 padding
            # 例：idx[0] = [a,b,c,PAD,PAD], mask[0]=[T,T,T,F,F] -> valid_tokens=[a,b,c]
            valid_tokens = idx[row][attention_mask[row]]
            # 左填充：真实 token 放到该行「最右端」，左边 max_len-valid_len 个位置保持 pad
            # 例：max_len=3, valid_len=3 -> 占据 [0,1,2]（无左填充）
            #     max_len=3, valid_len=1 -> 占据 [2]，索引 0,1 保持 pad
            packed_idx[row, max_len - valid_len:] = valid_tokens
            # 同步把对应位置在 mask 里标记为 True（真实），左边保持 False（padding）
            packed_mask[row, max_len - valid_len:] = True

        # 例（整体效果）：
        # 输入 idx = [[a,b,c,PAD,PAD],[d,PAD,PAD,PAD,PAD]]
        #     mask= [[T,T,T,F,F],[T,F,F,F,F]]，pad_token_id=0
        # 输出 packed_idx  = [[a,b,c],[0,0,d]]
        #      packed_mask = [[T,T,T],[F,F,T]]
        return packed_idx, packed_mask
    
    def forward(self, tokens: torch.Tensor, targets: Optional[torch.Tensor] = None, **kwargs) -> torch.Tensor:
        """
        模型前向传播。支持训练（给 targets 算 loss）和推理（只取末位 logits）两种模式。

        参数：
            - tokens:  Optional[torch.Tensor], 输入 token 张量，形状 (bsz, seq_len)
            - targets: Optional[torch.Tensor], 目标 token 张量（训练标签），形状同 tokens
            - kwargs:  其他关键字参数，可能包含 input_ids / labels / attention_mask

        返回：
            - self.OUT: CausalLMOutputWithPast，含 logits 和 last_loss 两个字段

        矩阵变化总览（训练分支，bsz=2, seq_len=3, dim=768, vocab=6144）：
            tokens  (2, 3)     --tok_embeddings-->  h  (2, 3, 768)
            h 经过 12 个 DecoderLayer 后形状不变     (2, 3, 768)
            h 经 self.norm 后仍为                      (2, 3, 768)
            logits = self.output(h)                ->  (2, 3, 6144)
        """

        # 兼容 HF 调用：若通过 input_ids / labels 传入，则覆盖位置参数
        if 'input_ids' in kwargs:
            tokens = kwargs['input_ids']
        if 'labels' in kwargs:
            targets = kwargs['labels']
        # 规范化 attention_mask -> (bsz, seq_len) bool 或 None
        attention_mask = self._prepare_attention_mask(kwargs.get('attention_mask'), tokens)

        # 前向传播函数
        _bsz, seqlen = tokens.shape          # 例：(2, 3)
        # 通过词嵌入层和 Dropout 层
        # tokens (2,3) -> h (2,3,768)：每个 token id 映射成 dim 维向量
        h = self.tok_embeddings(tokens)
        h = self.dropout(h)
        # 获取相对位置嵌入（RoPE）的频率，只取当前序列长度对应的前 seqlen 行
        # 预计算时长度是 max_seq_len，这里截到实际 seqlen 以匹配当前序列
        freqs_cos = self.freqs_cos[:seqlen]   # (seqlen, head_dim//2)
        freqs_sin = self.freqs_sin[:seqlen]

        # 依次通过每个 DecoderLayer（自注意力 + FFN + 残差），形状始终 (bsz, seq_len, dim)
        for layer in self.layers:
            h = layer(h, freqs_cos, freqs_sin, attention_mask=attention_mask)
        # 最后一层 RMSNorm
        h = self.norm(h)                      # (2, 3, 768)

        if targets is not None:
            # ===== 训练分支：计算损失 =====
            # 所有位置都映射到词表，得到完整 logits
            logits = self.output(h)           # (2, 3, 6144)
            # 确定忽略的 target 索引（padding 位置不算 loss）
            ignore_index = self.args.pad_token_id if self.args.pad_token_id is not None else 0
            # 若 target 里有 -100（HF 约定），也一并忽略
            if torch.any(targets == -100):
                ignore_index = -100
            # 交叉熵：把 logits 和 targets 展平成 (bsz*seq_len, vocab) / (bsz*seq_len)
            # reduction='none' 返回逐位置的 loss（不取平均），保存在 self.last_loss
            self.last_loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),   # (6, 6144)
                targets.view(-1),                   # (6,)
                ignore_index=ignore_index,
                reduction='none'
            )
        else:
            # ===== 推理分支：只取「最后一个有效位置」的 logits =====
            if attention_mask is None:
                # 没有 mask：直接取每条序列最后一个位置的 hidden，再映射到词表
                # h[:, [-1], :] -> (bsz, 1, dim) -> logits (bsz, 1, vocab)
                logits = self.output(h[:, [-1], :])
            else:
                # 有 mask：先把所有位置都映射成 logits，再挑出每条序列最后一个「真实」token 的位置
                # （左填充下，真实 token 右对齐，最后一个 True 的位置就是真实内容的末尾）
                full_logits = self.output(h)        # (bsz, seq_len, vocab)
                # 每条序列的有效长度（True 的个数），clamp(min=1) 防止全 pad 时下标变 -1
                last_token_pos = attention_mask.long().sum(dim=1).clamp(min=1) - 1   # (bsz,)
                # 按 (行, 末位位置) 取出对应 logits，再 unsqueeze 回 (bsz, 1, vocab)
                logits = full_logits[torch.arange(_bsz, device=tokens.device), last_token_pos].unsqueeze(1)
            self.last_loss = None

        # 把结果装进 CausalLMOutputWithPast 并返回
        self.OUT.__setitem__('logits', logits)
        self.OUT.__setitem__('last_loss', self.last_loss)
        return self.OUT

    
    @torch.inference_mode()
    def generate(
            self,
            idx,
            stop_id=None,
            max_new_tokens=256,
            temperature=1.0,
            top_k=None,
            attention_mask: Optional[torch.Tensor] = None,
            pad_token_id: Optional[int] = None
    ):
        """
        给定输入序列 idx（形状为 (bsz, seq_len) 的长整型张量），通过多次生成新 token 来完成序列。
        在 model.eval() 模式下运行。效率较低的采样版本，没有使用键 k/v cache。

        参数：
            idx:             (bsz, seq_len) 的 prompt token 张量，如 [[a,b,c,PAD],[d,e,PAD,PAD]]
            stop_id:         停止符 token id，生成到它就结束该条序列
            max_new_tokens:  最多生成多少个新 token（默认 256）
            temperature:     采样温度；0.0 走贪婪解码，否则按温度缩放后采样
            top_k:           只从概率最高的 k 个 token 里采样；None 表示不限制
            attention_mask:  (bsz, seq_len) 或 3D/4D，标记真实 token(True) 与 padding(False)
            pad_token_id:    填充符 id

        返回：
            (bsz, num_generated) 的只含「新生成部分」的 token 张量。
        """
        # 兜底确定填充符 id
        if pad_token_id is None:
            pad_token_id = self.args.pad_token_id if self.args.pad_token_id is not None else 0
        # 1) 规范化 attention_mask：None/3D/4D/int 都统一成 (bsz, seq_len) 的 bool 张量
        attention_mask = self._prepare_attention_mask(attention_mask, idx)
        # 2) 左填充：真实内容右对齐、pad 在左，并压缩掉尾部 padding
        #    idx:  [[a,b,c,PAD],[d,e,PAD,PAD]]  ->  [[a,b,c],[PAD,d,e]]
        #    mask: [[T,T,T,F],[T,T,F,F]]        ->  [[T,T,T],[F,T,T]]
        idx, attention_mask = self._left_pad_by_attention_mask(idx, attention_mask, pad_token_id)

        # finished: (bsz,) 的 bool，标记每条序列是否已结束，初始全 False
        finished = torch.zeros(idx.size(0), dtype=torch.bool, device=idx.device)   # [False, False]
        # index 记录 prompt 长度（左填充后的 seq_len），最后用它切片只返回生成部分
        index = idx.shape[1]                                                        # 3

        # 主循环：每次迭代生成一个新 token
        for _ in range(max_new_tokens):
            # ---- 4a. 上下文超长则截断，只保留最后 max_seq_len 个 token ----
            idx_cond = idx if idx.size(1) <= self.args.max_seq_len else idx[:, -self.args.max_seq_len:]
            mask_cond = None
            if attention_mask is not None:
                mask_cond = attention_mask if attention_mask.size(1) <= self.args.max_seq_len else attention_mask[:, -self.args.max_seq_len:]

            # ---- 4b. 前向传播，只取最后一个位置的 logits ----
            # self(...) 输出 (bsz, seq_len, vocab_size)；[:, -1, :] 取末位 -> (bsz, vocab_size)
            # 左填充保证末位一定是真实 token，自回归只需预测「下一个」
            logits = self(idx_cond, attention_mask=mask_cond).logits
            logits = logits[:, -1, :] # 只保留最后一个时间步的输出

            # ---- 4c. 根据 temperature/top_k 选择下一个 token ----
            if temperature == 0.0:
                # 分支 A：贪婪解码，直接取每行最大 logits 的索引（argmax）
                # logits: [[2.0,0.5,0.1,1.0,0.3],
                #          [1.0,2.5,0.8,0.2,0.1]]  ->  idx_next: [[0],[1]]
                _, idx_next = torch.topk(logits, k=1, dim=-1)
            else:
                # 分支 B：温度采样
                # 缩放 logits 并应用 softmax
                logits = logits / temperature        # 温度>1 更平滑(随机)，<1 更尖锐(确定)
                if top_k is not None:
                    # top-k：把低于第 k 大的 logits 置为 -inf，softmax 后概率为 0
                    v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                    logits[logits < v[:, [-1]]] = -float('Inf')
                probs = F.softmax(logits, dim=-1)    # (bsz, vocab_size) 概率分布
                idx_next = torch.multinomial(probs, num_samples=1)   # 按概率采样 1 个 -> (bsz, 1)

            # ---- 4d. 更新结束状态 ----
            prev_finished = finished.clone()         # 保存本轮之前的结束状态
            if stop_id is not None:
                if prev_finished.any():
                    # 已结束的序列不再生成新 token，用 fill_token 占位保持形状整齐
                    fill_token = pad_token_id if pad_token_id is not None else stop_id
                    idx_next = torch.where(prev_finished[:, None], torch.full_like(idx_next, fill_token), idx_next)
                # 之前结束过 或 本轮生成 stop_id -> 该序列标记为结束
                finished = prev_finished | idx_next[:, 0].eq(stop_id)

            # ---- 4e. 将新 token 拼接到序列末尾，并同步扩展 mask ----
            # idx: [[a,b,c],[PAD,d,e]] -> [[a,b,c,3],[PAD,d,e,1]]
            idx = torch.cat((idx, idx_next), dim=1)
            if attention_mask is not None:
                next_mask = torch.ones((attention_mask.size(0), 1), dtype=attention_mask.dtype, device=attention_mask.device)
                if prev_finished.any():
                    # 上轮已结束的序列，本轮生成的是占位 pad，mask 记为 False
                    next_mask[prev_finished] = False
                # mask: [[T,T,T],[F,T,T]] -> [[T,T,T,T],[F,T,T,T]]
                attention_mask = torch.cat((attention_mask, next_mask), dim=1)

            # ---- 4f. 所有序列都结束则提前退出循环 ----
            if stop_id is not None and finished.all():
                break

        # 用 index 切片，只返回新生成的 token
        # 例：最终 idx = [[a,b,c,3,2,<eos>],[PAD,d,e,1,4,<eos>]]，index=3
        #     返回 idx[:,3:] = [[3,2,<eos>],[1,4,<eos>]]，形状 (bsz, num_generated)
        return idx[:, index:] # 只返回生成的token
    
    def _greedy_decode(self, logits: torch.Tensor) -> torch.Tensor:
        """
        贪婪解码：选择概率最大的token

        Args:
            logits: 模型输出的logits，形状为 (batch_size, vocab_size)

        Returns:
            选择的token索引，形状为 (batch_size, 1)
        """
        _, idx_next = torch.topk(logits, k=1, dim=-1)
        return idx_next

    def _random_sample(self, logits: torch.Tensor, temperature: float = 1.0, top_k: int = None) -> torch.Tensor:
        """
        随机采样：基于概率分布随机选择token

        Args:
            logits: 模型输出的logits，形状为 (batch_size, vocab_size)
            temperature: 温度参数，控制随机性
            top_k: 只考虑概率最高的k个token

        Returns:
            选择的token索引，形状为 (batch_size, 1)
        """
        # 缩放 logits
        logits = logits / temperature

        # 应用top-k过滤
        if top_k is not None:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            # 将不在 top-k 内的 logits 设为负无穷
            logits[logits < v[:, [-1]]] = -float('Inf')

        # 计算概率并采样
        probs = F.softmax(logits, dim=-1)
        idx_next = torch.multinomial(probs, num_samples=1)
        return idx_next

    def _beam_search(self, idx: torch.Tensor, max_new_tokens: int, num_beams: int,
                     temperature: float = 1.0, top_k: int = None, stop_id: int = None) -> torch.Tensor:
        """
        束搜索：维护多个候选序列，选择最优路径

        束搜索的核心思想：在每一步生成时，不是只选择一个最佳token，
        而是保留多个候选路径，最终选择累积概率最高的完整序列。

        Args:
            idx: 输入序列，形状为 (batch_size, seq_len)
            max_new_tokens: 最大生成token数量
            num_beams: 束宽度，表示保留的候选路径数量
            temperature: 温度参数，控制分布的平滑程度
            top_k: top-k过滤参数，限制候选token范围
            stop_id: 停止生成的token ID，遇到则停止

        Returns:
            生成的token序列，形状为 (batch_size, generated_length)
            只返回新生成的部分，不包含原始输入序列
        """
        # 获取输入序列的基本信息
        batch_size = idx.shape[0]  # 批次大小，通常为1
        seq_len = idx.shape[1]     # 输入序列长度

        # 初始化束：创建 num_beams 个候选序列
        beams = [idx.clone() for _ in range(num_beams)]
        # 初始化每个候选序列的累积对数概率分数
        beam_scores = torch.zeros(num_beams, device=idx.device)
        # 第一个候选是原始输入序列，分数为0
        beam_scores[0] = 0.0
        # 其他候选初始分数设为负无穷，表示尚未生成
        beam_scores[1:] = float('-inf')

        # 主循环：逐步生成新的token，最多生成 max_new_tokens 个
        for step in range(max_new_tokens):
            # 每轮迭代收集新的候选序列和分数
            new_beams = []   # 新的候选序列列表
            new_scores = []  # 对应的分数列表

            # 遍历当前的所有候选序列
            for beam_idx, beam in enumerate(beams):
                # 跳过无效候选（分数为负无穷的序列）
                if beam_scores[beam_idx] == float('-inf'):
                    continue

                # 序列长度检查：如果超过最大长度，截取最后的部分
                beam_cond = beam if beam.size(1) <= self.args.max_seq_len else beam[:, -self.args.max_seq_len:]

                # 前向传播：获取模型对当前序列的预测
                output = self(beam_cond)
                # 提取最后一个位置的logits，用于预测下一个token
                logits = output.logits[:, -1, :]  # 形状: (1, vocab_size)

                # 温度缩放：调整logits的分布
                if temperature != 1.0:
                    logits = logits / temperature
                    # 温度 > 1：分布更平滑，增加随机性
                    # 温度 < 1：分布更尖锐，更确定

                # Top-k过滤：限制候选token的范围，提高质量
                if top_k is not None:
                    # 找到logits中前top_k个最大的值
                    v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                    # 将不在前top_k内的logits设为负无穷
                    logits[logits < v[:, [-1]]] = -float('Inf')
                    # 这样采样时只会考虑前top_k个token

                # 计算对数概率：使用log_softmax避免数值不稳定
                log_probs = F.log_softmax(logits, dim=-1)

                # 获取前 num_beams 个最可能的候选token
                # 注意：这里的top-k与上面的top-k不同
                # 上面的top-k是全局过滤，这里是束搜索的分支选择
                top_log_probs, top_indices = torch.topk(log_probs, k=num_beams, dim=-1)

                # 为当前候选序列生成 num_beams 个扩展序列
                for k in range(num_beams):
                    # 选择第k个候选token
                    token = top_indices[:, k:k+1]      # token ID
                    log_prob = top_log_probs[:, k]     # 对应的对数概率

                    # 扩展序列：将新token添加到当前序列末尾
                    new_beam = torch.cat([beam, token], dim=1)
                    # 更新累积分数：原序列分数 + 新token的对数概率
                    new_score = beam_scores[beam_idx] + log_prob.item()

                    # 保存新的候选序列和分数
                    new_beams.append(new_beam)
                    new_scores.append(new_score)

            # 安全检查：如果没有生成任何有效候选，提前结束
            if not new_beams:
                break

            # 筛选最佳候选：从所有新生成的候选中选择分数最高的 num_beams 个
            # 按分数降序排序，获取索引
            sorted_indices = sorted(range(len(new_scores)), key=lambda i: new_scores[i], reverse=True)
            # 选择前 num_beams 个最佳候选
            beams = [new_beams[i] for i in sorted_indices[:num_beams]]
            beam_scores = [new_scores[i] for i in sorted_indices[:num_beams]]

            # 停止条件检查：检查最佳序列是否以停止token结尾
            if stop_id is not None and beams[0][0, -1] == stop_id:
                break

        # 返回得分最高的序列，只返回新生成的部分（去掉原始输入）
        # beams[0] 是最终得分最高的完整序列
        # [:, seq_len:] 切片只保留生成部分
        return beams[0][:, seq_len:]

    @torch.inference_mode()
    def generate_super(self,
                       idx,
                       stop_id=None,
                       max_new_tokens=256,
                       temperature=1.0,
                       top_k=None,
                       do_sample=False,
                       num_beams=1,
                       attention_mask: Optional[torch.Tensor] = None,
                       pad_token_id: Optional[int] = None
                       ):
        """
        高级文本生成函数，支持三种解码策略：

        1. 贪婪解码（Greedy Search）：
           - 参数：do_sample=False, num_beams=1
           - 特点：每步选择概率最大的token，速度快、结果确定

        2. 随机采样（Random Sampling）：
           - 参数：do_sample=True, num_beams=1
           - 特点：基于概率分布随机采样，可配合temperature和top-k控制多样性

        3. 束搜索（Beam Search）：
           - 参数：do_sample=False, num_beams>1
           - 特点：维护多条候选路径，选择总概率最高的序列，质量更高但速度较慢

        Args:
            idx: 输入序列张量，形状为 (batch_size, seq_len)
            stop_id: 停止生成的token ID
            max_new_tokens: 最大生成token数量
            temperature: 温度参数，控制随机性，越高越随机
            top_k: 只考虑概率最高的k个token，None表示不考虑
            do_sample: 是否使用随机采样，False时使用确定性解码
            num_beams: 束搜索的束宽度，1表示不使用束搜索

        Returns:
            生成的token序列，形状为 (batch_size, generated_length)
        """
        # 参数验证
        if temperature <= 0:
            temperature = 0.001  # 避免除零错误
        if num_beams < 1:
            num_beams = 1
        if top_k is not None and top_k < 1:
            top_k = None
        if pad_token_id is None:
            pad_token_id = self.args.pad_token_id if self.args.pad_token_id is not None else 0
        attention_mask = self._prepare_attention_mask(attention_mask, idx)
        idx, attention_mask = self._left_pad_by_attention_mask(idx, attention_mask, pad_token_id)

        # 束搜索逻辑
        if not do_sample and num_beams > 1:
            return self._beam_search(idx, max_new_tokens, num_beams, temperature, top_k, stop_id)

        # 贪婪解码和随机采样逻辑
        finished = torch.zeros(idx.size(0), dtype=torch.bool, device=idx.device)
        index = idx.shape[1]
        for _ in range(max_new_tokens):
            # 如果序列上下文过长，截断它到最大长度
            idx_cond = idx if idx.size(1) <= self.args.max_seq_len else idx[:, -self.args.max_seq_len:]
            mask_cond = None
            if attention_mask is not None:
                mask_cond = attention_mask if attention_mask.size(1) <= self.args.max_seq_len else attention_mask[:, -self.args.max_seq_len:]

            # 前向传播获取序列中最后一个位置的 logits
            logits = self(idx_cond, attention_mask=mask_cond).logits
            logits = logits[:, -1, :] # 只保留最后一个时间步的输出

            # 根据参数选择解码策略
            if do_sample:
                idx_next = self._random_sample(logits, temperature, top_k)
            else:
                # 当temperature=0时使用贪婪解码
                if temperature < 0.1:
                    idx_next = self._greedy_decode(logits)
                else:
                    # 低温度下的随机采样（接近贪婪）
                    idx_next = self._random_sample(logits, temperature, top_k)

            prev_finished = finished.clone()
            if stop_id is not None:
                if prev_finished.any():
                    fill_token = pad_token_id if pad_token_id is not None else stop_id
                    idx_next = torch.where(prev_finished[:, None], torch.full_like(idx_next, fill_token), idx_next)
                finished = prev_finished | idx_next[:, 0].eq(stop_id)

            # 将选择的token添加到序列中
            idx = torch.cat((idx, idx_next), dim=1)
            if attention_mask is not None:
                next_mask = torch.ones((attention_mask.size(0), 1), dtype=attention_mask.dtype, device=attention_mask.device)
                if prev_finished.any():
                    next_mask[prev_finished] = False
                attention_mask = torch.cat((attention_mask, next_mask), dim=1)

            if stop_id is not None and finished.all():
                break

        return idx[:, index:] # 只返回生成的token

if __name__ == '__main__':
    tokenizer = AutoTokenizer.from_pretrained("tokenizer_k")
    args = ModelConfig(
        dim=1024,
        n_layers=18,
        pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0,
    )
    # 实例化LLaMA2Model
    model = Transformer(args=args)
    # 计算model的全部参数
    num_params = sum(p.numel() for p in model.parameters())
    print(f'LLM总参数量：{num_params / 1e6:.3f} 百万')

    prompt = "你好呀，今天吃什么呢？你过得怎么样嘞？"
    text = f"{tokenizer.bos_token}{prompt}{tokenizer.eos_token}"
    print(f"Input text: {text}")

    input_id = tokenizer(text).data['input_ids']
    print("input_ids :", input_id)
    print("dcode_str :", tokenizer.decode(input_id))

    X = torch.tensor(input_id[:-1]).unsqueeze(0)
    Y = torch.tensor(input_id[1:]).unsqueeze(0)
    print("X shape :", X.shape)
    print("Y shape :", Y.shape)

    # 将输入张量传入模型
    output = model(X, Y)
