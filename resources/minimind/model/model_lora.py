import torch
from torch import optim, nn


# 定义Lora网络结构
class LoRA(nn.Module):
    """
    LoRA（Low-Rank Adaptation，低秩自适应）微调模块。

    核心思想：不直接微调原始权重矩阵 W（参数多、显存大），而是「冻结 W，只训练一个低秩增量 ΔW」。
    把 ΔW 分解成两个小矩阵的乘积 B·A，从而把可训练参数量从 in×out 降到 rank×(in+out)。

        y = Wx + ΔW·x = Wx + (B·A)·x = Wx + B(Ax)

    其中 A 把输入从 in 维压到 rank 维（下采样），B 把 rank 维升回 out 维（上采样）。
    rank 越小，可训练参数越少，但表达能力也越弱；rank 是「参数效率 vs 容量」的旋钮。

    初始化策略（关键，保证训练从「不改变原模型」出发）：
        A 用高斯初始化（提供随机方向），B 用全 0 初始化，
        于是训练开始时 ΔW = B·A = 0，LoRA 输出恒为 0，模型行为与原模型完全一致，
        之后训练才逐渐学出增量。这样避免了一开始就破坏预训练权重。

    参数量对比（in=out=768、rank=16）：
        全量微调 W：768×768 ≈ 59 万参数
        LoRA 增量：rank×(in+out) = 16×1536 ≈ 2.5 万参数（约 4%）

    具体矩阵举例（in=4、out=4、rank=2，单输入向量 x = [1,2,3,4]）：

        A（nn.Linear(4,2)，weight 形状 [2,4]）:
            A.weight = [[ 0.1, 0.2, -0.1,  0.3],
                        [ 0.0, 0.1,  0.2, -0.2]]

        B（nn.Linear(2,4)，weight 形状 [4,2]）:
            B.weight = [[ 0.5, -0.1],
                        [ 0.2,  0.3],
                        [-0.3,  0.1],
                        [ 0.1,  0.4]]

        先算 A(x) = A.weight @ x（把 4 维压到 2 维）:
            [ 0.1·1 + 0.2·2 + (-0.1)·3 + 0.3·4 ]   = [ 0.1 + 0.4 - 0.3 + 1.2 ]   = [ 1.4 ]
            [ 0.0·1 + 0.1·2 +  0.2·3 + (-0.2)·4 ]   = [ 0.0 + 0.2 + 0.6 - 0.8 ]   = [ 0.0 ]
            => A(x) = [1.4, 0.0]

        再算 B(A(x)) = B.weight @ [1.4, 0.0]（把 2 维升回 4 维）:
            [ 0.5·1.4 + (-0.1)·0 ]   = [  0.70 ]
            [ 0.2·1.4 +  0.3·0  ]   = [  0.28 ]
            [-0.3·1.4 +  0.1·0  ]   = [ -0.42 ]
            [ 0.1·1.4 +  0.4·0  ]   = [  0.14 ]
            => B(A(x)) = [0.70, 0.28, -0.42, 0.14]

        最终 forward 输出 = 原始线性层输出 Wx + [0.70, 0.28, -0.42, 0.14]。
        增量 ΔW = B.weight @ A.weight（[4,2]@[2,4]=[4,4]），上面算的是 ΔW·x。
    """
    def __init__(self, in_features, out_features, rank):
        super().__init__()
        self.rank = rank  # LoRA的秩（rank），控制低秩矩阵的大小
        self.A = nn.Linear(in_features, rank, bias=False)  # 低秩矩阵A
        self.B = nn.Linear(rank, out_features, bias=False)  # 低秩矩阵B
        # 矩阵A高斯初始化
        self.A.weight.data.normal_(mean=0.0, std=0.02)
        # 矩阵B全0初始化
        self.B.weight.data.zero_()

    def forward(self, x):
        # 低秩增量：x -> A -> B，即 B(A(x)) = (B·A)·x
        # 先由 A 降维到 rank，再由 B 升维回 out
        return self.B(self.A(x))


def apply_lora(model, rank=16):
    # 遍历模型所有子模块，给符合条件的 nn.Linear 注入 LoRA 增量
    for name, module in model.named_modules():
        # 只对「方阵」线性层（in_features == out_features）注入 LoRA。
        # 这是一个简化的选择策略：MiniMind 里满足条件的正好是注意力里的 q_proj 和 o_proj
        # （q_proj 768->8*96=768、o_proj 768->768），即主要微调注意力投影。
        if isinstance(module, nn.Linear) and module.in_features == module.out_features:
            # 为该层新建一个 LoRA 模块，并挂到 module.lora 属性上（搬到 model 所在设备）
            lora = LoRA(module.in_features, module.out_features, rank=rank).to(model.device)
            setattr(module, "lora", lora)
            # 保存原始 forward，稍后在新 forward 里调用（原权重 W 被冻结，不更新）
            original_forward = module.forward

            # 显式绑定
            # 关键：用默认参数 layer1=original_forward、layer2=lora 捕获当前循环的原始 forward 和 lora。
            # 这是为了避免 Python 闭包的「延迟绑定」陷阱——如果直接在函数体里引用 module/lora，
            # 循环结束后所有注入的 forward 都会指向最后一次迭代的对象。
            def forward_with_lora(x, layer1=original_forward, layer2=lora):
                # 输出 = 原始线性层输出 + LoRA 增量：Wx + B(Ax)
                return layer1(x) + layer2(x)

            # 替换该模块的 forward，之后调用即自动加上 LoRA 增量
            module.forward = forward_with_lora


def load_lora(model, path):
    # 加载仅含 LoRA 增量的 checkpoint（不含基础权重）
    state_dict = torch.load(path, map_location=model.device)
    # 去掉 DataParallel/DDP 常见的 'module.' 前缀（k[7:] 即去掉前 7 个字符 'module.'）
    state_dict = {(k[7:] if k.startswith('module.') else k): v for k, v in state_dict.items()}

    # 按模块名匹配：把 state_dict 里属于该模块的 LoRA 参数挑出来，load 到 module.lora
    for name, module in model.named_modules():
        if hasattr(module, 'lora'):
            # 例：state_dict 键 "model.layers.0.self_attn.q_proj.lora.A.weight"
            #     过滤前缀 f'{name}.lora.' 后得到 "A.weight"，与 module.lora 的参数名对齐
            lora_state = {k.replace(f'{name}.lora.', ''): v for k, v in state_dict.items() if f'{name}.lora.' in k}
            module.lora.load_state_dict(lora_state)


def save_lora(model, path):
    # 只保存 LoRA 增量（不保存基础权重），文件更小、便于分发
    raw_model = getattr(model, '_orig_mod', model)   # 兼容 torch.compile 包装（取原始模型）
    state_dict = {}
    for name, module in raw_model.named_modules():
        if hasattr(module, 'lora'):
            # 去掉可能的 'module.' 前缀，得到干净的模块名
            clean_name = name[7:] if name.startswith("module.") else name
            # 只存该模块的 lora 参数，并转 half（fp16）省一半体积
            lora_state = {f'{clean_name}.lora.{k}': v.cpu().half() for k, v in module.lora.state_dict().items()}
            state_dict.update(lora_state)
    torch.save(state_dict, path)


def merge_lora(model, lora_path, save_path):
    # 把 LoRA 增量「合并回」基础权重，得到可独立部署的完整模型（推理时无需再算 LoRA 分支）
    load_lora(model, lora_path)                      # 先把 LoRA 权重加载进 model
    raw_model = getattr(model, '_orig_mod', model)   # 兼容 torch.compile 包装
    # 拷贝基础权重（排除 .lora. 参数），并统一转 half
    state_dict = {k: v.cpu().half() for k, v in raw_model.state_dict().items() if '.lora.' not in k}
    for name, module in raw_model.named_modules():
        # 遍历所有普通线性层（跳过 LoRA 自身内部的 A/B 层）
        if isinstance(module, nn.Linear) and '.lora.' not in name:
            # 覆盖该层的 weight 为基础权重（因为上面 state_dict 里已有一份，这里重写保证 fresh）
            state_dict[f'{name}.weight'] = module.weight.data.clone().cpu().half()
            if hasattr(module, 'lora'):
                # 合并公式：W_new = W + B·A
                #   module.lora.B.weight 形状 [out, rank]，module.lora.A.weight 形状 [rank, in]
                #   二者相乘得到 [out, in]，即 ΔW，加到原 W [out, in] 上
                #   例（rank=2，out=4，in=4）：[4,2] @ [2,4] = [4,4]，加到 W [4,4]
                state_dict[f'{name}.weight'] += (module.lora.B.weight.data @ module.lora.A.weight.data).cpu().half()
    # 保存合并后的完整模型权重
    torch.save(state_dict, save_path)
