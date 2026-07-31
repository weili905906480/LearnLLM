from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from .special_tokens import DEFAULT_SPECIAL_TOKENS_SCHEME


def load_yaml(path: str | Path) -> dict[str, Any]:
    p = Path(path)
    seen: set[Path] = set()

    def _load(cur: Path) -> dict[str, Any]:
        cur = cur.resolve()
        if cur in seen:
            raise ValueError(f"cycle detected in !include: {cur}")
        seen.add(cur)

        class Loader(yaml.SafeLoader):
            pass

        def _include(loader: Loader, node: yaml.Node):
            rel = Path(loader.construct_scalar(node))
            inc = (cur.parent / rel).resolve()
            return _load(inc)

        Loader.add_constructor("!include", _include)
        return yaml.load(cur.read_text(encoding="utf-8"), Loader=Loader)

    raw = _load(p)
    if not isinstance(raw, dict):
        raise ValueError(f"expected yaml mapping at top-level: {p}")
    return raw


@dataclass(frozen=True)
class ModelConfig:
    name: str
    vocab_size: int
    max_seq_len: int

    hidden_size: int
    num_layers: int
    num_heads: int
    num_kv_heads: int

    # 特殊 token 如何占据词表前部的 ID。
    # 当前默认只使用 minimind2_chatml：<|endoftext|>/<|im_start|>/<|im_end|> + <img_*>
    special_tokens_scheme: str = DEFAULT_SPECIAL_TOKENS_SCHEME

    rope_theta: float = 1_000_000.0
    dropout: float = 0.0

    # FFN（前馈层）。
    mlp_intermediate_size: int | None = None

    # 特征 token 设置（最多 1 张图）。
    image_feat_dim: int = 768
    image_tokens: int = 49


@dataclass(frozen=True)
class DataConfig:
    train_dir: Path
    val_dir: Path | None = None

    # 可选的多数据集混合（按 batch 采样）。开启后，train_dir 视为默认数据集，
    # 训练会按 mix_weights 的概率从 mix_train_dirs 中抽取 batch。
    mix_train_dirs: tuple[Path, ...] | None = None
    mix_weights: tuple[float, ...] | None = None


@dataclass(frozen=True)
class TrainConfig:
    stage: str = 's0_text'  # s0_text|sft
    init_from: Path | None = None  # initialize model weights from checkpoint (model-only)

    device: str = 'auto'  # auto|cpu|cuda
    dtype: str = 'bf16'  # fp16|bf16|fp32

    # 冻结相关控制（减少遗忘 / 适配小显存）。
    # - 当 freeze_backbone=True 时，可选只解冻最后 N 个 block。
    freeze_backbone: bool = False
    unfreeze_last_n_layers: int = 0
    freeze_base_embed: bool = False
    freeze_special_embed: bool = False

    epochs: int = 1
    batch_size: int = 16
    grad_accum: int = 1

    lr: float = 4e-4
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    warmup_steps: int = 200
    max_steps: int | None = None

    log_every: int = 50
    save_every: int = 1000
    keep_last: int | None = None

    # 训练可视化（TensorBoard）。
    tb_enable: bool = True
    tb_dir: Path | None = None  # 默认：outputs/tb/<experiment>/<stage>（与 checkpoints 下 out_dir 的相对结构一致）
    tb_every: int = 50  # 标量写入频率（step）

    out_dir: Path = Path('checkpoints/seeker-omni')


@dataclass(frozen=True)
class ExperimentConfig:
    seed: int
    data: DataConfig
    model: ModelConfig
    train: TrainConfig

    @staticmethod
    def load(path: str | Path) -> 'ExperimentConfig':
        p = Path(path)
        raw: dict[str, Any] = load_yaml(p)

        d = raw['data']

        mix_dirs = d.get('mix_train_dirs')
        mix_train_dirs = tuple(Path(x) for x in mix_dirs) if mix_dirs else None

        mix_weights_raw = d.get('mix_weights')
        mix_weights = tuple(float(x) for x in mix_weights_raw) if mix_weights_raw else None

        data = DataConfig(
            train_dir=Path(d['train_dir']),
            val_dir=Path(d['val_dir']) if d.get('val_dir') else None,
            mix_train_dirs=mix_train_dirs,
            mix_weights=mix_weights,
        )

        m_raw = raw['model']
        base = m_raw.get("base") if isinstance(m_raw, dict) else None
        if base is not None:
            if not isinstance(base, dict):
                raise ValueError("model.base must be a mapping")
            m = dict(base)
            for k, v in m_raw.items():
                if k != "base":
                    m[k] = v
        else:
            m = m_raw
        mlp_intermediate_size = m.get('mlp_intermediate_size')

        model = ModelConfig(
            name=str(m['name']),
            vocab_size=int(m['vocab_size']),
            max_seq_len=int(m['max_seq_len']),
            hidden_size=int(m['hidden_size']),
            num_layers=int(m['num_layers']),
            num_heads=int(m['num_heads']),
            num_kv_heads=int(m['num_kv_heads']),
            special_tokens_scheme=str(m.get('special_tokens_scheme', DEFAULT_SPECIAL_TOKENS_SCHEME)),
            rope_theta=float(m.get('rope_theta', 1_000_000.0)),
            dropout=float(m.get('dropout', 0.0)),
            mlp_intermediate_size=(int(mlp_intermediate_size) if mlp_intermediate_size is not None else None),
            image_feat_dim=int(m.get('image_feat_dim', 768)),
            image_tokens=int(m.get('image_tokens', 49)),
        )

        init_from = raw.get('train', {}).get('init_from')
        train = TrainConfig(
            stage=str(raw['train'].get('stage', 's0_text')),
            init_from=(Path(str(init_from)) if init_from else None),
            device=str(raw['train'].get('device', 'auto')),
            dtype=str(raw['train'].get('dtype', 'bf16')),
            freeze_backbone=bool(raw['train'].get('freeze_backbone', False)),
            unfreeze_last_n_layers=int(raw['train'].get('unfreeze_last_n_layers', 0)),
            freeze_base_embed=bool(raw['train'].get('freeze_base_embed', False)),
            freeze_special_embed=bool(raw['train'].get('freeze_special_embed', False)),
            epochs=int(raw['train'].get('epochs', 1)),
            batch_size=int(raw['train'].get('batch_size', 16)),
            grad_accum=int(raw['train'].get('grad_accum', 1)),
            lr=float(raw['train'].get('lr', 4e-4)),
            weight_decay=float(raw['train'].get('weight_decay', 0.1)),
            grad_clip=float(raw['train'].get('grad_clip', 1.0)),
            warmup_steps=int(raw['train'].get('warmup_steps', 200)),
            max_steps=(int(raw['train']['max_steps']) if raw['train'].get('max_steps') is not None else None),
            log_every=int(raw['train'].get('log_every', 50)),
            save_every=int(raw['train'].get('save_every', 1000)),
            keep_last=(int(raw['train']['keep_last']) if raw['train'].get('keep_last') is not None else None),
            tb_enable=bool(raw['train'].get('tb_enable', True)),
            tb_dir=(Path(raw['train']['tb_dir']) if raw['train'].get('tb_dir') else None),
            tb_every=int(raw['train'].get('tb_every', raw['train'].get('log_every', 50))),
            out_dir=Path(raw['train'].get('out_dir', 'checkpoints/seeker-omni')),
        )

        return ExperimentConfig(seed=int(raw.get('seed', 42)), data=data, model=model, train=train)
