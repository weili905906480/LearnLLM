from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from ..config import ExperimentConfig
from ..train.checkpoint import latest_checkpoint
from ..train.loop import train
from ..train.seed import set_seed


def run(cfg: dict) -> None:
    cfg_paths = [str(x) for x in cfg["configs"]]
    auto_init = bool(cfg.get("auto_init", True))
    init_from_raw = cfg.get("init_from")
    init_from = Path(str(init_from_raw)) if init_from_raw else None

    prev_cfg: ExperimentConfig | None = None

    for i, cfg_path in enumerate(cfg_paths):
        exp = ExperimentConfig.load(cfg_path)

        if i == 0 and init_from is not None and init_from.exists():
            if exp.train.init_from is None or not Path(exp.train.init_from).exists():
                exp = replace(exp, train=replace(exp.train, init_from=init_from))

        if i > 0 and auto_init and prev_cfg is not None:
            prev_ckpt = latest_checkpoint(prev_cfg.train.out_dir)
            if prev_ckpt is None:
                raise RuntimeError(f"no checkpoint found in previous out_dir: {prev_cfg.train.out_dir}")

            cur_init = exp.train.init_from
            if cur_init is None or not Path(cur_init).exists():
                exp = replace(exp, train=replace(exp.train, init_from=prev_ckpt))

        print(f"\n==> stage {i+1}/{len(cfg_paths)}: {exp.train.stage} ({exp.model.name})", flush=True)
        print(f"    data: {exp.data.train_dir}", flush=True)
        print(f"    out:  {exp.train.out_dir}", flush=True)
        if exp.train.init_from is not None:
            print(f"    init_from: {exp.train.init_from}", flush=True)

        set_seed(exp.seed)
        train(exp)
        prev_cfg = exp
