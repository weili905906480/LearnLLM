import os
from pathlib import Path

from .config import load_yaml

def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _read_yaml(path: str | Path) -> dict:
    p = Path(path)
    return load_yaml(p)


def train() -> int:
    from .steps.train import run as train_run

    os.chdir(_project_root())
    print("== pipeline: train ==")
    cfg = _read_yaml("configs/train.yaml")
    train_run(cfg)
    return 0


def e2e() -> int:
    from .steps.e2e import run_from_yaml_config

    os.chdir(_project_root())
    print("== pipeline: e2e ==")
    cfg = _read_yaml("configs/e2e.yaml")
    return int(run_from_yaml_config(cfg))


def run_all() -> int:
    train()
    e2e()
    return 0
