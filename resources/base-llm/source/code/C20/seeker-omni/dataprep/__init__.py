import os
from pathlib import Path

from .data_paths import default_dataprep_cfg


def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def run_download() -> int:
    from .download import run as download_run

    os.chdir(_project_root())
    print("== dataprep: download ==")
    cfg = default_dataprep_cfg()
    download_run(cfg)
    return 0


def run_prepare() -> int:
    from .prepare import run as prepare_run

    os.chdir(_project_root())
    print("== dataprep: prepare ==")
    cfg = default_dataprep_cfg()
    prepare_run(cfg)
    return 0


def run_all() -> int:
    os.chdir(_project_root())
    run_download()
    run_prepare()
    return 0
