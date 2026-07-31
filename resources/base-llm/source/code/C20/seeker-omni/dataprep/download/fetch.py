import sys
import urllib.request
from pathlib import Path

from tqdm import tqdm


def download(url: str, dst: Path, *, overwrite: bool, timeout: int = 60, user_agent: str = "seeker-omni") -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() and dst.stat().st_size > 0 and not overwrite:
        return

    tmp_path = dst.with_suffix(dst.suffix + ".part")
    if tmp_path.exists():
        tmp_path.unlink()

    req = urllib.request.Request(url, headers={"User-Agent": str(user_agent)})
    with urllib.request.urlopen(req, timeout=int(timeout)) as r:
        total = r.headers.get("Content-Length")
        total_i = int(total) if total and total.isdigit() else None

        with tmp_path.open("wb") as f:
            pbar = tqdm(total=total_i, unit="B", unit_scale=True, desc=f"download:{dst.name}", file=sys.stdout)
            for chunk in iter(lambda: r.read(1024 * 1024), b""):
                f.write(chunk)
                pbar.update(len(chunk))
            pbar.close()

    if tmp_path.stat().st_size <= 0:
        raise RuntimeError(f"download produced empty file: {url}")

    if overwrite and dst.exists():
        dst.unlink(missing_ok=True)
    tmp_path.replace(dst)

