import json
import zipfile
from pathlib import Path

from ..data_paths import (
    FLICKR8K_DIR,
    FLICKR8K_IMAGES_DIR,
    FLICKR8K_IMAGES_ZIP,
    FLICKR8K_TEXT_DIR,
    FLICKR8K_TEXT_ZIP,
    FLICKR8K_TRAIN_LIST,
    FLICKR8K_ZHC_CAPTIONS,
    MM_TRAIN_JSONL,
)
from .cleaning import normalize_text
from .fetch import download

_MM_DEFAULT_SYSTEM = "你是一个只用中文回答的助手。"
_MM_DEFAULT_PROMPT = "请描述这张图片。"
_CLEANUP_ZIPS_AFTER_EXTRACT = True


def _first_existing_path(paths: list[Path], *, what: str) -> Path:
    for p in paths:
        if p.exists():
            return p
    tried = "\n".join([f"  - {p}" for p in paths])
    raise FileNotFoundError(f"{what} not found. tried:\n{tried}")


def _resolve_flickr8k_images_dir(root: Path) -> Path:
    return _first_existing_path(
        [
            root / "Flickr8k_Dataset",
            root / "Flicker8k_Dataset",
            FLICKR8K_IMAGES_DIR,
        ],
        what="Flickr8k images dir",
    )


def _resolve_flickr8k_train_list(text_root: Path) -> Path:
    return _first_existing_path(
        [
            text_root / "Flickr_8k.trainImages.txt",
            text_root / "Flickr8k_text" / "Flickr_8k.trainImages.txt",
            FLICKR8K_TRAIN_LIST,
        ],
        what="Flickr8k train list",
    )


def _extract_zip(zip_path: Path, dst_dir: Path, *, overwrite: bool) -> None:
    marker = dst_dir / f".extract_{zip_path.stem}.done"
    if marker.exists() and not overwrite:
        if _CLEANUP_ZIPS_AFTER_EXTRACT and zip_path.exists():
            zip_path.unlink(missing_ok=True)
        return

    if overwrite and marker.exists():
        marker.unlink()

    dst_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(dst_dir)
    marker.write_text("ok\n", encoding="utf-8")

    if _CLEANUP_ZIPS_AFTER_EXTRACT:
        zip_path.unlink(missing_ok=True)


def _load_filename_list(path: Path) -> set[str]:
    names: set[str] = set()
    with path.open("r", encoding="utf-8-sig") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            names.add(Path(s).name)
    return names


def _build_mm_jsonl_from_flickr8kcn(
    zhc_captions: Path,
    *,
    images_dir: Path,
    train_list: Path,
    out_jsonl: Path,
    system: str | None,
    prompt: str,
) -> None:
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    train_names = _load_filename_list(train_list)

    written = 0
    missing = 0
    with zhc_captions.open("r", encoding="utf-8-sig") as r, out_jsonl.open("w", encoding="utf-8", newline="\n") as w:
        for line in r:
            line = line.strip()
            if not line or " " not in line:
                continue
            key, cap = line.split(" ", 1)
            cap = normalize_text(cap)
            if not cap:
                continue

            parts = key.split("#")
            if len(parts) < 3:
                continue
            fname = parts[0]
            if Path(fname).name not in train_names:
                continue

            img_path = images_dir / fname
            if not img_path.is_file():
                missing += 1
                continue

            suffix = "-".join(parts[1:])
            ex_id = f"flickr8k-{Path(fname).stem}-{suffix}"
            row = {
                "id": ex_id,
                "system": (str(system) if system else None),
                "prompt": str(prompt),
                "answer": cap,
                "image": str(img_path),
            }
            w.write(json.dumps(row, ensure_ascii=False) + "\n")
            written += 1

    if written <= 0:
        raise RuntimeError(f"no mm samples written: {out_jsonl} (missing_images={missing})")
    print(f"ok: mm jsonl -> {out_jsonl} (rows={written}, missing_images={missing})")


def ensure_flickr8k_mm_jsonl(cfg: dict) -> None:
    mm_cfg = cfg.get("mm", {})
    mm_train_jsonl = MM_TRAIN_JSONL
    if mm_train_jsonl.exists() and mm_train_jsonl.stat().st_size > 0:
        print(f"skip: mm jsonl exists -> {mm_train_jsonl}")
        return

    if not bool(mm_cfg.get("download", True)):
        raise FileNotFoundError(f"mm.train_jsonl missing: {mm_train_jsonl}")

    overwrite_mm = bool(mm_cfg.get("overwrite_download", False))
    images_url = str(mm_cfg["images_url"])
    text_url = str(mm_cfg["text_url"])
    zhc_url = str(mm_cfg["zhc_captions_url"])

    dataset_dir = FLICKR8K_DIR
    images_zip = FLICKR8K_IMAGES_ZIP
    text_zip = FLICKR8K_TEXT_ZIP
    text_dir = FLICKR8K_TEXT_DIR
    zhc_path = FLICKR8K_ZHC_CAPTIONS

    system = _MM_DEFAULT_SYSTEM
    prompt = _MM_DEFAULT_PROMPT

    dataset_dir.mkdir(parents=True, exist_ok=True)

    download(images_url, images_zip, overwrite=overwrite_mm)
    download(text_url, text_zip, overwrite=overwrite_mm)
    download(zhc_url, zhc_path, overwrite=overwrite_mm)

    _extract_zip(images_zip, dataset_dir, overwrite=overwrite_mm)
    _extract_zip(text_zip, text_dir, overwrite=overwrite_mm)

    images_dir = _resolve_flickr8k_images_dir(dataset_dir)
    train_list = _resolve_flickr8k_train_list(text_dir)

    _build_mm_jsonl_from_flickr8kcn(
        zhc_path,
        images_dir=images_dir,
        train_list=train_list,
        out_jsonl=mm_train_jsonl,
        system=(str(system) if system else None),
        prompt=prompt,
    )
