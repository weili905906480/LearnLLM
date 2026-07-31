import unicodedata


def is_cjk(ch: str) -> bool:
    cp = ord(ch)
    return (
        (0x4E00 <= cp <= 0x9FFF)
        or (0x3400 <= cp <= 0x4DBF)
        or (0x20000 <= cp <= 0x2A6DF)
        or (0x2A700 <= cp <= 0x2B73F)
        or (0x2B740 <= cp <= 0x2B81F)
        or (0x2B820 <= cp <= 0x2CEAF)
    )


def text_stats(s: str) -> tuple[int, int, int, int]:
    zh = 0
    latin = 0
    other = 0
    vis = 0
    for ch in s:
        if ch.isspace():
            continue
        vis += 1
        if is_cjk(ch):
            zh += 1
        elif ("A" <= ch <= "Z") or ("a" <= ch <= "z"):
            latin += 1
        else:
            other += 1
    return zh, latin, other, vis


def normalize_text(s: str) -> str:
    s = (s or "").replace("\r", " ").replace("\n", " ").strip()
    s = " ".join(s.split())
    return s


def keep_text(
    s: str,
    *,
    min_chars: int,
    max_chars: int,
    min_zh_ratio: float,
    max_latin_ratio: float,
    max_symbol_ratio: float,
) -> bool:
    if not s:
        return False
    if "\ufffd" in s:
        return False
    for ch in s:
        cat = unicodedata.category(ch)
        if cat.startswith("C") and ch not in ("\t", "\n", "\r"):
            return False

    if len(s) < int(min_chars) or len(s) > int(max_chars):
        return False

    zh, latin, other, vis = text_stats(s)
    if vis <= 0:
        return False

    if (float(zh) / float(vis)) < float(min_zh_ratio):
        return False
    if (float(latin) / float(vis)) > float(max_latin_ratio):
        return False
    if (float(other) / float(vis)) > float(max_symbol_ratio):
        return False
    return True

