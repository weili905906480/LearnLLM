import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator


@dataclass(frozen=True)
class RawSample:
    id: str
    system: str | None
    prompt: str
    answer: str
    image: str | None


def iter_jsonl(path: str | Path) -> Iterator[RawSample]:
    p = Path(path)
    with p.open("r", encoding="utf-8-sig") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"invalid json at line {line_no}: {p}") from e

            yield RawSample(
                id=str(obj.get("id", f"line-{line_no}")),
                system=obj.get("system"),
                prompt=str(obj.get("prompt", "")),
                answer=str(obj.get("answer", "")),
                image=obj.get("image"),
            )
