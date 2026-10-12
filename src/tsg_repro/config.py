from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_config(path: str | Path) -> tuple[dict[str, Any], Path]:
    p = Path(path).expanduser().resolve()
    obj = json.loads(p.read_text(encoding="utf-8"))
    if not isinstance(obj, dict):
        raise ValueError("Configuration root must be a JSON object")
    return obj, p.parent


def resolve_path(value: str | Path | None, base: Path) -> Path | None:
    if value is None:
        return None
    p = Path(value).expanduser()
    return (p if p.is_absolute() else base / p).resolve()


def get_path(cfg: dict[str, Any], base: Path, *keys: str, required: bool = True) -> Path | None:
    cur: Any = cfg
    for key in keys:
        if not isinstance(cur, dict) or key not in cur:
            if required:
                raise KeyError("Missing config key: " + ".".join(keys))
            return None
        cur = cur[key]
    p = resolve_path(cur, base)
    if required and p is None:
        raise ValueError("Null path for required key: " + ".".join(keys))
    return p
