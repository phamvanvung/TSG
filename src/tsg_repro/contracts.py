from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def require_train_only_metadata(metadata: dict[str, Any]) -> None:
    """Hard firewall for map fitting artifacts."""
    split = str(metadata.get("split", "")).upper()
    if split not in {"TRAIN", "BASIS_FIT", "JOURNAL_GEOMETRY_FIT"}:
        raise ValueError(f"canonical map fitting requires a training/fit split, got {split!r}")
    if metadata.get("val_used") is not False:
        raise ValueError("map fit metadata must assert val_used=false")
    if metadata.get("pres_lost_labels_used") is not False:
        raise ValueError("map fit metadata must assert pres_lost_labels_used=false")
