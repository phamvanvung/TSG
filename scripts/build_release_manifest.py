#!/usr/bin/env python
"""Create deterministic SHA-256 manifest for source/release files."""
from __future__ import annotations

import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXCLUDED_PARTS = {".git", ".pytest_cache", "__pycache__", "results"}
EXCLUDED_FILES = {"MANIFEST.txt"}
EXCLUDED_RELATIVE = {
    Path("configs/paths.json"),
    Path("configs/environment.requirements.json.pre_freeze"),
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    rows = []
    for p in sorted(ROOT.rglob("*")):
        if not p.is_file():
            continue
        rel = p.relative_to(ROOT)
        if rel in EXCLUDED_RELATIVE or p.name in EXCLUDED_FILES or any(part in EXCLUDED_PARTS for part in rel.parts):
            continue
        if p.suffix in {".pyc", ".pyo"}:
            continue
        rows.append(f"{sha256(p)}  {rel.as_posix()}")
    out = ROOT / "MANIFEST.txt"
    out.write_text("\n".join(rows) + "\n", encoding="utf-8")
    print(f"wrote {len(rows)} entries -> {out}")


if __name__ == "__main__":
    main()
