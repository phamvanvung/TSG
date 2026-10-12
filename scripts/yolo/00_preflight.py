#!/usr/bin/env python
"""Verify frozen YOLO11s assets before any expensive reproduction stage."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from tsg_repro.config import get_path, load_config
from tsg_repro.data import load_manifest, validate_manifest_contract
from tsg_repro.hashing import sha256_file


REFERENCE = ROOT / "reference_results" / "yolo11s_frozen_hashes.json"
REFERENCE_KEYS = {
    "clean_teacher": "clean_teacher_best.pt",
    "none": "none_best.pt",
    "clean_kd": "clean_kd_best.pt",
    "gaussian_manifest": "gaussian_train_val_manifest.csv",
    "checkpoint_companion": "checkpoint_companion",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--allow-hash-mismatch",
        action="store_true",
        help="Development only: continue when an asset differs from the frozen study asset.",
    )
    args = parser.parse_args()

    cfg, base = load_config(args.config)
    frozen = json.loads(REFERENCE.read_text(encoding="utf-8"))["sha256"]
    items = {
        "clean_teacher": get_path(cfg, base, "checkpoints", "yolo11s_clean_teacher"),
        "none": get_path(cfg, base, "checkpoints", "yolo11s_none"),
        "clean_kd": get_path(cfg, base, "checkpoints", "yolo11s_clean_kd"),
        "gaussian_manifest": get_path(cfg, base, "data", "gaussian_manifest"),
        "checkpoint_companion": get_path(
            cfg, base, "checkpoints", "yolo_checkpoint_companion", required=False
        ),
    }

    out = {"status": "PASS", "reference": str(REFERENCE), "files": {}}
    for key, path in items.items():
        if path is None:
            if key == "checkpoint_companion":
                raise RuntimeError(
                    "The frozen YOLO checkpoints require "
                    "checkpoints.yolo_checkpoint_companion"
                )
            raise RuntimeError(f"Missing configured path for {key}")
        if not path.is_file():
            raise FileNotFoundError(path)
        actual = sha256_file(path)
        expected = frozen[REFERENCE_KEYS[key]]
        ok = actual == expected
        out["files"][key] = {
            "path": str(path),
            "sha256": actual,
            "expected_sha256": expected,
            "match": ok,
        }
        if not ok and not args.allow_hash_mismatch:
            raise RuntimeError(f"{key} SHA256 mismatch: {actual}")

    manifest = load_manifest(items["gaussian_manifest"])
    out["manifest_counts"] = validate_manifest_contract(manifest)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
