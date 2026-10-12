#!/usr/bin/env python
"""Verify frozen RT-DETR-L assets and cohort manifests before reproduction."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from tsg_repro.config import get_path, load_config
from tsg_repro.geometry_workflow import load_view_manifest
from tsg_repro.hashing import sha256_file

REFERENCE = ROOT / "reference_results" / "rtdetr_l_frozen_hashes.json"
REFERENCE_KEYS = {
    "rtdetr_l_clean_teacher": "clean_teacher_best.pt",
    "rtdetr_l_none": "none_best.pt",
    "rtdetr_l_clean_kd": "clean_kd_best.pt",
    "rtdetr_checkpoint_companion": "checkpoint_companion",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--allow-hash-mismatch",
        action="store_true",
        help="Development only: continue with assets different from the frozen study files.",
    )
    args = parser.parse_args()

    cfg, base = load_config(args.config)
    expected = json.loads(REFERENCE.read_text(encoding="utf-8"))["sha256"]
    assets = {}
    for key in REFERENCE_KEYS:
        path = get_path(cfg, base, "checkpoints", key)
        actual = sha256_file(path)
        target = expected[REFERENCE_KEYS[key]]
        match = actual == target
        assets[key] = {
            "path": str(path),
            "sha256": actual,
            "expected_sha256": target,
            "match": match,
        }
        if not match and not args.allow_hash_mismatch:
            raise RuntimeError(f"{key} SHA256 mismatch: {actual}")

    bridge = get_path(cfg, base, "artifacts", "rtdetr_step10_bridge")
    bridge_sha = sha256_file(bridge)
    frozen_bridge_sha = expected.get("historical_none_to_clean_kd_bridge")
    bridge_match = bridge_sha == frozen_bridge_sha
    if frozen_bridge_sha and not bridge_match and not args.allow_hash_mismatch:
        raise RuntimeError(f"rtdetr_step10_bridge SHA256 mismatch: {bridge_sha}")

    fit = get_path(cfg, base, "data", "rtdetr_geometry_fit_manifest")
    confirm = get_path(cfg, base, "data", "rtdetr_confirm_manifest")
    fit_df = load_view_manifest(fit, expected_sources=120, expected_views=360)
    confirm_df = load_view_manifest(confirm, expected_sources=200, expected_views=600)
    fit_sha = sha256_file(fit)
    frozen_fit_sha = expected.get("geometry_fit_gaussian_view_manifest")
    if frozen_fit_sha and fit_sha != frozen_fit_sha and not args.allow_hash_mismatch:
        raise RuntimeError(f"rtdetr_geometry_fit_manifest SHA256 mismatch: {fit_sha}")
    for name, df in (("geometry_fit", fit_df), ("confirm", confirm_df)):
        if "view_index" not in df.columns:
            raise RuntimeError(f"{name} manifest lacks frozen view_index")
        view_index = df["view_index"].astype(int)
        expected_indices = set(range(len(df)))
        if set(view_index) != expected_indices:
            raise RuntimeError(
                f"{name} view_index must be a permutation of 0..{len(df)-1}"
            )

    report = {
        "status": "RTDETR_REPRO_PREFLIGHT_PASS",
        "reference": str(REFERENCE),
        "assets": assets,
        "historical_step10_bridge": {
            "path": str(bridge),
            "sha256": bridge_sha,
            "expected_sha256": frozen_bridge_sha,
            "match": bridge_match,
        },
        "manifests": {
            "geometry_fit": {
                "sha256": fit_sha,
                "expected_sha256": frozen_fit_sha,
                "match": (fit_sha == frozen_fit_sha) if frozen_fit_sha else None,
                "sources": 120,
                "views": 360,
            },
            "confirm": {
                "sha256": sha256_file(confirm),
                "sources": 200,
                "views": 600,
            },
        },
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
