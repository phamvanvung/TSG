#!/usr/bin/env python
"""Ordered YOLO11s frozen-checkpoint/native-geometry reproduction driver for v1.0-rc2.

The runner executes every migrated YOLO checkpoint-level stage from the
repository root so that relative result paths are deterministic. The current chain extends through native-gradient C_g fitting and Exp15A/Exp16A confirmation. The exact historical random control is available only through a separately supplied SHA-locked authoritative Step-07 helper.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO / "src"))

from tsg_repro.config import get_path, load_config

STAGES = [
    ("preflight", "00_preflight.py"),
    ("data-audit", "01_audit_gaussian_manifest.py"),
    ("evaluation", "02_evaluate_frozen_checkpoints.py"),
    ("maps", "03_fit_canonical_maps_from_checkpoints.py"),
    ("cohort", "04_build_object_cohort.py"),
    ("ablation", "05_reproduce_canonical_ablation.py"),
    ("targeted-rescue", "06_reproduce_canonical_targeted_rescue.py"),
    ("task-geometry", "07_fit_task_sensitive_geometry.py"),
    ("exp15a", "08_run_exp15a_confirmation.py"),
    ("exp16a", "09_run_exp16a_tsog.py"),
]


def run(cmd: list[str]) -> None:
    print(" ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, cwd=REPO)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--through", choices=[name for name, _ in STAGES], default="exp16a"
    )
    parser.add_argument(
        "--from-stage",
        choices=[name for name, _ in STAGES],
        default="preflight",
        help="Resume at this stage instead of rerunning earlier completed stages.",
    )
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument(
        "--skip-pixel-audit",
        action="store_true",
        help="Skip expensive stored-vs-regenerated Gaussian pixel audit.",
    )
    parser.add_argument("--include-random-control", action="store_true", help="Run exact Gate-2 random control using the SHA-locked artifacts.exp4_step07_helper.")
    args = parser.parse_args()

    config = Path(args.config).expanduser().resolve()
    cfg, base = load_config(config)
    map_dir = get_path(cfg, base, "artifacts", "canonical_map_dir")
    names = [name for name, _ in STAGES]
    start = names.index(args.from_stage)
    stop = names.index(args.through)
    if start > stop:
        parser.error("--from-stage must not come after --through")

    for name, script in STAGES[start : stop + 1]:
        print("\n" + "=" * 100, flush=True)
        print("STAGE:", name, flush=True)
        cmd = [args.python, str(HERE / script), "--config", str(config)]
        if name == "data-audit" and not args.skip_pixel_audit:
            cmd.append("--pixel-audit")
        run(cmd)
        if name == "targeted-rescue" and args.include_random_control:
            print("\nHASH-LOCKED EXPERIMENT-4 RANDOM CONTROL", flush=True)
            run([args.python, str(HERE / "06b_reproduce_canonical_random_control.py"), "--config", str(config)])

        if name == "maps":
            print("\nMAP HASH/FIREWALL VALIDATION", flush=True)
            run(
                [
                    args.python,
                    str(REPO / "scripts" / "02_validate_teacher_canonical_maps.py"),
                    "--audit-dir",
                    str(map_dir),
                ]
            )


if __name__ == "__main__":
    main()
