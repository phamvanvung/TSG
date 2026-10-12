#!/usr/bin/env python
"""Fit the frozen teacher-canonical YOLO P4 maps from TRAIN checkpoints/views."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from tsg_repro.config import get_path, load_config
from tsg_repro.yolo_workflow import fit_teacher_canonical_maps_streaming


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", default=None)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    cfg, base = load_config(args.config)
    output = (
        Path(args.output).expanduser().resolve()
        if args.output
        else get_path(cfg, base, "artifacts", "canonical_map_dir")
    )
    report = fit_teacher_canonical_maps_streaming(
        get_path(cfg, base, "data", "gaussian_manifest"),
        get_path(cfg, base, "checkpoints", "yolo11s_clean_teacher"),
        get_path(cfg, base, "checkpoints", "yolo11s_none"),
        get_path(cfg, base, "checkpoints", "yolo11s_clean_kd"),
        get_path(cfg, base, "data", "yolo_none_args_yaml"),
        output,
        device_name=args.device,
        checkpoint_companion=get_path(
            cfg, base, "checkpoints", "yolo_checkpoint_companion", required=False
        ),
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
