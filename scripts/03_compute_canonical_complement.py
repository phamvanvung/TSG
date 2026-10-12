#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tsg_repro.mapping import validate_map  # noqa: E402
from tsg_repro.projection import (  # noqa: E402
    decompose_flattened_region,
    decompose_per_cell,
    reconstruction_relative_error,
)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cells", required=True, help="NPZ containing none_cells and teacher_cells")
    p.add_argument("--r-nt", required=True)
    p.add_argument("--r-st", required=True)
    p.add_argument("--mode", choices=["flattened-region", "per-cell"], required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--channels", type=int, default=256)
    a = p.parse_args()

    z = np.load(a.cells, allow_pickle=False)
    n = z["none_cells"]
    t = z["teacher_cells"]
    r_nt = np.load(a.r_nt, allow_pickle=False)
    r_st = np.load(a.r_st, allow_pickle=False)
    validate_map(r_nt, channels=a.channels)
    validate_map(r_st, channels=a.channels)

    if a.mode == "flattened-region":
        result = decompose_flattened_region(n, t, r_nt, r_st)
    else:
        result = decompose_per_cell(n, t, r_nt, r_st)

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out,
        alpha=result.alpha,
        none_teacher=result.none_teacher,
        explained_teacher=result.explained_teacher,
        complement_teacher=result.complement_teacher,
        explained_student=result.explained_student,
        complement_student=result.complement_student,
    )
    report = {
        "mode": a.mode,
        "cells": int(n.shape[0]),
        "channels": int(n.shape[1]),
        "teacher_reconstruction_relative_error": reconstruction_relative_error(
            result.none_teacher, result.explained_teacher, result.complement_teacher
        ),
        "output": str(out.resolve()),
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
