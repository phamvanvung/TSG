#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tsg_repro.geometry import build_task_geometry  # noqa: E402
from tsg_repro.hashing import sha256_file  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--gradients", required=True, help=".npy [N,D] normalized internally")
    p.add_argument("--out", required=True)
    p.add_argument("--channels", type=int, default=256)
    a = p.parse_args()

    g = np.load(a.gradients, allow_pickle=False)
    if g.ndim != 2 or g.shape[1] != a.channels:
        raise SystemExit(f"expected gradients [N,{a.channels}], got {g.shape}")
    c = build_task_geometry(g)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.save(out, c)
    print(json.dumps({
        "status": "PASS",
        "rows": int(g.shape[0]),
        "channels": int(g.shape[1]),
        "trace": float(np.trace(c)),
        "symmetry_error": float(np.linalg.norm(c-c.T, ord="fro")),
        "geometry_sha256": sha256_file(out),
    }, indent=2))


if __name__ == "__main__":
    main()
