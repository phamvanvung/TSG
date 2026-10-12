#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tsg_repro.geometry import orientation_score  # noqa: E402
from tsg_repro.statistics import cluster_bootstrap, partial_spearman, spearman  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--directions", required=True, help=".npy [N,D]")
    p.add_argument("--geometry", required=True, help="C_g .npy [D,D]")
    p.add_argument("--table", required=True, help="CSV aligned to direction rows")
    p.add_argument("--utility-col", required=True)
    p.add_argument("--magnitude-col", required=True)
    p.add_argument("--cluster-col", default="source_stem")
    p.add_argument("--out", required=True)
    p.add_argument("--reps", type=int, default=5000)
    p.add_argument("--seed", type=int, default=2027)
    a = p.parse_args()

    d = np.load(a.directions, allow_pickle=False)
    c = np.load(a.geometry, allow_pickle=False)
    df = pd.read_csv(a.table)
    if len(df) != len(d):
        raise SystemExit(f"table/direction row mismatch: {len(df)} vs {len(d)}")

    q = orientation_score(d, c)
    y = pd.to_numeric(df[a.utility_col], errors="raise").to_numpy(float)
    mag = pd.to_numeric(df[a.magnitude_col], errors="raise").to_numpy(float)
    clusters = df[a.cluster_col].astype(str).to_numpy()

    point_partial = partial_spearman(q, y, mag)
    point_simple = spearman(q, y)
    boot = cluster_bootstrap(
        {"q": q, "y": y, "mag": mag},
        clusters,
        lambda s: partial_spearman(s["q"], s["y"], s["mag"]),
        reps=a.reps,
        seed=a.seed,
    )

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    result = {
        "status": "COMPLETE",
        "rows": int(len(df)),
        "source_clusters": int(pd.Series(clusters).nunique()),
        "spearman_q_utility": point_simple,
        "partial_spearman_q_utility_given_magnitude": point_partial,
        "source_cluster_bootstrap": boot,
    }
    out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
