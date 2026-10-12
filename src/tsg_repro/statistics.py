from __future__ import annotations

from typing import Callable

import numpy as np
from scipy.stats import rankdata, spearmanr


Array = np.ndarray


def partial_spearman(x: Array, y: Array, z: Array) -> float:
    """Partial Spearman rho(x,y|z) via rank residualization."""
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    z = np.asarray(z, dtype=np.float64).reshape(-1)
    if not (len(x) == len(y) == len(z)):
        raise ValueError("x, y, z lengths differ")
    good = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
    x, y, z = x[good], y[good], z[good]
    if len(x) < 4:
        return float("nan")
    rx, ry, rz = rankdata(x), rankdata(y), rankdata(z)
    a = np.column_stack([np.ones(len(rz)), rz])
    bx, *_ = np.linalg.lstsq(a, rx, rcond=None)
    by, *_ = np.linalg.lstsq(a, ry, rcond=None)
    ex = rx - a @ bx
    ey = ry - a @ by
    sx = float(np.std(ex))
    sy = float(np.std(ey))
    if sx == 0 or sy == 0:
        return float("nan")
    return float(np.corrcoef(ex, ey)[0, 1])


def spearman(x: Array, y: Array) -> float:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    good = np.isfinite(x) & np.isfinite(y)
    if good.sum() < 3:
        return float("nan")
    return float(spearmanr(x[good], y[good]).statistic)


def cluster_bootstrap(
    values: dict[str, Array],
    clusters: Array,
    statistic: Callable[[dict[str, Array]], float],
    *,
    reps: int = 5000,
    seed: int = 2027,
    ci: tuple[float, float] = (2.5, 97.5),
) -> dict:
    """Source-cluster bootstrap preserving all rows within sampled clusters."""
    clusters = np.asarray(clusters)
    n = len(clusters)
    if any(len(np.asarray(v)) != n for v in values.values()):
        raise ValueError("all values must have the same row count as clusters")
    unique = np.unique(clusters)
    groups = {g: np.flatnonzero(clusters == g) for g in unique}
    rng = np.random.default_rng(seed)
    boots = np.empty(reps, dtype=np.float64)

    for b in range(reps):
        sampled = rng.choice(unique, size=len(unique), replace=True)
        idx = np.concatenate([groups[g] for g in sampled])
        sample = {k: np.asarray(v)[idx] for k, v in values.items()}
        boots[b] = statistic(sample)

    finite = boots[np.isfinite(boots)]
    if len(finite) == 0:
        raise ValueError("all bootstrap statistics are non-finite")
    lo, hi = np.percentile(finite, ci)
    point = statistic(values)
    return {
        "point": float(point),
        "ci_low": float(lo),
        "ci_high": float(hi),
        "reps": int(reps),
        "seed": int(seed),
        "clusters": int(len(unique)),
        "finite_reps": int(len(finite)),
    }
