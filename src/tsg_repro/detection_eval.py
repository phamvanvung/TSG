"""Architecture-independent CORE9 AP reconstruction and paired source bootstrap.

This module mirrors the frozen Experiment-5 evaluation contract used by the
RT-DETR-L journal replication: AP is reconstructed from per-image validator
statistics, the fixed CORE9 class set is never changed, and bootstrap draws
that omit any CORE9 target class are rejected rather than imputed.
"""
from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd

CORE9_IDS = (0, 1, 2, 3, 4, 5, 6, 7, 9)
SIGMAS = (15, 25, 50)
BOOTSTRAP_REPS = 5000
BOOTSTRAP_SEED = 2027


def _concat(records: Iterable[dict], key: str, *, shape_tail=()) -> np.ndarray:
    vals = [np.asarray(r[key]) for r in records]
    if not vals:
        if shape_tail:
            return np.empty((0, *shape_tail))
        return np.empty((0,))
    return np.concatenate(vals, axis=0)


def core9_ap_from_records(records: list[dict]) -> dict | None:
    """Reconstruct fixed-CORE9 AP from cached validator records.

    Returns ``None`` when a resampled source draw is missing any CORE9 target
    class. That is the frozen bootstrap policy: reject and resample rather than
    changing the class set or imputing an AP value.
    """
    if not records:
        return None

    tp = _concat(records, "tp", shape_tail=(10,)).astype(bool, copy=False)
    conf = _concat(records, "conf").astype(np.float64, copy=False)
    pred_cls = _concat(records, "pred_cls").astype(np.int64, copy=False)
    target_cls = _concat(records, "target_cls").astype(np.int64, copy=False)

    target_ids = set(int(x) for x in np.unique(target_cls))
    if not set(CORE9_IDS).issubset(target_ids):
        return None
    if tp.ndim != 2 or tp.shape[1] != 10:
        raise RuntimeError(f"Expected tp shape (N,10), found {tp.shape}")
    if len(conf) != len(pred_cls) or len(conf) != len(tp):
        raise RuntimeError("Cached prediction arrays have inconsistent lengths")

    # Ultralytics 8.4.131 frozen AP implementation.  The returned AP matrix is
    # (number of target classes, 10 IoU thresholds). With the BDD100K VAL
    # cohort, train/class-8 has no target support, so the unique target classes
    # are exactly CORE9.  The same requirement is enforced for bootstrap draws.
    from ultralytics.utils.metrics import ap_per_class

    result = ap_per_class(
        tp,
        conf,
        pred_cls,
        target_cls,
        plot=False,
        names={i: str(i) for i in range(10)},
    )
    if len(result) < 7:
        raise RuntimeError("Unexpected Ultralytics ap_per_class return contract")
    ap = np.asarray(result[5], dtype=np.float64)
    unique_classes = tuple(int(x) for x in np.asarray(result[6]).reshape(-1))
    if unique_classes != CORE9_IDS:
        return None
    if ap.shape != (len(CORE9_IDS), 10):
        raise RuntimeError(f"Unexpected AP matrix shape: {ap.shape}")

    return {
        "map50_95": float(ap.mean()),
        "map50": float(ap[:, 0].mean()),
        "ap": ap,
        "classes": unique_classes,
    }


def native_core9(metrics) -> dict:
    """Extract the fixed-CORE9 native metric from an Ultralytics result."""
    ids = tuple(int(x) for x in np.asarray(metrics.box.ap_class_index).reshape(-1))
    if ids != CORE9_IDS:
        raise RuntimeError(
            "Native AP class-index mismatch. "
            f"Expected {CORE9_IDS}, found {ids}"
        )
    ap = np.asarray(metrics.box.all_ap, dtype=np.float64)
    if ap.shape != (len(CORE9_IDS), 10):
        raise RuntimeError(f"Unexpected native AP matrix shape: {ap.shape}")
    return {
        "core9": {
            "map50_95": float(ap.mean()),
            "map50": float(ap[:, 0].mean()),
        },
        "ap_class_index": list(ids),
    }


def severity_macro(per_sigma: dict, metric_name: str = "map50_95") -> float:
    return float(np.mean([
        float(per_sigma[str(s)]["core9"][metric_name]) for s in SIGMAS
    ]))


def records_by_source(records: list[dict]) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for r in records:
        source = str(r["source_stem"])
        if source in out:
            raise RuntimeError(f"Duplicate record for source {source}")
        out[source] = r
    return out


def bootstrap_primary_delta(
    caches: dict[str, dict[int, dict[str, dict]]],
    source_order: list[str],
    *,
    comparator: str = "NONE",
    candidate: str = "CLEAN-KD",
    reps: int = BOOTSTRAP_REPS,
    seed: int = BOOTSTRAP_SEED,
) -> tuple[pd.DataFrame, dict]:
    """Frozen paired 250-source bootstrap with fixed CORE9 support."""
    rng = np.random.default_rng(int(seed))
    rows: list[dict] = []
    rejected = 0
    draws = 0
    n = len(source_order)
    if n == 0:
        raise RuntimeError("Empty source order")

    while len(rows) < int(reps):
        draws += 1
        idx = rng.integers(0, n, size=n)
        sampled_sources = [source_order[int(i)] for i in idx]
        row: dict[str, float] = {}
        macros: dict[str, float] = {}
        valid = True

        for arm in (comparator, candidate):
            per_sigma: dict[int, float] = {}
            for sigma in SIGMAS:
                records = [caches[arm][sigma][source] for source in sampled_sources]
                value = core9_ap_from_records(records)
                if value is None:
                    valid = False
                    break
                per_sigma[sigma] = float(value["map50_95"])
                row[f"{arm}_sigma{sigma}"] = per_sigma[sigma]
            if not valid:
                break
            macros[arm] = float(np.mean([per_sigma[s] for s in SIGMAS]))
            row[f"{arm}_macro"] = macros[arm]

        if not valid:
            rejected += 1
            continue

        for sigma in SIGMAS:
            row[f"delta_sigma{sigma}"] = (
                row[f"{candidate}_sigma{sigma}"] - row[f"{comparator}_sigma{sigma}"]
            )
        row["delta_macro"] = macros[candidate] - macros[comparator]
        rows.append(row)

    df = pd.DataFrame(rows)
    deltas = df["delta_macro"].to_numpy(dtype=np.float64)
    ci_low, ci_high = np.quantile(deltas, [0.025, 0.975])
    return df, {
        "ci_low": float(ci_low),
        "ci_high": float(ci_high),
        "valid_replicates": int(len(df)),
        "rejected_draws_missing_core9_support": int(rejected),
        "total_source_resample_draws": int(draws),
    }
