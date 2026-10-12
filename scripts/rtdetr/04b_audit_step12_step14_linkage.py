#!/usr/bin/env python
"""Audit semantic linkage between public RT-DETR Step-12 and Step-14 inputs.

A byte-identical CSV is accepted immediately. When a completed historical
Step-14 run records a different Step-12 CSV hash, this audit compares the
public Step-12 table with the configured historical Step-12 table and verifies
that every quantity that defines the Step-14 primary analysis is unchanged:

* identical cell population keyed by view/source/sigma/P4 location;
* identical top-25% selections by descending u_rel with (p4_y,p4_x) ties;
* identical within-selected-cell u_C ranks (the Step-14 primary predictor);
* identical frozen Step-12 exact intervention outcomes Delta_T, Delta_C, Y
  to numerical tolerance.

Differences in diagnostic columns, floating-point score magnitudes that do not
change selection/ranks, and CSV serialization do not invalidate scientific
linkage.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
LOCAL_CELLS = ROOT / "results/rtdetr_local_utility_confirmation/RTDETR_CONFIRM_CELL_RECORDS.csv"
STEP14_RESULT = ROOT / "results/rtdetr_finite_horizon_confirmation/RTDETR_FINITE_HORIZON_CONFIRMATION_RESULT.json"
OUT = ROOT / "results/rtdetr_step12_step14_linkage_audit.json"

KEYS = ["view_index", "source_stem", "sigma", "p4_y", "p4_x"]
SELECTION_COLUMNS = {"view_index", "p4_y", "p4_x", "u_rel", "u_C"}
OUTCOME_COLUMNS = ["Delta_T", "Delta_C", "Y"]
OUTCOME_ATOL = 1e-12


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def load_json(path: Path) -> dict:
    obj = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(obj, dict):
        raise RuntimeError(f"Expected JSON object: {path}")
    return obj


def resolve_config_path(raw: str, config_path: Path) -> Path:
    p = Path(str(raw)).expanduser()
    if not p.is_absolute():
        p = config_path.parent / p
    return p.resolve()


def selection(df: pd.DataFrame) -> dict[int, list[tuple[int, int]]]:
    out: dict[int, list[tuple[int, int]]] = {}
    for v, g in df.groupby("view_index", sort=True):
        ordered = g.sort_values(
            ["u_rel", "p4_y", "p4_x"],
            ascending=[False, True, True],
            kind="mergesort",
        ).reset_index(drop=True)
        if len(ordered) < 4:
            out[int(v)] = []
            continue
        k = int(np.floor(0.25 * len(ordered)))
        chosen = ordered.iloc[:k]
        out[int(v)] = [
            (int(r.p4_y), int(r.p4_x))
            for r in chosen.itertuples(index=False)
        ]
    return out


def selected_rank_audit(a: pd.DataFrame, b: pd.DataFrame, selected: dict[int, list[tuple[int, int]]]):
    changed = []
    max_abs = 0.0
    for v, cells in selected.items():
        if len(cells) < 2:
            continue
        aa = a[a["view_index"] == v].set_index(["p4_y", "p4_x"])
        bb = b[b["view_index"] == v].set_index(["p4_y", "p4_x"])
        ua = np.asarray([float(aa.loc[cell, "u_C"]) for cell in cells], dtype=np.float64)
        ub = np.asarray([float(bb.loc[cell, "u_C"]) for cell in cells], dtype=np.float64)
        if len(ua):
            max_abs = max(max_abs, float(np.max(np.abs(ua - ub))))
        ra = pd.Series(ua).rank(method="average").to_numpy()
        rb = pd.Series(ub).rank(method="average").to_numpy()
        if not np.array_equal(ra, rb):
            changed.append(int(v))
    return changed, max_abs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--output", default=str(OUT))
    args = ap.parse_args()

    config_path = Path(args.config).expanduser().resolve()
    config = load_json(config_path)
    local = LOCAL_CELLS.resolve()
    step14_path = STEP14_RESULT.resolve()
    output = Path(args.output).expanduser().resolve()

    if not local.is_file():
        raise FileNotFoundError(local)
    if not step14_path.is_file():
        raise FileNotFoundError(step14_path)

    step14 = load_json(step14_path)
    consumed_sha = str(step14["provenance"]["step12_cell_records_sha256"])
    local_sha = sha256_file(local)

    result = {
        "status": None,
        "mode": None,
        "public_step12_path": str(local),
        "public_step12_sha256": local_sha,
        "step14_recorded_input_sha256": consumed_sha,
        "exact_byte_linkage": local_sha == consumed_sha,
    }

    if local_sha == consumed_sha:
        result.update({
            "status": "PASS",
            "mode": "EXACT_BYTE_LINKAGE",
            "scientifically_equivalent_for_step14": True,
            "selection_changed_views": 0,
            "selected_u_C_rank_changed_views": 0,
        })
    else:
        raw_hist = config.get("artifacts", {}).get("rtdetr_step12_cell_records")
        if not raw_hist:
            raise RuntimeError(
                "Step-14 did not consume the public Step-12 CSV and config is missing "
                "artifacts.rtdetr_step12_cell_records for semantic linkage audit."
            )
        hist = resolve_config_path(str(raw_hist), config_path)
        if not hist.is_file():
            raise FileNotFoundError(hist)
        hist_sha = sha256_file(hist)
        if hist_sha != consumed_sha:
            raise RuntimeError(
                "Configured historical Step-12 CSV does not match the input recorded by Step 14.\n"
                f"Step14 recorded: {consumed_sha}\nConfigured file: {hist_sha}"
            )

        a = pd.read_csv(local)
        b = pd.read_csv(hist)
        required_columns = set(KEYS) | SELECTION_COLUMNS | set(OUTCOME_COLUMNS)
        missing_a = required_columns - set(a.columns)
        missing_b = required_columns - set(b.columns)
        if missing_a:
            raise RuntimeError(f"Public Step-12 table missing columns: {sorted(missing_a)}")
        if missing_b:
            raise RuntimeError(f"Historical Step-12 table missing columns: {sorted(missing_b)}")
        if a.duplicated(KEYS).any() or b.duplicated(KEYS).any():
            raise RuntimeError("Duplicate scientific cell keys found.")

        ak = set(map(tuple, a[KEYS].itertuples(index=False, name=None)))
        bk = set(map(tuple, b[KEYS].itertuples(index=False, name=None)))
        keys_equal = ak == bk

        merged = a.merge(b, on=KEYS, suffixes=("_public", "_historical"), validate="one_to_one") if keys_equal else None
        outcome_max_abs = {}
        outcomes_equal = bool(keys_equal)
        if merged is not None:
            for c in OUTCOME_COLUMNS:
                x = pd.to_numeric(merged[f"{c}_public"], errors="raise").to_numpy(np.float64)
                y = pd.to_numeric(merged[f"{c}_historical"], errors="raise").to_numpy(np.float64)
                d = np.abs(x - y)
                m = float(d.max()) if len(d) else 0.0
                outcome_max_abs[c] = m
                outcomes_equal = outcomes_equal and bool(np.allclose(x, y, rtol=0.0, atol=OUTCOME_ATOL))

        sa = selection(a) if keys_equal else {}
        sb = selection(b) if keys_equal else {}
        views = sorted(set(sa) | set(sb))
        selection_changed = [int(v) for v in views if sa.get(v, []) != sb.get(v, [])]
        rank_changed, max_selected_uc = selected_rank_audit(a, b, sa) if keys_equal and not selection_changed else ([], float("nan"))

        score_diffs = {}
        if merged is not None:
            for c in ("u_T", "u_C", "u_rel"):
                if c in a.columns and c in b.columns:
                    x = pd.to_numeric(merged[f"{c}_public"], errors="raise").to_numpy(np.float64)
                    y = pd.to_numeric(merged[f"{c}_historical"], errors="raise").to_numpy(np.float64)
                    score_diffs[c] = float(np.max(np.abs(x - y))) if len(x) else 0.0

        passed = bool(keys_equal and outcomes_equal and not selection_changed and not rank_changed)
        result.update({
            "status": "PASS" if passed else "FAIL",
            "mode": "SEMANTIC_STEP14_LINKAGE",
            "scientifically_equivalent_for_step14": passed,
            "historical_step12_path": str(hist),
            "historical_step12_sha256": hist_sha,
            "public_rows": int(len(a)),
            "historical_rows": int(len(b)),
            "public_columns": int(len(a.columns)),
            "historical_columns": int(len(b.columns)),
            "cell_key_sets_equal": bool(keys_equal),
            "public_only_cell_keys": int(len(ak - bk)),
            "historical_only_cell_keys": int(len(bk - ak)),
            "views_compared": int(len(views)),
            "selection_changed_views": int(len(selection_changed)),
            "selection_changed_view_examples": selection_changed[:20],
            "selected_u_C_rank_changed_views": int(len(rank_changed)),
            "selected_u_C_rank_changed_view_examples": rank_changed[:20],
            "max_selected_u_C_abs_difference": max_selected_uc,
            "outcome_tolerance": OUTCOME_ATOL,
            "outcome_max_abs_differences": outcome_max_abs,
            "step12_score_max_abs_differences_descriptive": score_diffs,
            "interpretation": (
                "CSV bytes differ, but the cell population, Step-14 top-25% selection, "
                "selected-cell u_C ranks, and frozen exact Step-12 intervention outcomes "
                "are unchanged. The tables are scientifically equivalent for Step 14."
                if passed else
                "The public and historical Step-12 tables are not scientifically equivalent for Step 14."
            ),
        })

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, allow_nan=False))
    raise SystemExit(0 if result["status"] == "PASS" else 2)


if __name__ == "__main__":
    main()
