"""Frozen RT-DETR Step-12 local-utility and Step-14 finite-horizon statistics."""
from __future__ import annotations

import math
from typing import Iterable

import numpy as np
import pandas as pd


def _rank_average(x) -> np.ndarray:
    return pd.Series(np.asarray(x, dtype=np.float64)).rank(method="average").to_numpy(dtype=np.float64)


def spearman_zero_constant(x, y) -> float:
    """Step-12 convention: average-rank Spearman; constant vector -> 0.0."""
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    good = np.isfinite(x) & np.isfinite(y)
    x, y = x[good], y[good]
    if len(x) < 2:
        return float("nan")
    rx, ry = _rank_average(x), _rank_average(y)
    if float(np.std(rx)) <= 0.0 or float(np.std(ry)) <= 0.0:
        return 0.0
    return float(np.corrcoef(rx, ry)[0, 1])


def spearman_nan_constant(x, y) -> float:
    """Step-14 convention: average-rank Spearman; constant vector -> NaN."""
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    good = np.isfinite(x) & np.isfinite(y)
    x, y = x[good], y[good]
    if len(x) < 2:
        return float("nan")
    rx, ry = _rank_average(x), _rank_average(y)
    if float(np.std(rx)) <= 0.0 or float(np.std(ry)) <= 0.0:
        return float("nan")
    rho = float(np.corrcoef(rx, ry)[0, 1])
    return rho if np.isfinite(rho) else float("nan")


def _bootstrap_mean(values, *, reps: int = 5000, seed: int = 2027) -> dict:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    values = values[np.isfinite(values)]
    if len(values) < 1:
        raise ValueError("bootstrap requires at least one finite source statistic")
    rng = np.random.default_rng(seed)
    boot = np.empty(reps, dtype=np.float64)
    n = len(values)
    for b in range(reps):
        boot[b] = float(np.mean(values[rng.integers(0, n, size=n)]))
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return {
        "point": float(np.mean(values)),
        "ci95": [float(lo), float(hi)],
        "bootstrap_reps": int(reps),
        "bootstrap_seed": int(seed),
        "source_count": int(n),
    }


def step12_view_statistics(cell_records: pd.DataFrame, *, tail_fraction: float = 0.25) -> pd.DataFrame:
    """Build the frozen 600-view Step-12 Gate-A/Gate-B statistics."""
    required = {"view_index", "source_stem", "sigma", "p4_y", "p4_x", "u_rel", "Y", "Delta_T", "Delta_C", "u_T", "u_C"}
    missing = required - set(cell_records.columns)
    if missing:
        raise ValueError(f"Step-12 cell records missing columns: {sorted(missing)}")
    rows = []
    for (view_index, source, sigma), g in cell_records.groupby(["view_index", "source_stem", "sigma"], sort=True):
        g = g.sort_values(["p4_y", "p4_x"], kind="mergesort")
        n = len(g)
        gate_a_eligible = n >= 2
        rho_rel_y = spearman_zero_constant(g["u_rel"], g["Y"]) if gate_a_eligible else float("nan")
        rho_t_dt = spearman_zero_constant(g["u_T"], g["Delta_T"]) if gate_a_eligible else float("nan")
        rho_c_dc = spearman_zero_constant(g["u_C"], g["Delta_C"]) if gate_a_eligible else float("nan")
        gate_b_eligible = n >= 4
        tail_k = 0
        enrichment = float("nan")
        if gate_b_eligible:
            tail_k = max(1, int(math.floor(float(tail_fraction) * n)))
            ordered = g.sort_values(["u_rel", "p4_y", "p4_x"], ascending=[True, True, True], kind="mergesort")
            low = ordered.iloc[:tail_k]
            high = ordered.iloc[-tail_k:]
            enrichment = float(high["Y"].mean() - low["Y"].mean())
        rows.append({
            "view_index": int(view_index), "source_stem": str(source), "sigma": int(sigma), "n_cells": int(n),
            "gate_a_eligible": bool(gate_a_eligible), "rho_rel_y": rho_rel_y,
            "rho_uT_deltaT": rho_t_dt, "rho_uC_deltaC": rho_c_dc,
            "gate_b_eligible": bool(gate_b_eligible), "tail_k": int(tail_k),
            "enrichment_top_minus_bottom_Y": enrichment,
        })
    return pd.DataFrame(rows).sort_values("view_index", kind="mergesort").reset_index(drop=True)


def step12_source_statistics(view_stats: pd.DataFrame, *, expected_sigmas: tuple[int, ...] = (15, 25, 50)) -> pd.DataFrame:
    rows = []
    for source, g in view_stats.groupby("source_stem", sort=True):
        sigmas = tuple(sorted(g["sigma"].astype(int).tolist()))
        if sigmas != tuple(expected_sigmas):
            raise ValueError(f"source {source} does not contain exactly {expected_sigmas}: {sigmas}")
        a = g.loc[g["gate_a_eligible"], "rho_rel_y"].to_numpy(dtype=np.float64)
        b = g.loc[g["gate_b_eligible"], "enrichment_top_minus_bottom_Y"].to_numpy(dtype=np.float64)
        a = a[np.isfinite(a)]; b = b[np.isfinite(b)]
        rows.append({
            "source_stem": str(source),
            "gate_a_eligible_views": int(len(a)),
            "gate_a_source_stat": float(np.mean(a)) if len(a) else 0.0,
            "gate_b_eligible_views": int(len(b)),
            "gate_b_source_stat": float(np.mean(b)) if len(b) else float("nan"),
        })
    return pd.DataFrame(rows)


def step12_confirmation(cell_records: pd.DataFrame, *, reps: int = 5000, seed: int = 2027) -> dict:
    views = step12_view_statistics(cell_records)
    sources = step12_source_statistics(views)
    gate_a = _bootstrap_mean(sources["gate_a_source_stat"].to_numpy(dtype=np.float64), reps=reps, seed=seed)
    bvals = sources["gate_b_source_stat"].to_numpy(dtype=np.float64)
    gate_b = _bootstrap_mean(bvals[np.isfinite(bvals)], reps=reps, seed=seed)
    gate_a["pass"] = bool(gate_a["ci95"][0] > 0.0)
    gate_b["pass"] = bool(gate_b["ci95"][0] > 0.0)
    advance = bool(gate_a["pass"] and gate_b["pass"])
    return {"view_stats": views, "source_stats": sources, "gate_A": gate_a, "gate_B": gate_b, "advance": advance}


def freeze_step14_selection(step12_cells: pd.DataFrame, *, min_support: int = 4, top_fraction: float = 0.25) -> pd.DataFrame:
    """Freeze Step-14 selected cells from Step-12 outcomes before new model access."""
    rows = []
    for (view_index, source, sigma), g in step12_cells.groupby(["view_index", "source_stem", "sigma"], sort=True):
        n = len(g)
        if n < min_support:
            continue
        k = int(math.floor(top_fraction * n))
        if k < 1:
            raise RuntimeError("eligible finite-horizon view produced k<1")
        ordered = g.sort_values(["u_rel", "p4_y", "p4_x"], ascending=[False, True, True], kind="mergesort")
        for order, r in enumerate(ordered.iloc[:k].itertuples(index=False)):
            rows.append({
                "view_index": int(view_index), "source_stem": str(source), "sigma": int(sigma),
                "support_n": int(n), "selected_k": int(k), "selection_order": int(order),
                "class_id_representative": int(getattr(r, "class_id_representative", getattr(r, "class_id", -1))),
                "p4_y": int(r.p4_y), "p4_x": int(r.p4_x),
                "frozen_u_T": float(r.u_T), "frozen_u_C": float(r.u_C), "frozen_u_rel": float(r.u_rel),
            })
    return pd.DataFrame(rows)


def step14_confirmation(selected_outcomes: pd.DataFrame, all_views: pd.DataFrame, *, reps: int = 5000, seed: int = 2027, expected_sigmas: tuple[int, ...] = (15, 25, 50)) -> dict:
    """Rebuild Step-14 primary correlations centrally from frozen u_C + exact gains."""
    req = {"view_index", "source_stem", "sigma", "frozen_u_C", "Delta_small", "Delta_large"}
    missing = req - set(selected_outcomes.columns)
    if missing:
        raise ValueError(f"Step-14 selected outcomes missing: {sorted(missing)}")
    view_rows=[]
    grouped={int(v):g for v,g in selected_outcomes.groupby("view_index",sort=False)}
    for r in all_views.sort_values("view_index",kind="mergesort").itertuples(index=False):
        vi=int(r.view_index); g=grouped.get(vi)
        if g is None or len(g)<2:
            rs=rl=d=float("nan"); defined=False
        else:
            rs=spearman_nan_constant(g["frozen_u_C"],g["Delta_small"])
            rl=spearman_nan_constant(g["frozen_u_C"],g["Delta_large"])
            defined=bool(np.isfinite(rs) and np.isfinite(rl)); d=float(rs-rl) if defined else float("nan")
        view_rows.append({"view_index":vi,"source_stem":str(r.source_stem),"sigma":int(r.sigma),"selected_cells":0 if g is None else int(len(g)),"rho_small":rs,"rho_large":rl,"paired_difference":d,"primary_defined":defined})
    views=pd.DataFrame(view_rows)
    source_rows=[]
    for source,g in views.groupby("source_stem",sort=True):
        if tuple(sorted(g["sigma"].astype(int).tolist()))!=tuple(expected_sigmas):
            raise ValueError(f"source {source} severity set changed")
        d=g.loc[g["primary_defined"]]
        if len(d):
            rs=float(d["rho_small"].mean()); rl=float(d["rho_large"].mean()); dr=float(d["paired_difference"].mean())
        else: rs=rl=dr=0.0
        source_rows.append({"source_stem":str(source),"defined_views":int(len(d)),"rho_small_source":rs,"rho_large_source":rl,"delta_rho_source":dr,"conservative_zero_no_defined_view":bool(len(d)==0)})
    sources=pd.DataFrame(source_rows)
    primary=_bootstrap_mean(sources["delta_rho_source"].to_numpy(dtype=np.float64),reps=reps,seed=seed)
    rho_small=float(sources["rho_small_source"].mean()); rho_large=float(sources["rho_large_source"].mean())
    if abs((rho_small-rho_large)-primary["point"])>1e-10:
        raise RuntimeError("source-balanced rho difference does not equal Delta_rho")
    primary.update({"rho_small_source_balanced":rho_small,"rho_large_source_balanced":rho_large,"pass":bool(primary["point"]>0.0 and primary["ci95"][0]>0.0)})
    return {"view_results":views,"source_results":sources,"primary":primary}


def historical_mapped_complement_torch(n, t, bridge, *, eps: float = 1e-12):
    """Historical Experiment-17 Step-10/11 cell complement: n_C projected on raw t, then n_C @ R."""
    import torch
    n = n.float(); t = t.float(); bridge = bridge.float()
    denom = torch.dot(t, t)
    if (not torch.isfinite(denom)) or float(denom.detach().cpu()) <= eps:
        raise RuntimeError("teacher vector has insufficient energy for historical projection")
    coeff = torch.dot(n, t) / denom
    n_e = coeff * t
    n_c = n - n_e
    m = torch.matmul(n_c, bridge)
    if not torch.isfinite(m).all():
        raise RuntimeError("historical mapped complement contains non-finite values")
    return n_e, n_c, m
