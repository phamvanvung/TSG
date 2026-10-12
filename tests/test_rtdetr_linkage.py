from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/rtdetr/04b_audit_step12_step14_linkage.py"
spec = importlib.util.spec_from_file_location("rtdetr_linkage_audit", SCRIPT)
mod = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(mod)


def frame(delta=0.0):
    rows = []
    for view in (0, 1):
        for i, u in enumerate([4.0, 3.0, 2.0, 1.0]):
            rows.append({
                "view_index": view,
                "source_stem": f"s{view}",
                "sigma": 15,
                "p4_y": i,
                "p4_x": 0,
                "u_rel": u + delta * (i + 1),
                "u_C": 10.0 + u + delta * (i + 1),
                "Delta_T": 0.1 * i,
                "Delta_C": 0.2 * i,
                "Y": 0.1 * i,
            })
    return pd.DataFrame(rows)


def test_selection_invariant_to_non_rank_changing_fp_drift():
    a = frame(0.0)
    b = frame(1e-7)
    assert mod.selection(a) == mod.selection(b)
    changed, _ = mod.selected_rank_audit(a, b, mod.selection(a))
    assert changed == []


def test_selection_detects_rank_change():
    a = frame(0.0)
    b = frame(0.0)
    mask = (b.view_index == 0) & (b.p4_y == 3)
    b.loc[mask, "u_rel"] = 100.0
    assert mod.selection(a) != mod.selection(b)
