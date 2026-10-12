import numpy as np

import tsg_repro.detection_eval as de


def test_records_by_source_rejects_duplicates():
    rows = [
        {"source_stem": "a"},
        {"source_stem": "b"},
    ]
    assert set(de.records_by_source(rows)) == {"a", "b"}


def test_bootstrap_primary_delta_pairing(monkeypatch):
    # Replace AP reconstruction with a deterministic scalar encoded in target_cls;
    # this tests source-paired resampling and candidate-minus-comparator arithmetic
    # without requiring Ultralytics in the source-validation container.
    def fake(records):
        vals = np.array([float(np.asarray(r["target_cls"])[0]) for r in records])
        return {"map50_95": float(vals.mean())}

    monkeypatch.setattr(de, "core9_ap_from_records", fake)
    sources = ["s0", "s1", "s2"]
    caches = {"NONE": {}, "CLEAN-KD": {}}
    for arm, offset in (("NONE", 0.0), ("CLEAN-KD", 1.0)):
        for sigma in de.SIGMAS:
            caches[arm][sigma] = {
                s: {"target_cls": np.array([i + offset], dtype=float)}
                for i, s in enumerate(sources)
            }
    df, summary = de.bootstrap_primary_delta(
        caches, sources, reps=20, seed=2027
    )
    assert len(df) == 20
    assert np.allclose(df["delta_macro"], 1.0)
    assert np.isclose(summary["ci_low"], 1.0)
    assert np.isclose(summary["ci_high"], 1.0)
    assert summary["rejected_draws_missing_core9_support"] == 0
