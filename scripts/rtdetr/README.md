# RT-DETR-L end-to-end checkpoint reproduction — v1.0.0-rc2

Ordered stages:

1. `00_preflight.py` — SHA-lock checkpoints, custom companion, manifests, and the historical Step-10 NONE→CLEAN-KD bridge.
2. `01_fit_canonical_maps.py` — fit teacher-canonical `R_NT/R_ST` on `JOURNAL_GEOMETRY_FIT`.
3. `02_fit_task_sensitive_geometry.py` — fit native-gradient `C_g` from CLEAN-KD only.
4. `03_run_local_utility_confirmation.py` — reproduce frozen Step-12 local task utility on 200 sources / 600 views.
5. `04_run_finite_horizon_confirmation.py` — freeze Step-14 selection from Step-12 records, then reproduce alpha=.25 vs alpha=4 finite-horizon outcomes using frozen Step-12 `u_C` as the primary predictor.
6. `05_run_orientation_confirmation.py` — reproduce the final teacher-canonical mapped-complement orientation confirmation.

Run through the complete chain:

```bash
python scripts/rtdetr/run_rtdetr_reproduction.py \
  --config configs/paths.json \
  --through orientation
```

The Step-12/14 chain intentionally uses the **historical frozen Step-10 bridge**, while the final orientation stage uses the later teacher-canonical correction. They answer different frozen scientific questions and are not silently merged.

The native RT-DETR contract remains: eval/frozen model, lazy criterion initialized once, explicit `model.loss(batch)`, and the same SHA-derived per-view RNG seed reset before every baseline/intervention loss call.
