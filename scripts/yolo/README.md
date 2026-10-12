# YOLO11s frozen-checkpoint/native-geometry reproduction — v1.0-rc2

The public YOLO path is independent of the private experiment directory tree.

## Ordered stages

1. `00_preflight.py` — frozen checkpoint, manifest, and companion SHA-256 checks.
2. `01_audit_gaussian_manifest.py` — exact deterministic Gaussian-view audit.
3. `02_evaluate_frozen_checkpoints.py` — NONE/CLEAN-KD Gaussian VAL evaluation.
4. `03_fit_canonical_maps_from_checkpoints.py` — TRAIN-only `R_NT`/`R_ST` Procrustes fit.
5. `04_build_object_cohort.py` — native one-to-one VAL object matching.
6. `05_reproduce_canonical_ablation.py` — canonical flattened-region complement ablation.
7. `06_reproduce_canonical_targeted_rescue.py` — canonical TARGETED rescue.
8. `06b_reproduce_canonical_random_control.py` — exact targeted-vs-random Gate-2 **only when the hash-locked historical Step-07 helper is supplied**.
9. `07_fit_task_sensitive_geometry.py` — native CLEAN-KD task-gradient `C_g` fit on 120/360 BASIS_FIT.
10. `08_run_exp15a_confirmation.py` — teacher-canonical Exp15A on the independent 120/360 confirmation cohort.
11. `09_run_exp16a_tsog.py` — independent 120/360 mapped-complement replication and teacher-residual direction-family test.

Run through Exp16A:

```bash
python scripts/yolo/run_yolo_reproduction.py \
  --config configs/paths.json \
  --through exp16a
```

Add exact Gate-2 random control:

```bash
python scripts/yolo/run_yolo_reproduction.py \
  --config configs/paths.json \
  --through exp16a \
  --include-random-control
```

## Locked geometry

Experiment-4 uses cellwise N→T followed by one **flattened-region** projection. Exp15A/16A use **per-cell** teacher-canonical directions at unique spatial CORE9 object-center cells.

## Locked native-loss behavior

The CLEAN-KD model is frozen. Native loss requires raw Detect output, so only the Detect container flag is set to `training=True`; all child modules and BatchNorm layers remain eval. See `docs/NATIVE_LOSS_CONTRACT.md`.
