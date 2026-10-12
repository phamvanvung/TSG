# Code architecture

The public code separates mathematical invariants from detector-specific replay.

## Scientific kernel

- `mapping.py` — row-vector orthogonal maps and teacher-canonical coordinate transforms.
- `projection.py` — flattened-region versus per-cell decomposition.
- `native_geometry.py` — canonical per-cell mapped complement, teacher residual, support tensors, task-sensitive orientation.
- `geometry.py` — artifact-level `C_g` construction/orientation helpers.
- `statistics.py` — Spearman, partial Spearman via rank residualization, and source-cluster bootstrap.
- `local_utility.py` — frozen RT-DETR Step-12/14 rank conventions, source aggregation, selection freeze, and historical mapped-complement helper.
- `data.py` / `geometry_workflow.py` — manifest contracts and confirmation-table orchestration.

## Detector replay

- `yolo_runtime.py` — frozen YOLO dataset, inference, direct Detect-head replay, and native-loss configuration.
- `rtdetr_runtime.py` — frozen RT-DETR dataset/model loading.
- `native_model.py` — shared native loss, P4 leaf-gradient/intervention hooks, unique spatial CORE9 support, and RT per-view RNG seeding.

## Drivers

- `scripts/yolo/` — YOLO checkpoint reproduction through Exp16A.
- `scripts/rtdetr/` — RT-DETR teacher-canonical/native-geometry reproduction plus historical Step-12/14 local-utility and finite-horizon replay.

Drivers validate inputs, call shared scientific/runtime APIs, and serialize results. Coordinate formulas and statistical definitions should not be duplicated in experiment scripts.
