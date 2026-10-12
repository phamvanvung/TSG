# Canonical cell bundle

For `scripts/03_compute_canonical_complement.py`, use an NPZ with:

- `none_cells`: `[K, 256]`, raw NONE-space P4 cell vectors.
- `teacher_cells`: `[K, 256]`, paired clean-teacher P4 cell vectors.

Use `--mode flattened-region` for the Experiment-4/5 family. The script maps each NONE cell to teacher coordinates and computes one projection coefficient over the flattened `K x 256` region.

Use `--mode per-cell` for the task-sensitive-geometry family. Each P4 cell receives its own 256-D projection coefficient before the complement is mapped teacher→CLEAN-KD.
