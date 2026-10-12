# Migration from the private experiment tree

The private experiment tree contains scientifically useful history, but the public repository should expose one authoritative path only.

## Old pattern to remove

Any code that does this is invalid when `n` and `t` come from different learned feature coordinates:

```python
alpha = dot(n, t) / dot(t, t)
n_c = n - alpha * t
```

## Canonical flattened-region pattern

```python
n_t_cells = none_cells @ R_NT
n_t = n_t_cells.reshape(-1)
t = teacher_cells.reshape(-1)
alpha = dot(n_t, t) / dot(t, t)
c_t = (n_t - alpha * t).reshape(-1, 256)
c_s = c_t @ R_ST.T
```

## Canonical per-cell pattern

```python
n_t = none_cells @ R_NT
alpha_i = sum(n_t * teacher_cells, axis=1) / sum(teacher_cells**2, axis=1)
c_t = n_t - alpha_i[:, None] * teacher_cells
c_s = c_t @ R_ST.T
```

The public package implements both patterns centrally. Model-specific scripts should never hand-code them again.

## v1.0-rc1 — Experiment-17 Step-12/14 closure

The remaining RT-DETR-L chain is now represented by:

```text
historical Step 12 -> scripts/rtdetr/03_run_local_utility_confirmation.py
historical Step 14 -> scripts/rtdetr/04_run_finite_horizon_confirmation.py
final canonical TSG -> scripts/rtdetr/05_run_orientation_confirmation.py
```

Step 12/14 intentionally preserve the historical Step-10 NONE→CLEAN-KD bridge. The later teacher-canonical correction is used by the final TSG orientation chain and is not retroactively substituted into the frozen Step-12/14 result label.
