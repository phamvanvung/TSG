# Frozen native-loss replay contracts

The task-sensitive-geometry results depend on gradients and finite interventions under the detector's **native** loss. These rules are part of the scientific protocol.

## YOLO11s

1. Load the frozen CLEAN-KD checkpoint and training `args.yaml`.
2. Set `model.args` to the frozen training arguments and reset `model.criterion`.
3. Put the complete model in eval mode and freeze every parameter.
4. Set only the Detect container's `training` attribute to `True`, so the native criterion receives raw head output.
5. Immediately put every Detect child back into eval mode.
6. Require every BatchNorm module to remain in eval mode.
7. For a task gradient, replace module-19 P4 with an identical detached leaf tensor and differentiate native loss only with respect to that leaf.

Calling `Detect.train()` is forbidden because it recursively changes child BatchNorm state.

## RT-DETR-L

1. Load the frozen custom checkpoint and companion; keep the model in eval mode and freeze parameters.
2. If the standalone checkpoint has `criterion=None`, initialize it exactly once with `model.init_criterion()` and require it to be callable.
3. Use `model.loss(batch)` explicitly.
4. For every view, derive the replay seed from SHA-256 of:

```text
ConsRestore|Experiment17|Step11RTDETRLossReplay|view={view_index}
```

using the first four digest bytes as an unsigned big-endian integer.
5. Reset Python, NumPy, CPU Torch, and CUDA Torch RNG state to that same seed before each plain, gradient, or counterfactual loss call.
6. Interventions replace/add only the frozen module-24 P4 representation; no parameter gradient or optimizer step is permitted.

This per-view seed reset is essential because RT-DETR native loss contains stochastic matching/denoising behavior even in an otherwise frozen replay.


## Coordinate arithmetic for checkpoint interventions

Teacher-canonical maps are loaded as `float32` tensors on the replay device.
The N→T map, per-cell projection used by TSG/TSOG, and T→S map are therefore
computed in the same float32 tensor arithmetic family as the frozen P4
representations. Float64 is used only for downstream summary statistics such
as `Q_M`, norms, correlations, and bootstrap aggregation.

If a frozen view manifest contains `view_index`, all checkpoint geometry fits
and confirmations preserve ascending `view_index` order. This is relevant for
last-bit reproducibility of accumulated second moments and Procrustes cross
products.

## RT-DETR Step-12/14 historical-coordinate contract

The frozen local-utility/finite-horizon chain predates the teacher-canonical coordinate audit and is therefore reproduced without redefining its direction:

```text
n_E = ((n^T t)/(t^T t)) t
n_C = n - n_E
m   = n_C @ R_step10_none_to_clean_kd
```

Step 12 uses exact single-cell replacements `s→t` and `s→s+m`. Step 14 freezes selected cells from Step-12 `u_rel` before model access, then evaluates `s+0.25m` and `s+4m`. Its **primary predictor is the frozen Step-12 `u_C`**. Recomputed values are diagnostic only because tiny FP32/CUDA replay differences must not alter frozen selection or the primary statistic.

Teacher-canonical `R_NT/R_ST` remains authoritative for the later TSG orientation chain; it is not substituted into Step 12/14 under the same result label.
