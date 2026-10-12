"""Torch helpers for frozen native detector loss, P4 gradients, and interventions."""
from __future__ import annotations

import copy
import hashlib
import random
import numpy as np
import torch


def scalarize_native_loss(result):
    """Return a scalar native detector loss without assuming one Ultralytics head type."""
    x = result[0] if isinstance(result, (tuple, list)) else result
    if isinstance(x, dict):
        tensor_values = [v for v in x.values() if torch.is_tensor(v)]
        device = tensor_values[0].device if tensor_values else torch.device("cpu")
        vals = [
            v if torch.is_tensor(v) else torch.as_tensor(v, dtype=torch.float32, device=device)
            for v in x.values()
        ]
        if not vals:
            raise RuntimeError("native detector loss dictionary is empty")
        loss = sum(v.sum() for v in vals)
        items = x
    elif torch.is_tensor(x):
        loss = x.sum()
        items = None
    else:
        loss = torch.as_tensor(float(x), dtype=torch.float32)
        items = None
    if loss.ndim != 0:
        loss = loss.sum()
    if not torch.isfinite(loss):
        raise RuntimeError("native detector loss is non-finite")
    return loss, items


def set_all_rng(seed: int) -> None:
    """Reset Python/NumPy/Torch RNGs for frozen stochastic loss replay."""
    seed = int(seed) & 0xFFFFFFFF
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def rtdetr_loss_seed(view_index: int) -> int:
    """Frozen Experiment-17 per-view native-loss seed."""
    token = (
        "ConsRestore|Experiment17|Step11RTDETRLossReplay|"
        f"view={int(view_index)}"
    )
    return int.from_bytes(hashlib.sha256(token.encode("utf-8")).digest()[:4], "big", signed=False)


def _loss_forward(model, batch, *, initialize_criterion: bool, rng_seed: int | None):
    if initialize_criterion:
        ensure_criterion(model)
    if rng_seed is not None:
        set_all_rng(rng_seed)
    # RT-DETR's locked replay contract calls model.loss(batch) explicitly;
    # YOLO's locked replay contract uses model(batch) after configuring the
    # Detect container for raw-output native criterion mode.
    return model.loss(batch) if initialize_criterion else model(batch)


def ensure_criterion(model) -> None:
    """Initialize lazy RT-DETR criterion once when required."""
    if getattr(model, "criterion", None) is None and hasattr(model, "init_criterion"):
        model.criterion = model.init_criterion()
    crit = getattr(model, "criterion", None)
    if crit is None or not callable(crit):
        raise RuntimeError("model criterion is not callable after initialization")


def native_loss_plain(model, batch, *, initialize_criterion: bool = False, rng_seed: int | None = None) -> float:
    with torch.no_grad():
        result = _loss_forward(model, batch, initialize_criterion=initialize_criterion, rng_seed=rng_seed)
        loss, _ = scalarize_native_loss(result)
    return float(loss.detach().cpu())


def native_loss_and_p4_gradient(model, batch, *, p4_module: int, channels: int, hw: int, initialize_criterion: bool = False, rng_seed: int | None = None):
    """Return L, detached leaf P4 S, and downstream dL/dS without parameter grads."""
    holder = {}
    def hook(_module, _inp, out):
        if not torch.is_tensor(out):
            raise RuntimeError("P4 module did not return a Tensor")
        leaf = out.detach().clone().requires_grad_(True)
        holder["p4"] = leaf
        return leaf
    h = model.model[p4_module].register_forward_hook(hook)
    try:
        result = _loss_forward(model, batch, initialize_criterion=initialize_criterion, rng_seed=rng_seed)
    finally:
        h.remove()
    if "p4" not in holder:
        raise RuntimeError("P4 hook did not fire")
    loss, _ = scalarize_native_loss(result)
    s = holder["p4"]
    expected = (int(batch["img"].shape[0]), channels, hw, hw)
    if tuple(s.shape) != expected:
        raise RuntimeError(f"unexpected P4 shape {tuple(s.shape)} != {expected}")
    grad = torch.autograd.grad(loss, s, retain_graph=False, create_graph=False, allow_unused=False)[0]
    if grad is None or not torch.isfinite(grad).all():
        raise RuntimeError("invalid P4 task gradient")
    return float(loss.detach().cpu()), s.detach(), grad.detach()


def capture_eval_p4(model, image: torch.Tensor, *, p4_module: int, channels: int, hw: int) -> torch.Tensor:
    holder = {}
    def hook(_module, _inp, out):
        if not torch.is_tensor(out):
            raise RuntimeError("P4 module did not return a Tensor")
        holder["p4"] = out
    h = model.model[p4_module].register_forward_hook(hook)
    try:
        with torch.no_grad():
            _ = model(image)
    finally:
        h.remove()
    p4 = holder.get("p4")
    if p4 is None or tuple(p4.shape) != (int(image.shape[0]), channels, hw, hw):
        raise RuntimeError("failed to capture expected P4")
    return p4.detach()


def native_loss_with_cell_replacement(
    model,
    batch,
    y: int,
    x: int,
    replacement: torch.Tensor,
    *,
    p4_module: int,
    channels: int,
    hw: int,
    initialize_criterion: bool = False,
    rng_seed: int | None = None,
) -> float:
    """Evaluate the frozen native loss after replacing one P4 cell.

    This is the exact intervention primitive used by the RT-DETR Step-11/12/14
    chain: the module output is detached/cloned, one ``[:, y, x]`` vector is
    replaced, and the same per-view RNG seed is reset before every loss call.
    """
    y, x = int(y), int(x)
    if not (0 <= y < hw and 0 <= x < hw):
        raise RuntimeError(f"invalid P4 cell y={y}, x={x}")
    replacement = replacement.detach()
    if tuple(replacement.shape) != (channels,):
        raise RuntimeError(
            f"replacement vector shape {tuple(replacement.shape)} != ({channels},)"
        )

    def hook(_module, _inp, out):
        if not torch.is_tensor(out):
            raise RuntimeError("P4 intervention module did not return a Tensor")
        expected = (1, channels, hw, hw)
        if tuple(out.shape) != expected:
            raise RuntimeError(f"P4 intervention shape {tuple(out.shape)} != {expected}")
        changed = out.detach().clone()
        changed[0, :, y, x] = replacement.to(device=changed.device, dtype=changed.dtype)
        return changed

    h = model.model[p4_module].register_forward_hook(hook)
    try:
        with torch.no_grad():
            result = _loss_forward(
                model, batch, initialize_criterion=initialize_criterion, rng_seed=rng_seed
            )
            loss, _ = scalarize_native_loss(result)
    finally:
        h.remove()
    return float(loss.detach().cpu())


def native_loss_with_p4_direction(model, batch, direction: torch.Tensor, alpha: float, *, p4_module: int, initialize_criterion: bool = False, rng_seed: int | None = None) -> float:
    """Evaluate L(S + alpha*D), replacing the whole P4 tensor with frozen D support."""
    direction = direction.detach()
    def hook(_module, _inp, out):
        if not torch.is_tensor(out) or tuple(out.shape) != tuple(direction.shape):
            raise RuntimeError("P4 intervention shape mismatch")
        return out.detach().clone() + float(alpha) * direction.to(device=out.device, dtype=out.dtype)
    h = model.model[p4_module].register_forward_hook(hook)
    try:
        with torch.no_grad():
            result = _loss_forward(model, batch, initialize_criterion=initialize_criterion, rng_seed=rng_seed)
            loss, _ = scalarize_native_loss(result)
    finally:
        h.remove()
    return float(loss.detach().cpu())


def core9_object_center_cells(batch: dict, *, hw: int = 40, excluded_class: int = 8) -> list[tuple[int, int, int]]:
    """Unique spatial CORE9 P4 cells as ``(representative_class, y, x)``.

    The intervention unit is the spatial cell, not the object/class tuple. If
    multiple CORE9 objects map to the same P4 location, that location appears
    once. The representative class is retained only for diagnostics and does
    not affect the geometry or intervention.
    """
    batch_idx = batch["batch_idx"].detach().cpu().reshape(-1).numpy()
    cls = batch["cls"].detach().cpu().reshape(-1).numpy()
    bboxes = batch["bboxes"].detach().cpu().numpy()
    if bboxes.ndim != 2 or bboxes.shape[1] != 4:
        raise RuntimeError(f"unexpected bbox shape {bboxes.shape}")
    if not (len(batch_idx) == len(cls) == len(bboxes)):
        raise RuntimeError("batch_idx / cls / bboxes lengths differ")
    spatial = {}
    for bi, c, box in zip(batch_idx, cls, bboxes):
        if int(round(float(bi))) != 0:
            continue
        class_id = int(round(float(c)))
        if class_id == excluded_class:
            continue
        xc, yc = float(box[0]), float(box[1])
        if not (0.0 <= xc <= 1.0 and 0.0 <= yc <= 1.0):
            raise RuntimeError("expected normalized xywh bboxes")
        x = min(hw - 1, max(0, int(np.floor(xc * hw))))
        y = min(hw - 1, max(0, int(np.floor(yc * hw))))
        key = (y, x)
        # Deterministic human-readable representative only. Geometry depends
        # solely on the spatial location.
        if key not in spatial or class_id < spatial[key]:
            spatial[key] = class_id
    return [(spatial[(y, x)], y, x) for y, x in sorted(spatial)]


def verify_paired_geometry(degraded: dict, clean: dict) -> None:
    """Require identical transformed GT geometry for paired views."""
    if tuple(degraded["img"].shape) != tuple(clean["img"].shape):
        raise RuntimeError("clean/degraded image tensor shapes differ")
    for key in ("batch_idx", "cls", "bboxes"):
        if key not in degraded or key not in clean:
            raise RuntimeError(f"paired batch missing {key}")
        a, b = degraded[key], clean[key]
        if torch.is_tensor(a) and torch.is_tensor(b):
            if not torch.equal(a.detach().cpu(), b.detach().cpu()):
                raise RuntimeError(f"clean/degraded paired geometry differs in {key}")
        elif a != b:
            raise RuntimeError(f"clean/degraded paired geometry differs in {key}")


def clone_batch_to_device(batch: dict, device) -> dict:
    out={}
    for k,v in batch.items():
        out[k]=v.to(device,non_blocking=False) if torch.is_tensor(v) else copy.deepcopy(v)
    if "img" not in out:
        raise RuntimeError("batch missing img")
    img=out["img"]
    if not torch.is_floating_point(img): img=img.float()
    if float(img.max()) > 1.0: img=img/255.0
    out["img"]=img
    return out
