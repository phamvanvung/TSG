"""Architecture-neutral task-sensitive geometry and native-loss intervention helpers."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np


def trace_normalized_gradient_geometry(gradients: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """C_g = trace_normalize(sym(mean(g_hat g_hat^T)))."""
    g = np.asarray(gradients, dtype=np.float64)
    if g.ndim != 2:
        raise ValueError("gradients must be [N,C]")
    norms = np.linalg.norm(g, axis=1)
    good = np.isfinite(norms) & (norms > eps) & np.isfinite(g).all(axis=1)
    if not np.any(good):
        raise ValueError("no valid non-zero gradients")
    gh = g[good] / norms[good, None]
    c = (gh.T @ gh) / len(gh)
    c = 0.5 * (c + c.T)
    tr = float(np.trace(c))
    if not np.isfinite(tr) or tr <= 0:
        raise ValueError("invalid geometry trace")
    return c / tr


def task_sensitive_orientation(direction_cells: np.ndarray, cg: np.ndarray) -> float:
    """D * sum_i d_i^T C_g d_i / sum_i ||d_i||^2, D=channel dimension."""
    d = np.asarray(direction_cells, dtype=np.float64)
    c = np.asarray(cg, dtype=np.float64)
    if d.ndim != 2 or c.shape != (d.shape[1], d.shape[1]):
        raise ValueError("direction/C_g shape mismatch")
    den = float(np.sum(d * d))
    if den <= 0 or not np.isfinite(den):
        return float("nan")
    numer = float(np.einsum("ni,ij,nj->", d, c, d))
    return float(d.shape[1] * numer / den)


def direction_norm(direction_cells: np.ndarray) -> float:
    return float(np.linalg.norm(np.asarray(direction_cells, dtype=np.float64)))


def canonical_mapped_complement_cells(
    none_cells: np.ndarray,
    teacher_cells: np.ndarray,
    r_nt: np.ndarray,
    r_st: np.ndarray,
    eps: float = 1e-20,
) -> np.ndarray:
    """Per-cell teacher-canonical mapped complement M in student coordinates.

    Row-vector convention:
      n_T = n_N @ R_NT
      c_T = n_T - proj_{t_T}(n_T), per support cell
      m_S = c_T @ R_ST.T
    """
    n = np.asarray(none_cells, dtype=np.float64)
    t = np.asarray(teacher_cells, dtype=np.float64)
    rnt = np.asarray(r_nt, dtype=np.float64)
    rst = np.asarray(r_st, dtype=np.float64)
    if n.shape != t.shape or n.ndim != 2:
        raise ValueError("NONE/teacher cells must share [K,C] shape")
    c = n.shape[1]
    if rnt.shape != (c, c) or rst.shape != (c, c):
        raise ValueError("canonical map shape mismatch")
    nt = n @ rnt
    energy = np.sum(t * t, axis=1)
    alpha = np.zeros(len(t), dtype=np.float64)
    good = energy > eps
    alpha[good] = np.sum(nt[good] * t[good], axis=1) / energy[good]
    ct = nt - alpha[:, None] * t
    return ct @ rst.T


def canonical_teacher_residual_cells(
    student_cells: np.ndarray,
    teacher_cells: np.ndarray,
    r_st: np.ndarray,
) -> np.ndarray:
    """Teacher residual E=T_S-S in student coordinates."""
    s = np.asarray(student_cells, dtype=np.float64)
    t = np.asarray(teacher_cells, dtype=np.float64)
    rst = np.asarray(r_st, dtype=np.float64)
    if s.shape != t.shape or s.ndim != 2:
        raise ValueError("student/teacher cells must share [K,C] shape")
    if rst.shape != (s.shape[1], s.shape[1]):
        raise ValueError("R_ST shape mismatch")
    return t @ rst.T - s


def canonical_mapped_complement_cells_torch(
    none_cells,
    teacher_cells,
    r_nt,
    r_st,
    eps: float = 1e-20,
):
    """Torch/float-preserving form of the per-cell canonical mapped complement.

    This is the checkpoint-replay path. It intentionally performs the N->T
    map, projection, and T->S map in the tensors' existing dtype/device so a
    float32 CUDA replay follows the same arithmetic family as the frozen
    intervention engines.
    """
    import torch

    n = none_cells
    t = teacher_cells
    if not (torch.is_tensor(n) and torch.is_tensor(t)):
        raise TypeError("none_cells and teacher_cells must be torch tensors")
    if n.shape != t.shape or n.ndim != 2:
        raise ValueError("NONE/teacher cells must share [K,C] shape")
    c = int(n.shape[1])
    if tuple(r_nt.shape) != (c, c) or tuple(r_st.shape) != (c, c):
        raise ValueError("canonical map shape mismatch")
    nt = n @ r_nt.to(device=n.device, dtype=n.dtype)
    tt = t.to(device=n.device, dtype=n.dtype)
    energy = torch.sum(tt * tt, dim=1)
    numer = torch.sum(nt * tt, dim=1)
    alpha = torch.where(energy > float(eps), numer / energy.clamp_min(float(eps)), torch.zeros_like(energy))
    ct = nt - alpha[:, None] * tt
    return ct @ r_st.to(device=n.device, dtype=n.dtype).transpose(0, 1)


def canonical_teacher_residual_cells_torch(student_cells, teacher_cells, r_st):
    """Torch/float-preserving teacher residual E=T_S-S in student coordinates."""
    import torch

    s = student_cells
    t = teacher_cells
    if not (torch.is_tensor(s) and torch.is_tensor(t)):
        raise TypeError("student_cells and teacher_cells must be torch tensors")
    if s.shape != t.shape or s.ndim != 2:
        raise ValueError("student/teacher cells must share [K,C] shape")
    c = int(s.shape[1])
    if tuple(r_st.shape) != (c, c):
        raise ValueError("R_ST shape mismatch")
    rst = r_st.to(device=s.device, dtype=s.dtype)
    tt = t.to(device=s.device, dtype=s.dtype)
    return tt @ rst.transpose(0, 1) - s


def support_direction_tensor_torch(cells, coords: Iterable[tuple[int, int]], *, channels: int, hw: int):
    """Place K torch row-vectors at support coordinates into [1,C,H,W]."""
    import torch

    if not torch.is_tensor(cells):
        raise TypeError("cells must be a torch tensor")
    coords = list(coords)
    if tuple(cells.shape) != (len(coords), channels):
        raise ValueError("cell/coordinate count mismatch")
    out = torch.zeros((1, channels, hw, hw), dtype=cells.dtype, device=cells.device)
    for vec, (y, x) in zip(cells, coords):
        if not (0 <= y < hw and 0 <= x < hw):
            raise ValueError(f"invalid support coordinate {(y,x)}")
        out[0, :, y, x] = vec
    return out


def support_direction_tensor(cells: np.ndarray, coords: Iterable[tuple[int, int]], *, channels: int, hw: int) -> np.ndarray:
    """Place K row-vectors at support coordinates into [1,C,H,W]."""
    arr = np.asarray(cells, dtype=np.float64)
    coords = list(coords)
    if arr.shape != (len(coords), channels):
        raise ValueError("cell/coordinate count mismatch")
    out = np.zeros((1, channels, hw, hw), dtype=np.float64)
    for vec, (y, x) in zip(arr, coords):
        if not (0 <= y < hw and 0 <= x < hw):
            raise ValueError(f"invalid support coordinate {(y,x)}")
        out[0, :, y, x] = vec
    return out
