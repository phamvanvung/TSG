from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .mapping import map_none_to_teacher, map_teacher_to_student


Array = np.ndarray


@dataclass(frozen=True)
class FlattenedProjection:
    alpha: float
    none_teacher: Array
    explained_teacher: Array
    complement_teacher: Array
    explained_student: Array
    complement_student: Array


@dataclass(frozen=True)
class PerCellProjection:
    alpha: Array
    none_teacher: Array
    explained_teacher: Array
    complement_teacher: Array
    explained_student: Array
    complement_student: Array


def _check_cells(none_cells: Array, teacher_cells: Array) -> tuple[Array, Array]:
    n = np.asarray(none_cells, dtype=np.float64)
    t = np.asarray(teacher_cells, dtype=np.float64)
    if n.ndim != 2 or t.ndim != 2:
        raise ValueError("cell arrays must be [num_cells, channels]")
    if n.shape != t.shape:
        raise ValueError(f"NONE/teacher cell shapes differ: {n.shape} vs {t.shape}")
    if not np.isfinite(n).all() or not np.isfinite(t).all():
        raise ValueError("cell arrays contain non-finite values")
    return n, t


def decompose_flattened_region(
    none_cells: Array,
    teacher_cells: Array,
    r_nt: Array,
    r_st: Array,
    *,
    eps: float = 1e-20,
) -> FlattenedProjection:
    """Canonical Experiment-4/5-style flattened-region decomposition.

    1. Map each NONE cell into teacher coordinates.
    2. Flatten the complete region and compute ONE scalar projection coefficient.
    3. Decompose in teacher coordinates.
    4. Map teacher-space components into CLEAN-KD coordinates with R_ST.T.
    """
    n, t = _check_cells(none_cells, teacher_cells)
    n_t = map_none_to_teacher(n, r_nt)

    nf = n_t.reshape(-1)
    tf = t.reshape(-1)
    denom = float(tf @ tf)
    if not np.isfinite(denom) or denom <= eps:
        raise ValueError("teacher region has zero/non-finite energy")

    alpha = float((nf @ tf) / denom)
    e_t = (alpha * tf).reshape(t.shape)
    c_t = n_t - e_t

    e_s = map_teacher_to_student(e_t, r_st)
    c_s = map_teacher_to_student(c_t, r_st)
    return FlattenedProjection(alpha, n_t, e_t, c_t, e_s, c_s)


def decompose_per_cell(
    none_cells: Array,
    teacher_cells: Array,
    r_nt: Array,
    r_st: Array,
    *,
    eps: float = 1e-20,
) -> PerCellProjection:
    """Canonical Exp15A/16A-style per-cell 256-D decomposition."""
    n, t = _check_cells(none_cells, teacher_cells)
    n_t = map_none_to_teacher(n, r_nt)
    denom = np.einsum("ij,ij->i", t, t)
    if np.any(~np.isfinite(denom)) or np.any(denom <= eps):
        raise ValueError("at least one teacher cell has zero/non-finite energy")
    alpha = np.einsum("ij,ij->i", n_t, t) / denom
    e_t = alpha[:, None] * t
    c_t = n_t - e_t
    e_s = map_teacher_to_student(e_t, r_st)
    c_s = map_teacher_to_student(c_t, r_st)
    return PerCellProjection(alpha, n_t, e_t, c_t, e_s, c_s)


def reconstruction_relative_error(a: Array, b: Array, c: Array) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    c = np.asarray(c, dtype=np.float64)
    denom = np.linalg.norm(a.reshape(-1))
    if denom <= 0:
        return float(np.linalg.norm((a - (b + c)).reshape(-1)))
    return float(np.linalg.norm((a - (b + c)).reshape(-1)) / denom)
