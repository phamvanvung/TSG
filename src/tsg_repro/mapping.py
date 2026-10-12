from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np


Array = np.ndarray


@dataclass(frozen=True)
class CanonicalMaps:
    """Orthogonal maps into teacher coordinates.

    Row-vector convention:
        N_T = N @ R_NT
        S_T = S @ R_ST
        T_S = T @ R_ST.T

    `R_NT` maps NONE -> teacher and `R_ST` maps CLEAN-KD -> teacher.
    """

    r_nt: Array
    r_st: Array

    @property
    def r_ts(self) -> Array:
        """Teacher -> CLEAN-KD inverse map for an orthogonal R_ST."""
        return self.r_st.T


def _as_2d(x: Array, name: str) -> Array:
    x = np.asarray(x)
    if x.ndim != 2:
        raise ValueError(f"{name} must be 2-D [rows, channels], got {x.shape}")
    if not np.isfinite(x).all():
        raise ValueError(f"{name} contains non-finite values")
    return x


def fit_orthogonal_map(source: Array, target: Array) -> Array:
    """Fit R minimizing ||source @ R - target||_F with R^T R = I.

    This is the uncentered orthogonal Procrustes solution appropriate for a
    coordinate alignment/rotation. Source and target must be paired rows.
    """
    source = _as_2d(source, "source").astype(np.float64, copy=False)
    target = _as_2d(target, "target").astype(np.float64, copy=False)
    if source.shape != target.shape:
        raise ValueError(
            f"source/target shapes must match, got {source.shape} vs {target.shape}"
        )
    cross = source.T @ target
    u, _, vt = np.linalg.svd(cross, full_matrices=False)
    r = u @ vt
    return r.astype(np.float64, copy=False)


def fit_teacher_canonical_maps(
    none_features: Array,
    clean_kd_features: Array,
    teacher_features: Array,
) -> CanonicalMaps:
    """Fit TRAIN-only NONE->teacher and CLEAN-KD->teacher maps."""
    n = _as_2d(none_features, "none_features")
    s = _as_2d(clean_kd_features, "clean_kd_features")
    t = _as_2d(teacher_features, "teacher_features")
    if not (n.shape == s.shape == t.shape):
        raise ValueError(f"paired feature shapes differ: N={n.shape}, S={s.shape}, T={t.shape}")
    return CanonicalMaps(
        r_nt=fit_orthogonal_map(n, t),
        r_st=fit_orthogonal_map(s, t),
    )


def map_none_to_teacher(x: Array, r_nt: Array) -> Array:
    return np.asarray(x) @ np.asarray(r_nt)


def map_student_to_teacher(x: Array, r_st: Array) -> Array:
    return np.asarray(x) @ np.asarray(r_st)


def map_teacher_to_student(x: Array, r_st: Array) -> Array:
    return np.asarray(x) @ np.asarray(r_st).T


def orthogonality_error(r: Array) -> float:
    r = _as_2d(r, "r").astype(np.float64, copy=False)
    if r.shape[0] != r.shape[1]:
        raise ValueError(f"orthogonal map must be square, got {r.shape}")
    eye = np.eye(r.shape[0], dtype=np.float64)
    return float(np.linalg.norm(r.T @ r - eye, ord="fro") / np.linalg.norm(eye, ord="fro"))


def identity_distance(r: Array) -> float:
    r = _as_2d(r, "r").astype(np.float64, copy=False)
    if r.shape[0] != r.shape[1]:
        raise ValueError(f"map must be square, got {r.shape}")
    eye = np.eye(r.shape[0], dtype=np.float64)
    return float(np.linalg.norm(r - eye, ord="fro") / np.linalg.norm(eye, ord="fro"))


def validate_map(
    r: Array,
    *,
    channels: int = 256,
    max_orthogonality_error: float = 1e-5,
) -> None:
    r = _as_2d(r, "r")
    if r.shape != (channels, channels):
        raise ValueError(f"expected map shape {(channels, channels)}, got {r.shape}")
    err = orthogonality_error(r)
    if err >= max_orthogonality_error:
        raise ValueError(
            f"orthogonality error {err:.8e} >= {max_orthogonality_error:.8e}"
        )


def paired_alignment_error(source: Array, target: Array, r: Array) -> float:
    source = _as_2d(source, "source").astype(np.float64, copy=False)
    target = _as_2d(target, "target").astype(np.float64, copy=False)
    mapped = source @ np.asarray(r, dtype=np.float64)
    denom = np.linalg.norm(target, ord="fro")
    if denom <= 0:
        raise ValueError("target has zero Frobenius norm")
    return float(np.linalg.norm(mapped - target, ord="fro") / denom)
