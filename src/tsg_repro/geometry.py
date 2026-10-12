from __future__ import annotations

import numpy as np


Array = np.ndarray


def build_task_geometry(gradients: Array, *, eps: float = 1e-12) -> Array:
    """Build trace-normalized task-sensitive channel geometry C_g.

    Each row is a channel gradient vector. Rows are L2-normalized before their
    outer products are averaged. A final symmetrization and trace normalization
    are numerical safeguards.
    """
    g = np.asarray(gradients, dtype=np.float64)
    if g.ndim != 2:
        raise ValueError(f"gradients must be [N,D], got {g.shape}")
    if not np.isfinite(g).all():
        raise ValueError("gradients contain non-finite values")
    norms = np.linalg.norm(g, axis=1)
    keep = norms > eps
    if not np.any(keep):
        raise ValueError("no non-zero gradients")
    gh = g[keep] / norms[keep, None]
    c = (gh.T @ gh) / gh.shape[0]
    c = 0.5 * (c + c.T)
    tr = float(np.trace(c))
    if not np.isfinite(tr) or tr <= eps:
        raise ValueError("invalid geometry trace")
    c /= tr
    return c


def orientation_score(directions: Array, c_g: Array, *, eps: float = 1e-20) -> Array:
    """Compute q(d)=D d^T C_g d / ||d||^2 row-wise."""
    d = np.asarray(directions, dtype=np.float64)
    c = np.asarray(c_g, dtype=np.float64)
    one = False
    if d.ndim == 1:
        d = d[None, :]
        one = True
    if d.ndim != 2 or c.ndim != 2 or c.shape[0] != c.shape[1] or d.shape[1] != c.shape[0]:
        raise ValueError(f"incompatible shapes directions={d.shape}, C_g={c.shape}")
    denom = np.einsum("ij,ij->i", d, d)
    num = np.einsum("ij,jk,ik->i", d, c, d)
    q = np.full(d.shape[0], np.nan, dtype=np.float64)
    good = denom > eps
    q[good] = d.shape[1] * num[good] / denom[good]
    return q[0] if one else q


def orientation_score_set(directions: Array, c_g: Array, *, eps: float = 1e-20) -> float:
    """Energy-weighted orientation score for a set of selected cells."""
    d = np.asarray(directions, dtype=np.float64)
    c = np.asarray(c_g, dtype=np.float64)
    if d.ndim != 2 or c.shape != (d.shape[1], d.shape[1]):
        raise ValueError(f"incompatible shapes directions={d.shape}, C_g={c.shape}")
    denom = float(np.einsum("ij,ij->", d, d))
    if denom <= eps:
        return float("nan")
    num = float(np.einsum("ij,jk,ik->", d, c, d))
    return float(d.shape[1] * num / denom)
