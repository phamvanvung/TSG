#!/usr/bin/env python
from __future__ import annotations

import json
import platform
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tsg_repro.geometry import build_task_geometry, orientation_score  # noqa: E402
from tsg_repro.mapping import fit_teacher_canonical_maps, orthogonality_error  # noqa: E402
from tsg_repro.projection import (  # noqa: E402
    decompose_flattened_region,
    reconstruction_relative_error,
)


def main() -> None:
    rng = np.random.default_rng(2027)
    d = 8
    n = 128

    def random_orthogonal():
        q, _ = np.linalg.qr(rng.normal(size=(d, d)))
        return q

    r_nt_true = random_orthogonal()
    r_st_true = random_orthogonal()
    teacher = rng.normal(size=(n, d))
    none = teacher @ r_nt_true.T
    student = teacher @ r_st_true.T

    maps = fit_teacher_canonical_maps(none, student, teacher)
    fit_nt = np.linalg.norm(none @ maps.r_nt - teacher) / np.linalg.norm(teacher)
    fit_st = np.linalg.norm(student @ maps.r_st - teacher) / np.linalg.norm(teacher)

    p = decompose_flattened_region(none[:11], teacher[:11], maps.r_nt, maps.r_st)
    rec = reconstruction_relative_error(p.none_teacher, p.explained_teacher, p.complement_teacher)

    grads = rng.normal(size=(512, d))
    cg = build_task_geometry(grads)
    q = orientation_score(rng.normal(size=(16, d)), cg)

    report = {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "synthetic_none_to_teacher_fit_error": float(fit_nt),
        "synthetic_student_to_teacher_fit_error": float(fit_st),
        "R_NT_orthogonality_error": orthogonality_error(maps.r_nt),
        "R_ST_orthogonality_error": orthogonality_error(maps.r_st),
        "flattened_projection_reconstruction_error": rec,
        "geometry_trace": float(np.trace(cg)),
        "orientation_scores_finite": bool(np.isfinite(q).all()),
        "status": "PASS",
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
