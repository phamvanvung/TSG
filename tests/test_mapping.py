import numpy as np

from tsg_repro.mapping import (
    fit_teacher_canonical_maps,
    map_none_to_teacher,
    map_student_to_teacher,
    map_teacher_to_student,
    orthogonality_error,
)


def test_exact_synthetic_maps():
    rng = np.random.default_rng(7)
    d = 12
    n = 300
    qn, _ = np.linalg.qr(rng.normal(size=(d, d)))
    qs, _ = np.linalg.qr(rng.normal(size=(d, d)))
    t = rng.normal(size=(n, d))
    none = t @ qn.T
    student = t @ qs.T
    maps = fit_teacher_canonical_maps(none, student, t)
    assert np.linalg.norm(map_none_to_teacher(none, maps.r_nt) - t) / np.linalg.norm(t) < 1e-12
    assert np.linalg.norm(map_student_to_teacher(student, maps.r_st) - t) / np.linalg.norm(t) < 1e-12
    assert np.linalg.norm(map_teacher_to_student(t, maps.r_st) - student) / np.linalg.norm(student) < 1e-12
    assert orthogonality_error(maps.r_nt) < 1e-12
    assert orthogonality_error(maps.r_st) < 1e-12
