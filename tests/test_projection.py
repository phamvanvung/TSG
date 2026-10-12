import numpy as np

from tsg_repro.projection import (
    decompose_flattened_region,
    decompose_per_cell,
    reconstruction_relative_error,
)


def test_flattened_and_per_cell_are_distinct_but_reconstruct():
    rng = np.random.default_rng(8)
    k, d = 9, 6
    qn, _ = np.linalg.qr(rng.normal(size=(d, d)))
    qs, _ = np.linalg.qr(rng.normal(size=(d, d)))
    t = rng.normal(size=(k, d))
    n_teacher = rng.normal(size=(k, d))
    n = n_teacher @ qn.T

    a = decompose_flattened_region(n, t, qn, qs)
    b = decompose_per_cell(n, t, qn, qs)
    assert reconstruction_relative_error(a.none_teacher, a.explained_teacher, a.complement_teacher) < 1e-12
    assert reconstruction_relative_error(b.none_teacher, b.explained_teacher, b.complement_teacher) < 1e-12
    assert np.ndim(a.alpha) == 0
    assert b.alpha.shape == (k,)
    assert not np.allclose(a.complement_teacher, b.complement_teacher)
