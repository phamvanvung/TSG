import numpy as np

from tsg_repro.geometry import build_task_geometry, orientation_score


def test_trace_one_and_symmetric():
    rng = np.random.default_rng(9)
    g = rng.normal(size=(500, 16))
    c = build_task_geometry(g)
    assert np.allclose(c, c.T, atol=1e-14)
    assert np.isclose(np.trace(c), 1.0, atol=1e-14)


def test_isotropic_reference_is_one():
    rng = np.random.default_rng(10)
    d = 20
    c = np.eye(d) / d
    x = rng.normal(size=(50, d))
    q = orientation_score(x, c)
    assert np.allclose(q, 1.0, atol=1e-12)
