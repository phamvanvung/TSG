import numpy as np

from tsg_repro.statistics import cluster_bootstrap, partial_spearman


def test_partial_spearman_removes_shared_monotone_covariate():
    rng = np.random.default_rng(11)
    z = np.linspace(-2, 2, 300)
    x = z + 0.2 * rng.normal(size=len(z))
    y = z + 0.2 * rng.normal(size=len(z))
    rho = partial_spearman(x, y, z)
    assert abs(rho) < 0.2


def test_cluster_bootstrap_is_deterministic():
    rng = np.random.default_rng(12)
    clusters = np.repeat(np.arange(20), 3)
    x = rng.normal(size=len(clusters))
    y = x + 0.1 * rng.normal(size=len(clusters))
    fn = lambda s: float(np.corrcoef(s["x"], s["y"])[0, 1])
    a = cluster_bootstrap({"x": x, "y": y}, clusters, fn, reps=100, seed=2027)
    b = cluster_bootstrap({"x": x, "y": y}, clusters, fn, reps=100, seed=2027)
    assert a == b


def test_view_manifest_preserves_frozen_view_index_order(tmp_path):
    import pandas as pd
    from tsg_repro.geometry_workflow import load_view_manifest
    rows=[]
    for vi,(src,sig) in enumerate([("b",25),("a",15),("b",15),("a",50),("a",25),("b",50)]):
        rows.append({"view_index":vi,"source_stem":src,"sigma":sig,"clean_image":"c","degraded_image":"d","label_file":"l"})
    path=tmp_path/"m.csv"; pd.DataFrame(rows[::-1]).to_csv(path,index=False)
    out=load_view_manifest(path,expected_sources=2,expected_views=6)
    assert out.view_index.tolist()==list(range(6))
