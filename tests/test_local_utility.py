import numpy as np
import pandas as pd
from tsg_repro.local_utility import (
    spearman_zero_constant, spearman_nan_constant,
    step12_confirmation, freeze_step14_selection, step14_confirmation,
)

def _cells():
    rows=[]
    # two sources x three views, four cells each; u_rel and Y perfectly aligned.
    vi=0
    for source in ('a','b'):
        for sigma in (15,25,50):
            for k in range(4):
                rows.append({'view_index':vi,'source_stem':source,'sigma':sigma,'p4_y':k,'p4_x':0,
                    'u_T':0.0,'u_C':float(k+1),'u_rel':float(k+1),'Delta_T':0.0,'Delta_C':float(k+1),'Y':float(k+1),
                    'class_id_representative':2})
            vi+=1
    return pd.DataFrame(rows)

def test_step12_constant_convention():
    assert spearman_zero_constant([1,1],[1,2]) == 0.0
    assert np.isnan(spearman_nan_constant([1,1],[1,2]))

def test_step12_gates_positive():
    out=step12_confirmation(_cells(),reps=100,seed=2027)
    assert out['gate_A']['point'] == 1.0
    assert out['gate_B']['point'] > 0
    assert out['advance']

def test_step14_selection_is_descending_urel():
    sel=freeze_step14_selection(_cells(),min_support=4,top_fraction=.25)
    assert len(sel)==6
    assert set(sel['frozen_u_rel'])=={4.0}

def test_step14_source_balanced_difference():
    cells=_cells();sel=freeze_step14_selection(cells,min_support=4,top_fraction=.5)
    # Need >=2 selected cells per view. Frozen u_C [4,3]. Small gain follows u_C,
    # large gain reverses it => per-view difference 2.
    rows=[]
    for r in sel.itertuples(index=False):
        rows.append({**r._asdict(),'Delta_small':float(r.frozen_u_C),'Delta_large':-float(r.frozen_u_C)})
    views=cells[['view_index','source_stem','sigma']].drop_duplicates()
    out=step14_confirmation(pd.DataFrame(rows),views,reps=100,seed=2027)
    assert np.isclose(out['primary']['rho_small_source_balanced'], 1.0)
    assert np.isclose(out['primary']['rho_large_source_balanced'], -1.0)
    assert np.isclose(out['primary']['point'], 2.0)
    assert out['primary']['pass']
