#!/usr/bin/env python
"""Static fail-closed audit of coordinate and finite-horizon scientific contracts."""
from __future__ import annotations
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]

def need(path,parts):
 s=(ROOT/path).read_text()
 miss=[x for x in parts if x not in s]
 if miss:raise RuntimeError(f'{path}: missing frozen contract tokens: {miss}')
 return s

def main():
 hist=need(Path('src/tsg_repro/local_utility.py'),['historical_mapped_complement_torch','torch.dot(n, t)','torch.matmul(n_c, bridge)','spearman_zero_constant','spearman_nan_constant'])
 step12=need(Path('scripts/rtdetr/03_run_local_utility_confirmation.py'),['rtdetr_step10_bridge','historical_mapped_complement_torch','native_loss_with_cell_replacement','step12_confirmation'])
 step14=need(Path('scripts/rtdetr/04_run_finite_horizon_confirmation.py'),['freeze_step14_selection','frozen_u_C','frozen_u_rel','Delta_small','Delta_large','recomputed_step12_scores_used_as_scientific_gate'])
 orient=need(Path('scripts/rtdetr/05_run_orientation_confirmation.py'),['R_NT_NONE_TO_TEACHER.npy','R_ST_CLEAN_KD_TO_TEACHER.npy','canonical_mapped_complement_cells_torch'])
 if 'rtdetr_step10_bridge' in orient or 'historical_mapped_complement_torch' in orient:raise RuntimeError('canonical RT-DETR orientation imports historical Step-10 projection path')
 if 'R_NT_NONE_TO_TEACHER.npy' in step12 or 'R_ST_CLEAN_KD_TO_TEACHER.npy' in step12:raise RuntimeError('historical Step-12 was silently teacher-canonicalized')
 if 'R_NT_NONE_TO_TEACHER.npy' in step14 or 'R_ST_CLEAN_KD_TO_TEACHER.npy' in step14:raise RuntimeError('historical Step-14 was silently teacher-canonicalized')
 report={'status':'PASS','historical_projection_isolated_to_step12_14':True,'canonical_orientation_uses_R_NT_R_ST':True,'step14_primary_predictor_token':'frozen_u_C','step12_constant_spearman':'0.0','step14_constant_spearman':'NaN'}
 print(json.dumps(report,indent=2))
if __name__=='__main__':main()
