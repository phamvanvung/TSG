#!/usr/bin/env python
"""Verify the frozen Step-14 RT-DETR finite-horizon source-level artifact.

Legacy compatibility verifier. In v1.0-rc1 the primary checkpoint path is migrated; this artifact-only utility remains because the primary predictor is
frozen Step-12 u_C, not C_g. Supply the authoritative Step-14 SOURCE_RESULTS
CSV; this script does not recompute or substitute the predictor.
"""
from __future__ import annotations
import argparse,json,sys
from pathlib import Path
import numpy as np,pandas as pd
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'src'))
from tsg_repro.hashing import sha256_file

def main():
 p=argparse.ArgumentParser();p.add_argument('--source-results',required=True);p.add_argument('--reps',type=int,default=5000);p.add_argument('--seed',type=int,default=2027);a=p.parse_args();path=Path(a.source_results).expanduser().resolve();df=pd.read_csv(path);req={'source_stem','rho_small_source','rho_large_source','delta_rho_source'};missing=req-set(df.columns)
 if missing:raise ValueError(f'missing columns {sorted(missing)}')
 if len(df)!=200 or df.source_stem.nunique()!=200:raise ValueError('expected 200 source rows')
 d=df.delta_rho_source.to_numpy(float);point=float(d.mean());rng=np.random.default_rng(a.seed);boot=np.asarray([d[rng.integers(0,len(d),size=len(d))].mean() for _ in range(a.reps)]);lo,hi=np.percentile(boot,[2.5,97.5]);result={'status':'RTDETR_FINITE_HORIZON_ARTIFACT_VERIFIED','rho_small_source_balanced':float(df.rho_small_source.mean()),'rho_large_source_balanced':float(df.rho_large_source.mean()),'delta_rho':point,'ci95':[float(lo),float(hi)],'pass':bool(point>0 and lo>0),'bootstrap_reps':a.reps,'bootstrap_seed':a.seed,'source_results_sha256':sha256_file(path),'reproduction_level':'artifact-level: frozen Step-12 u_C is not recomputed by this script'};print(json.dumps(result,indent=2))
if __name__=='__main__':main()
