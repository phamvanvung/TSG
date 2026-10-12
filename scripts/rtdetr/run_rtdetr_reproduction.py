#!/usr/bin/env python
"""Ordered RT-DETR-L checkpoint reproduction driver for the v1.0 candidate."""
from __future__ import annotations
import argparse,subprocess,sys
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
STAGES=[
 ('preflight','00_preflight.py'),
 ('evaluation','01_evaluate_frozen_checkpoints.py'),
 ('maps','01_fit_canonical_maps.py'),
 ('task-geometry','02_fit_task_sensitive_geometry.py'),
 ('local-utility','03_run_local_utility_confirmation.py'),
 ('finite-horizon','04_run_finite_horizon_confirmation.py'),
 ('linkage-audit','04b_audit_step12_step14_linkage.py'),
 ('orientation','05_run_orientation_confirmation.py'),
]
def main():
 p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--through',choices=[x[0] for x in STAGES],default='orientation');p.add_argument('--from-stage',choices=[x[0] for x in STAGES],default='preflight');p.add_argument('--python',default=sys.executable);a=p.parse_args();cfg=str(Path(a.config).expanduser().resolve());names=[x[0] for x in STAGES];start=names.index(a.from_stage);stop=names.index(a.through)
 if start>stop:p.error('--from-stage must not come after --through')
 for name,script in STAGES[start:stop+1]:
  print('\n'+'='*100);print('STAGE:',name,flush=True);cmd=[a.python,str(HERE/script),'--config',cfg];print(' '.join(cmd),flush=True);subprocess.run(cmd,check=True,cwd=ROOT)
if __name__=='__main__':main()
