#!/usr/bin/env python
"""Fit YOLO11s C_g from native CLEAN-KD P4 task gradients on frozen basis-fit views."""
from __future__ import annotations
import argparse,json,shutil,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2]; sys.path.insert(0,str(ROOT/'src'))
from tsg_repro.config import load_config,get_path
from tsg_repro.geometry_workflow import load_view_manifest
from tsg_repro.yolo_runtime import import_checkpoint_companion,load_hyp,build_dataset,preprocess_one,safe_symlink,require_yolo_environment,P4_MODULE,P4_CHANNELS,P4_HW,configure_native_replay_model
from tsg_repro.native_model import native_loss_and_p4_gradient,core9_object_center_cells
from tsg_repro.hashing import sha256_file

def main():
 p=argparse.ArgumentParser(); p.add_argument('--config',required=True); p.add_argument('--output',default='results/yolo_task_geometry'); p.add_argument('--device',default='cuda:0'); a=p.parse_args()
 cfg,base=load_config(a.config); manifest_path=get_path(cfg,base,'data','yolo_tsg_basis_manifest'); df=load_view_manifest(manifest_path,expected_sources=120,expected_views=360)
 out=(ROOT/Path(a.output)).resolve() if not Path(a.output).is_absolute() else Path(a.output).resolve()
 if out.exists(): raise FileExistsError(out)
 tmp=out.parent/f'.{out.name}_stage'; shutil.rmtree(tmp,ignore_errors=True)
 for r in df.itertuples(index=False):
  src=Path(r.degraded_image).resolve(); lab=Path(r.label_file).resolve(); stem=str(r.generated_stem)
  safe_symlink(src,tmp/'images'/f'{stem}{src.suffix.lower()}'); safe_symlink(lab,tmp/'labels'/f'{stem}.txt')
 torch=require_yolo_environment(require_cuda=a.device.startswith('cuda'))
 import_checkpoint_companion(get_path(cfg,base,'checkpoints','yolo_checkpoint_companion',required=False))
 from ultralytics import YOLO
 device=torch.device(a.device); model=YOLO(str(get_path(cfg,base,'checkpoints','yolo11s_clean_kd'))).model.float().to(device)
 hyp=load_hyp(get_path(cfg,base,'data','yolo_none_args_yaml')); configure_native_replay_model(model,hyp); ds=build_dataset(tmp/'images',hyp,'YOLO TSG basis: '); idx={Path(x).stem:i for i,x in enumerate(ds.im_files)}
 cross=np.zeros((P4_CHANNELS,P4_CHANNELS),np.float64); ngrad=0; eligible=0
 for j,r in enumerate(df.itertuples(index=False)):
  batch=preprocess_one(ds[idx[str(r.generated_stem)]],device); cells=core9_object_center_cells(batch,hw=P4_HW[0])
  if len(cells)<4: continue
  _,_,g=native_loss_and_p4_gradient(model,batch,p4_module=P4_MODULE,channels=P4_CHANNELS,hw=P4_HW[0])
  for _,y,x in cells:
   v=g[0,:,y,x].detach().cpu().numpy().astype(np.float64); n=np.linalg.norm(v)
   if np.isfinite(n) and n>1e-12: v/=n; cross+=np.outer(v,v); ngrad+=1
  eligible+=1
  if (j+1)%25==0: print(f'views {j+1}/{len(df)} valid_gradients={ngrad}',flush=True)
 if eligible != 360:
  raise RuntimeError(f'Expected all 360 BASIS_FIT views eligible; found {eligible}')
 if ngrad != 5631:
  raise RuntimeError(f'Expected 5631 valid normalized gradient vectors; found {ngrad}')
 cg=cross/ngrad; cg=.5*(cg+cg.T); cg/=np.trace(cg)
 vals,vecs=np.linalg.eigh(cg); order=np.argsort(vals)[::-1]; vals=vals[order]; vecs=vecs[:,order]
 out.mkdir(parents=True,exist_ok=False); np.save(out/'C_g.npy',cg); np.save(out/'C_g_eigenvalues_desc.npy',vals); np.save(out/'C_g_eigenvectors_desc.npy',vecs)
 report={'status':'YOLO11S_TASK_SENSITIVE_GEOMETRY_FIT_COMPLETE','basis_fit':{'sources':120,'views':360,'eligible_views':eligible,'normalized_gradient_vectors':ngrad,'support':'all unique CORE9 object-center P4 cells','minimum_support_cells':4},'geometry':{'trace':float(np.trace(cg)),'symmetry_error':float(np.linalg.norm(cg-cg.T)),'min_eigenvalue':float(vals.min()),'sha256':sha256_file(out/'C_g.npy')},'firewall':{'clean_kd_only':True,'teacher_loaded':False,'none_loaded':False,'counterfactual_outcomes_used':False},'provenance':{'basis_manifest_sha256':sha256_file(manifest_path)}}
 (out/'TASK_SENSITIVE_GEOMETRY_FIT.json').write_text(json.dumps(report,indent=2)+'\n'); shutil.rmtree(tmp,ignore_errors=True); print(json.dumps(report,indent=2))
if __name__=='__main__': main()
