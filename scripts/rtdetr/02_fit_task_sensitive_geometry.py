#!/usr/bin/env python
"""Fit RT-DETR-L C_g from native CLEAN-KD module-24 gradients on JOURNAL_GEOMETRY_FIT."""
from __future__ import annotations
import argparse,json,shutil,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'src'))
from tsg_repro.config import load_config,get_path
from tsg_repro.geometry_workflow import load_view_manifest
from tsg_repro.rtdetr_runtime import *
from tsg_repro.yolo_runtime import import_checkpoint_companion
from tsg_repro.native_model import native_loss_and_p4_gradient,core9_object_center_cells,rtdetr_loss_seed
from tsg_repro.hashing import sha256_file

def main():
 p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--output',default='results/rtdetr_task_geometry');p.add_argument('--device',default='cuda:0');a=p.parse_args();cfg,base=load_config(a.config);mf=get_path(cfg,base,'data','rtdetr_geometry_fit_manifest');df=load_view_manifest(mf,expected_sources=120,expected_views=360);
 if 'view_index' not in df.columns: raise ValueError('RT-DETR geometry-fit manifest must contain frozen view_index')
 out=(ROOT/Path(a.output)).resolve() if not Path(a.output).is_absolute() else Path(a.output).resolve();
 if out.exists():raise FileExistsError(out)
 tmp=out.parent/f'.{out.name}_stage';shutil.rmtree(tmp,ignore_errors=True)
 for r in df.itertuples(index=False):src=Path(r.degraded_image).resolve();lab=Path(r.label_file).resolve();stem=str(r.generated_stem);safe_symlink(src,tmp/'images'/f'{stem}{src.suffix.lower()}');safe_symlink(lab,tmp/'labels'/f'{stem}.txt')
 torch=require_rtdetr_environment(require_cuda=a.device.startswith('cuda'));import_checkpoint_companion(get_path(cfg,base,'checkpoints','rtdetr_checkpoint_companion',required=False));device=torch.device(a.device);_,student=load_model(get_path(cfg,base,'checkpoints','rtdetr_l_clean_kd'),device);ds=build_dataset(tmp/'images','RTDETR Cg: ');idx={Path(x).stem:i for i,x in enumerate(ds.im_files)};cross=np.zeros((256,256),np.float64);ng=0;eligible=0
 for j,r in enumerate(df.itertuples(index=False)):
  b=preprocess_batch(fresh_collate_one(ds,idx[str(r.generated_stem)]),device);cells=core9_object_center_cells(b,hw=40)
  if len(cells)<4:continue
  seed=rtdetr_loss_seed(int(r.view_index)); _,_,grad=native_loss_and_p4_gradient(student,b,p4_module=24,channels=256,hw=40,initialize_criterion=True,rng_seed=seed)
  for _,y,x in cells:
   v=grad[0,:,y,x].cpu().numpy().astype(np.float64);n=np.linalg.norm(v)
   if np.isfinite(n) and n>1e-12:v/=n;cross+=np.outer(v,v);ng+=1
  eligible+=1
 if eligible != 360:raise RuntimeError(f'Expected all 360 JOURNAL_GEOMETRY_FIT views eligible; found {eligible}')
 if ng != 5769:raise RuntimeError(f'Expected 5769 valid normalized gradient vectors; found {ng}')
 cg=cross/ng;cg=.5*(cg+cg.T);cg/=np.trace(cg);vals,vecs=np.linalg.eigh(cg);o=np.argsort(vals)[::-1];vals=vals[o];vecs=vecs[:,o];out.mkdir(parents=True,exist_ok=False);np.save(out/'C_g.npy',cg);np.save(out/'C_g_eigenvalues_desc.npy',vals);np.save(out/'C_g_eigenvectors_desc.npy',vecs);rep={'status':'RTDETR_TASK_SENSITIVE_GEOMETRY_FIT_COMPLETE','basis_fit':{'sources':120,'views':360,'eligible_views':eligible,'normalized_gradient_vectors':ng},'geometry':{'trace':float(np.trace(cg)),'min_eigenvalue':float(vals.min()),'sha256':sha256_file(out/'C_g.npy')},'firewall':{'clean_kd_only':True,'teacher_loaded':False,'none_loaded':False,'bridge_loaded':False,'counterfactual_outcomes_used':False}};(out/'TASK_SENSITIVE_GEOMETRY_FIT.json').write_text(json.dumps(rep,indent=2)+'\n');shutil.rmtree(tmp,ignore_errors=True);print(json.dumps(rep,indent=2))
if __name__=='__main__':main()
