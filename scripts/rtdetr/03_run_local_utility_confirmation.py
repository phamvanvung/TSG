#!/usr/bin/env python
"""Reproduce frozen RT-DETR Experiment-17 Step-12 local-utility confirmation."""
from __future__ import annotations
import argparse,json,shutil,sys
from pathlib import Path
import numpy as np,pandas as pd
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'src'))
from tsg_repro.config import load_config,get_path
from tsg_repro.geometry_workflow import load_view_manifest
from tsg_repro.hashing import sha256_file
from tsg_repro.rtdetr_runtime import *
from tsg_repro.yolo_runtime import import_checkpoint_companion
from tsg_repro.native_model import (capture_eval_p4,core9_object_center_cells,native_loss_and_p4_gradient,native_loss_plain,native_loss_with_cell_replacement,rtdetr_loss_seed,verify_paired_geometry)
from tsg_repro.local_utility import historical_mapped_complement_torch,step12_confirmation

EXPECTED_BRIDGE_SHA='bf9f8f683f0d92a21d0f5a09127dc03d0547cff622446c39975cd6cca9b4eb60'
LOSS_ATOL=5e-6; P4_ATOL=1e-6

def _out(root,value):
 p=Path(value);return p.resolve() if p.is_absolute() else (root/p).resolve()

def main():
 p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--output',default='results/rtdetr_local_utility_confirmation');p.add_argument('--device',default='cuda:0');a=p.parse_args()
 cfg,base=load_config(a.config);mf=get_path(cfg,base,'data','rtdetr_confirm_manifest');views=load_view_manifest(mf,expected_sources=200,expected_views=600)
 if 'view_index' not in views.columns: raise ValueError('RT-DETR confirmation manifest must contain frozen view_index')
 bridge_path=get_path(cfg,base,'artifacts','rtdetr_step10_bridge')
 if sha256_file(bridge_path)!=EXPECTED_BRIDGE_SHA: raise RuntimeError('historical RT-DETR Step-10 bridge SHA256 mismatch')
 out=_out(ROOT,a.output)
 if out.exists(): raise FileExistsError(out)
 stage=out.parent/f'.{out.name}_stage';shutil.rmtree(stage,ignore_errors=True)
 for r in views.itertuples(index=False):
  stem=str(r.generated_stem);lab=Path(r.label_file).resolve()
  for branch,key in [('degraded','degraded_image'),('clean','clean_image')]:
   src=Path(getattr(r,key)).resolve();safe_symlink(src,stage/branch/'images'/f'{stem}{src.suffix.lower()}');safe_symlink(lab,stage/branch/'labels'/f'{stem}.txt')
 torch=require_rtdetr_environment(require_cuda=a.device.startswith('cuda'));import_checkpoint_companion(get_path(cfg,base,'checkpoints','rtdetr_checkpoint_companion'));device=torch.device(a.device)
 _,teacher=load_model(get_path(cfg,base,'checkpoints','rtdetr_l_clean_teacher'),device);_,none=load_model(get_path(cfg,base,'checkpoints','rtdetr_l_none'),device);_,student=load_model(get_path(cfg,base,'checkpoints','rtdetr_l_clean_kd'),device)
 bridge=torch.from_numpy(np.load(bridge_path,allow_pickle=False).astype(np.float32,copy=False)).to(device)
 if tuple(bridge.shape)!=(256,256): raise RuntimeError('historical bridge shape changed')
 dds=build_dataset(stage/'degraded/images','RTDETR Step12 degraded: ');cds=build_dataset(stage/'clean/images','RTDETR Step12 clean: ')
 di={Path(x).stem:i for i,x in enumerate(dds.im_files)};ci={Path(x).stem:i for i,x in enumerate(cds.im_files)}
 if len(di)!=600 or len(ci)!=600: raise RuntimeError('Step-12 staged view count changed')
 rows=[];max_loss_replay=0.0;max_p4_replay=0.0
 for j,r in enumerate(views.itertuples(index=False)):
  stem=str(r.generated_stem);db=preprocess_batch(fresh_collate_one(dds,di[stem]),device);cb=preprocess_batch(fresh_collate_one(cds,ci[stem]),device);verify_paired_geometry(db,cb);seed=rtdetr_loss_seed(int(r.view_index))
  T=capture_eval_p4(teacher,cb['img'],p4_module=24,channels=256,hw=40);N=capture_eval_p4(none,db['img'],p4_module=24,channels=256,hw=40);S_eval=capture_eval_p4(student,db['img'],p4_module=24,channels=256,hw=40)
  Lplain=native_loss_plain(student,db,initialize_criterion=True,rng_seed=seed);L0,S,G=native_loss_and_p4_gradient(student,db,p4_module=24,channels=256,hw=40,initialize_criterion=True,rng_seed=seed)
  le=abs(Lplain-L0);pe=float(torch.max(torch.abs(S_eval.float()-S.float())).detach().cpu());max_loss_replay=max(max_loss_replay,le);max_p4_replay=max(max_p4_replay,pe)
  if le>LOSS_ATOL: raise RuntimeError(f'native loss replay error view={r.view_index}: {le}')
  if pe>P4_ATOL: raise RuntimeError(f'P4 replay error view={r.view_index}: {pe}')
  cells=core9_object_center_cells(db,hw=40)
  if not cells: raise RuntimeError(f'no CORE9 support cells: {stem}')
  for class_id,y,x in cells:
   s=S[0,:,y,x].float();g=G[0,:,y,x].float();t=T[0,:,y,x].float();n=N[0,:,y,x].float();_,_,m=historical_mapped_complement_torch(n,t,bridge);d_t=t-s
   u_t=float((-torch.dot(g,d_t)).detach().cpu());u_c=float((-torch.dot(g,m)).detach().cpu());u_rel=u_c-u_t
   L_T=native_loss_with_cell_replacement(student,db,y,x,t,p4_module=24,channels=256,hw=40,initialize_criterion=True,rng_seed=seed)
   L_C=native_loss_with_cell_replacement(student,db,y,x,s+m,p4_module=24,channels=256,hw=40,initialize_criterion=True,rng_seed=seed)
   dt=L0-L_T;dc=L0-L_C;Y=dc-dt
   vals=[u_t,u_c,u_rel,L0,L_T,L_C,dt,dc,Y]
   if not np.isfinite(vals).all(): raise RuntimeError(f'non-finite Step-12 cell result: {stem} ({y},{x})')
   rows.append({'view_index':int(r.view_index),'source_stem':str(r.source_stem),'sigma':int(r.sigma),'generated_stem':stem,'class_id_representative':int(class_id),'p4_y':int(y),'p4_x':int(x),'u_T':u_t,'u_C':u_c,'u_rel':u_rel,'L0':L0,'L_T':L_T,'L_C':L_C,'Delta_T':dt,'Delta_C':dc,'Y':Y,'norm_m':float(torch.linalg.vector_norm(m).detach().cpu()),'norm_g':float(torch.linalg.vector_norm(g).detach().cpu())})
  if (j+1)%20==0: print(f'views {j+1}/600  cells {len(rows)}',flush=True)
 rec=pd.DataFrame(rows)
 if rec['view_index'].nunique()!=600 or rec['source_stem'].nunique()!=200: raise RuntimeError('incomplete Step-12 local-utility population')
 calc=step12_confirmation(rec,reps=5000,seed=2027);ga=calc['gate_A'];gb=calc['gate_B'];advance=calc['advance']
 result={'status':'EXPERIMENT17_STEP12_RTDETR_MAPPED_COMPLEMENT_LOCAL_UTILITY_CONFIRM_COMPLETE','scientific_result':True,'population':{'journal_split':'JOURNAL_CONFIRM','sources':200,'views':600,'cell_records':int(len(rec))},'primary_gate_A':{'definition':'mean source-level within-view Spearman(u_rel,Y)','point':ga['point'],'ci95':ga['ci95'],'bootstrap_reps':5000,'bootstrap_seed':2027,'pass':ga['pass']},'primary_gate_B':{'definition':'mean source-level within-view top25%-bottom25% exact-Y enrichment ranked by u_rel','point':gb['point'],'ci95':gb['ci95'],'bootstrap_reps':5000,'bootstrap_seed':2027,'sources_with_eligible_views':gb['source_count'],'pass':gb['pass']},'advance':advance,'conclusion':'RTDETR_MAPPED_COMPLEMENT_LOCAL_UTILITY_REPLICATES' if advance else 'RTDETR_MAPPED_COMPLEMENT_LOCAL_UTILITY_NOT_CONFIRMED','decision':'READY_FOR_RTDETR_FINITE_INTERVENTION_HORIZON' if advance else 'CLOSE_RTDETR_MAPPED_COMPLEMENT_LOCAL_UTILITY_REPLICATION_NO_TUNING','implementation_integrity':{'all_600_views_present':True,'max_native_loss_replay_error':max_loss_replay,'max_p4_replay_error':max_p4_replay,'primary_bridge':'historical Step-10 NONE->CLEAN-KD','teacher_canonical_correction_applied':False},'provenance':{'confirm_manifest_sha256':sha256_file(mf),'historical_step10_bridge_sha256':sha256_file(bridge_path)}}
 out.mkdir(parents=True,exist_ok=False);rec.to_csv(out/'RTDETR_CONFIRM_CELL_RECORDS.csv',index=False);calc['view_stats'].to_csv(out/'RTDETR_CONFIRM_VIEW_STATS.csv',index=False);calc['source_stats'].to_csv(out/'RTDETR_CONFIRM_SOURCE_STATS.csv',index=False);(out/'RTDETR_MAPPED_COMPLEMENT_LOCAL_UTILITY_RESULT.json').write_text(json.dumps(result,indent=2)+'\n')
 shutil.rmtree(stage,ignore_errors=True);print(json.dumps(result,indent=2))
if __name__=='__main__':main()
