#!/usr/bin/env python
"""Reproduce frozen RT-DETR Experiment-17 Step-14 finite-intervention horizon."""
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
from tsg_repro.native_model import capture_eval_p4,native_loss_and_p4_gradient,native_loss_plain,native_loss_with_cell_replacement,rtdetr_loss_seed,verify_paired_geometry
from tsg_repro.local_utility import historical_mapped_complement_torch,freeze_step14_selection,step14_confirmation

EXPECTED_BRIDGE_SHA='bf9f8f683f0d92a21d0f5a09127dc03d0547cff622446c39975cd6cca9b4eb60'
LOSS_ATOL=5e-6;P4_ATOL=1e-6

def _resolve(root,value):
 p=Path(value);return p.resolve() if p.is_absolute() else (root/p).resolve()

def main():
 p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--step12-cells');p.add_argument('--output',default='results/rtdetr_finite_horizon_confirmation');p.add_argument('--device',default='cuda:0');a=p.parse_args()
 cfg,base=load_config(a.config);mf=get_path(cfg,base,'data','rtdetr_confirm_manifest');views=load_view_manifest(mf,expected_sources=200,expected_views=600)
 if 'view_index' not in views.columns: raise ValueError('RT-DETR confirmation manifest must contain frozen view_index')
 local_step12=ROOT/'results/rtdetr_local_utility_confirmation/RTDETR_CONFIRM_CELL_RECORDS.csv'
 if a.step12_cells:
  cell_path=Path(a.step12_cells).expanduser().resolve()
 elif local_step12.is_file():
  # End-to-end public reproduction must consume the Step-12 records produced
  # by this same package run. A configured external historical file remains a
  # standalone fallback only when the local Step-12 stage has not been run.
  cell_path=local_step12.resolve()
 else:
  cell_path=get_path(cfg,base,'artifacts','rtdetr_step12_cell_records',required=False)
  if cell_path is None: cell_path=local_step12
 cell_path=cell_path.resolve()
 if not cell_path.is_file(): raise FileNotFoundError(f'Step-12 cell records required: {cell_path}')
 cells=pd.read_csv(cell_path);selection=freeze_step14_selection(cells,min_support=4,top_fraction=.25)
 if selection.empty: raise RuntimeError('Step-14 frozen selection is empty')
 out=_resolve(ROOT,a.output)
 if out.exists(): raise FileExistsError(out)
 out.mkdir(parents=True,exist_ok=False);selection.to_csv(out/'RTDETR_HORIZON_CELL_SELECTION_BEFORE_MODEL_ACCESS.csv',index=False)
 view_sel=views[['view_index','source_stem','sigma','generated_stem']].copy();support=cells.groupby('view_index').size().rename('support_n');view_sel=view_sel.merge(support,on='view_index',how='left');view_sel['support_n']=view_sel['support_n'].fillna(0).astype(int);view_sel['eligible_support_ge4']=view_sel['support_n']>=4;view_sel.to_csv(out/'RTDETR_HORIZON_VIEW_SELECTION_BEFORE_MODEL_ACCESS.csv',index=False)
 bridge_path=get_path(cfg,base,'artifacts','rtdetr_step10_bridge')
 if sha256_file(bridge_path)!=EXPECTED_BRIDGE_SHA: raise RuntimeError('historical RT-DETR Step-10 bridge SHA256 mismatch')
 stage=out.parent/f'.{out.name}_stage';shutil.rmtree(stage,ignore_errors=True)
 for r in views.itertuples(index=False):
  stem=str(r.generated_stem);lab=Path(r.label_file).resolve()
  for branch,key in [('degraded','degraded_image'),('clean','clean_image')]:
   src=Path(getattr(r,key)).resolve();safe_symlink(src,stage/branch/'images'/f'{stem}{src.suffix.lower()}');safe_symlink(lab,stage/branch/'labels'/f'{stem}.txt')
 torch=require_rtdetr_environment(require_cuda=a.device.startswith('cuda'));import_checkpoint_companion(get_path(cfg,base,'checkpoints','rtdetr_checkpoint_companion'));device=torch.device(a.device)
 _,teacher=load_model(get_path(cfg,base,'checkpoints','rtdetr_l_clean_teacher'),device);_,none=load_model(get_path(cfg,base,'checkpoints','rtdetr_l_none'),device);_,student=load_model(get_path(cfg,base,'checkpoints','rtdetr_l_clean_kd'),device);bridge=torch.from_numpy(np.load(bridge_path,allow_pickle=False).astype(np.float32,copy=False)).to(device)
 dds=build_dataset(stage/'degraded/images','RTDETR Step14 degraded: ');cds=build_dataset(stage/'clean/images','RTDETR Step14 clean: ');di={Path(x).stem:i for i,x in enumerate(dds.im_files)};ci={Path(x).stem:i for i,x in enumerate(cds.im_files)}
 sel_by_view={int(v):g.sort_values('selection_order',kind='mergesort') for v,g in selection.groupby('view_index',sort=False)};manifest_by_view={int(r.view_index):r for r in views.itertuples(index=False)};outcomes=[];max_uc=0.0;max_ur=0.0;max_loss=0.0;max_p4=0.0
 for j,(vi,sg) in enumerate(sorted(sel_by_view.items())):
  r=manifest_by_view[vi];stem=str(r.generated_stem);db=preprocess_batch(fresh_collate_one(dds,di[stem]),device);cb=preprocess_batch(fresh_collate_one(cds,ci[stem]),device);verify_paired_geometry(db,cb);seed=rtdetr_loss_seed(vi)
  T=capture_eval_p4(teacher,cb['img'],p4_module=24,channels=256,hw=40);N=capture_eval_p4(none,db['img'],p4_module=24,channels=256,hw=40);S_eval=capture_eval_p4(student,db['img'],p4_module=24,channels=256,hw=40);Lplain=native_loss_plain(student,db,initialize_criterion=True,rng_seed=seed);L0,S,G=native_loss_and_p4_gradient(student,db,p4_module=24,channels=256,hw=40,initialize_criterion=True,rng_seed=seed)
  le=abs(Lplain-L0);pe=float(torch.max(torch.abs(S_eval.float()-S.float())).detach().cpu());max_loss=max(max_loss,le);max_p4=max(max_p4,pe)
  if le>LOSS_ATOL: raise RuntimeError(f'Step-14 loss replay error view={vi}: {le}')
  if pe>P4_ATOL: raise RuntimeError(f'Step-14 P4 replay error view={vi}: {pe}')
  for q in sg.itertuples(index=False):
   y=int(q.p4_y);x=int(q.p4_x);s=S[0,:,y,x].float();g=G[0,:,y,x].float();t=T[0,:,y,x].float();n=N[0,:,y,x].float();_,_,m=historical_mapped_complement_torch(n,t,bridge);u_c=float((-torch.dot(g,m)).detach().cpu());u_t=float((-torch.dot(g,t-s)).detach().cpu());u_rel=u_c-u_t;max_uc=max(max_uc,abs(u_c-float(q.frozen_u_C)));max_ur=max(max_ur,abs(u_rel-float(q.frozen_u_rel)))
   Ls=native_loss_with_cell_replacement(student,db,y,x,s+.25*m,p4_module=24,channels=256,hw=40,initialize_criterion=True,rng_seed=seed);Ll=native_loss_with_cell_replacement(student,db,y,x,s+4.0*m,p4_module=24,channels=256,hw=40,initialize_criterion=True,rng_seed=seed)
   ds=L0-Ls;dl=L0-Ll
   if not np.isfinite([ds,dl,u_c,u_rel]).all(): raise RuntimeError(f'non-finite Step-14 result view={vi} cell=({y},{x})')
   outcomes.append({'view_index':vi,'source_stem':str(r.source_stem),'sigma':int(r.sigma),'p4_y':y,'p4_x':x,'selection_order':int(q.selection_order),'frozen_u_C':float(q.frozen_u_C),'frozen_u_rel':float(q.frozen_u_rel),'replayed_u_C':u_c,'replayed_u_rel':u_rel,'Delta_small':ds,'Delta_large':dl,'L0':L0,'L_alpha_0_25':Ls,'L_alpha_4':Ll})
  if (j+1)%20==0: print(f'eligible/selected views processed {j+1}/{len(sel_by_view)}',flush=True)
 outcome_df=pd.DataFrame(outcomes);calc=step14_confirmation(outcome_df,views[['view_index','source_stem','sigma']],reps=5000,seed=2027);pr=calc['primary']
 result={'status':'EXPERIMENT17_STEP14_RTDETR_FINITE_HORIZON_CONFIRM_COMPLETE','scientific_result':True,'population':{'journal_split':'JOURNAL_CONFIRM','sources':200,'views':600,'eligible_support_ge4_views':int(view_sel.eligible_support_ge4.sum()),'selected_cells':int(len(selection)),'primary_defined_views':int(calc['view_results'].primary_defined.sum()),'sources_with_valid_primary_view':int((calc['source_results'].defined_views>0).sum())},'primary_confirmation':{'small_alpha':.25,'large_alpha':4.0,'rho_small_source_balanced':pr['rho_small_source_balanced'],'rho_large_source_balanced':pr['rho_large_source_balanced'],'delta_rho_point':pr['point'],'ci95':pr['ci95'],'bootstrap_reps':5000,'bootstrap_seed':2027,'pass':pr['pass']},'conclusion':'RTDETR_FINITE_INTERVENTION_HORIZON_REPLICATES' if pr['pass'] else 'RTDETR_FINITE_INTERVENTION_HORIZON_NOT_CONFIRMED','decision':'READY_FOR_RTDETR_TASK_SENSITIVE_GEOMETRY_ORIENTATION_REPLICATION','integrity':{'primary_horizon_predictor_from_frozen_step12_u_C':True,'recomputed_step12_scores_used_as_scientific_gate':False,'selection_frozen_before_step14_model_access':True,'selection_replay_max_u_C_abs_error':max_uc,'selection_replay_max_u_rel_abs_error':max_ur,'max_native_loss_replay_error':max_loss,'max_p4_replay_error':max_p4},'provenance':{'confirm_manifest_sha256':sha256_file(mf),'step12_cell_records_sha256':sha256_file(cell_path),'historical_step10_bridge_sha256':sha256_file(bridge_path)}}
 outcome_df.to_csv(out/'RTDETR_HORIZON_SELECTED_CELL_RESULTS.csv',index=False);calc['view_results'].to_csv(out/'RTDETR_HORIZON_VIEW_RESULTS.csv',index=False);calc['source_results'].to_csv(out/'RTDETR_HORIZON_SOURCE_RESULTS.csv',index=False);(out/'RTDETR_FINITE_HORIZON_CONFIRMATION_RESULT.json').write_text(json.dumps(result,indent=2)+'\n');shutil.rmtree(stage,ignore_errors=True);print(json.dumps(result,indent=2))
if __name__=='__main__':main()
