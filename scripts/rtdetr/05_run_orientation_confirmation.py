#!/usr/bin/env python
"""RT-DETR-L checkpoint/native-loss mapped-complement TSG orientation confirmation."""
from __future__ import annotations
import argparse,json,shutil,sys
from pathlib import Path
import numpy as np,pandas as pd
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'src'))
from tsg_repro.config import load_config,get_path
from tsg_repro.geometry_workflow import load_view_manifest,orientation_confirmation
from tsg_repro.rtdetr_runtime import *
from tsg_repro.yolo_runtime import import_checkpoint_companion
from tsg_repro.native_model import native_loss_plain,native_loss_with_p4_direction,capture_eval_p4,core9_object_center_cells,rtdetr_loss_seed,verify_paired_geometry
from tsg_repro.native_geometry import canonical_mapped_complement_cells_torch,task_sensitive_orientation,direction_norm,support_direction_tensor_torch
from tsg_repro.hashing import sha256_file

def main():
 p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--output',default='results/rtdetr_orientation_confirmation');p.add_argument('--device',default='cuda:0');a=p.parse_args();cfg,base=load_config(a.config);mf=get_path(cfg,base,'data','rtdetr_confirm_manifest');df=load_view_manifest(mf,expected_sources=200,expected_views=600);
 if 'view_index' not in df.columns: raise ValueError('RT-DETR confirmation manifest must contain frozen view_index')
 out=(ROOT/Path(a.output)).resolve() if not Path(a.output).is_absolute() else Path(a.output).resolve();
 if out.exists():raise FileExistsError(out)
 tmp=out.parent/f'.{out.name}_stage';shutil.rmtree(tmp,ignore_errors=True)
 for r in df.itertuples(index=False):
  stem=str(r.generated_stem);lab=Path(r.label_file).resolve()
  for branch,key in [('degraded','degraded_image'),('clean','clean_image')]:src=Path(getattr(r,key)).resolve();safe_symlink(src,tmp/branch/'images'/f'{stem}{src.suffix.lower()}');safe_symlink(lab,tmp/branch/'labels'/f'{stem}.txt')
 torch=require_rtdetr_environment(require_cuda=a.device.startswith('cuda'));import_checkpoint_companion(get_path(cfg,base,'checkpoints','rtdetr_checkpoint_companion',required=False));device=torch.device(a.device);_,teacher=load_model(get_path(cfg,base,'checkpoints','rtdetr_l_clean_teacher'),device);_,none=load_model(get_path(cfg,base,'checkpoints','rtdetr_l_none'),device);_,student=load_model(get_path(cfg,base,'checkpoints','rtdetr_l_clean_kd'),device);dds=build_dataset(tmp/'degraded/images','RTDETR confirm degraded: ');cds=build_dataset(tmp/'clean/images','RTDETR confirm clean: ');di={Path(x).stem:i for i,x in enumerate(dds.im_files)};ci={Path(x).stem:i for i,x in enumerate(cds.im_files)};maps=get_path(cfg,base,'artifacts','rtdetr_canonical_map_dir');rnt=torch.from_numpy(np.load(maps/'R_NT_NONE_TO_TEACHER.npy').astype(np.float32,copy=False)).to(device);rst=torch.from_numpy(np.load(maps/'R_ST_CLEAN_KD_TO_TEACHER.npy').astype(np.float32,copy=False)).to(device);cg_path=get_path(cfg,base,'artifacts','rtdetr_task_geometry');cg=np.load(cg_path);rows=[]
 for j,r in enumerate(df.itertuples(index=False)):
  stem=str(r.generated_stem);db=preprocess_batch(fresh_collate_one(dds,di[stem]),device);cb=preprocess_batch(fresh_collate_one(cds,ci[stem]),device);verify_paired_geometry(db,cb);cells=core9_object_center_cells(db,hw=40);coords=[(y,x) for _,y,x in cells]
  if not coords:raise RuntimeError(f'No CORE9 support cells for RT-DETR confirmation view {stem}')
  tp=capture_eval_p4(teacher,cb['img'],p4_module=24,channels=256,hw=40);np4=capture_eval_p4(none,db['img'],p4_module=24,channels=256,hw=40);sp=capture_eval_p4(student,db['img'],p4_module=24,channels=256,hw=40);tc=torch.stack([tp[0,:,y,x] for y,x in coords]);nc=torch.stack([np4[0,:,y,x] for y,x in coords]);mc_t=canonical_mapped_complement_cells_torch(nc,tc,rnt,rst);mc=mc_t.detach().cpu().numpy().astype(np.float64,copy=False);q=task_sensitive_orientation(mc,cg);norm=direction_norm(mc);d=support_direction_tensor_torch(mc_t,coords,channels=256,hw=40);seed=rtdetr_loss_seed(int(r.view_index));l0=native_loss_plain(student,db,initialize_criterion=True,rng_seed=seed);la=native_loss_with_p4_direction(student,db,d,.25,p4_module=24,initialize_criterion=True,rng_seed=seed);rows.append({'source_stem':str(r.source_stem),'sigma':int(r.sigma),'support_cells':len(coords),'Q_M':q,'norm_M':norm,'L0':l0,'L_alpha_0_25':la,'Y_M':abs(l0-la)})
  if (j+1)%25==0:print(f'views {j+1}/{len(df)}',flush=True)
 rec=pd.DataFrame(rows);
 if len(rec)!=600 or rec.source_stem.nunique()!=200:raise RuntimeError(f'Incomplete RT-DETR confirmation: views={len(rec)} sources={rec.source_stem.nunique()}')
 out.mkdir(parents=True,exist_ok=False);rec.to_csv(out/'PER_VIEW.csv',index=False);g=orientation_confirmation(rec,reps=5000,seed=2027);result={'status':'RTDETR_TASK_SENSITIVE_ORIENTATION_CONFIRMATION_COMPLETE','sources':int(rec.source_stem.nunique()),'views':int(len(rec)),'alpha':.25,'primary_confirmation':g['gate_B_partial_spearman'],'ordinary_spearman_descriptive':g['gate_A_spearman'],'provenance':{'confirm_manifest_sha256':sha256_file(mf),'Cg_sha256':sha256_file(cg_path),'R_NT_sha256':sha256_file(maps/'R_NT_NONE_TO_TEACHER.npy'),'R_ST_sha256':sha256_file(maps/'R_ST_CLEAN_KD_TO_TEACHER.npy')}};(out/'RTDETR_TASK_SENSITIVE_ORIENTATION_RESULT.json').write_text(json.dumps(result,indent=2)+'\n');shutil.rmtree(tmp,ignore_errors=True);print(json.dumps(result,indent=2))
if __name__=='__main__':main()
