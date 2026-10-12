#!/usr/bin/env python
"""Checkpoint/native-loss YOLO11s teacher-canonical Exp15A orientation confirmation."""
from __future__ import annotations
import argparse,json,shutil,sys
from pathlib import Path
import numpy as np,pandas as pd
ROOT=Path(__file__).resolve().parents[2]; sys.path.insert(0,str(ROOT/'src'))
from tsg_repro.config import load_config,get_path
from tsg_repro.geometry_workflow import load_view_manifest,orientation_confirmation
from tsg_repro.native_geometry import canonical_mapped_complement_cells_torch,task_sensitive_orientation,direction_norm,support_direction_tensor_torch
from tsg_repro.native_model import native_loss_plain,native_loss_with_p4_direction,capture_eval_p4,core9_object_center_cells,verify_paired_geometry
from tsg_repro.yolo_runtime import import_checkpoint_companion,load_hyp,build_dataset,preprocess_one,safe_symlink,require_yolo_environment,P4_MODULE,P4_CHANNELS,P4_HW,configure_native_replay_model
from tsg_repro.hashing import sha256_file

def main():
 p=argparse.ArgumentParser(); p.add_argument('--config',required=True); p.add_argument('--output',default='results/yolo_exp15a_confirmation'); p.add_argument('--device',default='cuda:0'); a=p.parse_args(); cfg,base=load_config(a.config)
 mf=get_path(cfg,base,'data','yolo_tsg_confirm_manifest'); df=load_view_manifest(mf,expected_sources=120,expected_views=360); out=(ROOT/Path(a.output)).resolve() if not Path(a.output).is_absolute() else Path(a.output).resolve();
 if out.exists(): raise FileExistsError(out)
 tmp=out.parent/f'.{out.name}_stage'; shutil.rmtree(tmp,ignore_errors=True)
 for r in df.itertuples(index=False):
  stem=str(r.generated_stem); lab=Path(r.label_file).resolve()
  for branch,key in [('degraded','degraded_image'),('clean','clean_image')]:
   src=Path(getattr(r,key)).resolve(); safe_symlink(src,tmp/branch/'images'/f'{stem}{src.suffix.lower()}'); safe_symlink(lab,tmp/branch/'labels'/f'{stem}.txt')
 torch=require_yolo_environment(require_cuda=a.device.startswith('cuda')); import_checkpoint_companion(get_path(cfg,base,'checkpoints','yolo_checkpoint_companion',required=False)); from ultralytics import YOLO
 device=torch.device(a.device); teacher=YOLO(str(get_path(cfg,base,'checkpoints','yolo11s_clean_teacher'))).model.float().to(device).eval(); none=YOLO(str(get_path(cfg,base,'checkpoints','yolo11s_none'))).model.float().to(device).eval(); student=YOLO(str(get_path(cfg,base,'checkpoints','yolo11s_clean_kd'))).model.float().to(device).eval()
 for m in (teacher,none):
  for q in m.parameters(): q.requires_grad_(False)
 hyp=load_hyp(get_path(cfg,base,'data','yolo_none_args_yaml')); configure_native_replay_model(student,hyp); dds=build_dataset(tmp/'degraded/images',hyp,'YOLO Exp15A degraded: '); cds=build_dataset(tmp/'clean/images',hyp,'YOLO Exp15A clean: '); di={Path(x).stem:i for i,x in enumerate(dds.im_files)}; ci={Path(x).stem:i for i,x in enumerate(cds.im_files)}
 maps=get_path(cfg,base,'artifacts','canonical_map_dir'); rnt=torch.from_numpy(np.load(maps/'R_NT_NONE_TO_TEACHER.npy').astype(np.float32,copy=False)).to(device); rst=torch.from_numpy(np.load(maps/'R_ST_CLEAN_KD_TO_TEACHER.npy').astype(np.float32,copy=False)).to(device); cg_path=get_path(cfg,base,'artifacts','yolo_task_geometry',required=False) or (ROOT/'results/yolo_task_geometry/C_g.npy'); cg=np.load(cg_path)
 records=[]
 for j,r in enumerate(df.itertuples(index=False)):
  stem=str(r.generated_stem); db=preprocess_one(dds[di[stem]],device); cb=preprocess_one(cds[ci[stem]],device); verify_paired_geometry(db,cb); cells=core9_object_center_cells(db,hw=P4_HW[0]); coords=[(y,x) for _,y,x in cells]
  if len(coords)==0: raise RuntimeError(f'No CORE9 support cells for confirmation view {stem}')
  tp=capture_eval_p4(teacher,cb['img'],p4_module=P4_MODULE,channels=P4_CHANNELS,hw=P4_HW[0]); np4=capture_eval_p4(none,db['img'],p4_module=P4_MODULE,channels=P4_CHANNELS,hw=P4_HW[0]); sp=capture_eval_p4(student,db['img'],p4_module=P4_MODULE,channels=P4_CHANNELS,hw=P4_HW[0])
  tc=torch.stack([tp[0,:,y,x] for y,x in coords]); nc=torch.stack([np4[0,:,y,x] for y,x in coords]); mc_t=canonical_mapped_complement_cells_torch(nc,tc,rnt,rst); mc=mc_t.detach().cpu().numpy().astype(np.float64,copy=False)
  q=task_sensitive_orientation(mc,cg); norm=direction_norm(mc); d=support_direction_tensor_torch(mc_t,coords,channels=P4_CHANNELS,hw=P4_HW[0]); l0=native_loss_plain(student,db); l025=native_loss_with_p4_direction(student,db,d,.25,p4_module=P4_MODULE); l1=native_loss_with_p4_direction(student,db,d,1.0,p4_module=P4_MODULE)
  records.append({'source_stem':str(r.source_stem),'sigma':int(r.sigma),'generated_stem':stem,'support_cells':len(coords),'Q_M':q,'norm_M':norm,'L0':l0,'L_alpha_0_25':l025,'L_alpha_1':l1,'Delta_0_25':l0-l025,'Delta_1':l0-l1,'Y_M':abs(l0-l025)})
  if (j+1)%20==0: print(f'views {j+1}/{len(df)}',flush=True)
 rec=pd.DataFrame(records);
 if len(rec)!=360 or rec.source_stem.nunique()!=120: raise RuntimeError(f'Incomplete Exp15A confirmation: views={len(rec)} sources={rec.source_stem.nunique()}')
 out.mkdir(parents=True,exist_ok=False); rec.to_csv(out/'PER_VIEW.csv',index=False); gates=orientation_confirmation(rec,reps=5000,seed=2027)
 result={'status':'YOLO11S_EXP15A_TSG_CONFIRMATION_COMPLETE','sources':int(rec.source_stem.nunique()),'views':int(len(rec)),'alpha_local':.25,'alpha_full_descriptive':1.0,'support':'all unique CORE9 object-center P4 cells','primary_confirmation':gates,'advance':bool(gates['gate_A_spearman']['pass'] and gates['gate_B_partial_spearman']['pass']),'provenance':{'confirm_manifest_sha256':sha256_file(mf),'Cg_sha256':sha256_file(cg_path),'R_NT_sha256':sha256_file(maps/'R_NT_NONE_TO_TEACHER.npy'),'R_ST_sha256':sha256_file(maps/'R_ST_CLEAN_KD_TO_TEACHER.npy')}}
 (out/'EXP15A_TSG_CONFIRMATION_RESULT.json').write_text(json.dumps(result,indent=2)+'\n'); shutil.rmtree(tmp,ignore_errors=True); print(json.dumps(result,indent=2))
if __name__=='__main__':
 import torch
 main()
