#!/usr/bin/env python
from __future__ import annotations
import argparse,json,shutil,sys,tempfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]; sys.path.insert(0,str(ROOT/'src'))
from tsg_repro.config import load_config,get_path
from tsg_repro.data import load_manifest,SIGMAS
from tsg_repro.yolo_runtime import CORE9_IDS,BDD10_NAMES,IMGSZ,EXPECTED_ULTRALYTICS_VERSION,require_yolo_environment,safe_symlink,import_checkpoint_companion

def stage(part,root,sigma):
 img=root/'images'; lab=root/'labels'; img.mkdir(parents=True); lab.mkdir(parents=True)
 for r in part[part.sigma==sigma].sort_values('source_stem').itertuples(index=False):
  stem=f'{r.source_stem}__sigma{sigma}'; p=Path(r.degraded_image).resolve(); safe_symlink(p,img/f'{stem}{p.suffix.lower()}'); safe_symlink(Path(r.label_file).resolve(),lab/f'{stem}.txt')
 y=root/f'sigma{sigma}.yaml'; y.write_text('path: '+str(root.resolve())+'\ntrain: images\nval: images\nnames:\n'+''.join(f'  {i}: {BDD10_NAMES[i]}\n' for i in range(10))); return y

def main():
 p=argparse.ArgumentParser(); p.add_argument('--config',required=True); p.add_argument('--output',default='results/yolo_frozen_evaluation.json'); p.add_argument('--device',default='0'); a=p.parse_args()
 require_yolo_environment(); from ultralytics import YOLO
 cfg,base=load_config(a.config); import_checkpoint_companion(get_path(cfg,base,'checkpoints','yolo_checkpoint_companion',required=False)); df=load_manifest(get_path(cfg,base,'data','gaussian_manifest')); val=df[df.consrestore_split=='VAL']
 arms={'NONE':get_path(cfg,base,'checkpoints','yolo11s_none'),'CLEAN-KD':get_path(cfg,base,'checkpoints','yolo11s_clean_kd')}
 result={'status':'YOLO_FROZEN_EVALUATION_COMPLETE','endpoint':'CORE9 severity-macro mAP50-95','per_arm':{}}
 tmp=Path(tempfile.mkdtemp(prefix='tsg_eval_'))
 try:
  yamls={s:stage(val,tmp/f'sigma{s}',s) for s in SIGMAS}
  for name,ckpt in arms.items():
   model=YOLO(str(ckpt)); vals=[]; per={}
   for s in SIGMAS:
    m=model.val(data=str(yamls[s]),imgsz=IMGSZ,batch=16,device=a.device,workers=4,plots=False,save=False,verbose=False)
    maps=list(map(float,m.box.maps)); core=float(sum(maps[i] for i in CORE9_IDS)/len(CORE9_IDS)); vals.append(core)
    per[str(s)]={'bdd10_map50_95':float(m.box.map),'bdd10_map50':float(m.box.map50),'core9_map50_95':core}
   result['per_arm'][name]={'per_sigma':per,'core9_severity_macro_map50_95':float(sum(vals)/len(vals))}
 finally: shutil.rmtree(tmp,ignore_errors=True)
 op=Path(a.output); op.parent.mkdir(parents=True,exist_ok=True); op.write_text(json.dumps(result,indent=2)+'\n'); print(json.dumps(result,indent=2))
if __name__=='__main__': main()
