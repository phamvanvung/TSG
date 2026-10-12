"""Frozen RT-DETR-L dataset/model helpers for the public reproduction chain."""
from __future__ import annotations
import copy,gc,os
from pathlib import Path

BDD10_NAMES={0:'bike',1:'bus',2:'car',3:'motor',4:'person',5:'rider',6:'traffic light',7:'traffic sign',8:'train',9:'truck'}
IMGSZ=640; P4_MODULE=24; P4_CHANNELS=256; P4_HW=40

def require_rtdetr_environment(require_cuda=True):
 import torch,ultralytics
 if str(ultralytics.__version__)!='8.4.131': raise RuntimeError(f'Expected ultralytics 8.4.131; found {ultralytics.__version__}')
 if require_cuda and not torch.cuda.is_available(): raise RuntimeError('CUDA required for locked RT-DETR reproduction')
 return torch

def clear_cuda():
 import torch
 gc.collect()
 if torch.cuda.is_available():
  torch.cuda.empty_cache()
  try: torch.cuda.ipc_collect()
  except Exception: pass

def safe_symlink(target:Path,link:Path):
 target=target.expanduser().resolve()
 if not target.is_file(): raise FileNotFoundError(target)
 link.parent.mkdir(parents=True,exist_ok=True)
 if link.exists() or link.is_symlink():
  if not link.is_symlink() or link.resolve()!=target: raise RuntimeError(f'staged link mismatch: {link}')
  return
 os.symlink(str(target),str(link))

def build_dataset(image_dir:Path,prefix='RTDETR repro: '):
 from ultralytics.cfg import get_cfg
 from ultralytics.models.rtdetr.val import RTDETRDataset
 hyp=get_cfg(); hyp.imgsz=IMGSZ; hyp.seed=2027; hyp.deterministic=False; hyp.amp=False; hyp.cache=False; hyp.single_cls=False; hyp.classes=None
 return RTDETRDataset(img_path=str(image_dir.resolve()),imgsz=IMGSZ,batch_size=1,augment=False,hyp=copy.deepcopy(hyp),rect=False,cache=None,single_cls=False,prefix=prefix,classes=None,data={'names':BDD10_NAMES,'nc':10,'channels':3},fraction=1.0)

def fresh_collate_one(dataset,index):
 item=copy.deepcopy(dataset[int(index)])
 return dataset.collate_fn([item])

def preprocess_batch(batch,device):
 import torch
 out={}
 for k,v in batch.items(): out[k]=v.to(device,non_blocking=False) if isinstance(v,torch.Tensor) else v
 if 'img' not in out: raise RuntimeError('RT-DETR batch missing img')
 image=out['img']
 if not torch.is_floating_point(image): image=image.float()
 if float(image.max())>1.0: image=image/255.0
 out['img']=image
 return out

def load_model(path:Path,device):
 from ultralytics import RTDETR
 wrapper=RTDETR(str(path)); model=wrapper.model.float().to(device).eval()
 for p in model.parameters(): p.requires_grad_(False)
 if len(model.model)<=P4_MODULE: raise RuntimeError('RT-DETR checkpoint lacks module 24')
 if int(getattr(model.model[-1],'nc',-1))!=10: raise RuntimeError('Expected BDD10 RT-DETR checkpoint')
 return wrapper,model
