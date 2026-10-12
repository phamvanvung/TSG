#!/usr/bin/env python
from __future__ import annotations
import argparse,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]; sys.path.insert(0,str(ROOT/'src'))
from tsg_repro.config import load_config,get_path
from tsg_repro.data import load_manifest,validate_manifest_contract,audit_pixel_regeneration

def main():
 p=argparse.ArgumentParser(); p.add_argument('--config',required=True); p.add_argument('--pixel-audit',action='store_true'); p.add_argument('--output',default='results/yolo_gaussian_audit.json'); a=p.parse_args()
 cfg,base=load_config(a.config); manifest=get_path(cfg,base,'data','gaussian_manifest')
 df=load_manifest(manifest); out={'status':'YOLO_GAUSSIAN_MANIFEST_AUDIT_PASS','counts':validate_manifest_contract(df),'pixel_audit':None}
 if a.pixel_audit: out['pixel_audit']=audit_pixel_regeneration(df)
 op=Path(a.output); op.parent.mkdir(parents=True,exist_ok=True); op.write_text(json.dumps(out,indent=2)+'\n'); print(op.resolve())
if __name__=='__main__': main()
