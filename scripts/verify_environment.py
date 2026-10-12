#!/usr/bin/env python
"""Capture and verify the runtime against the v1 release environment contract."""
from __future__ import annotations
import argparse,json,platform,sys
from pathlib import Path

def ver(name):
 try:
  m=__import__(name);return getattr(m,'__version__',None)
 except Exception:return None

def capture():
 r={'python':sys.version.split()[0],'platform':platform.platform(),'numpy':ver('numpy'),'scipy':ver('scipy'),'pandas':ver('pandas'),'torch':ver('torch'),'ultralytics':ver('ultralytics')}
 try:
  import torch
  r['cuda_available']=bool(torch.cuda.is_available());r['torch_cuda']=torch.version.cuda;r['cudnn']=int(torch.backends.cudnn.version()) if torch.backends.cudnn.is_available() else None;r['gpu']=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
 except Exception:r['cuda_available']=None
 return r

def main():
 p=argparse.ArgumentParser();p.add_argument('--requirements',default=str(Path(__file__).resolve().parents[1]/'configs/environment.requirements.json'));p.add_argument('--output');p.add_argument('--require-cuda',action='store_true');a=p.parse_args();req=json.loads(Path(a.requirements).read_text());obs=capture();m=[]
 for k,v in req.get('required_exact',{}).items():
  if obs.get(k)!=v:m.append({'field':k,'expected':v,'observed':obs.get(k)})
 for k,v in req.get('freeze_from_successful_reference_run',{}).items():
  if v is not None and obs.get(k)!=v:m.append({'field':k,'expected':v,'observed':obs.get(k)})
 if a.require_cuda and obs.get('cuda_available') is not True:m.append({'field':'cuda_available','expected':True,'observed':obs.get('cuda_available')})
 report={'status':'PASS' if not m else 'FAIL','observed':obs,'mismatches':m,'contract_complete':all(v is not None for v in req.get('freeze_from_successful_reference_run',{}).values())}
 text=json.dumps(report,indent=2)+'\n';print(text,end='')
 if a.output:Path(a.output).write_text(text)
 raise SystemExit(0 if not m else 2)
if __name__=='__main__':main()
