"""Reusable staging, feature-map fitting, and frozen-model evaluation helpers."""
from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from .data import load_manifest
from .hashing import sha256_file
from .mapping import fit_orthogonal_map, orthogonality_error, identity_distance
from .yolo_runtime import P4_MODULE, P4_CHANNELS, P4_HW, safe_symlink, require_yolo_environment


def stage_manifest_views(manifest: pd.DataFrame, root: Path, split: str = "VAL") -> pd.DataFrame:
    part=manifest[manifest["consrestore_split"].astype(str).str.upper()==split.upper()].copy()
    rows=[]
    for rec in part.sort_values(["source_stem","sigma"]).itertuples(index=False):
        stem=f"{rec.source_stem}__sigma{int(rec.sigma)}"
        for branch,image in (("degraded",rec.degraded_image),("clean",rec.clean_image)):
            p=Path(image).resolve(); safe_symlink(p,root/branch/"images"/f"{stem}{p.suffix.lower()}")
            safe_symlink(Path(rec.label_file).resolve(),root/branch/"labels"/f"{stem}.txt")
        rows.append({"source_stem":str(rec.source_stem),"sigma":int(rec.sigma),"generated_stem":stem,"label_file":str(Path(rec.label_file).resolve())})
    return pd.DataFrame(rows)


def _load_yolo(path: Path, device):
    from ultralytics import YOLO
    wrapper=YOLO(str(path)); model=wrapper.model.float().to(device).eval()
    for p in model.parameters(): p.requires_grad_(False)
    return wrapper,model


def fit_teacher_canonical_maps_streaming(
    manifest_csv: Path,
    teacher_ckpt: Path,
    none_ckpt: Path,
    clean_kd_ckpt: Path,
    none_args_yaml: Path,
    output_dir: Path,
    *,
    device_name: str="cuda:0",
    checkpoint_companion: Path | None = None,
) -> dict:
    """Fit R_NT and R_ST from all TRAIN P4 cells without storing giant feature arrays."""
    torch=require_yolo_environment(require_cuda=device_name.startswith("cuda"))
    from .yolo_runtime import load_hyp,build_dataset,preprocess_one,capture_modules,remove_handles,clear_cuda,import_checkpoint_companion
    import_checkpoint_companion(checkpoint_companion)
    manifest=load_manifest(manifest_csv)
    train=manifest[manifest.consrestore_split=="TRAIN"].sort_values(["source_stem","sigma"]).reset_index(drop=True)
    if len(train)!=4500 or train.source_stem.nunique()!=1500:
        raise RuntimeError("Expected 1500 TRAIN sources / 4500 TRAIN views")
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite canonical map directory: {output_dir}")
    tmp=output_dir.parent/("."+output_dir.name+"_stage")
    if tmp.exists(): shutil.rmtree(tmp)
    staged=stage_manifest_views(manifest,tmp,"TRAIN")
    hyp=load_hyp(none_args_yaml)
    deg=build_dataset(tmp/"degraded/images",hyp,"canonical-map degraded: ")
    clean=build_dataset(tmp/"clean/images",hyp,"canonical-map clean: ")
    di={Path(p).stem:i for i,p in enumerate(deg.im_files)}; ci={Path(p).stem:i for i,p in enumerate(clean.im_files)}
    device=torch.device(device_name); clear_cuda()
    _,teacher=_load_yolo(teacher_ckpt,device); _,none=_load_yolo(none_ckpt,device); _,student=_load_yolo(clean_kd_ckpt,device)
    cross_nt=np.zeros((P4_CHANNELS,P4_CHANNELS),dtype=np.float64)
    cross_st=np.zeros_like(cross_nt); count=0
    for k,row in staged.iterrows():
        stem=row.generated_stem; db=preprocess_one(deg[di[stem]],device); cb=preprocess_one(clean[ci[stem]],device)
        stores=[]
        for model,batch in ((teacher,cb),(none,db),(student,db)):
            store,handles=capture_modules(model,[P4_MODULE])
            with torch.inference_mode(): _=model(batch["img"])
            remove_handles(handles); stores.append(store[P4_MODULE])
        t,n,s=stores
        for x in (t,n,s):
            if tuple(x.shape)!=(1,P4_CHANNELS,*P4_HW): raise RuntimeError(f"Unexpected P4 shape {tuple(x.shape)}")
        T=t[0].permute(1,2,0).reshape(-1,P4_CHANNELS).detach().cpu().numpy().astype(np.float64)
        N=n[0].permute(1,2,0).reshape(-1,P4_CHANNELS).detach().cpu().numpy().astype(np.float64)
        S=s[0].permute(1,2,0).reshape(-1,P4_CHANNELS).detach().cpu().numpy().astype(np.float64)
        cross_nt += N.T@T; cross_st += S.T@T; count += len(T)
        if (k+1)%100==0: print(f"map-fit views {k+1}/{len(staged)}")
    def solve(cross):
        u,_,vt=np.linalg.svd(cross,full_matrices=False); return (u@vt).astype(np.float64)
    r_nt=solve(cross_nt); r_st=solve(cross_st)
    output_dir.mkdir(parents=True,exist_ok=False)
    # Preserve the frozen public artifact convention: maps are serialized as float64.
    # Runtime intervention code casts them to model dtype (float32) only after loading.
    np.save(output_dir/"R_NT_NONE_TO_TEACHER.npy",r_nt.astype(np.float64))
    np.save(output_dir/"R_ST_CLEAN_KD_TO_TEACHER.npy",r_st.astype(np.float64))
    report={"status":"TSG_TEACHER_CANONICAL_COORDINATE_AUDIT_COMPLETE","scope":{"train_used":True,"val_used":False,"pres_lost_labels_used":False},"fit":{"views":int(len(staged)),"vectors":int(count),"channels":P4_CHANNELS},"maps":{}}
    for key,r,name in (("R_NT",r_nt,"R_NT_NONE_TO_TEACHER.npy"),("R_ST",r_st,"R_ST_CLEAN_KD_TO_TEACHER.npy")):
        report["maps"][key]={"file":name,"sha256":sha256_file(output_dir/name),"orthogonality_error":orthogonality_error(r),"identity_distance":identity_distance(r)}
    (output_dir/"TEACHER_CANONICAL_COORDINATE_AUDIT.json").write_text(json.dumps(report,indent=2)+"\n")
    shutil.rmtree(tmp,ignore_errors=True)
    return report
