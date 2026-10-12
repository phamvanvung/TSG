"""Shared table/statistics helpers for checkpoint-level geometry confirmations."""
from __future__ import annotations
import numpy as np
import pandas as pd
from .statistics import cluster_bootstrap, partial_spearman, spearman

REQUIRED_VIEW_COLUMNS={"source_stem","sigma","clean_image","degraded_image","label_file"}

def load_view_manifest(path, *, expected_sources=None, expected_views=None):
    df=pd.read_csv(path)
    missing=REQUIRED_VIEW_COLUMNS-set(df.columns)
    if missing: raise ValueError(f"view manifest missing columns: {sorted(missing)}")
    df["source_stem"]=df["source_stem"].astype(str)
    df["sigma"]=pd.to_numeric(df["sigma"],errors="raise").astype(int)
    if expected_views is not None and len(df)!=expected_views: raise ValueError(f"expected {expected_views} views; got {len(df)}")
    if expected_sources is not None and df.source_stem.nunique()!=expected_sources: raise ValueError(f"expected {expected_sources} sources; got {df.source_stem.nunique()}")
    if set(df.sigma.unique())!={15,25,50}: raise ValueError("expected sigma set {15,25,50}")
    if expected_sources is not None and not (df.groupby("source_stem").size()==3).all(): raise ValueError("each source must contribute exactly 3 views")
    if "generated_stem" not in df.columns: df["generated_stem"]=[f"{s}__sigma{g}" for s,g in zip(df.source_stem,df.sigma)]
    if "view_index" in df.columns:
        df["view_index"]=pd.to_numeric(df["view_index"],errors="raise").astype(int)
        if df["view_index"].duplicated().any(): raise ValueError("view_index must be unique")
        return df.sort_values("view_index",kind="mergesort").reset_index(drop=True)
    return df.sort_values(["source_stem","sigma"],kind="mergesort").reset_index(drop=True)

def orientation_confirmation(records: pd.DataFrame, *, q="Q_M", y="Y_M", norm="norm_M", reps=5000, seed=2027):
    vals={"q":records[q].to_numpy(float),"y":records[y].to_numpy(float),"norm":records[norm].to_numpy(float)}
    clusters=records.source_stem.astype(str).to_numpy()
    a=cluster_bootstrap(vals,clusters,lambda d:spearman(d["q"],d["y"]),reps=reps,seed=seed)
    b=cluster_bootstrap(vals,clusters,lambda d:partial_spearman(d["q"],d["y"],d["norm"]),reps=reps,seed=seed)
    a["pass"]=bool(a["point"]>0 and a["ci_low"]>0)
    b["pass"]=bool(b["point"]>0 and b["ci_low"]>0)
    return {"gate_A_spearman":a,"gate_B_partial_spearman":b}
