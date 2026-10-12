#!/usr/bin/env python
"""Frozen RT-DETR-L Gaussian VAL detector-performance reproduction.

Reproduces the paper's CORE9 severity-macro mAP50-95 comparison between
RTDETR-NONE and RTDETR-CLEAN-KD and its paired 250-source bootstrap.
No training, tuning, checkpoint reselection, or journal-cohort access.
"""
from __future__ import annotations

import argparse
import gc
import importlib.util
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "src"))
from tsg_repro.config import get_path, load_config
from tsg_repro.data import load_manifest, validate_manifest_contract
from tsg_repro.detection_eval import (
    BOOTSTRAP_REPS,
    BOOTSTRAP_SEED,
    CORE9_IDS,
    SIGMAS,
    bootstrap_primary_delta,
    native_core9,
    records_by_source,
    severity_macro,
    core9_ap_from_records,
)
from tsg_repro.hashing import sha256_file

EXPECTED_ULTRALYTICS_VERSION = "8.4.131"
EXPECTED_VAL_SOURCES = 250
EXPECTED_VAL_VIEWS = 750
IMGSZ = 640
BDD10_NAMES = {
    0: "bike", 1: "bus", 2: "car", 3: "motor", 4: "person",
    5: "rider", 6: "traffic light", 7: "traffic sign", 8: "train", 9: "truck",
}
STEM_LOOKUP: dict[str, tuple[str, int]] = {}


def clear_cuda() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        try:
            torch.cuda.ipc_collect()
        except Exception:
            pass


def safe_symlink(target: Path, link: Path) -> None:
    target = target.expanduser().resolve()
    if not target.is_file():
        raise FileNotFoundError(target)
    link.parent.mkdir(parents=True, exist_ok=True)
    if link.exists() or link.is_symlink():
        raise RuntimeError(f"Refusing staged overwrite: {link}")
    os.symlink(str(target), str(link))


def import_companion(path: Path) -> None:
    spec = importlib.util.spec_from_file_location(path.stem, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import checkpoint companion: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[path.stem] = module
    spec.loader.exec_module(module)


def resolve_manifest_path(raw: object, manifest_path: Path) -> Path:
    p = Path(str(raw)).expanduser()
    if not p.is_absolute():
        p = manifest_path.parent / p
    return p.resolve()


def stage_val(
    val: pd.DataFrame,
    root: Path,
    manifest_path: Path,
) -> tuple[pd.DataFrame, dict[int, Path]]:
    global STEM_LOOKUP
    STEM_LOOKUP = {}
    yamls: dict[int, Path] = {}
    rows: list[dict] = []
    for sigma in SIGMAS:
        sroot = root / f"sigma{sigma}"
        img_dir, lab_dir = sroot / "images", sroot / "labels"
        img_dir.mkdir(parents=True)
        lab_dir.mkdir(parents=True)
        part = val[val["sigma"] == sigma].sort_values("source_stem", kind="mergesort").reset_index(drop=True)
        if len(part) != EXPECTED_VAL_SOURCES:
            raise RuntimeError(f"sigma{sigma}: expected 250 views, found {len(part)}")
        for ordinal, row in enumerate(part.itertuples(index=False)):
            source = str(row.source_stem)
            stem = f"{ordinal:03d}__{source}__sigma{sigma}"
            image = resolve_manifest_path(row.degraded_image, manifest_path)
            label = resolve_manifest_path(row.label_file, manifest_path)

            # Verify frozen VAL bytes when the manifest carries hashes.
            if hasattr(row, "degraded_sha256"):
                expected = str(row.degraded_sha256).strip().lower()
                if expected and sha256_file(image) != expected:
                    raise RuntimeError(f"Degraded-image SHA256 mismatch: {image}")
            if hasattr(row, "label_sha256"):
                expected = str(row.label_sha256).strip().lower()
                if expected and sha256_file(label) != expected:
                    raise RuntimeError(f"Label SHA256 mismatch: {label}")
            safe_symlink(image, img_dir / f"{stem}{image.suffix.lower()}")
            safe_symlink(label, lab_dir / f"{stem}.txt")
            STEM_LOOKUP[stem] = (source, int(sigma))
            rows.append({"staged_stem": stem, "source_stem": source, "sigma": int(sigma)})
        yml = root / f"sigma{sigma}.yaml"
        yml.write_text(yaml.safe_dump({
            "path": str(sroot.resolve()),
            # Ultralytics 8.4.131 schema requires both keys for a detection dataset.
            # model.val(split='val') still evaluates only the val entry.
            "train": "images", "val": "images", "names": BDD10_NAMES,
        }, sort_keys=False), encoding="utf-8")
        yamls[int(sigma)] = yml
    return pd.DataFrame(rows), yamls


def cached_validator_class():
    from ultralytics.models.rtdetr.val import RTDETRValidator

    class CachedRTDETRValidator(RTDETRValidator):
        last_instance = None
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.image_stats = []
            type(self).last_instance = self
        def update_metrics(self, preds, batch):
            super().update_metrics(preds, batch)
            for si, pred in enumerate(preds):
                pbatch = self._prepare_batch(si, batch)
                predn = self._prepare_pred(pred)
                stats = self._process_batch(predn, pbatch)
                if not isinstance(stats, dict) or "tp" not in stats:
                    raise RuntimeError("RTDETRValidator _process_batch did not return tp")
                stem = Path(pbatch["im_file"]).stem
                if stem not in STEM_LOOKUP:
                    raise RuntimeError(f"Unknown staged stem: {stem}")
                source, sigma = STEM_LOOKUP[stem]
                tpv = stats["tp"]
                tp = (tpv.detach().cpu().numpy() if torch.is_tensor(tpv) else np.asarray(tpv)).astype(bool, copy=False)
                conf = predn["conf"].detach().cpu().numpy().astype(np.float64, copy=False) if predn["conf"].numel() else np.empty((0,), np.float64)
                pred_cls = (
                    predn["cls"].detach().cpu().numpy().reshape(-1).astype(np.int64, copy=False)
                    if predn["cls"].numel() else np.empty((0,), np.int64)
                )
                target_cls = (
                    pbatch["cls"].detach().cpu().numpy().reshape(-1).astype(np.int64, copy=False)
                )
                if tp.ndim != 2 or tp.shape[1] != 10 or len(tp) != len(conf) or len(pred_cls) != len(conf):
                    raise RuntimeError(f"Invalid cached prediction shapes for {stem}")
                self.image_stats.append({
                    "source_stem": source, "sigma": int(sigma), "tp": tp,
                    "conf": conf, "pred_cls": pred_cls, "target_cls": target_cls,
                })
    return CachedRTDETRValidator


def evaluate_arm(checkpoint: Path, yamls: dict[int, Path], *, device: str, batch: int, workers: int, run_root: Path) -> tuple[dict, dict]:
    from ultralytics import RTDETR
    result = {"checkpoint": str(checkpoint), "checkpoint_sha256": sha256_file(checkpoint), "per_sigma": {}}
    caches: dict[int, dict[str, dict]] = {}
    for sigma in SIGMAS:
        clear_cuda()
        Validator = cached_validator_class()
        model = RTDETR(str(checkpoint))
        metrics = model.val(
            validator=Validator, data=str(yamls[int(sigma)]), split="val",
            imgsz=IMGSZ, batch=int(batch), workers=int(workers), device=device,
            plots=False, save=False, save_json=False, save_txt=False, verbose=False,
            project=str(run_root), name=f"sigma{sigma}", exist_ok=True,
        )
        records = Validator.last_instance.image_stats if Validator.last_instance is not None else []
        if len(records) != EXPECTED_VAL_SOURCES:
            raise RuntimeError(f"sigma{sigma}: expected 250 cached images, got {len(records)}")
        native = native_core9(metrics)
        reconstructed = core9_ap_from_records(records)
        if reconstructed is None:
            raise RuntimeError(f"sigma{sigma}: full VAL cache lacks fixed CORE9 support")
        diff = float(reconstructed["map50_95"]) - float(native["core9"]["map50_95"])
        if abs(diff) > 5e-6:
            raise RuntimeError(f"sigma{sigma}: cached/native AP mismatch {diff:+.9g}")
        native["cached_reconstructed_core9_map50_95"] = float(reconstructed["map50_95"])
        native["cached_minus_native_core9_map50_95"] = diff
        result["per_sigma"][str(sigma)] = native
        caches[int(sigma)] = records_by_source(records)
        del model
        clear_cuda()
    result["severity_macro"] = {"core9": {
        "map50_95": severity_macro(result["per_sigma"], "map50_95"),
        "map50": severity_macro(result["per_sigma"], "map50"),
    }}
    return result, caches


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--device", default="0")
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--workers", type=int, default=8)
    a = p.parse_args()

    cfg, base = load_config(a.config)
    manifest_path = get_path(cfg, base, "data", "gaussian_manifest")
    none_ckpt = get_path(cfg, base, "checkpoints", "rtdetr_l_none")
    ckd_ckpt = get_path(cfg, base, "checkpoints", "rtdetr_l_clean_kd")
    companion = get_path(cfg, base, "checkpoints", "rtdetr_checkpoint_companion")
    if not torch.cuda.is_available() and str(a.device) != "cpu":
        raise RuntimeError("CUDA is required for the frozen RT-DETR reference evaluation")
    import ultralytics
    if str(ultralytics.__version__) != EXPECTED_ULTRALYTICS_VERSION:
        raise RuntimeError(f"Expected Ultralytics {EXPECTED_ULTRALYTICS_VERSION}; found {ultralytics.__version__}")
    import_companion(companion)

    df = load_manifest(manifest_path)
    validate_manifest_contract(df)
    val = df[df["consrestore_split"] == "VAL"].copy().sort_values(["source_stem", "sigma"], kind="mergesort").reset_index(drop=True)
    if len(val) != EXPECTED_VAL_VIEWS or val["source_stem"].nunique() != EXPECTED_VAL_SOURCES:
        raise RuntimeError("Frozen VAL population changed")
    source_order = sorted(val["source_stem"].unique().tolist())

    out = ROOT / "results" / "rtdetr_gaussian_val_evaluation"
    if out.exists():
        raise RuntimeError(f"Refusing to overwrite existing output: {out}")
    out.mkdir(parents=True)

    with tempfile.TemporaryDirectory(prefix="tsg_rtdetr_val_") as td:
        stage_df, yamls = stage_val(val, Path(td), manifest_path)
        stage_df.to_csv(out / "VAL_EVALUATION_STAGE.csv", index=False)
        none, none_cache = evaluate_arm(none_ckpt, yamls, device=a.device, batch=a.batch, workers=a.workers, run_root=out / "native_none")
        ckd, ckd_cache = evaluate_arm(ckd_ckpt, yamls, device=a.device, batch=a.batch, workers=a.workers, run_root=out / "native_clean_kd")

    caches = {"NONE": none_cache, "CLEAN-KD": ckd_cache}
    boot_df, boot = bootstrap_primary_delta(caches, source_order)
    boot_df.to_csv(out / "RTDETR_PAIRED_SOURCE_BOOTSTRAP.csv", index=False)

    none_macro = float(none["severity_macro"]["core9"]["map50_95"])
    ckd_macro = float(ckd["severity_macro"]["core9"]["map50_95"])
    delta_per_sigma = {
        str(s): float(ckd["per_sigma"][str(s)]["core9"]["map50_95"] - none["per_sigma"][str(s)]["core9"]["map50_95"])
        for s in SIGMAS
    }
    result = {
        "status": "RTDETR_GAUSSIAN_VAL_REPRODUCTION_COMPLETE",
        "primary_endpoint": "CORE9 severity-macro mAP50-95",
        "population": {"sources": 250, "views": 750, "sigmas": list(SIGMAS)},
        "results": {"RTDETR-NONE": none, "RTDETR-CLEAN-KD": ckd},
        "none_macro": none_macro,
        "clean_kd_macro": ckd_macro,
        "delta_macro": ckd_macro - none_macro,
        "delta_per_sigma": delta_per_sigma,
        "bootstrap": {"unit": "VAL source cluster", "paired": True, "reps": BOOTSTRAP_REPS, "seed": BOOTSTRAP_SEED, **boot},
        "conclusion": (
            "CLEAN_KD_POSITIVE_ON_RTDETR_BUT_PAIRED_SOURCE_CI_INCLUDES_ZERO"
            if ckd_macro > none_macro and float(boot["ci_low"]) <= 0.0
            else "SEE_NUMERICAL_RESULT"
        ),
        "firewall": {"training": False, "checkpoint_reselection": False, "journal_geometry_fit": False, "journal_confirm": False},
        "provenance": {
            "gaussian_manifest_sha256": sha256_file(manifest_path),
            "none_checkpoint_sha256": sha256_file(none_ckpt),
            "clean_kd_checkpoint_sha256": sha256_file(ckd_ckpt),
            "companion_sha256": sha256_file(companion),
        },
    }
    path = out / "RTDETR_GAUSSIAN_VAL_RESULT.json"
    path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))

if __name__ == "__main__":
    main()
