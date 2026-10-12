#!/usr/bin/env python
"""Rebuild the frozen YOLO VAL object cohort by authoritative one-to-one matching."""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from tsg_repro.config import get_path, load_config
from tsg_repro.data import load_manifest
from tsg_repro.yolo_runtime import (
    clear_cuda,
    det_to_original,
    import_checkpoint_companion,
    load_hyp,
    match_gt_status,
    nms_native,
    preprocess_one,
    primary_prediction,
    read_gt,
    require_yolo_environment,
    build_dataset,
)
from tsg_repro.yolo_workflow import stage_manifest_views

EXPECTED = {
    "rows": 14754,
    "sources": 250,
    "none_detected": 8091,
    "preserved": 7115,
    "lost": 976,
    "primary_source_clusters": 249,
}


def load_model(path: Path, device):
    from ultralytics import YOLO

    model = YOLO(str(path)).model.float().to(device).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def infer_statuses(model, batch, gt_rows, input_hw, original_hw):
    import torch

    with torch.inference_mode():
        raw = model(batch["img"])
    det = nms_native(primary_prediction(raw))[0]
    pred = det_to_original(det, input_hw=input_hw, original_hw=original_hw)
    return match_gt_status(gt_rows, pred)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--output",
        default="results/yolo_object_cohort/FROZEN_OBJECT_COHORT.csv",
    )
    args = parser.parse_args()

    torch = require_yolo_environment()
    cfg, base = load_config(args.config)
    import_checkpoint_companion(
        get_path(cfg, base, "checkpoints", "yolo_checkpoint_companion", required=False)
    )
    manifest = load_manifest(get_path(cfg, base, "data", "gaussian_manifest"))

    temp_root = Path(tempfile.mkdtemp(prefix="tsg_cohort_"))
    staged = stage_manifest_views(manifest, temp_root, "VAL")
    hyp = load_hyp(get_path(cfg, base, "data", "yolo_none_args_yaml"))
    degraded = build_dataset(temp_root / "degraded" / "images", hyp, "cohort degraded: ")
    degraded_index = {Path(path).stem: i for i, path in enumerate(degraded.im_files)}

    device = torch.device("cuda:0")
    clear_cuda()
    none_model = load_model(get_path(cfg, base, "checkpoints", "yolo11s_none"), device)
    clean_kd_model = load_model(
        get_path(cfg, base, "checkpoints", "yolo11s_clean_kd"), device
    )

    records: list[dict] = []
    sequence = 0
    try:
        for view_index, row in staged.iterrows():
            item = degraded[degraded_index[row.generated_stem]]
            batch = preprocess_one(item, device)
            original_hw = tuple(map(int, item["ori_shape"]))
            input_hw = tuple(int(x) for x in batch["img"].shape[2:])
            gt_rows = read_gt(row.label_file, original_hw[1], original_hw[0])

            none_status = infer_statuses(
                none_model, batch, gt_rows, input_hw, original_hw
            )
            clean_kd_status = infer_statuses(
                clean_kd_model, batch, gt_rows, input_hw, original_hw
            )

            for gt, none, clean_kd in zip(gt_rows, none_status, clean_kd_status):
                records.append(
                    {
                        "exp4_row_id": f"E4V_{sequence:05d}",
                        "source_stem": str(row.source_stem),
                        "sigma": int(row.sigma),
                        "gt_index": int(gt["gt_index"]),
                        "class_id": int(gt["class_id"]),
                        "none_detected": bool(none["detected"]),
                        "clean_kd_detected": bool(clean_kd["detected"]),
                        "none_conf": none["matched_conf"],
                        "none_iou": none["matched_iou"],
                        "clean_kd_conf": clean_kd["matched_conf"],
                        "clean_kd_iou": clean_kd["matched_iou"],
                    }
                )
                sequence += 1

            if (view_index + 1) % 50 == 0 or (view_index + 1) == len(staged):
                print(f"cohort views             : {view_index + 1}/{len(staged)}")
    finally:
        shutil.rmtree(temp_root, ignore_errors=True)

    cohort = pd.DataFrame(records)
    primary = cohort[cohort["none_detected"]]
    summary = {
        "rows": int(len(cohort)),
        "sources": int(cohort["source_stem"].nunique()),
        "none_detected": int(cohort["none_detected"].sum()),
        "preserved": int(
            (cohort["none_detected"] & cohort["clean_kd_detected"]).sum()
        ),
        "lost": int(
            (cohort["none_detected"] & ~cohort["clean_kd_detected"]).sum()
        ),
        "primary_source_clusters": int(primary["source_stem"].nunique()),
    }
    if summary != EXPECTED:
        raise RuntimeError(
            f"Object cohort does not match frozen counts: {summary} != {EXPECTED}"
        )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    cohort.to_csv(output, index=False)
    (output.parent / "OBJECT_COHORT_SUMMARY.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
