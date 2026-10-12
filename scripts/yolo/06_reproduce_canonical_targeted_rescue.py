#!/usr/bin/env python
"""Checkpoint-level TARGETED teacher-canonical mapped-complement rescue.

This script reproduces the checkpoint-level canonical targeted intervention branch:

    N_T = cellwise(N @ R_NT)
    c_T = flattened-region complement to the clean teacher
    c_S = cellwise(c_T @ R_ST.T)
    S[R_H] <- S[R_H] + c_S

The historical norm-matched deterministic RANDOM control is intentionally not
redefined. Its exact generator must be migrated before the Gate-2
TARGETED-vs-RANDOM contrast is advertised as checkpoint-level reproduced.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from tsg_repro.config import get_path, load_config
from tsg_repro.data import load_manifest
from tsg_repro.yolo_runtime import (
    DETECT_MODULE,
    HEAD_RADIUS,
    P3_MODULE,
    P4_MODULE,
    P5_MODULE,
    build_dataset,
    canonical_flat_decomposition,
    capture_modules,
    det_to_original,
    dilate_square,
    exact_box_mask_from_xywh,
    flatten_region,
    import_checkpoint_companion,
    item_input_boxes_xywh,
    load_hyp,
    match_gt_status,
    nms_native,
    preprocess_one,
    primary_prediction,
    read_gt,
    remove_handles,
    replace_region_from_vector,
    require_yolo_environment,
    teacher_to_student_flat,
)
from tsg_repro.yolo_workflow import stage_manifest_views

EXPECTED_PRIMARY = 8091
EXPECTED_LOST = 976
EXPECTED_PRESERVED = 7115


def parse_bool(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series
    parsed = series.astype(str).str.strip().str.lower().map(
        {"true": True, "false": False, "1": True, "0": False}
    )
    if parsed.isna().any():
        raise RuntimeError(f"Could not parse boolean column {series.name}")
    return parsed.astype(bool)


def load_model(path: Path, device):
    from ultralytics import YOLO

    model = YOLO(str(path)).model.float().to(device).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--cohort", default=None)
    parser.add_argument("--map-dir", default=None)
    parser.add_argument("--output", default="results/yolo_canonical_targeted_rescue")
    args = parser.parse_args()

    torch = require_yolo_environment()
    cfg, base = load_config(args.config)
    import_checkpoint_companion(
        get_path(cfg, base, "checkpoints", "yolo_checkpoint_companion", required=False)
    )
    manifest = load_manifest(get_path(cfg, base, "data", "gaussian_manifest"))

    cohort_path = (
        Path(args.cohort).expanduser().resolve()
        if args.cohort
        else get_path(cfg, base, "data", "object_cohort")
    )
    cohort = pd.read_csv(cohort_path)
    cohort["source_stem"] = cohort["source_stem"].astype(str)
    cohort["sigma"] = pd.to_numeric(cohort["sigma"], errors="raise").astype(int)
    cohort["none_detected"] = parse_bool(cohort["none_detected"])
    cohort["clean_kd_detected"] = parse_bool(cohort["clean_kd_detected"])

    primary = cohort[cohort["none_detected"]].copy()
    primary["group"] = np.where(
        primary["clean_kd_detected"], "PRESERVED", "LOST"
    )
    if (
        len(primary) != EXPECTED_PRIMARY
        or int((primary["group"] == "LOST").sum()) != EXPECTED_LOST
        or int((primary["group"] == "PRESERVED").sum()) != EXPECTED_PRESERVED
    ):
        raise RuntimeError("Frozen cohort does not match 8091/976/7115")

    map_dir = (
        Path(args.map_dir).expanduser().resolve()
        if args.map_dir
        else get_path(cfg, base, "artifacts", "canonical_map_dir")
    )
    r_nt_np = np.load(map_dir / "R_NT_NONE_TO_TEACHER.npy", allow_pickle=False)
    r_st_np = np.load(map_dir / "R_ST_CLEAN_KD_TO_TEACHER.npy", allow_pickle=False)
    if r_nt_np.shape != (256, 256) or r_st_np.shape != (256, 256):
        raise RuntimeError(
            f"Unexpected map shapes: R_NT={r_nt_np.shape}, R_ST={r_st_np.shape}"
        )

    output = Path(args.output)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite output: {output.resolve()}")

    temp_root = Path(tempfile.mkdtemp(prefix="tsg_rescue_"))
    staged = stage_manifest_views(manifest, temp_root, "VAL")
    hyp = load_hyp(get_path(cfg, base, "data", "yolo_none_args_yaml"))
    degraded = build_dataset(
        temp_root / "degraded" / "images", hyp, "rescue degraded: "
    )
    clean = build_dataset(temp_root / "clean" / "images", hyp, "rescue clean: ")
    degraded_index = {Path(path).stem: i for i, path in enumerate(degraded.im_files)}
    clean_index = {Path(path).stem: i for i, path in enumerate(clean.im_files)}

    device = torch.device("cuda:0")
    r_nt = torch.from_numpy(r_nt_np.astype(np.float32, copy=False)).to(device)
    r_st = torch.from_numpy(r_st_np.astype(np.float32, copy=False)).to(device)
    teacher = load_model(
        get_path(cfg, base, "checkpoints", "yolo11s_clean_teacher"), device
    )
    none = load_model(get_path(cfg, base, "checkpoints", "yolo11s_none"), device)
    student = load_model(
        get_path(cfg, base, "checkpoints", "yolo11s_clean_kd"), device
    )
    student_detect = student.model[DETECT_MODULE]

    primary_by_key = {
        (str(key[0]), int(key[1])): group.sort_values("gt_index")
        for key, group in primary.groupby(["source_stem", "sigma"], sort=False)
    }

    records: list[dict] = []
    baseline_mismatches = 0
    try:
        for view_index, row in staged.iterrows():
            key = (str(row.source_stem), int(row.sigma))
            targets = primary_by_key.get(key)
            if targets is None or len(targets) == 0:
                continue

            degraded_item = degraded[degraded_index[row.generated_stem]]
            clean_item = clean[clean_index[row.generated_stem]]
            degraded_batch = preprocess_one(degraded_item, device)
            clean_batch = preprocess_one(clean_item, device)
            input_hw = tuple(int(x) for x in degraded_batch["img"].shape[2:])
            original_hw = tuple(map(int, degraded_item["ori_shape"]))
            gt_rows = read_gt(row.label_file, original_hw[1], original_hw[0])
            input_boxes = item_input_boxes_xywh(degraded_item)

            teacher_store, teacher_handles = capture_modules(teacher, [P4_MODULE])
            none_store, none_handles = capture_modules(none, [P4_MODULE])
            student_store, student_handles = capture_modules(
                student, [P3_MODULE, P4_MODULE, P5_MODULE]
            )
            with torch.inference_mode():
                _ = teacher(clean_batch["img"])
                _ = none(degraded_batch["img"])
                _ = student(degraded_batch["img"])
            remove_handles(teacher_handles)
            remove_handles(none_handles)
            remove_handles(student_handles)

            teacher_p4 = teacher_store[P4_MODULE]
            none_p4 = none_store[P4_MODULE]
            student_p3 = student_store[P3_MODULE]
            student_p4 = student_store[P4_MODULE]
            student_p5 = student_store[P5_MODULE]

            with torch.inference_mode():
                baseline_det = nms_native(
                    primary_prediction(
                        student_detect(
                            [
                                student_p3.clone(),
                                student_p4.clone(),
                                student_p5.clone(),
                            ]
                        )
                    )
                )[0]
            baseline = match_gt_status(
                gt_rows,
                det_to_original(baseline_det, input_hw, original_hw),
            )

            frozen_rows = cohort[
                (cohort["source_stem"] == str(row.source_stem))
                & (cohort["sigma"] == int(row.sigma))
            ].sort_values("gt_index")
            if len(frozen_rows) != len(gt_rows):
                raise RuntimeError(f"Frozen cohort/GT mismatch for {row.generated_stem}")
            for frozen in frozen_rows.itertuples(index=False):
                baseline_mismatches += int(
                    bool(frozen.clean_kd_detected)
                    != bool(baseline[int(frozen.gt_index)]["detected"])
                )
            if baseline_mismatches:
                raise RuntimeError(
                    f"Direct-head CLEAN-KD baseline mismatch by {row.generated_stem}"
                )

            for record in targets.itertuples(index=False):
                gt_index = int(record.gt_index)
                head_mask = dilate_square(
                    exact_box_mask_from_xywh(
                        input_boxes[gt_index], input_hw=input_hw
                    ),
                    HEAD_RADIUS,
                )

                teacher_vec, teacher_rows, teacher_cols = flatten_region(
                    teacher_p4, head_mask
                )
                none_vec, none_rows, none_cols = flatten_region(none_p4, head_mask)
                student_vec, student_rows, student_cols = flatten_region(
                    student_p4, head_mask
                )
                if not (
                    torch.equal(teacher_rows, none_rows)
                    and torch.equal(teacher_cols, none_cols)
                    and torch.equal(teacher_rows, student_rows)
                    and torch.equal(teacher_cols, student_cols)
                ):
                    raise RuntimeError("Teacher/NONE/CLEAN-KD region indexing mismatch")

                _, _, complement_t, _, _ = canonical_flat_decomposition(
                    none_vec, teacher_vec, r_nt
                )
                mapped_complement = teacher_to_student_flat(complement_t, r_st)
                intervened_p4 = replace_region_from_vector(
                    student_p4,
                    student_vec + mapped_complement,
                    student_rows,
                    student_cols,
                )
                with torch.inference_mode():
                    intervened_det = nms_native(
                        primary_prediction(
                            student_detect(
                                [
                                    student_p3.clone(),
                                    intervened_p4,
                                    student_p5.clone(),
                                ]
                            )
                        )
                    )[0]
                after = match_gt_status(
                    gt_rows,
                    det_to_original(intervened_det, input_hw, original_hw),
                )[gt_index]

                group = "PRESERVED" if bool(record.clean_kd_detected) else "LOST"
                records.append(
                    {
                        "exp4_row_id": record.exp4_row_id,
                        "source_stem": str(row.source_stem),
                        "sigma": int(row.sigma),
                        "gt_index": gt_index,
                        "class_id": int(record.class_id),
                        "group": group,
                        "head_support_cells": int(head_mask.sum()),
                        "mapped_nC_norm": float(
                            torch.linalg.vector_norm(mapped_complement).item()
                        ),
                        "targeted_after_detected": bool(after["detected"]),
                        "targeted_after_conf": float(after["matched_conf"]),
                        "targeted_after_iou": float(after["matched_iou"]),
                    }
                )

            if (view_index + 1) % 50 == 0 or (view_index + 1) == len(staged):
                print(f"rescue views             : {view_index + 1}/{len(staged)}")
    finally:
        shutil.rmtree(temp_root, ignore_errors=True)

    result = pd.DataFrame(records)
    if len(result) != EXPECTED_PRIMARY:
        raise RuntimeError(
            f"Expected {EXPECTED_PRIMARY} intervention rows; found {len(result)}"
        )
    lost = result[result["group"] == "LOST"]
    preserved = result[result["group"] == "PRESERVED"]
    if len(lost) != EXPECTED_LOST or len(preserved) != EXPECTED_PRESERVED:
        raise RuntimeError("Unexpected LOST/PRESERVED rescue counts")

    output.mkdir(parents=True, exist_ok=False)
    result.to_csv(output / "PER_OBJECT_TARGETED_RESCUE.csv", index=False)
    summary = {
        "status": "YOLO_TEACHER_CANONICAL_TARGETED_RESCUE_COMPLETE",
        "rows": int(len(result)),
        "lost_n": int(len(lost)),
        "preserved_n": int(len(preserved)),
        "targeted_lost_rescued": int(lost["targeted_after_detected"].sum()),
        "targeted_lost_rescue_rate": float(lost["targeted_after_detected"].mean()),
        "targeted_preserved_harmed": int(
            (~preserved["targeted_after_detected"]).sum()
        ),
        "targeted_preserved_harm_rate": float(
            (~preserved["targeted_after_detected"]).mean()
        ),
        "baseline_integrity": {"direct_head_mismatches": baseline_mismatches},
        "random_control_status": (
            "SEPARATE_SHA_LOCKED_DRIVER; use 06b_reproduce_canonical_random_control.py "
            "for the Gate-2 targeted-vs-random contrast"
        ),
        "construction": (
            "N->T before flattened-region projection; c_T -> CLEAN-KD with R_ST.T"
        ),
    }
    (output / "TARGETED_RESCUE.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
