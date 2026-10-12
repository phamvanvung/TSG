#!/usr/bin/env python
"""Checkpoint-level teacher-canonical Experiment-4 complement ablation.

Canonical rule for the architecture-locked P4 head-support region R_H:

    N_T = cellwise(N @ R_NT)
    e_T = proj_T(N_T) using one scalar for the complete flattened region
    c_T = N_T - e_T
    e_N = cellwise(e_T @ R_NT.T)
    N[R_H] <- e_N

P3, P5, model weights, thresholds, and features outside R_H remain unchanged.
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
)
from tsg_repro.yolo_workflow import stage_manifest_views

BOOTSTRAP_REPS = 5000
BOOTSTRAP_SEED = 2027
EXPECTED_PRIMARY = 8091
EXPECTED_LOST = 976
EXPECTED_PRESERVED = 7115
EXPECTED_SOURCE_CLUSTERS = 249


def parse_bool(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series
    parsed = series.astype(str).str.strip().str.lower().map(
        {"true": True, "false": False, "1": True, "0": False}
    )
    if parsed.isna().any():
        raise RuntimeError(f"Could not parse boolean column {series.name}")
    return parsed.astype(bool)


def bootstrap_damage_difference(result: pd.DataFrame) -> dict:
    clusters = []
    for _, group in result.groupby("source_stem", sort=True):
        lost = group[group["group"] == "LOST"]
        preserved = group[group["group"] == "PRESERVED"]
        clusters.append(
            (
                int(lost["damaged"].sum()),
                len(lost),
                int(preserved["damaged"].sum()),
                len(preserved),
            )
        )

    cluster_array = np.asarray(clusters, dtype=np.float64)
    if len(cluster_array) != EXPECTED_SOURCE_CLUSTERS:
        raise RuntimeError(
            f"Expected {EXPECTED_SOURCE_CLUSTERS} source clusters; "
            f"found {len(cluster_array)}"
        )

    rng = np.random.default_rng(BOOTSTRAP_SEED)
    values = []
    k = len(cluster_array)
    for _ in range(BOOTSTRAP_REPS):
        index = rng.integers(0, k, size=k)
        lost_damaged, lost_n, pres_damaged, pres_n = cluster_array[index].sum(axis=0)
        if lost_n > 0 and pres_n > 0:
            values.append(lost_damaged / lost_n - pres_damaged / pres_n)

    values = np.asarray(values, dtype=np.float64)
    return {
        "bootstrap_reps": BOOTSTRAP_REPS,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "reps_valid": int(len(values)),
        "mean": float(values.mean()),
        "ci_low": float(np.quantile(values, 0.025)),
        "ci_high": float(np.quantile(values, 0.975)),
    }


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
    parser.add_argument("--output", default="results/yolo_canonical_ablation")
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
        or int(primary["source_stem"].nunique()) != EXPECTED_SOURCE_CLUSTERS
    ):
        raise RuntimeError("Frozen cohort does not match 8091/976/7115/249")

    map_dir = (
        Path(args.map_dir).expanduser().resolve()
        if args.map_dir
        else get_path(cfg, base, "artifacts", "canonical_map_dir")
    )
    r_nt_np = np.load(map_dir / "R_NT_NONE_TO_TEACHER.npy", allow_pickle=False)
    if r_nt_np.shape != (256, 256):
        raise RuntimeError(f"Unexpected R_NT shape: {r_nt_np.shape}")

    output = Path(args.output)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite output: {output.resolve()}")

    temp_root = Path(tempfile.mkdtemp(prefix="tsg_ablation_"))
    staged = stage_manifest_views(manifest, temp_root, "VAL")
    hyp = load_hyp(get_path(cfg, base, "data", "yolo_none_args_yaml"))
    degraded = build_dataset(
        temp_root / "degraded" / "images", hyp, "ablation degraded: "
    )
    clean = build_dataset(temp_root / "clean" / "images", hyp, "ablation clean: ")
    degraded_index = {Path(path).stem: i for i, path in enumerate(degraded.im_files)}
    clean_index = {Path(path).stem: i for i, path in enumerate(clean.im_files)}

    device = torch.device("cuda:0")
    r_nt = torch.from_numpy(r_nt_np.astype(np.float32, copy=False)).to(device)
    teacher = load_model(
        get_path(cfg, base, "checkpoints", "yolo11s_clean_teacher"), device
    )
    none = load_model(get_path(cfg, base, "checkpoints", "yolo11s_none"), device)
    detect = none.model[DETECT_MODULE]

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
            none_store, none_handles = capture_modules(
                none, [P3_MODULE, P4_MODULE, P5_MODULE]
            )
            with torch.inference_mode():
                _ = teacher(clean_batch["img"])
                _ = none(degraded_batch["img"])
            remove_handles(teacher_handles)
            remove_handles(none_handles)

            teacher_p4 = teacher_store[P4_MODULE]
            p3 = none_store[P3_MODULE]
            p4 = none_store[P4_MODULE]
            p5 = none_store[P5_MODULE]

            with torch.inference_mode():
                baseline_det = nms_native(
                    primary_prediction(detect([p3.clone(), p4.clone(), p5.clone()]))
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
                    bool(frozen.none_detected)
                    != bool(baseline[int(frozen.gt_index)]["detected"])
                )
            if baseline_mismatches:
                raise RuntimeError(
                    f"Direct-head NONE baseline mismatch by {row.generated_stem}"
                )

            for record in targets.itertuples(index=False):
                gt_index = int(record.gt_index)
                exact_mask = exact_box_mask_from_xywh(
                    input_boxes[gt_index], input_hw=input_hw
                )
                head_mask = dilate_square(exact_mask, HEAD_RADIUS)
                teacher_vec, teacher_rows, teacher_cols = flatten_region(
                    teacher_p4, head_mask
                )
                none_vec, none_rows, none_cols = flatten_region(p4, head_mask)
                if not (
                    torch.equal(teacher_rows, none_rows)
                    and torch.equal(teacher_cols, none_cols)
                ):
                    raise RuntimeError("Teacher/NONE region indexing mismatch")

                alpha, e_t, c_t, e_n, c_n = canonical_flat_decomposition(
                    none_vec, teacher_vec, r_nt
                )
                p4_ablated = replace_region_from_vector(
                    p4, e_n, none_rows, none_cols
                )
                with torch.inference_mode():
                    intervened_det = nms_native(
                        primary_prediction(
                            detect([p3.clone(), p4_ablated, p5.clone()])
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
                        "exact_box_cells": int(exact_mask.sum()),
                        "head_support_cells": int(head_mask.sum()),
                        "projection_alpha_teacher": float(alpha.item()),
                        "baseline_detected": True,
                        "after_detected": bool(after["detected"]),
                        "after_conf": float(after["matched_conf"]),
                        "after_iou": float(after["matched_iou"]),
                        "damaged": bool(not after["detected"]),
                        "teacher_norm": float(torch.linalg.vector_norm(teacher_vec).item()),
                        "none_norm": float(torch.linalg.vector_norm(none_vec).item()),
                        "nE_norm": float(torch.linalg.vector_norm(e_n).item()),
                        "nC_norm": float(torch.linalg.vector_norm(c_n).item()),
                        "teacher_e_norm": float(torch.linalg.vector_norm(e_t).item()),
                        "teacher_c_norm": float(torch.linalg.vector_norm(c_t).item()),
                    }
                )

            if (view_index + 1) % 50 == 0 or (view_index + 1) == len(staged):
                print(f"ablation views           : {view_index + 1}/{len(staged)}")
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
        raise RuntimeError("Unexpected LOST/PRESERVED intervention counts")

    gap = float(lost["damaged"].mean() - preserved["damaged"].mean())
    bootstrap = bootstrap_damage_difference(result)

    output.mkdir(parents=True, exist_ok=False)
    result.to_csv(output / "PER_OBJECT_NC_ABLATION.csv", index=False)
    summary = {
        "status": "YOLO_TEACHER_CANONICAL_NC_ABLATION_COMPLETE",
        "rows": int(len(result)),
        "lost_n": int(len(lost)),
        "preserved_n": int(len(preserved)),
        "lost_damaged": int(lost["damaged"].sum()),
        "preserved_damaged": int(preserved["damaged"].sum()),
        "lost_damage_rate": float(lost["damaged"].mean()),
        "preserved_damage_rate": float(preserved["damaged"].mean()),
        "difference_lost_minus_preserved": gap,
        "source_cluster_bootstrap": bootstrap,
        "coordinate_rule": (
            "cellwise N->T; flattened-region scalar projection in teacher; "
            "teacher components -> N with R_NT.T"
        ),
        "baseline_integrity": {"direct_head_mismatches": baseline_mismatches},
    }
    (output / "TEACHER_CANONICAL_NC_ABLATION.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
