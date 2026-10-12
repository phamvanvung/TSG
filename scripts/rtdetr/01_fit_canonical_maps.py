#!/usr/bin/env python
"""Fit RT-DETR-L teacher-canonical N->T and S->T maps on JOURNAL_GEOMETRY_FIT."""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from tsg_repro.config import get_path, load_config
from tsg_repro.geometry_workflow import load_view_manifest
from tsg_repro.hashing import sha256_file
from tsg_repro.mapping import identity_distance, orthogonality_error
from tsg_repro.native_model import capture_eval_p4, verify_paired_geometry
from tsg_repro.rtdetr_runtime import (
    P4_CHANNELS,
    P4_HW,
    P4_MODULE,
    build_dataset,
    fresh_collate_one,
    load_model,
    preprocess_batch,
    require_rtdetr_environment,
    safe_symlink,
)
from tsg_repro.yolo_runtime import import_checkpoint_companion

EXPECTED_SOURCES = 120
EXPECTED_VIEWS = 360
EXPECTED_VECTORS = EXPECTED_VIEWS * P4_HW * P4_HW


def _solve(cross: np.ndarray) -> np.ndarray:
    u, _, vt = np.linalg.svd(cross, full_matrices=False)
    return u @ vt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--output", default="results/rtdetr_teacher_canonical_coordinate_audit"
    )
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    cfg, base = load_config(args.config)
    manifest_path = get_path(cfg, base, "data", "rtdetr_geometry_fit_manifest")
    views = load_view_manifest(
        manifest_path,
        expected_sources=EXPECTED_SOURCES,
        expected_views=EXPECTED_VIEWS,
    )

    output = Path(args.output)
    if not output.is_absolute():
        output = (ROOT / output).resolve()
    else:
        output = output.resolve()
    if output.exists():
        raise FileExistsError(output)

    stage = output.parent / f".{output.name}_stage"
    shutil.rmtree(stage, ignore_errors=True)
    for row in views.itertuples(index=False):
        stem = str(row.generated_stem)
        label = Path(row.label_file).resolve()
        for branch, key in (("degraded", "degraded_image"), ("clean", "clean_image")):
            image = Path(getattr(row, key)).resolve()
            safe_symlink(
                image,
                stage / branch / "images" / f"{stem}{image.suffix.lower()}",
            )
            safe_symlink(label, stage / branch / "labels" / f"{stem}.txt")

    torch = require_rtdetr_environment(require_cuda=args.device.startswith("cuda"))
    import_checkpoint_companion(
        get_path(cfg, base, "checkpoints", "rtdetr_checkpoint_companion")
    )
    device = torch.device(args.device)
    _, teacher = load_model(
        get_path(cfg, base, "checkpoints", "rtdetr_l_clean_teacher"), device
    )
    _, none = load_model(get_path(cfg, base, "checkpoints", "rtdetr_l_none"), device)
    _, student = load_model(
        get_path(cfg, base, "checkpoints", "rtdetr_l_clean_kd"), device
    )

    degraded_ds = build_dataset(stage / "degraded/images", "RTDETR maps degraded: ")
    clean_ds = build_dataset(stage / "clean/images", "RTDETR maps clean: ")
    degraded_index = {Path(x).stem: i for i, x in enumerate(degraded_ds.im_files)}
    clean_index = {Path(x).stem: i for i, x in enumerate(clean_ds.im_files)}
    if len(degraded_index) != EXPECTED_VIEWS or len(clean_index) != EXPECTED_VIEWS:
        raise RuntimeError("RT-DETR teacher-canonical staging view count changed")

    cross_nt = np.zeros((P4_CHANNELS, P4_CHANNELS), dtype=np.float64)
    cross_st = np.zeros((P4_CHANNELS, P4_CHANNELS), dtype=np.float64)
    vector_count = 0

    for j, row in enumerate(views.itertuples(index=False)):
        stem = str(row.generated_stem)
        degraded = preprocess_batch(
            fresh_collate_one(degraded_ds, degraded_index[stem]), device
        )
        clean = preprocess_batch(fresh_collate_one(clean_ds, clean_index[stem]), device)
        verify_paired_geometry(degraded, clean)

        t = capture_eval_p4(
            teacher,
            clean["img"],
            p4_module=P4_MODULE,
            channels=P4_CHANNELS,
            hw=P4_HW,
        )
        n = capture_eval_p4(
            none,
            degraded["img"],
            p4_module=P4_MODULE,
            channels=P4_CHANNELS,
            hw=P4_HW,
        )
        s = capture_eval_p4(
            student,
            degraded["img"],
            p4_module=P4_MODULE,
            channels=P4_CHANNELS,
            hw=P4_HW,
        )
        t_cells = t[0].permute(1, 2, 0).reshape(-1, P4_CHANNELS).cpu().numpy().astype(np.float64)
        n_cells = n[0].permute(1, 2, 0).reshape(-1, P4_CHANNELS).cpu().numpy().astype(np.float64)
        s_cells = s[0].permute(1, 2, 0).reshape(-1, P4_CHANNELS).cpu().numpy().astype(np.float64)
        cross_nt += n_cells.T @ t_cells
        cross_st += s_cells.T @ t_cells
        vector_count += len(t_cells)
        if (j + 1) % 30 == 0:
            print(f"views {j+1}/{EXPECTED_VIEWS}", flush=True)

    if vector_count != EXPECTED_VECTORS:
        raise RuntimeError(
            f"Expected {EXPECTED_VECTORS} fit vectors; observed {vector_count}"
        )

    r_nt = _solve(cross_nt)
    r_st = _solve(cross_st)
    output.mkdir(parents=True, exist_ok=False)
    np.save(output / "R_NT_NONE_TO_TEACHER.npy", r_nt)
    np.save(output / "R_ST_CLEAN_KD_TO_TEACHER.npy", r_st)

    report = {
        "status": "RTDETR_TEACHER_CANONICAL_COORDINATE_AUDIT_COMPLETE",
        "scope": {
            "split": "JOURNAL_GEOMETRY_FIT",
            "sources": EXPECTED_SOURCES,
            "views": EXPECTED_VIEWS,
            "confirm_used": False,
            "outcome_labels_used": False,
        },
        "feature": {
            "module": P4_MODULE,
            "channels": P4_CHANNELS,
            "spatial": [P4_HW, P4_HW],
            "fit_vectors_per_map": vector_count,
            "row_vector_mapping": "x @ R",
            "fit": "uncentered orthogonal Procrustes over every P4 cell",
        },
        "maps": {},
        "provenance": {"geometry_fit_manifest_sha256": sha256_file(manifest_path)},
    }
    for key, matrix, name in (
        ("R_NT", r_nt, "R_NT_NONE_TO_TEACHER.npy"),
        ("R_ST", r_st, "R_ST_CLEAN_KD_TO_TEACHER.npy"),
    ):
        report["maps"][key] = {
            "sha256": sha256_file(output / name),
            "orthogonality_error": orthogonality_error(matrix),
            "identity_distance": identity_distance(matrix),
        }

    (output / "TEACHER_CANONICAL_COORDINATE_AUDIT.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    shutil.rmtree(stage, ignore_errors=True)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
