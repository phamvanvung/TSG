#!/usr/bin/env python
"""Verify completed YOLO11s + RT-DETR-L checkpoint runs against rc2 targets.

Stable v1.0 is fail-closed: numerical closure alone is insufficient. The exact
successful CUDA environment must also be frozen and verified.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from tsg_repro.hashing import sha256_file


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def close(a, b, atol, rtol):
    return math.isclose(float(a), float(b), abs_tol=atol, rel_tol=rtol)


def req(path):
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(p)
    return p


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--results-root", default=str(ROOT / "results"))
    p.add_argument(
        "--reference",
        default=str(ROOT / "reference_results/v1.0_rc2_reference_targets.json"),
    )
    p.add_argument(
        "--environment-contract",
        default=str(ROOT / "configs/environment.requirements.json"),
    )
    p.add_argument("--environment-report")
    p.add_argument("--atol", type=float, default=5e-6)
    p.add_argument("--rtol", type=float, default=1e-6)
    a = p.parse_args()

    rr = Path(a.results_root).resolve()
    ref = load(a.reference)
    mismatches = []

    def eq(label, obs, exp):
        if obs != exp:
            mismatches.append({"field": label, "expected": exp, "observed": obs})

    def num(label, obs, exp):
        if not close(obs, exp, a.atol, a.rtol):
            mismatches.append({"field": label, "expected": exp, "observed": obs})

    # ----------------------------------------------------------------------------------
    # YOLO: rebuild geometry is an audit artifact; confirmatory downstream stages must
    # explicitly report the frozen authoritative maps/C_g hashes.
    # ----------------------------------------------------------------------------------
    yref = ref["yolo11s"]
    yauth = yref["authoritative_downstream_artifacts"]
    ycg_path = req(rr / "yolo_task_geometry/C_g.npy")
    ycg = np.load(ycg_path, allow_pickle=False)
    eq("yolo.rebuild_C_g.shape", list(ycg.shape), [256, 256])
    num("yolo.rebuild_C_g.trace", float(np.trace(ycg)), 1.0)

    yr = load(req(rr / "yolo_canonical_targeted_vs_random/CANONICAL_TARGETED_VS_RANDOM.json"))
    y15 = load(req(rr / "yolo_exp15a_confirmation/EXP15A_TSG_CONFIRMATION_RESULT.json"))
    y16 = load(req(rr / "yolo_exp16a_tsog/EXP16A_TSOG_CONFIRMATION_RESULT.json"))

    # Provenance is the scientific coordinate/geometry contract for Exp15A/16A.
    eq("yolo.exp15a.R_NT.sha256", y15["provenance"]["R_NT_sha256"], yauth["teacher_canonical_R_NT_sha256"])
    eq("yolo.exp15a.R_ST.sha256", y15["provenance"]["R_ST_sha256"], yauth["teacher_canonical_R_ST_sha256"])
    eq("yolo.exp15a.C_g.sha256", y15["provenance"]["Cg_sha256"], yauth["task_geometry_sha256"])
    eq("yolo.exp16a.C_g.sha256", y16["provenance"]["Cg_sha256"], yauth["task_geometry_sha256"])

    q = yref["canonical_random_control"]
    eq("yolo.random.targeted_lost_rescued", yr["targeted_lost_rescued"], q["targeted_lost_rescued"])
    eq("yolo.random.random_lost_rescued", yr["random_lost_rescued"], q["random_lost_rescued"])
    eq("yolo.random.targeted_preserved_harmed", yr["targeted_preserved_harmed"], q["targeted_preserved_harmed"])
    eq("yolo.random.random_preserved_harmed", yr["random_preserved_harmed"], q["random_preserved_harmed"])
    num("yolo.random.delta_utility", yr["delta_utility_bootstrap"]["point"], q["delta_utility_point"])
    for i, exp in enumerate(q["ci95"]):
        num(f"yolo.random.ci95[{i}]", [yr["delta_utility_bootstrap"]["ci_low"], yr["delta_utility_bootstrap"]["ci_high"]][i], exp)
    eq("yolo.random.gate_pass", yr["gate_pass"], q["gate_pass"])

    q = yref["exp15a"]
    g = y15["primary_confirmation"]
    num("yolo.exp15a.ordinary", g["gate_A_spearman"]["point"], q["ordinary_spearman"])
    num("yolo.exp15a.partial", g["gate_B_partial_spearman"]["point"], q["partial_spearman"])
    for i, exp in enumerate(q["partial_ci95"]):
        obs = [g["gate_B_partial_spearman"]["ci_low"], g["gate_B_partial_spearman"]["ci_high"]][i]
        num(f"yolo.exp15a.partial_ci95[{i}]", obs, exp)

    q = yref["exp16a"]
    num("yolo.exp16a.M", y16["gate_A_mapped_complement"]["point"], q["mapped_complement_partial_spearman"])
    num("yolo.exp16a.E", y16["gate_B_teacher_residual"]["point"], q["teacher_residual_partial_spearman"])
    eq("yolo.exp16a.conclusion", y16["conclusion"], q["conclusion"])

    # ----------------------------------------------------------------------------------
    # RT-DETR detector evaluation + mechanism chain.
    # ----------------------------------------------------------------------------------
    rref = ref["rtdetr_l"]
    rv = load(req(rr / "rtdetr_gaussian_val_evaluation/RTDETR_GAUSSIAN_VAL_RESULT.json"))
    q = rref["gaussian_val_detection"]
    num("rtdetr.val.none_macro", rv["none_macro"], q["none_macro"])
    num("rtdetr.val.clean_kd_macro", rv["clean_kd_macro"], q["clean_kd_macro"])
    num("rtdetr.val.delta_macro", rv["delta_macro"], q["delta_macro"])
    for s, exp in q["delta_per_sigma"].items():
        num(f"rtdetr.val.delta_sigma{s}", rv["delta_per_sigma"][str(s)], exp)
    num("rtdetr.val.ci_low", rv["bootstrap"]["ci_low"], q["ci95"][0])
    num("rtdetr.val.ci_high", rv["bootstrap"]["ci_high"], q["ci95"][1])
    eq("rtdetr.val.conclusion", rv["conclusion"], q["conclusion"])

    rnt = req(rr / "rtdetr_teacher_canonical_coordinate_audit/R_NT_NONE_TO_TEACHER.npy")
    rst = req(rr / "rtdetr_teacher_canonical_coordinate_audit/R_ST_CLEAN_KD_TO_TEACHER.npy")
    eq("rtdetr.R_NT.sha256", sha256_file(rnt), rref["teacher_canonical_R_NT_sha256"])
    eq("rtdetr.R_ST.sha256", sha256_file(rst), rref["teacher_canonical_R_ST_sha256"])
    eq("rtdetr.C_g.sha256", sha256_file(req(rr / "rtdetr_task_geometry/C_g.npy")), rref["task_geometry_sha256"])

    r12 = load(req(rr / "rtdetr_local_utility_confirmation/RTDETR_MAPPED_COMPLEMENT_LOCAL_UTILITY_RESULT.json"))
    local_step12_cells = req(rr / "rtdetr_local_utility_confirmation/RTDETR_CONFIRM_CELL_RECORDS.csv")
    r14 = load(req(rr / "rtdetr_finite_horizon_confirmation/RTDETR_FINITE_HORIZON_CONFIRMATION_RESULT.json"))
    ro = load(req(rr / "rtdetr_orientation_confirmation/RTDETR_TASK_SENSITIVE_ORIENTATION_RESULT.json"))
    # End-to-end linkage: exact byte linkage is ideal for a fresh public run.
    # A completed historical Step-14 run may have consumed a byte-different
    # Step-12 CSV that is scientifically equivalent for Step 14. In that case
    # require the explicit semantic linkage audit rather than rejecting solely
    # on CSV serialization / non-rank-changing FP differences.
    step14_input_sha = r14["provenance"]["step12_cell_records_sha256"]
    local_step12_sha = sha256_file(local_step12_cells)
    if step14_input_sha != local_step12_sha:
        linkage_path = rr / "rtdetr_step12_step14_linkage_audit.json"
        if not linkage_path.is_file():
            mismatches.append({
                "field": "rtdetr.step14.step12_cell_records_linkage",
                "expected": local_step12_sha,
                "observed": step14_input_sha,
                "detail": "Run scripts/rtdetr/04b_audit_step12_step14_linkage.py."
            })
        else:
            linkage = load(linkage_path)
            eq("rtdetr.step14.linkage_audit.status", linkage.get("status"), "PASS")
            eq("rtdetr.step14.linkage_audit.scientific_equivalence", linkage.get("scientifically_equivalent_for_step14"), True)
            eq("rtdetr.step14.linkage_audit.public_sha", linkage.get("public_step12_sha256"), local_step12_sha)
            eq("rtdetr.step14.linkage_audit.step14_input_sha", linkage.get("step14_recorded_input_sha256"), step14_input_sha)
            eq("rtdetr.step14.linkage_audit.selection_changed_views", linkage.get("selection_changed_views"), 0)
            eq("rtdetr.step14.linkage_audit.selected_u_C_rank_changed_views", linkage.get("selected_u_C_rank_changed_views"), 0)

    q = rref["step12_local_utility"]
    num("rtdetr.step12.gate_A", r12["primary_gate_A"]["point"], q["gate_A_point"])
    num("rtdetr.step12.gate_B", r12["primary_gate_B"]["point"], q["gate_B_point"])
    eq("rtdetr.step12.conclusion", r12["conclusion"], q["conclusion"])

    q = rref["step14_finite_horizon"]
    pc = r14["primary_confirmation"]
    num("rtdetr.step14.rho_small", pc["rho_small_source_balanced"], q["rho_small_source_balanced"])
    num("rtdetr.step14.rho_large", pc["rho_large_source_balanced"], q["rho_large_source_balanced"])
    num("rtdetr.step14.delta", pc["delta_rho_point"], q["delta_rho_point"])
    for i, exp in enumerate(q["ci95"]):
        num(f"rtdetr.step14.ci95[{i}]", pc["ci95"][i], exp)
    eq("rtdetr.step14.conclusion", r14["conclusion"], q["conclusion"])
    eq("rtdetr.step14.frozen_u_C_primary", r14["integrity"]["primary_horizon_predictor_from_frozen_step12_u_C"], True)
    eq("rtdetr.step14.recomputed_scores_not_gate", r14["integrity"]["recomputed_step12_scores_used_as_scientific_gate"], False)

    q = rref["canonical_orientation"]
    oc = ro["primary_confirmation"]
    num("rtdetr.orientation.partial", oc["point"], q["partial_spearman"])
    num("rtdetr.orientation.ci_low", oc["ci_low"], q["ci95"][0])
    num("rtdetr.orientation.ci_high", oc["ci_high"], q["ci95"][1])

    # ----------------------------------------------------------------------------------
    # Stable release requires the successful reference execution environment.
    # ----------------------------------------------------------------------------------
    contract = load(a.environment_contract)
    frozen_env = contract.get("freeze_from_successful_reference_run", {})
    complete = bool(frozen_env) and all(v is not None for v in frozen_env.values())
    env_ok = False
    if a.environment_report:
        er = load(a.environment_report)
        env_ok = er.get("status") == "PASS" and bool(er.get("contract_complete"))

    scientific_ok = not mismatches
    if scientific_ok and complete and env_ok:
        status = "FULL_CHECKPOINT_REPRODUCTION_PASS"
    elif scientific_ok:
        status = "SCIENTIFIC_NUMERICAL_TARGETS_PASS_ENVIRONMENT_LOCK_INCOMPLETE"
    else:
        status = "FAIL"

    report = {
        "status": status,
        "scientific_targets_pass": scientific_ok,
        "stable_v1_environment_contract_complete": complete,
        "environment_report_pass": env_ok,
        "mismatches": mismatches,
        "atol": a.atol,
        "rtol": a.rtol,
    }
    print(json.dumps(report, indent=2))
    raise SystemExit(0 if scientific_ok else 2)


if __name__ == "__main__":
    main()
