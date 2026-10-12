# Reproduction status matrix — v1.0.0

rc2 incorporates the first complete author-machine checkpoint execution and corrects the release targets/bookkeeping exposed by that run. No new method selection or tuning is introduced.

| Scientific output | v1.0 level | Entry point | Frozen boundary |
|---|---|---|---|
| Gaussian TRAIN/VAL contract | Checkpoint/data-level | `scripts/yolo/01_audit_gaussian_manifest.py` | Exact SHA-derived generator |
| YOLO frozen metrics / rebuild maps / object cohort | Checkpoint-level | `scripts/yolo/02...04_*` | Frozen checkpoints/manifests |
| YOLO canonical Experiment-4 ablation/rescue | Checkpoint-level | `scripts/yolo/05...06_*` | Frozen authoritative teacher-canonical maps downstream |
| YOLO targeted-vs-random Gate-2 | Hash-locked helper checkpoint-level | `scripts/yolo/06b_*` | Exact Step-07 helper SHA only; no surrogate |
| YOLO `C_g` rebuild audit | Checkpoint-level audit | `scripts/yolo/07_fit_task_sensitive_geometry.py` | CLEAN-KD native loss only |
| YOLO Exp15A / Exp16A confirmation | Checkpoint-level | `scripts/yolo/08...09_*` | Frozen authoritative YOLO maps + frozen authoritative `C_g` |
| RT-DETR asset/cohort/bridge preflight | Checkpoint-level | `scripts/rtdetr/00_preflight.py` | Three checkpoints + companion + historical bridge hashes |
| RT-DETR Gaussian VAL detector performance | **Checkpoint-level added in rc2** | `scripts/rtdetr/01_evaluate_frozen_checkpoints.py` | Frozen VAL, fixed CORE9, paired 250-source bootstrap |
| RT-DETR teacher-canonical `R_NT`,`R_ST` | Checkpoint-level | `scripts/rtdetr/01_fit_canonical_maps.py` | JOURNAL_GEOMETRY_FIT, 576,000 cells/map |
| RT-DETR `C_g` | Checkpoint-level | `scripts/rtdetr/02_fit_task_sensitive_geometry.py` | CLEAN-KD native loss only |
| RT-DETR Step-12 local utility | Checkpoint-level | `scripts/rtdetr/03_run_local_utility_confirmation.py` | Historical frozen Step-10 NONE→CLEAN-KD bridge; 200/600 confirmation |
| RT-DETR Step-14 finite horizon | Checkpoint-level | `scripts/rtdetr/04_run_finite_horizon_confirmation.py` | Local public Step-12 table preferred; frozen Step-12 `u_C` is primary predictor |
| RT-DETR teacher-canonical orientation | Checkpoint-level | `scripts/rtdetr/05_run_orientation_confirmation.py` | Public `R_NT/R_ST` + public `C_g`; 200/600, alpha=.25 |
| Cross-chain numerical closure | Verification-level | `scripts/verify_end_to_end.py` | rc2 successful reference-run targets + Step12→14 linkage |
| Stable-v1 environment closure | Pending final freeze | `scripts/verify_environment.py` | exact Python/NumPy/SciPy/pandas/PyTorch/CUDA/cuDNN/GPU |

## Successful checkpoint execution already established

The author-machine rc1 execution established:

- YOLO canonical Gate-2 targeted-vs-random PASS;
- YOLO teacher-canonical Exp15A PASS;
- YOLO Exp16A mapped-complement replication PASS and teacher-residual family FAIL;
- RT-DETR Step-12 local utility PASS;
- RT-DETR Step-14 finite-horizon PASS under the corrected authoritative replay semantics;
- RT-DETR teacher-canonical orientation PASS.

Those reconciled values are frozen in `reference_results/v1.0_rc2_reference_targets.json`.

## Why RT-DETR has two coordinate paths

The frozen Step-12/14 finite-horizon experiment is reproduced **as originally defined**: raw NONE is decomposed against the clean-teacher vector and the resulting complement is mapped with the historical Step-10 NONE→CLEAN-KD Procrustes bridge. The final TSG orientation uses the later teacher-canonical coordinate correction (`R_NT`, `R_ST`). rc2 does not retroactively rewrite Step-12/14, because that would create a different experiment.

## Public-chain correction in rc2

When the public Step-12 output exists, Step 14 now consumes that local `RTDETR_CONFIRM_CELL_RECORDS.csv` before consulting any optional external fallback. For a fresh run, exact provenance-SHA linkage is expected. For the already-completed author run, the historical Step-12 CSV and the public regenerated CSV are byte-different because the public table omits diagnostic columns and has small non-rank-changing FP differences. `scripts/rtdetr/04b_audit_step12_step14_linkage.py` therefore verifies semantic linkage: identical 9,975 cell identities, identical Step-14 top-25% selections, identical selected-cell `u_C` ranks, and identical frozen exact Step-12 outcomes (`Delta_T`, `Delta_C`, `Y`). The end-to-end verifier accepts either exact SHA linkage or this explicit semantic-equivalence PASS.

## Stable v1.0 closure status

- stage bundle-safe small author artifacts/companions with `scripts/stage_public_reference_artifacts.py`;
- run the new RT-DETR Gaussian VAL detector stage under the author reference environment;
- run the Step-12→Step-14 linkage audit; exact SHA linkage is accepted for fresh runs and semantic equivalence is accepted for the completed historical run;
- **Complete:** exact Python/NumPy/SciPy/pandas/PyTorch/CUDA/cuDNN/GPU versions were captured and frozen;
- **Complete:** `FULL_CHECKPOINT_REPRODUCTION_PASS` was obtained;
- sanitize the archive, rebuild `MANIFEST.txt`, test archive integrity, and record archive SHA-256.
