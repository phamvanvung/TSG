#!/usr/bin/env python
"""
ConsRestore / RestorDet Experiment 4 — Step 07
================================================

Purpose:
    Specificity control for the positive Step-06 functional n_C ablation.

Scientific question:
    Is the larger LOST-object damage observed when removing n_C specific to
    removal of that complementary component, or are LOST objects simply more
    fragile to any equally large perturbation in the same P4 region?

Step-06 targeted intervention:
    n = n_E + n_C
    targeted: n -> n_E
    targeted perturbation delta_C = -n_C

Step-07 random control:
    construct deterministic delta_R such that
        delta_R ⟂ t
        delta_R ⟂ n_C
        ||delta_R||_2 = ||n_C||_2
    and apply
        n -> n + delta_R
    inside the same architecture-locked R_H only.

The extra orthogonality to n_C prevents the random control from accidentally
cancelling the actual n_C direction.

Primary endpoint:
    damage = originally NONE-detected object becomes undetected.

Primary specificity contrast:
    Delta_specific =
        [Damage_C(LOST) - Damage_C(PRESERVED)]
      - [Damage_R(LOST) - Damage_R(PRESERVED)]

where Damage_C comes from frozen Step 06 and Damage_R from this control.

Primary gate:
    source-cluster bootstrap 95% CI for Delta_specific entirely > 0.

No training, no threshold/radius tuning, no CLEAN-KD intervention,
no DEVTEST, no ACDC.

Run from:
    /AUTHOR_HOME/Documents/Researching/ConsRestore/experiment4
"""

from __future__ import annotations

import gc
import hashlib
import importlib
import json
import os
import shutil
import sys
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import numpy as np
import pandas as pd
import torch
import yaml


PROJECT = Path("/AUTHOR_HOME/Documents/Researching/ConsRestore")
EXP1 = PROJECT / "experiment1"
EXP4 = PROJECT / "experiment4"

LOCK_JSON = EXP4 / "results/00_scientific_lock/EXPERIMENT4_SCIENTIFIC_LOCK.json"
STEP01_JSON = EXP4 / "results/01_input_audit/INPUT_AUDIT.json"
STEP04_JSON = EXP4 / "results/04_head_support_audit/HEAD_SUPPORT_AUDIT.json"
STEP05_JSON = EXP4 / "results/05_none_replay/NONE_REPLAY_AUDIT.json"
STEP06_JSON = EXP4 / "results/06_functional_nc_ablation/FUNCTIONAL_NC_GATE.json"
STEP06_CSV = EXP4 / "results/06_functional_nc_ablation/PER_OBJECT_NC_ABLATION.csv"
COHORT_CSV = EXP4 / "results/01_input_audit/FROZEN_OBJECT_COHORT.csv"

NONE_ARGS = EXP1 / "results/13_scientific_training/runs/none_50e/args.yaml"

OUT = Path("results/07_random_control")
TMP = Path("results/.tmp_exp4_step07_random_control")

CONF = 0.25
MATCH_IOU = 0.50
NMS_IOU = 0.70
MAX_DET = 300
IMGSZ = 640

P3_MODULE = 16
P4_MODULE = 19
P5_MODULE = 22
DETECT_MODULE = 23
HEAD_RADIUS = 2

BOOTSTRAP_REPS = 5000
BOOTSTRAP_SEED = 2027
RANDOM_CONTROL_SEED = 2027

EXPECTED_ROWS = 14754
EXPECTED_NONE_DETECTED = 8091
EXPECTED_PRESERVED = 7115
EXPECTED_LOST = 976
EXPECTED_PRIMARY_SOURCE_CLUSTERS = 249
EXPECTED_VAL_VIEWS = 750
EXPECTED_P4_HW = (40, 40)
EXPECTED_ULTRALYTICS_VERSION = "8.4.131"

BDD10_NAMES = {
    0: "bike",
    1: "bus",
    2: "car",
    3: "motor",
    4: "person",
    5: "rider",
    6: "traffic light",
    7: "traffic sign",
    8: "train",
    9: "truck",
}

EXP1_COMPANION_FILE = EXP1 / "consrestore_cacd_native.py"
EXPECTED_COMPANION_SHA256 = (
    "d82f20215ac37848cea810b8e4e5ee7fb169af24ea8819305088fe29490d121b"
)


def fail(message: str) -> None:
    raise RuntimeError(message)


def sha256_file(path: Path) -> str:
    if not path.is_file():
        fail(f"Missing required file: {path}")
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def clear_cuda() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        try:
            torch.cuda.ipc_collect()
        except Exception:
            pass


def prepare_checkpoint_companion():
    if not EXP1_COMPANION_FILE.is_file():
        fail(f"Missing checkpoint companion: {EXP1_COMPANION_FILE}")

    actual = sha256_file(EXP1_COMPANION_FILE)
    if actual != EXPECTED_COMPANION_SHA256:
        fail(
            "Checkpoint companion SHA256 mismatch.\n"
            f"Expected: {EXPECTED_COMPANION_SHA256}\n"
            f"Actual  : {actual}"
        )

    exp1_dir = str(EXP1.resolve())
    if exp1_dir not in sys.path:
        sys.path.insert(0, exp1_dir)

    module = importlib.import_module("consrestore_cacd_native")
    if Path(module.__file__).resolve() != EXP1_COMPANION_FILE.resolve():
        fail(f"Imported companion from unexpected location: {module.__file__}")

    return {
        "path": str(EXP1_COMPANION_FILE.resolve()),
        "sha256": actual,
    }


def safe_symlink(target: Path, link: Path) -> None:
    target = target.resolve()
    if not target.is_file():
        fail(f"Missing symlink target: {target}")
    link.parent.mkdir(parents=True, exist_ok=True)
    if link.exists() or link.is_symlink():
        link.unlink()
    os.symlink(str(target), str(link))


def load_hyp():
    from ultralytics.cfg import get_cfg

    raw = yaml.safe_load(NONE_ARGS.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        fail("Could not parse NONE args.yaml.")
    return get_cfg(overrides=raw)


def build_dataset(image_dir: Path, hyp, prefix: str):
    from ultralytics.data.dataset import YOLODataset

    return YOLODataset(
        img_path=str(image_dir),
        imgsz=IMGSZ,
        cache=False,
        augment=False,
        hyp=hyp,
        prefix=prefix,
        rect=False,
        batch_size=1,
        stride=32,
        pad=0.0,
        single_cls=False,
        classes=None,
        fraction=1.0,
        data={
            "names": BDD10_NAMES,
            "nc": 10,
            "channels": 3,
        },
        task="detect",
    )


def preprocess_one(item: dict, device: torch.device):
    from ultralytics.data.dataset import YOLODataset

    batch = YOLODataset.collate_fn([item])

    for key, value in list(batch.items()):
        if isinstance(value, torch.Tensor):
            if key == "img":
                batch[key] = (
                    value.to(device, non_blocking=False)
                    .float()
                    .div_(255.0)
                )
            else:
                batch[key] = value.to(device, non_blocking=False)

    return batch


def primary_prediction(raw):
    if torch.is_tensor(raw):
        return raw
    if (
        isinstance(raw, (tuple, list))
        and len(raw) > 0
        and torch.is_tensor(raw[0])
    ):
        return raw[0]
    fail(f"Could not identify decoded prediction tensor from {type(raw)}")


def nms_native(pred):
    try:
        from ultralytics.utils.ops import non_max_suppression
    except ImportError:
        from ultralytics.utils.nms import non_max_suppression

    return non_max_suppression(
        pred,
        conf_thres=CONF,
        iou_thres=NMS_IOU,
        classes=None,
        agnostic=False,
        multi_label=True,
        max_det=MAX_DET,
        nc=10,
    )


def read_gt(label_file: Path, width: int, height: int):
    rows = []
    text = label_file.read_text(encoding="utf-8").strip()
    if not text:
        return rows

    for gt_index, line in enumerate(text.splitlines()):
        parts = line.split()
        if len(parts) < 5:
            fail(f"Malformed YOLO label: {label_file}")

        cls = int(float(parts[0]))
        xc, yc, bw, bh = map(float, parts[1:5])

        rows.append({
            "gt_index": gt_index,
            "class_id": cls,
            "box": np.asarray(
                [
                    (xc - bw / 2.0) * width,
                    (yc - bh / 2.0) * height,
                    (xc + bw / 2.0) * width,
                    (yc + bh / 2.0) * height,
                ],
                dtype=np.float64,
            ),
        })
    return rows


def iou_one_to_many(box, boxes):
    if len(boxes) == 0:
        return np.empty((0,), dtype=np.float64)

    x1 = np.maximum(box[0], boxes[:, 0])
    y1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[2], boxes[:, 2])
    y2 = np.minimum(box[3], boxes[:, 3])

    inter = (
        np.maximum(0.0, x2 - x1)
        * np.maximum(0.0, y2 - y1)
    )

    area_a = (
        max(0.0, box[2] - box[0])
        * max(0.0, box[3] - box[1])
    )
    area_b = (
        np.maximum(0.0, boxes[:, 2] - boxes[:, 0])
        * np.maximum(0.0, boxes[:, 3] - boxes[:, 1])
    )

    union = area_a + area_b - inter

    return np.divide(
        inter,
        union,
        out=np.zeros_like(inter),
        where=union > 0,
    )


def match_gt_status(gt_rows, pred):
    status = [
        {
            "detected": False,
            "matched_conf": float("nan"),
            "matched_iou": float("nan"),
        }
        for _ in gt_rows
    ]

    if not gt_rows or len(pred["boxes"]) == 0:
        return status

    gt_boxes = np.stack([g["box"] for g in gt_rows], axis=0)
    gt_cls = np.asarray([g["class_id"] for g in gt_rows], dtype=np.int64)

    used_gt = np.zeros(len(gt_rows), dtype=bool)
    order = np.argsort(-pred["conf"])

    for pi in order:
        candidates = np.where(
            (~used_gt)
            & (gt_cls == pred["cls"][pi])
        )[0]
        if len(candidates) == 0:
            continue

        ious = iou_one_to_many(
            pred["boxes"][pi],
            gt_boxes[candidates],
        )

        j = int(np.argmax(ious))
        best = float(ious[j])

        if best >= MATCH_IOU:
            gi = int(candidates[j])
            used_gt[gi] = True
            status[gi] = {
                "detected": True,
                "matched_conf": float(pred["conf"][pi]),
                "matched_iou": best,
            }

    return status


def det_to_original(det, input_hw, original_hw):
    from ultralytics.utils.ops import scale_boxes

    if det.numel() == 0:
        return {
            "boxes": np.empty((0, 4), dtype=np.float64),
            "conf": np.empty((0,), dtype=np.float64),
            "cls": np.empty((0,), dtype=np.int64),
        }

    det = det.clone()
    det[:, :4] = scale_boxes(
        input_hw,
        det[:, :4],
        original_hw,
    )

    d = det.detach().cpu()
    return {
        "boxes": d[:, :4].numpy().astype(np.float64),
        "conf": d[:, 4].numpy().astype(np.float64),
        "cls": d[:, 5].numpy().astype(np.int64),
    }


def item_input_boxes_xywh(item: dict):
    b = item["bboxes"]
    if isinstance(b, torch.Tensor):
        b = b.detach().cpu().numpy()
    else:
        b = np.asarray(b)

    b = np.asarray(b, dtype=np.float64).reshape(-1, 4)
    if len(b) == 0:
        return b

    h, w = map(int, item["resized_shape"])

    if np.nanmax(np.abs(b)) <= 1.5:
        out = b.copy()
        out[:, [0, 2]] *= w
        out[:, [1, 3]] *= h
        return out

    return b


def exact_box_mask_from_xywh(xywh, h_cells: int, w_cells: int, input_hw):
    h_img, w_img = input_hw
    xc, yc, bw, bh = map(float, xywh)

    x1 = xc - bw / 2.0
    y1 = yc - bh / 2.0
    x2 = xc + bw / 2.0
    y2 = yc + bh / 2.0

    cell_w = w_img / float(w_cells)
    cell_h = h_img / float(h_cells)

    xs = (np.arange(w_cells, dtype=np.float64) + 0.5) * cell_w
    ys = (np.arange(h_cells, dtype=np.float64) + 0.5) * cell_h

    inside_x = (xs >= x1) & (xs <= x2)
    inside_y = (ys >= y1) & (ys <= y2)

    mask = np.outer(inside_y, inside_x)

    if not mask.any():
        c = int(np.floor(xc / cell_w))
        r = int(np.floor(yc / cell_h))
        c = int(np.clip(c, 0, w_cells - 1))
        r = int(np.clip(r, 0, h_cells - 1))
        mask[r, c] = True

    return mask


def dilate_square(mask: np.ndarray, radius: int):
    h, w = mask.shape
    out = np.zeros_like(mask, dtype=bool)

    rr, cc = np.where(mask)
    for r, c in zip(rr, cc):
        r0 = max(0, int(r) - radius)
        r1 = min(h, int(r) + radius + 1)
        c0 = max(0, int(c) - radius)
        c1 = min(w, int(c) + radius + 1)
        out[r0:r1, c0:c1] = True
    return out


def flatten_region(feature: torch.Tensor, mask: np.ndarray):
    rr, cc = np.where(mask)
    order = np.lexsort((cc, rr))
    rr = rr[order]
    cc = cc[order]

    rt = torch.as_tensor(rr, device=feature.device, dtype=torch.long)
    ct = torch.as_tensor(cc, device=feature.device, dtype=torch.long)

    cells = feature[0, :, rt, ct].permute(1, 0).contiguous()
    vec = cells.reshape(-1)
    return vec, rt, ct


def replace_region_from_vector(
    feature: torch.Tensor,
    vector: torch.Tensor,
    rows: torch.Tensor,
    cols: torch.Tensor,
):
    out = feature.clone()
    c = int(feature.shape[1])
    m = int(len(rows))
    cells = vector.reshape(m, c)
    out[0, :, rows, cols] = cells.transpose(0, 1)
    return out


def project_n_onto_t(n: torch.Tensor, t: torch.Tensor):
    denom = torch.dot(t, t)
    if not torch.isfinite(denom) or float(denom.item()) <= 1e-20:
        fail("Teacher vector has zero/nonfinite energy.")

    alpha = torch.dot(n, t) / denom
    n_e = alpha * t
    n_c = n - n_e
    return alpha, n_e, n_c


def stable_object_seed(exp4_row_id: str) -> int:
    msg = f"{RANDOM_CONTROL_SEED}|{exp4_row_id}".encode("utf-8")
    digest = hashlib.sha256(msg).digest()
    return int.from_bytes(digest[:8], byteorder="little", signed=False)


def make_random_control_delta(
    exp4_row_id: str,
    t: torch.Tensor,
    n_c: torch.Tensor,
):
    """
    Deterministic random vector with:
        delta_R ⟂ t
        delta_R ⟂ n_C
        ||delta_R|| = ||n_C||

    Because n_C is already orthogonal to t by construction, the two projection
    removals are numerically stable and define a random direction outside both
    the teacher direction and the actual complementary direction.
    """
    target_norm = torch.linalg.vector_norm(n_c)
    if not torch.isfinite(target_norm):
        fail(f"Nonfinite n_C norm for {exp4_row_id}.")

    if float(target_norm.item()) <= 1e-20:
        return torch.zeros_like(n_c), 0.0, 0.0, 0.0

    seed = stable_object_seed(exp4_row_id)
    rng = np.random.default_rng(seed)
    z_np = rng.standard_normal(int(n_c.numel()), dtype=np.float32)
    z = torch.from_numpy(z_np).to(device=n_c.device, dtype=n_c.dtype)

    # Remove teacher direction.
    tt = torch.dot(t, t)
    if float(tt.item()) > 1e-20:
        z = z - (torch.dot(z, t) / tt) * t

    # Remove actual n_C direction.
    cc = torch.dot(n_c, n_c)
    if float(cc.item()) > 1e-20:
        z = z - (torch.dot(z, n_c) / cc) * n_c

    z_norm = torch.linalg.vector_norm(z)
    if not torch.isfinite(z_norm) or float(z_norm.item()) <= 1e-20:
        fail(f"Degenerate random-control direction for {exp4_row_id}.")

    delta = z * (target_norm / z_norm)

    # Numerical diagnostics.
    cos_t = float(
        torch.dot(delta, t).item()
        / max(float(torch.linalg.vector_norm(delta).item()
                    * torch.linalg.vector_norm(t).item()), 1e-20)
    )
    cos_c = float(
        torch.dot(delta, n_c).item()
        / max(float(torch.linalg.vector_norm(delta).item()
                    * torch.linalg.vector_norm(n_c).item()), 1e-20)
    )
    norm_ratio = float(
        torch.linalg.vector_norm(delta).item()
        / max(float(target_norm.item()), 1e-20)
    )

    return delta, cos_t, cos_c, norm_ratio


def capture_modules(model, indices):
    storage = {}
    handles = []

    for idx in indices:
        def hook(_module, _inp, out, idx=idx):
            if not torch.is_tensor(out):
                fail(f"Module {idx} output is not a tensor.")
            storage[idx] = out.detach().clone()
        handles.append(model.model[idx].register_forward_hook(hook))

    return storage, handles


def remove_handles(handles):
    for h in handles:
        h.remove()


def parse_bool_series(series):
    if series.dtype == bool:
        return series
    mapped = series.astype(str).str.strip().str.lower().map({
        "true": True,
        "false": False,
        "1": True,
        "0": False,
    })
    if mapped.isna().any():
        fail(f"Could not parse boolean column {series.name}.")
    return mapped.astype(bool)


def bootstrap_specificity(df: pd.DataFrame):
    """
    Source-cluster bootstrap of:
      ([C_LOST - C_PRES]) - ([R_LOST - R_PRES])

    C = Step-06 targeted n_C removal damage
    R = Step-07 matched random-control damage
    """
    clusters = []

    for source, g in df.groupby("source_stem", sort=True):
        lost = g[g["group"] == "LOST"]
        pres = g[g["group"] == "PRESERVED"]

        clusters.append({
            "source": source,
            "lost_n": int(len(lost)),
            "pres_n": int(len(pres)),
            "c_lost": int(lost["targeted_damaged"].sum()),
            "c_pres": int(pres["targeted_damaged"].sum()),
            "r_lost": int(lost["random_damaged"].sum()),
            "r_pres": int(pres["random_damaged"].sum()),
        })

    if len(clusters) != EXPECTED_PRIMARY_SOURCE_CLUSTERS:
        fail(
            f"Expected {EXPECTED_PRIMARY_SOURCE_CLUSTERS} primary source "
            f"clusters; found {len(clusters)}."
        )

    ln = np.asarray([x["lost_n"] for x in clusters], dtype=np.float64)
    pn = np.asarray([x["pres_n"] for x in clusters], dtype=np.float64)
    cl = np.asarray([x["c_lost"] for x in clusters], dtype=np.float64)
    cp = np.asarray([x["c_pres"] for x in clusters], dtype=np.float64)
    rl = np.asarray([x["r_lost"] for x in clusters], dtype=np.float64)
    rp = np.asarray([x["r_pres"] for x in clusters], dtype=np.float64)

    rng = np.random.default_rng(BOOTSTRAP_SEED)
    vals = np.empty(BOOTSTRAP_REPS, dtype=np.float64)

    k = len(clusters)

    for b in range(BOOTSTRAP_REPS):
        idx = rng.integers(0, k, size=k)

        lost_n = ln[idx].sum()
        pres_n = pn[idx].sum()

        if lost_n <= 0 or pres_n <= 0:
            vals[b] = np.nan
            continue

        c_gap = (
            cl[idx].sum() / lost_n
            - cp[idx].sum() / pres_n
        )
        r_gap = (
            rl[idx].sum() / lost_n
            - rp[idx].sum() / pres_n
        )

        vals[b] = c_gap - r_gap

    vals = vals[np.isfinite(vals)]
    if len(vals) < int(0.99 * BOOTSTRAP_REPS):
        fail("Too many invalid bootstrap replicates.")

    return {
        "reps_valid": int(len(vals)),
        "mean": float(np.mean(vals)),
        "ci_low": float(np.quantile(vals, 0.025)),
        "ci_high": float(np.quantile(vals, 0.975)),
    }


def main():
    cwd = Path.cwd().resolve()
    if cwd.name != "experiment4":
        fail(
            "Run Step 07 from "
            "/AUTHOR_HOME/Documents/Researching/ConsRestore/experiment4."
        )

    if OUT.exists():
        fail(f"Refusing to overwrite existing output: {OUT.resolve()}")
    if TMP.exists():
        fail(
            f"Temporary Step-07 directory exists: {TMP.resolve()}\n"
            "Inspect/remove it before rerunning."
        )

    for p in [
        LOCK_JSON,
        STEP01_JSON,
        STEP04_JSON,
        STEP05_JSON,
        STEP06_JSON,
        STEP06_CSV,
        COHORT_CSV,
        NONE_ARGS,
    ]:
        if not p.is_file():
            fail(f"Required prior artifact missing: {p}")

    lock = json.loads(LOCK_JSON.read_text(encoding="utf-8"))
    audit1 = json.loads(STEP01_JSON.read_text(encoding="utf-8"))
    audit4 = json.loads(STEP04_JSON.read_text(encoding="utf-8"))
    audit5 = json.loads(STEP05_JSON.read_text(encoding="utf-8"))
    audit6 = json.loads(STEP06_JSON.read_text(encoding="utf-8"))

    if lock.get("status") != "EXPERIMENT4_STEP00_SCIENTIFIC_LOCKED":
        fail("Unexpected Step-00 status.")
    if audit1.get("status") != "EXPERIMENT4_STEP01_INPUT_AUDIT_COMPLETE":
        fail("Unexpected Step-01 status.")
    if audit4.get("status") != "EXPERIMENT4_STEP04_HEAD_SUPPORT_LOCKED":
        fail("Unexpected Step-04 status.")
    if audit5.get("replay_pass") is not True:
        fail("Step-05 exact replay did not pass.")
    if audit6.get("status") != "EXPERIMENT4_STEP06_FUNCTIONAL_NC_ABLATION_COMPLETE":
        fail("Unexpected Step-06 status.")
    if audit6.get("gate_pass") is not True:
        fail("Step-06 functional n_C gate did not pass; Step 07 is not justified.")
    if int(audit4["support_lock"]["locked_symmetric_radius"]) != HEAD_RADIUS:
        fail("Step-04 head-support radius changed.")

    companion = prepare_checkpoint_companion()

    import ultralytics
    if str(ultralytics.__version__) != EXPECTED_ULTRALYTICS_VERSION:
        fail(
            f"Expected Ultralytics {EXPECTED_ULTRALYTICS_VERSION}; "
            f"found {ultralytics.__version__}."
        )

    if not torch.cuda.is_available():
        fail("CUDA is required.")

    frozen = lock["frozen_models"]
    teacher_spec = frozen["clean_teacher"]
    none_spec = frozen["none_degraded_student"]

    teacher_ckpt = Path(teacher_spec["path"])
    none_ckpt = Path(none_spec["path"])

    if sha256_file(teacher_ckpt) != teacher_spec["sha256"]:
        fail("Clean-teacher checkpoint changed.")
    if sha256_file(none_ckpt) != none_spec["sha256"]:
        fail("NONE checkpoint changed.")

    manifest_spec = lock["data"]["manifest"]
    manifest_path = Path(manifest_spec["path"])
    if sha256_file(manifest_path) != manifest_spec["sha256"]:
        fail("Gaussian manifest changed.")

    if sha256_file(COHORT_CSV) != audit1["frozen_cohort"]["sha256"]:
        fail("Frozen cohort changed.")

    cohort = pd.read_csv(COHORT_CSV)
    cohort["source_stem"] = cohort["source_stem"].astype(str)
    cohort["sigma"] = pd.to_numeric(cohort["sigma"], errors="raise").astype(int)
    cohort["gt_index"] = pd.to_numeric(cohort["gt_index"], errors="raise").astype(int)
    cohort["class_id"] = pd.to_numeric(cohort["class_id"], errors="raise").astype(int)
    cohort["none_detected"] = parse_bool_series(cohort["none_detected"])
    cohort["clean_kd_detected"] = parse_bool_series(cohort["clean_kd_detected"])

    primary = cohort[cohort["none_detected"]].copy()
    primary["group"] = np.where(
        primary["clean_kd_detected"],
        "PRESERVED",
        "LOST",
    )

    if len(primary) != EXPECTED_NONE_DETECTED:
        fail("Unexpected NONE-detected primary count.")
    if primary["source_stem"].nunique() != EXPECTED_PRIMARY_SOURCE_CLUSTERS:
        fail("Unexpected primary source-cluster count.")

    step06 = pd.read_csv(STEP06_CSV)
    if len(step06) != EXPECTED_NONE_DETECTED:
        fail("Unexpected Step-06 per-object row count.")
    if step06["exp4_row_id"].duplicated().any():
        fail("Duplicate Step-06 object IDs.")

    step06["targeted_damaged"] = parse_bool_series(step06["damaged"])

    step06_key = step06.set_index("exp4_row_id")
    primary_ids = set(primary["exp4_row_id"])
    if set(step06_key.index) != primary_ids:
        fail("Step-06 object IDs do not match the primary cohort.")

    manifest = pd.read_csv(manifest_path)
    val = (
        manifest[
            manifest["consrestore_split"]
            .astype(str)
            .str.upper()
            == "VAL"
        ]
        .sort_values(["source_stem", "sigma"])
        .reset_index(drop=True)
    )

    if len(val) != EXPECTED_VAL_VIEWS:
        fail("Unexpected VAL-view count.")

    deg_img_dir = TMP / "degraded" / "images"
    deg_lab_dir = TMP / "degraded" / "labels"
    clean_img_dir = TMP / "clean" / "images"
    clean_lab_dir = TMP / "clean" / "labels"

    staged_rows = []

    for row in val.itertuples(index=False):
        source = str(row.source_stem)
        sigma = int(row.sigma)
        stem = f"{source}__sigma{sigma}"

        deg = Path(row.degraded_image).resolve()
        clean = Path(row.clean_image).resolve()
        label = Path(row.label_file).resolve()

        for p in [deg, clean, label]:
            if not p.is_file():
                fail(f"Missing staged source: {p}")

        safe_symlink(deg, deg_img_dir / f"{stem}{deg.suffix.lower()}")
        safe_symlink(label, deg_lab_dir / f"{stem}.txt")

        safe_symlink(clean, clean_img_dir / f"{stem}{clean.suffix.lower()}")
        safe_symlink(label, clean_lab_dir / f"{stem}.txt")

        staged_rows.append({
            "source_stem": source,
            "sigma": sigma,
            "generated_stem": stem,
            "label_file": str(label),
        })

    staged = pd.DataFrame(staged_rows)

    hyp = load_hyp()
    deg_dataset = build_dataset(
        deg_img_dir,
        hyp,
        prefix="Exp4 Step07 degraded: ",
    )
    clean_dataset = build_dataset(
        clean_img_dir,
        hyp,
        prefix="Exp4 Step07 clean: ",
    )

    deg_index = {Path(p).stem: i for i, p in enumerate(deg_dataset.im_files)}
    clean_index = {Path(p).stem: i for i, p in enumerate(clean_dataset.im_files)}

    primary_by_key = {
        (str(k[0]), int(k[1])): g.sort_values("gt_index")
        for k, g in primary.groupby(["source_stem", "sigma"], sort=False)
    }

    print("=" * 120)
    print("ConsRestore / RestorDet Experiment 4 — Step 07")
    print("=" * 120)
    print("purpose                 : matched random specificity control")
    print("control                  : n -> n + delta_R in R_H")
    print("delta_R constraints      : orthogonal to t and n_C")
    print("norm match               : ||delta_R|| = ||n_C||")
    print("head-support radius      :", HEAD_RADIUS)
    print("primary rows             :", EXPECTED_NONE_DETECTED)
    print("primary source clusters  :", EXPECTED_PRIMARY_SOURCE_CLUSTERS)
    print("PRESERVED / LOST         :", EXPECTED_PRESERVED, "/", EXPECTED_LOST)
    print("random seed              :", RANDOM_CONTROL_SEED)
    print("bootstrap reps / seed    :", BOOTSTRAP_REPS, "/", BOOTSTRAP_SEED)
    print("phase                    : loading frozen teacher and NONE models")

    from ultralytics import YOLO

    device = torch.device("cuda:0")
    clear_cuda()

    teacher_wrapper = YOLO(str(teacher_ckpt))
    teacher = teacher_wrapper.model.to(device).eval()
    for p in teacher.parameters():
        p.requires_grad_(False)

    none_wrapper = YOLO(str(none_ckpt))
    none_model = none_wrapper.model.to(device).eval()
    for p in none_model.parameters():
        p.requires_grad_(False)

    detect = none_model.model[DETECT_MODULE]

    records = []
    intervention_count = 0
    direct_head_mismatches = 0

    print("phase                    : frozen random-control interventions")

    for vi, srow in staged.iterrows():
        source = str(srow["source_stem"])
        sigma = int(srow["sigma"])
        stem = str(srow["generated_stem"])
        key = (source, sigma)

        targets = primary_by_key.get(key)
        if targets is None or len(targets) == 0:
            continue

        deg_item = deg_dataset[deg_index[stem]]
        clean_item = clean_dataset[clean_index[stem]]

        deg_batch = preprocess_one(deg_item, device)
        clean_batch = preprocess_one(clean_item, device)

        input_hw = tuple(int(x) for x in deg_batch["img"].shape[2:])
        original_hw = tuple(int(x) for x in deg_item["ori_shape"])

        if tuple(deg_batch["img"].shape) != tuple(clean_batch["img"].shape):
            fail(f"Clean/degraded input shape mismatch for {stem}.")
        if tuple(map(int, clean_item["ori_shape"])) != original_hw:
            fail(f"Clean/degraded original shape mismatch for {stem}.")

        gt_rows = read_gt(
            Path(srow["label_file"]),
            width=original_hw[1],
            height=original_hw[0],
        )

        input_boxes = item_input_boxes_xywh(deg_item)
        clean_input_boxes = item_input_boxes_xywh(clean_item)

        if len(input_boxes) != len(gt_rows):
            fail(f"Transformed GT count mismatch for {stem}.")
        if not np.allclose(input_boxes, clean_input_boxes, atol=1e-5, rtol=1e-6):
            fail(f"Clean/degraded transformed GT geometry differs for {stem}.")

        # Clean teacher P4.
        t_store, t_handles = capture_modules(teacher, [P4_MODULE])
        with torch.inference_mode():
            _ = teacher(clean_batch["img"])
        remove_handles(t_handles)
        t_p4 = t_store[P4_MODULE]

        # NONE P3/P4/P5.
        n_store, n_handles = capture_modules(
            none_model,
            [P3_MODULE, P4_MODULE, P5_MODULE],
        )
        with torch.inference_mode():
            _ = none_model(deg_batch["img"])
        remove_handles(n_handles)

        p3 = n_store[P3_MODULE]
        p4 = n_store[P4_MODULE]
        p5 = n_store[P5_MODULE]

        if tuple(p4.shape[2:]) != EXPECTED_P4_HW:
            fail(f"Unexpected P4 shape for {stem}: {tuple(p4.shape)}")
        if tuple(t_p4.shape) != tuple(p4.shape):
            fail(f"Teacher/NONE P4 shape mismatch for {stem}.")

        # Direct-head baseline replay.
        with torch.inference_mode():
            base_raw = detect([p3.clone(), p4.clone(), p5.clone()])
            base_pred = primary_prediction(base_raw)
            base_det = nms_native(base_pred)[0]

        baseline_pred = det_to_original(
            base_det,
            input_hw=input_hw,
            original_hw=original_hw,
        )
        baseline_statuses = match_gt_status(gt_rows, baseline_pred)

        base_rows_all = cohort[
            (cohort["source_stem"] == source)
            & (cohort["sigma"] == sigma)
        ].sort_values("gt_index")

        if len(base_rows_all) != len(gt_rows):
            fail(f"Frozen cohort/GT count mismatch for {stem}.")

        for b in base_rows_all.itertuples(index=False):
            gi = int(b.gt_index)
            if bool(b.none_detected) != bool(baseline_statuses[gi]["detected"]):
                direct_head_mismatches += 1

        if direct_head_mismatches > 0:
            fail(
                "Direct Detect-head replay mismatch before random control; "
                f"first encountered by {stem}."
            )

        for rec in targets.itertuples(index=False):
            gi = int(rec.gt_index)

            if not baseline_statuses[gi]["detected"]:
                fail(f"Primary target not detected in baseline: {rec.exp4_row_id}")

            exact_mask = exact_box_mask_from_xywh(
                input_boxes[gi],
                h_cells=EXPECTED_P4_HW[0],
                w_cells=EXPECTED_P4_HW[1],
                input_hw=input_hw,
            )
            head_mask = dilate_square(exact_mask, HEAD_RADIUS)

            t_vec, tr, tc = flatten_region(t_p4, head_mask)
            n_vec, nr, nc = flatten_region(p4, head_mask)

            if not torch.equal(tr, nr) or not torch.equal(tc, nc):
                fail("Teacher/NONE region indexing mismatch.")

            _alpha, _n_e, n_c = project_n_onto_t(n_vec, t_vec)

            delta_r, cos_rt, cos_rc, norm_ratio = make_random_control_delta(
                rec.exp4_row_id,
                t_vec,
                n_c,
            )

            n_random = n_vec + delta_r

            p4_random = replace_region_from_vector(
                p4,
                n_random,
                nr,
                nc,
            )

            with torch.inference_mode():
                raw_r = detect([
                    p3.clone(),
                    p4_random,
                    p5.clone(),
                ])
                pred_r = primary_prediction(raw_r)
                det_r = nms_native(pred_r)[0]

            pred_r = det_to_original(
                det_r,
                input_hw=input_hw,
                original_hw=original_hw,
            )
            status_r = match_gt_status(gt_rows, pred_r)[gi]

            targeted_row = step06_key.loc[rec.exp4_row_id]
            targeted_damaged = bool(targeted_row["targeted_damaged"])

            group = (
                "PRESERVED"
                if bool(rec.clean_kd_detected)
                else "LOST"
            )

            records.append({
                "exp4_row_id": rec.exp4_row_id,
                "source_stem": source,
                "sigma": sigma,
                "gt_index": gi,
                "class_id": int(rec.class_id),
                "group": group,
                "exact_box_cells": int(exact_mask.sum()),
                "head_support_cells": int(head_mask.sum()),
                "nC_norm": float(torch.linalg.vector_norm(n_c).item()),
                "random_delta_norm": float(
                    torch.linalg.vector_norm(delta_r).item()
                ),
                "random_norm_ratio": float(norm_ratio),
                "random_cos_teacher": float(cos_rt),
                "random_cos_nC": float(cos_rc),
                "targeted_damaged": targeted_damaged,
                "random_after_detected": bool(status_r["detected"]),
                "random_after_conf": float(status_r["matched_conf"]),
                "random_after_iou": float(status_r["matched_iou"]),
                "random_damaged": bool(not status_r["detected"]),
            })

            intervention_count += 1

            del p4_random, raw_r, pred_r, det_r, status_r

            if intervention_count % 500 == 0:
                print(
                    f"objects controlled       : "
                    f"{intervention_count}/{EXPECTED_NONE_DETECTED}"
                )

        del (
            deg_batch,
            clean_batch,
            t_p4,
            p3,
            p4,
            p5,
            base_raw,
            base_pred,
            base_det,
        )

        if (vi + 1) % 100 == 0 or (vi + 1) == len(staged):
            clear_cuda()

    if intervention_count != EXPECTED_NONE_DETECTED:
        fail(
            f"Expected {EXPECTED_NONE_DETECTED} controls; "
            f"performed {intervention_count}."
        )

    result = pd.DataFrame(records)

    if len(result) != EXPECTED_NONE_DETECTED:
        fail("Unexpected control-result row count.")
    if result["exp4_row_id"].duplicated().any():
        fail("Duplicate object IDs.")

    lost = result[result["group"] == "LOST"].copy()
    pres = result[result["group"] == "PRESERVED"].copy()

    if len(lost) != EXPECTED_LOST or len(pres) != EXPECTED_PRESERVED:
        fail("Unexpected LOST/PRESERVED counts.")

    # Step-06 targeted damage.
    c_lost_rate = float(lost["targeted_damaged"].mean())
    c_pres_rate = float(pres["targeted_damaged"].mean())
    c_gap = c_lost_rate - c_pres_rate

    # Step-07 random-control damage.
    r_lost_rate = float(lost["random_damaged"].mean())
    r_pres_rate = float(pres["random_damaged"].mean())
    r_gap = r_lost_rate - r_pres_rate

    delta_specific = c_gap - r_gap

    # Persist expensive control results before bootstrap/reporting.
    shutil.rmtree(TMP, ignore_errors=True)
    OUT.mkdir(parents=True, exist_ok=False)

    csv_path = OUT / "PER_OBJECT_RANDOM_CONTROL.csv"
    json_path = OUT / "SPECIFICITY_GATE.json"
    md_path = OUT / "SPECIFICITY_GATE.md"

    result.to_csv(csv_path, index=False)

    print("phase                    : source-cluster specificity bootstrap")
    boot = bootstrap_specificity(result)

    gate_pass = bool(boot["ci_low"] > 0.0)

    decision = (
        "PASS_NC_REMOVAL_SPECIFIC_BEYOND_MATCHED_RANDOM_PERTURBATION"
        if gate_pass
        else "FAIL_NC_REMOVAL_NOT_SPECIFIC_BEYOND_MATCHED_RANDOM_PERTURBATION"
    )

    max_abs_cos_t = float(result["random_cos_teacher"].abs().max())
    max_abs_cos_c = float(result["random_cos_nC"].abs().max())
    max_norm_error = float((result["random_norm_ratio"] - 1.0).abs().max())

    audit = {
        "status": "EXPERIMENT4_STEP07_RANDOM_CONTROL_COMPLETE",
        "decision": decision,
        "gate_pass": gate_pass,
        "scientific_question": (
            "Is the Step-06 LOST-vs-PRESERVED damage gap specific to "
            "removing n_C rather than generic sensitivity to an equally "
            "large perturbation in R_H?"
        ),
        "control": {
            "definition": "n -> n + delta_R within R_H",
            "delta_R_orthogonal_to_teacher": True,
            "delta_R_orthogonal_to_nC": True,
            "delta_R_norm_equals_nC_norm": True,
            "seed": RANDOM_CONTROL_SEED,
            "stable_per_object_seed": "SHA256(global_seed|exp4_row_id)",
        },
        "primary": {
            "targeted_damage_rate_lost": c_lost_rate,
            "targeted_damage_rate_preserved": c_pres_rate,
            "targeted_gap": c_gap,
            "random_damage_rate_lost": r_lost_rate,
            "random_damage_rate_preserved": r_pres_rate,
            "random_gap": r_gap,
            "delta_specific": delta_specific,
            "source_cluster_bootstrap": boot,
            "gate_rule": "95% CI lower bound for delta_specific > 0",
        },
        "control_diagnostics": {
            "max_abs_cos_delta_teacher": max_abs_cos_t,
            "max_abs_cos_delta_nC": max_abs_cos_c,
            "max_abs_norm_ratio_minus_1": max_norm_error,
            "direct_head_replay_mismatches": direct_head_mismatches,
        },
        "counts": {
            "rows": int(len(result)),
            "lost": int(len(lost)),
            "preserved": int(len(pres)),
            "primary_source_clusters": int(
                result["source_stem"].nunique()
            ),
            "random_damaged_lost": int(lost["random_damaged"].sum()),
            "random_damaged_preserved": int(pres["random_damaged"].sum()),
            "targeted_damaged_lost": int(lost["targeted_damaged"].sum()),
            "targeted_damaged_preserved": int(pres["targeted_damaged"].sum()),
        },
        "provenance": {
            "step00_lock_sha256": sha256_file(LOCK_JSON),
            "step01_audit_sha256": sha256_file(STEP01_JSON),
            "step04_audit_sha256": sha256_file(STEP04_JSON),
            "step05_audit_sha256": sha256_file(STEP05_JSON),
            "step06_gate_sha256": sha256_file(STEP06_JSON),
            "step06_per_object_sha256": sha256_file(STEP06_CSV),
            "frozen_cohort_sha256": sha256_file(COHORT_CSV),
            "teacher_checkpoint_sha256": sha256_file(teacher_ckpt),
            "none_checkpoint_sha256": sha256_file(none_ckpt),
            "manifest_sha256": sha256_file(manifest_path),
            "checkpoint_companion": companion,
            "ultralytics_version": str(ultralytics.__version__),
        },
        "firewall": {
            "training": False,
            "threshold_tuning": False,
            "radius_tuning": False,
            "clean_kd_feature_intervention": False,
            "devtest": False,
            "acdc": False,
        },
    }

    json_path.write_text(
        json.dumps(audit, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )

    md_path.write_text(
        f"""# ConsRestore / RestorDet Experiment 4 — Step 07

## Status

`EXPERIMENT4_STEP07_RANDOM_CONTROL_COMPLETE`

## Control

For each object:

```text
delta_R ⟂ t
delta_R ⟂ n_C
||delta_R|| = ||n_C||
n -> n + delta_R
```

inside the same locked `R_H` used in Step 06.

## Primary specificity result

```text
Targeted n_C removal:
  LOST damage rate          {c_lost_rate:+.8f}
  PRESERVED damage rate     {c_pres_rate:+.8f}
  LOST - PRESERVED          {c_gap:+.8f}

Matched random control:
  LOST damage rate          {r_lost_rate:+.8f}
  PRESERVED damage rate     {r_pres_rate:+.8f}
  LOST - PRESERVED          {r_gap:+.8f}

Delta_specific              {delta_specific:+.8f}
95% CI                      [{boot['ci_low']:+.8f}, {boot['ci_high']:+.8f}]
```

Primary gate:

```text
95% CI lower bound for Delta_specific > 0
```

Gate pass:

```text
{gate_pass}
```

Decision:

```text
{decision}
```

## Firewall

```text
training                     NO
threshold tuning             NO
radius tuning                NO
CLEAN-KD feature intervention NO
DEVTEST / ACDC               NO / NO
```
""",
        encoding="utf-8",
    )

    print()
    print("[SPECIFICITY CONTROL]")
    print("targeted LOST damage    :", f"{c_lost_rate:+.8f}")
    print("targeted PRES damage    :", f"{c_pres_rate:+.8f}")
    print("targeted gap            :", f"{c_gap:+.8f}")
    print()
    print("random LOST damage      :", f"{r_lost_rate:+.8f}")
    print("random PRES damage      :", f"{r_pres_rate:+.8f}")
    print("random gap              :", f"{r_gap:+.8f}")
    print()
    print("Delta_specific          :", f"{delta_specific:+.8f}")
    print(
        "95% CI Delta_specific   :",
        f"[{boot['ci_low']:+.8f}, {boot['ci_high']:+.8f}]",
    )
    print()
    print("GATE PASS               :", gate_pass)
    print("DECISION                :", decision)
    print()
    print("max |cos(delta_R,t)|    :", f"{max_abs_cos_t:.3e}")
    print("max |cos(delta_R,n_C)|  :", f"{max_abs_cos_c:.3e}")
    print("max norm-ratio error    :", f"{max_norm_error:.3e}")
    print()
    print("training                : NO")
    print("CLEAN-KD intervention   : NO")
    print("DEVTEST / ACDC          : NO / NO")
    print()
    print("STEP 07 COMPLETE")
    print("result JSON             :", json_path.resolve())
    print("per-object control      :", csv_path.resolve())


if __name__ == "__main__":
    main()
