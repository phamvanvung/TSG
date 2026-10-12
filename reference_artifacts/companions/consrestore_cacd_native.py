#!/usr/bin/env python
from __future__ import annotations

import copy
import torch
import torch.nn.functional as F
from ultralytics.nn.tasks import DetectionModel

P4_MODULE_INDEX = 19
EXPECTED_P4_CHANNELS = 256
EXPECTED_P4_HW = 40
EPS = 1e-6
CACD_OFFSETS_DY_DX = (
    (0, 0),
    (-1, -1), (-1, 0), (-1, 1),
    (0, -1), (0, 1),
    (1, -1), (1, 0), (1, 1),
)


def fail(message: str) -> None:
    raise RuntimeError(message)


def capture_p4_once(model: torch.nn.Module, image: torch.Tensor) -> torch.Tensor:
    box = {"value": None}

    def hook(_m, _i, output):
        if not isinstance(output, torch.Tensor):
            fail(f"P4 hook expected Tensor, got {type(output)}.")
        box["value"] = output

    if not hasattr(model, "model") or len(model.model) <= P4_MODULE_INDEX:
        fail(f"Model has no module index {P4_MODULE_INDEX}.")

    h = model.model[P4_MODULE_INDEX].register_forward_hook(hook)
    try:
        _ = model(image)
    finally:
        h.remove()

    if box["value"] is None:
        fail("P4 hook did not fire.")
    return box["value"]


def validate_p4(name: str, feat: torch.Tensor, batch_size: int) -> None:
    expected = (batch_size, EXPECTED_P4_CHANNELS, EXPECTED_P4_HW, EXPECTED_P4_HW)
    if tuple(feat.shape) != expected:
        fail(f"{name} P4: expected {expected}, found {tuple(feat.shape)}.")


def clean_kd_mean(student_p4: torch.Tensor, clean_teacher_p4: torch.Tensor) -> torch.Tensor:
    s = F.normalize(student_p4.float(), p=2, dim=1, eps=EPS)
    t = F.normalize(clean_teacher_p4.detach().float(), p=2, dim=1, eps=EPS)
    return (1.0 - (s * t).sum(dim=1)).mean()


def valid_mask(h: int, w: int, dy: int, dx: int, device) -> torch.Tensor:
    yy = torch.arange(h, device=device).view(h, 1)
    xx = torch.arange(w, device=device).view(1, w)
    return (
        (yy + dy >= 0) & (yy + dy < h)
        & (xx + dx >= 0) & (xx + dx < w)
    )


def cacd_target_and_stats(
    degraded_teacher_p4: torch.Tensor,
    clean_teacher_p4: torch.Tensor,
):
    td = F.normalize(degraded_teacher_p4.detach().float(), p=2, dim=1, eps=EPS)
    tc = F.normalize(clean_teacher_p4.detach().float(), p=2, dim=1, eps=EPS)

    if td.shape != tc.shape:
        fail(f"CACD teacher shape mismatch: {tuple(td.shape)} vs {tuple(tc.shape)}.")

    _b, _c, h, w = tc.shape
    padded = F.pad(tc, (1, 1, 1, 1), mode="constant", value=0.0)

    sims = []
    for dy, dx in CACD_OFFSETS_DY_DX:
        cand = padded[:, :, 1 + dy:1 + dy + h, 1 + dx:1 + dx + w]
        sim = (td * cand).sum(dim=1)
        mask = valid_mask(h, w, dy, dx, tc.device)
        sims.append(sim.masked_fill(~mask.unsqueeze(0), float("-inf")))

    sim_stack = torch.stack(sims, dim=1)
    best = sim_stack.argmax(dim=1)

    target = torch.zeros_like(tc)
    for k, (dy, dx) in enumerate(CACD_OFFSETS_DY_DX):
        cand = padded[:, :, 1 + dy:1 + dy + h, 1 + dx:1 + dx + w]
        target = torch.where((best == k).unsqueeze(1), cand, target)

    best_sim = sim_stack.gather(1, best.unsqueeze(1)).squeeze(1)
    center_sim = sim_stack[:, 0]

    stats = {
        "noncenter_fraction": float((best != 0).float().mean().cpu().item()),
        "center_cosine_mean": float(center_sim.mean().cpu().item()),
        "best_cosine_mean": float(best_sim.mean().cpu().item()),
        "local_gain_mean": float((best_sim - center_sim).mean().cpu().item()),
    }
    return target.detach(), stats


def cacd_kd_mean(student_p4, degraded_teacher_p4, clean_teacher_p4):
    target, stats = cacd_target_and_stats(degraded_teacher_p4, clean_teacher_p4)
    s = F.normalize(student_p4.float(), p=2, dim=1, eps=EPS)
    kd = (1.0 - (s * target).sum(dim=1)).mean()
    return kd, stats


class ConsRestoreKDDetectionModel(DetectionModel):
    """Importable runtime-teacher model for CLEAN-KD and CACD."""

    def __init__(self, cfg="yolo11s.yaml", ch=3, nc=None, verbose=True):
        super().__init__(cfg=cfg, ch=ch, nc=nc, verbose=verbose)
        self._consrestore_arm = "CLEAN-KD"
        self._consrestore_lambda_kd = 0.0
        object.__setattr__(self, "_consrestore_teacher", None)

        self._consrestore_kd_count = 0
        self._consrestore_kd_sum = 0.0
        self._consrestore_first_kd_p4_grad_norm = None
        self._consrestore_first_student_p4_shape = None
        self._consrestore_first_teacher_clean_p4_shape = None
        self._consrestore_first_teacher_degraded_p4_shape = None
        self._consrestore_cacd_stat_count = 0
        self._consrestore_cacd_noncenter_sum = 0.0
        self._consrestore_cacd_gain_sum = 0.0

    def __getstate__(self):
        state = copy.copy(self.__dict__)
        state["_consrestore_teacher"] = None
        return state

    def configure_consrestore(self, arm: str, lambda_kd: float) -> None:
        arm = str(arm).upper()
        if arm not in {"CLEAN-KD", "CACD"}:
            fail(f"Unsupported KD arm: {arm}")
        self._consrestore_arm = arm
        self._consrestore_lambda_kd = float(lambda_kd)

    def attach_teacher(self, teacher: torch.nn.Module) -> None:
        teacher.eval()
        for p in teacher.parameters():
            p.requires_grad_(False)
        object.__setattr__(self, "_consrestore_teacher", teacher)

    def student_predict_with_p4(self, image: torch.Tensor):
        box = {"value": None}

        def hook(_m, _i, output):
            if not isinstance(output, torch.Tensor):
                fail(f"Student P4 hook expected Tensor, got {type(output)}.")
            box["value"] = output

        h = self.model[P4_MODULE_INDEX].register_forward_hook(hook)
        try:
            preds = self.predict(image)
        finally:
            h.remove()

        if box["value"] is None:
            fail("Student P4 hook did not fire.")
        return preds, box["value"]

    def loss(self, batch, preds=None):
        # Native validation/inference loss only.
        if not self.training:
            return super().loss(batch, preds=preds)

        teacher = getattr(self, "_consrestore_teacher", None)
        if teacher is None:
            fail("KD training reached model without attached teacher.")
        if "teacher_img" not in batch:
            fail("KD training batch missing teacher_img.")

        if getattr(self, "criterion", None) is None:
            self.criterion = self.init_criterion()

        preds, student_p4 = self.student_predict_with_p4(batch["img"])
        det_loss, loss_items = self.criterion(preds, batch)
        b = int(batch["img"].shape[0])
        validate_p4("student", student_p4, b)

        with torch.no_grad():
            td = (
                capture_p4_once(teacher, batch["img"])
                if self._consrestore_arm == "CACD"
                else None
            )
            tc = capture_p4_once(teacher, batch["teacher_img"])

        validate_p4("teacher clean", tc, b)
        if td is not None:
            validate_p4("teacher degraded", td, b)

        if self._consrestore_arm == "CLEAN-KD":
            kd_mean = clean_kd_mean(student_p4, tc)
            stats = None
        else:
            kd_mean, stats = cacd_kd_mean(student_p4, td, tc)

        if not torch.isfinite(kd_mean):
            fail(f"Non-finite {self._consrestore_arm} KD mean.")

        if self._consrestore_first_kd_p4_grad_norm is None:
            g = torch.autograd.grad(
                kd_mean,
                student_p4,
                retain_graph=True,
                allow_unused=True,
            )[0]
            if g is None:
                fail(f"{self._consrestore_arm} has no gradient to student P4.")
            gn = float(g.detach().float().pow(2).sum().sqrt().cpu().item())
            if gn <= 0.0:
                fail(f"{self._consrestore_arm} P4 gradient is zero.")
            self._consrestore_first_kd_p4_grad_norm = gn
            self._consrestore_first_student_p4_shape = list(student_p4.shape)
            self._consrestore_first_teacher_clean_p4_shape = list(tc.shape)
            if td is not None:
                self._consrestore_first_teacher_degraded_p4_shape = list(td.shape)

        self._consrestore_kd_count += 1
        self._consrestore_kd_sum += float(kd_mean.detach().cpu().item())

        if stats is not None:
            self._consrestore_cacd_stat_count += 1
            self._consrestore_cacd_noncenter_sum += stats["noncenter_fraction"]
            self._consrestore_cacd_gain_sum += stats["local_gain_mean"]

        kd_batch = kd_mean * float(b)
        total_loss = det_loss + self._consrestore_lambda_kd * kd_batch
        return total_loss, loss_items
