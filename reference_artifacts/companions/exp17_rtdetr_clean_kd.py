"""
Experiment 17 — importable RT-DETR CLEAN-KD model companion.

This module must remain importable when checkpoints created by the custom
RT-DETR CLEAN-KD trainer are loaded.

The teacher is attached as a non-registered reference so it is not included in
student state_dict/checkpoints/optimizer parameter groups.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from ultralytics.nn.tasks import RTDETRDetectionModel


P4_MODULE_INDEX = 24
P4_CHANNELS = 256
P4_HW = 40


class CleanKDRTDETRDetectionModel(RTDETRDetectionModel):
    """
    RT-DETR model with an optional frozen clean-teacher P4 cosine-KD objective.

    Training batch contract:
        batch["teacher_img"] exists -> detection + CLEAN-KD
        batch["teacher_img"] absent -> native RT-DETR loss only

    The latter behavior is required for ordinary validation.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        object.__setattr__(self, "_clean_teacher_ref", None)
        object.__setattr__(self, "_student_hook_handle", None)
        object.__setattr__(self, "_teacher_hook_handle", None)

        self._student_p4 = None
        self._teacher_p4 = None
        self._lambda_kd = None

        # Smoke/provenance counters; not trainable tensors.
        self._kd_count = 0
        self._kd_sum = 0.0
        self._last_kd = None
        self._last_det = None
        self._last_total = None

    def _remove_clean_kd_hooks(self):
        for attr in ("_student_hook_handle", "_teacher_hook_handle"):
            handle = getattr(self, attr, None)
            if handle is not None:
                handle.remove()
                object.__setattr__(self, attr, None)

    def attach_clean_teacher(self, teacher, lambda_kd: float):
        self._remove_clean_kd_hooks()

        if teacher is None:
            raise RuntimeError("CLEAN-KD teacher cannot be None.")

        value = float(lambda_kd)
        if not (value > 0.0):
            raise RuntimeError(f"lambda_kd must be positive, got {value}")

        teacher.eval()
        for p in teacher.parameters():
            p.requires_grad_(False)

        if len(self.model) <= P4_MODULE_INDEX:
            raise RuntimeError(
                f"Student has {len(self.model)} modules; "
                f"cannot tap module {P4_MODULE_INDEX}."
            )

        if len(teacher.model) <= P4_MODULE_INDEX:
            raise RuntimeError(
                f"Teacher has {len(teacher.model)} modules; "
                f"cannot tap module {P4_MODULE_INDEX}."
            )

        object.__setattr__(self, "_clean_teacher_ref", teacher)
        self._lambda_kd = value

        def student_hook(_module, _inputs, output):
            if not isinstance(output, torch.Tensor):
                raise RuntimeError(
                    "Student RT-DETR module-24 output is not a tensor."
                )
            self._student_p4 = output

        def teacher_hook(_module, _inputs, output):
            if not isinstance(output, torch.Tensor):
                raise RuntimeError(
                    "Teacher RT-DETR module-24 output is not a tensor."
                )
            self._teacher_p4 = output.detach()

        student_handle = self.model[P4_MODULE_INDEX].register_forward_hook(
            student_hook
        )
        teacher_handle = teacher.model[P4_MODULE_INDEX].register_forward_hook(
            teacher_hook
        )

        object.__setattr__(
            self,
            "_student_hook_handle",
            student_handle,
        )
        object.__setattr__(
            self,
            "_teacher_hook_handle",
            teacher_handle,
        )

    def loss(self, batch, preds=None):
        # Validation / native inference-loss path: no KD.
        if "teacher_img" not in batch:
            return super().loss(batch, preds=preds)

        teacher = getattr(self, "_clean_teacher_ref", None)
        if teacher is None:
            raise RuntimeError(
                "CLEAN-KD batch provided but clean teacher is not attached."
            )

        if self._lambda_kd is None:
            raise RuntimeError(
                "CLEAN-KD batch provided but lambda_kd is not configured."
            )

        teacher_img = batch["teacher_img"]

        if not isinstance(teacher_img, torch.Tensor):
            raise RuntimeError("teacher_img is not a tensor.")

        if tuple(teacher_img.shape) != tuple(batch["img"].shape):
            raise RuntimeError(
                "Student/teacher image batch shapes differ: "
                f"{tuple(batch['img'].shape)} vs "
                f"{tuple(teacher_img.shape)}"
            )

        self._student_p4 = None
        self._teacher_p4 = None

        with torch.no_grad():
            _ = teacher(teacher_img)

        if self._teacher_p4 is None:
            raise RuntimeError("Teacher module-24 hook did not fire.")

        det_loss, loss_items = super().loss(batch, preds=preds)

        if self._student_p4 is None:
            raise RuntimeError("Student module-24 hook did not fire.")

        student_p4 = self._student_p4
        teacher_p4 = self._teacher_p4

        expected = (
            int(batch["img"].shape[0]),
            P4_CHANNELS,
            P4_HW,
            P4_HW,
        )

        if tuple(student_p4.shape) != expected:
            raise RuntimeError(
                f"Student P4 shape {tuple(student_p4.shape)} != {expected}"
            )

        if tuple(teacher_p4.shape) != expected:
            raise RuntimeError(
                f"Teacher P4 shape {tuple(teacher_p4.shape)} != {expected}"
            )

        cosine_map = F.cosine_similarity(
            student_p4,
            teacher_p4,
            dim=1,
            eps=1e-8,
        )

        kd_mean = (1.0 - cosine_map).mean()
        kd_batch = kd_mean * float(batch["img"].shape[0])

        if not torch.isfinite(kd_batch):
            raise RuntimeError("CLEAN-KD loss is non-finite.")

        total = det_loss + self._lambda_kd * kd_batch

        if not torch.isfinite(total):
            raise RuntimeError("RT-DETR CLEAN-KD total loss is non-finite.")

        self._kd_count += 1
        self._kd_sum += float(kd_mean.detach().cpu())
        self._last_kd = float(kd_mean.detach().cpu())
        self._last_det = float(det_loss.detach().cpu())
        self._last_total = float(total.detach().cpu())

        # Do not add a fourth logged loss component: RTDETRTrainer's native
        # loss_names/validator expect the standard three reporting items.
        return total, loss_items
