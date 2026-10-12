# v1.0.0

rc2 is the reconciliation and public-release-hardening successor to `v1.0.0-rc1-end-to-end`.

## Scientific reconciliation

- Replaces stale rc1 RT-DETR target values with the successful checkpoint-level reference execution under the corrected authoritative Step-14 replay protocol.
- Freezes the successful RT-DETR teacher-canonical map hashes, task-sensitive `C_g` hash, Step-12 local-utility result, Step-14 finite-horizon result, and canonical orientation result.
- Preserves the older target JSON as an explicitly named legacy audit record; it is not a release gate.
- Distinguishes YOLO rebuild-audit geometry from the frozen authoritative maps/`C_g` actually used by Exp15A/Exp16A.

## Added

- first-class RT-DETR Gaussian VAL detector-performance reproduction, including paired 250-source CORE9 bootstrap;
- end-to-end Step-12→Step-14 cell-table hash linkage check;
- `--from-stage` support for both detector-chain drivers;
- public staging tool for bundle-safe small author-generated artifacts and checkpoint companion source;
- exact NumPy/SciPy/pandas fields in the stable environment freeze;
- `docs/RC2_RECONCILIATION.md` and strengthened stable-release checklist.

## Folded-in rc1 hotfixes

- Ultralytics 8.4.131 temporary YOLO validation YAMLs now provide both `train:` and `val:` schema keys while still executing only `model.val()`;
- YOLO native predictions are cloned before NMS to avoid in-place mutation of PyTorch inference tensors.

## Stable-v1 gate

Stable `v1.0` is emitted only after:

1. bundle-safe reference artifacts are staged;
2. both checkpoint chains complete under the public rc2 tree;
3. the successful exact environment is frozen;
4. `scripts/verify_end_to_end.py` reports `FULL_CHECKPOINT_REPRODUCTION_PASS`;
5. private-path/archive integrity checks and the release manifest pass.
