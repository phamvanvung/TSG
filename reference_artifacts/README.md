# Bundled small reference artifacts

Stable public v1.0 should bundle the small, author-generated scientific intermediates and checkpoint companion source files that are required for exact replay but are not third-party datasets/checkpoints.

They are staged from the successful author machine with:

```bash
python scripts/stage_public_reference_artifacts.py --config configs/paths.json
```

The staging script verifies all known frozen SHA-256 values before copying. BDD100K imagery/labels and detector checkpoints are deliberately **not** copied; users must obtain/provide those under their applicable licenses.

Expected bundled items:

- YOLO teacher-canonical `R_NT` and `R_ST`;
- frozen YOLO task-sensitive `C_g`;
- Experiment-4 random-control helper as either exact historical source or a provenance-locked path-sanitized derivative with unchanged `make_random_control_delta` AST;
- YOLO checkpoint companion source;
- RT-DETR checkpoint companion source;
- frozen historical RT-DETR Step-10 bridge;
- YOLO native-loss `args.yaml` used for replay.
