# Distribution and licensing boundary

This reproducibility package is designed so that the public source archive does **not** redistribute BDD100K imagery/annotations or the trained YOLO11s / RT-DETR-L checkpoints. Users supply those assets locally and the preflight scripts verify their frozen SHA-256 identities.

Small project-authored replay artifacts that are necessary to reproduce the frozen scientific protocol can be staged with `scripts/stage_public_reference_artifacts.py` after the authors verify that they are appropriate to redistribute. The staging script never copies the BDD100K dataset or detector checkpoints.

## Repository license

The project-authored source code in this repository is released under the **BSD 3-Clause License**. See the top-level `LICENSE` file. This repository license does not override the licenses or terms governing third-party packages, datasets, pretrained model files, or other externally supplied materials.

Third-party packages, datasets, and model files remain subject to their own licenses/terms regardless of the license selected for this repository.
