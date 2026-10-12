#!/usr/bin/env python
"""Freeze the successful reproduction-machine environment into the release contract."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

FIELDS = ("python", "torch", "torch_cuda", "cudnn", "gpu", "numpy", "scipy", "pandas")


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    p = argparse.ArgumentParser()
    p.add_argument("--report", required=True, help="JSON produced by verify_environment.py")
    p.add_argument(
        "--requirements",
        default=str(root / "configs" / "environment.requirements.json"),
    )
    p.add_argument("--write", action="store_true", help="Write the frozen values in place")
    a = p.parse_args()

    report_path = Path(a.report).expanduser().resolve()
    req_path = Path(a.requirements).expanduser().resolve()
    report = json.loads(report_path.read_text(encoding="utf-8"))
    req = json.loads(req_path.read_text(encoding="utf-8"))

    if report.get("status") != "PASS":
        raise RuntimeError("Environment report must have status PASS before freezing")
    obs = report.get("observed", {})
    if obs.get("cuda_available") is not True:
        raise RuntimeError("Successful stable reference environment must have CUDA available")
    if obs.get("ultralytics") != req.get("required_exact", {}).get("ultralytics"):
        raise RuntimeError("Ultralytics version differs from the required exact release version")

    frozen = {}
    for key in FIELDS:
        value = obs.get(key)
        if value is None:
            raise RuntimeError(f"Environment report lacks required observed field: {key}")
        frozen[key] = value

    req["freeze_from_successful_reference_run"] = frozen
    text = json.dumps(req, indent=2) + "\n"

    print("[FROZEN REFERENCE ENVIRONMENT]")
    for key in FIELDS:
        print(f"{key:<12}: {frozen[key]}")

    if a.write:
        backup = req_path.with_suffix(req_path.suffix + ".pre_freeze")
        if not backup.exists():
            backup.write_bytes(req_path.read_bytes())
        req_path.write_text(text, encoding="utf-8")
        print(f"written      : {req_path}")
        print(f"backup       : {backup}")
    else:
        print("\nDry run only. Re-run with --write to update the contract.")


if __name__ == "__main__":
    main()
