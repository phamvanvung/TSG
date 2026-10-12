#!/usr/bin/env python
"""Capture runtime versions used for a reproduction run."""
from __future__ import annotations

import json
import platform
import sys


def version_or_none(module_name: str):
    try:
        module = __import__(module_name)
        return getattr(module, "__version__", None)
    except Exception:
        return None


def main() -> None:
    report = {
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": version_or_none("numpy"),
        "scipy": version_or_none("scipy"),
        "pandas": version_or_none("pandas"),
        "torch": version_or_none("torch"),
        "ultralytics": version_or_none("ultralytics"),
    }
    try:
        import torch

        report["cuda_available"] = bool(torch.cuda.is_available())
        report["torch_cuda"] = torch.version.cuda
        report["cudnn"] = (
            int(torch.backends.cudnn.version())
            if torch.backends.cudnn.is_available()
            else None
        )
        report["gpu"] = (
            torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
        )
    except Exception:
        report["cuda_available"] = None
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
