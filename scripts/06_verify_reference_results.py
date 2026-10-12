#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tsg_repro.reference import compare_numeric_tree  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--observed", required=True)
    p.add_argument("--reference", required=True)
    p.add_argument("--atol", type=float, default=1e-8)
    p.add_argument("--rtol", type=float, default=1e-6)
    a = p.parse_args()
    observed = json.loads(Path(a.observed).read_text(encoding="utf-8"))
    reference = json.loads(Path(a.reference).read_text(encoding="utf-8"))
    mismatches = compare_numeric_tree(observed, reference, atol=a.atol, rtol=a.rtol)
    print(json.dumps({"status": "PASS" if not mismatches else "FAIL", "mismatches": mismatches}, indent=2))
    raise SystemExit(0 if not mismatches else 2)


if __name__ == "__main__":
    main()
