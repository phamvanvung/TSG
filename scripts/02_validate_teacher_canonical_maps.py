#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tsg_repro.hashing import sha256_file  # noqa: E402
from tsg_repro.mapping import identity_distance, orthogonality_error, validate_map  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--audit-dir", required=True)
    p.add_argument("--reference", default=str(ROOT / "reference_results/canonical_mapping_hashes.json"))
    p.add_argument("--channels", type=int, default=256)
    p.add_argument("--require-reference-map-hashes", action="store_true")
    a = p.parse_args()

    d = Path(a.audit_dir)
    r_nt_p = d / "R_NT_NONE_TO_TEACHER.npy"
    r_st_p = d / "R_ST_CLEAN_KD_TO_TEACHER.npy"
    audit_p = d / "TEACHER_CANONICAL_COORDINATE_AUDIT.json"
    for x in (r_nt_p, r_st_p, audit_p):
        if not x.is_file():
            raise SystemExit(f"missing required artifact: {x}")

    r_nt = np.load(r_nt_p, allow_pickle=False)
    r_st = np.load(r_st_p, allow_pickle=False)
    validate_map(r_nt, channels=a.channels)
    validate_map(r_st, channels=a.channels)

    audit = json.loads(audit_p.read_text(encoding="utf-8"))
    if audit.get("status") != "TSG_TEACHER_CANONICAL_COORDINATE_AUDIT_COMPLETE":
        raise SystemExit("unexpected audit status")
    scope = audit.get("scope", {})
    if scope.get("val_used") is not False or scope.get("pres_lost_labels_used") is not False:
        raise SystemExit("fit firewall violated: VAL or PRES/LOST labels were used")

    observed = {
        "R_NT_NONE_TO_TEACHER.npy": sha256_file(r_nt_p),
        "R_ST_CLEAN_KD_TO_TEACHER.npy": sha256_file(r_st_p),
        "TEACHER_CANONICAL_COORDINATE_AUDIT.json": sha256_file(audit_p),
    }
    ref = json.loads(Path(a.reference).read_text(encoding="utf-8")) if Path(a.reference).is_file() else {}
    expected = ref.get("sha256", {})
    hash_match = {k: (expected.get(k) == v if k in expected else None) for k, v in observed.items()}

    if a.require_reference_map_hashes:
        required = ("R_NT_NONE_TO_TEACHER.npy", "R_ST_CLEAN_KD_TO_TEACHER.npy")
        bad = [k for k in required if hash_match.get(k) is not True]
        if bad:
            raise SystemExit("frozen canonical map hash mismatch: " + ", ".join(bad))

    report = {
        "status": "PASS",
        "shape": list(r_nt.shape),
        "R_NT_orthogonality_error": orthogonality_error(r_nt),
        "R_ST_orthogonality_error": orthogonality_error(r_st),
        "R_NT_identity_distance": identity_distance(r_nt),
        "R_ST_identity_distance": identity_distance(r_st),
        "sha256": observed,
        "reference_hash_match": hash_match,
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
