#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tsg_repro.contracts import require_train_only_metadata  # noqa: E402
from tsg_repro.hashing import sha256_file  # noqa: E402
from tsg_repro.mapping import (  # noqa: E402
    fit_teacher_canonical_maps,
    identity_distance,
    orthogonality_error,
    paired_alignment_error,
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--features", required=True, help="NPZ with none, clean_kd, teacher arrays")
    p.add_argument("--metadata", required=True, help="JSON asserting fit-only scope")
    p.add_argument("--out", required=True)
    p.add_argument("--channels", type=int, default=256)
    return p.parse_args()


def main():
    a = parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=False)

    meta = json.loads(Path(a.metadata).read_text(encoding="utf-8"))
    require_train_only_metadata(meta)

    z = np.load(a.features, allow_pickle=False)
    required = {"none", "clean_kd", "teacher"}
    missing = required - set(z.files)
    if missing:
        raise SystemExit(f"missing arrays in NPZ: {sorted(missing)}")
    n = z["none"]
    s = z["clean_kd"]
    t = z["teacher"]
    if n.ndim != 2 or n.shape[1] != a.channels:
        raise SystemExit(f"expected [N,{a.channels}] feature arrays, got {n.shape}")

    maps = fit_teacher_canonical_maps(n, s, t)
    r_nt_path = out / "R_NT_NONE_TO_TEACHER.npy"
    r_st_path = out / "R_ST_CLEAN_KD_TO_TEACHER.npy"
    np.save(r_nt_path, maps.r_nt.astype(np.float64))
    np.save(r_st_path, maps.r_st.astype(np.float64))

    audit = {
        "status": "TSG_TEACHER_CANONICAL_COORDINATE_AUDIT_COMPLETE",
        "convention": {
            "feature_vectors": "row vectors",
            "none_to_teacher": "N_T = N @ R_NT",
            "clean_kd_to_teacher": "S_T = S @ R_ST",
            "teacher_to_clean_kd": "T_S = T @ R_ST.T",
        },
        "scope": {
            "split": meta.get("split"),
            "val_used": False,
            "pres_lost_labels_used": False,
        },
        "shape": list(maps.r_nt.shape),
        "rows": int(n.shape[0]),
        "metrics": {
            "r_nt_orthogonality_error": orthogonality_error(maps.r_nt),
            "r_st_orthogonality_error": orthogonality_error(maps.r_st),
            "r_nt_identity_distance": identity_distance(maps.r_nt),
            "r_st_identity_distance": identity_distance(maps.r_st),
            "none_to_teacher_relative_fit_error": paired_alignment_error(n, t, maps.r_nt),
            "clean_kd_to_teacher_relative_fit_error": paired_alignment_error(s, t, maps.r_st),
        },
        "artifacts": {
            "R_NT_NONE_TO_TEACHER.npy": sha256_file(r_nt_path),
            "R_ST_CLEAN_KD_TO_TEACHER.npy": sha256_file(r_st_path),
            "input_features_sha256": sha256_file(a.features),
            "input_metadata_sha256": sha256_file(a.metadata),
        },
    }
    audit_path = out / "TEACHER_CANONICAL_COORDINATE_AUDIT.json"
    audit_path.write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
