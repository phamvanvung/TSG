#!/usr/bin/env python
"""Stage small author-generated replay artifacts for the public stable release.

This intentionally excludes BDD100K data and trained detector checkpoints.

Historical Python helpers may contain author-machine absolute path literals.  A
public derivative is allowed only when:
  1. the historical source bytes match their frozen SHA-256;
  2. sanitization changes only known private path tokens; and
  3. the scientific callable used by the public replay has an identical AST
     (ignoring its docstring and source-location metadata) before/after.
"""
from __future__ import annotations

import argparse
import ast
import copy
import hashlib
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from tsg_repro.config import get_path, load_config
from tsg_repro.hashing import sha256_file

EXPECTED = {
    "yolo_R_NT": "2caa1712e6493ca25f5ceb30dfe4d831eb4485e76c51bbf5372571fc0e0e330f",
    "yolo_R_ST": "cebc13623188e0ee6b83410f318a53798bfc8264b52f55237b4558bd76b55cd8",
    "yolo_C_g": "e765f903ca3308bb090b6e0985fd14ba7c46151da2cf2c4d7b339faa34a9f64d",
    "exp4_random_helper": "7aa5b6aa43aa35c90f977413dd7056e1c00b07f58b02629c4af2b9a5fc9861f8",
    "yolo_companion": "d82f20215ac37848cea810b8e4e5ee7fb169af24ea8819305088fe29490d121b",
    "rtdetr_companion": "eafb2d02a08977149d24486755a2a4d43a9c0359f3a8a8b840ce5a8882aa471e",
    "rtdetr_step10_bridge": "bf9f8f683f0d92a21d0f5a09127dc03d0547cff622446c39975cd6cca9b4eb60",
}

# Deliberately narrow substitutions.  We do not perform broad regex redaction
# because that could silently alter executable scientific logic.
PRIVATE_PATH_REPLACEMENTS = {
    "/home/" + "mdptlab/": "/AUTHOR_HOME/",
    "C:\\Users\\": "C:\\AUTHOR_HOME\\",
}


def assert_public_text_safe(path: Path) -> None:
    if path.suffix.lower() not in {".py", ".yaml", ".yml", ".json", ".txt", ".md"}:
        return
    text = path.read_text(encoding="utf-8", errors="replace")
    found = [token for token in PRIVATE_PATH_REPLACEMENTS if token in text]
    if found:
        raise RuntimeError(
            f"Staged public text artifact contains a private machine path: {path}; "
            f"tokens={found}. Create a reviewed portable derivative before public release."
        )


def _callable_ast_sha_from_text(text: str, callable_name: str) -> str:
    tree = ast.parse(text)
    matches = [
        node for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == callable_name
    ]
    if len(matches) != 1:
        raise RuntimeError(
            f"Expected exactly one top-level callable {callable_name!r}; found {len(matches)}"
        )
    node = copy.deepcopy(matches[0])
    # A docstring is documentation, not executable scientific logic.  Removing
    # it from the fingerprint also allows a private path in a docstring to be
    # sanitized without changing the scientific-callable fingerprint.
    if (
        node.body
        and isinstance(node.body[0], ast.Expr)
        and isinstance(node.body[0].value, ast.Constant)
        and isinstance(node.body[0].value.value, str)
    ):
        node.body = node.body[1:]
    dumped = ast.dump(node, annotate_fields=True, include_attributes=False)
    return hashlib.sha256(dumped.encode("utf-8")).hexdigest()


def copy_checked(src: Path, dst: Path, expected: str | None) -> dict:
    src = src.resolve()
    if not src.is_file():
        raise FileNotFoundError(src)
    actual = sha256_file(src)
    if expected is not None and actual != expected:
        raise RuntimeError(
            f"SHA256 mismatch for {src}\nExpected: {expected}\nActual  : {actual}"
        )
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    copied = sha256_file(dst)
    if copied != actual:
        raise RuntimeError(f"Copy integrity failure: {dst}")
    assert_public_text_safe(dst)
    return {
        "source_name": src.name,
        "public_path": str(dst.relative_to(ROOT)),
        "sha256": copied,
        "portable_derivative": False,
    }


def copy_portable_python_derivative(
    src: Path,
    dst: Path,
    expected_source_sha256: str,
    scientific_callable: str,
) -> dict:
    """Create a path-sanitized derivative without changing scientific callable logic."""
    src = src.resolve()
    if not src.is_file():
        raise FileNotFoundError(src)

    historical_sha = sha256_file(src)
    if historical_sha != expected_source_sha256:
        raise RuntimeError(
            f"Historical helper SHA256 mismatch for {src}\n"
            f"Expected: {expected_source_sha256}\nActual  : {historical_sha}"
        )

    original = src.read_text(encoding="utf-8")
    before_ast_sha = _callable_ast_sha_from_text(original, scientific_callable)

    portable = original
    replacements_applied = []
    for private, replacement in PRIVATE_PATH_REPLACEMENTS.items():
        count = portable.count(private)
        if count:
            portable = portable.replace(private, replacement)
            replacements_applied.append({
                "kind": (
                    "unix_author_home_prefix"
                    if private.startswith("/home/")
                    else "windows_user_home_prefix"
                ),
                "replacement": replacement,
                "count": int(count),
            })

    if not replacements_applied:
        # Nothing private was present; retain exact bytes and exact provenance.
        return copy_checked(src, dst, expected_source_sha256)

    # Parse/compile the public derivative before writing it.
    ast.parse(portable)
    compile(portable, str(dst), "exec")
    after_ast_sha = _callable_ast_sha_from_text(portable, scientific_callable)
    if after_ast_sha != before_ast_sha:
        raise RuntimeError(
            f"Portable sanitization changed scientific callable {scientific_callable!r}. "
            "Refusing to stage derivative."
        )

    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(portable, encoding="utf-8")
    assert_public_text_safe(dst)
    public_sha = sha256_file(dst)

    return {
        "source_name": src.name,
        "public_path": str(dst.relative_to(ROOT)),
        "sha256": public_sha,
        "portable_derivative": True,
        "historical_source_sha256": historical_sha,
        "scientific_callable": scientific_callable,
        "scientific_callable_ast_sha256": after_ast_sha,
        "sanitization": {
            "policy": "known_private_path_literals_only",
            "replacements": replacements_applied,
        },
    }


def copy_portable_text_derivative(
    src: Path,
    dst: Path,
    expected_source_sha256: str | None = None,
) -> dict:
    """Create a text derivative by replacing only known private path prefixes.

    This is intended for non-executable provenance/configuration text such as
    args.yaml.  The output bytes are derived from the historical source by
    exact literal substitution only; no parser round-trip or reformatting is
    allowed, so every non-path byte remains unchanged.
    """
    src = src.resolve()
    if not src.is_file():
        raise FileNotFoundError(src)

    historical_sha = sha256_file(src)
    if expected_source_sha256 is not None and historical_sha != expected_source_sha256:
        raise RuntimeError(
            f"Historical text artifact SHA256 mismatch for {src}\n"
            f"Expected: {expected_source_sha256}\nActual  : {historical_sha}"
        )

    original = src.read_text(encoding="utf-8")
    portable = original
    replacements_applied = []
    for private, replacement in PRIVATE_PATH_REPLACEMENTS.items():
        count = portable.count(private)
        if count:
            portable = portable.replace(private, replacement)
            replacements_applied.append({
                "kind": (
                    "unix_author_home_prefix"
                    if private.startswith("/home/")
                    else "windows_user_home_prefix"
                ),
                "replacement": replacement,
                "count": int(count),
            })

    if not replacements_applied:
        return copy_checked(src, dst, expected_source_sha256)

    # Strong byte-level guardrail: the derivative must be exactly what is
    # obtained by the allow-listed literal replacements and nothing else.
    reconstructed = original
    for private, replacement in PRIVATE_PATH_REPLACEMENTS.items():
        reconstructed = reconstructed.replace(private, replacement)
    if portable != reconstructed:
        raise RuntimeError(
            f"Portable text sanitization made a non-allow-listed change: {src}"
        )

    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(portable, encoding="utf-8")
    assert_public_text_safe(dst)
    public_sha = sha256_file(dst)

    return {
        "source_name": src.name,
        "public_path": str(dst.relative_to(ROOT)),
        "sha256": public_sha,
        "portable_derivative": True,
        "historical_source_sha256": historical_sha,
        "sanitization": {
            "policy": "exact_known_private_path_prefix_replacement_only",
            "non_path_bytes_changed": False,
            "replacements": replacements_applied,
        },
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--destination", default=str(ROOT / "reference_artifacts"))
    a = p.parse_args()
    cfg, base = load_config(a.config)
    dest = Path(a.destination).expanduser().resolve()
    if dest != (ROOT / "reference_artifacts").resolve():
        raise RuntimeError("For release safety, destination must be repository/reference_artifacts")

    maps = get_path(cfg, base, "artifacts", "canonical_map_dir")
    items = {}
    items["yolo_R_NT"] = copy_checked(
        maps / "R_NT_NONE_TO_TEACHER.npy",
        dest / "yolo/R_NT_NONE_TO_TEACHER.npy",
        EXPECTED["yolo_R_NT"],
    )
    items["yolo_R_ST"] = copy_checked(
        maps / "R_ST_CLEAN_KD_TO_TEACHER.npy",
        dest / "yolo/R_ST_CLEAN_KD_TO_TEACHER.npy",
        EXPECTED["yolo_R_ST"],
    )
    items["yolo_C_g"] = copy_checked(
        get_path(cfg, base, "artifacts", "yolo_task_geometry"),
        dest / "yolo/C_g.npy",
        EXPECTED["yolo_C_g"],
    )
    items["exp4_random_helper"] = copy_portable_python_derivative(
        get_path(cfg, base, "artifacts", "exp4_step07_helper"),
        dest / "yolo/07_random_control.py",
        EXPECTED["exp4_random_helper"],
        "make_random_control_delta",
    )
    items["yolo_companion"] = copy_checked(
        get_path(cfg, base, "checkpoints", "yolo_checkpoint_companion"),
        dest / "companions/consrestore_cacd_native.py",
        EXPECTED["yolo_companion"],
    )
    items["rtdetr_companion"] = copy_checked(
        get_path(cfg, base, "checkpoints", "rtdetr_checkpoint_companion"),
        dest / "companions/exp17_rtdetr_clean_kd.py",
        EXPECTED["rtdetr_companion"],
    )
    items["rtdetr_step10_bridge"] = copy_checked(
        get_path(cfg, base, "artifacts", "rtdetr_step10_bridge"),
        dest / "rtdetr/RTDETR_NONE_TO_CLEAN_KD_P4_ORTHOGONAL_BRIDGE.npy",
        EXPECTED["rtdetr_step10_bridge"],
    )
    # args.yaml has no independent historical hash in the public lock.
    # Record its source SHA at staging time and create a public derivative
    # only by exact replacement of known private home-path prefixes.  No
    # parser round-trip is used, so all non-path bytes/hyperparameters remain
    # identical to the source artifact.
    items["yolo_none_args_yaml"] = copy_portable_text_derivative(
        get_path(cfg, base, "data", "yolo_none_args_yaml"),
        dest / "yolo/NONE_args.yaml",
        None,
    )

    report = {
        "status": "PUBLIC_REFERENCE_ARTIFACTS_STAGED",
        "excluded": ["BDD100K data", "detector checkpoints"],
        "items": items,
    }
    report_path = dest / "STAGED_REFERENCE_ARTIFACTS.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    assert_public_text_safe(report_path)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
