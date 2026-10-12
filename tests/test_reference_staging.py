from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_portable_python_derivative_preserves_scientific_callable(tmp_path):
    root = Path(__file__).resolve().parents[1]
    stage = _load(root / "scripts/stage_public_reference_artifacts.py", "stage_refs_test")

    src = tmp_path / "historical.py"
    private = "/home/" + "mdptlab/"
    src.write_text(
        'from pathlib import Path\n'
        f'PROJECT = Path("{private}Documents/Researching/ConsRestore")\n'
        'def make_random_control_delta(row_id, teacher, complement):\n'
        '    """scientific callable"""\n'
        '    return row_id, teacher, complement, 1.0\n',
        encoding="utf-8",
    )
    expected = hashlib.sha256(src.read_bytes()).hexdigest()
    dst = tmp_path / "public.py"
    old_root = stage.ROOT
    try:
        stage.ROOT = tmp_path
        info = stage.copy_portable_python_derivative(
            src, dst, expected, "make_random_control_delta"
        )
    finally:
        stage.ROOT = old_root

    assert info["portable_derivative"] is True
    assert info["historical_source_sha256"] == expected
    assert ("/home/" + "mdptlab/") not in dst.read_text(encoding="utf-8")
    assert "/AUTHOR_HOME/" in dst.read_text(encoding="utf-8")


def test_random_control_verifier_accepts_staged_portable_derivative(tmp_path):
    root = Path(__file__).resolve().parents[1]
    mod = _load(
        root / "scripts/yolo/06b_reproduce_canonical_random_control.py",
        "random_control_verify_test",
    )

    refs = tmp_path / "reference_artifacts"
    helper = refs / "yolo/07_random_control.py"
    helper.parent.mkdir(parents=True)
    helper.write_text(
        'def make_random_control_delta(row_id, teacher, complement):\n'
        '    return row_id, teacher, complement, 1.0\n',
        encoding="utf-8",
    )
    helper_sha = hashlib.sha256(helper.read_bytes()).hexdigest()
    ast_sha = mod._callable_ast_sha(helper, "make_random_control_delta")
    report = {
        "items": {
            "exp4_random_helper": {
                "portable_derivative": True,
                "historical_source_sha256": mod.EXPECTED_STEP07_SHA256,
                "sha256": helper_sha,
                "scientific_callable": "make_random_control_delta",
                "scientific_callable_ast_sha256": ast_sha,
            }
        }
    }
    (refs / "STAGED_REFERENCE_ARTIFACTS.json").write_text(
        json.dumps(report), encoding="utf-8"
    )
    result = mod.verify_step07_helper(helper)
    assert result["mode"] == "PORTABLE_PATH_SANITIZED_DERIVATIVE"


def test_portable_text_derivative_changes_only_private_path_prefixes(tmp_path):
    root = Path(__file__).resolve().parents[1]
    stage = _load(root / "scripts/stage_public_reference_artifacts.py", "stage_refs_yaml_test")

    src = tmp_path / "NONE_args.yaml"
    private = "/home/" + "mdptlab/"
    original = (
        "epochs: 50\n"
        "seed: 2027\n"
        "box: 7.5\n"
        "cls: 0.5\n"
        "dfl: 1.5\n"
        f"project: {private}Documents/Researching/ConsRestore/experiment1/runs\n"
        f"data: {private}Documents/Researching/datasets/bdd/data.yaml\n"
        "name: none_50e\n"
    )
    src.write_text(original, encoding="utf-8")
    dst = tmp_path / "public" / "NONE_args.yaml"

    old_root = stage.ROOT
    try:
        stage.ROOT = tmp_path
        info = stage.copy_portable_text_derivative(src, dst, None)
    finally:
        stage.ROOT = old_root

    public = dst.read_text(encoding="utf-8")
    expected = original.replace(private, "/AUTHOR_HOME/")
    assert public == expected
    assert info["portable_derivative"] is True
    assert info["sanitization"]["non_path_bytes_changed"] is False
    assert info["sanitization"]["policy"] == (
        "exact_known_private_path_prefix_replacement_only"
    )
    assert private not in public
    assert "epochs: 50" in public
    assert "seed: 2027" in public
    assert "box: 7.5" in public
    assert "cls: 0.5" in public
    assert "dfl: 1.5" in public
