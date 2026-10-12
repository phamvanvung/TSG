"""Deterministic BDD100K Gaussian view generation and manifest auditing."""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from PIL import Image

TRAIN_VAL_GENERATOR_VERSION = "consrestore_exp1_training_gaussian_v1"
DEVTEST_GENERATOR_VERSION = "consrestore_gaussian_devtest_v1"
MASTER_SEED = 2027
SIGMAS = (15, 25, 50)

REQUIRED_MANIFEST_COLUMNS = {
    "consrestore_split", "source_stem", "sigma", "seed_uint64",
    "clean_image", "degraded_image", "label_file",
}


def gaussian_view_seed(
    split: str,
    source_stem: str,
    sigma: int,
    *,
    generator_version: str = TRAIN_VAL_GENERATOR_VERSION,
    master_seed: int = MASTER_SEED,
) -> int:
    token = (
        f"{generator_version}|master_seed={master_seed}|"
        f"split={split}|source={source_stem}|sigma={int(sigma)}"
    )
    digest = hashlib.sha256(token.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], byteorder="big", signed=False)


def generate_gaussian(clean_rgb: np.ndarray, sigma: int, seed: int) -> np.ndarray:
    clean_rgb = np.asarray(clean_rgb)
    if clean_rgb.dtype != np.uint8:
        raise ValueError(f"Expected uint8 clean image, found {clean_rgb.dtype}")
    rng = np.random.default_rng(int(seed))
    noise = rng.normal(0.0, float(sigma), size=clean_rgb.shape).astype(np.float32)
    degraded = clean_rgb.astype(np.float32) + noise
    return np.clip(np.rint(degraded), 0.0, 255.0).astype(np.uint8)


def write_png_rgb(array: np.ndarray, destination: str | Path) -> None:
    p = Path(destination)
    p.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.asarray(array, dtype=np.uint8), mode="RGB").save(
        p, format="PNG", optimize=False
    )


def load_manifest(path: str | Path) -> pd.DataFrame:
    p = Path(path)
    df = pd.read_csv(p)
    missing = REQUIRED_MANIFEST_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"Manifest missing columns: {sorted(missing)}")
    df = df.copy()
    df["consrestore_split"] = df["consrestore_split"].astype(str).str.upper()
    df["source_stem"] = df["source_stem"].astype(str)
    df["sigma"] = pd.to_numeric(df["sigma"], errors="raise").astype(int)
    if not set(df["sigma"]).issubset(set(SIGMAS)):
        raise ValueError(f"Unexpected sigma values: {sorted(set(df['sigma']))}")
    return df


def validate_manifest_contract(
    df: pd.DataFrame,
    *,
    expected_train_sources: int = 1500,
    expected_val_sources: int = 250,
    expected_train_views: int = 4500,
    expected_val_views: int = 750,
) -> dict:
    if df[["consrestore_split", "source_stem", "sigma"]].duplicated().any():
        raise ValueError("Duplicate (split, source_stem, sigma) rows")
    if set(df["consrestore_split"]) != {"TRAIN", "VAL"}:
        raise ValueError("TRAIN/VAL manifest must contain exactly TRAIN and VAL")
    tr = df[df["consrestore_split"] == "TRAIN"]
    va = df[df["consrestore_split"] == "VAL"]
    actual = {
        "train_sources": int(tr["source_stem"].nunique()),
        "val_sources": int(va["source_stem"].nunique()),
        "train_views": int(len(tr)),
        "val_views": int(len(va)),
    }
    expected = {
        "train_sources": expected_train_sources,
        "val_sources": expected_val_sources,
        "train_views": expected_train_views,
        "val_views": expected_val_views,
    }
    if actual != expected:
        raise ValueError(f"Manifest counts differ: actual={actual}, expected={expected}")
    for split, part in (("train", tr), ("val", va)):
        counts = part.groupby("source_stem")["sigma"].agg(lambda x: tuple(sorted(x)))
        if not all(v == SIGMAS for v in counts):
            raise ValueError(f"Every {split} source must have sigma {SIGMAS}")
    return actual


def audit_pixel_regeneration(
    df: pd.DataFrame,
    *,
    samples_per_split_sigma: int = 10,
) -> list[dict]:
    rows: list[dict] = []
    for split_upper, split_lower in (("TRAIN", "train"), ("VAL", "val")):
        for sigma in SIGMAS:
            sub = df[(df.consrestore_split == split_upper) & (df.sigma == sigma)].copy()
            def rank(stem: str) -> str:
                token = f"consrestore_exp1_step09_pixel_audit_v1|{split_lower}|sigma{sigma}|{stem}"
                return hashlib.sha256(token.encode()).hexdigest()
            sub["_rank"] = sub["source_stem"].map(rank)
            chosen = sub.sort_values(["_rank", "source_stem"]).head(samples_per_split_sigma)
            if len(chosen) != samples_per_split_sigma:
                raise ValueError("Insufficient rows for fixed pixel audit")
            for rec in chosen.itertuples(index=False):
                expected_seed = gaussian_view_seed(split_lower, rec.source_stem, sigma)
                if int(rec.seed_uint64) != expected_seed:
                    raise ValueError(f"Seed mismatch: {rec.source_stem} sigma{sigma}")
                clean = np.asarray(Image.open(rec.clean_image).convert("RGB"), dtype=np.uint8)
                stored = np.asarray(Image.open(rec.degraded_image).convert("RGB"), dtype=np.uint8)
                regenerated = generate_gaussian(clean, sigma, expected_seed)
                if not np.array_equal(stored, regenerated):
                    raise ValueError(f"Pixel regeneration mismatch: {rec.source_stem} sigma{sigma}")
                rows.append({"split": split_upper, "sigma": sigma,
                             "source_stem": rec.source_stem, "seed_uint64": expected_seed,
                             "result": "PASS"})
    return rows
