"""Utilities for validating versioned dataset files against their manifest."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = ROOT / "metadata" / "dataset_manifest.json"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_manifest() -> dict:
    if not MANIFEST_PATH.is_file():
        raise FileNotFoundError(
            f"Dataset manifest not found: {MANIFEST_PATH}. "
            "Run src/create_datasets.py first."
        )
    return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))


def load_verified_split(split_name: str) -> tuple[pd.DataFrame, dict]:
    """Load a split only if its file hash and row count match the manifest."""
    manifest = load_manifest()
    if split_name not in manifest.get("splits", {}):
        raise KeyError(
            f"Split {split_name!r} is not present in {MANIFEST_PATH}."
        )

    details = manifest["splits"][split_name]
    path = ROOT / details["path"]

    if not path.is_file():
        raise FileNotFoundError(f"Dataset split not found: {path}")

    actual_hash = sha256_file(path)
    expected_hash = details["sha256"]
    if actual_hash != expected_hash:
        raise ValueError(
            f"SHA-256 mismatch for {split_name} dataset {path}.\n"
            f"Expected: {expected_hash}\n"
            f"Actual:   {actual_hash}\n"
            "Do not train/evaluate against a modified dataset. "
            "Regenerate intentionally or restore the expected file."
        )

    frame = pd.read_csv(path)
    if len(frame) != int(details["row_count"]):
        raise ValueError(
            f"Row-count mismatch for {split_name}: manifest says "
            f"{details['row_count']}, file has {len(frame)}."
        )
    if list(frame.columns) != details["columns"]:
        raise ValueError(
            f"Column mismatch for {split_name}: manifest says "
            f"{details['columns']}, file has {list(frame.columns)}."
        )

    return frame, manifest
