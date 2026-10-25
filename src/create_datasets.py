#!/usr/bin/env python3
"""Create reproducible train/validation/test CSVs and a dataset manifest."""
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data/raw/houses.csv"
OUT = ROOT / "data/processed"
MANIFEST = ROOT / "metadata/dataset_manifest.json"

DATASET_VERSION = "v1"
RANDOM_STATE = 42
REQUIRED_COLUMNS = {"Bedrooms", "Area", "Location", "Age", "Price"}


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    if not SOURCE.is_file():
        raise FileNotFoundError(f"Put your existing source CSV at {SOURCE}")

    data = pd.read_csv(SOURCE)
    missing = REQUIRED_COLUMNS - set(data.columns)
    if missing:
        raise ValueError(
            f"Missing required columns: {sorted(missing)}. "
            f"Found: {list(data.columns)}"
        )
    if len(data) < 10:
        raise ValueError(f"Only {len(data)} rows; too few for this 3-way split.")
    if data[list(REQUIRED_COLUMNS)].isnull().any().any():
        raise ValueError("Required columns contain nulls; handle them deliberately first.")

    # 70% train; reserve 30%, then split that evenly into 15% validation and 15% test.
    train_df, holdout_df = train_test_split(
        data, test_size=0.30, random_state=RANDOM_STATE, shuffle=True
    )
    validation_df, test_df = train_test_split(
        holdout_df, test_size=0.50, random_state=RANDOM_STATE, shuffle=True
    )

    OUT.mkdir(parents=True, exist_ok=True)
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    splits = {}
    for name, frame in {
        "train": train_df,
        "validation": validation_df,
        "test": test_df,
    }.items():
        path = OUT / f"{name}.csv"
        frame.to_csv(path, index=False, lineterminator="\n")
        splits[name] = {
            "path": path.relative_to(ROOT).as_posix(),
            "sha256": sha256_file(path),
            "row_count": int(len(frame)),
            "columns": list(frame.columns),
        }

    manifest = {
        "dataset_version": DATASET_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_dataset": {
            "path": SOURCE.relative_to(ROOT).as_posix(),
            "sha256": sha256_file(SOURCE),
            "row_count": int(len(data)),
            "columns": list(data.columns),
        },
        "split_configuration": {
            "method": "random_train_test_split",
            "random_state": RANDOM_STATE,
            "shuffle": True,
            "ratios": {"train": 0.70, "validation": 0.15, "test": 0.15},
        },
        "splits": splits,
    }
    MANIFEST.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    print("Created dataset splits:")
    for name, details in splits.items():
        print(f"  {details['path']}: {details['row_count']} rows")
    print(f"Manifest: {MANIFEST.relative_to(ROOT).as_posix()}")
    print("Warning: rerunning overwrites the current splits and manifest.")


if __name__ == "__main__":
    main()
