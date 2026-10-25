#!/usr/bin/env python3
"""Verify source and processed CSV hashes against the dataset manifest."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "metadata/dataset_manifest.json"


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def check(label, relative_path, expected):
    path = ROOT / relative_path
    if not path.is_file():
        print(f"FAIL: {label}: missing file {relative_path}")
        return False
    actual = sha256_file(path)
    if actual != expected:
        print(f"FAIL: {label}: hash mismatch")
        print(f"  expected: {expected}")
        print(f"  actual:   {actual}")
        return False
    print(f"PASS: {label}: hash matches")
    return True


def main():
    if not MANIFEST.is_file():
        raise FileNotFoundError("Manifest missing; run python src/create_datasets.py first.")
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    results = []
    source = manifest["source_dataset"]
    results.append(check("source dataset", source["path"], source["sha256"]))
    for name, details in manifest["splits"].items():
        results.append(check(name, details["path"], details["sha256"]))
    if not all(results):
        raise SystemExit("Dataset verification failed.")
    print("All dataset hashes verified successfully.")


if __name__ == "__main__":
    main()
