"""Reusable guards for house-price model inputs, predictions, and metrics."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score

FEATURE_COLUMNS = ["Bedrooms", "Area", "Location", "Age"]
TARGET_COLUMN = "Price"
REQUIRED_COLUMNS = FEATURE_COLUMNS + [TARGET_COLUMN]


def validate_dataset_frame(data: pd.DataFrame, split_name: str = "dataset") -> None:
    """Raise a clear error for missing columns or null required values."""
    missing = sorted(set(REQUIRED_COLUMNS) - set(data.columns))
    if missing:
        raise ValueError(f"{split_name} dataset is missing columns: {missing}")

    null_columns = data[REQUIRED_COLUMNS].columns[
        data[REQUIRED_COLUMNS].isna().any()
    ].tolist()
    if null_columns:
        raise ValueError(
            f"{split_name} dataset contains null required values in: {null_columns}"
        )

    for column in ["Bedrooms", "Area", "Age", "Price"]:
        try:
            values = pd.to_numeric(data[column], errors="raise").to_numpy(dtype=float)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{split_name} column {column!r} must be numeric") from exc
        if not np.isfinite(values).all():
            raise ValueError(
                f"{split_name} column {column!r} contains non-finite values"
            )

    if not data["Location"].map(lambda value: isinstance(value, str) and bool(value.strip())).all():
        raise ValueError(f"{split_name} column 'Location' must contain non-empty strings")


def validate_predictions(predictions, expected_rows: int) -> np.ndarray:
    """Require exactly one finite numeric prediction per input row."""
    try:
        values = np.asarray(predictions, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError("Model predictions must be numeric") from exc

    if values.shape != (expected_rows,):
        raise ValueError(
            f"Prediction shape mismatch: expected ({expected_rows},), got {values.shape}"
        )
    if not np.isfinite(values).all():
        raise ValueError("Model predictions contain non-finite values (NaN or infinity)")
    return values


def validate_quality_metrics(
    metrics: dict, min_r2: float = 0.90, max_mae: float = 1_000_000
) -> None:
    """Fail closed when metrics are missing, non-finite, or outside quality gates."""
    for name in ("mae", "r2"):
        if name not in metrics:
            raise ValueError(f"Required model metric {name!r} is missing")
        try:
            value = float(metrics[name])
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Model metric {name!r} must be numeric") from exc
        if not np.isfinite(value):
            raise ValueError(f"Model metric {name!r} is non-finite")

    if float(metrics["r2"]) < min_r2:
        raise RuntimeError(
            f"Model quality check failed: R²={float(metrics['r2']):.4f}, required >= {min_r2}"
        )
    if float(metrics["mae"]) > max_mae:
        raise RuntimeError(
            f"Model quality check failed: MAE=₹{float(metrics['mae']):,.2f}, required <= ₹{max_mae:,.0f}"
        )


def evaluate_predictions(model, X_eval, y_eval) -> dict:
    """Predict, validate output shape/values, then calculate regression metrics."""
    predictions = validate_predictions(model.predict(X_eval), expected_rows=len(X_eval))
    if len(y_eval) != len(X_eval):
        raise ValueError(
            f"Target length mismatch: expected {len(X_eval)} values, got {len(y_eval)}"
        )
    return {
        "mae": float(mean_absolute_error(y_eval, predictions)),
        "r2": float(r2_score(y_eval, predictions)),
    }
