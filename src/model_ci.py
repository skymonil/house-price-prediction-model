# src/model_ci.py
import json
import os
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline

from src.dataset_utils import load_verified_split
from src.evaluate import evaluate_model
from src.preprocess import create_preprocessor

MIN_R2 = 0.90
MAX_MAE = 1_000_000

MODEL_PATH = Path("models/model_ci.pkl")
METRICS_PATH = Path("models/metrics.json")

MLFLOW_EXPERIMENT = "house-price-prediction-v2"
MODEL_NAME = "HousePricePredictor"
MLFLOW_TRACKING_URI = os.environ.get(
    "MLFLOW_TRACKING_URI",
    "https://mlflow.615915.xyz",
)


def split_features_target(data: pd.DataFrame, split_name: str):
    required_columns = ["Bedrooms", "Area", "Location", "Age", "Price"]
    missing = set(required_columns) - set(data.columns)
    if missing:
        raise ValueError(
            f"{split_name} dataset is missing columns: {sorted(missing)}"
        )
    if data[required_columns].isna().any().any():
        raise ValueError(f"{split_name} dataset contains null required values.")

    X = data[["Bedrooms", "Area", "Location", "Age"]]
    y = pd.to_numeric(data["Price"], errors="raise")
    return X, y


def train_model():
    print("[1/6] Loading verified training and validation datasets...", flush=True)
    train_data, manifest = load_verified_split("train")
    validation_data, _ = load_verified_split("validation")

    X_train, y_train = split_features_target(train_data, "train")
    X_validation, y_validation = split_features_target(
        validation_data, "validation"
    )

    print(
        f"[1/6] Train rows: {len(train_data)}; "
        f"validation rows: {len(validation_data)}",
        flush=True,
    )

    print("[2/6] Building model...", flush=True)
    model = Pipeline(
        steps=[
            ("preprocessor", create_preprocessor()),
            (
                "model",
                RandomForestRegressor(
                    n_estimators=200,
                    random_state=42,
                ),
            ),
        ]
    )

    print("[2/6] Training model...", flush=True)
    model.fit(X_train, y_train)
    print("[2/6] Model training completed.", flush=True)

    # Validation data is used for candidate quality gates/model selection.
    metrics = evaluate_model(model, X_validation, y_validation)
    print("[3/6] Validation evaluation completed.", flush=True)

    dataset_metadata = {
        "dataset_version": str(manifest.get("dataset_version", "unknown")),
        "source_sha256": manifest["source_dataset"]["sha256"],
        "train_sha256": manifest["splits"]["train"]["sha256"],
        "validation_sha256": manifest["splits"]["validation"]["sha256"],
        "train_rows": len(train_data),
        "validation_rows": len(validation_data),
        "git_commit": os.environ.get("GITHUB_SHA", "local"),
    }
    return model, metrics, dataset_metadata


def validate_model(metrics):
    print("\nValidation Evaluation", flush=True)
    print("---------------------", flush=True)
    print(f"MAE: ₹{metrics['mae']:,.2f}", flush=True)
    print(f"R²:  {metrics['r2']:.4f}", flush=True)
    print("\nQuality Gate", flush=True)
    print("------------", flush=True)
    print(f"Required R²  >= {MIN_R2}", flush=True)
    print(f"Required MAE <= ₹{MAX_MAE:,.0f}", flush=True)

    if metrics["r2"] < MIN_R2:
        raise RuntimeError(
            f"Model quality check failed: R²={metrics['r2']:.4f}, "
            f"required >= {MIN_R2}"
        )
    if metrics["mae"] > MAX_MAE:
        raise RuntimeError(
            f"Model quality check failed: MAE=₹{metrics['mae']:,.2f}, "
            f"required <= ₹{MAX_MAE:,.0f}"
        )
    print("\nModel quality check PASSED.", flush=True)


def save_artifacts(model, metrics):
    print("\n[4/6] Saving local CI artifacts...", flush=True)
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    METRICS_PATH.write_text(json.dumps(metrics, indent=4), encoding="utf-8")
    print(f"[4/6] Model saved to {MODEL_PATH}", flush=True)
    print(f"[4/6] Metrics saved to {METRICS_PATH}", flush=True)


def log_to_mlflow(model, metrics, dataset_metadata):
    print("\n[5/6] Connecting to MLflow...", flush=True)
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT)

    with mlflow.start_run() as run:
        print(f"MLflow Run ID: {run.info.run_id}", flush=True)
        params = {
            "model_type": "RandomForestRegressor",
            "n_estimators": 200,
            "random_state": 42,
            "training_dataset": "data/processed/train.csv",
            "evaluation_dataset": "data/processed/validation.csv",
            **dataset_metadata,
        }
        mlflow.log_params(params)
        mlflow.log_metrics({
            "validation_mae": metrics["mae"],
            "validation_r2": metrics["r2"],
        })
        mlflow.sklearn.log_model(
            sk_model=model,
            name="house-price-model",
            skops_trusted_types=["sklearn.tree._tree.Tree"],
        )
        return run.info.run_id


def register_model(run_id):
    print("\n[6/6] Registering model...", flush=True)
    model_uri = f"runs:/{run_id}/house-price-model"
    registered_model = mlflow.register_model(
        model_uri=model_uri,
        name=MODEL_NAME,
    )

    candidate_version_path = Path("models/candidate_version.txt")
    candidate_version_path.parent.mkdir(parents=True, exist_ok=True)
    candidate_version_path.write_text(
        f"{registered_model.version}\n",
        encoding="utf-8",
    )

    print("Model registered successfully.", flush=True)
    print(f"Model name:    {registered_model.name}", flush=True)
    print(f"Model version: {registered_model.version}", flush=True)
    print(f"Candidate file: {candidate_version_path}", flush=True)
    return registered_model


if __name__ == "__main__":
    print("==========================================", flush=True)
    print("       HOUSE PRICE MODEL CI", flush=True)
    print("==========================================", flush=True)

    model, metrics, dataset_metadata = train_model()
    validate_model(metrics)
    save_artifacts(model, metrics)

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    run_id = log_to_mlflow(model, metrics, dataset_metadata)
    register_model(run_id)

    print("\n==========================================", flush=True)
    print("       MODEL CI COMPLETED SUCCESSFULLY", flush=True)
    print("==========================================", flush=True)
