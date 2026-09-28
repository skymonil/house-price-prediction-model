import json
import os
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from src.evaluate import evaluate_model
from src.preprocess import create_preprocessor


# ============================================================
# Model Quality Thresholds
# ============================================================

MIN_R2 = 0.90
MAX_MAE = 1_000_000


# ============================================================
# Local CI Artifacts
# ============================================================

MODEL_PATH = Path("models/model_ci.pkl")
METRICS_PATH = Path("models/metrics.json")


# ============================================================
# MLflow Configuration
# ============================================================

MLFLOW_EXPERIMENT = "house-price-prediction-v2"
MODEL_NAME = "HousePricePredictor"

MLFLOW_TRACKING_URI = os.environ.get(
    "MLFLOW_TRACKING_URI",
    "https://mlflow.615915.xyz",
)


# ============================================================
# Train Model
# ============================================================

def train_model():
    print("[1/6] Loading dataset...", flush=True)

    data = pd.read_csv("data/houses.csv")

    X = data[["Bedrooms", "Area", "Location", "Age"]]
    y = data["Price"]

    print(f"[1/6] Dataset loaded: {len(data)} rows", flush=True)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
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

    metrics = evaluate_model(
        model,
        X_test,
        y_test,
    )

    print("[3/6] Model evaluation completed.", flush=True)

    return model, metrics


# ============================================================
# Model Quality Gate
# ============================================================

def validate_model(metrics):
    print("\nModel Evaluation", flush=True)
    print("----------------", flush=True)
    print(
        f"MAE: ₹{metrics['mae']:,.2f}",
        flush=True,
    )
    print(
        f"R²:  {metrics['r2']:.4f}",
        flush=True,
    )

    print("\nQuality Gate", flush=True)
    print("------------", flush=True)
    print(
        f"Required R²  >= {MIN_R2}",
        flush=True,
    )
    print(
        f"Required MAE <= ₹{MAX_MAE:,.0f}",
        flush=True,
    )

    if metrics["r2"] < MIN_R2:
        raise RuntimeError(
            "Model quality check failed: "
            f"R²={metrics['r2']:.4f}, "
            f"required >= {MIN_R2}"
        )

    if metrics["mae"] > MAX_MAE:
        raise RuntimeError(
            "Model quality check failed: "
            f"MAE=₹{metrics['mae']:,.2f}, "
            f"required <= ₹{MAX_MAE:,.0f}"
        )

    print(
        "\nModel quality check PASSED.",
        flush=True,
    )


# ============================================================
# Save Local CI Artifacts
# ============================================================

def save_artifacts(model, metrics):
    print(
        "\n[4/6] Saving local CI artifacts...",
        flush=True,
    )

    MODEL_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    joblib.dump(
        model,
        MODEL_PATH,
    )

    with open(
        METRICS_PATH,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            metrics,
            file,
            indent=4,
        )

    print(
        f"[4/6] Model saved to {MODEL_PATH}",
        flush=True,
    )
    print(
        f"[4/6] Metrics saved to {METRICS_PATH}",
        flush=True,
    )


# ============================================================
# Log Model to MLflow
# ============================================================

def log_to_mlflow(model, metrics):
    print(
        "\n[5/6] Connecting to MLflow...",
        flush=True,
    )

    print(
        f"MLflow URI: {MLFLOW_TRACKING_URI}",
        flush=True,
    )

    mlflow.set_tracking_uri(
        MLFLOW_TRACKING_URI
    )

    print(
        f"MLflow experiment: {MLFLOW_EXPERIMENT}",
        flush=True,
    )

    mlflow.set_experiment(
        MLFLOW_EXPERIMENT
    )

    print(
        "Starting MLflow run...",
        flush=True,
    )

    with mlflow.start_run() as run:

        print(
            f"MLflow Run ID: {run.info.run_id}",
            flush=True,
        )

        mlflow.log_params(
            {
                "model_type": "RandomForestRegressor",
                "n_estimators": 200,
                "random_state": 42,
                "test_size": 0.2,
            }
        )

        print(
            "Parameters logged.",
            flush=True,
        )

        mlflow.log_metrics(
            {
                "mae": metrics["mae"],
                "r2": metrics["r2"],
            }
        )

        print(
            "Metrics logged.",
            flush=True,
        )

        print(
            "Logging sklearn model to MLflow...",
            flush=True,
        )

        mlflow.sklearn.log_model(
            sk_model=model,
            name="house-price-model",
            skops_trusted_types=[
                "sklearn.tree._tree.Tree"
            ],
        )

        print(
            "Model successfully logged to MLflow.",
            flush=True,
        )

        return run.info.run_id


# ============================================================
# Register Model
# ============================================================

def register_model(run_id):
    print(
        "\n[6/6] Registering model...",
        flush=True,
    )

    model_uri = (
        f"runs:/{run_id}/house-price-model"
    )

    print(
        f"Model URI: {model_uri}",
        flush=True,
    )

    registered_model = mlflow.register_model(
        model_uri=model_uri,
        name=MODEL_NAME,
    )

    print(
        "\nModel registered successfully.",
        flush=True,
    )
    print(
        f"Model name:    {registered_model.name}",
        flush=True,
    )
    print(
        f"Model version: {registered_model.version}",
        flush=True,
    )

    return registered_model


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":

    print(
        "==========================================",
        flush=True,
    )
    print(
        "       HOUSE PRICE MODEL CI",
        flush=True,
    )
    print(
        "==========================================",
        flush=True,
    )

    # 1. Train
    model, metrics = train_model()

    # 2. Quality gate
    validate_model(metrics)

    # 3. Save local artifacts
    save_artifacts(
        model,
        metrics,
    )

    # 4. Configure MLflow
    mlflow.set_tracking_uri(
        MLFLOW_TRACKING_URI
    )

    # 5. Log model
    run_id = log_to_mlflow(
        model,
        metrics,
    )

    # 6. Register model
    register_model(
        run_id
    )

    print(
        "\n==========================================",
        flush=True,
    )
    print(
        "       MODEL CI COMPLETED SUCCESSFULLY",
        flush=True,
    )
    print(
        "==========================================",
        flush=True,
    )