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
    data = pd.read_csv("data/houses.csv")

    X = data[["Bedrooms", "Area", "Location", "Age"]]
    y = data["Price"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
    )

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

    model.fit(X_train, y_train)

    metrics = evaluate_model(
        model,
        X_test,
        y_test,
    )

    return model, metrics


# ============================================================
# Model Quality Gate
# ============================================================

def validate_model(metrics):
    print("\nModel Evaluation")
    print("----------------")
    print(f"MAE: ₹{metrics['mae']:,.2f}")
    print(f"R²:  {metrics['r2']:.4f}")

    print("\nQuality Gate")
    print("------------")
    print(f"Required R²  >= {MIN_R2}")
    print(f"Required MAE <= ₹{MAX_MAE:,.0f}")

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

    print("\nModel quality check PASSED.")


# ============================================================
# Save Local CI Artifacts
# ============================================================

def save_artifacts(model, metrics):
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

    print("\nModel artifacts saved:")
    print(f"  Model:   {MODEL_PATH}")
    print(f"  Metrics: {METRICS_PATH}")


# ============================================================
# Log Model to MLflow
# ============================================================

def log_to_mlflow(model, metrics):
    mlflow.set_tracking_uri(
        MLFLOW_TRACKING_URI
    )

    mlflow.set_experiment(
        MLFLOW_EXPERIMENT
    )

    with mlflow.start_run() as run:

        mlflow.log_params(
            {
                "model_type": "RandomForestRegressor",
                "n_estimators": 200,
                "random_state": 42,
                "test_size": 0.2,
            }
        )

        mlflow.log_metrics(
            {
                "mae": metrics["mae"],
                "r2": metrics["r2"],
            }
        )

        # Store the sklearn model under this artifact path.
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="house-price-model",
        )

        run_id = run.info.run_id

        print("\nMLflow run logged successfully.")
        print(f"Run ID: {run_id}")

        return run_id


# ============================================================
# Register Model
# ============================================================

def register_model(run_id):
    model_uri = (
        f"runs:/{run_id}/house-price-model"
    )

    registered_model = mlflow.register_model(
        model_uri=model_uri,
        name=MODEL_NAME,
    )

    print("\nModel registered successfully.")
    print(f"Model name:    {registered_model.name}")
    print(f"Model version: {registered_model.version}")

    return registered_model


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":

    # --------------------------------------------------------
    # 1. Train
    # --------------------------------------------------------

    model, metrics = train_model()

    # --------------------------------------------------------
    # 2. Quality Gate
    # --------------------------------------------------------

    validate_model(metrics)

    # --------------------------------------------------------
    # 3. Save CI Artifacts
    # --------------------------------------------------------

    save_artifacts(
        model,
        metrics,
    )

    # --------------------------------------------------------
    # 4. Configure MLflow
    # --------------------------------------------------------

    mlflow.set_tracking_uri(
        MLFLOW_TRACKING_URI
    )

    # --------------------------------------------------------
    # 5. Log Model
    # --------------------------------------------------------

    run_id = log_to_mlflow(
        model,
        metrics,
    )

    # --------------------------------------------------------
    # 6. Register Model
    # --------------------------------------------------------

    register_model(
        run_id
    )