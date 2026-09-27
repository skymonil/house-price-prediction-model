import json
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from src.preprocess import create_preprocessor
from src.evaluate import evaluate_model
import os

MIN_R2 = 0.90
MAX_MAE = 1_000_000

MODEL_PATH = Path("models/model_ci.pkl")
METRICS_PATH = Path("models/metrics.json")

MLFLOW_EXPERIMENT = "house-price-prediction-v2"


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

    metrics = evaluate_model(model, X_test, y_test)

    return model, metrics


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
            f"Model quality check failed: "
            f"R²={metrics['r2']:.4f}, required >= {MIN_R2}"
        )

    if metrics["mae"] > MAX_MAE:
        raise RuntimeError(
            f"Model quality check failed: "
            f"MAE=₹{metrics['mae']:,.2f}, "
            f"required <= ₹{MAX_MAE:,.0f}"
        )

    print("\nModel quality check PASSED.")


def save_artifacts(model, metrics):
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(model, MODEL_PATH)

    with open(METRICS_PATH, "w") as file:
        json.dump(metrics, file, indent=4)

    print("\nModel artifacts saved:")
    print(f"  Model:   {MODEL_PATH}")
    print(f"  Metrics: {METRICS_PATH}")


def log_to_mlflow(model, metrics):
    mlflow.set_tracking_uri(
        os.environ["MLFLOW_TRACKING_URI"]
    )

    mlflow.set_experiment(MLFLOW_EXPERIMENT)

    with mlflow.start_run() as run:
        mlflow.log_params(
            {
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

        mlflow.sklearn.log_model(
            model,
            name="house-price-model",
            skops_trusted_types=[
                "sklearn.tree._tree.Tree"
            ],
        )

        run_id = run.info.run_id

        print("\nMLflow run logged successfully.")
        print(f"Run ID: {run_id}")

        return run_id
    
    mlflow.set_tracking_uri(
    os.environ.get("MLFLOW_TRACKING_URI", "https://mlflow.615915.xyz")
)

    mlflow.set_experiment(MLFLOW_EXPERIMENT)

    with mlflow.start_run():
        mlflow.log_params(
            {
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

        mlflow.sklearn.log_model(
            model,
            name="house-price-model",
            skops_trusted_types=[
           "sklearn.tree._tree.Tree"
         ]   ,
        )

        print("\nMLflow run logged successfully.")

def register_model(run_id):
    model_name = "HousePricePredictor"

    model_uri = f"runs:/{run_id}/house-price-model"

    registered_model = mlflow.register_model(
        model_uri=model_uri,
        name=model_name,
    )

    print("\nModel registered successfully.")
    print(f"Model name: {registered_model.name}")
    print(f"Model version: {registered_model.version}")

    return registered_model

if __name__ == "__main__":
    model, metrics = train_model()

    # 1. Quality gate
    validate_model(metrics)

    # 2. Save local CI artifacts
    save_artifacts(model, metrics)

    # 3. Log successful model to MLflow
    run_id = log_to_mlflow(model, metrics)

    # 4. Register only after quality gate passes
    register_model(run_id)