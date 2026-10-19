
import argparse
import json
import os
from pathlib import Path

import mlflow
import mlflow.sklearn
import pandas as pd
from mlflow import MlflowClient
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split


MODEL_NAME = "HousePricePredictor"
CHAMPION_ALIAS = "champion"

MIN_R2 = 0.90
MAX_MAE = 1_000_000

# Candidate must improve MAE by at least 2%,
# without losing more than 0.01 in R².
MIN_MAE_IMPROVEMENT = 0.02
MAX_R2_REGRESSION = 0.01

DATA_PATH = Path("data/houses.csv")
REPORT_DIR = Path("reports")


def evaluate_model(model, X_test, y_test):
    predictions = model.predict(X_test)

    return {
        "mae": float(mean_absolute_error(y_test, predictions)),
        "r2": float(r2_score(y_test, predictions)),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Compare a candidate model with the MLflow champion."
    )
    parser.add_argument(
        "--candidate-version",
        required=True,
        help="Registered MLflow model version to evaluate",
    )
    args = parser.parse_args()

    tracking_uri = os.getenv(
        "MLFLOW_TRACKING_URI",
        "http://localhost:5000",
    )
    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient()

    candidate_version = str(args.candidate_version)

    # Resolve the champion alias to a specific registered version.
    champion = client.get_model_version_by_alias(
        MODEL_NAME,
        CHAMPION_ALIAS,
    )

    if str(champion.version) == candidate_version:
        raise ValueError(
            "Candidate version is already the champion. "
            "Choose a different candidate version."
        )

    candidate_metadata = client.get_model_version(
        MODEL_NAME,
        candidate_version,
    )

    print(f"Tracking URI: {tracking_uri}")
    print(f"Champion: {MODEL_NAME}/{champion.version}")
    print(f"Candidate: {MODEL_NAME}/{candidate_version}")

    # Use the same held-out rows for both models.
    df = pd.read_csv(DATA_PATH)

    required_columns = [
        "Bedrooms", "Area", "Location", "Age", "Price"
    ]
    missing = set(required_columns) - set(df.columns)
    if missing:
        raise ValueError(f"Dataset is missing columns: {sorted(missing)}")

    if df[required_columns].isna().any().any():
        raise ValueError(
            "Dataset contains missing required values. "
            "Apply the same documented cleaning used in training first."
        )

    X = df[["Bedrooms", "Area", "Location", "Age"]]
    y = pd.to_numeric(df["Price"], errors="raise")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
    )

    # These are complete sklearn pipelines, including preprocessing.
    champion_model = mlflow.sklearn.load_model(
        f"models:/{MODEL_NAME}@{CHAMPION_ALIAS}"
    )
    candidate_model = mlflow.sklearn.load_model(
        f"models:/{MODEL_NAME}/{candidate_version}"
    )

    champion_metrics = evaluate_model(
        champion_model, X_test, y_test
    )
    candidate_metrics = evaluate_model(
        candidate_model, X_test, y_test
    )

    mae_improvement = (
        (champion_metrics["mae"] - candidate_metrics["mae"])
        / champion_metrics["mae"]
        if champion_metrics["mae"] > 0
        else 0.0
    )
    r2_change = candidate_metrics["r2"] - champion_metrics["r2"]

    absolute_gates_pass = (
        candidate_metrics["mae"] <= MAX_MAE
        and candidate_metrics["r2"] >= MIN_R2
    )

    relative_gates_pass = (
        mae_improvement >= MIN_MAE_IMPROVEMENT
        and r2_change >= -MAX_R2_REGRESSION
    )

    recommend_promotion = absolute_gates_pass and relative_gates_pass

    report = {
        "model_name": MODEL_NAME,
        "champion_version": str(champion.version),
        "candidate_version": candidate_version,
        "candidate_run_id": candidate_metadata.run_id,
        "test_rows": len(X_test),
        "champion": champion_metrics,
        "candidate": candidate_metrics,
        "mae_improvement_fraction": mae_improvement,
        "r2_change": r2_change,
        "absolute_gates_pass": absolute_gates_pass,
        "relative_gates_pass": relative_gates_pass,
        "recommend_promotion": recommend_promotion,
    }

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    report_path = REPORT_DIR / (
        f"comparison-v{champion.version}-vs-v{candidate_version}.json"
    )
    report_path.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    # Keep comparison results separate from training runs.
    mlflow.set_experiment("house-price-model-comparison")

    with mlflow.start_run(run_name=f"compare-v{candidate_version}"):
        mlflow.log_params({
            "model_name": MODEL_NAME,
            "champion_version": str(champion.version),
            "candidate_version": candidate_version,
            "test_rows": len(X_test),
        })
        mlflow.log_metrics({
            "champion_mae": champion_metrics["mae"],
            "candidate_mae": candidate_metrics["mae"],
            "champion_r2": champion_metrics["r2"],
            "candidate_r2": candidate_metrics["r2"],
            "mae_improvement_fraction": mae_improvement,
            "r2_change": r2_change,
            "absolute_gates_pass": float(absolute_gates_pass),
            "relative_gates_pass": float(relative_gates_pass),
            "recommend_promotion": float(recommend_promotion),
        })
        mlflow.log_artifact(str(report_path))

    print("\n=== MODEL COMPARISON ===")
    print(f"Test rows:       {len(X_test)}")
    print(
        f"Champion MAE:    ₹{champion_metrics['mae']:,.2f}"
    )
    print(
        f"Candidate MAE:   ₹{candidate_metrics['mae']:,.2f}"
    )
    print(f"MAE improvement: {mae_improvement:.2%}")
    print(f"Champion R²:     {champion_metrics['r2']:.4f}")
    print(f"Candidate R²:    {candidate_metrics['r2']:.4f}")
    print(f"R² change:       {r2_change:+.4f}")
    print(f"Absolute gates:  {'PASS' if absolute_gates_pass else 'FAIL'}")
    print(f"Relative gates:  {'PASS' if relative_gates_pass else 'FAIL'}")
    print(f"Report:          {report_path}")

    if recommend_promotion:
        print("\nRECOMMENDATION: Candidate is eligible for manual approval.")
    else:
        print("\nRECOMMENDATION: Do not promote this candidate.")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
