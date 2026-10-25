# src/compare_models.py
import argparse
import json
import os
from pathlib import Path

import mlflow
import mlflow.sklearn
import pandas as pd
from mlflow import MlflowClient
from sklearn.metrics import mean_absolute_error, r2_score

from src.dataset_utils import load_verified_split

MODEL_NAME = "HousePricePredictor"
CHAMPION_ALIAS = "champion"

MIN_R2 = 0.90
MAX_MAE = 1_000_000
MIN_MAE_IMPROVEMENT = 0.02
MAX_R2_REGRESSION = 0.01

REPORT_DIR = Path("reports")


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


def evaluate_model(model, X_eval, y_eval):
    predictions = model.predict(X_eval)
    return {
        "mae": float(mean_absolute_error(y_eval, predictions)),
        "r2": float(r2_score(y_eval, predictions)),
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

    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "https://mlflow.615915.xyz")
    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient()

    candidate_version = str(args.candidate_version)
    champion_metadata = client.get_model_version_by_alias(
        MODEL_NAME, CHAMPION_ALIAS
    )
    champion_version = str(champion_metadata.version)

    if champion_version == candidate_version:
        raise ValueError(
            "Candidate version is already the champion. "
            "Choose a different candidate version."
        )

    candidate_metadata = client.get_model_version(
        MODEL_NAME, candidate_version
    )

    # Use the exact same fixed validation rows for both models.
    # Do not use test.csv here; it should be reserved for final evaluation.
    validation_data, manifest = load_verified_split("validation")
    X_validation, y_validation = split_features_target(
        validation_data, "validation"
    )

    print(f"Tracking URI: {tracking_uri}")
    print(f"Champion: {MODEL_NAME}/{champion_version}")
    print(f"Candidate: {MODEL_NAME}/{candidate_version}")
    print(
        f"Validation dataset: {manifest['dataset_version']} "
        f"({len(validation_data)} rows)"
    )

    # Load the exact resolved champion version, not the moving alias again.
    champion_model = mlflow.sklearn.load_model(
        f"models:/{MODEL_NAME}/{champion_version}"
    )
    candidate_model = mlflow.sklearn.load_model(
        f"models:/{MODEL_NAME}/{candidate_version}"
    )

    champion_metrics = evaluate_model(
        champion_model, X_validation, y_validation
    )
    candidate_metrics = evaluate_model(
        candidate_model, X_validation, y_validation
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
        "champion_version": champion_version,
        "champion_run_id": champion_metadata.run_id,
        "candidate_version": candidate_version,
        "candidate_run_id": candidate_metadata.run_id,
        "evaluation": {
            "dataset_version": manifest.get("dataset_version", "unknown"),
            "dataset_path": manifest["splits"]["validation"]["path"],
            "dataset_sha256": manifest["splits"]["validation"]["sha256"],
            "source_dataset_sha256": manifest["source_dataset"]["sha256"],
            "evaluation_type": "validation",
            "row_count": len(validation_data),
            "git_commit": os.environ.get("GITHUB_SHA", "local"),
        },
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
        f"comparison-v{champion_version}-vs-v{candidate_version}.json"
    )
    report_path.write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
    )

    mlflow.set_experiment("house-price-model-comparison")
    with mlflow.start_run(run_name=f"compare-v{candidate_version}"):
        mlflow.log_params({
            "model_name": MODEL_NAME,
            "champion_version": champion_version,
            "candidate_version": candidate_version,
            "evaluation_type": "validation",
            "dataset_version": str(manifest.get("dataset_version", "unknown")),
            "validation_sha256": manifest["splits"]["validation"]["sha256"],
            "source_sha256": manifest["source_dataset"]["sha256"],
            "validation_rows": len(validation_data),
            "git_commit": os.environ.get("GITHUB_SHA", "local"),
        })
        mlflow.log_metrics({
            "champion_validation_mae": champion_metrics["mae"],
            "candidate_validation_mae": candidate_metrics["mae"],
            "champion_validation_r2": champion_metrics["r2"],
            "candidate_validation_r2": candidate_metrics["r2"],
            "mae_improvement_fraction": mae_improvement,
            "r2_change": r2_change,
            "absolute_gates_pass": float(absolute_gates_pass),
            "relative_gates_pass": float(relative_gates_pass),
            "recommend_promotion": float(recommend_promotion),
        })
        mlflow.log_artifact(str(report_path))

    print("\n=== MODEL COMPARISON (VALIDATION DATA) ===")
    print(f"Validation rows: {len(validation_data)}")
    print(f"Champion MAE:    ₹{champion_metrics['mae']:,.2f}")
    print(f"Candidate MAE:   ₹{candidate_metrics['mae']:,.2f}")
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
