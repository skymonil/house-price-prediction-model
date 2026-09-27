import os
import joblib
import mlflow
import mlflow.sklearn
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from src.preprocess import create_preprocessor
from src.evaluate import evaluate_model

MLFLOW_TRACKING_URI = os.getenv(
    "MLFLOW_TRACKING_URI",
    "http://localhost:5000",
)

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)


DATA_PATH = "data/houses.csv"
MODEL_PATH = "models/model.pkl"

EXPERIMENT_NAME = "house-price-prediction-v2"
MODEL_NAME = "HousePricePredictor"


def load_data():
    df = pd.read_csv(DATA_PATH)

    print(f"Dataset loaded: {len(df)} rows")

    X = df[["Bedrooms", "Area", "Location", "Age"]]
    y = df["Price"]

    return X, y


def create_model():
    preprocessor = create_preprocessor()

    model = RandomForestRegressor(
        n_estimators=200,
        random_state=42,
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )

    return pipeline


def train():
    X, y = load_data()

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
    )

    pipeline = create_model()

    mlflow.set_experiment(EXPERIMENT_NAME)

    with mlflow.start_run(run_name="RandomForest"):

        pipeline.fit(X_train, y_train)

        metrics = evaluate_model(
            pipeline,
            X_test,
            y_test,
        )

        mlflow.log_param(
            "model",
            "RandomForest",
        )

        mlflow.log_param(
            "n_estimators",
            200,
        )

        mlflow.log_param(
            "random_state",
            42,
        )

        mlflow.log_metric(
            "mae",
            metrics["mae"],
        )

        mlflow.log_metric(
            "r2",
            metrics["r2"],
        )

        joblib.dump(
            pipeline,
            MODEL_PATH,
        )

        model_info = mlflow.sklearn.log_model(
            pipeline,
            name="model",
            skops_trusted_types=[
                "sklearn.tree._tree.Tree"
            ],
        )

        registered_model = mlflow.register_model(
            model_uri=model_info.model_uri,
            name=MODEL_NAME,
        )

        print("\nModel Evaluation")
        print("----------------")
        print("Model: Random Forest")
        print("n_estimators: 200")
        print(f"MAE: ₹{metrics['mae']:,.2f}")
        print(f"R²: {metrics['r2']:.4f}")

        print("\nMLflow")
        print("------")
        print(f"Model: {registered_model.name}")
        print(f"Version: {registered_model.version}")

        print(f"\nModel saved to {MODEL_PATH}")


if __name__ == "__main__":
    train()