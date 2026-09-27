import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from src.preprocess import create_preprocessor
from src.evaluate import evaluate_model


# Quality thresholds
MIN_R2 = 0.90
MAX_MAE = 1_000_000


def train_model():
    # Load dataset
    data = pd.read_csv("data/houses.csv")

    X = data[["Bedrooms", "Area", "Location", "Age"]]
    y = data["Price"]

    # Same split used by the training pipeline
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
    )

    # Create preprocessing + model pipeline
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

    # Train
    model.fit(X_train, y_train)

    # Evaluate
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


if __name__ == "__main__":
    _, metrics = train_model()
    validate_model(metrics)