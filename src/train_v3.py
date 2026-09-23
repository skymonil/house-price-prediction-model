import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
import os
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

mlflow.set_tracking_uri(
    os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
)
# -------------------------
# Load dataset
# -------------------------

df = pd.read_csv("data/houses.csv")

print(f"Dataset loaded: {len(df)} rows")


# -------------------------
# Features and target
# -------------------------

X = df[["Bedrooms", "Area", "Location", "Age"]]
y = df["Price"]


# -------------------------
# Train/test split
# -------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# -------------------------
# Preprocessing
# -------------------------

categorical_features = ["Location"]
numeric_features = ["Bedrooms", "Area", "Age"]

preprocessor = ColumnTransformer(
    transformers=[
        (
            "location",
            OneHotEncoder(handle_unknown="ignore"),
            categorical_features
        )
    ],
    remainder="passthrough"
)


# -------------------------
# Model
# -------------------------

n_estimators = 200
random_state = 42

model = RandomForestRegressor(
    n_estimators=n_estimators,
    random_state=random_state
)


# -------------------------
# Complete ML pipeline
# -------------------------

pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("model", model)
    ]
)


# -------------------------
# MLflow experiment
# -------------------------

mlflow.set_experiment("house-price-prediction-v2")

with mlflow.start_run(run_name="RandomForest-V3"):

    # Train complete pipeline
    pipeline.fit(X_train, y_train)

    # Predict using complete pipeline
    predictions = pipeline.predict(X_test)

    # Metrics
    mae = mean_absolute_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)

    # Log parameters
    mlflow.log_param("model", "RandomForest")
    mlflow.log_param("n_estimators", n_estimators)
    mlflow.log_param("random_state", random_state)

    # Log metrics
    mlflow.log_metric("mae", mae)
    mlflow.log_metric("r2", r2)

    # Save complete pipeline locally
    model_path = "models/model_v3.pkl"

    joblib.dump(
        pipeline,
        model_path
    )

    # Log complete pipeline to MLflow
    model_info = mlflow.sklearn.log_model(
    pipeline,
    name="model",
    skops_trusted_types=[
        "sklearn.tree._tree.Tree"
    ]
)
    registered_model = mlflow.register_model(
    model_uri=model_info.model_uri,
    name="HousePricePredictor"
)

    print(
    f"\nRegistered model: {registered_model.name}"
)
    print(
    f"Model version: {registered_model.version}"
)
    # -------------------------
    # Output
    # -------------------------

    print("\nV3 Model Evaluation")
    print("-------------------")
    print("Model: Random Forest")
    print(f"n_estimators: {n_estimators}")
    print(f"MAE: ₹{mae:,.2f}")
    print(f"R²:  {r2:.4f}")

    print("\nModel saved to models/model_v3.pkl")