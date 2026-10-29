import os
import time
from contextlib import asynccontextmanager

import mlflow
import mlflow.sklearn

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
from prometheus_client import Counter, Gauge, Histogram
from prometheus_fastapi_instrumentator import Instrumentator
from mlflow.tracking import MlflowClient

from src.predict import predict_price


# --------------------------------------------------
# Configuration
# --------------------------------------------------

MLFLOW_TRACKING_URI = os.getenv(
    "MLFLOW_TRACKING_URI",
    "http://localhost:5000",
)

MLFLOW_MODEL_URI = os.getenv(
    "MLFLOW_MODEL_URI",
    "models:/HousePricePredictor@champion",
)


# --------------------------------------------------
# Custom Prometheus metrics
# --------------------------------------------------

# Identifies the model version currently loaded by this API.
MODEL_INFO = Gauge(
    "house_price_model_info",
    "Information about the model currently loaded by the API.",
    ["model_name", "model_version"],
)

# Counts predictions successfully produced by the model.
PREDICTIONS_TOTAL = Counter(
    "house_price_predictions_total",
    "Total number of successful house price predictions.",
    ["model_name", "model_version"],
)

# Tracks the distribution of predicted prices in Indian rupees.
PREDICTION_VALUE = Histogram(
    "house_price_prediction_value_rupees",
    "Distribution of predicted house prices in INR.",
    ["model_name", "model_version"],
    buckets=(
        500_000,
        1_000_000,
        2_500_000,
        5_000_000,
        7_500_000,
        10_000_000,
        15_000_000,
        25_000_000,
        50_000_000,
    ),
)

# Measures execution time of the prediction function.
INFERENCE_DURATION = Histogram(
    "house_price_model_inference_seconds",
    "Time spent executing the house price prediction function.",
    ["model_name", "model_version"],
    buckets=(
        0.001,
        0.005,
        0.01,
        0.025,
        0.05,
        0.1,
        0.25,
        0.5,
        1.0,
        2.5,
        5.0,
    ),
)

# Counts model prediction errors.
PREDICTION_ERRORS_TOTAL = Counter(
    "house_price_prediction_errors_total",
    "Total number of errors during house price prediction.",
    ["model_name", "model_version", "error_type"],
)


# --------------------------------------------------
# Model identity
# --------------------------------------------------

def get_model_identity():
    """
    Resolve the actual MLflow model version being loaded.

    Supports version-based URIs and registered model aliases.
    """

    parts = MLFLOW_MODEL_URI.split("/")

    model_name = "HousePricePredictor"
    model_version = "unknown"

    try:
        if len(parts) >= 3 and parts[0] == "models:":
            model_name = parts[1]

            version_or_alias = parts[2]

            client = MlflowClient(
                tracking_uri=MLFLOW_TRACKING_URI
            )

            if version_or_alias.startswith("@"):
                alias = version_or_alias[1:]

                model_version_info = (
                    client.get_model_version_by_alias(
                        model_name,
                        alias,
                    )
                )

                model_version = str(model_version_info.version)

            else:
                model_version = version_or_alias

    except Exception:
        # Do not expose MLflow metadata errors through the API.
        # The model load itself will still determine startup success.
        model_version = "unknown"

    return model_name, model_version


# --------------------------------------------------
# Model loading
# --------------------------------------------------

def load_model():
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

    model = mlflow.sklearn.load_model(
        MLFLOW_MODEL_URI
    )

    model_name, model_version = get_model_identity()

    return model, model_name, model_version


# --------------------------------------------------
# FastAPI lifespan
# --------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.model = None
    app.state.model_name = "HousePricePredictor"
    app.state.model_version = "unknown"

    if not getattr(app.state, "testing", False):
        model, model_name, model_version = load_model()

        app.state.model = model
        app.state.model_name = model_name
        app.state.model_version = model_version

        # Mark the loaded model version as active.
        MODEL_INFO.labels(
            model_name=model_name,
            model_version=model_version,
        ).set(1)

    yield

    app.state.model = None


app = FastAPI(
    title="House Price Prediction API",
    lifespan=lifespan,
)

Instrumentator().instrument(app).expose(
    app,
    endpoint="/metrics",
)


# --------------------------------------------------
# Request schema
# --------------------------------------------------

class HouseInput(BaseModel):
    bedrooms: int
    area: float
    location: str
    age: int


# --------------------------------------------------
# Prediction endpoint
# --------------------------------------------------

@app.post("/predict")
def predict(
    house: HouseInput,
    request: Request,
):
    model = getattr(
        request.app.state,
        "model",
        None,
    )

    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Model is not loaded",
        )

    model_name = request.app.state.model_name
    model_version = request.app.state.model_version

    labels = {
        "model_name": model_name,
        "model_version": model_version,
    }

    start_time = time.perf_counter()

    try:
        predicted_price = predict_price(
            model=model,
            bedrooms=house.bedrooms,
            area=house.area,
            location=house.location,
            age=house.age,
        )

        predicted_price = float(predicted_price)

        # Record only successful predictions.
        PREDICTIONS_TOTAL.labels(
            **labels
        ).inc()

        PREDICTION_VALUE.labels(
            **labels
        ).observe(predicted_price)

        return {
            "predicted_price": predicted_price,
        }

    except Exception as exc:
        PREDICTION_ERRORS_TOTAL.labels(
            **labels,
            error_type=type(exc).__name__,
        ).inc()

        raise

    finally:
        elapsed = time.perf_counter() - start_time

        INFERENCE_DURATION.labels(
            **labels
        ).observe(elapsed)


# --------------------------------------------------
# Health endpoint
# --------------------------------------------------

@app.get("/health")
def health():
    return {
        "status": "ok",
    }


# --------------------------------------------------
# Readiness endpoint
# --------------------------------------------------

@app.get("/ready")
def ready(request: Request):
    model = getattr(
        request.app.state,
        "model",
        None,
    )

    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Model is not loaded",
        )

    return {
        "status": "ready",
    }