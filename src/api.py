import os
from contextlib import asynccontextmanager

import mlflow
import mlflow.sklearn
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel

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
# Model loading
# --------------------------------------------------

def load_model():
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

    return mlflow.sklearn.load_model(MLFLOW_MODEL_URI)


# --------------------------------------------------
# FastAPI lifespan
# --------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.model = None

    if not getattr(app.state, "testing", False):
        app.state.model = load_model()

    yield

    app.state.model = None


app = FastAPI(
    title="House Price Prediction API",
    lifespan=lifespan,
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
    model = getattr(request.app.state, "model", None)

    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Model is not loaded",
        )

    predicted_price = predict_price(
        model=model,
        bedrooms=house.bedrooms,
        area=house.area,
        location=house.location,
        age=house.age,
    )

    return {
        "predicted_price": predicted_price,
    }


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
    model = getattr(request.app.state, "model", None)

    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Model is not loaded",
        )

    return {
        "status": "ready",
    }