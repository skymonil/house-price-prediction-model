import os
from contextlib import asynccontextmanager

import mlflow
import mlflow.sklearn
from fastapi import FastAPI, Request
from pydantic import BaseModel

from src.predict import predict_price

@asynccontextmanager
async def lifespan(app: FastAPI):
    if not getattr(app.state, "testing", False):
        app.state.model = load_model()

    yield

    app.state.model = None
# --------------------------------------------------
# Configuration
# --------------------------------------------------

MLFLOW_TRACKING_URI = os.getenv(
    "MLFLOW_TRACKING_URI",
    "http://localhost:5000",
)

MLFLOW_MODEL_URI = os.getenv(
    "MLFLOW_MODEL_URI",
    "models:/HousePricePredictor/1",
)


# --------------------------------------------------
# Model loading
# --------------------------------------------------

def load_model():
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

    return mlflow.sklearn.load_model(
        MLFLOW_MODEL_URI
    )


# --------------------------------------------------
# FastAPI lifespan
# --------------------------------------------------


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
    model = request.app.state.model

    predicted_price = predict_price(
        model=model,
        bedrooms=house.bedrooms,
        area=house.area,
        location=house.location,
        age=house.age,
    )

    return {
        "predicted_price": predicted_price
    }


# --------------------------------------------------
# Health endpoint
# --------------------------------------------------

@app.get("/health")
def health():
    return {
        "status": "ok"
    }