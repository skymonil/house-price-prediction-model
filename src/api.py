import mlflow
import mlflow.sklearn
import pandas as pd
import os
from fastapi import FastAPI
from pydantic import BaseModel


# --------------------------------------------------
# 1. Create FastAPI application
# --------------------------------------------------

app = FastAPI(title="House Price Prediction API")


# --------------------------------------------------
# 2. Load registered MLflow model
# --------------------------------------------------

MLFLOW_TRACKING_URI = os.getenv(
    "MLFLOW_TRACKING_URI",
    "http://localhost:5000"
)

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

model = mlflow.sklearn.load_model(
    "models:/HousePricePredictor/1"
)


# --------------------------------------------------
# 3. Define prediction request
# --------------------------------------------------

class HouseInput(BaseModel):
    bedrooms: int
    area: float
    location: str
    age: int


# --------------------------------------------------
# 4. Prediction endpoint
# --------------------------------------------------

@app.post("/predict")
def predict_price(house: HouseInput):

    # Convert request into DataFrame
    new_house = pd.DataFrame({
        "Bedrooms": [house.bedrooms],
        "Area": [house.area],
        "Location": [house.location],
        "Age": [house.age]
    })

    # Pipeline handles:
    # Data → Preprocessor → Random Forest → Prediction
    predicted_price = model.predict(new_house)

    return {
        "predicted_price": predicted_price[0]
    }


# --------------------------------------------------
# 5. Health endpoint
# --------------------------------------------------

@app.get("/health")
def health():
    return {"status": "ok"}