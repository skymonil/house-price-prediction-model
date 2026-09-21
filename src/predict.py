import joblib
import pandas as pd

from fastapi import FastAPI
from pydantic import BaseModel


# --------------------------------------------------
# 1. Create FastAPI application
# --------------------------------------------------

app = FastAPI(title="House Price Prediction API")


# --------------------------------------------------
# 2. Load model + preprocessor
# --------------------------------------------------

artifact = joblib.load("models/model.pkl")

model = artifact["model"]
preprocessor = artifact["preprocessor"]


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

    # Apply the same preprocessing used during training
    new_house_processed = preprocessor.transform(new_house)

    # Make prediction
    predicted_price = model.predict(new_house_processed)

    return {
        "predicted_price": predicted_price[0]
    }

@app.get("/health")
def health():
    return {"status": "ok"}