import mlflow
import pandas as pd


# -------------------------
# Load registered model
# -------------------------

model_uri = "models:/HousePricePredictor/1"

model = mlflow.sklearn.load_model(model_uri)


# -------------------------
# New house
# -------------------------

new_house = pd.DataFrame({
    "Bedrooms": [3],
    "Area": [1200],
    "Location": ["Mumbai"],
    "Age": [5]
})


# -------------------------
# Prediction
# -------------------------

predicted_price = model.predict(new_house)


print(f"Predicted Price: ₹{predicted_price[0]:,.2f}")