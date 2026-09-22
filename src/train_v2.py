import joblib
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder


# Load dataset
df = pd.read_csv("data/houses.csv")

print(f"Dataset loaded: {len(df)} rows")


# Features and target
X = df[["Bedrooms", "Area", "Location", "Age"]]
y = df["Price"]


# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# Feature types
categorical_features = ["Location"]
numeric_features = ["Bedrooms", "Area", "Age"]


# Preprocessing
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


# Fit preprocessing only on training data
X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)


# V2 model: Random Forest
model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)


# Train model
model.fit(X_train_processed, y_train)


# Make predictions
predictions = model.predict(X_test_processed)


# Evaluate model
mae = mean_absolute_error(y_test, predictions)
r2 = r2_score(y_test, predictions)


print("\nV2 Model Evaluation")
print("-------------------")
print("Model: Random Forest")
print(f"MAE: ₹{mae:,.2f}")
print(f"R²:  {r2:.4f}")


# Save model and preprocessor
model_artifact = {
    "model": model,
    "preprocessor": preprocessor
}

joblib.dump(
    model_artifact,
    "models/model_v2.pkl"
)

print("\nV2 model saved to models/model_v2.pkl")