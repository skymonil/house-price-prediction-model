import joblib
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder


# --------------------------------------------------
# 1. Load dataset
# --------------------------------------------------

df = pd.read_csv("data/houses.csv")

print(f"Dataset loaded: {len(df)} rows")


# --------------------------------------------------
# 2. Define features and target
# --------------------------------------------------

X = df[["Bedrooms", "Area", "Location", "Age"]]
y = df["Price"]


# --------------------------------------------------
# 3. Train/test split
# --------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# --------------------------------------------------
# 4. Preprocessing
# --------------------------------------------------

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


# --------------------------------------------------
# 5. Transform training and test data
# --------------------------------------------------

X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)


# --------------------------------------------------
# 6. Train model
# --------------------------------------------------

model = LinearRegression()

model.fit(X_train_processed, y_train)


# --------------------------------------------------
# 7. Evaluate model
# --------------------------------------------------

predictions = model.predict(X_test_processed)

mae = mean_absolute_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

print("\nModel evaluation")
print("----------------")
print(f"MAE: ₹{mae:,.2f}")
print(f"R²:  {r2:.4f}")


# --------------------------------------------------
# 8. Save model + preprocessing
# --------------------------------------------------

model_artifact = {
    "model": model,
    "preprocessor": preprocessor
}

joblib.dump(
    model_artifact,
    "models/model.pkl"
)

print("\nModel saved to models/model.pkl")