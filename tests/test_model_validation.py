"""Regression tests for invalid datasets, model outputs, and quality gates."""
import numpy as np
import pandas as pd
import pytest
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.model_validation import (
    evaluate_predictions,
    validate_dataset_frame,
    validate_predictions,
    validate_quality_metrics,
)


@pytest.fixture
def valid_data():
    return pd.DataFrame(
        {
            "Bedrooms": [2, 3, 4, 2],
            "Area": [800, 1200, 1600, 900],
            "Location": ["Mumbai", "Pune", "Mumbai", "Pune"],
            "Age": [5, 8, 3, 10],
            "Price": [8_000_000, 12_000_000, 16_000_000, 9_000_000],
        }
    )


def test_valid_dataset_splits_features_and_target(valid_data):
    validate_dataset_frame(valid_data, "test")
    X = valid_data[["Bedrooms", "Area", "Location", "Age"]]
    y = valid_data["Price"]
    assert list(X.columns) == ["Bedrooms", "Area", "Location", "Age"]
    assert len(X) == len(y) == 4


def test_missing_required_column_is_rejected(valid_data):
    invalid = valid_data.drop(columns=["Area"])
    with pytest.raises(ValueError, match="missing columns.*Area"):
        validate_dataset_frame(invalid, "test")


def test_null_required_value_is_rejected(valid_data):
    invalid = valid_data.copy()
    invalid.loc[1, "Location"] = None
    with pytest.raises(ValueError, match="null required values"):
        validate_dataset_frame(invalid, "test")


def test_non_numeric_feature_is_rejected(valid_data):
    invalid = valid_data.copy()
    invalid["Area"] = invalid["Area"].astype(object)
    invalid.loc[0, "Area"] = "large"
    with pytest.raises(ValueError, match="Area.*numeric"):
        validate_dataset_frame(invalid, "test")


def test_non_finite_dataset_value_is_rejected(valid_data):
    invalid = valid_data.copy()
    invalid["Price"] = invalid["Price"].astype(float)
    invalid.loc[0, "Price"] = np.inf
    with pytest.raises(ValueError, match="non-finite"):
        validate_dataset_frame(invalid, "test")


def test_unexpected_category_is_handled_by_preprocessor(valid_data):
    """A new location must not crash the OneHotEncoder-based model pipeline."""
    X = valid_data[["Bedrooms", "Area", "Location", "Age"]]
    y = valid_data["Price"]
    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", StandardScaler(), ["Bedrooms", "Area", "Age"]),
            ("categorical", OneHotEncoder(handle_unknown="ignore"), ["Location"]),
        ]
    )
    model = Pipeline(
        [
            ("preprocessor", preprocessor),
            ("model", RandomForestRegressor(n_estimators=10, random_state=42)),
        ]
    )
    model.fit(X, y)
    unexpected = pd.DataFrame(
        {"Bedrooms": [2], "Area": [1000], "Location": ["Nashik"], "Age": [4]}
    )
    predictions = validate_predictions(model.predict(unexpected), expected_rows=1)
    assert predictions.shape == (1,)
    assert np.isfinite(predictions).all()


def test_prediction_shape_must_match_input_rows():
    with pytest.raises(ValueError, match="Prediction shape mismatch"):
        validate_predictions(np.array([[100.0], [200.0]]), expected_rows=2)


@pytest.mark.parametrize("bad_predictions", [[np.nan], [np.inf], [-np.inf]])
def test_non_finite_predictions_are_rejected(bad_predictions):
    with pytest.raises(ValueError, match="non-finite"):
        validate_predictions(bad_predictions, expected_rows=1)


def test_model_evaluation_rejects_wrong_prediction_count():
    class WrongShapeModel:
        def predict(self, X):
            return np.array([100.0])  # one prediction even if X has two rows

    X = pd.DataFrame({"Bedrooms": [2, 3]})
    y = pd.Series([100.0, 200.0])
    with pytest.raises(ValueError, match="Prediction shape mismatch"):
        evaluate_predictions(WrongShapeModel(), X, y)


def test_model_evaluation_rejects_nan_prediction():
    class NaNModel:
        def predict(self, X):
            return np.full(len(X), np.nan)

    X = pd.DataFrame({"Bedrooms": [2, 3]})
    y = pd.Series([100.0, 200.0])
    with pytest.raises(ValueError, match="non-finite"):
        evaluate_predictions(NaNModel(), X, y)


def test_deliberately_bad_model_fails_quality_gates():
    bad_metrics = {"mae": 5_000_000, "r2": 0.10}
    with pytest.raises(RuntimeError, match="R²=.*required"):
        validate_quality_metrics(bad_metrics)


def test_non_finite_metrics_fail_quality_gates():
    with pytest.raises(ValueError, match="non-finite"):
        validate_quality_metrics({"mae": np.inf, "r2": 0.95})
