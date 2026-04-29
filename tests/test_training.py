import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from training import build_preprocessor, evaluate_predictions


def test_build_preprocessor_has_expected_transformers():
    categorical = ["Sex", "Embarked", "Pclass"]
    numerical = ["Age", "SibSp", "Parch", "Fare"]

    preprocessor = build_preprocessor(categorical, numerical)

    assert set(preprocessor.transformers[0][2]) == set(categorical)
    assert set(preprocessor.transformers[1][2]) == set(numerical)
    assert isinstance(preprocessor.transformers[0][1], OneHotEncoder)
    assert isinstance(preprocessor.transformers[1][1], StandardScaler)


def test_build_preprocessor_transforms_sample_dataframe():
    categorical = ["Sex", "Embarked", "Pclass"]
    numerical = ["Age", "SibSp", "Parch", "Fare"]
    df = pd.DataFrame(
        {
            "Sex": ["male", "female", "female"],
            "Embarked": ["S", "C", "Q"],
            "Pclass": [3, 1, 2],
            "Age": [22.0, 38.0, 26.0],
            "SibSp": [1, 1, 0],
            "Parch": [0, 0, 0],
            "Fare": [7.25, 71.2833, 7.925],
        }
    )

    preprocessor = build_preprocessor(categorical, numerical)
    transformed = preprocessor.fit_transform(df)

    assert transformed.shape[0] == len(df)
    assert transformed.shape[1] > len(numerical)


def test_evaluate_predictions_returns_expected_metrics():
    y_true = np.array([1, 0, 1, 0, 1])
    y_pred = np.array([1, 0, 0, 0, 1])

    metrics = evaluate_predictions(y_true, y_pred)

    assert "accuracy" in metrics
    assert "classification_report" in metrics
    assert "confusion_matrix" in metrics
    assert metrics["accuracy"] == 0.8
    assert metrics["confusion_matrix"].shape == (2, 2)
    assert "0" in metrics["classification_report"]
    assert "1" in metrics["classification_report"]
