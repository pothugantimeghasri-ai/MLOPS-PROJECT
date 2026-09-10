import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from src.data_processing import preprocess_data
from src.model import FEATURES, create_model, load_model, predict_application, save_model, train_model


@pytest.fixture
def sample_data() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "annual_income": [75000, 95000, 110000, 58000, 72000, 130000],
            "loan_amount": [150000, 160000, 180000, 90000, 130000, 220000],
            "employment_years": [2, 4, 5, 1, 3, 7],
            "credit_history_months": [18, 24, 36, 12, 30, 60],
            "debt_to_income": [0.35, 0.28, 0.25, 0.42, 0.31, 0.22],
            "previous_defaults": [0, 0, 0, 1, 0, 0],
            "approved": [0, 1, 1, 0, 0, 1],
        }
    )


def test_model_trains_successfully(sample_data: pd.DataFrame) -> None:
    X = sample_data[FEATURES]
    y = sample_data["approved"]
    model = train_model(X, y)
    assert isinstance(model, LogisticRegression)


def test_predictions_are_0_or_1(sample_data: pd.DataFrame) -> None:
    model = train_model(sample_data[FEATURES], sample_data["approved"])
    prediction, probability = predict_application(model, {
        "annual_income": 75000,
        "loan_amount": 200000,
        "employment_years": 6,
        "credit_history_months": 60,
        "debt_to_income": 0.30,
        "previous_defaults": 0,
    })
    assert prediction in [0, 1]
    assert 0.0 <= probability <= 1.0


def test_saved_model_can_be_loaded(tmp_path) -> None:
    model = create_model()
    model.fit([[70000, 120000, 2, 18, 0.35, 0], [90000, 150000, 5, 36, 0.25, 0]], [0, 1])
    save_path = tmp_path / "loan_model.joblib"
    save_model(model, save_path)
    loaded = load_model(save_path)
    assert isinstance(loaded, LogisticRegression)


def test_feature_order_remains_consistent() -> None:
    assert FEATURES == [
        "annual_income",
        "loan_amount",
        "employment_years",
        "credit_history_months",
        "debt_to_income",
        "previous_defaults",
    ]
