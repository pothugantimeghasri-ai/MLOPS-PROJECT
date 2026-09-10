from pathlib import Path

import pandas as pd
import pytest

from src.data_processing import (
    REQIRED_COLUMNS,
    check_missing_values,
    ensure_numeric_columns,
    validate_business_rules,
    validate_required_columns,
)


@pytest.fixture
def valid_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "annual_income": [75000, 95000],
            "loan_amount": [150000, 160000],
            "employment_years": [2, 4],
            "credit_history_months": [18, 24],
            "debt_to_income": [0.35, 0.28],
            "previous_defaults": [0, 0],
            "approved": [0, 1],
        }
    )


def test_required_columns_exist(valid_df: pd.DataFrame) -> None:
    validate_required_columns(valid_df)
    assert all(col in valid_df.columns for col in [
        "annual_income",
        "loan_amount",
        "employment_years",
        "credit_history_months",
        "debt_to_income",
        "previous_defaults",
        "approved",
    ])


def test_missing_values_are_detected() -> None:
    df = pd.DataFrame(
        {
            "annual_income": [75000, None],
            "loan_amount": [150000, 160000],
            "employment_years": [2, 4],
            "credit_history_months": [18, 24],
            "debt_to_income": [0.35, 0.28],
            "previous_defaults": [0, 0],
            "approved": [0, 1],
        }
    )
    with pytest.raises(ValueError):
        check_missing_values(df)


def test_invalid_approved_values_are_rejected() -> None:
    df = pd.DataFrame(
        {
            "annual_income": [75000, 95000],
            "loan_amount": [150000, 160000],
            "employment_years": [2, 4],
            "credit_history_months": [18, 24],
            "debt_to_income": [0.35, 0.28],
            "previous_defaults": [0, 0],
            "approved": [0, 2],
        }
    )
    with pytest.raises(ValueError):
        validate_business_rules(df)


def test_numeric_conversion_and_validation(valid_df: pd.DataFrame) -> None:
    converted = ensure_numeric_columns(valid_df.copy())
    for col in [
        "annual_income",
        "loan_amount",
        "employment_years",
        "credit_history_months",
        "debt_to_income",
        "previous_defaults",
        "approved",
    ]:
        assert pd.api.types.is_numeric_dtype(converted[col])

    validate_business_rules(converted)
