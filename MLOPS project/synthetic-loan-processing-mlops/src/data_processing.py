from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_PATH = DATA_DIR / "synthetic_loans.csv"
PROCESSED_DATA_PATH = DATA_DIR / "processed_loans.csv"

REQUIRED_COLUMNS = [
    "annual_income",
    "loan_amount",
    "employment_years",
    "credit_history_months",
    "debt_to_income",
    "previous_defaults",
    "approved",
]

# Backward-compatible alias used by tests and user-facing validation helpers.
REQIRED_COLUMNS = REQUIRED_COLUMNS

NUMERIC_COLUMNS = [
    "annual_income",
    "loan_amount",
    "employment_years",
    "credit_history_months",
    "debt_to_income",
    "previous_defaults",
    "approved",
]


def validate_required_columns(df: pd.DataFrame) -> None:
    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def check_missing_values(df: pd.DataFrame) -> None:
    missing = df.isna().sum()
    if missing.any():
        raise ValueError(f"Missing values found:\n{missing[missing > 0].to_string()}")


def ensure_numeric_columns(df: pd.DataFrame) -> pd.DataFrame:
    for column in NUMERIC_COLUMNS:
        if column not in df.columns:
            raise ValueError(f"Column '{column}' is missing.")
        try:
            df[column] = pd.to_numeric(df[column], errors="raise")
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Column '{column}' contains invalid numeric values.") from exc
    return df


def validate_business_rules(df: pd.DataFrame) -> None:
    if (df["annual_income"] <= 0).any():
        raise ValueError("annual_income must be positive for all rows.")
    if (df["loan_amount"] <= 0).any():
        raise ValueError("loan_amount must be positive for all rows.")
    if ((df["debt_to_income"] < 0) | (df["debt_to_income"] > 1)).any():
        raise ValueError("debt_to_income must be between 0 and 1 inclusive.")
    if not df["approved"].isin([0, 1]).all():
        raise ValueError("approved column must contain only 0 or 1 values.")


def load_dataset(path: Path | str) -> pd.DataFrame:
    dataset_path = Path(path)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset file not found: {dataset_path}")
    return pd.read_csv(dataset_path)


def preprocess_data(raw_path: Path | str = RAW_DATA_PATH, output_path: Path | str = PROCESSED_DATA_PATH) -> pd.DataFrame:
    df = load_dataset(raw_path)
    validate_required_columns(df)
    check_missing_values(df)
    df = ensure_numeric_columns(df)
    validate_business_rules(df)

    processed_path = Path(output_path)
    processed_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(processed_path, index=False)

    print(f"Row count: {len(df)}")
    print(f"Columns: {list(df.columns)}")
    print("Missing values per column:")
    print(df.isna().sum().to_string())
    print("Class counts:")
    print(df["approved"].value_counts().sort_index().to_string())
    print(f"Saved cleaned data to: {processed_path}")
    print("Data preprocessing completed successfully.")
    return df


def main() -> None:
    preprocess_data()


if __name__ == "__main__":
    main()
