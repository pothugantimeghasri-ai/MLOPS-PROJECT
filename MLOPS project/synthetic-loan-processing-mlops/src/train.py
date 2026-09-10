from __future__ import annotations

import json
from pathlib import Path

from sklearn.model_selection import train_test_split

from data_processing import PROCESSED_DATA_PATH, preprocess_data
from model import FEATURES, save_model, train_model

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "models" / "loan_model.joblib"
METADATA_PATH = PROJECT_ROOT / "models" / "loan_model_metadata.json"


def train_pipeline() -> None:
    print("Running preprocessing...")
    preprocess_data()

    print("Loading processed dataset...")
    import pandas as pd

    df = pd.read_csv(PROCESSED_DATA_PATH)
    X = df[FEATURES]
    y = df["approved"]

    print("Splitting data into train and test sets...")
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    print("Training logistic regression model...")
    model = train_model(X_train, y_train)

    print(f"Saving model to: {MODEL_PATH}")
    save_model(model, MODEL_PATH, METADATA_PATH)

    metadata = {
        "model_version": "1.0.0",
        "features": FEATURES,
        "model_type": "LogisticRegression",
        "train_shape": list(X_train.shape),
        "test_shape": list(X_test.shape),
    }
    with open(METADATA_PATH, "w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2)

    print(f"Saved metadata to: {METADATA_PATH}")
    print("Model training completed successfully.")


def main() -> None:
    train_pipeline()


if __name__ == "__main__":
    main()
