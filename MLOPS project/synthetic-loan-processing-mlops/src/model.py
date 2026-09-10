from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression

FEATURES = [
    "annual_income",
    "loan_amount",
    "employment_years",
    "credit_history_months",
    "debt_to_income",
    "previous_defaults",
]

MODEL_VERSION = "1.0.0"


def create_model() -> LogisticRegression:
    return LogisticRegression(max_iter=1000, random_state=42)


def train_model(X_train, y_train) -> LogisticRegression:
    model = create_model()
    model.fit(X_train, y_train)
    return model


def save_model(model: LogisticRegression, path: str | Path, metadata_path: str | Path | None = None) -> None:
    model_path = Path(path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path)

    metadata = {
        "model_version": MODEL_VERSION,
        "features": FEATURES,
        "model_type": "LogisticRegression",
    }

    output_metadata_path = Path(metadata_path) if metadata_path is not None else model_path.with_suffix(".json")
    output_metadata_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_metadata_path, "w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2)


def load_model(path: str | Path) -> LogisticRegression:
    model_path = Path(path)
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    return joblib.load(model_path)


def predict_application(model: LogisticRegression, application: dict[str, Any]) -> tuple[int, float]:
    ordered_values = [application[feature] for feature in FEATURES]
    feature_frame = pd.DataFrame([ordered_values], columns=FEATURES)
    probability = model.predict_proba(feature_frame)[0]
    prediction = int(model.predict(feature_frame)[0])
    positive_probability = float(probability[1]) if len(probability) > 1 else float(probability[0])
    return prediction, positive_probability
