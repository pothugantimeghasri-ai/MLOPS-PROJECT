from __future__ import annotations

import json
from pathlib import Path

from model import FEATURES, load_model, predict_application

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "models" / "loan_model.joblib"
METADATA_PATH = PROJECT_ROOT / "models" / "loan_model_metadata.json"


def predict_example() -> None:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model file not found: {MODEL_PATH}. Train the model first.")
    if not METADATA_PATH.exists():
        raise FileNotFoundError(f"Model metadata not found: {METADATA_PATH}.")

    with open(METADATA_PATH, "r", encoding="utf-8") as file:
        metadata = json.load(file)

    application = {
        "annual_income": 75000,
        "loan_amount": 200000,
        "employment_years": 6,
        "credit_history_months": 60,
        "debt_to_income": 0.30,
        "previous_defaults": 0,
    }

    model = load_model(MODEL_PATH)
    prediction, probability = predict_application(model, application)

    print("Educational-only prediction result")
    print(f"Model version: {metadata.get('model_version', 'unknown')}")
    print(f"Features used: {metadata.get('features', FEATURES)}")
    print(f"Input application: {application}")
    print(f"Predicted class: {prediction}")
    print(f"Probability of approval: {probability:.4f}")
    print("This project is for educational purposes only and must not be used for real lending decisions.")


def main() -> None:
    predict_example()


if __name__ == "__main__":
    main()
