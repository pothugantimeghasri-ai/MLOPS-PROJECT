from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split

from model import FEATURES, load_model

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DATA_PATH = PROJECT_ROOT / "data" / "processed_loans.csv"
MODEL_PATH = PROJECT_ROOT / "models" / "loan_model.joblib"
REPORT_PATH = PROJECT_ROOT / "reports" / "evaluation.json"


def evaluate_model() -> dict[str, float | list[list[int]]]:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found at {MODEL_PATH}. Train the model first.")
    if not PROCESSED_DATA_PATH.exists():
        raise FileNotFoundError(f"Processed data not found at {PROCESSED_DATA_PATH}.")

    df = pd.read_csv(PROCESSED_DATA_PATH)
    X = df[FEATURES]
    y = df["approved"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    model = load_model(MODEL_PATH)
    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions, zero_division=0)
    recall = recall_score(y_test, predictions, zero_division=0)
    f1 = f1_score(y_test, predictions, zero_division=0)
    matrix = confusion_matrix(y_test, predictions).tolist()

    metrics = {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1),
        "confusion_matrix": matrix,
        "notes": "This is an educational synthetic dataset. Small sample sizes should be interpreted carefully and are not used for real lending decisions.",
    }

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(REPORT_PATH, "w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2)

    print("Evaluation metrics:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 score: {f1:.4f}")
    print("Confusion matrix:")
    print(matrix)
    print(f"Saved evaluation report to: {REPORT_PATH}")
    return metrics


def main() -> None:
    evaluate_model()


if __name__ == "__main__":
    main()
