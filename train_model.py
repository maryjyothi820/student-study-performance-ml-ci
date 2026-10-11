
import json

import joblib
import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def create_dataset():
    rng = np.random.default_rng(42)
    n = 300

    data = pd.DataFrame({
        "study_hours": rng.uniform(0, 10, n).round(1),
        "attendance": rng.integers(50, 101, n),
        "assignments_completed": rng.integers(0, 11, n),
        "sleep_hours": rng.uniform(4, 10, n).round(1),
        "social_media_hours": rng.uniform(0, 8, n).round(1),
        "previous_score": rng.integers(30, 101, n)
    })

    score = (
        5 * data["study_hours"]
        + 0.25 * data["attendance"]
        + 2 * data["assignments_completed"]
        + 0.5 * data["sleep_hours"]
        - 1.5 * data["social_media_hours"]
        + 0.25 * data["previous_score"]
    )

    data["result"] = np.where(score >= 45, "Pass", "Fail")
    return data


def train_model():
    print("Creating dataset...")
    data = create_dataset()
    data.to_csv("student_results.csv", index=False)

    print("Dataset created successfully.")
    print("Number of records:", len(data))

    features = [
        "study_hours",
        "attendance",
        "assignments_completed",
        "sleep_hours",
        "social_media_hours",
        "previous_score"
    ]

    X = data[features]
    y = data["result"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    print("Training records:", len(X_train))
    print("Testing records :", len(X_test))

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(max_iter=1000))
    ])

    print("Training model...")
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    matrix = confusion_matrix(
        y_test, predictions, labels=["Fail", "Pass"]
    )

    print("\nModel Evaluation")
    print("----------------")
    print("Accuracy:", round(accuracy, 4))
    print("Confusion Matrix (Fail, Pass):")
    print(matrix)

    joblib.dump(model, "student_result_model.pkl")
    print("Model saved as student_result_model.pkl")

    metrics = {
        "accuracy": float(accuracy),
        "training_records": len(X_train),
        "testing_records": len(X_test),
        "features": features,
        "classes": ["Fail", "Pass"]
    }

    with open("metrics.json", "w") as file:
        json.dump(metrics, file, indent=4)

    print("Metrics saved as metrics.json")


if __name__ == "__main__":
    train_model()
