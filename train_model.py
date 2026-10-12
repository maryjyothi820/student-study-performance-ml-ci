```python
import json
import joblib
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def load_dataset():
    print("Loading student performance dataset...")

    data = pd.read_csv("student_results.csv")

    required_columns = [
        "study_hours",
        "attendance",
        "assignments_completed",
        "sleep_hours",
        "social_media_hours",
        "previous_score",
        "result"
    ]

    missing = [
        col for col in required_columns
        if col not in data.columns
    ]

    if missing:
        raise ValueError(f"Missing columns: {missing}")

    data = data[required_columns].dropna()

    if len(data) < 10:
        raise ValueError("Dataset must contain at least 10 valid rows.")

    print("Dataset loaded successfully.")
    print("Number of records:", len(data))
    print(data["result"].value_counts())

    return data


def train_model():
    data = load_dataset()

    features = [
        "study_hours",
        "attendance",
        "assignments_completed",
        "sleep_hours",
        "social_media_hours",
        "previous_score"
    ]

    X = data[features]
    y = data["result"].astype(str).str.strip()

    if y.nunique() < 2:
        raise ValueError("The result column must contain Pass and Fail classes.")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(max_iter=1000))
    ])

    print("Training model...")
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)

    matrix = confusion_matrix(
        y_test,
        predictions,
        labels=["Fail", "Pass"]
    )

    print("\nModel Evaluation")
    print("----------------")
    print("Accuracy:", round(accuracy, 4))
    print("Confusion Matrix (Fail, Pass):")
    print(matrix)

    joblib.dump(model, "student_result_model.pkl")
    print("Model saved.")

    metrics = {
        "accuracy": float(accuracy),
        "training_records": int(len(X_train)),
        "testing_records": int(len(X_test)),
        "features": features,
        "classes": sorted(y.unique().tolist())
    }

    with open("metrics.json", "w") as file:
        json.dump(metrics, file, indent=4)

    print("Metrics saved to metrics.json")


if __name__ == "__main__":
    train_model()
```
