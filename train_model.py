import json

import joblib
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


DATA_FILE = "student_results.csv"


def load_dataset():
    """Load the existing student dataset without replacing it."""
    print(f"Loading existing dataset: {DATA_FILE}")
    data = pd.read_csv(DATA_FILE)

    required_columns = [
        "study_hours",
        "attendance",
        "assignments_completed",
        "sleep_hours",
        "social_media_hours",
        "previous_score",
        "result",
    ]

    missing_columns = [column for column in required_columns if column not in data.columns]
    if missing_columns:
        raise ValueError(
            "Dataset is missing required columns: " + ", ".join(missing_columns)
        )

    if data.empty:
        raise ValueError(f"{DATA_FILE} has no data rows.")

    data = data.dropna(subset=required_columns)

    print("Dataset loaded successfully.")
    print("Number of records:", len(data))
    print("Result counts:")
    print(data["result"].value_counts().to_string())

    return data


def train_model():
    data = load_dataset()

    features = [
        "study_hours",
        "attendance",
        "assignments_completed",
        "sleep_hours",
        "social_media_hours",
        "previous_score",
    ]

    X = data[features]
    y = data["result"].astype(str)

    if y.nunique() < 2:
        raise ValueError("The result column must contain both Pass and Fail classes.")

    # Stratification needs at least two examples in each class.
    class_counts = y.value_counts()
    if class_counts.min() < 2:
        raise ValueError(
            "Each result class needs at least 2 rows for a stratified train/test split."
        )

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )

    print("Training records:", len(X_train))
    print("Testing records :", len(X_test))

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(max_iter=1000)),
    ])

    print("Training model...")
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)

    # Use the classes actually present in the dataset for the matrix.
    labels = sorted(y.unique())
    matrix = confusion_matrix(y_test, predictions, labels=labels)

    print("\nModel Evaluation")
    print("----------------")
    print("Accuracy:", round(accuracy, 4))
    print(f"Confusion Matrix ({', '.join(labels)}):")
    print(matrix)

    model_file = "student_result_model.pkl"
    joblib.dump(model, model_file)
    print(f"Model saved as {model_file}")

    metrics = {
        "accuracy": float(accuracy),
        "training_records": int(len(X_train)),
        "testing_records": int(len(X_test)),
        "features": features,
        "classes": labels,
    }

    with open("metrics.json", "w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=4)

    print("Metrics saved as metrics.json")


if __name__ == "__main__":
    train_model()
