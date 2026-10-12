
from flask import Flask, request, jsonify
import joblib
import pandas as pd
from pathlib import Path

app = Flask(__name__)

# Load the trained student result model
MODEL_PATH = Path(__file__).parent / "student_result_model.pkl"
model = joblib.load(MODEL_PATH)

# Input features from student_results.csv
FEATURES = [
    "study_hours",
    "attendance",
    "assignments_completed",
    "sleep_hours",
    "social_media_hours",
    "previous_score",
]


# Home endpoint
@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "message": "Student Performance Prediction API is running"
    })


# Health endpoint for Docker and GitHub Actions
@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "healthy"
    }), 200


# Student result prediction endpoint
@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(silent=True)

    if not isinstance(data, dict):
        return jsonify({
            "error": "Send a JSON object"
        }), 400

    # Check for missing features
    missing = [feature for feature in FEATURES if feature not in data]

    if missing:
        return jsonify({
            "error": "Missing required features",
            "missing_features": missing
        }), 400

    # Validate numeric inputs
    try:
        values = {
            feature: float(data[feature])
            for feature in FEATURES
        }
    except (TypeError, ValueError):
        return jsonify({
            "error": "All features must be numeric"
        }), 400

    # Prepare input and predict
    input_data = pd.DataFrame([values], columns=FEATURES)
    prediction = model.predict(input_data)[0]

    return jsonify({
        "prediction": str(prediction)
    }), 200


# Start Flask so the Docker container can be reached
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
