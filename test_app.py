
import pytest
from app import app


@pytest.fixture
def client():
    app.config["TESTING"] = True
    with app.test_client() as test_client:
        yield test_client


def test_home_endpoint(client):
    response = client.get("/")
    assert response.status_code == 200
    assert b"Student Performance Prediction API is running" in response.data


def test_predict_endpoint(client):
    response = client.post(
        "/predict",
        json={
            "study_hours": 4,
            "attendance": 85,
            "assignments_completed": 8,
            "sleep_hours": 7,
            "social_media_hours": 2,
            "previous_score": 75
        }
    )

    assert response.status_code == 200
    assert "prediction" in response.get_json()
    assert response.get_json()["prediction"] in ["Pass", "Fail"]


def test_missing_features(client):
    response = client.post(
        "/predict",
        json={"study_hours": 4}
    )

    assert response.status_code == 400
    assert "Missing required features" in response.get_json()["error"]


def test_invalid_numeric_value(client):
    response = client.post(
        "/predict",
        json={
            "study_hours": "abc",
            "attendance": 85,
            "assignments_completed": 8,
            "sleep_hours": 7,
            "social_media_hours": 2,
            "previous_score": 75
        }
    )

    assert response.status_code == 400
    assert "All features must be numeric" in response.get_json()["error"]
