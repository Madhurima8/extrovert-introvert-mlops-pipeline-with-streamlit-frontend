from fastapi.testclient import TestClient
from src.api import app


client = TestClient(app)


def test_health_endpoint():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}

def test_predict_with_valid_input():
    payload = {
        "Time_spent_Alone": 5,
        "Stage_fear": "Yes",
        "Social_event_attendance": 4,
        "Going_outside": 3,
        "Drained_after_socializing": "Yes",
        "Friends_circle_size": 6,
        "Post_frequency": 3
    }

    response = client.post("/predict", json=payload)
    # print(response.json())

    assert response.status_code == 200

    data = response.json()

    assert "model_name" in data
    assert "predicted_label" in data
    assert "predicted_class" in data
    assert "predicted_proba" in data

    assert data["predicted_label"] in ["introvert", "extrovert"]
    assert data["predicted_class"] in [0, 1]
    assert 0.0 <= data["predicted_proba"] <= 1.0

def test_predict_rejects_out_of_range_input():
    payload = {
        "Time_spent_Alone": 20,
        "Stage_fear": "Yes",
        "Social_event_attendance": 4,
        "Going_outside": 3,
        "Drained_after_socializing": "Yes",
        "Friends_circle_size": 6,
        "Post_frequency": 3
    }

    response = client.post("/predict", json=payload)

    assert response.status_code == 422

def test_predict_rejects_missing_field():
    payload = {
        "Time_spent_Alone": 5,
        "Stage_fear": "Yes",
        "Social_event_attendance": 4,
        "Going_outside": 3,
        "Drained_after_socializing": "Yes",
        "Friends_circle_size": 6
    }

    response = client.post("/predict", json=payload)

    assert response.status_code == 422

def test_predict_rejects_invalid_categorical_input():
    payload = {
        "Time_spent_Alone": 5,
        "Stage_fear": "Maybe",
        "Social_event_attendance": 4,
        "Going_outside": 3,
        "Drained_after_socializing": "Yes",
        "Friends_circle_size": 6,
        "Post_frequency": 3
    }

    response = client.post("/predict", json=payload)

    assert response.status_code == 422

def test_predict_accepts_case_insensitive_categorical_input():
    payload = {
        "Time_spent_Alone": 5,
        "Stage_fear": "YES",
        "Social_event_attendance": 4,
        "Going_outside": 3,
        "Drained_after_socializing": "no",
        "Friends_circle_size": 6,
        "Post_frequency": 3
    }

    response = client.post("/predict", json=payload)

    assert response.status_code == 200