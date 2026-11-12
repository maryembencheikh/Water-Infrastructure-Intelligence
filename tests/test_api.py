from fastapi.testclient import TestClient
from src.api.main import app

client = TestClient(app)


def test_health_check():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "healthy"
    assert response.json()["model_loaded"] is True


def test_prediction():
    response = client.post(
        "/predict",
        json={"id": 12345}
    )

    assert response.status_code == 200

    data = response.json()

    assert data["id"] == 12345
    assert data["prediction"] in [
        "functional",
        "functional needs repair",
        "non functional"
    ]
    assert abs(sum(data["probabilities"].values()) - 1) < 1e-6


def test_unknown_field():
    response = client.post(
        "/predict",
        json={"id": 12345, "unknown_field": "test"}
    )

    assert response.status_code == 422


def test_invalid_numeric_input():
    response = client.post(
        "/predict",
        json={"id": 12345, "amount_tsh": "invalid"}
    )

    assert response.status_code == 422