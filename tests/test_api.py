from fastapi.testclient import TestClient

from src.api.main import app


client = TestClient(app)


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_attack_types():
    response = client.get("/api/v1/attacks/types")
    assert response.status_code == 200
    data = response.json()
    assert "label_flipping" in data["attacks"]
    assert "backdoor" in data["attacks"]


def test_secure_predict():
    response = client.post(
        "/api/v1/predict",
        json={"amount": 950, "merchant_risk": 0.75},
    )
    assert response.status_code == 200
    data = response.json()
    assert 0.0 <= data["fraud_probability"] <= 1.0
    assert "controls_active" in data
