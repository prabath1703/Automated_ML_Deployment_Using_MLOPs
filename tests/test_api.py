from fastapi.testclient import TestClient
from src.api.app import app

client = TestClient(app)

def test_health_check():
    response = client.get("/")
    assert response.status_code == 200

def test_prediction_success():
    payload = {
        "tenure": 12,
        "MonthlyCharges": 70,
        "TotalCharges": 840
    }

    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    assert "churn_prediction" in response.json()
    assert response.json()["churn_prediction"] in [0, 1]

