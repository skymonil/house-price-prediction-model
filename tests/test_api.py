from fastapi.testclient import TestClient

from src.api import app


class FakeModel:

    def predict(self, data):
        return [10_000_000.0]
    
app.state.testing = True

def test_health():
    app.state.model = FakeModel()

    with TestClient(app) as client:
        response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_prediction():
    app.state.model = FakeModel()

    with TestClient(app) as client:
        response = client.post(
            "/predict",
            json={
                "bedrooms": 3,
                "area": 1200,
                "location": "Mumbai",
                "age": 5,
            },
        )

    assert response.status_code == 200

    result = response.json()

    assert "predicted_price" in result
    assert result["predicted_price"] == 10_000_000.0
