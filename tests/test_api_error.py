import pytest
from httpx import AsyncClient, ASGITransport
from src.api.main import app


@pytest.mark.asyncio
async def test_predict():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        test_data = {
            "credit_score": 600,
            "geography": "Germany",
            "gender": "Male",
            "age": 30,
            "tenure": 5,
            "balance": 50000.0,
            "num_of_products": 2,
            "has_cr_card": 1,
            "is_active_member": 1,
            "estimated_salary": 70000.0
        }

        response = await ac.post("/predict", json=test_data)
        assert response.status_code == 200
        assert "prediction" in response.json()
        assert "probability" in response.json()
        assert response.json()["prediction"] == 0


@pytest.mark.asyncio
async def test_predict_invalid_data():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        test_data = {
            "credit_score": "invalid",  # Invalid value (should be a float)
            "geography": "Germany",
            "gender": "Male",
            "age": 30,
            "tenure": 5,
            "balance": 50000.0,
            "num_of_products": 2,
            "has_cr_card": 1,
            "is_active_member": 1,
            "estimated_salary": 70000.0
        }
        response = await ac.post("/predict", json=test_data)
        assert response.status_code == 422

