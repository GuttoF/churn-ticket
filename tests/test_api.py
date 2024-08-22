import pytest
from httpx import AsyncClient, ASGITransport
from src.api.main import app


@pytest.mark.asyncio
async def test_predict():
    # Use ASGITransport to wrap the FastAPI app
    transport = ASGITransport(app=app)

    # AsyncClient to simulate an HTTP request to the API
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        # Input data for the test
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

        # Make a POST request to the /predict endpoint
        response = await ac.post("/predict", json=test_data)

        # Check if the status code is 200
        assert response.status_code == 200

        # Check if the response contains the "prediction" key
        assert "prediction" in response.json()

        # Check if the response contains the "probability" key
        assert "probability" in response.json()

        # Verify the expected prediction value
        assert response.json()["prediction"] == 0
