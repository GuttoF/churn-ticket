import pytest
import pandas as pd
from src.api.services import make_prediction, load_model, load_threshold


@pytest.fixture
def input_data():
    return pd.DataFrame([{
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
    }])


def test_make_prediction(input_data):
    # Load the model and threshold
    model = load_model()
    threshold = load_threshold()

    # Make the prediction
    prediction, prediction_proba = make_prediction(model, threshold, input_data)

    # Verify if the prediction is correct
    assert prediction == 0
    assert 0 <= prediction_proba <= 1  # The probability should be between 0 and 1
