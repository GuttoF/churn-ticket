from fastapi import APIRouter
from pydantic import BaseModel
from src.api.services import load_model, load_threshold, make_prediction
import pandas as pd

router = APIRouter()

# Pydantic BaseModel
class PredictionInput(BaseModel):
    credit_score: float
    geography: str
    gender: str
    age: int
    tenure: float
    balance: float
    num_of_products: int
    has_cr_card: int
    is_active_member: int
    estimated_salary: float

# Load threshold and begin API
model = load_model()
threshold = load_threshold()

@router.post("/predict")
def predict(input_data: PredictionInput):
    try:
        # Convert input data to DataFrame
        input_df = pd.DataFrame([input_data.model_dump()])

        # Make prediction
        prediction, prediction_proba = make_prediction(model, threshold, input_df)

        return {"prediction": prediction, "probability": prediction_proba}
    except Exception as e:
        return {"error": str(e)}