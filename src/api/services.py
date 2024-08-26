import pickle
from pathlib import Path

import pandas as pd

from .feature_engineering import FeatureEngineering as fe

model = None
threshold = None

# Paths
current_path = Path(__file__)
model_path = current_path.parents[2] / "src" / "models" / "model.pkl"
threshold_path = current_path.parents[2] / "src" / "models" / "threshold.pkl"


def load_model():
    global model
    if model is None:
        with open(model_path, "rb") as model_file:
            model = pickle.load(model_file)
    return model


def load_threshold():
    global threshold
    if threshold is None:
        with open(threshold_path, "rb") as threshold_file:
            threshold = pickle.load(threshold_file)
    return threshold


def make_prediction(model, threshold, input_data: pd.DataFrame):
    # Apply feature engineering transformations
    transformed_data = fe.transform_data_inference(input_data)

    # Guarantee that the columns are in the correct order and add missing columns with value 0
    correct_column_order = [
        "credit_score",
        "age",
        "tenure",
        "balance",
        "num_of_products",
        "has_cr_card",
        "is_active_member",
        "estimated_salary",
        "age_squared",
        "balance_sqrt",
        "credit_score_num_of_products",
        "age_balance",
        "engagement_score",
        "customer_value_normalized",
        "product_density",
        "balance_salary_ratio",
        "credit_score_age_ratio",
        "tenure_age_ratio",
        "credit_salary_ratio",
        "credit_score_gender",
        "balance_age",
        "salary_rank_geography",
        "geography_germany",
        "geography_spain",
        "gender_male",
        "tenure_group_long_standing",
        "tenure_group_new",
        "life_stage_adulthood",
        "life_stage_middle_age",
        "life_stage_senior",
        "balance_indicator_low",
        "cs_category_low",
        "cs_category_medium",
    ]

    # Add missing columns with value 0 to deal with NA values
    if transformed_data.isnull().values.any():
        transformed_data.fillna(0, inplace=True)

    # Reorganize the columns according to the correct order
    transformed_data = transformed_data[correct_column_order]

    # Make the prediction
    prediction_proba = model.predict_proba(transformed_data)[:, 1]

    # Apply the threshold to determine the final class
    prediction = (prediction_proba >= threshold).astype(int)

    return int(prediction[0]), float(prediction_proba[0])
