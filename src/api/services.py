import pickle
import pandas as pd
from .feature_engineering import perform_transformations

def load_model():
    with open('src/models/model.pkl', 'rb') as f:
        model = pickle.load(f)
    return model

def load_threshold():
    with open('src/models/threshold.pkl', 'rb') as f:
        threshold = pickle.load(f)
    return threshold

def make_prediction(model, threshold, input_data: pd.DataFrame):
    # Apply feature engineering transformations
    transformed_data = perform_transformations(input_data)

    # Guarantee that the columns are in the correct order and add missing columns with value 0
    correct_column_order = [
        'credit_score', 'age', 'tenure', 'balance', 'num_of_products', 'has_cr_card',
        'is_active_member', 'estimated_salary', 'age_squared', 'balance_sqrt',
        'credit_score_num_of_products', 'age_balance', 'engagement_score',
        'customer_value_normalized', 'product_density', 'balance_salary_ratio',
        'credit_score_age_ratio', 'tenure_age_ratio', 'credit_salary_ratio',
        'credit_score_gender', 'balance_age', 'salary_rank_geography',
        'geography_germany', 'geography_spain', 'gender_male',
        'tenure_group_long_standing', 'tenure_group_new', 'life_stage_adulthood',
        'life_stage_middle_age', 'life_stage_senior', 'balance_indicator_low',
        'cs_category_low', 'cs_category_medium'
    ]

    # Add missing columns with value 0 to deal with NA values
    # Add an if any NA values are present, print a warning message
    if transformed_data.isnull().values.any():
        print("Warning: NA values are present in the input data. Replacing with 0.")
        transformed_data.fillna(0, inplace=True)

    # Reorganize the columns according to the correct order
    transformed_data = transformed_data[correct_column_order]

    # Make the prediction
    prediction_proba = model.predict_proba(transformed_data)[:, 1]

    # Apply the threshold to determine the final class
    prediction = (prediction_proba >= threshold).astype(int)

    return int(prediction[0]), float(prediction_proba[0])
