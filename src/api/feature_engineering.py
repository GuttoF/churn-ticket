import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler


def perform_transformations(dataframe: pd.DataFrame) -> pd.DataFrame:
    """
    Perform feature engineering transformations on the given dataframe.

    Parameters:
    - dataframe (pandas.DataFrame): Input dataframe to be transformed.

    Returns:
    - dataframe (pandas.DataFrame): Transformed dataframe.
    """
    # Non-linear transformations
    dataframe["age_squared"] = dataframe["age"] ** 2
    dataframe["balance_sqrt"] = np.sqrt(dataframe["balance"])

    # Multiplicative interactions
    dataframe["credit_score_num_of_products"] = (
            dataframe["credit_score"] * dataframe["num_of_products"]
    )
    dataframe["age_balance"] = dataframe["age"] * dataframe["balance"]
    dataframe["engagement_score"] = (
            (dataframe["is_active_member"] * 0.5)
            + (dataframe["has_cr_card"] * 0.3)
            + (dataframe["num_of_products"] * 0.2)
    )

    # Division interactions
    epsilon = 1e-6
    dataframe["customer_value_normalized"] = (
            dataframe["balance"] / dataframe["balance"].max()
            + dataframe["num_of_products"] / dataframe["num_of_products"].max()
            + dataframe["estimated_salary"] / dataframe["estimated_salary"].max()
    )
    dataframe["product_density"] = dataframe["num_of_products"] / (
            dataframe["balance"] + epsilon
    )
    dataframe["balance_salary_ratio"] = (
            dataframe["balance"] / dataframe["estimated_salary"]
    )
    dataframe["credit_score_age_ratio"] = (
            dataframe["credit_score"] / dataframe["age"]
    )
    dataframe["tenure_age_ratio"] = dataframe["tenure"] / dataframe["age"]
    dataframe["credit_salary_ratio"] = (
            dataframe["credit_score"] / dataframe["estimated_salary"]
    )

    # Manual One-Hot Encoding for 'geography'
    dataframe["geography_germany"] = (dataframe["geography"] == "Germany").astype(int)
    dataframe["geography_spain"] = (dataframe["geography"] == "Spain").astype(int)

    # Group by transformations (if applicable)
    if "gender" in dataframe.columns:
        dataframe["credit_score_gender"] = dataframe.groupby("gender")[
            "credit_score"
        ].transform("mean")
        dataframe["gender_male"] = (dataframe["gender"] == "Male").astype(int)

    if "age" in dataframe.columns:
        dataframe["balance_age"] = dataframe.groupby("age")["balance"].transform("mean")

    if "geography" in dataframe.columns:
        dataframe["salary_rank_geography"] = dataframe.groupby("geography")[
            "estimated_salary"
        ].rank(method="dense")

    # Indicator transformations
    dataframe["balance_indicator_low"] = np.where(
        dataframe["balance"] < 150000, 1, 0
    )

    # Binning transformations with unique labels
    dataframe["life_stage_adulthood"] = (dataframe["age"] >= 20) & (dataframe["age"] < 35)
    dataframe["life_stage_middle_age"] = (dataframe["age"] >= 35) & (dataframe["age"] < 50)
    dataframe["life_stage_senior"] = dataframe["age"] >= 50

    dataframe["tenure_group_long_standing"] = dataframe["tenure"] > 6
    dataframe["tenure_group_new"] = dataframe["tenure"] <= 3

    dataframe["cs_category_low"] = dataframe["credit_score"] <= 500
    dataframe["cs_category_medium"] = (dataframe["credit_score"] > 500) & (dataframe["credit_score"] <= 700)

    # Convert boolean to int
    dataframe["life_stage_adulthood"] = dataframe["life_stage_adulthood"].astype(int)
    dataframe["life_stage_middle_age"] = dataframe["life_stage_middle_age"].astype(int)
    dataframe["life_stage_senior"] = dataframe["life_stage_senior"].astype(int)
    dataframe["tenure_group_long_standing"] = dataframe["tenure_group_long_standing"].astype(int)
    dataframe["tenure_group_new"] = dataframe["tenure_group_new"].astype(int)
    dataframe["cs_category_low"] = dataframe["cs_category_low"].astype(int)
    dataframe["cs_category_medium"] = dataframe["cs_category_medium"].astype(int)

    return dataframe


def apply_log_transformation(data: pd.DataFrame, columns: list) -> None:
    """
    Applies logarithmic transformation to specified columns in the list of datasets.

    Parameters:
    - data (list of pd.DataFrame): List of DataFrames to apply the transformation.
    - columns (list of str): List of column names to which the logarithmic transformation will be applied.
    """
    for df in data:
        for col in columns:
            df[col] = np.log1p(df[col])
    return None


def apply_scaler(data: list, columns: list, scaler_path: str):
    """
    Applies scaling to specified columns in the list of datasets using the provided scaler.

    Parameters:
    - data (list of pd.DataFrame): List of DataFrames to apply the scaling.
    - columns (list of str): List of column names to which scaling will be applied.
    - scaler_path (str): Path to the saved scaler model.
    """
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)

    for df in data:
        df[columns] = scaler.transform(df[columns])
    return None


# Application of feature engineering, transformations, and scaling during inference
def transform_data_inference(X):
    """
    Transforms input data by applying feature engineering, logarithmic transformations,
    and scaling. This is done in the order required for model inference.

    Parameters:
    - X (pd.DataFrame): The input dataframe to be transformed.

    Returns:
    - X (pd.DataFrame): The transformed dataframe ready for model inference.
    """
    # First, apply feature engineering transformations
    X = perform_transformations(X)

    # Then, apply log transformations and scaling
    log_cols = ["balance", "balance_sqrt", "age_balance", "age", "age_squared"]
    standard_scaler_cols = ["credit_score_num_of_products", "engagement_score"]
    min_max_scaler_cols = ["credit_score", "tenure"]
    robust_scaler_cols = [
        "age", "balance", "estimated_salary", "age_squared", "balance_sqrt", "age_balance",
        "product_density", "balance_salary_ratio", "credit_score_age_ratio", "tenure_age_ratio", "credit_salary_ratio"
    ]

    apply_log_transformation([X], log_cols)
    apply_scaler([X], standard_scaler_cols, 'src/scalers/standard_scaler.pkl')
    apply_scaler([X], min_max_scaler_cols, 'src/scalers/min_max_scaler.pkl')
    apply_scaler([X], robust_scaler_cols, 'src/scalers/robust_scaler.pkl')

    return X

