import pickle

import numpy as np
import pandas as pd


class FeatureEngineering:
    """
    Class for performing feature engineering on a given dataframe.
    """

    def __init__(self):
        pass

    @staticmethod
    def perform_transformations(dataframe: pd.DataFrame) -> pd.DataFrame:
        """
        Perform feature transformations on the given dataframe.

        Parameters:
        - dataframe (pandas.DataFrame): Input dataframe to be transformed.

        Returns:
        - dataframe (pandas.DataFrame): Transformed dataframe.
        """
        # Define a small epsilon to avoid division by zero and log(0)
        epsilon = 1e-6

        # Conditions to zero handling
        dataframe["num_of_products"] = dataframe["num_of_products"].apply(
            lambda x: x + 1 if x == 0 else x
        )
        dataframe["credit_score"] = dataframe["credit_score"].apply(
            lambda x: x + 1 if x == 0 else x
        )
        dataframe["balance"] = dataframe["balance"].apply(
            lambda x: x + 1 if x == 0 else x
        )
        dataframe["estimated_salary"] = dataframe["estimated_salary"].apply(
            lambda x: x + 1 if x == 0 else x
        )

        # Non-linear transformations with safe handling for zeros
        dataframe["age_squared"] = dataframe["age"] ** 2
        dataframe["balance_sqrt"] = np.sqrt(dataframe["balance"] + epsilon)

        # Multiplicative interactions
        dataframe["credit_score_num_of_products"] = (
            dataframe["credit_score"] * dataframe["num_of_products"]
        )
        dataframe["age_balance"] = dataframe["age"] * dataframe["balance"]

        # Division interactions with safe handling for zeros
        dataframe["customer_value_normalized"] = (
            dataframe["balance"] / dataframe["balance"].max()
            + dataframe["num_of_products"] / dataframe["num_of_products"].max()
            + dataframe["estimated_salary"] / dataframe["estimated_salary"].max()
        )
        dataframe["product_density"] = dataframe["num_of_products"] / (
            dataframe["balance"] + epsilon
        )
        dataframe["balance_salary_ratio"] = dataframe["balance"] / (
            dataframe["estimated_salary"] + epsilon
        )
        dataframe["credit_score_age_ratio"] = dataframe["credit_score"] / (
            dataframe["age"] + epsilon
        )
        dataframe["tenure_age_ratio"] = dataframe["tenure"] / (
            dataframe["age"] + epsilon
        )
        dataframe["credit_salary_ratio"] = dataframe["credit_score"] / (
            dataframe["estimated_salary"] + epsilon
        )

        dataframe["engagement_score"] = (
            (dataframe["is_active_member"] * 0.5)
            + (dataframe["has_cr_card"] * 0.3)
            + (dataframe["num_of_products"] * 0.2)
        )

        # Logarithmic transformations with epsilon
        dataframe["log_balance"] = np.log1p(dataframe["balance"] + epsilon)
        dataframe["log_credit_score"] = np.log1p(dataframe["credit_score"] + epsilon)

        # Manual One-Hot Encoding for 'geography'
        dataframe["geography_germany"] = (dataframe["geography"] == "Germany").astype(
            int
        )
        dataframe["geography_spain"] = (dataframe["geography"] == "Spain").astype(int)
        dataframe["geography_france"] = (dataframe["geography"] == "France").astype(int)

        dataframe["gender_male"] = (dataframe["gender"] == "Male").astype(int)
        dataframe["gender_female"] = (dataframe["gender"] == "Female").astype(int)

        # Group by transformations
        dataframe["credit_score_gender"] = dataframe.groupby("gender")[
            "credit_score"
        ].transform("mean")

        dataframe["balance_age"] = dataframe.groupby("age")["balance"].transform("mean")
        dataframe["salary_rank_geography"] = dataframe.groupby("geography")[
            "estimated_salary"
        ].rank(method="dense")

        # Indicator transformations
        dataframe["balance_indicator"] = np.where(
            dataframe["balance"] >= 150000, "high", "low"
        )
        dataframe["balance_indicator_low"] = np.where(
            dataframe["balance"] < 150000, 1, 0
        )
        dataframe["balance_indicator_high"] = np.where(
            dataframe["balance"] >= 150000, 1, 0
        )

        # Binning transformations
        dataframe["life_stage"] = pd.cut(
            dataframe["age"],
            bins=[0, 20, 35, 50, np.inf],
            labels=["adolescence", "adulthood", "middle_age", "senior"],
        )
        life_stage_dummies = pd.get_dummies(
            dataframe["life_stage"], prefix="life_stage", drop_first=False
        )
        dataframe = pd.concat([dataframe, life_stage_dummies], axis=1)

        dataframe["tenure_group"] = np.where(
            dataframe["tenure"] > 6,
            "long_standing",
            np.where(dataframe["tenure"] <= 3, "new", "medium"),
        )
        dataframe["tenure_group_long_standing"] = np.where(
            dataframe["tenure"] > 6, 1, 0
        )
        dataframe["tenure_group_new"] = np.where(dataframe["tenure"] <= 3, 1, 0)
        dataframe["tenure_group_medium"] = np.where(
            (dataframe["tenure"] > 3) & (dataframe["tenure"] <= 6), 1, 0
        )

        dataframe["cs_category"] = pd.cut(
            dataframe["credit_score"],
            bins=[0, 500, 700, np.inf],
            labels=["low", "medium", "high"],
        )
        dataframe["cs_category_low"] = np.where(dataframe["credit_score"] <= 500, 1, 0)
        dataframe["cs_category_medium"] = np.where(
            (dataframe["credit_score"] > 500) & (dataframe["credit_score"] <= 700), 1, 0
        )
        dataframe["cs_category_high"] = np.where(dataframe["credit_score"] > 700, 1, 0)

        return dataframe

    @staticmethod
    def apply_log_transformation(data: pd.DataFrame, columns: list) -> None:
        """
        Applies logarithmic transformation to specified columns in the list of datasets.

        Parameters:
        - data (pd.DataFrame): DataFrame to apply the transformation.
        - columns (list of str): List of column names to which the logarithmic transformation will be applied.
        """
        epsilon = 1e-6
        for col in columns:
            # If there is a 0 value, add a small epsilon to avoid division by zero
            if (data[col] == 0).any():
                data[col] = np.log1p(data[col] + epsilon)
            else:
                data[col] = np.log1p(data[col])

        return None

    @staticmethod
    def apply_scaler(data: pd.DataFrame, columns: list, scaler_path: str):
        """
        Applies scaling to specified columns in the dataframe using the provided scaler.

        Parameters:
        - data (pd.DataFrame): DataFrame to apply the scaling.
        - columns (list of str): List of column names to which scaling will be applied.
        - scaler_path (str): Path to the saved scaler model.
        """
        with open(scaler_path, "rb") as f:
            scaler = pickle.load(f)
        data[columns] = scaler.transform(data[columns])
        return None

    @staticmethod
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
        X = FeatureEngineering.perform_transformations(X)

        # Then, apply log transformations and scaling
        log_cols = ["balance", "balance_sqrt", "age_balance", "age", "age_squared"]
        standard_scaler_cols = ["credit_score_num_of_products", "engagement_score"]
        min_max_scaler_cols = ["credit_score", "tenure"]
        robust_scaler_cols = [
            "age",
            "balance",
            "estimated_salary",
            "age_squared",
            "balance_sqrt",
            "age_balance",
            "product_density",
            "balance_salary_ratio",
            "credit_score_age_ratio",
            "tenure_age_ratio",
            "credit_salary_ratio",
        ]

        FeatureEngineering.apply_log_transformation(X, log_cols)
        FeatureEngineering.apply_scaler(
            X, standard_scaler_cols, "src/scalers/standard_scaler.pkl"
        )
        FeatureEngineering.apply_scaler(
            X, min_max_scaler_cols, "src/scalers/min_max_scaler.pkl"
        )
        FeatureEngineering.apply_scaler(
            X, robust_scaler_cols, "src/scalers/robust_scaler.pkl"
        )

        return X
