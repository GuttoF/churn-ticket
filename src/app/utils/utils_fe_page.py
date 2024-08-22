import numpy as np
import pandas as pd

class FeatureEngineering:
    """
    Class for performing feature engineering on a given dataframe.

    Parameters:
    - seed (int): Random seed for reproducibility.
    """

    def __init__(self, seed=None):
        self.seed = seed

    @staticmethod
    def _perform_transformations(dataframe: pd.DataFrame) -> pd.DataFrame:
        """
        Perform feature transformations on the given dataframe.

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

        # Indicator transformations
        dataframe["balance_indicator"] = np.where(
            dataframe["balance"] >= 150000, "high", "low"
        )

        # Binning transformations
        dataframe["life_stage"] = pd.cut(
            dataframe["age"],
            bins=[0, 20, 35, 50, np.inf],
            labels=["adolescence", "adulthood", "middle_age", "senior"],
        )
        dataframe["cs_category"] = pd.cut(
            dataframe["credit_score"],
            bins=[0, 500, 700, np.inf],
            labels=["low", "medium", "high"],
        )
        dataframe["tenure_group"] = pd.cut(
            dataframe["tenure"],
            bins=[-1, 3, 6, np.inf],
            labels=["new", "intermediate", "long_standing"],
        )

        # Group by transformations
        dataframe["credit_score_gender"] = dataframe.groupby("gender")[
            "credit_score"
        ].transform("mean")
        dataframe["balance_age"] = dataframe.groupby("age")["balance"].transform("mean")
        dataframe["salary_rank_geography"] = dataframe.groupby("geography")[
            "estimated_salary"
        ].rank(method="dense")

        return dataframe