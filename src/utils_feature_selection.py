import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.feature_selection import RFE
from sklearn.preprocessing import MinMaxScaler, RobustScaler, OneHotEncoder, StandardScaler


def multiple_histplots(data: pd.DataFrame, rows: int, cols: int):
    """
    Shows a matrix with hisplots of selected features.

    Args:
        data ([dataframe]): [Insert all categorical attributes in the dataset]
        rows ([int]): [Insert the number of rows of the subplot]
        cols ([int]): [Insert the number of columns of the subplot]

    Returns:
        [Image]: [A matrix plot with histplots]
    """

    for i, col in enumerate(data.columns, 1):
        plt.subplot(rows, cols, i)
        ax = sns.histplot(data[col], kde = True)
        plt.ylabel('')

    return ax

def apply_log_transformation(datasets: pd.DataFrame, columns: list) -> None:
    """
    Applies logarithmic transformation to specified columns in a list of datasets,
    replacing the original values in the DataFrames.

    Args:
        datasets (list of pd.DataFrame): List of DataFrames where the transformation will be applied.
        columns (list of str): List of column names to which the logarithmic transformation will be applied.

    Returns:
        None
    """
    for df in datasets:
        for col in columns:
            df[col] = np.log1p(df[col])

    return None

def apply_one_hot_encoder(datasets: pd.DataFrame, columns: list):
    """
    Applies One-Hot Encoding to specified columns in a list of datasets using sklearn OneHotEncoder,
    replacing the original columns in the DataFrames.

    Args:
        datasets (list of pd.DataFrame): List of DataFrames where the encoding will be applied.
        columns (list of str): List of column names to which One-Hot Encoding will be applied.

    Returns:
        None
    """
    encoder = OneHotEncoder(sparse_output=False, drop='first')

    for df in datasets:
        # Apply One-Hot Encoding
        encoded_columns = pd.DataFrame(encoder.fit_transform(df[columns]),
                                       columns=encoder.get_feature_names_out(columns),
                                       index=df.index)

        # Drop original columns and concatenate encoded columns
        df.drop(columns=columns, inplace=True)
        df[encoded_columns.columns] = encoded_columns

        # Put columns to snake_case
        df.columns = [col.lower().replace(" ", "_") for col in df.columns]

    return None

def apply_standard_scaler(datasets: list, columns: list):
    """
    Applies Standard Scaling to specified columns in a list of datasets using sklearn StandardScaler,
    replacing the original columns in the DataFrames.

    Args:
        datasets (list of pd.DataFrame): List of DataFrames where the scaling will be applied.
        columns (list of str): List of column names to which Standard Scaling will be applied.

    Returns:
        None
    """
    scaler = StandardScaler()

    for df in datasets:
        # Apply Standard Scaling
        df[columns] = scaler.fit_transform(df[columns])

    return None


def apply_min_max_scaler(datasets: list, columns: list):
    """
    Applies Min-Max Scaling to specified columns in a list of datasets using sklearn MinMaxScaler,
    replacing the original columns in the DataFrames.

    Args:
        datasets (list of pd.DataFrame): List of DataFrames where the scaling will be applied.
        columns (list of str): List of column names to which Min-Max Scaling will be applied.

    Returns:
        None
    """
    scaler = MinMaxScaler()

    for df in datasets:
        # Apply Min-Max Scaling
        df[columns] = scaler.fit_transform(df[columns])

    return None

def apply_robust_scaler(datasets: list, columns: list):
    """
    Applies Robust Scaling to specified columns in a list of datasets using sklearn RobustScaler,
    replacing the original columns in the DataFrames.

    Args:
        datasets (list of pd.DataFrame): List of DataFrames where the scaling will be applied.
        columns (list of str): List of column names to which Robust Scaling will be applied.

    Returns:
        None
    """
    scaler = RobustScaler()

    for df in datasets:
        # Apply Robust Scaling
        df[columns] = scaler.fit_transform(df[columns])

    return None

def plot_feature_importance(
        X_train: pd.DataFrame,
        y_train: pd.Series,
        model_type: str = 'ExtraTrees',
        n_estimators: int = 200,
        n_jobs: int = 14,
        random_state: int = 42
):
    """
    Function to calculate and plot feature importance using ExtraTreesClassifier or RandomForestClassifier.

    Parameters:
    - X_train (DataFrame): The training features.
    - y_train (Series or DataFrame): The corresponding labels.
    - model_type (str): The model to use for feature importance ('ExtraTrees' or 'RandomForest').
    - n_estimators (int): The number of trees in the forest (default is 200).
    - n_jobs (int): The number of processors to use for computation (default is 14).
    - random_state (int): The seed for the random number generator (default is 42).

    Returns:
    - DataFrame: A DataFrame containing features and their respective importance, sorted from most to least important.
    - Displays: A bar plot showing the feature importance.
    """

    # Select the model based on 'model_type'
    if model_type == 'ExtraTrees':
        model = ExtraTreesClassifier(n_estimators=n_estimators, n_jobs=n_jobs, random_state=random_state)
    elif model_type == 'RandomForest':
        model = RandomForestClassifier(n_estimators=n_estimators, n_jobs=n_jobs, random_state=random_state)
    else:
        raise ValueError("model_type must be 'ExtraTrees' or 'RandomForest'")

    # Fit the model on the training data
    model.fit(X_train, y_train)

    # Create a DataFrame with features and their importance scores
    feature_selection = pd.DataFrame({
        'feature': X_train.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False).reset_index(drop=True)

    # Plot the feature importance
    plt.figure(figsize=(10, 8))
    ax = sns.barplot(x='importance', y='feature', data=feature_selection)
    ax.set_title(f'Feature Importance with {model_type}')
    plt.show()

    # Return the DataFrame with feature importance
    return feature_selection

def select_features_with_rfe(
        X_train: pd.DataFrame,
        y_train: pd.Series,
        n_features_to_select: int = 12,
        n_jobs: int = 14
):
    """
    Function to perform Recursive Feature Elimination (RFE) using RandomForestClassifier
    and return the selected features.

    Parameters:
    - X_train (DataFrame): The training features.
    - y_train (Series): The corresponding labels.
    - n_features_to_select (int): The number of features to select (default is 12).
    - n_jobs (int): The number of processors to use for computation (default is 14).

    Returns:
    - selected_columns (Index): An Index object containing the names of the selected columns.
    """

    # Initialize the RandomForestClassifier
    rf_selector = RandomForestClassifier(n_jobs=n_jobs)

    # Initialize RFE with the specified number of features to select
    rfe = RFE(estimator=rf_selector, n_features_to_select=n_features_to_select, step=1)

    # Fit RFE on the training data
    rfe = rfe.fit(X_train, y_train)

    # Get the selected columns
    selected_columns = X_train.columns[rfe.support_]

    return selected_columns