import polars as pl
import seaborn as sns
from matplotlib import pyplot as plt
import pandas as pd


# Classes
class DataDescription:
    def __init__(self, data):
        self.data = data

    def show_data(self):
        return self.data.head()

    def data_dimensions(self):
        return print(
            f"Number of Rows: {self.data.shape[0]}\nNumber of Columns: {self.data.shape[1]}"
        )

    def data_types(self):
        return {
            col_name: dtype
            for col_name, dtype in zip(self.data.columns, self.data.dtypes)
        }

    def check_na(self):
        return self.data.select(
            [pl.col(col).is_null().sum().alias(col) for col in self.data.columns]
        )

    def select_numeric_features(self):
        return self.data.select(
            pl.col(
                [
                    col
                    for col in self.data.columns
                    if self.data[col].dtype in [pl.Int64, pl.Float64]
                ]
            )
        )

    def select_categorical_features(self):
        return self.data.select(
            pl.col(
                [col for col in self.data.columns if self.data[col].dtype == pl.Utf8]
            )
        )


# Functions
def plot_features(data: pd.DataFrame, plot_type: str = "histplot") -> None:
    """
    Plots features of a DataFrame as either histograms or boxplots.

    Parameters:
    data (pd.DataFrame): The DataFrame containing the data to plot.
    plot_type (str): The type of plot to create. Must be either 'histplot' or 'boxplot'. Default is 'histplot'.


    Raises:
    ValueError: If plot_type is not 'histplot' or 'boxplot'.

    Returns:
    None
    """
    if plot_type not in ["histplot", "boxplot"]:
        raise ValueError("plot_type must be either 'histplot' or 'boxplot'")

    count = 1

    plt.figure(figsize=(15, 10))

    for col in data.columns:
        plt.subplot(3, 3, count)

        if plot_type == "histplot":
            axes = sns.histplot(data=data, x=col, kde=True)
            axes.title.set_text(f"Histogram of {col}")
        elif plot_type == "boxplot":
            axes = sns.boxplot(data=data, x=col, orient="h")
            axes.title.set_text(f"Catching outliers in {col}")

        plt.subplots_adjust(hspace=0.5, wspace=0.5)
        count += 1

    plt.show()

    return None


def categorical_metrics(data: pd.DataFrame, column: str):
    """
    Shows the absolute and percent values in categorical variables.

    Args:
        data ([dataframe]): [Insert all categorical attributes in the dataset]
        column ([str]): [Insert the column name]

    Returns:
        [dataframe]: [A dataframe with absolute and percent values]
    """

    return pd.DataFrame(
        {
            "absolute": data[column].value_counts(),
            "percent %": data[column].value_counts(normalize=True) * 100,
        }
    )
