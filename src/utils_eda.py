import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import scipy.stats as ss
from plotly.subplots import make_subplots


class DataVisualizer:
    def __init__(self, data: pd.DataFrame):
        """
        Initialize the DataVisualizer with a DataFrame.

        Args:
        - data (pd.DataFrame): The DataFrame containing the data for visualization.
        """
        self.data = data

    colors_list = [
        "#1D5B79",
        "#1D7865",
        "#78621D",
        "#784F1D",
        "#1B2F38",
        "#75CDBB",
        "#CDB875",
        "#CDA675",
        "#B7E2F7",
        "#E4FFF9",
    ]

    def multiple_distplots(self, columns: list) -> None:
        """
        Create multiple distribution plots (distplots) with histogram and KDE on the same scale.

        Args:
        - columns (list): A list of column names to be plotted.
        - colors (list, optional): A list of colors for each distribution plot. Defaults to None.
        """
        num_columns = len(columns)
        cols = int(num_columns**0.5)
        if cols * cols >= num_columns:
            rows = cols
        else:
            rows = cols + 1
            if cols * rows < num_columns:
                cols += 1

        fig = make_subplots(rows=rows, cols=cols, subplot_titles=columns)
        row, col = 1, 1

        for i, column in enumerate(columns):
            color_index = i % len(self.colors_list)
            column_data = self.data[column]
            fig.add_trace(
                go.Histogram(
                    x=column_data,
                    name=column + " Histogram",
                    nbinsx=30,
                    opacity=0.75,
                    marker_color=self.colors_list[color_index],
                    histnorm="probability density",
                ),
                row=row,
                col=col,
            )

            kde = ss.gaussian_kde(column_data)
            kde_x = np.linspace(column_data.min(), column_data.max(), 500)
            kde_y = kde(kde_x)

            fig.add_trace(
                go.Scatter(
                    x=kde_x,
                    y=kde_y,
                    mode="lines",
                    name=column + " KDE",
                    line=dict(color=self.colors_list[color_index], width=2),
                ),
                row=row,
                col=col,
            )

            col += 1
            if col > cols:
                col = 1
                row += 1

        fig.update_layout(
            title_text="Distribution Plots", height=200 * rows, showlegend=False
        )
        fig.show()

    def distribution_analysis(self, columns: list) -> None:
        """
        Create multiple distribution plots (distplots) and boxplots for the given columns.

        Args:
        - columns (list): A list of column names to be plotted.
        """
        cols = 2  # two columns per row, one for distplot and one for boxplot
        rows = len(columns)

        fig = make_subplots(
            rows=rows,
            cols=cols,
            subplot_titles=[f"{col}" for col in columns for _ in (0, 1)],
        )

        row = 1

        for i, column in enumerate(columns):
            color_index = i % len(self.colors_list)
            column_data = self.data[column]

            # Distribution Plot
            fig.add_trace(
                go.Histogram(
                    x=column_data,
                    name="Histogram",
                    nbinsx=30,
                    opacity=0.75,
                    marker_color=self.colors_list[color_index],
                    histnorm="probability density",
                ),
                row=row,
                col=1,
            )

            kde = ss.gaussian_kde(column_data)
            kde_x = np.linspace(column_data.min(), column_data.max(), 500)
            kde_y = kde(kde_x)

            fig.add_trace(
                go.Scatter(
                    x=kde_x,
                    y=kde_y,
                    mode="lines",
                    name=f"{column} KDE",
                    line=dict(color=self.colors_list[color_index], width=2),
                ),
                row=row,
                col=1,
            )

            # Boxplot
            fig.add_trace(
                go.Box(
                    x=column_data,
                    name="",
                    marker_color=self.colors_list[color_index],
                ),
                row=row,
                col=2,
            )

            row += 1

        fig.update_layout(
            title_text="Distribution and Boxplot Analysis",
            height=250 * rows,
            showlegend=False,
        )
        fig.show()

    def multiple_barplots(self, columns: list) -> None:
        """
        Generate multiple bar plots for specified columns in the DataFrame.

        Args:
        - columns (list): A list of column names to be plotted.
        """
        num_columns = len(columns)
        cols = int(num_columns**0.5)
        rows = (num_columns + cols - 1) // cols

        fig = make_subplots(rows=rows, cols=cols, subplot_titles=columns)

        index = 0
        for row in range(1, rows + 1):
            for col in range(1, cols + 1):
                if index < num_columns:
                    column_data = self.data[columns[index]].value_counts().reset_index()
                    column_data.columns = ["category", "count"]

                    fig.add_trace(
                        go.Bar(
                            x=column_data["category"],
                            y=column_data["count"],
                            name=columns[index],
                            marker_color=self.colors_list[
                                index % len(self.colors_list)
                            ],
                        ),
                        row=row,
                        col=col,
                    )
                index += 1

        fig.update_layout(title_text="Barplots", height=200 * rows, showlegend=False)
        fig.show()

    def correlation_heatmap(self, columns: list) -> None:
        """
        Generate a heatmap for the correlation matrix of specified columns in the DataFrame.
        """
        corr_matrix = self.data[columns].corr()
        fig = go.Figure(
            data=go.Heatmap(
                z=corr_matrix.values,
                x=corr_matrix.columns,
                y=corr_matrix.columns,
                colorscale="Greys",
                colorbar=dict(title="Correlation Coefficient"),
            )
        )

        annotations = []
        for i, row in enumerate(corr_matrix.values):
            for j, value in enumerate(row):
                annotations.append(
                    go.layout.Annotation(
                        x=corr_matrix.columns[j],
                        y=corr_matrix.columns[i],
                        text=str(round(value, 2)),
                        showarrow=False,
                        font=dict(color="black"),
                    )
                )

        fig.update_layout(
            title_text="Correlation Heatmap",
            height=1200,
            width=1200,
            annotations=annotations,
        )
        fig.show()

    def scatter_plot_matrix(self, columns: list, color_column=None) -> None:
        """
        Generate a scatter plot matrix for specified columns, optionally colored by another column.
        """
        if color_column:
            fig = px.scatter_matrix(
                self.data, dimensions=columns, color=self.data[color_column]
            )
        else:
            fig = px.scatter_matrix(
                self.data,
                dimensions=columns,
                color_discrete_sequence=self.colors_list,
            )

        fig.update_layout(
            title_text="Scatter Plot Matrix",
            height=1000,
            width=1000,
            xaxis_tickangle=-45,
            yaxis_tickangle=-45,
        )
        fig.show()

    def cramers_v(self, x, y):
        """
        Calculate Cramér's V statistic for categorical-categorical association.
        """
        confusion_matrix = pd.crosstab(x, y)
        chi2 = ss.chi2_contingency(confusion_matrix)[0]
        n = confusion_matrix.sum().sum()
        phi2 = chi2 / n
        r, k = confusion_matrix.shape
        phi2_corr = max(0, phi2 - ((k - 1) * (r - 1)) / (n - 1))
        r_corr = r - ((r - 1) ** 2) / (n - 1)
        k_corr = k - ((k - 1) ** 2) / (n - 1)
        return np.sqrt(phi2_corr / min((k_corr - 1), (r_corr - 1)))

    def categorical_heatmap(self, columns: list) -> None:
        """
        Generate a heatmap for the association between categorical columns.
        """
        cramer_matrix = pd.DataFrame(index=columns, columns=columns, dtype=float)
        for col1 in columns:
            for col2 in columns:
                if col1 == col2:
                    cramer_matrix.loc[col1, col2] = 1.0
                else:
                    cramer_matrix.loc[col1, col2] = self.cramers_v(
                        self.data[col1], self.data[col2]
                    )

        fig = go.Figure(
            data=go.Heatmap(
                z=cramer_matrix.values,
                x=cramer_matrix.columns,
                y=cramer_matrix.columns,
                colorscale="Greys",
                colorbar=dict(title="Cramér's V"),
            )
        )

        annotations = []
        for i, row in enumerate(cramer_matrix.values):
            for j, value in enumerate(row):
                annotations.append(
                    go.layout.Annotation(
                        x=cramer_matrix.columns[j],
                        y=cramer_matrix.columns[i],
                        text=str(round(value, 2)),
                        showarrow=False,
                        font=dict(color="black"),
                    )
                )

        fig.update_layout(
            title_text="Cramer's Heatmap",
            height=1200,
            width=1200,
            annotations=annotations,
        )
        fig.show()

    def statistic_test(self, featureA: str, featureB: str) -> None:
        """
        Perform a chi-square test of independence between two categorical variables.

        Parameters:
        - featureA: str
            The name of the first categorical variable.
        - featureB: str
            The name of the second categorical variable.

        Returns:
        None

        Prints the chi-square statistic, p-value, and the result of the hypothesis test.
        """
        table = pd.crosstab(self.data[featureA], self.data[featureB])
        result = ss.chi2_contingency(table)
        print(
            f"Chi2 Statistic: {round(result.statistic, 3)}\nP-value: {round(result.pvalue, 3)}"
        )

        if result.pvalue < 0.05:
            print("Reject the null hypothesis: The variables are dependent")
        else:
            print("Fail to reject the null hypothesis: The variables are independent")

    def hypotheses_1(self) -> None:
        credit_score_mean = self.data["credit_score"].mean()

        self.data["credit_score_category"] = self.data["credit_score"].apply(
            lambda x: "Above Average" if x >= credit_score_mean else "Below Average"
        )

        aux = self.data.groupby("credit_score_category")["exited"].mean().reset_index()

        title = "Churn Rate by Credit Score Category<br>"
        suptitle = "<sub>Shows the percentage of customers who <b>exited</b> based on their credit score being above or below average</sub>"
        info = title + suptitle

        fig = px.bar(
            aux,
            y="credit_score_category",
            x="exited",
            title=info,
            hover_data={"exited": True, "credit_score_category": False},
            labels={
                "exited": "Churn Rate",
                "credit_score_category": "Credit Score Category",
            },
            color="credit_score_category",
            color_discrete_sequence=[self.colors_list[0], self.colors_list[1]],
            barmode="group",
            text_auto=".2%",
        )

        fig.update_layout(
            xaxis_title="Credit Score Category",
            yaxis_title="Churn Rate",
            plot_bgcolor="white",
            paper_bgcolor="white",
            width=800,
            height=400,
        )

        fig.show()

    def hypotheses_2(self) -> None:
        self.data["group_product"] = self.data["num_of_products"].astype(str)

        self.data.sort_values("num_of_products", ascending=True, inplace=True)

        title = "Product Usage by Age Group<br>"
        suptitle = "<sub>How many products does each age group have?</sub>"
        info = title + suptitle

        fig = px.bar(
            self.data,
            x="life_stage",
            y="num_of_products",
            title=info,
            hover_data={
                "num_of_products": True,
                "life_stage": False,
                "group_product": False,
            },
            labels={
                "num_of_products": "Number of Products",
                "life_stage": "Age Group",
            },
            color="group_product",
            color_discrete_sequence=[
                self.colors_list[0],
                self.colors_list[1],
                self.colors_list[2],
                self.colors_list[3],
            ],
            barmode="group",
            text_auto=False,
        )

        legend_order = {"1": 1, "2": 2, "3": 3, "4": 4}

        for trace in fig.data:
            if trace.name in legend_order:
                trace.legendgroup = trace.name
                trace.legendrank = legend_order[trace.name]

        fig.update_yaxes(dtick=500)

        fig.update_layout(
            legend_title="Number of Products",
            yaxis_title="Total",
            xaxis_title="Age Group",
            plot_bgcolor="white",
            paper_bgcolor="white",
            width=800,
            height=400,
        )

        fig.show()

    def hypotheses_3(self) -> None:
        aux = self.data.groupby("geography")["balance"].mean().reset_index()

        aux.sort_values("balance", ascending=False, inplace=True)

        title = "Average Account Balance by Region<br>"
        suptitle = "<sub>What is the average balance on each country?</sub>"
        info = title + suptitle

        fig = px.bar(
            aux,
            y="geography",
            x="balance",
            title=info,
            hover_data={"balance": True, "geography": False},
            labels={"balance": "Average Account Balance", "geography": "Region"},
            color="geography",
            color_discrete_sequence=[
                self.colors_list[0],
                self.colors_list[1],
                self.colors_list[2],
            ],
            barmode="group",
            text_auto=True,
        )

        fig.update_layout(
            xaxis_title="Average Account Balance",
            yaxis_title="Region",
            plot_bgcolor="white",
            paper_bgcolor="white",
            width=800,
            height=400,
        )

        fig.show()

    def hypotheses_4(self) -> None:
        active_members = self.data[self.data["is_active_member"] == 1]
        gender_counts = active_members["gender"].value_counts().reset_index()
        gender_counts.columns = ["gender", "count"]

        active_by_gender = (
            self.data.groupby("gender")["is_active_member"].mean().reset_index()
        )
        active_by_gender.sort_values("is_active_member", ascending=False, inplace=True)

        colors_map = {"Male": self.colors_list[0], "Female": self.colors_list[1]}

        fig = make_subplots(
            rows=1,
            cols=2,
            specs=[[{"type": "bar"}, {"type": "pie"}]],
            subplot_titles=(
                "Proportion of Active Members by Gender",
                "Gender Proportion",
            ),
        )

        bar = px.bar(
            active_by_gender,
            x="gender",
            y="is_active_member",
            color="gender",
            color_discrete_map=colors_map,
            text_auto=".2%",
            labels={
                "is_active_member": "Active Members Proportion",
                "gender": "Gender",
            },
        )

        for trace in bar.data:
            # trace.width = [1] * len(active_by_gender)
            fig.add_trace(trace, row=1, col=1)

        pie = px.pie(
            gender_counts,
            values="count",
            names="gender",
            color="gender",
            color_discrete_map=colors_map,
        )

        for trace in pie.data:
            trace.showlegend = False
            fig.add_trace(trace, row=1, col=2)

        fig.update_layout(
            title_text="What is the most active gender?",
            plot_bgcolor="white",
            paper_bgcolor="white",
            width=800,
            height=400,
            legend_title_text="Gender",
            # xaxis_tickangle=90,
        )

        fig.show()

    def hypotheses_5(self) -> None:
        self.data["salary_category"] = pd.qcut(
            self.data["estimated_salary"],
            q=4,
            labels=["Low", "Medium", "High", "Very High"],
        )

        aux = (
            self.data.groupby("salary_category", observed=False)["has_cr_card"]
            .mean()
            .reset_index()
        )

        aux.sort_values("has_cr_card", ascending=False, inplace=True)

        title = "Credit Card Ownership by Salary Category<br>"
        suptitle = "<sub>What is the proportion of customers with a credit card in each salary category?</sub>"
        info = title + suptitle

        fig = px.bar(
            aux,
            y="salary_category",
            x="has_cr_card",
            title=info,
            color="salary_category",
            color_discrete_sequence=self.colors_list,
            labels={
                "has_cr_card": "Credit Card Ownership",
                "salary_category": "Salary Category",
            },
            barmode="group",
            text_auto=".2%",
        )

        fig.update_traces(width=[0.6] * len(aux))

        fig.update_layout(
            yaxis_title="Salary Category",
            xaxis_title="Credit Card Ownership",
            plot_bgcolor="white",
            paper_bgcolor="white",
            width=800,
            height=400,
        )

        fig.show()

    def hypotheses_6(self) -> None:
        aux = self.data.groupby("tenure")["exited"].mean().reset_index()

        title = "Churn Rate by Tenure<br>"
        suptitle = "<sub>Is there a relationship between <b>churn</b> and tenure?</sub>"
        info = title + suptitle

        aux.sort_values("tenure", ascending=True, inplace=True)

        fig = px.line(
            aux,
            x="tenure",
            y="exited",
            hover_data={"exited": True, "tenure": False},
            title=info,
            color_discrete_sequence=self.colors_list,
            markers=False,
        )

        fig.update_traces(
            line=dict(width=4),
        )

        fig.update_xaxes(tickvals=aux["tenure"].unique())

        fig.update_layout(
            xaxis_title="Tenure",
            yaxis_title="Churn Rate",
            plot_bgcolor="white",
            paper_bgcolor="white",
            width=800,
            height=400,
        )

        fig.show()

    def hypotheses_7(self) -> None:
        aux = self.data.groupby("balance_indicator")["exited"].mean().reset_index()

        aux.sort_values("balance_indicator", ascending=True, inplace=True)

        title = "Churn rate by Balance Category<br>"
        suptitle = "<sub>Is there a significant difference in the churn rate among customers with different account balances?</sub>"
        info = title + suptitle

        fig = px.bar(
            aux,
            y="balance_indicator",
            x="exited",
            title=info,
            hover_data={"balance_indicator": False, "exited": True},
            labels={
                "balance_indicator": "Balance Indicator",
                "exited": "Churn Rate",
            },
            color="balance_indicator",
            color_discrete_sequence=[
                self.colors_list[0],
                self.colors_list[1],
                self.colors_list[2],
            ],
            barmode="group",
            text_auto=".2%",
        )

        fig.update_layout(
            yaxis_title="Balance",
            xaxis_title="Churn Rate",
            plot_bgcolor="white",
            paper_bgcolor="white",
            width=800,
            height=400,
        )

        fig.show()

    def hypotheses_8(self) -> None:
        aux = (
            self.data.groupby("life_stage", observed=False)["has_cr_card"]
            .mean()
            .reset_index()
        )

        aux.sort_values("life_stage", ascending=True, inplace=True)

        title = "Credit_Card Ownership by Age<br>"
        suptitle = "<sub>Is there a significant difference in the Credit Card Ownership among customers with different age?</sub>"
        info = title + suptitle

        fig = px.bar(
            aux,
            y="life_stage",
            x="has_cr_card",
            title=info,
            hover_data={"life_stage": True, "has_cr_card": False},
            labels={
                "life_stage": "Age Group",
                "has_cr_card": "Credit Card Ownership",
            },
            color="life_stage",
            color_discrete_sequence=[
                self.colors_list[0],
                self.colors_list[1],
                self.colors_list[2],
                self.colors_list[3],
            ],
            barmode="group",
            text_auto=".2%",
        )

        fig.update_traces(width=[0.6] * len(aux))

        fig.update_layout(
            yaxis_title="Age Group",
            xaxis_title="Credit Card Ownership",
            plot_bgcolor="white",
            paper_bgcolor="white",
            width=800,
            height=400,
        )

        fig.show()
