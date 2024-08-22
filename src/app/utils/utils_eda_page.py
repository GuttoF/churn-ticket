import pandas as pd
import streamlit as st
import plotly.express as px
from plotly.subplots import make_subplots

class DataVisualizer:
    def __init__(self, data: pd.DataFrame):
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
        "#E4FFF9",]

    def hypotheses_1(self) -> None:
        credit_score_mean = self.data["credit_score"].mean()

        self.data["credit_score_category"] = self.data["credit_score"].apply(
            lambda x: "Above Average" if x >= credit_score_mean else "Below Average"
        )

        aux = self.data.groupby("credit_score_category")["exited"].mean().reset_index()

        title = "Taxa de Churn por Categoria de Score de Crédito<br>"
        suptitle = "<sub>Mostra a porcentagem de clientes que <b>saíram</b> com base na categoria de score de crédito acima ou abaixo da média</sub>"
        info = title + suptitle

        fig = px.bar(
            aux,
            y="credit_score_category",
            x="exited",
            title=info,
            hover_data={"exited": True, "credit_score_category": False},
            labels={
                "exited": "Taxa de Churn",
                "credit_score_category": "Categoria de Score de Crédito",
            },
            color="credit_score_category",
            color_discrete_sequence=[self.colors_list[0], self.colors_list[1]],
            barmode="group",
            text_auto=".2%",
        )

        fig.update_layout(
            xaxis_title="Categoria de Score de Crédito",
            yaxis_title="Taxa de Churn",
            width=800,
            height=400,
        )

        st.plotly_chart(fig)

    def hypotheses_2(self) -> None:
        self.data["group_product"] = self.data["num_of_products"].astype(str)

        self.data.sort_values("num_of_products", ascending=True, inplace=True)

        title = "Uso de Produtos por Grupo Etário<br>"
        suptitle = "<sub>Quantos produtos cada grupo etário possui?</sub>"
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
                "num_of_products": "Número de Produtos",
                "life_stage": "Grupo Etário",
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
            legend_title="Número de Produtos",
            yaxis_title="Total",
            xaxis_title="Grupo Etário",
            width=800,
            height=400,
        )

        st.plotly_chart(fig)

    def hypotheses_3(self) -> None:
        aux = self.data.groupby("geography")["balance"].mean().reset_index()

        aux.sort_values("balance", ascending=False, inplace=True)

        title = "Saldo Médio da Conta por Região<br>"
        suptitle = "<sub>Qual é o saldo médio em cada país?</sub>"
        info = title + suptitle

        fig = px.bar(
            aux,
            y="geography",
            x="balance",
            title=info,
            hover_data={"balance": True, "geography": False},
            labels={"balance": "Saldo Médio da Conta", "geography": "Região"},
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
            xaxis_title="Saldo Médio da Conta",
            yaxis_title="Região",
            width=800,
            height=400,
        )

        st.plotly_chart(fig)

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
                "Proporção de Membros Ativos por Gênero",
                "Proporção por Gênero",
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
                "is_active_member": "Proporção de Membros Ativos",
                "gender": "Gênero",
            },
        )

        for trace in bar.data:
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
            title_text="Qual é o gênero mais ativo?",
            width=800,
            height=400,
            legend_title_text="Gênero",
        )

        st.plotly_chart(fig)

    def hypotheses_5(self) -> None:
        self.data["salary_category"] = pd.qcut(
            self.data["estimated_salary"],
            q=4,
            labels=["Baixo", "Médio", "Alto", "Muito Alto"],
        )

        aux = (
            self.data.groupby("salary_category", observed=False)["has_cr_card"]
            .mean()
            .reset_index()
        )

        aux.sort_values("has_cr_card", ascending=False, inplace=True)

        title = "Propriedade de Cartão de Crédito por Categoria Salarial<br>"
        suptitle = "<sub>Qual é a proporção de clientes com cartão de crédito em cada categoria salarial?</sub>"
        info = title + suptitle

        fig = px.bar(
            aux,
            y="salary_category",
            x="has_cr_card",
            title=info,
            color="salary_category",
            color_discrete_sequence=self.colors_list,
            labels={
                "has_cr_card": "Propriedade de Cartão de Crédito",
                "salary_category": "Categoria Salarial",
            },
            barmode="group",
            text_auto=".2%",
        )

        fig.update_traces(width=[0.6] * len(aux))

        fig.update_layout(
            yaxis_title="Categoria Salarial",
            xaxis_title="Propriedade de Cartão de Crédito",
            width=800,
            height=400,
        )

        st.plotly_chart(fig)

    def hypotheses_6(self) -> None:
        aux = self.data.groupby("tenure")["exited"].mean().reset_index()

        title = "Taxa de Churn por Tempo de Permanência (Tenure)<br>"
        suptitle = "<sub>Existe uma relação entre <b>churn</b> e tempo de permanência?</sub>"
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
            xaxis_title="Tempo de Permanência",
            yaxis_title="Taxa de Churn",
            width=800,
            height=400,
        )

        st.plotly_chart(fig)

    def hypotheses_7(self) -> None:
        aux = self.data.groupby("balance_indicator")["exited"].mean().reset_index()

        aux.sort_values("balance_indicator", ascending=True, inplace=True)

        title = "Taxa de Churn por Categoria de Saldo<br>"
        suptitle = "<sub>Existe uma diferença significativa na taxa de churn entre clientes com diferentes saldos de conta?</sub>"
        info = title + suptitle

        fig = px.bar(
            aux,
            y="balance_indicator",
            x="exited",
            title=info,
            hover_data={"balance_indicator": False, "exited": True},
            labels={
                "balance_indicator": "Indicador de Saldo",
                "exited": "Taxa de Churn",
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
            yaxis_title="Indicador de Saldo",
            xaxis_title="Taxa de Churn",
            width=800,
            height=400,
        )

        st.plotly_chart(fig)

    def hypotheses_8(self) -> None:
        aux = (
            self.data.groupby("life_stage", observed=False)["has_cr_card"]
            .mean()
            .reset_index()
        )

        aux.sort_values("life_stage", ascending=True, inplace=True)

        title = "Propriedade de Cartão de Crédito por Idade<br>"
        suptitle = "<sub>Existe uma diferença significativa na propriedade de cartão de crédito entre clientes de diferentes idades?</sub>"
        info = title + suptitle

        fig = px.bar(
            aux,
            y="life_stage",
            x="has_cr_card",
            title=info,
            hover_data={"life_stage": True, "has_cr_card": False},
            labels={
                "life_stage": "Grupo Etário",
                "has_cr_card": "Propriedade de Cartão de Crédito",
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
            yaxis_title="Grupo Etário",
            xaxis_title="Propriedade de Cartão de Crédito",
            width=800,
            height=400,
        )

        st.plotly_chart(fig)
