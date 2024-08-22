import streamlit as st
import pandas as pd
import requests
import plotly.express as px
from pathlib import Path
from utils.utils_fe_page import FeatureEngineering

colors_list = [
    "#1D5B79", "#1D7865", "#78621D", "#784F1D", "#1B2F38",
    "#75CDBB", "#CDB875", "#CDA675", "#B7E2F7", "#E4FFF9",
]

def run():
    st.title("Informações dos Top Clients")

    path = Path().resolve().parent
    # Local path
    #data_path = path / "churn-ticket/data"
    # Docker path
    data_path = path / "app/data"
    X_test = pd.read_parquet(data_path / "processed/X_test.parquet")

    fe = FeatureEngineering()
    X_test = fe._perform_transformations(X_test)

    url = "http://api:8000/predict"
    predictions = []
    probabilities = []

    for index, row in X_test.iterrows():
        input_data = {
            "credit_score": row["credit_score"],
            "geography": row["geography"],
            "gender": row["gender"],
            "age": row["age"],
            "tenure": row["tenure"],
            "balance": row["balance"],
            "num_of_products": row["num_of_products"],
            "has_cr_card": row["has_cr_card"],
            "is_active_member": row["is_active_member"],
            "estimated_salary": row["estimated_salary"]
        }

        response = requests.post(url, json=input_data)
        if response.status_code == 200:
            result = response.json()
            predictions.append(result['prediction'])
            probabilities.append(result['probability'])
        else:
            predictions.append(None)
            probabilities.append(None)

    X_test['predict'] = predictions
    X_test['predict_proba'] = probabilities

    high_churn = X_test[X_test['predict'] == 1]
    low_churn = X_test[X_test['predict'] == 0]

    high_churn_count = len(high_churn)
    low_churn_count = len(low_churn)

    active_filter = st.selectbox("Filtrar por Membros Ativos", options=["Todos", "Membros Ativos", "Membros Inativos"])

    if active_filter == "Membros Ativos":
        high_churn = high_churn[high_churn['is_active_member'] == 1]
        low_churn = low_churn[low_churn['is_active_member'] == 1]
    elif active_filter == "Membros Inativos":
        high_churn = high_churn[high_churn['is_active_member'] == 0]
        low_churn = low_churn[low_churn['is_active_member'] == 0]

    top_high_churn = high_churn.sort_values(by="predict_proba", ascending=False).head(10)
    top_low_churn = low_churn.sort_values(by="predict_proba", ascending=True).head(10)

    columns_to_show = ['estimated_salary', 'num_of_products', 'is_active_member', 'tenure',
                       'balance', 'age', 'credit_score', 'gender', 'predict_proba']

    top_high_churn_df = top_high_churn[columns_to_show].rename(columns={'predict_proba': 'Probabilidade de Churn'})
    top_low_churn_df = top_low_churn[columns_to_show].rename(columns={'predict_proba': 'Probabilidade de Churn'})

    country_counts = X_test['geography'].value_counts().reset_index()
    country_counts.columns = ['geography', 'count']

    fig1 = px.choropleth(
        country_counts,
        locations="geography",
        locationmode="country names",
        color="count",
        hover_name="geography",
        title="",
        color_continuous_scale=colors_list,
        scope="europe",
    )
    fig1.update_geos(showcoastlines=False, showland=False, fitbounds="locations")
    fig1.update_layout(geo=dict(bgcolor='rgba(0,0,0,0)'))

    col1, col2 = st.columns([1.5, 1])

    with col1:
        st.subheader("Ganhos/Perdas de Churn")
        col1a, col1b = st.columns(2)

        with col1a:
            st.metric("Alta Probabilidade de Churn", high_churn_count)

        with col1b:
            st.metric("Baixa Probabilidade de Churn", low_churn_count)

        st.subheader("Média de Valores por Probabilidade de Churn")

        high_churn_means = high_churn[
            ['num_of_products', 'age', 'estimated_salary', 'credit_score', 'balance', 'tenure']].mean().round(2)
        low_churn_means = low_churn[
            ['num_of_products', 'age', 'estimated_salary', 'credit_score', 'balance', 'tenure']].mean().round(2)

        # Rename columns
        high_churn_means = high_churn_means.rename('Alta Probabilidade de Churn')
        low_churn_means = low_churn_means.rename('Baixa Probabilidade de Churn')

        # Combine dataframes
        combined_means = pd.concat([high_churn_means, low_churn_means], axis=1)

        st.dataframe(combined_means)

        st.subheader("Distribuição de Clientes por País")
        st.plotly_chart(fig1, use_container_width=True)

    with col2:
        st.subheader("Top 10 Clientes com Alta Probabilidade de Churn")
        st.dataframe(top_high_churn_df)

        st.subheader("Top 10 Clientes com Baixa Probabilidade de Churn")
        st.dataframe(top_low_churn_df)

        st.subheader(":thinking_face: Informações úteis")
        st.markdown("""
        Algumas observações ao analisar esses dados:
        - O país onde vive é relevante para predição;
        - Número de produtos, idade, pontuação de crédito, saldo e gênero são as variáveis mais importantes;
        Essas informações podem ser usadas para melhorar as estratégias de retenção de clientes e ajustar o foco das campanhas de marketing.
        """)


if __name__ == "__main__":
    run()