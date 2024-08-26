import os

import pandas as pd
import requests
import streamlit as st
from utils.utils_business_translation import (
    get_churn_predictions_from_api,
    knapsack_solver,
    top_clients,
)


def get_prediction(input_data):
    api_url = os.getenv("API_URL")
    # api_url = "http://localhost:8000/predict"
    # if not api_url:
    #    raise ValueError("API_URL not found in environment variables")

    response = requests.post(api_url, json=input_data)
    if response.status_code == 200:
        result = response.json()
        return result
    return None


def process_batch(batch_df):
    predictions = []
    probabilities = []
    for index, row in batch_df.iterrows():
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
            "estimated_salary": row["estimated_salary"],
        }
        result = get_prediction(input_data)
        if result:
            predictions.append(result["prediction"])
            probabilities.append(result["probability"])
        else:
            predictions.append(None)
            probabilities.append(None)

    batch_df["predict"] = predictions
    batch_df["predict_proba"] = probabilities
    return batch_df


def run():
    st.title(":moneybag: Simulação de Retenção de Clientes Utilizando Tickets")

    st.markdown("""
    Vamos fazer algumas considerações sobre o retorno financeiro dos clientes.
    - Retorno de 15% se o salário estimado for menor que a média;
    - Retorno de 20% se o salário estimado for igual à média e menor que duas vezes a média;
    - Retorno de 25% se o salário estimado for igual ou maior que duas vezes a média;

    Com isso obtemos as seguintes informações:

    """)

    api_url = os.getenv("API_URL")
    # api_url = "http://localhost:8000/predict"
    if not api_url:
        raise ValueError("API_URL not found in environment variables")

    return_clients, churn_loss, total_return, df_simulation = (
        get_churn_predictions_from_api(api_url)
    )

    result_return_clients = (
        f"€{return_clients:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")
    )
    result_churn_loss = (
        f"€{churn_loss:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")
    )
    result_total_return = (
        f"{total_return:,.2f}%".replace(",", "X").replace(".", ",").replace("X", ".")
    )

    st.metric("Retorno Total Estimado", result_return_clients)
    st.metric("Perda de Churn", result_churn_loss)
    st.metric("Retorno Total", result_total_return)
    st.metric("Churn Rate atual", "20.17%")

    st.subheader("Cenários de Retenção de Clientes")
    st.write("""
    Para a simulação de retenção de clientes, vamos considerar dois cenários:
    - **Cenário 1**: Investimento igual para todos os clientes no top N, sendo o ticket calculado pelo total de investimento dividido pelo número de clientes.
    - **Cenário 2**: Investimento variável para cada cliente, com base em sua probabilidade de churn e retorno financeiro esperado.
    """)

    st.subheader("Como Funciona o Cenário 2?")

    st.info("""
    Imagine que você vai fazer uma viagem e tem uma mochila com capacidade limitada, ou seja, só cabe um certo peso ou volume de itens. Você tem vários objetos que gostaria de levar, como roupas, comida, livros, e eletrônicos, mas não consegue levar tudo porque a mochila é limitada.

    Cada objeto tem um peso (ou ocupa um espaço) e também tem um valor para você (como a utilidade ou o quanto você gosta do item). O objetivo do problema da mochila é decidir quais itens levar na mochila para maximizar o valor total dos itens que você leva.
    """)

    st.write(
        """
        Usando o problema da mochila com probabilidades e um orçamento de exemplo:

        - p(churn) >= Altíssima(ex: 0,95): Altíssima: Cliente com alta probabilidade de ficar com um cupom de alto valor.
        - Alta(ex: 0,85) >= p(churn) < Alta: Cliente que pode ficar com um cupom de valor médio.
        - Média(ex: 0,70) >= p(churn) < Média: Cliente que pode ficar com um cupom de valor baixo.
        - Abaixo da média >= p(churn) < Abaixo da média: Cliente que pode ficar sem cupom.
        """
    )

    st.sidebar.subheader("Simulação")

    top_n_options = [10, 15, 20, 25, 30, 35, 40, 45, 50]
    top_n = st.sidebar.selectbox(
        "Quantidade de Clientes no Top", top_n_options, index=2
    )

    higher_prob = st.sidebar.slider(
        "Limite de Probabilidade Altíssima",
        min_value=0.5,
        max_value=1.0,
        value=0.95,
        step=0.01,
    )
    high_prob = st.sidebar.slider(
        "Limite de Probabilidade Alta",
        min_value=0.5,
        max_value=1.0,
        value=0.90,
        step=0.01,
    )
    medium_prob = st.sidebar.slider(
        "Limite de Probabilidade Média",
        min_value=0.5,
        max_value=1.0,
        value=0.70,
        step=0.01,
    )

    incentives_list = st.sidebar.text_input(
        "Lista de Incentivos (separados por vírgula)", "200,100,50"
    )
    incentives_list = list(map(int, incentives_list.split(",")))

    investment_value = st.sidebar.number_input(
        "Valor de Investimento para Ambos os Cenários",
        min_value=1000,
        max_value=100000,
        value=10000,
    )

    simulation_1 = top_clients(
        scenario_name="Cenário 1",
        data=df_simulation,
        probability="probability",
        prediction="prediction",
        clients_return="financial_return",
        number_of_clients=top_n,
        incentive_value=investment_value / top_n,
        max_investment=investment_value,
    )

    df_simulation_2 = df_simulation[df_simulation["prediction"] == 1].copy()

    incentives = [
        incentives_list[0]
        if row["probability"] >= higher_prob
        else incentives_list[1]
        if high_prob <= row["probability"] < higher_prob
        else incentives_list[2]
        if medium_prob <= row["probability"] < high_prob
        else 0  # No incentive
        for _, row in df_simulation_2.iterrows()
    ]
    df_simulation_2["incentive"] = incentives

    simulation_2 = knapsack_solver(
        scenario_name="Cenário 2",
        data=df_simulation_2,
        probability="probability",
        prediction="prediction",
        clients_return="financial_return",
        number_of_clients=top_n,
        W=investment_value,
        incentive_value="incentive",
        max_investment=investment_value,
    )

    compare_df = pd.concat([simulation_1, simulation_2], axis=0)
    compare_df.reset_index(drop=True, inplace=True)
    compare_df = compare_df.T

    st.write(compare_df)

    st.markdown("""
    ### Explicação das Métricas

    1. **Investimento:**
        - **O que é?**: É o total de dinheiro que decidimos investir em incentivos para reter os clientes mais propensos a churn.
        - **Como foi calculada?**: É a soma dos incentivos financeiros oferecidos a cada cliente selecionado para retenção.

    2. **Lucro:**
        - **O que é?**: O lucro representa o valor que conseguimos "ganhar" após subtrair o custo dos incentivos oferecidos da receita recuperada.
        - **Como foi calculada?**: Para cada cliente, subtraímos o valor do incentivo do retorno financeiro que ele gerou, e depois somamos esses valores para todos os clientes.

    3. **ROI (%):**
        - **O que é?**: O Retorno sobre Investimento (ROI) mostra o quanto ganhamos em relação ao que investimos em retenção, expresso em porcentagem.
        - **Como foi calculada?**: Dividimos o "Lucro" total pelo "Investimento" total e multiplicamos por 100 para obter a porcentagem.

    4. **Redução do Churn (%):**
        - **O que é?**: Essa métrica indica a porcentagem de redução no churn, ou seja, quanto conseguimos reduzir a saída de clientes com nossas ações
        """)


if __name__ == "__main__":
    run()
