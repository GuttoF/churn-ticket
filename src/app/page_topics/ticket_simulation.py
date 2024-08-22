import streamlit as st
from utils.utils_business_translation import get_churn_predictions_from_api, top_clients, knapsack_solver
import pandas as pd

def run():

    st.title(":moneybag: Simulação de Retenção de Clientes Utilizando Tickets")

    st.markdown("""
    Vamos fazer algumas considerações sobre o retorno financeiro dos clientes.
    - Retorno de 15% se o salário estimado for menor que a média;
    - Retorno de 20% se o salário estimado for igual à média e menor que duas vezes a média;
    - Retorno de 25% se o salário estimado for igual ou maior que duas vezes a média;
    
    Com isso obtemos as seguintes informações:
    
    """)

    return_clients, churn_loss, total_return, df_simulation = get_churn_predictions_from_api()

    result_return_clients = f"€{return_clients}"
    result_churn_loss = f"€{churn_loss}"
    result_total_return = f"{total_return}%"
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

    st.sidebar.subheader("Personalizar Simulação")

    top_n = st.sidebar.number_input("Quantidade de Clientes no Top", min_value=10, max_value=500, value=20, step=10)

    incentives_list = st.sidebar.text_input("Lista de Incentivos (separados por vírgula)", "200,100,50")
    incentives_list = list(map(int, incentives_list.split(',')))

    investment_value = st.sidebar.number_input("Valor de Investimento para Ambos os Cenários", min_value=1000, max_value=100000, value=10000)

    simulation_1 = top_clients(
        scenario_name="Cenário 1",
        data=df_simulation,
        probability="probability",
        prediction="prediction",
        clients_return="financial_return",
        churn_loss=churn_loss,
        number_of_clients=top_n,
        incentive_value=investment_value / top_n,
        max_investment=investment_value
    )

    df_simulation_2 = df_simulation[df_simulation["prediction"] == 1].copy()

    incentives = [
        incentives_list[0] if row["probability"] >= 0.95 else
        incentives_list[1] if 0.90 <= row["probability"] < 0.95 else
        incentives_list[2]
        for _, row in df_simulation_2.iterrows()
    ]
    df_simulation_2["incentive"] = incentives

    simulation_2 = knapsack_solver(
        scenario_name="Cenário 2",
        data=df_simulation_2,
        probability="probability",
        prediction="prediction",
        clients_return="financial_return",
        churn_loss=churn_loss,
        number_of_clients=top_n,
        W=investment_value,
        incentive_value="incentive",
        max_investment=investment_value
    )

    compare_df = pd.concat([simulation_1, simulation_2], axis=0)
    compare_df.reset_index(drop=True, inplace=True)
    compare_df = compare_df.T

    st.write(compare_df)

    st.markdown("""
    ### Explicação das Métricas

    1. **Receita Recuperada:**
        - **O que é?**: A receita recuperada representa o valor total que conseguimos salvar ao evitar que clientes propensos a churn realmente saiam.
        - **Como foi calculada?**: Para cada cliente, calculamos o retorno financeiro esperado se ele permanecer no banco. Somamos o retorno de todos os clientes que foram corretamente identificados como churners e que, com as ações de retenção, permaneceram.

    2. **Investimento:**
        - **O que é?**: É o total de dinheiro que decidimos investir em incentivos para reter os clientes mais propensos a churn.
        - **Como foi calculada?**: É a soma dos incentivos financeiros oferecidos a cada cliente selecionado para retenção.

    3. **Lucro:**
        - **O que é?**: O lucro representa o valor que conseguimos "ganhar" após subtrair o custo dos incentivos oferecidos da receita recuperada.
        - **Como foi calculada?**: Para cada cliente, subtraímos o valor do incentivo do retorno financeiro que ele gerou, e depois somamos esses valores para todos os clientes.

    4. **ROI (%):**
        - **O que é?**: O Retorno sobre Investimento (ROI) mostra o quanto ganhamos em relação ao que investimos em retenção, expresso em porcentagem.
        - **Como foi calculada?**: Dividimos o "Lucro" total pelo "Investimento" total e multiplicamos por 100 para obter a porcentagem.

    5. **Redução do Churn (%):**
        - **O que é?**: Essa métrica indica a porcentagem de redução no churn, ou seja, quanto conseguimos reduzir a saída de clientes com nossas ações.
        - **Como foi calculada?**: Calculamos a porcentagem de clientes que estavam prestes a churnar, mas que foram retidos com sucesso, em relação ao número total de clientes com probabilidade de churn.
    """)

if __name__ == "__main__":
    run()
