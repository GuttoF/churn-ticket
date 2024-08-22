import streamlit as st
import pandas as pd
import duckdb
from pathlib import Path
from utils.utils_eda_page import DataVisualizer
from utils.utils_fe_page import FeatureEngineering

def text_step(subheader: str, markdown: str):
    st.subheader(f"Hipótese {subheader}")
    st.markdown(markdown)

def run():
    st.title("Análise Exploratória de Dados (EDA)")

    # Ler o DataFrame
    path = Path().resolve().parent
    # Local path
    #data_path = path / "churn-ticket/data/interim"
    # Docker path
    data_path = path / "app/data/interim"
    conn_path = str(data_path / "churn.db")
    conn = duckdb.connect(database=conn_path, read_only=False)
    query = conn.execute("SELECT * FROM churn")
    df = pd.DataFrame(query.fetchdf())
    conn.close()

    fe = FeatureEngineering()
    df = fe._perform_transformations(df)

    eda = DataVisualizer(df)

    h1_sub = "1: Clientes com pontuação de crédito abaixo da média têm maior probabilidade de sair do banco."
    h1_mark = "**Verdadeiro:** Clientes com uma pontuação de crédito abaixo da média são mais propensos a deixar o banco."
    text_step(h1_sub, h1_mark)
    eda.hypotheses_1()

    h2_sub = "2: Os clientes adultos têm um ou dois produtos bancários."
    h2_mark = "**Verdadeiro** Semelhante às outras faixas etárias, clientes adultos, geralmente, usam 1 ou 2 produtos bancários."
    text_step(h2_sub, h2_mark)
    eda.hypotheses_2()

    h3_sub = "3: O saldo médio da conta varia significativamente entre as diferentes regiões geográficas."
    h3_mark = "**Verdadeiro** O saldo médio da conta varia significativamente entre diferentes regiões geográficas."
    text_step(h3_sub, h3_mark)
    eda.hypotheses_3()

    h4_sub = "4: Os homens têm maior probabilidade de serem membros activos do que as mulheres."
    h4_mark = "**Verdadeiro** Em um pouco de diferença, os homens são mais propensos a serem membros ativos"
    text_step(h4_sub, h4_mark)
    eda.hypotheses_4()

    h5_sub = "5: Clientes com salários mais altos têm maior probabilidade de ter cartão de crédito."
    h5_mark = "**Falso** A maioria dos clientes tem cartão de crédito, independentemente do salário."
    text_step(h5_sub, h5_mark)
    eda.hypotheses_5()

    h6_sub = "6: Clientes com maior tempo de permanência têm menos probabilidade de deixar o banco."
    h6_mark = "**Falso** A taxa de rotatividade é semelhante para clientes com diferentes mandatos."
    text_step(h6_sub, h6_mark)
    eda.hypotheses_6()

    h7_sub = "7: Clientes com saldos mais altos têm maior probabilidade de sair do banco."
    h7_mark = "**Verdadeiro** Clientes com saldos mais altos têm maior probabilidade de sair do banco."
    text_step(h7_sub, h7_mark)
    eda.hypotheses_7()

    h8_sub = "8: Clientes mais velhos têm maior probabilidade de ter cartão de crédito."
    h8_mark = "**Falso** A maioria dos clientes têm cartão de crédito, independente da idade."
    text_step(h8_sub, h8_mark)
    eda.hypotheses_8()

    st.subheader("Conclusão")
    st.markdown("""
    A análise exploratória revelou alguns insights importantes sobre o comportamento dos clientes:

    - O país onde o cliente mora aparenta ser relevante, dado o valor de saldo médio.
    - Clientes adultos têm mais produtos bancários do que seniors e adolescentes.
    - Como a maioria dos clientes possuem cartão de crédito (cerca de 70%), essa variável não é tão relevante para previsão de churn.
    - O tempo de permanência no banco não aparenta ter uma relevância significativa no churn.

    Essas informações podem ser usadas para melhorar as estratégias de retenção de clientes e ajustar o foco das campanhas de marketing.
    """)

if __name__ == "__main__":
    run()
