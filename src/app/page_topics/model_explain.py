import pickle
from pathlib import Path

import pandas as pd
import streamlit as st
from utils.utils_model_explain import model_plot

# Paths
current_path = Path(__file__)
X_test_path = current_path.parents[3] / "data" / "processed" / "X_test_fs.parquet"
y_test_path = current_path.parents[3] / "data" / "processed" / "y_test.pkl"
model_path = current_path.parents[3] / "src" / "models" / "model.pkl"
threshold_path = current_path.parents[3] / "src" / "models" / "threshold.pkl"

# Load model and threshold
with open(model_path, "rb") as model_file:
    model = pickle.load(model_file)

with open(threshold_path, "rb") as threshold_file:
    threshold = pickle.load(threshold_file)

# Load data
X_test = pd.read_parquet(X_test_path)

with open(y_test_path, "rb") as y_test_file:
    y_test = pickle.load(y_test_file)


# Plot
fig = model_plot(model, X_test, y_test, threshold)


def run():
    st.title(":thinking_face: Explicação do Modelo")

    st.write("""
    **Modelo CatBoost**  
    O modelo utilizado neste aplicativo é o **CatBoost**. O CatBoost é uma técnica que cria várias "árvores de decisão" para tomar decisões baseadas nos dados. Essas árvores de decisão são como regras que ajudam a prever se um cliente vai ou não cancelar seus serviços.

    **Por que o CatBoost?**  
    O CatBoost é particularmente bom em lidar com dados categóricos, como o país ou gênero do cliente, e é capaz de capturar relacionamentos complexos entre as variáveis de forma eficiente. Isso resulta em previsões mais precisas e robustas, o que é essencial para entender e prever o comportamento dos clientes.

    **Como o modelo foi treinado?**  
    O modelo foi treinado usando dados históricos de clientes, incluindo informações como pontuação de crédito, saldo, número de produtos adquiridos, e outras características. A partir desses dados, o modelo aprendeu a identificar padrões que indicam uma maior probabilidade de churn.

    **Métrica Utilizada: Recall**  
    A métrica principal utilizada para avaliar o desempenho do modelo é o **recall**. O recall é uma medida que nos diz a capacidade do modelo em identificar corretamente os clientes que realmente irão entrar em churn.  
    Imagine que o modelo tem que escolher entre muitos clientes quais deles irão sair do banco. O recall nos mostra, entre todos os clientes que realmente cancelaram, quantos deles o modelo conseguiu prever corretamente. É uma métrica muito importante quando queremos minimizar a chance de perder clientes que poderiam ser retidos. O resultado do recall foi de **77,6%**, considerado como um valor satisfatório.

    **Curva ROC e AUC**  
    Outra ferramenta que usamos para avaliar o modelo é a **Curva ROC** (Receiver Operating Characteristic). Essa curva nos mostra como o modelo se comporta ao tentar distinguir entre clientes que irão entrar em churn e os que não irão.  
    A curva ROC traça a taxa de verdadeiros positivos (clientes corretamente identificados como churn) contra a taxa de falsos positivos (clientes incorretamente identificados como churn).  
    Um bom modelo terá uma curva ROC que se aproxima do canto superior esquerdo do gráfico. A área sob a curva ROC, conhecida como **AUC** (Area Under the Curve), nos dá uma ideia geral de quão bem o modelo está separando as duas classes (churn e não churn). Quanto mais próxima de 1 for a AUC, melhor o modelo.

    **Matriz de Confusão**  
    A matriz de confusão é uma ferramenta que usamos para avaliar a performance do modelo de forma detalhada. Ela mostra como as previsões do modelo se comparam com os valores reais em termos de **quatro categorias**:

    - **Verdadeiros Positivos (VP)**: O modelo previu corretamente que o cliente iria entrar em churn.
    - **Falsos Positivos (FP)**: O modelo previu que o cliente iria entrar em churn, mas ele não entrou.
    - **Verdadeiros Negativos (VN)**: O modelo previu corretamente que o cliente não iria entrar em churn.
    - **Falsos Negativos (FN)**: O modelo previu que o cliente não iria entrar em churn, mas ele entrou.
    """)

    st.subheader("Resultados do Modelo")

    st.plotly_chart(fig)

    st.markdown("""
    **Verdadeiros Positivos (VP = 324):**  
    O modelo identificou corretamente 324 clientes que realmente fizeram churn (ou seja, o modelo previu churn e o cliente realmente deixou o banco). Isso é positivo, pois esses são os clientes que você quer identificar para possíveis ações de retenção.
    
    **Falsos Positivos (FP = 370):**  
    O modelo previu que 370 clientes fariam churn, mas esses clientes, na realidade, não deixaram o banco. Isso pode ser um problema, dependendo do impacto de tomar ações de retenção em clientes que não estavam propensos a churn. No entanto, esses falsos positivos podem ser menos problemáticos se as ações de retenção forem de baixo custo ou se o objetivo principal for garantir que nenhum cliente importante seja perdido.
    
    **Verdadeiros Negativos (VN = 1223):**  
    O modelo corretamente identificou 1203 clientes que não fizeram churn (o modelo previu "não churn" e o cliente realmente permaneceu). Isso mostra que o modelo tem uma boa capacidade de identificar clientes fiéis.
    
    **Falsos Negativos (FN = 83):**  
    Existem 83 clientes que o modelo previu que permaneceriam no banco, mas que realmente fizeram churn. Este é um ponto de preocupação, especialmente se o objetivo principal do modelo for minimizar as perdas de clientes. Esses falsos negativos representam oportunidades perdidas de retenção.
    
    ### Por que esse resultado pode ser considerado bom?
    
    - **Alto Número de Verdadeiros Positivos e Verdadeiros Negativos:**  
      A maioria das previsões feitas pelo modelo são corretas, indicando uma boa performance geral. Um número relativamente alto de verdadeiros positivos (324) e verdadeiros negativos (1223) sugere que o modelo é eficaz em identificar corretamente os clientes churn e não churn.
    
    - **Baixo Número de Falsos Negativos:**  
      O número de falsos negativos (83) é relativamente baixo comparado ao número total de clientes, o que significa que o modelo é capaz de identificar a maioria dos clientes que estão em churn, minimizando o risco de perda de clientes importantes sem ação de retenção.

    """)


if __name__ == "__main__":
    run()
