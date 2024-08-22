import requests
import pandas as pd
from pathlib import Path
import duckdb
from utils.utils_fe_page import FeatureEngineering

def get_churn_predictions_from_api(api_url="http://api:8000/predict"):
    path = Path().resolve().parent
    # Local path
    #data_path = path / "churn-ticket/data"
    # Docker path
    data_path = path / "app/data"

    conn_path = str(data_path / "interim/churn.db")
    conn = duckdb.connect(database=conn_path, read_only=False)
    query = conn.execute("SELECT * FROM churn")
    df_raw = pd.DataFrame(query.fetchdf())
    conn.close()

    X_test_old = pd.read_parquet(data_path / "processed/X_test.parquet")
    estimated_salary = X_test_old["estimated_salary"]

    X_test = pd.read_parquet(data_path / "processed/X_test.parquet")

    fe = FeatureEngineering()
    X_test = fe._perform_transformations(X_test)

    y_test = pd.read_pickle(data_path / "processed/y_test.pkl")

    predictions = []
    probabilities = []

    for _, row in X_test.iterrows():
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

        response = requests.post(api_url, json=input_data)
        if response.status_code == 200:
            result = response.json()
            predictions.append(result['prediction'])
            probabilities.append(result['probability'])
        else:
            predictions.append(None)
            probabilities.append(None)

    # Create a DataFrame for the results
    y_hat = pd.DataFrame(predictions, columns=['prediction'])
    y_hat_proba = pd.DataFrame(probabilities, columns=['probability'])

    def return_per_client():
        df_raw["estimated_salary_regular"] = estimated_salary

        # df_raw is a dataframe before the transformations
        salary_mean = round(df_raw["estimated_salary"].mean(), 2)

        # Predictions and Results
        y_test_frame = y_test.to_frame().reset_index(drop=True)
        y_proba = y_hat_proba.reset_index(drop=True)
        y_predict = y_hat.reset_index(drop=True)

        # Estimated salary without mms
        estimated_salary_frame = estimated_salary.to_frame().reset_index(drop=True)

        # Creating a dataframe with the results
        df_simulation = pd.concat(
            (y_test_frame, y_proba, y_predict, estimated_salary_frame), axis=1
        )

        # Verify threshold
        df_simulation["threshold"] = df_simulation["probability"].apply(
            lambda x: "negative" if x <= 0.4 else "positive"
        )

        # Reorder columns
        df_simulation = df_simulation[
            ["estimated_salary", "exited", "prediction", "probability", "threshold"]
        ]

        df_simulation["financial_return"] = df_simulation["estimated_salary"].apply(
            lambda x: x * 0.15
            if x < salary_mean
            else x * 0.2
            if salary_mean <= x < 2 * salary_mean
            else x * 0.25
        )

        return_clients = round(df_simulation["financial_return"].sum(), 2)

        churn_loss = round(
            df_simulation[df_simulation["exited"] == 1]["financial_return"].sum(), 2
        )

        total_return = round((churn_loss / return_clients) * 100, 2)
        return return_clients, churn_loss, total_return, df_simulation

    return return_per_client()


def top_clients(
    scenario_name: str,
    data: pd.DataFrame,
    probability: str,
    prediction: str,
    clients_return: str,
    churn_loss: float,
    number_of_clients: int,
    incentive_value: float,
    max_investment: float,
):
    """
    Calcula o retorno esperado e identifica os principais clientes a serem direcionados para retenção.

    Parâmetros:
    - scenario_name (str): Nome do cenário.
    - data (pd.DataFrame): Dados adicionais para o cenário.
    - probability (str): Nome da coluna de probabilidade no DataFrame.
    - prediction (str): Nome da coluna de predição no DataFrame.
    - clients_return (str): Nome da coluna que contém o retorno financeiro dos clientes.
    - churn_loss (float): Perda estimada devido ao churn por cliente.
    - number_of_clients (int): Número de clientes principais a serem direcionados.
    - incentive_value (float): Valor do incentivo oferecido aos clientes para retenção.
    - max_investment (float): Valor máximo de investimento permitido.

    Retorna:
    - DataFrame contendo os principais clientes a serem direcionados para retenção.
    """
    data.sort_values(by=probability, ascending=False, inplace=True)

    top_value = data.iloc[:number_of_clients, :].copy()

    total_incentive = incentive_value * number_of_clients

    if total_incentive > max_investment:
        incentive_value = max_investment / number_of_clients
        total_incentive = incentive_value * number_of_clients

    top_value["incentive"] = incentive_value

    top_value["recover"] = top_value.apply(
        lambda x: x[clients_return] if x[prediction] == 1 else 0, axis=1
    )

    top_value["profit"] = top_value["recover"] - top_value["incentive"]

    recovered_revenue = round(top_value["recover"].sum(), 2)

    roi = round(top_value["profit"].sum() / total_incentive * 100, 2)

    churn_reduction = round(
        len(top_value[(top_value[prediction] == 1) & (top_value["exited"] == 1)]) / number_of_clients * 100, 2
    )

    dataframe = pd.DataFrame(
        {
            "Cenário": scenario_name,
            "Receita Recuperada": f"€{recovered_revenue}",
            "Investimento": f"€{total_incentive:.2f}",
            "Lucro": f"€{top_value['profit'].sum():.2f}",
            "ROI": f"{roi}%",
            "Redução do Churn": f"{churn_reduction}%",
        },
        index=[0],
    )

    return dataframe

def knapsack_solver(
    scenario_name: str,
    data: pd.DataFrame,
    probability: str,
    prediction: str,
    clients_return: str,
    churn_loss: float,
    number_of_clients: int,
    W: int,
    incentive_value: str,
    max_investment: float,
):
    """
    Solucionador do problema da mochila para otimização do valor de retenção dos clientes.

    Parâmetros:
    - scenario_name (str): Nome do cenário.
    - data (pd.DataFrame): Dados adicionais para o cenário.
    - probability (str): Nome da coluna de probabilidade no DataFrame.
    - prediction (str): Nome da coluna de predição no DataFrame.
    - clients_return (str): Nome da coluna que contém o retorno financeiro dos clientes.
    - churn_loss (float): Perda estimada devido ao churn por cliente.
    - number_of_clients (int): Número de clientes principais a serem direcionados.
    - W (int): Capacidade máxima de investimento.
    - incentive_value (str): Nome da coluna que contém o valor do incentivo.
    - max_investment (float): Valor máximo de investimento permitido.

    Retorna:
    - DataFrame contendo os clientes selecionados com o retorno máximo.
    """
    data = data[data[prediction] == 1]

    data = data.sort_values(by=probability, ascending=False).head(number_of_clients)

    total_incentive = data[incentive_value].sum()
    if total_incentive > max_investment:
        scaling_factor = max_investment / total_incentive
        data[incentive_value] = data[incentive_value] * scaling_factor

    val = data[clients_return].astype(int).values
    wt = data[incentive_value].values
    n = len(val)

    K = [[0 for x in range(W + 1)] for x in range(n + 1)]

    for i in range(n + 1):
        for w in range(W + 1):
            if i == 0 or w == 0:
                K[i][w] = 0
            elif wt[i - 1] <= w:
                K[i][w] = max(val[i - 1] + K[i - 1][w - wt[i - 1]], K[i - 1][w])
            else:
                K[i][w] = K[i - 1][w]

    max_val = K[n][W]

    keep = [False] * n
    res = max_val
    w = W
    for i in range(n, 0, -1):
        if res <= 0:
            break
        if res == K[i - 1][w]:
            continue
        else:
            keep[i - 1] = True
            res = res - val[i - 1]
            w = w - wt[i - 1]

    selected_data = data[keep]

    selected_data["recover"] = selected_data.apply(
        lambda x: x[clients_return] if x["exited"] == 1 else 0, axis=1
    )
    selected_data["profit"] = selected_data["recover"] - selected_data[incentive_value]

    recovered_revenue = round(selected_data["recover"].sum(), 2)

    roi = round(selected_data["profit"].sum() / selected_data[incentive_value].sum() * 100, 2)

    churn_reduction = round(
        len(selected_data[(selected_data[prediction] == 1) & (selected_data["exited"] == 1)]) / number_of_clients * 100, 2
    )

    dataframe = pd.DataFrame(
        {
            "Cenário": scenario_name,
            "Receita Recuperada": f"€{recovered_revenue}",
            "Investimento": f"€{selected_data[incentive_value].sum():.2f}",
            "Lucro": f"€{selected_data['profit'].sum():.2f}",
            "ROI": f"{roi}%",
            "Redução do Churn": f"{churn_reduction}%",
        },
        index=[0],
    )

    del K
    return dataframe
