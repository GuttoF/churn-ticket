import pandas as pd

def top_clients_result(
    scenario_name: str,
    data: pd.DataFrame,
    probability: str,
    prediction: str,
    clients_return: str,
    churn_loss: float,
    number_of_clients: int,
    incentive_value: float,
):
    """
    Calculates the expected return and identifies the top clients to target for retention.

    Parameters:
    - scenario_name (str): Name of the scenario.
    - data (pd.DataFrame): Additional data for the scenario.
    - probability (str): Probability column name in the dataframe.
    - clients_return (str): Column name for the clients' return in the dataframe.
    - churn_loss (float): Estimated loss due to churn per client.
    - number_of_clients (int): Number of top clients to target for retention.
    - incentive_value (float): Incentive value offered to the clients for retention.

    Returns:
    - top_clients_df (DataFrame): DataFrame containing the top clients to target for retention.
    - expected_return (float): Expected return in terms of revenue from the targeted clients.

    """
    # sort values by probability
    data.sort_values(by=probability, ascending=False, inplace=True)

    # select the top clients with the highest probability
    top_value = data.iloc[:number_of_clients, :].copy()  # Create a copy explicitly

    # send an incentive
    top_value.loc[:, "incentive"] = incentive_value

    # recover per client
    top_value.loc[:, "recover"] = top_value.apply(
        lambda x: x[clients_return] if x["exited"] == 1 else 0, axis=1
    )

    # profit
    top_value.loc[:, "profit"] = top_value["recover"] - top_value["incentive"]

    # total recovered
    recovered_revenue = round(top_value["recover"].sum(), 2)

    # loss recovered in percentage
    loss_recovered = round(recovered_revenue / churn_loss * 100, 2)

    # sum of incentives
    sum_incentives = round(top_value["incentive"].sum(), 2)

    # profit sum
    profit_sum = round(top_value["profit"].sum(), 2)

    # ROI
    roi = round(profit_sum / sum_incentives * 100, 2)

    # calculate possible churn reduction in percentage
    churn_by_model = top_value[
        (top_value["exited"] == 1) & (top_value[prediction] == 1)
    ]
    churn_real = round((len(churn_by_model) / len(data[data["exited"] == 1])) * 100, 2)

    dataframe = pd.DataFrame(
        {
            "Scenario": scenario_name,
            "Recovered Revenue": "$" + str(recovered_revenue),
            "Loss Recovered": str(loss_recovered) + "%",
            "Investment": "$" + str(sum_incentives),
            "Profit": "$" + str(profit_sum),
            "ROI": str(roi) + "%",
            "Clients Recovered": str(len(churn_by_model)) + " clients",
            "Churn Reduction": str(churn_real) + "%",
        },
        index=[0],
    )

    return dataframe


def knapsack_solver(
    scenario_name: str,
    data: pd.DataFrame,
    prediction: str,
    clients_return: str,
    churn_loss: float,
    W: int,
    incentive: list,
):
    """
    A knapsack problem algorithm is a constructive approach to combinatorial optimization. Given set of items, each with a specific weight and a value. The algorithm determine each item's number to include with a total weight is less than a given limit.
    reference: https://www.geeksforgeeks.org/python-program-for-dynamic-programming-set-10-0-1-knapsack-problem/
    Parameters:
        scenario_name (str): [Name of the scenario]
        data (pd.DataFrame): [Additional data for the scenario]
        prediction (str): [Prediction column name in the dataframe]
        clients_return (str): [Column name for the clients' return in the dataframe]
        churn_loss (float): [Estimated loss due to churn per client]
        W (int): [Maximum weight capacity]
        incentive (list): [List of the incentive values offered to the clients for retention]
    Returns:
        [DataFrame]: [A dataframe with the results calculated]
    """
    # filter clients in churn according model
    data = data[data[prediction] == 1]

    # set parameters for the knapsack function
    val = data[clients_return].astype(int).values  # return per client
    wt = data[incentive].values  # incentive value per client

    # number of items in values
    n = len(val)

    # set K with 0 values
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

    # select items that maximizes the output
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

    # dataframe with selected clients that maximizes output value
    data = data[keep]

    # Recover per client
    data["recover"] = data.apply(
        lambda x: x[clients_return] if x["exited"] == 1 else 0, axis=1
    )

    # Calculate profit
    data["profit"] = data["recover"] - data["incentive"]

    # Calculate the total recovered
    recovered_revenue = round(data["recover"].sum(), 2)

    # Calculate loss recovered in percent
    loss_recovered = round((recovered_revenue / churn_loss) * 100, 2)

    # Calculate the sum of incentives
    sum_incentives = round(data["incentive"].sum(), 2)

    # Calculate profit sum
    profit = round(data["profit"].sum(), 2)

    # Calculate ROI in percent
    roi = round((profit / sum_incentives) * 100, 2)

    # Calculate possible churn reduction in %
    churn_by_model = data[(data["exited"] == 1) & (data[prediction] == 1)]
    churn_real = round((len(churn_by_model) / len(data[data["exited"] == 1])) * 100, 2)

    dataframe = pd.DataFrame(
        {
            "Scenario": scenario_name,
            "Recovered Revenue": "$" + str(recovered_revenue),
            "Loss Recovered": str(loss_recovered) + "%",
            "Investment": "$" + str(sum_incentives),
            "Profit": "$" + str(profit),
            "ROI": str(roi) + "%",
            "Clients Recovered": str(len(churn_by_model)) + " clients",
            "Churn Reduction": str(churn_real) + "%",
        },
        index=[0],
    )

    del K
    return dataframe