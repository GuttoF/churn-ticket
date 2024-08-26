import requests
import streamlit as st
import os

def run():
    st.title(":robot_face: Previsão de Churn")

    st.subheader("Insira os dados do cliente para prever o churn")

    with st.container():
        credit_score = st.number_input("Pontuação de Crédito", min_value=100)
        geography = st.selectbox("Geografia", ["Germany", "France", "Spain"])
        gender = st.selectbox("Gênero", ["Male", "Female"])
        age = st.number_input("Idade", min_value=18, max_value=100)
        tenure = st.number_input("Tempo de Permanência", min_value=0, max_value=10)
        balance = st.number_input("Saldo", min_value=0.0)
        num_of_products = st.number_input(
            "Número de Produtos", min_value=1, max_value=4
        )
        has_cr_card = st.selectbox("Possui Cartão de Crédito?", [0, 1])
        is_active_member = st.selectbox("É Membro Ativo?", [0, 1])
        estimated_salary = st.number_input("Salário Estimado", min_value=0.0)

    # Make prediction
    if st.button("Clique aqui para realizar a previsão"):
        # Entry data
        input_data = {
            "credit_score": credit_score,
            "geography": geography,
            "gender": gender,
            "age": age,
            "tenure": tenure,
            "balance": balance,
            "num_of_products": num_of_products,
            "has_cr_card": has_cr_card,
            "is_active_member": is_active_member,
            "estimated_salary": estimated_salary,
        }

        # API URL
        api_url = os.getenv("API_URL")
        # api_url = "http://127.0.0.1:8000/predict"
        if not api_url:
            raise ValueError("API_URL not found in environment variables")

        # POST
        response = requests.post(api_url, json=input_data)

        try:
            response_json = response.json()
            prediction = response_json["prediction"]
            if prediction == 1:
                st.warning("O cliente está propenso a sair do banco.")
            else:
                st.success("O cliente não está propenso a sair do banco.")

            probability = response_json["probability"]
            probability = "{:.2f}%".format(probability * 100)
            st.info(f"A probabilidade é de {probability}.")

        except requests.exceptions.JSONDecodeError:
            st.error("Erro ao tentar decodificar a resposta como JSON")


if __name__ == "__main__":
    run()
