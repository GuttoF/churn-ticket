import streamlit as st
from page_topics import home, model_explain, eda, predict, top_clients, ticket_simulation, about

st.set_page_config(page_title="Previsão de Churn da Top Bank",
                   page_icon=":bank:",
                   layout="wide")

PAGES = {
    "Home": home.run,
    "Explicação do Modelo": model_explain.run,
    "Análise Exploratória de Dados": eda.run,
    "Top Clients": top_clients.run,
    "Previsão de Churn": predict.run,
    "Simulação de Ticket": ticket_simulation.run,
    "Sobre": about.run
}

def main():
    # Add lateral bar
    st.sidebar.title("Navegação")
    selection = st.sidebar.selectbox("Ir para", list(PAGES.keys()))

    page = PAGES[selection]
    page()

if __name__ == "__main__":
    main()