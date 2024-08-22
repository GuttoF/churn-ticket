import streamlit as st
from page_topics import home, eda


st.set_page_config(page_title="Previsão de Churn da Top Bank",
                   page_icon=":bank:",
                   layout="wide")

PAGES = ["Home", "Análise Exploratória de Dados"]

def main():
    # Add lateral bar
    st.sidebar.title("Navegação")
    selection = st.sidebar.selectbox("Ir para", PAGES)

    if selection == "Home":
        home.run()
    elif selection == "Análise Exploratória de Dados":
        eda.run()


if __name__ == "__main__":
    main()