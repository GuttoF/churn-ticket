import streamlit as st

def run():
    st.title(":bank: Previsão de Churn da Top Bank")

    st.write("""
    Este aplicativo utiliza um modelo de machine learning para prever o **Churn** dos clientes da Top Bank.

    **O que é Churn?**  
    Churn refere-se ao momento em que um cliente decide deixar de fazer negócios com a empresa. Identificar quais clientes têm maior probabilidade de churn permite que a empresa tome ações proativas para retê-los.

    **Como funciona o modelo de Machine Learning?**  
    Machine learning é uma tecnologia que permite que computadores aprendam a partir de dados. Em vez de seguir instruções explícitas, o modelo aprende padrões a partir dos dados históricos para fazer previsões sobre novos dados.
    """)

    st.subheader(":floppy_disk: Dicionário do Banco de Dados")
    st.markdown(
        """
        - **row_number**: O número das colunas;
        - **customer_id**: Identificador único de clientes;
        - **sobrenome**: Sobrenome do cliente;
        - **credit_score**: Pontuação de crédito dos clientes para o mercado financeiro;
        - **geography**: O país do cliente;
        - **gender**: O gênero do cliente;
        - **age**: A idade do cliente;
        - **tenure**: Número de anos que o cliente está no banco;
        - **balance**: O valor que o cliente tem em sua conta;
        - **num_of_products**: A quantidade de produtos que o cliente comprou;
        - **has_cr_card**: Se o cliente tiver cartão de crédito;
        - **isActiveMember**: Se o cliente estiver ativo (nos últimos 12 meses);
        - **estimated_salary**: Estimativa do salário anual dos clientes;
        - **exited**: Se o cliente for um churn (*variável alvo*)
        """
    )


if __name__ == "__main__":
    run()
