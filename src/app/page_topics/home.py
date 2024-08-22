import streamlit as st

def run():
    st.title(":bank: Previsão de Churn da Top Bank")
    st.write("""
    Este é um aplicativo simples que prevê o **Churn** de clientes em um banco.
    Antes de começarmos, vamos entender algumas coisas.
    
    **O que é churn?**  
    Churn refere-se ao momento em que um cliente deixa de fazer negócios com uma empresa. Entender quais clientes têm maior probabilidade de churn pode ajudar as empresas a tomarem medidas proativas para retê-los.

    """)

    st.markdown(
        """
        **Colunas do banco de dados**
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
        - **estimated_salary**: Estimativadosalárioanualdosclientes;
        - **exited**: Se o cliente for um churn (*variável alvo*)
        """
            )

if __name__ == "__main__":
    run()