import streamlit as st
import plotly.express as px
import pandas as pd
import plotly.graph_objects as go

def run():
    st.title("📊 Informações")

    st.header("🔍 Visão Geral")
    st.write("""
    Este projeto implementa um sistema de previsão de churn bancário utilizando:
    - **Modelo**: Modelo de Machine Learning CatBoost
    - **Ciclo de Vida**: 20 dias de treino + 10 dias de teste + retreinamento contínuo e monitoramento
    - **Conformidade**: LGPD (dados anonimizados)
    """)

    st.header("⏳ Cronograma do Projeto")

    df_timeline = pd.DataFrame([
        {"Task": "Treinamento", "Start": "2025-04-01", "Finish": "2025-04-20"},
        {"Task": "Teste/Validação", "Start": "2025-04-21", "Finish": "2025-05-01"},
        {"Task": "Implantação", "Start": "2025-05-02", "Finish": "2025-05-15"},
        {"Task": "Ciclo de Retreino", "Start": "2025-05-16", "Finish": "2025-06-15"},
        {"Task": "Ciclo de Retreino", "Start": "2025-06-16", "Finish": "2025-07-15"},
        {"Task": "Ciclo de Retreino", "Start": "2025-07-16", "Finish": "2025-08-15"},
        {"Task": "Ciclo de Retreino", "Start": "2025-08-16", "Finish": "2025-09-15"},
        {"Task": "Ciclo de Retreino", "Start": "2025-09-16", "Finish": "2025-10-15"},
        {"Task": "Ciclo de Retreino", "Start": "2025-10-16", "Finish": "2025-11-15"},
        {"Task": "Ciclo de Retreino", "Start": "2025-11-16", "Finish": "2025-12-15"},
        {"Task": "Ciclo de Retreino", "Start": "2025-12-16", "Finish": "2026-01-15"},
        {"Task": "Monitoramento", "Start": "2025-05-16", "Finish": "2026-01-15"}
    ])

    df_timeline["Start"] = pd.to_datetime(df_timeline["Start"])
    df_timeline["Finish"] = pd.to_datetime(df_timeline["Finish"])

    fig = px.timeline(df_timeline, x_start="Start", x_end="Finish", y="Task",
                      color="Task", title="📅 Fases do Projeto (Abril 2025 - Janeiro 2026)")

    fig.update_yaxes(title="Fase")
    fig.update_xaxes(title="Data", tickformat="%b %d %Y")
    fig.update_layout(height=400, showlegend=False)

    st.plotly_chart(fig, use_container_width=True)

    nodes = [
        "Início", "Treinamento", "Teste/Validação", "Implantação",
        "Ciclo de Retreino", "Monitoramento", ""
    ]

    sources = [0, 1, 2, 3, 3, 4, 5]
    targets = [1, 2, 3, 4, 5, 6, 6]
    values =  [1, 1, 1, 0.8, 0.2, 1, 1]

    labels = [
        "Início (01/04/2025)",
        "20 dias (01/04-20/04/2025)",
        "10 dias (21/04-01/05/2025)",
        "Produção (02/05-15/05/2025)",
        "8 ciclos mensais (16/05/2025-15/01/2026)",
        "Paralelo (16/05/2025-15/01/2026)",
        "Finalização"
    ]

    fig = go.Figure(go.Sankey(
        node=dict(
            pad=20, thickness=25, line=dict(color='black', width=1),
            label=nodes,
        ),
        link=dict(
            source=sources, target=targets, value=values,
            label=labels
        )
    ))

    fig.update_layout(

        height=400,
        font=dict(size=12)
    )

    st.plotly_chart(fig, use_container_width=True)

    with st.expander("Detalhes dos Ciclos de Retreino"):
        st.markdown("""
        **Ciclos de Retreino**:
        - Ciclo 1: 16/05/2025 - 15/06/2025
        - Ciclo 2: 16/06/2025 - 15/07/2025
        - Ciclo 3: 16/07/2025 - 15/08/2025
        - Ciclo 4: 16/08/2025 - 15/09/2025
        - Ciclo 5: 16/09/2025 - 15/10/2025
        - Ciclo 6: 16/10/2025 - 15/11/2025
        - Ciclo 7: 16/11/2025 - 15/12/2025
        - Ciclo 8: 16/12/2025 - 15/01/2026
        """)

    st.header("🛠️ Stack Tecnológico")
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Principais Bibliotecas")
        st.markdown("""
        - Python
        - CatBoost
        - FastAPI
        - Pandas/Numpy
        """)

    with col2:
        st.subheader("Infraestrutura")
        st.markdown("""
        - Docker
        - Kubernetes
        - DuckDB
        - GitHub Actions (CI/CD)
        """)

    st.header("🛡️ Conformidade com LGPD")
    st.markdown("""
    - **Anonimização**: Dados sensíveis são pseudonimizados
    - **Acesso**: Restrito a equipe autorizada
    - **Retenção**: Dados são apagados após 6 meses de inatividade
    - **Transparência**: Clientes podem solicitar acesso aos dados
    """)
    st.success("✅ Projeto totalmente compatível com a Lei Geral de Proteção de Dados (LGPD)")





if __name__ == "__main__":
    run()
