import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.metrics import auc, confusion_matrix, roc_curve


# Definir a função para gerar o gráfico
def model_plot(model, X_val: pd.DataFrame, y_val: pd.Series, threshold: float = 0.5):
    """
    Generate a model evaluation plot with confusion matrix and ROC curve.

    Parameters:
    - model (object): Fitted model object.
    - X_val (pd.DataFrame): Validation features.
    - y_val (pd.Series): Validation target.
    - threshold (float): Classification threshold (default = 0.5).

    Returns:
    - fig (plotly.graph_objects.Figure): Plotly figure object.
    """

    # Define class names
    class_names = ["Not Churn", "Churn"]

    y_prob = model.predict_proba(X_val)[:, 1]
    y_pred = (y_prob >= threshold).astype(int)

    # Compute ROC curve and ROC area
    fpr, tpr, _ = roc_curve(y_val, y_prob)
    roc_auc = auc(fpr, tpr)

    # Compute confusion matrix
    cm = confusion_matrix(y_val, y_pred)

    # Create subplots
    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=("Matriz de Confusão", "Curva ROC"),
        specs=[[{"type": "heatmap"}, {"type": "scatter"}]],
    )

    # Add confusion matrix
    fig.add_trace(
        go.Heatmap(
            z=cm,
            x=class_names,
            y=class_names,
            colorscale="Blues",
            showscale=True,
            text=cm,
            texttemplate="%{text}",
            hovertemplate="Verdadeiro: %{y}<br>Previsto: %{x}<br>Contagem: %{z}<extra></extra>",
            colorbar=dict(title="Count", thickness=15, x=0.45),
        ),
        row=1,
        col=1,
    )

    # Add ROC curve
    fig.add_trace(
        go.Scatter(
            x=fpr,
            y=tpr,
            mode="lines",
            name=f"ROC curve (area = {roc_auc:.2f})",
            line=dict(color="darkorange", width=2),
        ),
        row=1,
        col=2,
    )
    fig.add_trace(
        go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode="lines",
            name="Adivinhação Aleatória",
            line=dict(color="navy", width=2, dash="dash"),
        ),
        row=1,
        col=2,
    )

    # Update layout
    fig.update_xaxes(title_text="Previsto", row=1, col=1)
    fig.update_yaxes(title_text="Verdade", row=1, col=1)
    fig.update_xaxes(title_text="Taxa de Falsos Positivos", row=1, col=2)
    fig.update_yaxes(title_text="Taxa de Verdadeiros Positivos", row=1, col=2)
    fig.update_layout(
        title_text="Model Evaluation",
        height=500,
        width=1000,
        showlegend=False,
        template="plotly_white",
    )

    return fig
