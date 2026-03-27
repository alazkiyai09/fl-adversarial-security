"""
Reusable Plotly chart components for the dashboard.
Provides consistent styling and efficient rendering.
"""

import plotly.graph_objects as go
import plotly.express as px
import numpy as np
from typing import List, Dict, Tuple, Optional
from core.data_models import TrainingRound, ClientMetric


# Chart theme constants
COLORS = {
    "primary": "#667eea",
    "secondary": "#764ba2",
    "success": "#28a745",
    "warning": "#ffc107",
    "danger": "#dc3545",
    "info": "#17a2b8",
    "client_default": "#6c757d",
    "client_anomaly": "#dc3545",
    "background": "#f8f9fa"
}

LAYOUT_TEMPLATE = go.layout.Template()
LAYOUT_TEMPLATE.layout = {
    "font": {"family": "system-ui, -apple-system, sans-serif"},
    "paper_bgcolor": "rgba(0,0,0,0)",
    "plot_bgcolor": "rgba(0,0,0,0)",
    "autotypenumbers": "strict",
    "margin": {"l": 0, "r": 0, "t": 30, "b": 0},
    "xaxis": {
        "gridcolor": "#e9ecef",
        "zerolinecolor": "#dee2e6"
    },
    "yaxis": {
        "gridcolor": "#e9ecef",
        "zerolinecolor": "#dee2e6"
    },
    "legend": {
        "orientation": "h",
        "yanchor": "bottom",
        "y": 1.02,
        "xanchor": "right",
        "x": 1
    }
}


def plot_training_curves(
    rounds: List[TrainingRound],
    show_loss: bool = True,
    show_accuracy: bool = True
) -> go.Figure:
    """
    Plot training loss and accuracy curves.

    Args:
        rounds: List of training rounds
        show_loss: Whether to show loss curve
        show_accuracy: Whether to show accuracy curve

    Returns:
        Plotly figure
    """
    if not rounds:
        return _create_empty_figure("No training data available")

    fig = go.Figure()

    x_data = [r.round_num for r in rounds]

    # Loss curve
    if show_loss:
        loss_values = [r.global_loss for r in rounds]
        fig.add_trace(go.Scatter(
            x=x_data,
            y=loss_values,
            mode="lines+markers",
            name="Loss",
            line=dict(color=COLORS["danger"], width=2),
            marker=dict(size=6),
            hovertemplate="Round %{x}: Loss = %{y:.4f}<extra></extra>"
        ))

    # Accuracy curve
    if show_accuracy:
        acc_values = [r.global_accuracy for r in rounds]
        fig.add_trace(go.Scatter(
            x=x_data,
            y=acc_values,
            mode="lines+markers",
            name="Accuracy",
            line=dict(color=COLORS["success"], width=2),
            marker=dict(size=6),
            yaxis="y2",
            hovertemplate="Round %{x}: Acc = %{y:.2%}<extra></extra>"
        ))

    # Dual y-axis layout
    if show_loss and show_accuracy:
        fig.update_layout(
            yaxis=dict(
                title="Loss",
                title_font=dict(size=12, color=COLORS["danger"]),
                tickfont=dict(color=COLORS["danger"]),
                side="left"
            ),
            yaxis2=dict(
                title="Accuracy",
                title_font=dict(size=12, color=COLORS["success"]),
                tickfont=dict(color=COLORS["success"]),
                overlaying="y",
                side="right",
                tickformat=".0%"
            )
        )
    elif show_accuracy:
        fig.update_layout(yaxis=dict(tickformat=".0%"))

    fig.update_xaxes(title_text="Round")
    fig.update_layout(template=LAYOUT_TEMPLATE, height=300)

    return fig


def plot_client_reputation(
    client_metrics: Dict[int, List[ClientMetric]],
    show_history: bool = False
) -> go.Figure:
    """
    Plot client reputation scores.

    Args:
        client_metrics: Dictionary mapping client_id to metrics
        show_history: Whether to show history or just current values

    Returns:
        Plotly bar chart or line chart
    """
    if not client_metrics:
        return _create_empty_figure("No client data available")

    fig = go.Figure()

    if show_history:
        # Line chart showing reputation over time
        for client_id, metrics in sorted(client_metrics.items()):
            reputations = [m.reputation_score for m in metrics]
            rounds = [i for i in range(len(metrics))]

            color = COLORS["client_anomaly"] if reputations[-1] < 0.5 else COLORS["client_default"]

            fig.add_trace(go.Scatter(
                x=rounds,
                y=reputations,
                mode="lines+markers",
                name=f"Client {client_id}",
                line=dict(color=color, width=2),
                hovertemplate=f"Client {client_id}: %{{y:.2f}}<extra></extra>"
            ))
    else:
        # Bar chart of current reputations
        client_ids = []
        reputations = []
        colors = []

        for client_id, metrics in sorted(client_metrics.items()):
            if metrics:
                client_ids.append(f"C{client_id}")
                reputations.append(metrics[-1].reputation_score)
                if metrics[-1].reputation_score < 0.5:
                    colors.append(COLORS["client_anomaly"])
                else:
                    colors.append(COLORS["success"])

        fig.add_trace(go.Bar(
            x=client_ids,
            y=reputations,
            marker_color=colors,
            hovertemplate="Client %{x}: Rep = %{y:.2f}<extra></extra>"
        ))

        fig.update_yaxes(range=[0, 1], title="Reputation Score")

    fig.update_layout(
        template=LAYOUT_TEMPLATE,
        height=300,
        hovermode="x unified" if show_history else "x"
    )

    return fig


def plot_client_metrics_table(
    client_metrics: Dict[int, ClientMetric]
) -> List[Dict]:
    """
    Prepare client metrics for table display.

    Args:
        client_metrics: Dictionary mapping client_id to metric

    Returns:
        List of dictionaries for st.dataframe
    """
    data = []

    for client_id, metric in sorted(client_metrics.items()):
        data.append({
            "Client ID": client_id,
            "Accuracy": f"{metric.accuracy:.2%}",
            "Loss": f"{metric.loss:.4f}",
            "Data Size": metric.data_size,
            "Status": metric.status.title(),
            "Anomaly Score": f"{metric.anomaly_score:.3f}",
            "Reputation": f"{metric.reputation_score:.3f}"
        })

    return data


def plot_security_timeline(
    events: List,
    max_events: int = 50
) -> go.Figure:
    """
    Plot security events on a timeline.

    Args:
        events: List of security events
        max_events: Maximum number of events to display

    Returns:
        Plotly timeline chart
    """
    if not events:
        return _create_empty_figure("No security events")

    events = events[-max_events:]

    fig = go.Figure()

    severity_colors = {
        "critical": COLORS["danger"],
        "high": "#fd7e14",
        "medium": COLORS["warning"],
        "low": COLORS["info"]
    }

    for event in events:
        color = severity_colors.get(event.severity, COLORS["info"])

        fig.add_trace(go.Scatter(
            x=[event.round_num],
            y=[event.severity],
            mode="markers",
            name=event.event_type.replace("_", " ").title(),
            marker=dict(
                size=15,
                color=color,
                symbol="diamond",
                line=dict(color="white", width=1)
            ),
            hovertemplate=(
                f"<b>{event.event_type.replace('_', ' ').title()}</b><br>"
                f"Round: {event.round_num}<br>"
                f"Severity: {event.severity}<br>"
                f"{event.message}<br>"
                "<extra></extra>"
            ),
            showlegend=False
        ))

    fig.update_layout(
        template=LAYOUT_TEMPLATE,
        height=250,
        xaxis_title="Round",
        yaxis_title="Severity"
    )

    # Custom y-axis order
    fig.update_yaxes(
        categoryarray=["critical", "high", "medium", "low"],
        categoryorder="array"
    )

    return fig


def plot_privacy_budget(
    epsilon_spent: List[float],
    epsilon_total: float
) -> go.Figure:
    """
    Plot privacy budget over time.

    Args:
        epsilon_spent: List of epsilon values spent per round
        epsilon_total: Total privacy budget

    Returns:
        Plotly figure showing privacy usage
    """
    if not epsilon_spent:
        return _create_empty_figure("No privacy data")

    rounds = list(range(1, len(epsilon_spent) + 1))
    cumulative = np.cumsum(epsilon_spent)
    remaining = epsilon_total - cumulative

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=rounds,
        y=cumulative,
        mode="lines+markers",
        name="ε Spent",
        fill="tozeroy",
        line=dict(color=COLORS["warning"], width=2),
        hovertemplate="Round %{x}: ε = %{y:.2f}<extra></extra>"
    ))

    fig.add_trace(go.Scatter(
        x=rounds,
        y=remaining,
        mode="lines",
        name="ε Remaining",
        line=dict(color=COLORS["success"], width=2, dash="dash"),
        hovertemplate="Round %{x}: ε remaining = %{y:.2f}<extra></extra>"
    ))

    # Budget line
    fig.add_hline(
        y=epsilon_total,
        line_dash="dot",
        line_color=COLORS["danger"],
        annotation_text="Budget Limit"
    )

    fig.update_layout(
        template=LAYOUT_TEMPLATE,
        height=300,
        xaxis_title="Round",
        yaxis_title="Epsilon (ε)"
    )

    return fig


def plot_tsne_clustering(
    embeddings: np.ndarray,
    labels: np.ndarray,
    anomalies: Optional[np.ndarray] = None
) -> go.Figure:
    """
    Plot client clustering using t-SNE.

    Args:
        embeddings: 2D coordinates from t-SNE
        labels: Cluster labels
        anomalies: Boolean array indicating anomalies

    Returns:
        Plotly scatter plot
    """
    if embeddings is None or len(embeddings) == 0:
        return _create_empty_figure("No clustering data")

    fig = go.Figure()

    # Color by cluster
    unique_labels = np.unique(labels)
    colors = px.colors.qualitative.Set2

    for i, label in enumerate(unique_labels):
        mask = labels == label
        color = COLORS["client_anomaly"] if (anomalies is not None and anomalies[mask].any()) else colors[i % len(colors)]

        fig.add_trace(go.Scatter(
            x=embeddings[mask, 0],
            y=embeddings[mask, 1],
            mode="markers",
            name=f"Cluster {label}",
            marker=dict(
                size=12,
                color=color,
                opacity=0.7,
                line=dict(color="white", width=1)
            ),
            hovertemplate="Client (%{x:.2f}, %{y:.2f})<extra></extra>"
        ))

    fig.update_layout(
        template=LAYOUT_TEMPLATE,
        height=400,
        xaxis_title="t-SNE 1",
        yaxis_title="t-SNE 2"
    )

    return fig


def plot_attack_success_rate(
    attack_results: List[Dict]
) -> go.Figure:
    """
    Plot attack success rate over rounds.

    Args:
        attack_results: List of attack results per round

    Returns:
        Plotly figure
    """
    if not attack_results:
        return _create_empty_figure("No attack data")

    rounds = [r["round"] for r in attack_results]
    success_rates = [r["success_rate"] for r in attack_results]

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=rounds,
        y=success_rates,
        mode="lines+markers",
        name="Attack Success Rate",
        line=dict(color=COLORS["danger"], width=2),
        fill="tozeroy",
        hovertemplate="Round %{x}: %{y:.1%}<extra></extra>"
    ))

    fig.update_layout(
        template=LAYOUT_TEMPLATE,
        height=250,
        xaxis_title="Round",
        yaxis_title="Success Rate",
        yaxis_tickformat=".0%"
    )

    return fig


def plot_convergence_progress(
    target_rounds: int,
    current_round: int,
    is_converged: bool = False
) -> go.Figure:
    """
    Plot training convergence progress.

    Args:
        target_rounds: Total rounds planned
        current_round: Current round number
        is_converged: Whether model has converged

    Returns:
        Plotly progress indicator
    """
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=current_round,
        domain={"x": [0, 1], "y": [0, 1]},
        title={"text": f"Training Progress{' (Converged!)' if is_converged else ''}"},
        delta={"reference": target_rounds},
        gauge={
            "axis": {"range": [0, target_rounds]},
            "bar": {"color": COLORS["success"] if is_converged else COLORS["primary"]},
            "steps": [
                {"range": [0, target_rounds * 0.5], "color": "#e9ecef"},
                {"range": [target_rounds * 0.5, target_rounds * 0.8], "color": "#dee2e6"}
            ],
            "threshold": {
                "line": {"color": COLORS["success"], "width": 4},
                "thickness": 0.75,
                "value": target_rounds * 0.9
            }
        }
    ))

    fig.update_layout(
        template=LAYOUT_TEMPLATE,
        height=250,
        margin=dict(l=0, r=0, t=50, b=0)
    )

    return fig


def _create_empty_figure(message: str) -> go.Figure:
    """Create an empty placeholder figure."""
    fig = go.Figure()
    fig.add_annotation(
        text=message,
        xref="paper",
        yref="paper",
        x=0.5,
        y=0.5,
        showarrow=False,
        font=dict(size=14, color="#6c757d")
    )
    fig.update_layout(
        template=LAYOUT_TEMPLATE,
        height=250,
        xaxis={"showgrid": False, "showticklabels": False},
        yaxis={"showgrid": False, "showticklabels": False}
    )
    return fig
