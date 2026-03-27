"""
Training Monitor Page
Real-time visualization of FL training progress.
"""

import streamlit as st
import time
from src.dashboard.app.components.charts import (
    plot_training_curves,
    plot_convergence_progress
)
from src.dashboard.app.components.metrics import (
    metric_card,
    status_badge,
    progress_bar,
    client_status_grid
)
from src.dashboard.app.components.controls import training_control_panel, refresh_control
from src.dashboard.app.components.data_fetchers import (
    get_training_summary,
    get_recent_rounds,
    wait_for_refresh
)


def show_page() -> None:
    """Display the Training Monitor page."""

    st.markdown('<div class="dashboard-header"><h1>⚡ Training Monitor</h1><p>Real-time federated learning training progress</p></div>',
                unsafe_allow_html=True)

    # Training controls
    training_control_panel()

    st.markdown("---")

    # Refresh control
    refresh_control()

    # Get data
    summary = get_training_summary()
    recent_rounds = get_recent_rounds(50)

    if not recent_rounds:
        st.info("👈 Start training to see real-time metrics")
        return

    # Top metrics row
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        metric_card(
            "Current Round",
            summary.get("current_round", 0),
            color="primary"
        )

    with col2:
        metric_card(
            "Global Accuracy",
            f"{summary.get('global_accuracy', 0):.1%}",
            color="success"
        )

    with col3:
        metric_card(
            "Global Loss",
            f"{summary.get('global_loss', 0):.4f}",
            color="warning"
        )

    with col4:
        metric_card(
            "Active Clients",
            summary.get("active_clients", 0),
            color="info"
        )

    st.markdown("---")

    # Main charts
    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown("### 📈 Training Curves")
        fig = plot_training_curves(recent_rounds, show_loss=True, show_accuracy=True)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("### 🎯 Convergence Progress")
        latest_round = recent_rounds[-1] if recent_rounds else None
        if latest_round:
            fig = plot_convergence_progress(
                target_rounds=st.session_state.fl_config.num_rounds,
                current_round=latest_round.round_num,
                is_converged=latest_round.loss_delta < 0.001
            )
            st.plotly_chart(fig, use_container_width=True)

        st.markdown("### 📊 Round Statistics")
        if latest_round:
            col_a, col_b = st.columns(2)
            with col_a:
                st.metric("Loss Delta", f"{latest_round.loss_delta:+.4f}")
            with col_b:
                st.metric("Acc Delta", f"{latest_round.accuracy_delta:+.2%}")

    # Client status grid
    st.markdown("---")
    st.markdown("### 👥 Client Status")

    if latest_round and latest_round.per_client_metrics:
        client_statuses = {
            m.client_id: m.status
            for m in latest_round.per_client_metrics
        }
        client_status_grid(client_statuses, columns=10)

    # Auto-refresh
    if st.session_state.get("auto_refresh", True) and st.session_state.get("training_running", False):
        wait_for_refresh()
