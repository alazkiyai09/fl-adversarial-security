"""
Client Analytics Page
Detailed per-client metrics and reputation tracking.
"""

import streamlit as st
import pandas as pd
from src.dashboard.app.components.charts import (
    plot_client_reputation,
    plot_client_metrics_table
)
from src.dashboard.app.components.metrics import (
    metric_card,
    status_badge
)
from src.dashboard.app.components.data_fetchers import (
    get_client_metrics_summary,
    get_client_reputations,
    wait_for_refresh
)


def show_page() -> None:
    """Display the Client Analytics page."""

    st.markdown('<div class="dashboard-header"><h1>👥 Client Analytics</h1><p>Detailed per-client metrics and analysis</p></div>',
                unsafe_allow_html=True)

    # Get data
    client_summary = get_client_metrics_summary()
    reputations = get_client_reputations()

    if not client_summary:
        st.info("No client data available. Start training first.")
        return

    # Summary metrics
    col1, col2, col3 = st.columns(3)

    with col1:
        metric_card("Total Clients", len(client_summary), color="info")

    with col2:
        anomaly_count = sum(1 for m in client_summary.values() if m.get("anomaly_score", 0) > 0.5)
        metric_card("Flagged Clients", anomaly_count, color="danger" if anomaly_count > 0 else "success")

    with col3:
        avg_reputation = sum(reputations.values()) / len(reputations) if reputations else 0
        metric_card("Avg Reputation", f"{avg_reputation:.3f}", color="primary")

    st.markdown("---")

    # Client metrics table
    st.markdown("### 📊 Client Metrics Table")

    # Prepare data for table
    table_data = [
        {
            "Client ID": cid,
            "Accuracy": f"{m['accuracy']:.2%}",
            "Loss": f"{m['loss']:.4f}",
            "Status": m['status'].title(),
            "Anomaly Score": f"{m['anomaly_score']:.3f}",
            "Reputation": f"{m['reputation']:.3f}"
        }
        for cid, m in sorted(client_summary.items())
    ]

    df = pd.DataFrame(table_data)

    # Color code status column
    st.dataframe(
        df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Client ID": st.column_config.NumberColumn("Client ID", width="small"),
            "Accuracy": st.column_config.TextColumn("Accuracy", width="medium"),
            "Loss": st.column_config.TextColumn("Loss", width="medium"),
            "Status": st.column_config.TextColumn("Status", width="small"),
            "Anomaly Score": st.column_config.TextColumn("Anomaly", width="medium"),
            "Reputation": st.column_config.TextColumn("Reputation", width="medium")
        }
    )

    st.markdown("---")

    # Reputation chart
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### 🏆 Client Reputations")

        # Prepare client metrics for reputation chart
        from core.data_models import ClientMetric
        client_metrics = {}
        mc = st.session_state.metrics_collector

        for cid in client_summary.keys():
            metrics = mc.get_client_metrics(client_id=cid, n_recent=10).get(cid, [])
            if metrics:
                client_metrics[cid] = metrics

        if client_metrics:
            show_history = st.checkbox("Show History", value=False)
            fig = plot_client_reputation(client_metrics, show_history=show_history)
            st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("### ⚠️ Anomaly Detection")

        # Sort by anomaly score
        sorted_clients = sorted(
            client_summary.items(),
            key=lambda x: x[1]["anomaly_score"],
            reverse=True
        )

        # Show top anomalous clients
        anomalous = [(cid, m) for cid, m in sorted_clients if m["anomaly_score"] > 0.3]

        if anomalous:
            for cid, m in anomalous[:10]:
                score = m["anomaly_score"]
                color = "🔴" if score > 0.7 else "🟡" if score > 0.5 else "🟠"

                st.markdown(f"""
                <div style="display: flex; justify-content: space-between;
                           padding: 0.5rem; border-bottom: 1px solid #e9ecef;">
                    <span><strong>Client {cid}</strong></span>
                    <span>{color} {score:.3f}</span>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("No significant anomalies detected")

    # Auto-refresh
    if st.session_state.get("auto_refresh", True) and st.session_state.get("training_running", False):
        wait_for_refresh()
