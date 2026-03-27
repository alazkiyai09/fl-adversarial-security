"""
Privacy Budget Page
DP epsilon tracking, secure aggregation, and privacy-utility trade-off.
"""

import streamlit as st
from src.dashboard.app.components.charts import plot_privacy_budget
from src.dashboard.app.components.metrics import metric_card, info_box, progress_bar
from src.dashboard.app.components.data_fetchers import get_privacy_status, wait_for_refresh


def show_page() -> None:
    """Display the Privacy Budget page."""

    st.markdown('<div class="dashboard-header"><h1>🔐 Privacy Budget</h1><p>Differential privacy and secure aggregation tracking</p></div>',
                unsafe_allow_html=True)

    # Get data
    privacy_status = get_privacy_status()

    if not privacy_status:
        st.info("Privacy tracking not enabled. Enable DP in configuration.")
        return

    # Summary cards
    col1, col2, col3 = st.columns(3)

    with col1:
        metric_card(
            "Total Budget (ε)",
            f"{privacy_status['epsilon_total']:.2f}",
            color="primary"
        )

    with col2:
        spent_pct = privacy_status['percentage_used']
        metric_card(
            "Epsilon Spent",
            f"{privacy_status['epsilon_spent']:.2f} ({spent_pct:.1f}%)",
            color="warning" if spent_pct > 80 else "success"
        )

    with col3:
        metric_card(
            "Epsilon Remaining",
            f"{privacy_status['epsilon_remaining']:.2f}",
            color="danger" if privacy_status['epsilon_remaining'] < privacy_status['epsilon_total'] * 0.2 else "success"
        )

    st.markdown("---")

    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown("### 📈 Privacy Budget Over Time")

        # Get epsilon history
        mc = st.session_state.metrics_collector

        # Extract epsilon per round from history
        epsilon_history = []
        for round_data in mc.training_history:
            epsilon_history.append(round_data.epsilon_spent)

        if epsilon_history:
            fig = plot_privacy_budget(
                epsilon_spent=epsilon_history,
                epsilon_total=privacy_status['epsilon_total']
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No privacy data available yet")

    with col2:
        st.markdown("### 🔒 Secure Aggregation")

        if privacy_status['secure_aggregation']:
            st.markdown(f"""
            <div style="background: #d4edda; color: #155724; padding: 1rem;
                       border-radius: 0.5rem; margin-bottom: 1rem;">
                <div style="font-size: 2rem; text-align: center;">🔐</div>
                <div style="text-align: center; font-weight: 700; margin: 0.5rem 0;">
                    Secure Aggregation Active
                </div>
                <div style="text-align: center; font-size: 0.875rem;">
                    Method: {privacy_status['encryption_method']}
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div style="background: #f8d7da; color: #721c24; padding: 1rem;
                       border-radius: 0.5rem; margin-bottom: 1rem;">
                <div style="font-size: 2rem; text-align: center;">⚠️</div>
                <div style="text-align: center; font-weight: 700; margin: 0.5rem 0;">
                    Secure Aggregation Inactive
                </div>
                <div style="text-align: center; font-size: 0.875rem;">
                    Updates are sent in plain text
                </div>
            </div>
            """, unsafe_allow_html=True)

        # DP parameters
        st.markdown("### ⚙️ DP Parameters")

        fl_config = st.session_state.fl_config

        st.markdown(f"""
        <div style="background: white; padding: 1rem; border-radius: 0.5rem;
                   border: 1px solid #dee2e6;">
            <div style="display: flex; justify-content: space-between; margin-bottom: 0.5rem;">
                <span><strong>Epsilon (ε)</strong></span>
                <span>{fl_config.dp_epsilon:.2f}</span>
            </div>
            <div style="display: flex; justify-content: space-between; margin-bottom: 0.5rem;">
                <span><strong>Delta (δ)</strong></span>
                <span>{fl_config.dp_delta:.6f}</span>
            </div>
            <div style="display: flex; justify-content: space-between; margin-bottom: 0.5rem;">
                <span><strong>Noise Multiplier</strong></span>
                <span>{fl_config.noise_multiplier:.2f}</span>
            </div>
            <div style="display: flex; justify-content: space-between;">
                <span><strong>Max Grad Norm</strong></span>
                <span>{fl_config.max_grad_norm:.2f}</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # Privacy-utility trade-off explanation
    st.markdown("### 📊 Privacy-Utility Trade-off")

    info_box(
        "Understanding the Trade-off",
        """
        **Higher epsilon (ε)** = Less privacy, better model accuracy
        **Lower epsilon (ε)** = More privacy, worse model accuracy

        **Noise Multiplier**: More noise = better privacy, worse convergence
        **Max Gradient Norm**: More clipping = better privacy, less information
        """,
        icon="📚",
        variant="info"
    )

    # Auto-refresh
    if st.session_state.get("auto_refresh", True) and st.session_state.get("training_running", False):
        wait_for_refresh()
