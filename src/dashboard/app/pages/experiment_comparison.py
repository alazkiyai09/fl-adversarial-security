"""
Experiment Comparison Page
Side-by-side comparison of different FL configurations.
"""

import streamlit as st
import pandas as pd
from src.dashboard.app.components.charts import plot_training_curves
from src.dashboard.app.components.metrics import metric_card
from src.dashboard.app.utils.session import save_experiment_to_history
from typing import List, Dict


def show_page() -> None:
    """Display the Experiment Comparison page."""

    st.markdown('<div class="dashboard-header"><h1>📊 Experiment Comparison</h1><p>Compare different FL configurations and results</p></div>',
                unsafe_allow_html=True)

    # Get experiment history
    experiments = st.session_state.get("experiments", [])

    if not experiments:
        st.info("👈 Complete some experiments to see comparisons here")
        st.markdown("""
        ### How to Use This Page

        1. Run experiments with different configurations
        2. Each completed experiment is automatically saved
        3. Compare results side-by-side
        4. Analyze which configuration performs best

        **Tip:** Use the Configuration Editor on other pages to change settings before each run.
        """)
        return

    # Experiment selection
    st.markdown("### 🔍 Select Experiments to Compare")

    experiment_names = [exp.get("name", f"Experiment {i+1}") for i, exp in enumerate(experiments)]

    selected = st.multiselect(
        "Choose experiments",
        range(len(experiments)),
        default=list(range(min(3, len(experiments)))),
        format_func=lambda i: experiment_names[i]
    )

    if not selected:
        st.warning("Select at least one experiment to compare")
        return

    selected_experiments = [experiments[i] for i in selected]

    st.markdown("---")

    # Summary comparison table
    st.markdown("### 📋 Summary Comparison")

    comparison_data = []
    for exp in selected_experiments:
        comparison_data.append({
            "Experiment": exp.get("name", "Unnamed"),
            "Rounds": exp.get("rounds_completed", 0),
            "Final Accuracy": f"{exp.get('final_accuracy', 0):.2%}",
            "Final Loss": f"{exp.get('final_loss', 0):.4f}",
            "Attacks Detected": exp.get("total_attacks_detected", 0),
            "Defenses Used": exp.get("total_defense_activations", 0),
            "Status": exp.get("status", "unknown").title()
        })

    df = pd.DataFrame(comparison_data)
    st.dataframe(df, use_container_width=True, hide_index=True)

    st.markdown("---")

    # Side-by-side charts
    st.markdown("### 📈 Training Curves Comparison")

    num_cols = min(3, len(selected_experiments))
    cols = st.columns(num_cols)

    for col, exp in zip(cols, selected_experiments):
        with col:
            st.markdown(f"**{exp.get('name', 'Experiment')}**")

            # Reconstruct TrainingRound objects
            from core.data_models import TrainingRound
            rounds = []
            for r in exp.get("training_history", []):
                # Reconstruct client metrics
                from core.data_models import ClientMetric
                client_metrics = [ClientMetric(**cm) for cm in r.get("per_client_metrics", [])]

                # Reconstruct security events
                from core.data_models import SecurityEvent
                security_events = [SecurityEvent(**se) for se in r.get("security_events", [])]

                # Create training round
                round_dict = {
                    **r,
                    "per_client_metrics": client_metrics,
                    "security_events": security_events
                }
                rounds.append(TrainingRound(**round_dict))

            if rounds:
                fig = plot_training_curves(rounds, show_loss=True, show_accuracy=True)
                st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # Detailed statistics comparison
    st.markdown("### 📊 Detailed Statistics")

    for exp in selected_experiments:
        with st.expander(f"Details: {exp.get('name', 'Experiment')}"):
            col1, col2 = st.columns(2)

            with col1:
                st.markdown("**Configuration**")
                config = exp.get("config", {})
                if config:
                    fl_config = config.get("fl", {})
                    st.json(fl_config)

            with col2:
                st.markdown("**Attack/Defense Config**")
                if config:
                    st.json({
                        "attack": config.get("attack", {}),
                        "defense": config.get("defense", {})
                    })

            st.markdown("**Training History Summary**")
            history = exp.get("training_history", [])
            if history:
                total_clients = sum(len(r.get("per_client_metrics", [])) for r in history)
                avg_clients = total_clients / len(history) if history else 0

                st.metric("Total Rounds", len(history))
                st.metric("Avg Clients/Round", f"{avg_clients:.1f}")

    # Save current experiment
    st.markdown("---")
    st.markdown("### 💾 Save Current Experiment")

    if st.session_state.metrics_collector.training_history:
        exp_name = st.text_input("Experiment Name", value=f"Experiment_{len(experiments) + 1}")

        if st.button("Save Current Experiment", use_container_width=True):
            save_experiment_to_history(exp_name)
            st.success(f"Saved as '{exp_name}'")
            st.rerun()
