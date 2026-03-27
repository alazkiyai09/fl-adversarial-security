"""
Interactive control components for the dashboard.
Start/stop/pause buttons, attack injection, configuration editors.
"""

import streamlit as st
from typing import Callable, Optional, Dict, Any
from src.dashboard.app.utils.session import (
    start_training, stop_training, pause_training, resume_training,
    is_training_active, inject_attack, deactivate_attack,
    update_fl_config, update_attack_config
)


def training_control_panel() -> None:
    """Display training control buttons (Start/Stop/Pause)."""
    st.markdown("### ⚡ Training Controls")

    cols = st.columns(4)

    with cols[0]:
        if st.button("▶️ Start", use_container_width=True, disabled=is_training_active()):
            start_training()
            st.rerun()

    with cols[1]:
        if st.button("⏸️ Pause", use_container_width=True,
                    disabled=not st.session_state.get("training_running", False)):
            pause_training()
            st.rerun()

    with cols[2]:
        if st.button("⏯️ Resume", use_container_width=True,
                    disabled=not st.session_state.get("training_paused", False)):
            resume_training()
            st.rerun()

    with cols[3]:
        if st.button("⏹️ Stop", use_container_width=True,
                    disabled=not st.session_state.get("training_running", False)):
            stop_training()
            st.rerun()


def attack_injection_panel() -> None:
    """Display attack injection controls for demonstration."""
    st.markdown("### ⚔️ Attack Injection")

    with st.expander("Inject Attack (Demo Mode)", expanded=False):
        attack_type = st.selectbox(
            "Attack Type",
            ["label_flipping", "backdoor", "byzantine", "poisoning"],
            label_visibility="collapsed"
        )

        num_attackers = st.slider(
            "Number of Attackers",
            min_value=1,
            max_value=5,
            value=2
        )

        start_round = st.number_input(
            "Start Round",
            min_value=0,
            value=st.session_state.get("current_round", 0) + 1
        )

        end_round = st.number_input(
            "End Round (0 = until stopped)",
            min_value=0,
            value=0
        )

        col1, col2 = st.columns(2)

        with col1:
            if st.button("🎯 Inject Attack", use_container_width=True):
                update_attack_config({
                    "attack_type": attack_type,
                    "num_attackers": num_attackers,
                    "start_round": start_round,
                    "end_round": end_round if end_round > 0 else None
                })
                inject_attack()
                st.success(f"Attack '{attack_type}' injected!")

        with col2:
            if st.button("🛡️ Stop Attack", use_container_width=True):
                deactivate_attack()
                st.success("Attack deactivated!")


def configuration_editor() -> None:
    """Display live configuration editor."""
    st.markdown("### ⚙️ Configuration")

    tab1, tab2, tab3 = st.tabs(["FL Config", "Attack Config", "Defense Config"])

    with tab1:
        _fl_config_editor()

    with tab2:
        _attack_config_editor()

    with tab3:
        _defense_config_editor()


def _fl_config_editor() -> None:
    """FL configuration editor."""
    config = st.session_state.fl_config

    with st.form("fl_config_form"):
        col1, col2 = st.columns(2)

        with col1:
            num_rounds = st.number_input("Rounds", min_value=1, value=config.num_rounds)
            num_clients = st.number_input("Clients", min_value=1, value=config.num_clients)
            clients_per_round = st.number_input(
                "Clients/Round",
                min_value=1,
                max_value=num_clients,
                value=config.clients_per_round
            )

        with col2:
            learning_rate = st.number_input(
                "Learning Rate",
                min_value=0.0001,
                max_value=1.0,
                value=config.learning_rate,
                format="%.4f"
            )
            local_epochs = st.number_input(
                "Local Epochs",
                min_value=1,
                value=config.local_epochs
            )
            batch_size = st.number_input(
                "Batch Size",
                min_value=1,
                value=config.batch_size
            )

        st.markdown("---")

        col1, col2 = st.columns(2)

        with col1:
            data_distribution = st.selectbox(
                "Data Distribution",
                ["iid", "non_iid_dirichlet", "non_iid_pathological"],
                index=["iid", "non_iid_dirichlet", "non_iid_pathological"].index(
                    config.data_distribution
                )
            )

        with col2:
            use_dp = st.checkbox("Use Differential Privacy", value=config.use_dp)

        if use_dp:
            col1, col2 = st.columns(2)
            with col1:
                dp_epsilon = st.number_input(
                    "DP Epsilon",
                    min_value=0.1,
                    value=config.dp_epsilon
                )
            with col2:
                dp_delta = st.number_input(
                    "DP Delta",
                    min_value=0.0,
                    max_value=1.0,
                    value=config.dp_delta,
                    format="%.6f"
                )

        submitted = st.form_submit_button("Update Configuration", use_container_width=True)

        if submitted:
            updates = {
                "num_rounds": num_rounds,
                "num_clients": num_clients,
                "clients_per_round": clients_per_round,
                "learning_rate": learning_rate,
                "local_epochs": local_epochs,
                "batch_size": batch_size,
                "data_distribution": data_distribution,
                "use_dp": use_dp
            }

            if use_dp:
                updates.update({"dp_epsilon": dp_epsilon, "dp_delta": dp_delta})

            update_fl_config(updates)
            st.success("Configuration updated!")


def _attack_config_editor() -> None:
    """Attack configuration editor."""
    config = st.session_state.attack_config

    with st.form("attack_config_form"):
        attack_type = st.selectbox(
            "Attack Type",
            ["label_flipping", "backdoor", "byzantine", "poisoning"],
            index=["label_flipping", "backdoor", "byzantine", "poisoning"].index(
                config.attack_type
            )
        )

        col1, col2 = st.columns(2)

        with col1:
            num_attackers = st.number_input(
                "Number of Attackers",
                min_value=1,
                value=config.num_attackers
            )
            start_round = st.number_input(
                "Start Round",
                min_value=0,
                value=config.start_round
            )

        with col2:
            end_round = st.number_input(
                "End Round (0 = continuous)",
                min_value=0,
                value=config.end_round if config.end_round else 0
            )

        # Attack-specific parameters
        if attack_type == "label_flipping":
            label_flip_ratio = st.slider(
                "Label Flip Ratio",
                min_value=0.0,
                max_value=1.0,
                value=config.label_flip_ratio
            )

        elif attack_type == "backdoor":
            backdoor_target_class = st.number_input(
                "Target Class",
                min_value=0,
                value=config.backdoor_target_class
            )

        elif attack_type == "byzantine":
            byzantine_type = st.selectbox(
                "Byzantine Type",
                ["random", "sign_flip", "scaled"],
                index=["random", "sign_flip", "scaled"].index(config.byzantine_type)
            )

        elif attack_type == "poisoning":
            poison_magnitude = st.number_input(
                "Poison Magnitude",
                min_value=0.1,
                value=config.poison_magnitude
            )

        submitted = st.form_submit_button("Update Attack Config", use_container_width=True)

        if submitted:
            updates = {
                "attack_type": attack_type,
                "num_attackers": num_attackers,
                "start_round": start_round,
                "end_round": end_round if end_round > 0 else None
            }

            if attack_type == "label_flipping":
                updates["label_flip_ratio"] = label_flip_ratio
            elif attack_type == "backdoor":
                updates["backdoor_target_class"] = backdoor_target_class
            elif attack_type == "byzantine":
                updates["byzantine_type"] = byzantine_type
            elif attack_type == "poisoning":
                updates["poison_magnitude"] = poison_magnitude

            update_attack_config(updates)
            st.success("Attack configuration updated!")


def _defense_config_editor() -> None:
    """Defense configuration editor."""
    config = st.session_state.defense_config

    with st.form("defense_config_form"):
        defense_type = st.selectbox(
            "Defense Type",
            ["none", "signguard", "krum", "foolsgold", "trim_mean", "median"],
            index=["none", "signguard", "krum", "foolsgold", "trim_mean", "median"].index(
                config.defense_type
            )
        )

        if defense_type != "none":
            col1, col2 = st.columns(2)

            with col1:
                anomaly_threshold = st.slider(
                    "Anomaly Threshold",
                    min_value=0.0,
                    max_value=1.0,
                    value=config.anomaly_threshold
                )

            with col2:
                reputation_threshold = st.slider(
                    "Reputation Threshold",
                    min_value=0.0,
                    max_value=1.0,
                    value=config.reputation_threshold
                )

            action_on_detection = st.selectbox(
                "Action on Detection",
                ["drop", "downweight", "flag_only"],
                index=["drop", "downweight", "flag_only"].index(
                    config.action_on_detection
                )
            )

            if defense_type == "signguard":
                signguard_window_size = st.number_input(
                    "Reputation Window Size",
                    min_value=1,
                    value=config.signguard_window_size
                )
                signguard_decay_factor = st.slider(
                    "Reputation Decay",
                    min_value=0.0,
                    max_value=1.0,
                    value=config.signguard_decay_factor
                )

            elif defense_type == "krum":
                krum_num_attackers = st.number_input(
                    "Expected Attackers",
                    min_value=0,
                    value=config.krum_num_attackers
                )

            elif defense_type == "trim_mean":
                trim_ratio = st.slider(
                    "Trim Ratio",
                    min_value=0.0,
                    max_value=0.5,
                    value=config.trim_ratio
                )

        submitted = st.form_submit_button("Update Defense Config", use_container_width=True)

        if submitted:
            updates = {"defense_type": defense_type}

            if defense_type != "none":
                updates.update({
                    "anomaly_threshold": anomaly_threshold,
                    "reputation_threshold": reputation_threshold,
                    "action_on_detection": action_on_detection
                })

                if defense_type == "signguard":
                    updates.update({
                        "signguard_window_size": signguard_window_size,
                        "signguard_decay_factor": signguard_decay_factor
                    })
                elif defense_type == "krum":
                    updates["krum_num_attackers"] = krum_num_attackers
                elif defense_type == "trim_mean":
                    updates["trim_ratio"] = trim_ratio

            from src.dashboard.app.utils.session import update_defense_config
            update_defense_config(updates)
            st.success("Defense configuration updated!")


def refresh_control() -> bool:
    """
    Display refresh control and return whether to auto-refresh.

    Returns:
        True if auto-refresh is enabled
    """
    col1, col2 = st.columns([3, 1])

    with col1:
        st.session_state.auto_refresh = st.checkbox(
            "Auto-refresh",
            value=st.session_state.get("auto_refresh", True)
        )

    with col2:
        if st.button("🔄 Refresh", use_container_width=True):
            st.rerun()

    return st.session_state.auto_refresh


def export_buttons() -> None:
    """Display export buttons (PDF/PNG)."""
    col1, col2 = st.columns(2)

    with col1:
        if st.button("📸 Export as PNG", use_container_width=True):
            st.info("PNG export: Configure in deployment settings")

    with col2:
        if st.button("📄 Export as PDF", use_container_width=True):
            st.info("PDF export: Configure in deployment settings")


def demo_mode_selector() -> None:
    """Display demo mode configuration."""
    st.markdown("### 🎭 Demo Mode")

    demo_mode = st.checkbox(
        "Enable Demo Mode",
        value=st.session_state.get("demo_mode", True),
        help="Use pre-recorded data for demonstrations"
    )

    if demo_mode != st.session_state.get("demo_mode", True):
        st.session_state.demo_mode = demo_mode
        st.rerun()

    if demo_mode:
        scenario = st.selectbox(
            "Select Demo Scenario",
            ["Normal Training", "Label Flipping Attack", "Backdoor Attack",
             "Byzantine Attack", "SignGuard Defense", "FoolsGold Defense"]
        )

        if st.button("Load Scenario", use_container_width=True):
            _load_demo_scenario(scenario)
            st.success(f"Loaded scenario: {scenario}")
