"""
Security Status Page
Attack detection alerts, defense actions, and threat monitoring.
"""

import streamlit as st
from src.dashboard.app.components.charts import plot_security_timeline, plot_attack_success_rate
from src.dashboard.app.components.metrics import (
    metric_card,
    threat_level_indicator,
    defense_status_card,
    info_box
)
from src.dashboard.app.components.data_fetchers import (
    get_security_events,
    get_attack_statistics,
    wait_for_refresh
)


def show_page() -> None:
    """Display the Security Status page."""

    st.markdown('<div class="dashboard-header"><h1>🛡️ Security Status</h1><p>Attack detection and defense monitoring</p></div>',
                unsafe_allow_html=True)

    # Get data
    events = get_security_events(50)
    attack_stats = get_attack_statistics()
    summary = st.session_state.metrics_collector.get_summary_statistics()

    # Threat level indicator
    threat_level = summary.get("threat_level", "low")

    col1, col2, col3 = st.columns([1, 2, 1])

    with col1:
        st.markdown("### Threat Level")
        threat_level_indicator(threat_level, show_label=False)

    with col2:
        st.markdown("### Quick Stats")
        stat_row({
            "Total Events": attack_stats.get("total_events", 0),
            "Attacks Detected": attack_stats.get("attacks_detected", 0),
            "Defenses Activated": attack_stats.get("defenses_activated", 0)
        })

    with col3:
        st.markdown("### ")
        if st.session_state.attack_active:
            info_box(
                "Attack Active",
                f"{st.session_state.attack_config.attack_type} attack is currently running!",
                icon="⚠️",
                variant="warning"
            )

    st.markdown("---")

    # Security timeline
    st.markdown("### 📅 Security Event Timeline")
    if events:
        fig = plot_security_timeline(events)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No security events recorded yet")

    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        # Recent events list
        st.markdown("### 🚨 Recent Events")

        if events:
            for event in events[-10:]:
                severity_colors = {
                    "critical": "🔴",
                    "high": "🟠",
                    "medium": "🟡",
                    "low": "🔵"
                }

                icon = severity_colors.get(event.severity, "⚪")

                st.markdown(f"""
                <div style="background: white; padding: 0.75rem; border-radius: 0.5rem;
                           border-left: 3px solid {'#dc3545' if event.severity in ['critical', 'high'] else '#ffc107'};
                           margin-bottom: 0.5rem; box-shadow: 0 1px 2px rgba(0,0,0,0.05);">
                    <div style="display: flex; justify-content: space-between; margin-bottom: 0.25rem;">
                        <span style="font-weight: 600;">{icon} {event.event_type.replace('_', ' ').title()}</span>
                        <span style="font-size: 0.75rem; color: #6c757d;">Round {event.round_num}</span>
                    </div>
                    <div style="font-size: 0.875rem;">{event.message}</div>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("No recent events")

    with col2:
        # Defense status
        st.markdown("### 🛡️ Defense Status")

        defense_config = st.session_state.defense_config
        defense_active = defense_config.defense_type != "none"

        defense_status_card(
            defense_type=defense_config.defense_type,
            is_active=defense_active,
            detections=attack_stats.get("defenses_activated", 0),
            clients_excluded=0  # Would be tracked in real implementation
        )

        # Attack statistics
        st.markdown("### 📊 Attack Statistics")

        st.json(attack_stats)

    # Auto-refresh
    if st.session_state.get("auto_refresh", True) and st.session_state.get("training_running", False):
        wait_for_refresh()
