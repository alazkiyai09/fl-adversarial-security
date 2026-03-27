"""
FL Security Dashboard - Main Entry Point
Real-time monitoring and visualization of federated learning security.
"""

import streamlit as st
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.data_models import FLConfig
from src.dashboard.app.utils.config import load_config, save_config
from src.dashboard.app.utils.session import init_session_state


def main():
    """Main application entry point."""

    # Page config
    st.set_page_config(
        page_title="FL Security Dashboard",
        page_icon="🔒",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Custom CSS for professional appearance
    st.markdown("""
    <style>
        /* Main theme adjustments */
        .main {
            padding-top: 1rem;
        }

        /* Status badges */
        .status-badge {
            padding: 0.25rem 0.75rem;
            border-radius: 1rem;
            font-size: 0.875rem;
            font-weight: 600;
            display: inline-block;
        }

        .status-active {
            background-color: #d4edda;
            color: #155724;
        }

        .status-warning {
            background-color: #fff3cd;
            color: #856404;
        }

        .status-danger {
            background-color: #f8d7da;
            color: #721c24;
        }

        /* Metric cards */
        .metric-card {
            background: white;
            padding: 1rem;
            border-radius: 0.5rem;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
            margin-bottom: 1rem;
        }

        /* Header styling */
        .dashboard-header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 1.5rem;
            border-radius: 0.5rem;
            color: white;
            margin-bottom: 1.5rem;
        }

        .dashboard-header h1 {
            margin: 0;
            font-size: 2rem;
            font-weight: 700;
        }

        .dashboard-header p {
            margin: 0.5rem 0 0 0;
            opacity: 0.9;
        }
    </style>
    """, unsafe_allow_html=True)

    # Initialize session state
    init_session_state()

    # Sidebar navigation
    with st.sidebar:
        st.markdown("### 🔒 FL Security Dashboard")
        st.markdown("---")

        page = st.radio(
            "Navigation",
            [
                "⚡ Training Monitor",
                "👥 Client Analytics",
                "🛡️ Security Status",
                "🔐 Privacy Budget",
                "📊 Experiment Comparison"
            ],
            label_visibility="collapsed"
        )

        st.markdown("---")

        # Quick stats in sidebar
        if st.session_state.get("metrics_collector"):
            metrics = st.session_state.metrics_collector.get_summary_statistics()
            st.markdown("#### Quick Stats")
            st.metric("Round", metrics.get("current_round", 0))
            st.metric("Accuracy", f"{metrics.get('global_accuracy', 0):.1%}")
            st.metric("Events", metrics.get("total_events", 0))

            # Threat level indicator
            threat = metrics.get("threat_level", "low")
            threat_colors = {
                "low": "🟢",
                "medium": "🟡",
                "high": "🟠",
                "critical": "🔴"
            }
            st.markdown(f"**Threat Level:** {threat_colors.get(threat, '⚪')} {threat.title()}")

        st.markdown("---")

        # Configuration
        st.markdown("#### ⚙️ Configuration")
        demo_mode = st.checkbox("Demo Mode", value=True)
        if demo_mode != st.session_state.get("demo_mode", True):
            st.session_state.demo_mode = demo_mode

        refresh_rate = st.slider(
            "Refresh Rate (ms)",
            min_value=500,
            max_value=10000,
            value=2000,
            step=500
        )
        st.session_state.refresh_rate = refresh_rate

        st.markdown("---")
        st.markdown("""
        <div style="font-size: 0.75rem; color: #666;">
        <strong>FL Security Dashboard v1.0</strong><br>
        Real-time monitoring of federated<br>
        learning training, attacks, and defenses.
        </div>
        """, unsafe_allow_html=True)

    # Main content area
    if page == "⚡ Training Monitor":
        # Import page
        from src.dashboard.app.pages.training_monitor import show_page
        show_page()

    elif page == "👥 Client Analytics":
        from src.dashboard.app.pages.client_analytics import show_page
        show_page()

    elif page == "🛡️ Security Status":
        from src.dashboard.app.pages.security_status import show_page
        show_page()

    elif page == "🔐 Privacy Budget":
        from src.dashboard.app.pages.privacy_budget import show_page
        show_page()

    elif page == "📊 Experiment Comparison":
        from src.dashboard.app.pages.experiment_comparison import show_page
        show_page()


if __name__ == "__main__":
    main()
