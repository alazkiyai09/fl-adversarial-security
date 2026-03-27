"""
Metric display components for the dashboard.
Provides styled metric cards, status badges, and progress bars.
"""

import streamlit as st
from typing import Optional, Dict, Any


def metric_card(
    title: str,
    value: Any,
    delta: Optional[str] = None,
    help_text: Optional[str] = None,
    color: str = "default"
) -> None:
    """
    Display a styled metric card.

    Args:
        title: Metric title
        value: Metric value
        delta: Optional change indicator
        help_text: Optional help tooltip
        color: Color scheme (default, success, warning, danger)
    """
    # Color mapping
    colors = {
        "default": "#6c757d",
        "success": "#28a745",
        "warning": "#ffc107",
        "danger": "#dc3545",
        "info": "#17a2b8"
    }

    border_color = colors.get(color, colors["default"])

    st.markdown(f"""
    <div style="background: white; padding: 1rem; border-radius: 0.5rem;
                border-left: 4px solid {border_color}; box-shadow: 0 1px 3px rgba(0,0,0,0.1);">
        <div style="font-size: 0.875rem; color: #6c757d; margin-bottom: 0.25rem;">
            {title}
        </div>
        <div style="font-size: 1.75rem; font-weight: 700; color: #212529;">
            {value}
        </div>
        {f'<div style="font-size: 0.875rem; color: {colors["success"] if "+" in (delta or "") else colors["danger"]}; margin-top: 0.25rem;">{delta}</div>' if delta else ''}
    </div>
    """, unsafe_allow_html=True)


def status_badge(
    status: str,
    size: str = "md"
) -> None:
    """
    Display a status badge.

    Args:
        status: Status text (active, idle, dropped, anomaly)
        size: Badge size (sm, md, lg)
    """
    sizes = {
        "sm": "0.75rem",
        "md": "0.875rem",
        "lg": "1rem"
    }

    colors = {
        "active": ("#d4edda", "#155724"),
        "idle": ("#fff3cd", "#856404"),
        "dropped": ("#f8d7da", "#721c24"),
        "anomaly": ("#f8d7da", "#721c24"),
        "completed": ("#d4edda", "#155724"),
        "running": ("#d1ecf1", "#0c5460"),
        "failed": ("#f8d7da", "#721c24")
    }

    bg_color, text_color = colors.get(status.lower(), ("#e2e3e5", "#383d41"))

    st.markdown(f"""
    <span style="background-color: {bg_color}; color: {text_color};
                  padding: 0.25rem 0.75rem; border-radius: 1rem;
                  font-size: {sizes.get(size, sizes["md"])};
                  font-weight: 600; display: inline-block;">
        {status.title()}
    </span>
    """, unsafe_allow_html=True)


def threat_level_indicator(
    level: str,
    show_label: bool = True
) -> None:
    """
    Display threat level indicator.

    Args:
        level: Threat level (none, low, medium, high, critical)
        show_label: Whether to show text label
    """
    icons = {
        "none": "🟢",
        "low": "🟢",
        "medium": "🟡",
        "high": "🟠",
        "critical": "🔴"
    }

    descriptions = {
        "none": "No threats detected",
        "low": "Low threat level",
        "medium": "Medium threat level",
        "high": "High threat level - Active attacks",
        "critical": "Critical threat level - Multiple attacks"
    }

    icon = icons.get(level, icons["none"])
    description = descriptions.get(level, "")

    if show_label:
        st.markdown(f"""
        <div style="text-align: center; padding: 1rem;">
            <div style="font-size: 3rem;">{icon}</div>
            <div style="font-size: 1.25rem; font-weight: 600; margin-top: 0.5rem;">
                {level.upper()} THREAT LEVEL
            </div>
            <div style="font-size: 0.875rem; color: #6c757d; margin-top: 0.25rem;">
                {description}
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f'<span style="font-size: 1.5rem;">{icon}</span>', unsafe_allow_html=True)


def progress_bar(
    current: int,
    total: int,
    label: Optional[str] = None,
    color: str = "#667eea"
) -> None:
    """
    Display a custom progress bar.

    Args:
        current: Current value
        total: Total value
        label: Optional label
        color: Bar color
    """
    percentage = min(100, (current / total * 100) if total > 0 else 0)

    st.markdown(f"""
    <div style="margin: 1rem 0;">
        {f'<div style="font-size: 0.875rem; margin-bottom: 0.5rem; display: flex; justify-content: space-between;">\
            <span>{label}</span>\
            <span>{current}/{total} ({percentage:.0f}%)</span>\
           </div>' if label else ''}
        <div style="background-color: #e9ecef; border-radius: 0.5rem; height: 1rem;
                    overflow: hidden;">
            <div style="background-color: {color}; height: 100%; width: {percentage}%;
                       transition: width 0.3s ease;"></div>
        </div>
    </div>
    """, unsafe_allow_html=True)


def client_status_grid(
    client_statuses: Dict[int, str],
    columns: int = 10
) -> None:
    """
    Display grid of client status indicators.

    Args:
        client_statuses: Dictionary mapping client_id to status
        columns: Number of columns in grid
    """
    colors = {
        "active": "#28a745",
        "idle": "#ffc107",
        "dropped": "#6c757d",
        "anomaly": "#dc3545"
    }

    html = '<div style="display: grid; grid-template-columns: repeat({}, "1fr"); gap: 0.5rem;">'.format(columns)

    for client_id, status in sorted(client_statuses.items()):
        color = colors.get(status, colors["idle"])
        html += f"""
        <div style="background-color: {color}; color: white; padding: 0.5rem;
                    border-radius: 0.25rem; text-align: center; font-size: 0.75rem;
                    font-weight: 600;">
            {client_id}
        </div>
        """

    html += "</div>"
    st.markdown(html, unsafe_allow_html=True)


def info_box(
    title: str,
    content: str,
    icon: str = "ℹ️",
    variant: str = "info"
) -> None:
    """
    Display an information box.

    Args:
        title: Box title
        content: Box content
        icon: Icon emoji
        variant: Box variant (info, warning, success, error)
    """
    variants = {
        "info": ("#d1ecf1", "#0c5460"),
        "warning": ("#fff3cd", "#856404"),
        "success": ("#d4edda", "#155724"),
        "error": ("#f8d7da", "#721c24")
    }

    bg_color, text_color = variants.get(variant, variants["info"])

    st.markdown(f"""
    <div style="background-color: {bg_color}; color: {text_color};
                padding: 1rem; border-radius: 0.5rem; margin: 1rem 0;">
        <div style="display: flex; align-items: center; margin-bottom: 0.5rem;">
            <span style="font-size: 1.5rem; margin-right: 0.5rem;">{icon}</span>
            <span style="font-weight: 700; font-size: 1rem;">{title}</span>
        </div>
        <div style="font-size: 0.875rem;">{content}</div>
    </div>
    """, unsafe_allow_html=True)


def stat_row(
    stats: Dict[str, Any]
) -> None:
    """
    Display a row of statistics.

    Args:
        stats: Dictionary of label-value pairs
    """
    cols = st.columns(len(stats))

    for col, (label, value) in zip(cols, stats.items()):
        with col:
            st.markdown(f"""
            <div style="text-align: center; padding: 0.5rem;">
                <div style="font-size: 1.5rem; font-weight: 700; color: #212529;">
                    {value}
                </div>
                <div style="font-size: 0.75rem; color: #6c757d; margin-top: 0.25rem;">
                    {label}
                </div>
            </div>
            """, unsafe_allow_html=True)


def defense_status_card(
    defense_type: str,
    is_active: bool,
    detections: int = 0,
    clients_excluded: int = 0
) -> None:
    """
    Display defense mechanism status card.

    Args:
        defense_type: Type of defense (signguard, krum, etc.)
        is_active: Whether defense is active
        detections: Number of detections
        clients_excluded: Number of clients excluded
    """
    status = "Active" if is_active else "Inactive"
    status_color = COLORS["success"] if is_active else COLORS["default"]

    st.markdown(f"""
    <div style="background: white; padding: 1rem; border-radius: 0.5rem;
                border-left: 4px solid {status_color}; box-shadow: 0 1px 3px rgba(0,0,0,0.1);">
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.5rem;">
            <div style="font-size: 1rem; font-weight: 700;">{defense_type.title()}</div>
            <div style="font-size: 0.75rem; background: {status_color}; color: white;
                       padding: 0.25rem 0.5rem; border-radius: 1rem;">{status}</div>
        </div>
        <div style="display: flex; justify-content: space-around; font-size: 0.875rem;">
            <div style="text-align: center;">
                <div style="font-weight: 700;">{detections}</div>
                <div style="color: #6c757d;">Detections</div>
            </div>
            <div style="text-align: center;">
                <div style="font-weight: 700;">{clients_excluded}</div>
                <div style="color: #6c757d;">Excluded</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)


# Color constants
COLORS = {
    "primary": "#667eea",
    "secondary": "#764ba2",
    "success": "#28a745",
    "warning": "#ffc107",
    "danger": "#dc3545",
    "info": "#17a2b8",
    "default": "#6c757d"
}
