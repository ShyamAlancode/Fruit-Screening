"""
ui_theme.py
===========
Centralized CSS styles, metric card helpers, and banner rendering for the CFRD dashboard.
Import and call inject_css() once at app startup.
"""

import streamlit as st

# ---------------------------------------------------------------------------
# Global CSS
# ---------------------------------------------------------------------------
_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.disclaimer-banner {
    background: linear-gradient(90deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
    color: #e0e0e0;
    padding: 10px 20px;
    border-radius: 8px;
    font-size: 0.78rem;
    font-weight: 500;
    text-align: center;
    letter-spacing: 0.03em;
    margin-bottom: 1.2rem;
    border-left: 4px solid #e94560;
}

.metric-card {
    background: linear-gradient(135deg, #1e3a5f 0%, #0d2137 100%);
    border: 1px solid #2d5a8e;
    border-radius: 12px;
    padding: 18px 22px;
    margin: 6px 0;
}

.metric-value {
    font-size: 2.2rem;
    font-weight: 700;
    color: #64b5f6;
}

.metric-label {
    font-size: 0.82rem;
    color: #90a4ae;
    text-transform: uppercase;
    letter-spacing: 0.08em;
}

.warning-box {
    background: rgba(239, 83, 80, 0.08);
    border: 1px solid rgba(239, 83, 80, 0.4);
    border-left: 4px solid #ef5350;
    border-radius: 8px;
    padding: 14px 18px;
    margin: 10px 0;
}

.info-box {
    background: rgba(33, 150, 243, 0.06);
    border: 1px solid rgba(33, 150, 243, 0.3);
    border-left: 4px solid #2196f3;
    border-radius: 8px;
    padding: 14px 18px;
    margin: 10px 0;
}

.success-box {
    background: rgba(76, 175, 80, 0.06);
    border: 1px solid rgba(76, 175, 80, 0.3);
    border-left: 4px solid #4caf50;
    border-radius: 8px;
    padding: 14px 18px;
    margin: 10px 0;
}

h1 { color: #e3f2fd; }
h2 { color: #bbdefb; border-bottom: 1px solid #1e3a5f; padding-bottom: 6px; }
h3 { color: #90caf9; }
</style>
"""


def inject_css() -> None:
    """Call once at app startup to inject global CSS."""
    st.markdown(_CSS, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Banner helpers
# ---------------------------------------------------------------------------

def show_disclaimer(banner_text: str) -> None:
    """Render the mandatory pre-hardware disclaimer banner on every page."""
    st.markdown(
        f'<div class="disclaimer-banner">🔬 {banner_text}</div>',
        unsafe_allow_html=True,
    )


def warning_box(html: str) -> None:
    st.markdown(f'<div class="warning-box">{html}</div>', unsafe_allow_html=True)


def info_box(html: str) -> None:
    st.markdown(f'<div class="info-box">{html}</div>', unsafe_allow_html=True)


def success_box(html: str) -> None:
    st.markdown(f'<div class="success-box">{html}</div>', unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Metric card
# ---------------------------------------------------------------------------

def metric_card(value: str, label: str, border_color: str = "#2d5a8e") -> None:
    st.markdown(
        f'<div class="metric-card" style="border-left:5px solid {border_color};">'
        f'<div class="metric-value">{value}</div>'
        f'<div class="metric-label">{label}</div>'
        f'</div>',
        unsafe_allow_html=True,
    )
