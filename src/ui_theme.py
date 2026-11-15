"""
Shared UI Theme and Styling for FlakyGuard AI Streamlit Dashboard.
Applies clean, research-oriented dark charcoal design with indigo/blue accents.
"""

import streamlit as st


def apply_theme():
    """Injects custom CSS styling for consistent typography, cards, and colors."""
    st.markdown(
        """
        <style>
        /* Base typography & palette */
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

        html, body, [class*="css"] {
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        }

        /* Metric cards */
        .metric-card {
            background-color: #1E293B;
            border: 1px solid #334155;
            border-radius: 8px;
            padding: 18px 20px;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.2);
            transition: transform 0.15s ease;
        }
        .metric-card:hover {
            transform: translateY(-2px);
            border-color: #6366F1;
        }
        .metric-label {
            font-size: 0.82rem;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            color: #94A3B8;
            font-weight: 600;
            margin-bottom: 6px;
        }
        .metric-value {
            font-size: 1.85rem;
            font-weight: 700;
            color: #F8FAFC;
        }
        .metric-sub {
            font-size: 0.75rem;
            color: #64748B;
            margin-top: 4px;
        }

        /* Badges */
        .badge-flaky {
            background-color: rgba(239, 68, 68, 0.2);
            color: #F87171;
            padding: 4px 10px;
            border-radius: 4px;
            font-weight: 600;
            font-size: 0.85rem;
            display: inline-block;
        }
        .badge-fp {
            background-color: rgba(59, 130, 246, 0.2);
            color: #60A5FA;
            padding: 4px 10px;
            border-radius: 4px;
            font-weight: 600;
            font-size: 0.85rem;
            display: inline-block;
        }
        .badge-info {
            background-color: rgba(99, 102, 241, 0.2);
            color: #818CF8;
            padding: 4px 10px;
            border-radius: 4px;
            font-weight: 600;
            font-size: 0.85rem;
            display: inline-block;
        }

        /* Data table headers */
        thead tr th {
            background-color: #1E293B !important;
            color: #E2E8F0 !important;
        }

        /* Code snippets */
        code, pre {
            font-family: 'JetBrains Mono', monospace !important;
        }
        </style>
        """,
        unsafe_allow_html=True
    )


def render_metric_card(label: str, value: str, subtext: str = ""):
    """Renders a custom styled metric card."""
    sub_html = f'<div class="metric-sub">{subtext}</div>' if subtext else ""
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
            {sub_html}
        </div>
        """,
        unsafe_allow_html=True
    )
