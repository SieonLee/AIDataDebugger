"""Reusable Streamlit UI components."""

from __future__ import annotations

import html

import streamlit as st

from data_debugger.visualizations import SEVERITY_COLORS


def inject_dashboard_css() -> None:
    st.markdown(
        """
        <style>
        .block-container {
            padding-top: 2rem;
            padding-bottom: 3rem;
        }
        div[data-testid="stMetric"] {
            background: #ffffff;
            border: 1px solid #e5e7eb;
            border-radius: 8px;
            padding: 14px 16px;
            box-shadow: 0 1px 2px rgba(15, 23, 42, 0.06);
        }
        .severity-badge {
            display: inline-block;
            border-radius: 999px;
            color: white;
            font-size: 0.76rem;
            font-weight: 700;
            letter-spacing: 0;
            padding: 0.18rem 0.55rem;
            text-transform: uppercase;
        }
        .issue-subtitle {
            color: #475569;
            font-size: 0.92rem;
            margin-top: -0.25rem;
            margin-bottom: 0.35rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def severity_badge(severity: str) -> str:
    color = SEVERITY_COLORS.get(severity, "#64748b")
    safe_severity = html.escape(severity)
    return f'<span class="severity-badge" style="background:{color};">{safe_severity}</span>'


def render_issue_panel(issue: dict) -> None:
    title = f"{issue['display_name']} - {issue['column']} - {issue['metric']}"
    with st.expander(title):
        st.markdown(severity_badge(issue["severity"]), unsafe_allow_html=True)
        st.markdown(f'<div class="issue-subtitle">{html.escape(issue["explanation"])}</div>', unsafe_allow_html=True)

        why_tab, impact_tab, fix_tab, code_tab, insight_tab = st.tabs(
            ["Why this matters", "ML/business impact", "Suggested remediation", "Example cleaning code", "AI insight"]
        )
        with why_tab:
            st.write(issue["why_this_matters"])
        with impact_tab:
            st.markdown(f"**ML impact:** {issue['ml_impact']}")
            st.markdown(f"**Business impact:** {issue['business_impact']}")
        with fix_tab:
            st.write(issue["suggested_fix"])
        with code_tab:
            st.code(issue["example_cleaning_code"], language="python")
        with insight_tab:
            st.info(issue["ai_insight"])
