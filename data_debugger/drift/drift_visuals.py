"""Plotly visuals for dataset drift dashboards."""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from data_debugger.visualizations import SEVERITY_COLORS


def psi_bar_chart(numeric_table: pd.DataFrame) -> go.Figure:
    if numeric_table.empty or "psi" not in numeric_table:
        return _empty_figure("No numeric PSI results available.")
    data = numeric_table.sort_values("psi", ascending=False)
    fig = px.bar(
        data,
        x="column",
        y="psi",
        color="severity",
        color_discrete_map=SEVERITY_COLORS,
        labels={"column": "Column", "psi": "PSI"},
    )
    fig.add_hline(y=0.10, line_dash="dash", line_color="#d97706")
    fig.add_hline(y=0.25, line_dash="dash", line_color="#dc2626")
    fig.update_layout(height=340, margin=dict(l=10, r=10, t=20, b=10), legend_title_text="")
    return fig


def missing_drift_chart(missing_table: pd.DataFrame) -> go.Figure:
    if missing_table.empty:
        return _empty_figure("No missing-rate comparison available.")
    data = missing_table.copy().sort_values("change", ascending=False).head(30)
    data["baseline_missing_rate"] = data["baseline_missing_rate"] * 100
    data["new_missing_rate"] = data["new_missing_rate"] * 100
    long = data.melt(
        id_vars="column",
        value_vars=["baseline_missing_rate", "new_missing_rate"],
        var_name="dataset",
        value_name="missing_rate",
    )
    fig = px.bar(
        long,
        x="column",
        y="missing_rate",
        color="dataset",
        barmode="group",
        labels={"column": "Column", "missing_rate": "Missing rate (%)"},
        color_discrete_sequence=["#64748b", "#2563eb"],
    )
    fig.update_layout(height=340, margin=dict(l=10, r=10, t=20, b=10), legend_title_text="")
    return fig


def category_frequency_comparison(
    baseline: pd.DataFrame,
    current: pd.DataFrame,
    column: str,
    top_n: int = 15,
) -> go.Figure:
    if column not in baseline.columns or column not in current.columns:
        return _empty_figure("Column is not present in both datasets.")
    base = baseline[column].dropna().astype(str).value_counts(normalize=True).rename("baseline")
    new = current[column].dropna().astype(str).value_counts(normalize=True).rename("new")
    categories = list((base.add(new, fill_value=0)).sort_values(ascending=False).head(top_n).index)
    data = (
        pd.DataFrame(
            {
                "category": categories,
                "baseline": [float(base.get(category, 0)) * 100 for category in categories],
                "new": [float(new.get(category, 0)) * 100 for category in categories],
            }
        )
        .melt(id_vars="category", var_name="dataset", value_name="frequency")
    )
    fig = px.bar(
        data,
        x="category",
        y="frequency",
        color="dataset",
        barmode="group",
        labels={"category": "Category", "frequency": "Frequency (%)"},
        color_discrete_sequence=["#64748b", "#7c3aed"],
    )
    fig.update_layout(height=360, margin=dict(l=10, r=10, t=20, b=10), legend_title_text="")
    return fig


def numeric_distribution_overlay(
    baseline: pd.DataFrame,
    current: pd.DataFrame,
    column: str,
) -> go.Figure:
    if column not in baseline.columns or column not in current.columns:
        return _empty_figure("Column is not present in both datasets.")
    data = pd.concat(
        [
            pd.DataFrame({"value": pd.to_numeric(baseline[column], errors="coerce"), "dataset": "baseline"}),
            pd.DataFrame({"value": pd.to_numeric(current[column], errors="coerce"), "dataset": "new"}),
        ],
        ignore_index=True,
    ).dropna()
    if data.empty:
        return _empty_figure("No numeric values available for overlay.")
    fig = px.histogram(
        data,
        x="value",
        color="dataset",
        barmode="overlay",
        opacity=0.55,
        nbins=35,
        labels={"value": column},
        color_discrete_sequence=["#64748b", "#0f766e"],
    )
    fig.update_layout(height=360, margin=dict(l=10, r=10, t=20, b=10), legend_title_text="")
    return fig


def drift_issue_severity_chart(issue_table: pd.DataFrame) -> go.Figure:
    if issue_table.empty or "severity" not in issue_table:
        return _empty_figure("No drift issues detected.")
    counts = issue_table["severity"].value_counts().reset_index()
    counts.columns = ["severity", "count"]
    fig = px.bar(
        counts,
        x="severity",
        y="count",
        color="severity",
        color_discrete_map=SEVERITY_COLORS,
        labels={"severity": "Severity", "count": "Drift issues"},
    )
    fig.update_layout(height=300, margin=dict(l=10, r=10, t=20, b=10), showlegend=False)
    return fig


def _empty_figure(message: str) -> go.Figure:
    fig = go.Figure()
    fig.add_annotation(text=message, showarrow=False, x=0.5, y=0.5, xref="paper", yref="paper")
    fig.update_layout(height=300, margin=dict(l=10, r=10, t=20, b=10))
    return fig
