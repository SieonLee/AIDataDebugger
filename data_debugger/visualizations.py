"""Plotly visualizations for the Streamlit app."""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


SEVERITY_COLORS = {
    "critical": "#dc2626",
    "warning": "#d97706",
    "minor": "#2563eb",
    "none": "#94a3b8",
}


def missing_values_chart(df: pd.DataFrame) -> go.Figure:
    missing = (
        df.isna()
        .mean()
        .mul(100)
        .rename("missing_rate")
        .reset_index()
        .rename(columns={"index": "column"})
        .sort_values("missing_rate", ascending=False)
    )
    missing = missing[missing["missing_rate"] > 0]
    if missing.empty:
        missing = pd.DataFrame({"column": ["No missing values"], "missing_rate": [0.0]})

    fig = px.bar(
        missing,
        x="column",
        y="missing_rate",
        labels={"column": "Column", "missing_rate": "Missing rate (%)"},
        color_discrete_sequence=["#2563eb"],
    )
    fig.update_layout(height=320, margin=dict(l=10, r=10, t=20, b=10), showlegend=False)
    return fig


def missing_value_heatmap(df: pd.DataFrame) -> go.Figure:
    if df.empty:
        fig = go.Figure()
        fig.add_annotation(text="No rows available for missingness heatmap.", showarrow=False)
        fig.update_layout(height=280, margin=dict(l=10, r=10, t=20, b=10))
        return fig

    sample = df.head(200).isna().astype(int)
    fig = px.imshow(
        sample.T,
        color_continuous_scale=[[0, "#f8fafc"], [1, "#dc2626"]],
        labels={"x": "Row sample", "y": "Column", "color": "Missing"},
        aspect="auto",
    )
    fig.update_layout(height=420, margin=dict(l=10, r=10, t=20, b=10), coloraxis_showscale=False)
    return fig


def severity_pie_chart(issues: list[dict]) -> go.Figure:
    counts = pd.Series([issue["severity"] for issue in issues]).value_counts()
    if counts.empty:
        counts = pd.Series({"none": 1})

    fig = px.pie(
        names=counts.index,
        values=counts.values,
        hole=0.55,
        color=counts.index,
        color_discrete_map=SEVERITY_COLORS,
    )
    fig.update_layout(height=320, margin=dict(l=10, r=10, t=20, b=10), legend_title_text="")
    return fig


def severity_distribution_chart(issues: list[dict]) -> go.Figure:
    severity_order = ["critical", "warning", "minor"]
    counts = pd.Series([issue["severity"] for issue in issues]).value_counts()
    chart_data = pd.DataFrame(
        {"severity": severity_order, "count": [int(counts.get(severity, 0)) for severity in severity_order]}
    )
    fig = px.bar(
        chart_data,
        x="severity",
        y="count",
        color="severity",
        color_discrete_map=SEVERITY_COLORS,
        labels={"severity": "Severity", "count": "Issues"},
    )
    fig.update_layout(height=320, margin=dict(l=10, r=10, t=20, b=10), showlegend=False)
    return fig


def cardinality_comparison_chart(df: pd.DataFrame) -> go.Figure:
    if df.shape[1] == 0:
        fig = go.Figure()
        fig.add_annotation(text="No columns available.", showarrow=False)
        fig.update_layout(height=320, margin=dict(l=10, r=10, t=20, b=10))
        return fig

    row_count = max(1, len(df))
    data = pd.DataFrame(
        {
            "column": list(map(str, df.columns)),
            "unique_values": [int(df[column].nunique(dropna=True)) for column in df.columns],
            "unique_ratio": [float(df[column].nunique(dropna=True) / row_count) for column in df.columns],
        }
    ).sort_values("unique_ratio", ascending=False)
    fig = px.bar(
        data.head(30),
        x="column",
        y="unique_ratio",
        hover_data=["unique_values"],
        labels={"column": "Column", "unique_ratio": "Unique ratio"},
        color_discrete_sequence=["#7c3aed"],
    )
    fig.update_layout(height=360, margin=dict(l=10, r=10, t=20, b=10), showlegend=False)
    return fig


def issue_breakdown_by_category(issues: list[dict]) -> go.Figure:
    categories = {
        "missing_values": "Completeness",
        "duplicate_rows": "Integrity",
        "constant_column": "Feature utility",
        "high_cardinality_categorical": "Feature encoding",
        "numeric_outliers_iqr": "Distribution",
        "likely_id_column": "Leakage/generalization",
        "too_many_unique_values": "Leakage/generalization",
        "potential_target_leakage": "Leakage/generalization",
        "suspicious_timestamp_granularity": "Train-serving reliability",
        "possible_wrong_dtype": "Schema reliability",
        "imbalanced_categorical": "Segment coverage",
    }
    data = pd.DataFrame(
        {
            "category": [categories.get(issue.get("issue_type"), "Other") for issue in issues],
            "severity": [issue.get("severity", "minor") for issue in issues],
        }
    )
    if data.empty:
        data = pd.DataFrame({"category": ["No issues"], "severity": ["none"]})

    grouped = data.value_counts(["category", "severity"]).reset_index(name="count")
    fig = px.bar(
        grouped,
        x="category",
        y="count",
        color="severity",
        color_discrete_map=SEVERITY_COLORS,
        labels={"category": "Issue category", "count": "Issues"},
    )
    fig.update_layout(height=360, margin=dict(l=10, r=10, t=20, b=10), legend_title_text="")
    return fig


def numeric_distribution_histogram(df: pd.DataFrame, column: str) -> go.Figure:
    fig = px.histogram(
        df,
        x=column,
        nbins=30,
        marginal="box",
        color_discrete_sequence=["#0f766e"],
    )
    fig.update_layout(height=360, margin=dict(l=10, r=10, t=20, b=10), yaxis_title="Rows")
    return fig


def correlation_heatmap(df: pd.DataFrame) -> go.Figure:
    numeric_df = df.select_dtypes(include="number")
    if numeric_df.shape[1] < 2:
        fig = go.Figure()
        fig.add_annotation(
            text="Need at least two numeric columns for correlation analysis.",
            showarrow=False,
            x=0.5,
            y=0.5,
            xref="paper",
            yref="paper",
        )
        fig.update_layout(height=320, margin=dict(l=10, r=10, t=20, b=10))
        return fig

    corr = numeric_df.corr(numeric_only=True)
    fig = px.imshow(
        corr,
        text_auto=".2f",
        color_continuous_scale="RdBu_r",
        zmin=-1,
        zmax=1,
        aspect="auto",
    )
    fig.update_layout(height=420, margin=dict(l=10, r=10, t=20, b=10))
    return fig
