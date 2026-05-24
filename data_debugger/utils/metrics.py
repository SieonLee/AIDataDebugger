"""Monitoring-style summary metrics."""

from __future__ import annotations

from typing import Any

import pandas as pd


def observability_metrics(df: pd.DataFrame, issues: list[dict[str, Any]], health_score: int) -> dict[str, Any]:
    issue_columns = {issue.get("column") for issue in issues if issue.get("column") not in {None, "dataset"}}
    issue_density = len(issues) / max(1, df.shape[1])
    problematic_columns_pct = len(issue_columns) / max(1, df.shape[1]) * 100

    if health_score >= 85:
        risk_level = "Low"
    elif health_score >= 65:
        risk_level = "Moderate"
    elif health_score >= 40:
        risk_level = "High"
    else:
        risk_level = "Critical"

    return {
        "issue_density": round(issue_density, 2),
        "problematic_columns_pct": round(problematic_columns_pct, 1),
        "estimated_ml_risk_level": risk_level,
        "cleanliness_trend": "Pending",
        "cleanliness_trend_detail": "Awaiting monitoring baseline",
    }
