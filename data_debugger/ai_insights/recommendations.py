"""Prioritized remediation recommendation engine."""

from __future__ import annotations

from typing import Any

from data_debugger.remediation import estimate_issue_impact


RISK_ORDER = {
    "potential_target_leakage": 0,
    "duplicate_rows": 1,
    "likely_id_column": 2,
    "too_many_unique_values": 3,
    "high_cardinality_categorical": 4,
    "numeric_outliers_iqr": 5,
    "missing_values": 6,
    "suspicious_timestamp_granularity": 7,
    "possible_wrong_dtype": 8,
}
SEVERITY_ORDER = {"critical": 0, "warning": 1, "minor": 2}


def generate_recommendations(issues: list[dict[str, Any]], limit: int = 6) -> list[dict[str, Any]]:
    """Prioritize issues by severity, estimated score impact, and downstream ML risk."""
    recommendations = []
    for issue in issues:
        impact = estimate_issue_impact(issue)
        recommendations.append(
            {
                "priority": 0,
                "title": issue.get("display_name", issue.get("issue_type", "Issue")),
                "column": issue.get("column", "dataset"),
                "severity": issue.get("severity", "minor"),
                "expected_impact": impact,
                "downstream_ml_risk": issue.get("ml_impact", ""),
                "recommended_action": issue.get("suggested_fix", issue.get("recommended_fix", "")),
                "issue_type": issue.get("issue_type", ""),
            }
        )

    recommendations.sort(
        key=lambda rec: (
            SEVERITY_ORDER.get(rec["severity"], 3),
            -rec["expected_impact"],
            RISK_ORDER.get(rec["issue_type"], 99),
        )
    )
    for index, recommendation in enumerate(recommendations[:limit], start=1):
        recommendation["priority"] = index
    return recommendations[:limit]
