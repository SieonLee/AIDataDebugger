"""Health score calculation."""

from __future__ import annotations

SEVERITY_DEDUCTIONS = {
    "critical": 15,
    "warning": 5,
    "minor": 2,
}


def calculate_health_score(issues: list[dict]) -> int:
    """Calculate a 0-100 dataset health score from detected issues."""
    score = 100
    for issue in issues:
        score -= SEVERITY_DEDUCTIONS.get(issue.get("severity"), 0)
    return max(0, min(100, score))


def count_by_severity(issues: list[dict]) -> dict[str, int]:
    """Return issue counts grouped by severity."""
    return {
        "critical": sum(1 for issue in issues if issue.get("severity") == "critical"),
        "warning": sum(1 for issue in issues if issue.get("severity") == "warning"),
        "minor": sum(1 for issue in issues if issue.get("severity") == "minor"),
    }


def score_breakdown(issues: list[dict]) -> list[dict]:
    """Return a transparent per-issue score deduction table."""
    running_score = 100
    rows: list[dict] = []
    for issue in issues:
        deduction = SEVERITY_DEDUCTIONS.get(issue.get("severity"), 0)
        running_score = max(0, running_score - deduction)
        rows.append(
            {
                "Issue": issue.get("display_name", issue.get("issue_type", "Issue")),
                "Column": issue.get("column", "dataset"),
                "Severity": issue.get("severity", "unknown"),
                "Deduction": -deduction,
                "Running score": running_score,
            }
        )
    return rows
