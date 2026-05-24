"""Drift risk scoring and severity helpers."""

from __future__ import annotations

from collections import Counter


DRIFT_DEDUCTIONS = {
    "critical": 18,
    "warning": 7,
    "minor": 2,
}


def severity_for_missing_increase(
    increase: float,
    warning_threshold: float = 0.05,
    critical_threshold: float = 0.20,
) -> str:
    if increase >= critical_threshold:
        return "critical"
    if increase >= warning_threshold:
        return "warning"
    if increase > 0:
        return "minor"
    if increase == 0:
        return "none"
    return "minor"


def severity_for_psi(psi: float) -> str:
    if psi > 0.25:
        return "critical"
    if psi >= 0.10:
        return "warning"
    return "minor"


def severity_for_category_shift(max_change: float, unseen_count: int, dominant_changed: bool) -> str:
    if max_change > 0.30 or unseen_count >= 10:
        return "critical"
    if max_change >= 0.10 or unseen_count > 0 or dominant_changed:
        return "warning"
    return "minor"


def calculate_drift_risk_score(issues: list[dict]) -> int:
    score = 100
    for issue in issues:
        score -= DRIFT_DEDUCTIONS.get(issue.get("severity"), 0)
    return max(0, min(100, score))


def drift_risk_label(score: int) -> str:
    if score >= 85:
        return "Stable"
    if score >= 65:
        return "Moderate drift"
    if score >= 40:
        return "High drift"
    return "Critical drift"


def severity_summary(issues: list[dict]) -> dict[str, int]:
    counts = Counter(issue.get("severity", "minor") for issue in issues)
    return {
        "critical": counts.get("critical", 0),
        "warning": counts.get("warning", 0),
        "minor": counts.get("minor", 0),
    }
