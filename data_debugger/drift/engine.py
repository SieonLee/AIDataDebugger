"""Backward-compatible drift comparison wrapper."""

from __future__ import annotations

import pandas as pd

from data_debugger.drift.drift_engine import generate_drift_summary


def compare_datasets(
    baseline: pd.DataFrame,
    current: pd.DataFrame,
    missing_warning_threshold: float = 0.05,
    missing_critical_threshold: float = 0.20,
) -> dict:
    """Return the full drift summary for existing imports."""
    return generate_drift_summary(
        baseline,
        current,
        missing_warning_threshold=missing_warning_threshold,
        missing_critical_threshold=missing_critical_threshold,
    )
