"""Dataset drift comparison tools."""

from data_debugger.drift.drift_engine import (
    compare_cardinality,
    compare_categorical_distributions,
    compare_missing_rates,
    compare_numeric_distributions,
    compare_schema,
    generate_drift_summary,
)


def compare_datasets(
    baseline,
    current,
    missing_warning_threshold=0.05,
    missing_critical_threshold=0.20,
):
    """Backward-compatible wrapper for the full drift summary."""
    return generate_drift_summary(
        baseline,
        current,
        missing_warning_threshold=missing_warning_threshold,
        missing_critical_threshold=missing_critical_threshold,
    )


__all__ = [
    "compare_cardinality",
    "compare_categorical_distributions",
    "compare_datasets",
    "compare_missing_rates",
    "compare_numeric_distributions",
    "compare_schema",
    "generate_drift_summary",
]
