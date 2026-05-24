"""Real dataset drift comparison engine."""

from __future__ import annotations

from typing import Any

import pandas as pd

from data_debugger.drift.drift_metrics import (
    categorical_distribution_shift,
    numeric_stats,
    population_stability_index,
)
from data_debugger.drift.drift_scoring import (
    calculate_drift_risk_score,
    drift_risk_label,
    severity_for_category_shift,
    severity_for_missing_increase,
    severity_for_psi,
    severity_summary,
)
from data_debugger.utils import normalize_dataframe_values


def _issue(
    drift_type: str,
    column: str,
    severity: str,
    metric: str,
    explanation: str,
    ml_risk: str,
    business_risk: str,
) -> dict[str, Any]:
    return {
        "drift_type": drift_type,
        "column": column,
        "severity": severity,
        "metric": metric,
        "explanation": explanation,
        "ml_risk": ml_risk,
        "business_risk": business_risk,
    }


def _normalize_column_name(column: object) -> str:
    return str(column).strip().lower()


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize columns for baseline-vs-new comparison.

    The drift engine compares metrics directly between aligned columns. Normalizing
    here prevents `Income` and ` income ` from being treated as different fields.
    """
    normalized = normalize_dataframe_values(df)
    seen: dict[str, int] = {}
    columns = []
    for column in normalized.columns:
        name = _normalize_column_name(column)
        if name in seen:
            seen[name] += 1
            name = f"{name}__{seen[name]}"
        else:
            seen[name] = 0
        columns.append(name)
    normalized.columns = columns
    return normalized


def compare_schema(baseline: pd.DataFrame, current: pd.DataFrame) -> dict[str, Any]:
    baseline = _normalize_columns(baseline)
    current = _normalize_columns(current)
    baseline_columns = set(map(str, baseline.columns))
    current_columns = set(map(str, current.columns))
    shared_columns = sorted(baseline_columns & current_columns)

    dtype_changes = []
    issues: list[dict[str, Any]] = []
    for column in shared_columns:
        baseline_dtype = str(baseline[column].dtype)
        current_dtype = str(current[column].dtype)
        if baseline_dtype != current_dtype:
            dtype_changes.append({"column": column, "baseline_dtype": baseline_dtype, "new_dtype": current_dtype})
            issues.append(
                _issue(
                    "schema_dtype_change",
                    column,
                    "warning",
                    f"{baseline_dtype} -> {current_dtype}",
                    f"`{column}` changed dtype from {baseline_dtype} to {current_dtype}.",
                    "Dtype changes can break feature transforms or create train-serving inconsistency.",
                    "A changed schema can indicate upstream ETL or export contract changes.",
                )
            )

    added_columns = sorted(current_columns - baseline_columns)
    removed_columns = sorted(baseline_columns - current_columns)
    for column in removed_columns:
        issues.append(
            _issue(
                "schema_removed_column",
                column,
                "critical",
                "column removed",
                f"`{column}` exists in baseline but is missing from the new dataset.",
                "A removed feature can break model scoring or silently drop a trained input.",
                "A removed field can invalidate dashboards, rules, or downstream reporting.",
            )
        )
    for column in added_columns:
        issues.append(
            _issue(
                "schema_added_column",
                column,
                "minor",
                "column added",
                f"`{column}` is new in the current dataset.",
                "New features may be useful, but should be validated before entering training.",
                "New fields can reflect product, tracking, or export changes.",
            )
        )

    return {
        "baseline_shape": baseline.shape,
        "new_shape": current.shape,
        "shared_columns": shared_columns,
        "added_columns": added_columns,
        "removed_columns": removed_columns,
        "dtype_changes": dtype_changes,
        "issues": issues,
    }


def compare_missing_rates(
    baseline: pd.DataFrame,
    current: pd.DataFrame,
    warning_threshold: float = 0.05,
    critical_threshold: float = 0.20,
) -> tuple[list[dict], pd.DataFrame]:
    baseline = _normalize_columns(baseline)
    current = _normalize_columns(current)
    rows = []
    issues = []
    for column in sorted(set(baseline.columns) & set(current.columns)):
        baseline_missing_count = int(baseline[column].isna().sum())
        new_missing_count = int(current[column].isna().sum())
        baseline_missing = baseline_missing_count / max(1, len(baseline))
        new_missing = new_missing_count / max(1, len(current))
        change = new_missing - baseline_missing
        count_delta = new_missing_count - baseline_missing_count
        severity = severity_for_missing_increase(change, warning_threshold, critical_threshold)
        rows.append(
            {
                "column": column,
                "baseline_missing_count": baseline_missing_count,
                "new_missing_count": new_missing_count,
                "missing_count_delta": count_delta,
                "baseline_missing_rate": baseline_missing,
                "new_missing_rate": new_missing,
                "change": change,
                "severity": severity,
            }
        )
        if change >= warning_threshold:
            issues.append(
                _issue(
                    "missing_value_drift",
                    column,
                    severity,
                    f"{baseline_missing:.1%} -> {new_missing:.1%} ({change:+.1%})",
                    f"`{column}` missingness increased by {change:.1%}.",
                    "Missingness drift can change learned feature behavior and introduce serving-time instability.",
                    "A sudden rise in missing values often indicates collection, integration, or process degradation.",
                )
            )
    return issues, pd.DataFrame(rows)


def compare_numeric_distributions(baseline: pd.DataFrame, current: pd.DataFrame) -> tuple[list[dict], pd.DataFrame]:
    baseline = _normalize_columns(baseline)
    current = _normalize_columns(current)
    shared = sorted(set(baseline.columns) & set(current.columns))
    numeric_columns = [
        column
        for column in shared
        if pd.api.types.is_numeric_dtype(baseline[column]) and pd.api.types.is_numeric_dtype(current[column])
    ]
    rows = []
    issues = []
    for column in numeric_columns:
        base_stats = numeric_stats(baseline[column])
        new_stats = numeric_stats(current[column])
        psi = population_stability_index(baseline[column], current[column])
        severity = severity_for_psi(psi)
        row = {
            "column": column,
            "psi": psi,
            "severity": severity,
            **{f"baseline_{key}": value for key, value in base_stats.items()},
            **{f"new_{key}": value for key, value in new_stats.items()},
        }
        rows.append(row)
        if severity in {"warning", "critical"}:
            issues.append(
                _issue(
                    "numeric_distribution_drift",
                    column,
                    severity,
                    f"PSI={psi:.3f}; mean {base_stats['mean']} -> {new_stats['mean']}",
                    f"`{column}` shows numeric distribution drift with PSI {psi:.3f}.",
                    "Distribution drift can degrade model calibration, ranking, and decision thresholds.",
                    "Feature movement may reflect market shifts, seasonality, tracking changes, or pipeline bugs.",
                )
            )
    return issues, pd.DataFrame(rows)


def compare_categorical_distributions(baseline: pd.DataFrame, current: pd.DataFrame) -> tuple[list[dict], pd.DataFrame]:
    baseline = _normalize_columns(baseline)
    current = _normalize_columns(current)
    shared = sorted(set(baseline.columns) & set(current.columns))
    categorical_columns = [
        column
        for column in shared
        if not pd.api.types.is_numeric_dtype(baseline[column]) or not pd.api.types.is_numeric_dtype(current[column])
    ]
    rows = []
    issues = []
    for column in categorical_columns:
        shift = categorical_distribution_shift(baseline[column], current[column])
        dominant_changed = shift["baseline_dominant"] != shift["new_dominant"]
        severity = severity_for_category_shift(
            shift["max_frequency_change"],
            len(shift["new_categories"]),
            dominant_changed,
        )
        rows.append(
            {
                "column": column,
                "severity": severity,
                "new_category_count": len(shift["new_categories"]),
                "removed_category_count": len(shift["removed_categories"]),
                "baseline_dominant": shift["baseline_dominant"],
                "new_dominant": shift["new_dominant"],
                "baseline_dominant_share": shift["baseline_dominant_share"],
                "new_dominant_share": shift["new_dominant_share"],
                "max_frequency_change": shift["max_frequency_change"],
                "new_categories": ", ".join(shift["new_categories"][:10]),
            }
        )
        if severity in {"warning", "critical"}:
            issues.append(
                _issue(
                    "categorical_distribution_drift",
                    column,
                    severity,
                    f"max frequency shift={shift['max_frequency_change']:.1%}; new categories={len(shift['new_categories'])}",
                    f"`{column}` categorical distribution changed between baseline and new data.",
                    "Category drift can create unseen labels, sparse encodings, and degraded generalization.",
                    "Changing category mix can indicate customer, market, product, or instrumentation shifts.",
                )
            )
    return issues, pd.DataFrame(rows)


def compare_cardinality(baseline: pd.DataFrame, current: pd.DataFrame) -> tuple[list[dict], pd.DataFrame]:
    baseline = _normalize_columns(baseline)
    current = _normalize_columns(current)
    rows = []
    issues = []
    for column in sorted(set(baseline.columns) & set(current.columns)):
        baseline_unique = int(baseline[column].nunique(dropna=True))
        new_unique = int(current[column].nunique(dropna=True))
        baseline_ratio = baseline_unique / max(1, len(baseline))
        new_ratio = new_unique / max(1, len(current))
        change = new_ratio - baseline_ratio
        rows.append(
            {
                "column": column,
                "baseline_unique": baseline_unique,
                "new_unique": new_unique,
                "baseline_unique_ratio": baseline_ratio,
                "new_unique_ratio": new_ratio,
                "change": change,
            }
        )
        if change >= 0.20:
            severity = "critical" if change > 0.40 else "warning"
            issues.append(
                _issue(
                    "cardinality_drift",
                    column,
                    severity,
                    f"unique ratio {baseline_ratio:.1%} -> {new_ratio:.1%}",
                    f"`{column}` cardinality increased sharply.",
                    "Cardinality drift can create sparse encodings and memorization-prone feature behavior.",
                    "A cardinality spike may reflect ID leakage, tracking changes, or new segmentation logic.",
                )
            )
    return issues, pd.DataFrame(rows)


def generate_drift_summary(
    baseline: pd.DataFrame,
    current: pd.DataFrame,
    missing_warning_threshold: float = 0.05,
    missing_critical_threshold: float = 0.20,
) -> dict[str, Any]:
    baseline = _normalize_columns(baseline)
    current = _normalize_columns(current)
    schema = compare_schema(baseline, current)
    missing_issues, missing_table = compare_missing_rates(
        baseline,
        current,
        warning_threshold=missing_warning_threshold,
        critical_threshold=missing_critical_threshold,
    )
    numeric_issues, numeric_table = compare_numeric_distributions(baseline, current)
    categorical_issues, categorical_table = compare_categorical_distributions(baseline, current)
    cardinality_issues, cardinality_table = compare_cardinality(baseline, current)

    issues = [
        *schema["issues"],
        *missing_issues,
        *numeric_issues,
        *categorical_issues,
        *cardinality_issues,
    ]
    drift_score = calculate_drift_risk_score(issues)
    explanation = _generate_drift_explanation(issues, drift_score)

    return {
        "baseline_shape": schema["baseline_shape"],
        "new_shape": schema["new_shape"],
        "baseline_normalized": baseline,
        "new_normalized": current,
        "shared_columns": len(schema["shared_columns"]),
        "added_columns": schema["added_columns"],
        "removed_columns": schema["removed_columns"],
        "dtype_changes": schema["dtype_changes"],
        "issues": issues,
        "issue_table": pd.DataFrame(issues),
        "missing_table": missing_table,
        "missing_display_table": _format_missing_display_table(missing_table),
        "numeric_table": numeric_table,
        "categorical_table": categorical_table,
        "cardinality_table": cardinality_table,
        "drift_score": drift_score,
        "risk_label": drift_risk_label(drift_score),
        "severity_summary": severity_summary(issues),
        "explanation": explanation,
    }


def _format_missing_display_table(missing_table: pd.DataFrame) -> pd.DataFrame:
    if missing_table.empty:
        return pd.DataFrame(columns=["Column", "Baseline Missing %", "New Missing %", "Delta %", "Severity"])
    display = missing_table.copy()
    return pd.DataFrame(
        {
            "Column": display["column"],
            "Baseline Missing %": display["baseline_missing_rate"].map(lambda value: f"{value:.1%}"),
            "New Missing %": display["new_missing_rate"].map(lambda value: f"{value:.1%}"),
            "Delta %": display["change"].map(lambda value: f"{value:+.1%}"),
            "Baseline Missing Count": display["baseline_missing_count"],
            "New Missing Count": display["new_missing_count"],
            "Delta Count": display["missing_count_delta"],
            "Severity": display["severity"],
        }
    )


def _generate_drift_explanation(issues: list[dict], drift_score: int) -> str:
    if not issues:
        return (
            "The baseline and new datasets look stable across schema, missingness, cardinality, "
            "numeric distributions, and categorical distributions. Continue monitoring as data volume grows."
        )

    issue_types = {issue["drift_type"] for issue in issues}
    parts = [f"The drift risk score is {drift_score}/100."]
    if "numeric_distribution_drift" in issue_types:
        parts.append(
            "Several numeric features experienced distribution movement, which may indicate seasonality, upstream pipeline changes, or production-serving inconsistencies."
        )
    if "categorical_distribution_drift" in issue_types:
        parts.append(
            "Categorical feature mix changed, which can introduce unseen categories or shift segment-level model behavior."
        )
    if "missing_value_drift" in issue_types:
        parts.append(
            "Missing value rates changed, suggesting potential data collection degradation or changes in source-system coverage."
        )
    if {"schema_removed_column", "schema_dtype_change"} & issue_types:
        parts.append(
            "Schema drift was detected and should be reviewed before using the new dataset for scoring, retraining, or reporting."
        )
    parts.append(
        "ML risk: drift can reduce calibration, increase train-serving skew, and make historical validation less representative. "
        "Business risk: downstream dashboards, rules, and decisions may no longer reflect the same population or process. "
        "Retraining implication: investigate high-severity drift first, then decide whether retraining, feature contract fixes, or monitoring thresholds are needed."
    )
    return "\n\n".join(parts)
