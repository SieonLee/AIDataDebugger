"""Issue-level remediation previews and fix application."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from data_debugger.checks import run_quality_checks
from data_debugger.issue_catalog import enrich_issues
from data_debugger.scoring import calculate_health_score


def apply_issue_fix(df: pd.DataFrame, issue: dict[str, Any]) -> tuple[pd.DataFrame, list[str]]:
    """Apply one conservative fix for a single issue to a copied DataFrame."""
    fixed = df.copy()
    issue_type = issue.get("issue_type")
    column = issue.get("column")
    actions: list[str] = []

    if issue_type == "duplicate_rows":
        before = len(fixed)
        fixed = fixed.drop_duplicates()
        actions.append(f"Dropped {before - len(fixed)} duplicate rows.")

    elif issue_type == "missing_values" and column in fixed.columns:
        if pd.api.types.is_numeric_dtype(fixed[column]) and not fixed[column].dropna().empty:
            fixed[column] = fixed[column].fillna(fixed[column].median())
            actions.append(f"Filled missing numeric values in {column} with the median.")
        else:
            mode = fixed[column].mode(dropna=True)
            fill_value = mode.iloc[0] if not mode.empty else "UNKNOWN"
            fixed[column] = fixed[column].fillna(fill_value)
            actions.append(f"Filled missing values in {column} with the most frequent value.")

    elif issue_type == "constant_column" and column in fixed.columns:
        fixed = fixed.drop(columns=[column])
        actions.append(f"Removed constant column {column}.")

    elif issue_type == "high_cardinality_categorical" and column in fixed.columns:
        top_categories = fixed[column].value_counts().nlargest(20).index
        fixed[column] = fixed[column].where(fixed[column].isin(top_categories), "OTHER")
        actions.append(f"Grouped rare categories in {column} into OTHER.")

    elif issue_type == "numeric_outliers_iqr" and column in fixed.columns:
        series = fixed[column].dropna()
        if pd.api.types.is_numeric_dtype(fixed[column]) and len(series) >= 4:
            q1 = series.quantile(0.25)
            q3 = series.quantile(0.75)
            iqr = q3 - q1
            if iqr != 0:
                fixed[column] = fixed[column].clip(q1 - 1.5 * iqr, q3 + 1.5 * iqr)
                actions.append(f"Capped IQR outliers in {column}.")

    elif issue_type in {"likely_id_column", "too_many_unique_values", "potential_target_leakage"} and column in fixed.columns:
        fixed = fixed.drop(columns=[column])
        actions.append(f"Removed leakage-prone or non-generalizable feature {column}.")

    elif issue_type == "possible_wrong_dtype" and column in fixed.columns:
        fixed[column] = pd.to_numeric(fixed[column].astype(str).str.replace(",", "", regex=False), errors="coerce")
        actions.append(f"Converted {column} to numeric with invalid values coerced to missing.")

    elif issue_type == "imbalanced_categorical" and column in fixed.columns:
        counts = fixed[column].value_counts()
        rare_values = counts[counts < max(2, len(fixed) * 0.01)].index
        fixed[column] = fixed[column].where(~fixed[column].isin(rare_values), "OTHER")
        actions.append(f"Grouped rare categories in imbalanced feature {column}.")

    elif issue_type == "suspicious_timestamp_granularity" and column in fixed.columns:
        parsed = pd.to_datetime(fixed[column], errors="coerce")
        fixed[f"{column}_dayofweek"] = parsed.dt.dayofweek
        fixed[f"{column}_hour"] = parsed.dt.hour
        fixed = fixed.drop(columns=[column])
        actions.append(f"Converted {column} into day-of-week and hour features, then removed raw timestamp.")

    if not actions:
        actions.append("No safe automatic fix is available for this issue.")

    return fixed, actions


def preview_issue_fix(df: pd.DataFrame, issue: dict[str, Any], roles: dict[str, Any] | None = None) -> dict[str, Any]:
    """Preview score and dataset changes if one issue fix were applied."""
    before_issues = enrich_issues(run_quality_checks(df, roles=roles))
    before_score = calculate_health_score(before_issues)
    fixed, actions = apply_issue_fix(df, issue)
    after_issues = enrich_issues(run_quality_checks(fixed, roles=roles))
    after_score = calculate_health_score(after_issues)

    return {
        "before_score": before_score,
        "after_score": after_score,
        "improvement": after_score - before_score,
        "before_shape": df.shape,
        "after_shape": fixed.shape,
        "before_missing": int(df.isna().sum().sum()),
        "after_missing": int(fixed.isna().sum().sum()),
        "actions": actions,
        "fixed_df": fixed,
        "after_issue_count": len(after_issues),
    }


def estimate_issue_impact(issue: dict[str, Any]) -> int:
    """Estimate score gain for recommendation ordering."""
    severity_base = {"critical": 12, "warning": 6, "minor": 3}.get(issue.get("severity"), 1)
    risk_bonus = {
        "duplicate_rows": 4,
        "constant_column": 4,
        "potential_target_leakage": 8,
        "likely_id_column": 5,
        "too_many_unique_values": 5,
        "high_cardinality_categorical": 4,
        "numeric_outliers_iqr": 3,
        "suspicious_timestamp_granularity": 4,
    }.get(issue.get("issue_type"), 0)
    return int(np.clip(severity_base + risk_bonus, 1, 20))
