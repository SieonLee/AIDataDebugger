"""Simple before/after cleaning simulation utilities."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from data_debugger.checks import run_quality_checks
from data_debugger.issue_catalog import enrich_issues
from data_debugger.scoring import calculate_health_score


def simulate_cleaning(
    df: pd.DataFrame,
    drop_duplicates: bool = True,
    fill_numeric_median: bool = True,
    remove_constant_columns: bool = True,
) -> tuple[pd.DataFrame, list[str]]:
    """Apply conservative cleaning actions to a copy of the dataset."""
    cleaned = df.copy()
    actions: list[str] = []

    if drop_duplicates and len(cleaned) > 0:
        before = len(cleaned)
        cleaned = cleaned.drop_duplicates()
        removed = before - len(cleaned)
        if removed:
            actions.append(f"Dropped {removed} duplicate rows.")

    if fill_numeric_median and not cleaned.empty:
        numeric_columns = cleaned.select_dtypes(include=[np.number]).columns
        filled_columns: list[str] = []
        for column in numeric_columns:
            if cleaned[column].isna().any() and not cleaned[column].dropna().empty:
                cleaned[column] = cleaned[column].fillna(cleaned[column].median())
                filled_columns.append(str(column))
        if filled_columns:
            actions.append(f"Filled numeric missing values with medians: {', '.join(filled_columns)}.")

    if remove_constant_columns and cleaned.shape[1] > 0:
        constant_columns = [
            column
            for column in cleaned.columns
            if not cleaned[column].isna().all() and cleaned[column].nunique(dropna=False) == 1
        ]
        if constant_columns:
            cleaned = cleaned.drop(columns=constant_columns)
            actions.append(f"Removed constant columns: {', '.join(map(str, constant_columns))}.")

    if not actions:
        actions.append("No conservative cleaning actions changed the dataset.")

    return cleaned, actions


def compare_cleaning_results(
    original_df: pd.DataFrame,
    cleaned_df: pd.DataFrame,
    roles: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Scan original and cleaned datasets and return a score comparison."""
    original_issues = enrich_issues(run_quality_checks(original_df, roles=roles))
    cleaned_issues = enrich_issues(run_quality_checks(cleaned_df, roles=roles))
    original_score = calculate_health_score(original_issues)
    cleaned_score = calculate_health_score(cleaned_issues)

    return {
        "original_score": original_score,
        "cleaned_score": cleaned_score,
        "improvement": cleaned_score - original_score,
        "original_issue_count": len(original_issues),
        "cleaned_issue_count": len(cleaned_issues),
        "cleaned_issues": cleaned_issues,
    }
