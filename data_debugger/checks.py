"""Rule-based data quality checks for the Data Debugger MVP."""

from __future__ import annotations

import re
from typing import Any

import numpy as np
import pandas as pd

from data_debugger.roles import columns_excluded_from_ml_checks, normalize_roles

Issue = dict[str, Any]

SAFE_CONTINUOUS_NAME_TOKENS = [
    "price",
    "volume",
    "temperature",
    "temp",
    "sensor",
    "reading",
    "return",
    "returns",
    "amount",
    "value",
    "rate",
    "ratio",
    "score",
    "measurement",
]
ID_NAME_TOKENS = ["id", "uuid", "guid", "key", "identifier"]


def _issue(
    issue_type: str,
    column: str,
    severity: str,
    metric: str,
    explanation: str,
    recommended_fix: str,
    feature_type: str | None = None,
) -> Issue:
    issue = {
        "issue_type": issue_type,
        "column": column,
        "severity": severity,
        "metric": metric,
        "explanation": explanation,
        "recommended_fix": recommended_fix,
    }
    if feature_type:
        issue["feature_type"] = feature_type
    return issue


def _is_categorical(series: pd.Series) -> bool:
    return pd.api.types.is_object_dtype(series) or isinstance(series.dtype, pd.CategoricalDtype)


def _is_numeric_looking(series: pd.Series) -> float:
    non_null = series.dropna()
    if non_null.empty:
        return 0.0

    as_text = non_null.astype(str).str.strip()
    numeric_pattern = re.compile(r"^-?[\d,]+(\.\d+)?$")
    numeric_like = as_text.map(lambda value: bool(numeric_pattern.match(value)))
    return float(numeric_like.mean())


def _looks_token_like(series: pd.Series) -> bool:
    non_null = series.dropna()
    if non_null.empty:
        return False

    as_text = non_null.astype(str).str.strip()
    long_values = as_text.str.len().median() >= 8
    compact_values = (~as_text.str.contains(r"\s", regex=True)).mean() >= 0.90
    contains_mixed_tokens = (
        (as_text.str.contains(r"[A-Za-z]", regex=True) & as_text.str.contains(r"\d", regex=True)).mean() >= 0.50
    )
    contains_separators = as_text.str.contains(r"[-_]", regex=True).mean() >= 0.50
    uuid_like = as_text.str.contains(r"^[0-9a-fA-F]{8}-[0-9a-fA-F-]{13,}$", regex=True).mean() >= 0.50
    return bool(long_values and compact_values and (contains_mixed_tokens or contains_separators or uuid_like))


def _looks_free_text_like(series: pd.Series) -> bool:
    non_null = series.dropna()
    if non_null.empty:
        return False
    as_text = non_null.astype(str).str.strip()
    return bool(as_text.str.len().median() >= 30)


def _is_integer_like(series: pd.Series) -> bool:
    non_null = series.dropna()
    if non_null.empty:
        return False
    if pd.api.types.is_integer_dtype(non_null):
        return True
    if pd.api.types.is_float_dtype(non_null):
        return bool((non_null % 1 == 0).all())
    return False


def _safe_continuous_name(column: str) -> bool:
    name = column.lower()
    return any(token in name for token in SAFE_CONTINUOUS_NAME_TOKENS)


def _name_suggests_id(column: str) -> bool:
    name = column.lower()
    return any(token in name for token in ID_NAME_TOKENS)


def _low_variance_numeric_semantics(series: pd.Series) -> bool:
    numeric = pd.to_numeric(series, errors="coerce").dropna()
    if len(numeric) < 10 or numeric.nunique() <= 2:
        return False
    mean_abs = abs(float(numeric.mean()))
    std = float(numeric.std(ddof=0))
    if mean_abs == 0:
        return False
    return (std / mean_abs) < 0.01


def _feature_type(column: str, series: pd.Series, row_count: int) -> str:
    name = column.lower()
    non_null = series.dropna()
    if any(token in name for token in ["date", "time", "timestamp", "created", "updated"]):
        return "timestamp-like"
    if _is_categorical(series):
        if _name_suggests_id(column) or _looks_token_like(series):
            return "identifier-like"
        if _looks_free_text_like(series):
            return "free-text-like"
        return "categorical"
    if pd.api.types.is_float_dtype(series):
        return "continuous numeric"
    if _is_integer_like(series):
        unique_ratio = float(non_null.nunique(dropna=True) / max(1, row_count))
        if _name_suggests_id(column) and unique_ratio >= 0.90:
            return "identifier-like"
        if unique_ratio >= 0.98 and not _safe_continuous_name(column):
            return "identifier-like"
        unique_count = int(non_null.nunique(dropna=True))
        if unique_count <= min(20, max(2, row_count // 10)):
            return "ordinal"
        return "continuous numeric"
    if pd.api.types.is_numeric_dtype(series):
        return "continuous numeric"
    return "categorical"


def run_quality_checks(df: pd.DataFrame, roles: dict[str, Any] | None = None) -> list[Issue]:
    """Run all MVP checks and return normalized issue dictionaries."""
    issues: list[Issue] = []
    row_count = len(df)
    roles = normalize_roles(roles)
    ml_excluded_columns = columns_excluded_from_ml_checks(roles)

    if row_count == 0 or df.shape[1] == 0:
        return [
            _issue(
                "empty_dataset",
                "dataset",
                "critical",
                f"shape={df.shape}",
                "The uploaded file does not contain usable rows or columns.",
                "Upload a non-empty CSV with a header row and at least one data row.",
            )
        ]

    issues.extend(_missing_value_issues(df))
    issues.extend(_duplicate_row_issues(df))
    issues.extend(_constant_column_issues(df))
    issues.extend(_categorical_cardinality_issues(df, skip_columns=ml_excluded_columns))
    issues.extend(_numeric_outlier_issues(df))
    id_like_issues = _id_like_column_issues(df, skip_columns=ml_excluded_columns)
    issues.extend(id_like_issues)
    issues.extend(_numeric_string_issues(df))
    issues.extend(_imbalanced_category_issues(df, skip_columns=set(roles["protected_columns"])))
    issues.extend(_timestamp_granularity_issues(df, configured_timestamp=roles["timestamp_column"]))
    issues.extend(_target_leakage_name_issues(df, configured_target=roles["target_column"]))
    issues.extend(_entity_id_role_issues(df, roles["entity_id_column"], roles["timestamp_column"]))
    id_like_columns = {issue["column"] for issue in id_like_issues}
    issues.extend(_relative_unique_issues(df, skip_columns=id_like_columns | ml_excluded_columns))
    issues.extend(_target_role_issues(df, roles["target_column"]))

    return issues


def _missing_value_issues(df: pd.DataFrame) -> list[Issue]:
    issues: list[Issue] = []
    for column in df.columns:
        missing_rate = float(df[column].isna().mean())
        if missing_rate == 0:
            continue

        if missing_rate > 0.40:
            severity = "critical"
        elif missing_rate >= 0.05:
            severity = "warning"
        else:
            severity = "minor"

        issues.append(
            _issue(
                "missing_values",
                column,
                severity,
                f"{missing_rate:.1%} missing",
                f"`{column}` has missing values, which can bias analysis or break model training if left untreated.",
                "Investigate why values are missing, then impute, add a missingness flag, or remove the field if it is not reliable.",
            )
        )
    return issues


def _duplicate_row_issues(df: pd.DataFrame) -> list[Issue]:
    duplicate_rate = float(df.duplicated().mean())
    if duplicate_rate == 0:
        return []

    severity = "critical" if duplicate_rate > 0.20 else "warning"
    return [
        _issue(
            "duplicate_rows",
            "dataset",
            severity,
            f"{duplicate_rate:.1%} duplicate rows",
            "Duplicate rows may inflate counts, leak repeated examples into train/test splits, or distort business metrics.",
            "Confirm whether repeated records are legitimate, then deduplicate using the correct business key.",
        )
    ]


def _constant_column_issues(df: pd.DataFrame) -> list[Issue]:
    issues: list[Issue] = []
    for column in df.columns:
        if df[column].isna().all():
            continue
        if df[column].nunique(dropna=False) == 1:
            issues.append(
                _issue(
                    "constant_column",
                    column,
                    "critical",
                    "1 unique value",
                    f"`{column}` contains only one value and carries no predictive or analytical signal.",
                    "Remove the column unless the constant value itself is needed for auditing or lineage.",
                )
            )
    return issues


def _categorical_cardinality_issues(df: pd.DataFrame, skip_columns: set[str] | None = None) -> list[Issue]:
    issues: list[Issue] = []
    skip_columns = skip_columns or set()
    row_count = len(df)
    if row_count < 2:
        return issues

    for column in df.columns:
        if column in skip_columns:
            continue
        series = df[column]
        if not _is_categorical(series):
            continue
        if _feature_type(str(column), series, row_count) != "categorical":
            continue

        unique_count = int(series.nunique(dropna=True))
        unique_rate = unique_count / row_count
        if unique_count >= 50 and unique_rate >= 0.30:
            issues.append(
                _issue(
                    "high_cardinality_categorical",
                    column,
                    "warning",
                    f"{unique_count} unique values ({unique_rate:.1%} of rows)",
                    f"`{column}` has many distinct categories, which can create sparse features and unstable model behavior.",
                    "Group rare categories, use target/frequency encoding with validation safeguards, or exclude if it is an identifier.",
                )
            )
    return issues


def _numeric_outlier_issues(df: pd.DataFrame) -> list[Issue]:
    issues: list[Issue] = []
    numeric_columns = df.select_dtypes(include=[np.number]).columns

    for column in numeric_columns:
        series = df[column].dropna()
        if len(series) < 4 or series.nunique() <= 1:
            continue

        q1 = float(series.quantile(0.25))
        q3 = float(series.quantile(0.75))
        iqr = q3 - q1
        if iqr == 0:
            continue

        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        outlier_rate = float(((series < lower) | (series > upper)).mean())
        if outlier_rate == 0:
            continue

        severity = "critical" if outlier_rate > 0.10 else "minor"
        issues.append(
            _issue(
                "numeric_outliers_iqr",
                column,
                severity,
                f"{outlier_rate:.1%} outside IQR bounds [{lower:.3g}, {upper:.3g}]",
                f"`{column}` contains values far from the central distribution, which may be errors or important edge cases.",
                "Inspect outlier records, validate units and entry rules, then cap, transform, segment, or keep with explicit justification.",
            )
        )
    return issues


def _id_like_column_issues(df: pd.DataFrame, skip_columns: set[str] | None = None) -> list[Issue]:
    issues: list[Issue] = []
    skip_columns = skip_columns or set()
    row_count = len(df)
    if row_count < 2:
        return issues

    for column in df.columns:
        if column in skip_columns:
            continue
        series = df[column].dropna()
        if series.empty:
            continue

        unique_rate = float(series.nunique(dropna=True) / row_count)
        column_name = str(column).lower()
        feature_type = _feature_type(str(column), df[column], row_count)
        name_suggests_id = _name_suggests_id(str(column))
        token_like = _looks_token_like(series)
        integer_identifier = _is_integer_like(series) and name_suggests_id
        low_variance_semantics = _low_variance_numeric_semantics(series) and not _safe_continuous_name(str(column))
        mostly_unique = unique_rate >= 0.95

        if feature_type == "continuous numeric":
            continue

        if mostly_unique and (name_suggests_id or token_like or integer_identifier or low_variance_semantics):
            issues.append(
                _issue(
                    "likely_id_column",
                    column,
                    "warning",
                    f"{unique_rate:.1%} unique values",
                    f"`{column}` looks like an identifier-like feature rather than a continuous measurement.",
                    "Keep it for joins and tracing, but exclude it from most model features unless it has a deliberate encoding strategy.",
                    feature_type=feature_type,
                )
            )
    return issues


def _numeric_string_issues(df: pd.DataFrame) -> list[Issue]:
    issues: list[Issue] = []
    for column in df.columns:
        series = df[column]
        if not _is_categorical(series):
            continue

        numeric_like_rate = _is_numeric_looking(series)
        if numeric_like_rate >= 0.80:
            issues.append(
                _issue(
                    "possible_wrong_dtype",
                    column,
                    "minor",
                    f"{numeric_like_rate:.1%} numeric-looking strings",
                    f"`{column}` is stored as text but most non-empty values look numeric.",
                    "Convert the column with `pandas.to_numeric`, after checking commas, currency symbols, and invalid placeholders.",
                )
            )
    return issues


def _imbalanced_category_issues(df: pd.DataFrame, skip_columns: set[str] | None = None) -> list[Issue]:
    issues: list[Issue] = []
    skip_columns = skip_columns or set()
    for column in df.columns:
        if column in skip_columns:
            continue
        series = df[column].dropna()
        if series.empty or not _is_categorical(df[column]) or series.nunique() < 2:
            continue

        top_share = float(series.value_counts(normalize=True).iloc[0])
        if top_share >= 0.95:
            issues.append(
                _issue(
                    "imbalanced_categorical",
                    column,
                    "warning",
                    f"top category is {top_share:.1%} of non-missing rows",
                    f"`{column}` is dominated by one category, so it may add little signal or hide minority behavior.",
                    "Check whether the imbalance is expected, combine rare labels, stratify evaluation, or monitor minority classes separately.",
                )
            )
    return issues


def _relative_unique_issues(df: pd.DataFrame, skip_columns: set[str] | None = None) -> list[Issue]:
    issues: list[Issue] = []
    skip_columns = skip_columns or set()
    row_count = len(df)
    if row_count < 10:
        return issues

    for column in df.columns:
        if column in skip_columns:
            continue
        unique_count = int(df[column].nunique(dropna=True))
        unique_rate = unique_count / row_count
        feature_type = _feature_type(str(column), df[column], row_count)
        if feature_type == "continuous numeric":
            continue
        if feature_type == "ordinal":
            continue
        if feature_type not in {"identifier-like", "free-text-like"}:
            continue
        if unique_count > 20 and unique_rate > 0.80:
            issues.append(
                _issue(
                    "too_many_unique_values",
                    column,
                    "warning" if feature_type == "identifier-like" else "minor",
                    f"{unique_count} unique values ({unique_rate:.1%} of rows)",
                    f"`{column}` has many unique values for its inferred feature type: {feature_type}. High uniqueness alone is insufficient evidence of leakage.",
                    "Validate whether this is an identifier, timestamp, free text, or leakage-prone field before using it. Near-unique numeric distributions may be normal for continuous measurements.",
                    feature_type=feature_type,
                )
            )
    return issues


def _timestamp_granularity_issues(df: pd.DataFrame, configured_timestamp: str = "") -> list[Issue]:
    issues: list[Issue] = []
    row_count = len(df)
    if row_count < 10:
        return issues

    for column in df.columns:
        if configured_timestamp and column == configured_timestamp:
            continue
        column_name = str(column).lower()
        if not any(token in column_name for token in ["date", "time", "timestamp", "created", "updated"]):
            continue

        parsed = pd.to_datetime(df[column], errors="coerce")
        parse_rate = float(parsed.notna().mean())
        if parse_rate < 0.80:
            continue

        unique_rate = float(parsed.nunique(dropna=True) / row_count)
        if unique_rate > 0.80:
            issues.append(
                _issue(
                    "suspicious_timestamp_granularity",
                    column,
                    "warning",
                    f"{unique_rate:.1%} unique timestamp values",
                    f"`{column}` has very granular timestamp values that may encode row order, recency, or post-event behavior.",
                    "Convert raw timestamps into stable features such as hour, day of week, age at event, or remove if unavailable at prediction time.",
                )
            )
    return issues


def _target_leakage_name_issues(df: pd.DataFrame, configured_target: str = "") -> list[Issue]:
    leakage_tokens = [
        "target",
        "label",
        "outcome",
        "churned",
        "converted",
        "conversion",
        "defaulted",
        "fraud",
        "post_",
        "after_",
    ]
    issues: list[Issue] = []
    for column in df.columns:
        if configured_target and column == configured_target:
            continue
        column_name = str(column).lower()
        if any(token in column_name for token in leakage_tokens):
            issues.append(
                _issue(
                    "potential_target_leakage",
                    column,
                    "warning",
                    "name suggests target or post-outcome signal",
                    f"`{column}` may represent a label, outcome, or post-event field. These fields can leak the answer into training.",
                    "Confirm prediction-time availability. Keep true labels out of feature matrices and remove post-outcome fields from training inputs.",
                )
            )
    return issues


def _target_role_issues(df: pd.DataFrame, target_column: str = "") -> list[Issue]:
    if not target_column or target_column not in df.columns:
        return []
    issues: list[Issue] = []
    target = df[target_column]
    missing_rate = float(target.isna().mean())
    if missing_rate > 0:
        severity = "critical" if missing_rate > 0.05 else "warning"
        issues.append(
            _issue(
                "target_quality_risk",
                target_column,
                severity,
                f"{missing_rate:.1%} target missing",
                f"The configured target column `{target_column}` has missing labels, which can invalidate supervised training rows.",
                "Filter unlabeled rows for supervised training or separate them into scoring/inference data.",
            )
        )
    non_null = target.dropna()
    if non_null.nunique(dropna=True) >= 2:
        top_share = float(non_null.value_counts(normalize=True).iloc[0])
        if top_share >= 0.90:
            issues.append(
                _issue(
                    "target_imbalance_risk",
                    target_column,
                    "warning",
                    f"top target class is {top_share:.1%}",
                    f"The configured target `{target_column}` is highly imbalanced, which can make naive metrics misleading.",
                    "Use stratified splits, class-aware metrics, baseline models, and consider resampling or threshold tuning.",
                )
            )
    return issues


def _entity_id_role_issues(df: pd.DataFrame, entity_id_column: str = "", timestamp_column: str = "") -> list[Issue]:
    if not entity_id_column or entity_id_column not in df.columns or len(df) < 2:
        return []
    key_columns = [entity_id_column]
    if timestamp_column and timestamp_column in df.columns:
        key_columns.append(timestamp_column)
    key_frame = df[key_columns].dropna()
    duplicate_rate = float(key_frame.duplicated().mean()) if not key_frame.empty else 0.0
    if duplicate_rate == 0:
        return []
    severity = "warning" if duplicate_rate <= 0.20 else "critical"
    key_label = " + ".join(key_columns)
    return [
        _issue(
            "entity_id_integrity_risk",
            entity_id_column,
            severity,
            f"{duplicate_rate:.1%} duplicate non-missing keys ({key_label})",
            f"The configured entity key `{key_label}` contains duplicate records.",
            "Confirm the intended dataset grain, then deduplicate, aggregate, or choose the correct entity/timestamp key.",
        )
    ]
