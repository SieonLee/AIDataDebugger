"""ML-aware issue metadata, investigation guidance, and remediation snippets."""

from __future__ import annotations

from typing import Any


ISSUE_GUIDANCE: dict[str, dict[str, str]] = {
    "missing_values": {
        "display_name": "Missing signal risk",
        "ml_impact": "Missing values can make feature distributions unstable, introduce training-serving skew, or force models to learn collection artifacts instead of real behavior.",
        "business_impact": "Missing values can hide process gaps and make operational metrics look better or worse than reality.",
        "why": "If missingness is systematic, the model may learn who was measured rather than what actually happened. Even simple imputation can bias downstream decisions when the missing pattern carries business meaning.",
        "fix": "Profile the missing pattern, decide whether missingness is informative, then impute with a defensible value and optionally add a missingness indicator.",
        "code": 'df["{column}_was_missing"] = df["{column}"].isna()\n'
        'df["{column}"] = df["{column}"].fillna(df["{column}"].median())',
    },
    "duplicate_rows": {
        "display_name": "Duplicate training signal",
        "ml_impact": "Duplicate rows can overweight repeated examples, inflate validation metrics, and leak the same record across train/test splits.",
        "business_impact": "Duplicates can overstate counts, revenue, events, or customer activity.",
        "why": "Repeated records make the dataset appear larger than it is and can cause a model or analysis to trust repeated observations too much.",
        "fix": "Confirm the business key and timestamp logic, then deduplicate only records that represent the same real-world event.",
        "code": "df = df.drop_duplicates()",
    },
    "constant_column": {
        "display_name": "Non-informative feature",
        "ml_impact": "Constant columns provide no predictive signal and can add noise to feature stores, model cards, and monitoring dashboards.",
        "business_impact": "A constant field may indicate broken instrumentation or a stale export pipeline.",
        "why": "A field with one value cannot help a model separate outcomes or help analysts compare segments.",
        "fix": "Remove the column unless it is required for lineage, compliance, or a fixed cohort marker.",
        "code": 'df = df.drop(columns=["{column}"])',
    },
    "high_cardinality_categorical": {
        "display_name": "Sparse categorical risk",
        "ml_impact": "High-cardinality categories can create sparse feature spaces, increase overfitting risk, and reduce model generalization.",
        "business_impact": "Long-tail labels can make segment reporting noisy and difficult to operationalize.",
        "why": "When many categories appear only a few times, models can memorize rare labels instead of learning reusable patterns.",
        "fix": "Use frequency encoding, target encoding with proper validation, grouping, or exclude the field if it behaves like an identifier.",
        "code": 'top_categories = df["{column}"].value_counts().nlargest(20).index\n'
        'df["{column}"] = df["{column}"].where(df["{column}"].isin(top_categories), "OTHER")',
    },
    "numeric_outliers_iqr": {
        "display_name": "Unstable distribution feature",
        "ml_impact": "Outliers can dominate loss functions, distort scaling, and make model behavior brittle around extreme values.",
        "business_impact": "Extreme values may represent fraud, unit errors, rare premium customers, or data entry mistakes.",
        "why": "Outliers are not always bad, but unmanaged extremes can change coefficients, splits, and aggregate metrics.",
        "fix": "Inspect outlier records, validate units, then cap, transform, segment, or preserve them as explicit edge cases.",
        "code": 'q1 = df["{column}"].quantile(0.25)\n'
        'q3 = df["{column}"].quantile(0.75)\n'
        "iqr = q3 - q1\n"
        'lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr\n'
        'df["{column}"] = df["{column}"].clip(lower, upper)',
    },
    "likely_id_column": {
        "display_name": "Memorization-prone identifier",
        "ml_impact": "Identifier-like fields can become memorization shortcuts and may act as train-time leakage signals.",
        "business_impact": "IDs are useful for joins and audits, but they rarely translate into an operational rule that generalizes.",
        "why": "A mostly unique key can let a model memorize historical rows instead of learning behavior that transfers to new records.",
        "fix": "Keep the field for tracing and joins, but exclude it from model features unless a deliberate encoding strategy is justified.",
        "code": 'feature_df = df.drop(columns=["{column}"])',
    },
    "possible_wrong_dtype": {
        "display_name": "Possible schema mismatch",
        "ml_impact": "Wrong dtypes can prevent numeric transformations, break validation code, or silently treat ordered quantities as categories.",
        "business_impact": "Schema drift can make dashboards and exports disagree about the same metric.",
        "why": "A numeric-looking text column usually means parsing, locale, currency, or placeholder values need attention before modeling.",
        "fix": "Clean formatting characters and convert with explicit error handling.",
        "code": 'df["{column}"] = pd.to_numeric(df["{column}"].astype(str).str.replace(",", ""), errors="coerce")',
    },
    "imbalanced_categorical": {
        "display_name": "Minority segment blind spot",
        "ml_impact": "A dominant category can hide poor performance on minority segments and make evaluation look more stable than it is.",
        "business_impact": "Small but important user, customer, or risk groups may be underserved or invisible in aggregate reporting.",
        "why": "When one category dominates, aggregate metrics often describe the majority group while minority behavior remains untested.",
        "fix": "Use stratified evaluation, monitor minority categories separately, and consider grouping rare categories.",
        "code": 'category_rates = df["{column}"].value_counts(normalize=True)\nprint(category_rates.head(20))',
    },
    "too_many_unique_values": {
        "display_name": "Possible train-time leakage signal",
        "ml_impact": "Near-unique non-continuous fields can leak entity, timestamp, or post-outcome information into training.",
        "business_impact": "Operational decisions based on leakage-prone fields may fail when deployed to new data.",
        "why": "A column with almost one value per row can behave like a row key, free text, or timestamp rather than a reusable signal. However, near-unique numeric distributions may be normal for continuous measurements such as prices, volumes, sensor readings, or returns.",
        "fix": "Validate whether the field is an identifier, timestamp, free-text attribute, or a legitimate continuous measurement before using it as a feature.",
        "code": 'feature_df = df.drop(columns=["{column}"])',
    },
    "suspicious_timestamp_granularity": {
        "display_name": "Suspicious timestamp granularity",
        "ml_impact": "Highly granular timestamps can encode row order, recency, or operational timing that will not generalize cleanly in deployment.",
        "business_impact": "Timestamp leakage can make backtests look strong while live decisions degrade when timing patterns shift.",
        "why": "Raw timestamps often reflect when an event was logged rather than a stable behavioral signal. They can become non-stationary shortcuts.",
        "fix": "Convert timestamps into prediction-time-safe features such as hour, day of week, tenure, or elapsed time known before the outcome.",
        "code": 'parsed = pd.to_datetime(df["{column}"], errors="coerce")\n'
        'df["{column}_dayofweek"] = parsed.dt.dayofweek\n'
        'df["{column}_hour"] = parsed.dt.hour\n'
        'df = df.drop(columns=["{column}"])',
    },
    "potential_target_leakage": {
        "display_name": "Potential target leakage",
        "ml_impact": "Target-like or post-outcome fields can leak the answer into training and produce offline metrics that collapse in production.",
        "business_impact": "Leakage-driven models create false confidence and can fail once deployed into real decision workflows.",
        "why": "If a field is created after the prediction moment, the model can learn information that will not exist at serving time.",
        "fix": "Confirm the prediction timestamp and remove labels, outcomes, and post-event fields from the feature set.",
        "code": 'feature_df = df.drop(columns=["{column}"])',
    },
    "target_quality_risk": {
        "display_name": "Target quality risk",
        "ml_impact": "Missing target labels can invalidate supervised training rows and make evaluation metrics unreliable.",
        "business_impact": "Unlabeled outcomes can hide process gaps or delay visibility into important decisions.",
        "why": "When the target is explicitly configured, label completeness becomes a first-order training readiness check.",
        "fix": "Separate unlabeled inference rows from supervised training rows, or repair the labeling pipeline before training.",
        "code": 'training_df = df[df["{column}"].notna()].copy()',
    },
    "entity_id_integrity_risk": {
        "display_name": "Entity ID integrity risk",
        "ml_impact": "Unexpected duplicate entity keys can corrupt joins, train/test splits, and entity-level evaluation.",
        "business_impact": "Duplicate business keys can double-count customers, assets, accounts, or transactions.",
        "why": "When entity and timestamp roles are configured, their combined key becomes a reliability contract rather than a generic cardinality warning.",
        "fix": "Confirm the intended grain of the dataset, then deduplicate, aggregate to entity level, or choose the correct key.",
        "code": 'duplicate_ids = df[df["{column}"].duplicated(keep=False)]',
    },
    "target_imbalance_risk": {
        "display_name": "Target imbalance risk",
        "ml_impact": "A highly imbalanced target can make accuracy misleading and hide poor minority-class recall.",
        "business_impact": "Rare but important outcomes can be under-detected in production decision workflows.",
        "why": "When one target class dominates, models can appear strong by predicting the majority class while missing the cases that matter most.",
        "fix": "Use stratified splits, class-aware metrics, calibration checks, and consider resampling or threshold tuning.",
        "code": 'target_rates = df["{column}"].value_counts(normalize=True)\nprint(target_rates)',
    },
    "empty_dataset": {
        "display_name": "Empty dataset",
        "ml_impact": "No model or analysis can be validated without usable rows and columns.",
        "business_impact": "The export or ingestion pipeline may be broken.",
        "why": "The debugger needs at least one data row and one column to inspect quality patterns.",
        "fix": "Check the export query, delimiter, encoding, and file contents before retrying.",
        "code": "# Re-export the dataset with a header row and data rows, then upload again.",
    },
}


def enrich_issue(issue: dict[str, Any]) -> dict[str, Any]:
    """Attach display names and investigation guidance to a detected issue."""
    guidance = ISSUE_GUIDANCE.get(issue["issue_type"], {})
    column = str(issue.get("column", "column"))
    enriched = dict(issue)
    enriched["display_name"] = guidance.get("display_name", issue["issue_type"].replace("_", " ").title())
    enriched["ml_impact"] = guidance.get("ml_impact", issue.get("explanation", ""))
    enriched["business_impact"] = guidance.get("business_impact", issue.get("explanation", ""))
    enriched["why_this_matters"] = guidance.get("why", issue.get("explanation", ""))
    enriched["suggested_fix"] = guidance.get("fix", issue.get("recommended_fix", ""))
    enriched["example_cleaning_code"] = guidance.get("code", "# Add a cleaning step for this issue.").format(column=column)
    enriched["ai_insight"] = _ai_insight(enriched)
    return enriched


def enrich_issues(issues: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [enrich_issue(issue) for issue in issues]


def top_risks(issues: list[dict[str, Any]], limit: int = 5) -> list[dict[str, Any]]:
    severity_order = {"critical": 0, "warning": 1, "minor": 2}
    return sorted(issues, key=lambda issue: severity_order.get(issue.get("severity", "minor"), 3))[:limit]


def scoring_rules() -> list[dict[str, str]]:
    return [
        {"Rule": "Missing rate > 40%", "Severity": "Critical"},
        {"Rule": "Constant column", "Severity": "Critical"},
        {"Rule": "Duplicate rows > 20%", "Severity": "Critical"},
        {"Rule": "Extreme outlier rate > 10%", "Severity": "Critical"},
        {"Rule": "Missing rate 5-40%", "Severity": "Warning"},
        {"Rule": "High-cardinality categorical feature", "Severity": "Warning"},
        {"Rule": "Likely identifier or leakage-prone key", "Severity": "Warning"},
        {"Rule": "Top category share >= 95%", "Severity": "Warning"},
        {"Rule": "Unique ratio > 80% with many distinct values", "Severity": "Warning"},
        {"Rule": "Low missing rate or small outlier rate", "Severity": "Minor"},
        {"Rule": "Numeric-looking values stored as text", "Severity": "Minor"},
        {"Rule": "Target-like or post-outcome column name", "Severity": "Warning"},
        {"Rule": "Highly granular timestamp feature", "Severity": "Warning"},
        {"Rule": "Configured target has missing labels", "Severity": "Warning/Critical"},
        {"Rule": "Configured target top class >= 90%", "Severity": "Warning"},
        {"Rule": "Configured entity ID has duplicate keys", "Severity": "Warning/Critical"},
    ]


def _ai_insight(issue: dict[str, Any]) -> str:
    issue_type = issue.get("issue_type", "")
    column = issue.get("column", "this column")
    metric = issue.get("metric", "the observed metric")
    insights = {
        "missing_values": f"The missingness pattern in `{column}` should be checked against the prediction moment. If missingness correlates with operational workflow, it may become a hidden segmentation feature.",
        "duplicate_rows": "Duplicate rows are a reliability smell because they can inflate both model confidence and business KPIs without adding new evidence.",
        "constant_column": f"`{column}` is currently a zero-variance feature. In production, this is often caused by stale instrumentation or an export default.",
        "high_cardinality_categorical": f"The combination of many categories and {metric} suggests `{column}` may fragment training signal into memorized pockets.",
        "numeric_outliers_iqr": f"The outlier pattern in `{column}` should be treated as a distribution stability question, not just a cleaning task.",
        "likely_id_column": f"`{column}` is likely useful for lineage, but risky as a feature because it can behave like a memorization shortcut.",
        "possible_wrong_dtype": f"`{column}` looks like a schema contract issue. Fixing it early reduces train-serving inconsistency risk.",
        "imbalanced_categorical": f"`{column}` may hide minority-segment behavior behind strong aggregate performance.",
        "too_many_unique_values": f"`{column}` has a near-unique pattern. This is risky for IDs, timestamps, and free text, but may be normal for continuous measurements.",
        "suspicious_timestamp_granularity": f"`{column}` may encode timing artifacts that shift over time, making it a non-stationary feature candidate.",
        "potential_target_leakage": f"`{column}` should be audited against the prediction timestamp before any model uses it as an input.",
        "target_quality_risk": f"`{column}` is configured as the target, so missing labels should be handled before supervised training.",
        "entity_id_integrity_risk": f"`{column}` is configured as the entity ID, so duplicate keys should be reconciled against the dataset grain.",
        "target_imbalance_risk": f"`{column}` is configured as the target, so class balance should be reviewed before training and evaluation.",
    }
    return insights.get(issue_type, "This issue should be reviewed for ML reliability, operational availability, and downstream business impact.")
