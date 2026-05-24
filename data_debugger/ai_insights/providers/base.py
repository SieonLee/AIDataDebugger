"""Provider interface and shared payload helpers for AI insight generation."""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from typing import Any


ISSUE_JSON_KEYS = {
    "why_this_matters",
    "ml_impact",
    "business_impact",
    "suggested_remediation",
    "example_cleaning_code",
    "ai_insight",
}


class AIProvider(ABC):
    """Common interface for local, remote, and rule-based AI providers."""

    name = "AI Provider"

    @abstractmethod
    def is_available(self) -> bool:
        """Return whether the provider can be used right now."""

    @abstractmethod
    def generate_issue_insight(self, issue: dict[str, Any], dataset_context: dict[str, Any]) -> dict[str, str]:
        """Return structured guidance for one deterministic issue."""

    @abstractmethod
    def generate_executive_summary(
        self,
        issues: list[dict[str, Any]],
        score: int,
        dataset_context: dict[str, Any],
    ) -> str:
        """Return a concise executive summary."""

    @abstractmethod
    def generate_fix_plan(
        self,
        issues: list[dict[str, Any]],
        score: int,
        dataset_context: dict[str, Any],
    ) -> str:
        """Return a prioritized remediation plan."""

    @abstractmethod
    def generate_data_contract(self, profile: dict[str, Any], issues: list[dict[str, Any]]) -> str:
        """Return a draft data contract."""

    def answer_dataset_question(
        self,
        question: str,
        issues: list[dict[str, Any]],
        score: int,
        dataset_context: dict[str, Any],
        drift_context: dict[str, Any] | None,
    ) -> str:
        return "This provider does not support dataset Q&A."

    def generate_cleaning_code(self, issue: dict[str, Any], dataset_context: dict[str, Any]) -> str:
        return f"```python\n{issue.get('example_cleaning_code', '# No code available.')}\n```"

    def generate_drift_explanation(self, drift_context: dict[str, Any]) -> str:
        return drift_context.get("rule_based_explanation") or drift_context.get("risk_label") or "No drift explanation available."


def structured_issue_payload(issue: dict[str, Any], dataset_context: dict[str, Any]) -> dict[str, Any]:
    column = issue.get("column", "dataset")
    column_context = dataset_context.get("columns", {}).get(column, {})
    return {
        "column_name": column,
        "issue_type": issue.get("issue_type"),
        "display_name": issue.get("display_name"),
        "severity": issue.get("severity"),
        "metric": issue.get("metric"),
        "missing_rate": column_context.get("missing_rate"),
        "unique_ratio": column_context.get("unique_ratio"),
        "outlier_rate": column_context.get("outlier_rate"),
        "dtype": column_context.get("dtype"),
        "sample_summary_stats": column_context.get("summary_stats"),
        "row_count": dataset_context.get("row_count"),
        "column_count": dataset_context.get("column_count"),
    }


def compact_dataset_context(dataset_context: dict[str, Any]) -> dict[str, Any]:
    return {
        "row_count": dataset_context.get("row_count"),
        "column_count": dataset_context.get("column_count"),
        "columns": {
            column: {
                "dtype": stats.get("dtype"),
                "missing_rate": stats.get("missing_rate"),
                "unique_ratio": stats.get("unique_ratio"),
                "outlier_rate": stats.get("outlier_rate"),
                "summary_stats": stats.get("summary_stats"),
            }
            for column, stats in list(dataset_context.get("columns", {}).items())[:50]
        },
    }


def compact_issues(issues: list[dict[str, Any]], limit: int = 30) -> list[dict[str, Any]]:
    return [
        {
            "issue_type": issue.get("issue_type"),
            "display_name": issue.get("display_name"),
            "column": issue.get("column"),
            "severity": issue.get("severity"),
            "metric": issue.get("metric"),
            "explanation": issue.get("explanation"),
            "recommended_fix": issue.get("recommended_fix"),
        }
        for issue in issues[:limit]
    ]


def parse_issue_json(content: str) -> dict[str, str] | None:
    content = content.strip()
    if not content:
        return None
    if "{" in content and "}" in content:
        content = content[content.find("{") : content.rfind("}") + 1]
    content = content.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    try:
        parsed = json.loads(content)
    except json.JSONDecodeError:
        return None
    if not isinstance(parsed, dict):
        return None
    return {
        key: str(parsed.get(key, "")).strip()
        for key in ISSUE_JSON_KEYS
        if parsed.get(key) is not None and str(parsed.get(key, "")).strip()
    }


def issue_insight_prompt(issue: dict[str, Any], dataset_context: dict[str, Any]) -> str:
    structured_issue = structured_issue_payload(issue, dataset_context)
    return (
        "Generate concise JSON for an ML dataset debugging issue. "
        "Use only the structured issue data provided. Do not invent dataset rows. "
        "Return exactly these keys: why_this_matters, ml_impact, business_impact, "
        "suggested_remediation, example_cleaning_code, ai_insight.\n\n"
        f"Structured issue data:\n{json.dumps(structured_issue, ensure_ascii=False)}"
    )


def executive_summary_prompt(issues: list[dict[str, Any]], score: int, dataset_context: dict[str, Any]) -> str:
    return (
        "Write a concise markdown executive summary for an ML dataset reliability report. "
        "Use only the structured issue metadata and dataset context. Do not request or infer raw rows. "
        "Cover top risks, ML impact, business impact, and remediation priorities.\n\n"
        f"Health score: {score}\n"
        f"Dataset context: {json.dumps(compact_dataset_context(dataset_context), ensure_ascii=False)}\n"
        f"Issues: {json.dumps(compact_issues(issues, limit=20), ensure_ascii=False)}"
    )


def fix_plan_prompt(issues: list[dict[str, Any]], score: int, dataset_context: dict[str, Any]) -> str:
    return (
        "Generate a prioritized remediation plan from detected deterministic issues. "
        "Return markdown with rows or bullets containing: priority, issue, reason, recommended action, expected impact. "
        "Do not add issues that are not in the input.\n\n"
        f"Health score: {score}\n"
        f"Dataset context: {json.dumps(compact_dataset_context(dataset_context), ensure_ascii=False)}\n"
        f"Issues: {json.dumps(compact_issues(issues), ensure_ascii=False)}"
    )


def cleaning_code_prompt(issue: dict[str, Any], dataset_context: dict[str, Any]) -> str:
    return (
        "Generate pandas cleaning code for the deterministic issue below. "
        "Do not execute code. Return only a short fenced python code block plus one sentence of caution. "
        "Use the provided column name exactly.\n\n"
        f"Issue: {json.dumps(structured_issue_payload(issue, dataset_context), ensure_ascii=False)}"
    )


def data_contract_prompt(dataset_context: dict[str, Any], issues: list[dict[str, Any]]) -> str:
    return (
        "Draft a practical data contract for this dataset using only metadata and detected issues. "
        "Include validation rules such as dtype expectations, missing-rate limits, uniqueness checks, "
        "accepted categorical values when inferable from metadata, and monitoring notes. "
        "Return concise markdown.\n\n"
        f"Dataset context: {json.dumps(compact_dataset_context(dataset_context), ensure_ascii=False)}\n"
        f"Issues: {json.dumps(compact_issues(issues), ensure_ascii=False)}"
    )


def qa_prompt(
    question: str,
    issues: list[dict[str, Any]],
    score: int,
    dataset_context: dict[str, Any],
    drift_context: dict[str, Any] | None,
) -> str:
    return (
        "Answer the user's dataset analysis question using only the provided metadata. "
        "Do not claim access to raw rows. If the metadata is insufficient, say what is missing. "
        "Be concise and operational.\n\n"
        f"Question: {question}\n"
        f"Health score: {score}\n"
        f"Dataset context: {json.dumps(compact_dataset_context(dataset_context), ensure_ascii=False)}\n"
        f"Detected issues: {json.dumps(compact_issues(issues), ensure_ascii=False)}\n"
        f"Drift context: {json.dumps(drift_context or {}, ensure_ascii=False)}"
    )


def drift_prompt(drift_context: dict[str, Any]) -> str:
    return (
        "Explain these deterministic dataset drift results for ML observability. "
        "Cover what changed, why it matters, ML risk, business risk, and retraining implications. "
        "Do not invent metrics that are not present.\n\n"
        f"Drift context: {json.dumps(drift_context, ensure_ascii=False)}"
    )
