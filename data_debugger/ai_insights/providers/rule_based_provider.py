"""Deterministic fallback provider."""

from __future__ import annotations

from typing import Any

from data_debugger.ai_insights.providers.base import AIProvider
from data_debugger.llm_report import generate_explanation


class RuleBasedProvider(AIProvider):
    name = "Rule-based only"

    def is_available(self) -> bool:
        return True

    def generate_issue_insight(self, issue: dict[str, Any], dataset_context: dict[str, Any]) -> dict[str, str]:
        return {
            "why_this_matters": issue.get("why_this_matters", issue.get("explanation", "")),
            "ml_impact": issue.get("ml_impact", issue.get("explanation", "")),
            "business_impact": issue.get("business_impact", issue.get("explanation", "")),
            "suggested_remediation": issue.get("suggested_fix", issue.get("recommended_fix", "")),
            "example_cleaning_code": issue.get("example_cleaning_code", "# No code available."),
            "ai_insight": issue.get("ai_insight", issue.get("explanation", "")),
        }

    def generate_executive_summary(
        self,
        issues: list[dict[str, Any]],
        score: int,
        dataset_context: dict[str, Any],
    ) -> str:
        overview = {
            "row_count": dataset_context.get("row_count"),
            "column_count": dataset_context.get("column_count"),
            "shape": (dataset_context.get("row_count"), dataset_context.get("column_count")),
        }
        return generate_explanation(issues, overview, score)

    def generate_fix_plan(
        self,
        issues: list[dict[str, Any]],
        score: int,
        dataset_context: dict[str, Any],
    ) -> str:
        if not issues:
            return "No remediation plan is needed because no deterministic issues were detected."
        severity_weight = {"critical": 15, "warning": 5, "minor": 2}
        sorted_issues = sorted(
            issues,
            key=lambda issue: severity_weight.get(issue.get("severity", "minor"), 0),
            reverse=True,
        )
        lines = ["### Rule-Based Fix Plan"]
        for priority, issue in enumerate(sorted_issues[:6], start=1):
            lines.append(
                f"- **Priority {priority} - {issue.get('display_name', issue.get('issue_type'))}** "
                f"(`{issue.get('column', 'dataset')}`): "
                f"{issue.get('suggested_fix', issue.get('recommended_fix', 'Review this issue.'))} "
                f"Expected impact: +{severity_weight.get(issue.get('severity', 'minor'), 2)} score."
            )
        return "\n".join(lines)

    def generate_data_contract(self, profile: dict[str, Any], issues: list[dict[str, Any]]) -> str:
        lines = ["### Rule-Based Data Contract Draft"]
        for column, stats in profile.get("columns", {}).items():
            lines.append(f"- `{column}` must remain `{stats.get('dtype')}`.")
            lines.append(f"- `{column}` missing rate should stay below 5% unless explicitly approved.")
            if stats.get("unique_ratio") == 1:
                lines.append(f"- `{column}` appears unique and should be monitored as a key or identifier.")
        if issues:
            lines.append("- Add validation checks for all detected critical and warning issues before production use.")
        return "\n".join(lines)

    def answer_dataset_question(
        self,
        question: str,
        issues: list[dict[str, Any]],
        score: int,
        dataset_context: dict[str, Any],
        drift_context: dict[str, Any] | None,
    ) -> str:
        critical = [issue for issue in issues if issue.get("severity") == "critical"]
        warnings = [issue for issue in issues if issue.get("severity") == "warning"]
        drift_text = ""
        if drift_context:
            drift_text = f" Drift risk is {drift_context.get('risk_label')} with score {drift_context.get('drift_score')}."
        return (
            f"Based on the deterministic analysis, the health score is {score}/100 with "
            f"{len(critical)} critical issues and {len(warnings)} warnings.{drift_text} "
            "For a more specific answer, ask about a particular column, issue type, or drift result."
        )

    def generate_cleaning_code(self, issue: dict[str, Any], dataset_context: dict[str, Any]) -> str:
        return f"```python\n{issue.get('example_cleaning_code', '# No code available.')}\n```"

    def generate_drift_explanation(self, drift_context: dict[str, Any]) -> str:
        return (
            drift_context.get("rule_based_explanation")
            or f"Drift risk is {drift_context.get('risk_label')} with score {drift_context.get('drift_score')}."
        )
