"""Markdown report generation."""

from __future__ import annotations

from typing import Any


def generate_markdown_report(
    overview: dict[str, Any],
    health_score: int,
    issues: list[dict[str, Any]],
    explanation: str,
    cleaning_comparison: dict[str, Any] | None = None,
) -> str:
    """Create a downloadable markdown report."""
    critical = [issue for issue in issues if issue["severity"] == "critical"]
    warnings = [issue for issue in issues if issue["severity"] == "warning"]
    minor = [issue for issue in issues if issue["severity"] == "minor"]

    lines = [
        "# AI Data Debugger Report",
        "",
        "## Executive Summary",
        f"This dataset scored **{health_score}/100** across automated ML and analytics quality checks.",
        f"The scan found **{len(critical)} critical**, **{len(warnings)} warning**, and **{len(minor)} minor** issues.",
        "The goal of this report is to explain which data risks matter, why they matter, and how to remediate them before modeling or business reporting.",
        "",
        "## Dataset Overview",
        f"- Rows: {overview['row_count']}",
        f"- Columns: {overview['column_count']}",
        f"- Memory usage: {overview['memory_usage_mb']} MB",
        f"- Column names: {', '.join(map(str, overview['columns']))}",
        "",
        "## Dataset Health Score",
        f"**{health_score}/100**",
        "",
        "## AI-Ready Explanation",
        explanation,
        "",
        "## Top Risks",
    ]

    if not issues:
        lines.append("No issues detected by the MVP checks.")
    else:
        for index, issue in enumerate(issues[:8], start=1):
            lines.extend(
                [
                    f"### {index}. {issue.get('display_name', issue['issue_type'].replace('_', ' ').title())}",
                    f"- Severity: {issue['severity']}",
                    f"- Column: {issue['column']}",
                    f"- Metric: {issue['metric']}",
                    f"- ML impact: {issue.get('ml_impact', issue['explanation'])}",
                    f"- Business impact: {issue.get('business_impact', issue['explanation'])}",
                    f"- AI insight: {issue.get('ai_insight', 'Review this issue for operational ML reliability.')}",
                    f"- Recommended fix: {issue.get('suggested_fix', issue['recommended_fix'])}",
                    "",
                    "```python",
                    issue.get("example_cleaning_code", "# No code snippet available."),
                    "```",
                    "",
                ]
            )

    lines.extend(
        [
            "## Recommended Remediation Plan",
            "1. Fix critical issues first, especially severe missingness, duplicates, constants, and extreme outliers.",
            "2. Remove or quarantine non-generalizable identifiers from model features.",
            "3. Review sparse categorical and high-uniqueness fields for leakage or overfitting risk.",
            "4. Re-run the debugger after cleaning and compare the health score.",
            "5. Add domain-specific validation before production use.",
            "",
            "## Cleaning Suggestions",
        ]
    )

    if issues:
        for issue in issues[:10]:
            lines.append(f"- `{issue['column']}`: {issue.get('suggested_fix', issue['recommended_fix'])}")
    else:
        lines.append("- No automated cleaning suggestions are needed from the current scan.")

    if cleaning_comparison:
        lines.extend(
            [
                "",
                "## Before/After Cleaning Comparison",
                f"- Original score: {cleaning_comparison['original_score']}",
                f"- Cleaned score: {cleaning_comparison['cleaned_score']}",
                f"- Improvement: {cleaning_comparison['improvement']:+d}",
                f"- Original issue count: {cleaning_comparison['original_issue_count']}",
                f"- Cleaned issue count: {cleaning_comparison['cleaned_issue_count']}",
                "",
                "### Cleaning Actions Applied",
            ]
        )
        for action in cleaning_comparison.get("actions", []):
            lines.append(f"- {action}")

    return "\n".join(lines)
