"""LLM and fallback explanation generation."""

from __future__ import annotations

import os
from collections import Counter
from typing import Any


def generate_explanation(
    issues: list[dict[str, Any]],
    overview: dict[str, Any],
    health_score: int,
) -> str:
    """Generate a polished explanation, using OpenAI when configured.

    The app remains fully functional without an API key by returning a rule-based
    summary. Importing OpenAI is delayed so the dependency can be optional.
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return _fallback_explanation(issues, overview, health_score)

    try:
        from openai import OpenAI

        client = OpenAI(api_key=api_key)
        response = client.chat.completions.create(
            model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a senior data quality analyst. Explain issues in practical "
                        "ML and business terms. Be concise, specific, and action-oriented."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"Dataset overview: {overview}\n"
                        f"Health score: {health_score}\n"
                        f"Issues: {issues}\n\n"
                        "Write a concise markdown report for an AI engineering SaaS product with: "
                        "executive summary, top risks, ML impact, business impact, and remediation plan."
                    ),
                },
            ],
            temperature=0.2,
            max_tokens=900,
        )
        return response.choices[0].message.content or _fallback_explanation(issues, overview, health_score)
    except Exception as exc:  # pragma: no cover - depends on external service
        fallback = _fallback_explanation(issues, overview, health_score)
        return f"{fallback}\n\n_Note: OpenAI explanation failed, so the rule-based summary was used. Error: {exc}_"


def _fallback_explanation(issues: list[dict[str, Any]], overview: dict[str, Any], health_score: int) -> str:
    if not issues:
        return (
            "### Executive Summary\n"
            f"The dataset has a health score of **{health_score}/100**. No common data quality issues were detected by the MVP checks. "
            "This does not guarantee the data is perfect, but it is a strong starting point for exploratory analysis or modeling.\n\n"
            "### Recommended Next Steps\n"
            "- Validate business definitions and target leakage risks.\n"
            "- Review sampling logic and confirm the dataset represents the intended population.\n"
            "- Add domain-specific checks before production use."
        )

    counts = Counter(issue["severity"] for issue in issues)
    most_common_types = Counter(issue["issue_type"] for issue in issues).most_common(3)
    top_types = ", ".join(issue_type.replace("_", " ") for issue_type, _ in most_common_types)

    critical_text = ""
    critical_issues = [issue for issue in issues if issue["severity"] == "critical"]
    if critical_issues:
        critical_text = (
            " Critical issues should be handled before downstream modeling or executive reporting because they can materially distort results."
        )

    return (
        "### Executive Summary\n"
        f"The dataset contains **{overview['row_count']} rows** and **{overview['column_count']} columns** with a health score of "
        f"**{health_score}/100**. The scan found **{len(issues)} issues**: "
        f"{counts.get('critical', 0)} critical, {counts.get('warning', 0)} warnings, and {counts.get('minor', 0)} minor concerns."
        f"{critical_text}\n\n"
        "### What Matters Most\n"
        f"The most frequent issue areas are: **{top_types}**. In an ML context, these can affect feature reliability, validation quality, "
        "and model interpretability. In a business context, they can change counts, hide bad collection processes, or make segments look healthier than they are.\n\n"
        "### ML Impact\n"
        "These issues can reduce generalization, create leakage-prone features, destabilize feature distributions, or make offline validation too optimistic.\n\n"
        "### Business Impact\n"
        "Operationally, the same issues can distort KPIs, hide data collection failures, and make stakeholder decisions depend on unreliable segments.\n\n"
        "### Recommended Remediation Plan\n"
        "- Fix critical issues first, especially missingness, duplicates, constants, and extreme outliers.\n"
        "- Decide which ID-like or high-cardinality fields are keys versus model features.\n"
        "- Re-run the debugger after cleaning to compare the health score and remaining risks."
    )
