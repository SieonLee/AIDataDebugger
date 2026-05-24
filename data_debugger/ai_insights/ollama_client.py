"""Optional local Ollama explanation layer."""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Any


OLLAMA_CHAT_URL = "http://localhost:11434/api/chat"
OLLAMA_TAGS_URL = "http://localhost:11434/api/tags"
ISSUE_JSON_KEYS = {
    "why_this_matters",
    "ml_impact",
    "business_impact",
    "suggested_remediation",
    "example_cleaning_code",
    "ai_insight",
}


def check_ollama_available(model: str = "llama3.1", timeout: float = 2.0) -> bool:
    """Return True when the selected local Ollama model responds to /api/chat."""
    payload = {
        "model": model,
        "stream": False,
        "messages": [
            {"role": "system", "content": "Return only JSON."},
            {"role": "user", "content": '{"status":"ok"}'},
        ],
        "format": "json",
        "options": {"temperature": 0, "num_predict": 16},
    }
    try:
        _post_chat(payload, timeout=timeout)
        return True
    except Exception:
        return False


def check_ollama_server_available(timeout: float = 2.0) -> bool:
    """Return True when the Ollama HTTP server is reachable."""
    try:
        request = urllib.request.Request(OLLAMA_TAGS_URL, method="GET")
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.status == 200
    except Exception:
        return False


def list_ollama_models(timeout: float = 2.0) -> list[str]:
    """List locally installed Ollama models without loading them."""
    try:
        request = urllib.request.Request(OLLAMA_TAGS_URL, method="GET")
        with urllib.request.urlopen(request, timeout=timeout) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except Exception:
        return []
    return [model.get("name", "") for model in payload.get("models", []) if model.get("name")]


def generate_issue_insight(
    issue: dict[str, Any],
    dataset_context: dict[str, Any],
    model: str,
    timeout: float = 8.0,
) -> dict[str, str] | None:
    """Generate concise structured issue guidance with local Ollama."""
    structured_issue = _structured_issue_payload(issue, dataset_context)
    prompt = (
        "Generate concise JSON for an ML dataset debugging issue. "
        "Use only the structured issue data provided. Do not invent dataset rows. "
        "Return exactly these keys: why_this_matters, ml_impact, business_impact, "
        "suggested_remediation, example_cleaning_code, ai_insight.\n\n"
        f"Structured issue data:\n{json.dumps(structured_issue, ensure_ascii=False)}"
    )
    payload = _chat_payload(model, prompt, max_tokens=500)
    try:
        return _parse_issue_json(_post_chat(payload, timeout=timeout))
    except Exception:
        return None


def generate_executive_summary(
    issues: list[dict[str, Any]],
    score: int,
    dataset_context: dict[str, Any],
    model: str,
    timeout: float = 10.0,
) -> str | None:
    """Generate a short executive summary using only compact metadata."""
    compact_issues = [_structured_issue_payload(issue, dataset_context) for issue in issues[:20]]
    prompt = (
        "Write a concise markdown executive summary for an ML dataset reliability report. "
        "Use only the structured issue metadata and dataset context. Do not request or infer raw rows. "
        "Cover top risks, ML impact, business impact, and remediation priorities.\n\n"
        f"Health score: {score}\n"
        f"Dataset context: {json.dumps(_compact_dataset_context(dataset_context), ensure_ascii=False)}\n"
        f"Issues: {json.dumps(compact_issues, ensure_ascii=False)}"
    )
    payload = _chat_payload(model, prompt, max_tokens=800, json_mode=False)
    try:
        response = _post_chat(payload, timeout=timeout)
    except Exception:
        return None
    content = response.get("message", {}).get("content", "").strip()
    return content or None


def answer_dataset_question(
    question: str,
    issues: list[dict[str, Any]],
    score: int,
    dataset_context: dict[str, Any],
    drift_context: dict[str, Any] | None,
    model: str,
    timeout: float = 12.0,
) -> str | None:
    """Answer a natural language question using only analysis metadata."""
    prompt = (
        "Answer the user's dataset analysis question using only the provided metadata. "
        "Do not claim access to raw rows. If the metadata is insufficient, say what is missing. "
        "Be concise and operational.\n\n"
        f"Question: {question}\n"
        f"Health score: {score}\n"
        f"Dataset context: {json.dumps(_compact_dataset_context(dataset_context), ensure_ascii=False)}\n"
        f"Detected issues: {json.dumps(_compact_issues(issues), ensure_ascii=False)}\n"
        f"Drift context: {json.dumps(drift_context or {}, ensure_ascii=False)}"
    )
    return _generate_text(model, prompt, timeout=timeout, max_tokens=700)


def generate_fix_plan(
    issues: list[dict[str, Any]],
    score: int,
    dataset_context: dict[str, Any],
    model: str,
    timeout: float = 12.0,
) -> str | None:
    """Generate a prioritized remediation plan from deterministic issue objects."""
    prompt = (
        "Generate a prioritized remediation plan from detected deterministic issues. "
        "Return markdown with rows or bullets containing: priority, issue, reason, recommended action, expected impact. "
        "Do not add issues that are not in the input.\n\n"
        f"Health score: {score}\n"
        f"Dataset context: {json.dumps(_compact_dataset_context(dataset_context), ensure_ascii=False)}\n"
        f"Issues: {json.dumps(_compact_issues(issues), ensure_ascii=False)}"
    )
    return _generate_text(model, prompt, timeout=timeout, max_tokens=900)


def generate_cleaning_code(
    issue: dict[str, Any],
    dataset_context: dict[str, Any],
    model: str,
    timeout: float = 10.0,
) -> str | None:
    """Generate pandas cleaning code for one issue. Preview only."""
    prompt = (
        "Generate pandas cleaning code for the deterministic issue below. "
        "Do not execute code. Return only a short fenced python code block plus one sentence of caution. "
        "Use the provided column name exactly.\n\n"
        f"Issue: {json.dumps(_structured_issue_payload(issue, dataset_context), ensure_ascii=False)}"
    )
    return _generate_text(model, prompt, timeout=timeout, max_tokens=500)


def generate_data_contract(
    dataset_context: dict[str, Any],
    issues: list[dict[str, Any]],
    model: str,
    timeout: float = 12.0,
) -> str | None:
    """Draft validation rules from metadata and detected issues."""
    prompt = (
        "Draft a practical data contract for this dataset using only metadata and detected issues. "
        "Include validation rules such as dtype expectations, missing-rate limits, uniqueness checks, "
        "accepted categorical values when inferable from metadata, and monitoring notes. "
        "Return concise markdown.\n\n"
        f"Dataset context: {json.dumps(_compact_dataset_context(dataset_context), ensure_ascii=False)}\n"
        f"Issues: {json.dumps(_compact_issues(issues), ensure_ascii=False)}"
    )
    return _generate_text(model, prompt, timeout=timeout, max_tokens=900)


def generate_drift_explanation(
    drift_context: dict[str, Any],
    model: str,
    timeout: float = 12.0,
) -> str | None:
    """Explain deterministic drift results using local Ollama."""
    prompt = (
        "Explain these deterministic dataset drift results for ML observability. "
        "Cover what changed, why it matters, ML risk, business risk, and retraining implications. "
        "Do not invent metrics that are not present.\n\n"
        f"Drift context: {json.dumps(drift_context, ensure_ascii=False)}"
    )
    return _generate_text(model, prompt, timeout=timeout, max_tokens=900)


def _chat_payload(model: str, prompt: str, max_tokens: int, json_mode: bool = True) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model": model,
        "stream": False,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are a senior ML data reliability engineer. Be concise, operational, "
                    "and practical. Never ask for raw data."
                ),
            },
            {"role": "user", "content": prompt},
        ],
        "options": {"temperature": 0, "num_predict": max_tokens},
    }
    if json_mode:
        payload["format"] = "json"
    return payload


def _generate_text(model: str, prompt: str, timeout: float, max_tokens: int) -> str | None:
    payload = _chat_payload(model, prompt, max_tokens=max_tokens, json_mode=False)
    try:
        response = _post_chat(payload, timeout=timeout)
    except Exception:
        return None
    content = response.get("message", {}).get("content", "").strip()
    return content or None


def _post_chat(payload: dict[str, Any], timeout: float) -> dict[str, Any]:
    request = urllib.request.Request(
        OLLAMA_CHAT_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def _parse_issue_json(response: dict[str, Any]) -> dict[str, str] | None:
    content = response.get("message", {}).get("content", "").strip()
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


def _structured_issue_payload(issue: dict[str, Any], dataset_context: dict[str, Any]) -> dict[str, Any]:
    column = issue.get("column", "dataset")
    column_context = dataset_context.get("columns", {}).get(column, {})
    return {
        "column_name": column,
        "issue_type": issue.get("issue_type"),
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


def _compact_dataset_context(dataset_context: dict[str, Any]) -> dict[str, Any]:
    return {
        "row_count": dataset_context.get("row_count"),
        "column_count": dataset_context.get("column_count"),
        "columns": {
            column: {
                "dtype": stats.get("dtype"),
                "missing_rate": stats.get("missing_rate"),
                "unique_ratio": stats.get("unique_ratio"),
                "outlier_rate": stats.get("outlier_rate"),
            }
            for column, stats in list(dataset_context.get("columns", {}).items())[:50]
        },
    }


def _compact_issues(issues: list[dict[str, Any]]) -> list[dict[str, Any]]:
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
        for issue in issues[:30]
    ]
