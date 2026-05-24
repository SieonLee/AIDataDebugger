"""Local Ollama provider adapter."""

from __future__ import annotations

from typing import Any

from data_debugger.ai_insights import ollama_client
from data_debugger.ai_insights.providers.base import AIProvider
from data_debugger.ai_insights.providers.rule_based_provider import RuleBasedProvider


class OllamaProvider(AIProvider):
    name = "Local Ollama"

    def __init__(self, model: str, timeout: float = 12.0) -> None:
        self.model = model
        self.timeout = timeout
        self.fallback = RuleBasedProvider()

    def is_available(self) -> bool:
        return bool(self.model) and ollama_client.check_ollama_available(model=self.model, timeout=20.0)

    def generate_issue_insight(self, issue: dict[str, Any], dataset_context: dict[str, Any]) -> dict[str, str]:
        return (
            ollama_client.generate_issue_insight(issue, dataset_context, model=self.model, timeout=8.0)
            or self.fallback.generate_issue_insight(issue, dataset_context)
        )

    def generate_executive_summary(
        self,
        issues: list[dict[str, Any]],
        score: int,
        dataset_context: dict[str, Any],
    ) -> str:
        return (
            ollama_client.generate_executive_summary(issues, score, dataset_context, model=self.model, timeout=10.0)
            or self.fallback.generate_executive_summary(issues, score, dataset_context)
        )

    def generate_fix_plan(
        self,
        issues: list[dict[str, Any]],
        score: int,
        dataset_context: dict[str, Any],
    ) -> str:
        return (
            ollama_client.generate_fix_plan(issues, score, dataset_context, model=self.model, timeout=self.timeout)
            or self.fallback.generate_fix_plan(issues, score, dataset_context)
        )

    def generate_data_contract(self, profile: dict[str, Any], issues: list[dict[str, Any]]) -> str:
        return (
            ollama_client.generate_data_contract(profile, issues, model=self.model, timeout=self.timeout)
            or self.fallback.generate_data_contract(profile, issues)
        )

    def answer_dataset_question(
        self,
        question: str,
        issues: list[dict[str, Any]],
        score: int,
        dataset_context: dict[str, Any],
        drift_context: dict[str, Any] | None,
    ) -> str:
        return (
            ollama_client.answer_dataset_question(
                question,
                issues,
                score,
                dataset_context,
                drift_context,
                model=self.model,
                timeout=self.timeout,
            )
            or self.fallback.answer_dataset_question(question, issues, score, dataset_context, drift_context)
        )

    def generate_cleaning_code(self, issue: dict[str, Any], dataset_context: dict[str, Any]) -> str:
        return (
            ollama_client.generate_cleaning_code(issue, dataset_context, model=self.model, timeout=10.0)
            or self.fallback.generate_cleaning_code(issue, dataset_context)
        )

    def generate_drift_explanation(self, drift_context: dict[str, Any]) -> str:
        return (
            ollama_client.generate_drift_explanation(drift_context, model=self.model, timeout=self.timeout)
            or self.fallback.generate_drift_explanation(drift_context)
        )
