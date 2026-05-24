"""OpenAI API provider adapter."""

from __future__ import annotations

import os
from typing import Any

from data_debugger.ai_insights.providers.base import (
    AIProvider,
    cleaning_code_prompt,
    data_contract_prompt,
    drift_prompt,
    executive_summary_prompt,
    fix_plan_prompt,
    issue_insight_prompt,
    parse_issue_json,
    qa_prompt,
)
from data_debugger.ai_insights.providers.rule_based_provider import RuleBasedProvider


class OpenAIProvider(AIProvider):
    name = "OpenAI API"

    def __init__(self, model: str, api_key: str | None = None, timeout: float = 20.0) -> None:
        self.model = model
        self.api_key = (api_key or os.getenv("OPENAI_API_KEY") or "").strip()
        self.timeout = timeout
        self.fallback = RuleBasedProvider()

    def is_available(self) -> bool:
        if not self.api_key or not self.model:
            return False
        try:
            self._generate_text("Reply with: ok", max_tokens=8)
            return True
        except Exception:
            return False

    def generate_issue_insight(self, issue: dict[str, Any], dataset_context: dict[str, Any]) -> dict[str, str]:
        try:
            content = self._generate_text(issue_insight_prompt(issue, dataset_context), max_tokens=500, json_mode=True)
            parsed = parse_issue_json(content)
            return parsed or self.fallback.generate_issue_insight(issue, dataset_context)
        except Exception:
            return self.fallback.generate_issue_insight(issue, dataset_context)

    def generate_executive_summary(
        self,
        issues: list[dict[str, Any]],
        score: int,
        dataset_context: dict[str, Any],
    ) -> str:
        try:
            return self._generate_text(executive_summary_prompt(issues, score, dataset_context), max_tokens=800)
        except Exception:
            return self.fallback.generate_executive_summary(issues, score, dataset_context)

    def generate_fix_plan(
        self,
        issues: list[dict[str, Any]],
        score: int,
        dataset_context: dict[str, Any],
    ) -> str:
        try:
            return self._generate_text(fix_plan_prompt(issues, score, dataset_context), max_tokens=900)
        except Exception:
            return self.fallback.generate_fix_plan(issues, score, dataset_context)

    def generate_data_contract(self, profile: dict[str, Any], issues: list[dict[str, Any]]) -> str:
        try:
            return self._generate_text(data_contract_prompt(profile, issues), max_tokens=900)
        except Exception:
            return self.fallback.generate_data_contract(profile, issues)

    def answer_dataset_question(
        self,
        question: str,
        issues: list[dict[str, Any]],
        score: int,
        dataset_context: dict[str, Any],
        drift_context: dict[str, Any] | None,
    ) -> str:
        try:
            return self._generate_text(qa_prompt(question, issues, score, dataset_context, drift_context), max_tokens=700)
        except Exception:
            return self.fallback.answer_dataset_question(question, issues, score, dataset_context, drift_context)

    def generate_cleaning_code(self, issue: dict[str, Any], dataset_context: dict[str, Any]) -> str:
        try:
            return self._generate_text(cleaning_code_prompt(issue, dataset_context), max_tokens=500)
        except Exception:
            return self.fallback.generate_cleaning_code(issue, dataset_context)

    def generate_drift_explanation(self, drift_context: dict[str, Any]) -> str:
        try:
            return self._generate_text(drift_prompt(drift_context), max_tokens=900)
        except Exception:
            return self.fallback.generate_drift_explanation(drift_context)

    def _generate_text(self, prompt: str, max_tokens: int, json_mode: bool = False) -> str:
        from openai import OpenAI

        client = OpenAI(api_key=self.api_key, timeout=self.timeout)
        kwargs: dict[str, Any] = {
            "model": self.model,
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
            "temperature": 0,
            "max_tokens": max_tokens,
        }
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        response = client.chat.completions.create(**kwargs)
        content = response.choices[0].message.content if response.choices else ""
        return (content or "").strip()
