"""AI-style recommendation helpers."""

from data_debugger.ai_insights.ollama_client import (
    answer_dataset_question,
    check_ollama_available,
    check_ollama_server_available,
    generate_cleaning_code,
    generate_data_contract,
    generate_drift_explanation,
    generate_executive_summary,
    generate_fix_plan,
    generate_issue_insight,
    list_ollama_models,
)
from data_debugger.ai_insights.recommendations import generate_recommendations

__all__ = [
    "check_ollama_available",
    "check_ollama_server_available",
    "answer_dataset_question",
    "generate_cleaning_code",
    "generate_data_contract",
    "generate_drift_explanation",
    "generate_executive_summary",
    "generate_fix_plan",
    "generate_issue_insight",
    "generate_recommendations",
    "list_ollama_models",
]
