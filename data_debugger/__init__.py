"""AI-powered Data Debugger package."""

from data_debugger.checks import run_quality_checks
from data_debugger.cleaning import compare_cleaning_results, simulate_cleaning
from data_debugger.drift import compare_datasets
from data_debugger.issue_catalog import enrich_issues
from data_debugger.llm_report import generate_explanation
from data_debugger.profiler import profile_dataframe
from data_debugger.remediation import apply_issue_fix, preview_issue_fix
from data_debugger.report_generator import generate_markdown_report
from data_debugger.scoring import calculate_health_score

__all__ = [
    "calculate_health_score",
    "compare_cleaning_results",
    "compare_datasets",
    "enrich_issues",
    "generate_explanation",
    "generate_markdown_report",
    "apply_issue_fix",
    "preview_issue_fix",
    "profile_dataframe",
    "run_quality_checks",
    "simulate_cleaning",
]
