"""Streamlit MVP for an AI-powered Data Debugger."""

from __future__ import annotations

import json
import os
import importlib.util
from io import BytesIO
from zipfile import ZipFile, ZIP_DEFLATED

import pandas as pd
import plotly.express as px
import streamlit as st

from data_debugger.ai_insights import (
    check_ollama_available,
    check_ollama_server_available,
    generate_recommendations,
    list_ollama_models,
)
from data_debugger.ai_insights.providers import AIProvider, OllamaProvider, OpenAIProvider, RuleBasedProvider
from data_debugger.cleaning import compare_cleaning_results, simulate_cleaning
from data_debugger.checks import run_quality_checks
from data_debugger.connectors import CSVConnector
from data_debugger.contracts import contract_to_json, contract_to_yaml, generate_data_contract as generate_validation_contract
from data_debugger.drift import compare_datasets
from data_debugger.drift.drift_visuals import (
    category_frequency_comparison,
    drift_issue_severity_chart,
    missing_drift_chart,
    numeric_distribution_overlay,
    psi_bar_chart,
)
from data_debugger.issue_catalog import enrich_issues, scoring_rules, top_risks
from data_debugger.llm_report import generate_explanation
from data_debugger.profiler import dtype_summary, profile_dataframe
from data_debugger.remediation import preview_issue_fix
from data_debugger.report_generator import generate_markdown_report
from data_debugger.roles import normalize_roles
from data_debugger.scoring import calculate_health_score, count_by_severity, score_breakdown
from data_debugger.storage import load_recent_runs, load_runs_for_dataset, save_analysis_run, schema_fingerprint_for_frame
from data_debugger.ui_components import inject_dashboard_css, severity_badge
from data_debugger.utils import observability_metrics
from data_debugger.visualizations import (
    cardinality_comparison_chart,
    correlation_heatmap,
    issue_breakdown_by_category,
    missing_value_heatmap,
    missing_values_chart,
    numeric_distribution_histogram,
    severity_distribution_chart,
    severity_pie_chart,
)


st.set_page_config(
    page_title="AI Data Debugger",
    page_icon=":mag:",
    layout="wide",
)


MAX_UPLOAD_SIZE_BYTES = 200 * 1024 * 1024
OLLAMA_MODELS = [
    "llama3.1",
    "llama3.2",
    "mistral",
    "qwen2.5-coder",
    "qwen3.5:latest",
    "qwen2.5:3b",
    "qwen2.5:1.5b",
    "qwen3:4b",
    "custom",
]
OPENAI_MODELS = ["gpt-4o-mini", "gpt-4.1-mini", "gpt-4.1", "custom"]


@st.cache_data(show_spinner=False)
def _cached_explanation(issues_json: str, overview_json: str, health_score: int) -> str:
    """Avoid repeated LLM calls on Streamlit reruns for the same dataset scan."""
    issues = json.loads(issues_json)
    overview = json.loads(overview_json)
    return generate_explanation(issues, overview, health_score)


@st.cache_data(show_spinner=False, ttl=30)
def _cached_ollama_available(model: str) -> bool:
    return check_ollama_available(model=model, timeout=20.0)


@st.cache_data(show_spinner=False, ttl=30)
def _cached_ollama_server_available() -> bool:
    return check_ollama_server_available(timeout=2.0)


@st.cache_data(show_spinner=False, ttl=30)
def _cached_ollama_models() -> list[str]:
    return list_ollama_models(timeout=2.0)


def _read_csv(uploaded_file) -> pd.DataFrame:
    try:
        uploaded_file.seek(0)
        return CSVConnector().load(uploaded_file)
    except pd.errors.EmptyDataError as exc:
        raise ValueError("The uploaded CSV is empty. Please upload a file with headers and data rows.") from exc
    except UnicodeDecodeError as exc:
        raise ValueError("The file encoding could not be read as a standard CSV. Try saving it as UTF-8.") from exc
    except pd.errors.ParserError as exc:
        raise ValueError("The CSV could not be parsed. Check delimiters, quotes, and malformed rows.") from exc


def _default_monitor_name(filename: str) -> str:
    name = os.path.splitext(os.path.basename(filename))[0]
    for prefix in ("baseline_", "broken_", "new_", "production_", "prod_"):
        if name.startswith(prefix):
            return name.removeprefix(prefix)
    return name


def _build_dataset_context(df: pd.DataFrame) -> dict:
    columns = {}
    row_count = len(df)
    for column in df.columns:
        series = df[column]
        summary_stats = {}
        outlier_rate = None
        if pd.api.types.is_numeric_dtype(series):
            numeric = pd.to_numeric(series, errors="coerce").dropna()
            if not numeric.empty:
                summary_stats = {
                    "mean": round(float(numeric.mean()), 4),
                    "std": round(float(numeric.std(ddof=0)), 4),
                    "min": round(float(numeric.min()), 4),
                    "p25": round(float(numeric.quantile(0.25)), 4),
                    "median": round(float(numeric.median()), 4),
                    "p75": round(float(numeric.quantile(0.75)), 4),
                    "max": round(float(numeric.max()), 4),
                }
                q1 = numeric.quantile(0.25)
                q3 = numeric.quantile(0.75)
                iqr = q3 - q1
                if iqr != 0:
                    outlier_rate = round(float(((numeric < q1 - 1.5 * iqr) | (numeric > q3 + 1.5 * iqr)).mean()), 4)
        columns[str(column)] = {
            "dtype": str(series.dtype),
            "missing_rate": round(float(series.isna().mean()), 4) if row_count else 0.0,
            "unique_ratio": round(float(series.nunique(dropna=True) / max(1, row_count)), 4),
            "outlier_rate": outlier_rate,
            "summary_stats": summary_stats,
        }

    return {
        "row_count": row_count,
        "column_count": df.shape[1],
        "columns": columns,
    }


def _ai_cache_get(key: str):
    return st.session_state.setdefault("ai_provider_cache", {}).get(key)


def _ai_cache_set(key: str, value):
    st.session_state.setdefault("ai_provider_cache", {})[key] = value
    return value


def _apply_ai_provider_insights(
    issues: list[dict],
    dataset_context: dict,
    provider: AIProvider,
    provider_id: str,
) -> list[dict]:
    context_json = json.dumps(dataset_context, sort_keys=True)
    updated_issues = []
    # Keep first-render latency bounded. Less severe issues keep deterministic text.
    ai_candidate_keys = {_issue_key(issue) for issue in top_risks(issues, limit=3)}
    for issue in issues:
        updated = dict(issue)
        insight = None
        if _issue_key(issue) in ai_candidate_keys:
            issue_json = json.dumps(issue, sort_keys=True)
            cache_key = f"issue:{provider_id}:{issue_json}:{context_json}"
            insight = _ai_cache_get(cache_key)
            if insight is None:
                insight = _ai_cache_set(cache_key, provider.generate_issue_insight(issue, dataset_context))
        if insight:
            updated["why_this_matters"] = insight.get("why_this_matters", updated.get("why_this_matters", ""))
            updated["ml_impact"] = insight.get("ml_impact", updated.get("ml_impact", ""))
            updated["business_impact"] = insight.get("business_impact", updated.get("business_impact", ""))
            updated["suggested_fix"] = insight.get("suggested_remediation", updated.get("suggested_fix", ""))
            updated["recommended_fix"] = updated["suggested_fix"]
            updated["example_cleaning_code"] = insight.get(
                "example_cleaning_code",
                updated.get("example_cleaning_code", ""),
            )
            updated["ai_insight"] = insight.get("ai_insight", updated.get("ai_insight", ""))
        updated_issues.append(updated)
    return updated_issues


def _compact_drift_context(drift_summary: dict | None) -> dict:
    if not drift_summary:
        return {}
    issue_table = drift_summary.get("issue_table")
    missing_table = drift_summary.get("missing_table")
    numeric_table = drift_summary.get("numeric_table")
    categorical_table = drift_summary.get("categorical_table")
    return {
        "drift_score": drift_summary.get("drift_score"),
        "risk_label": drift_summary.get("risk_label"),
        "severity_summary": drift_summary.get("severity_summary"),
        "added_columns": drift_summary.get("added_columns"),
        "removed_columns": drift_summary.get("removed_columns"),
        "dtype_changes": drift_summary.get("dtype_changes"),
        "issues": issue_table.head(20).to_dict("records") if isinstance(issue_table, pd.DataFrame) else [],
        "missing_drift": missing_table.head(30).to_dict("records") if isinstance(missing_table, pd.DataFrame) else [],
        "numeric_drift": numeric_table.sort_values("psi", ascending=False).head(20).to_dict("records")
        if isinstance(numeric_table, pd.DataFrame) and "psi" in numeric_table
        else [],
        "categorical_drift": categorical_table.head(20).to_dict("records")
        if isinstance(categorical_table, pd.DataFrame)
        else [],
        "rule_based_explanation": drift_summary.get("explanation"),
    }


def _issue_key(issue: dict) -> str:
    raw = f"{issue.get('issue_type')}-{issue.get('column')}-{issue.get('metric')}"
    return "".join(char if char.isalnum() else "_" for char in raw)


def _dataframe_to_parquet_bytes(df: pd.DataFrame) -> bytes | None:
    if importlib.util.find_spec("pyarrow") is None:
        return None
    buffer = BytesIO()
    df.to_parquet(buffer, index=False)
    return buffer.getvalue()


def _cleaned_dataset_zip(cleaned_df: pd.DataFrame, metadata: dict) -> bytes:
    buffer = BytesIO()
    with ZipFile(buffer, "w", compression=ZIP_DEFLATED) as archive:
        archive.writestr("cleaned_dataset_preview.csv", cleaned_df.to_csv(index=False))
        archive.writestr("cleaning_metadata.json", json.dumps(metadata, indent=2, sort_keys=True, default=str))
    return buffer.getvalue()


def _render_cleaned_export(
    cleaned_df: pd.DataFrame,
    label: str,
    original_rows: int,
    original_columns: int,
    original_score: int,
    original_issue_count: int,
    cleaned_score: int,
    cleaned_issue_count: int,
    key_prefix: str,
) -> None:
    st.markdown(f"**{label}**")
    row_col, col_col, score_col, issue_col = st.columns(4)
    row_col.metric("Rows", cleaned_df.shape[0], cleaned_df.shape[0] - original_rows)
    col_col.metric("Columns", cleaned_df.shape[1], cleaned_df.shape[1] - original_columns)
    score_col.metric("Health score", cleaned_score, cleaned_score - original_score)
    issue_col.metric("Issue count", cleaned_issue_count, cleaned_issue_count - original_issue_count, delta_color="inverse")
    metadata = {
        "label": label,
        "original": {
            "rows": original_rows,
            "columns": original_columns,
            "health_score": original_score,
            "issue_count": original_issue_count,
        },
        "cleaned_preview": {
            "rows": cleaned_df.shape[0],
            "columns": cleaned_df.shape[1],
            "health_score": cleaned_score,
            "issue_count": cleaned_issue_count,
        },
    }
    st.download_button(
        "Download cleaned dataset preview (CSV)",
        data=cleaned_df.to_csv(index=False).encode("utf-8"),
        file_name="cleaned_dataset_preview.csv",
        mime="text/csv",
        key=f"{key_prefix}_csv",
    )
    st.download_button(
        "Download cleaned preview package (ZIP)",
        data=_cleaned_dataset_zip(cleaned_df, metadata),
        file_name="cleaned_dataset_preview.zip",
        mime="application/zip",
        key=f"{key_prefix}_zip",
    )
    parquet_bytes = _dataframe_to_parquet_bytes(cleaned_df)
    if parquet_bytes:
        st.download_button(
            "Download cleaned dataset preview (Parquet)",
            data=parquet_bytes,
            file_name="cleaned_dataset_preview.parquet",
            mime="application/octet-stream",
            key=f"{key_prefix}_parquet",
        )
    else:
        st.caption("Parquet export is available when `pyarrow` is installed.")


def _render_role_setup(df: pd.DataFrame) -> dict:
    columns = [""] + list(map(str, df.columns))
    with st.expander("Dataset role setup", expanded=True):
        st.caption("Optional roles reduce ML false positives and prepare this scan for repeatable reliability workflows.")
        role_col_1, role_col_2, role_col_3 = st.columns(3)
        with role_col_1:
            target_column = st.selectbox("Target column", columns, index=0)
        with role_col_2:
            timestamp_column = st.selectbox("Timestamp column", columns, index=0)
        with role_col_3:
            entity_id_column = st.selectbox("Entity ID column", columns, index=0)
        protected_columns = st.multiselect("Protected columns to ignore", list(map(str, df.columns)))
        exclude_ml_columns = st.multiselect("Columns to exclude from ML checks", list(map(str, df.columns)))
    return normalize_roles(
        {
            "target_column": target_column,
            "timestamp_column": timestamp_column,
            "entity_id_column": entity_id_column,
            "protected_columns": protected_columns,
            "exclude_ml_columns": exclude_ml_columns,
        }
    )


def _render_issue_card(issue: dict, df: pd.DataFrame, roles: dict | None = None) -> None:
    title = f"{issue['display_name']} - {issue['column']} - {issue['metric']}"
    with st.expander(title):
        st.markdown(severity_badge(issue["severity"]), unsafe_allow_html=True)
        st.caption(f"Affected column: {issue['column']}")

        why_tab, impact_tab, fix_tab, code_tab, insight_tab, action_tab = st.tabs(
            ["Why this matters", "Impact", "Remediation", "Code", "AI insight", "Apply fix"]
        )
        with why_tab:
            st.write(issue["why_this_matters"])
        with impact_tab:
            st.markdown(f"**ML impact:** {issue['ml_impact']}")
            st.markdown(f"**Business impact:** {issue['business_impact']}")
        with fix_tab:
            st.write(issue["suggested_fix"])
        with code_tab:
            st.code(issue["example_cleaning_code"], language="python")
        with insight_tab:
            st.info(issue["ai_insight"])
        with action_tab:
            preview = preview_issue_fix(df, issue, roles=roles)
            before_col, after_col, gain_col = st.columns(3)
            before_col.metric("Current score", preview["before_score"])
            after_col.metric("After fix", preview["after_score"])
            gain_col.metric("Improvement", f"{preview['improvement']:+d}")
            st.caption(
                f"Shape: {preview['before_shape']} -> {preview['after_shape']} | "
                f"Missing cells: {preview['before_missing']} -> {preview['after_missing']}"
            )
            for action in preview["actions"]:
                st.markdown(f"- {action}")
            _render_cleaned_export(
                preview["fixed_df"],
                "Cleaned dataset preview",
                df.shape[0],
                df.shape[1],
                preview["before_score"],
                len(enrich_issues(run_quality_checks(df, roles=roles))),
                preview["after_score"],
                preview["after_issue_count"],
                f"issue_{_issue_key(issue)}",
            )
            if st.button("Apply fix and rescan", key=f"apply_{_issue_key(issue)}"):
                st.session_state["working_df"] = preview["fixed_df"]
                st.session_state.setdefault("applied_fixes", []).extend(preview["actions"])
                st.session_state.pop("cleaning_comparison", None)
                st.rerun()


def _severity_table(issues: list[dict], severity: str, df: pd.DataFrame, roles: dict | None = None) -> None:
    selected = [issue for issue in issues if issue["severity"] == severity]
    if not selected:
        st.success(f"No {severity} issues found.")
        return

    for issue in selected:
        _render_issue_card(issue, df, roles=roles)


def _health_label(score: int) -> str:
    if score >= 85:
        return "Strong"
    if score >= 65:
        return "Needs review"
    if score >= 40:
        return "High risk"
    return "Critical"


def _score_uploaded_df(df: pd.DataFrame, roles: dict | None = None) -> int:
    return calculate_health_score(enrich_issues(run_quality_checks(df, roles=roles)))


def main() -> None:
    inject_dashboard_css()
    st.title("AI Data Debugger")
    st.caption("An AI copilot for debugging ML and analytics datasets.")

    st.sidebar.header("AI Provider")
    use_ai = st.sidebar.toggle("Use AI Explanation", value=False)
    provider_choice = st.sidebar.selectbox(
        "Provider",
        ["Rule-based only", "Local Ollama", "OpenAI API"],
        disabled=not use_ai,
    )
    st.sidebar.caption("API providers receive dataset metadata and issue summaries, not raw data rows.")

    ai_provider: AIProvider = RuleBasedProvider()
    provider_available = False
    provider_id = "rule-based"
    ollama_model = ""
    openai_model = ""

    if not use_ai or provider_choice == "Rule-based only":
        st.sidebar.info("AI Provider: Rule-based deterministic fallback")
    elif provider_choice == "Local Ollama":
        installed_models = _cached_ollama_models()
        model_options = list(dict.fromkeys([*installed_models, *OLLAMA_MODELS]))
        selected_model = st.sidebar.selectbox("Ollama model", model_options, index=0)
        custom_model = ""
        if selected_model == "custom":
            custom_model = st.sidebar.text_input("Custom Ollama model name", value="")
        ollama_model = custom_model.strip() if selected_model == "custom" and custom_model.strip() else selected_model
        server_available = _cached_ollama_server_available()
        if selected_model == "custom" and not custom_model.strip():
            st.sidebar.warning("Local AI: Enter a custom model name")
        elif not server_available:
            st.sidebar.warning("Local AI: Server not running, using rule-based fallback")
        else:
            provider_available = _cached_ollama_available(ollama_model)
        if provider_available:
            st.sidebar.success(f"Local AI: Connected ({ollama_model})")
            ai_provider = OllamaProvider(model=ollama_model)
            provider_id = f"ollama:{ollama_model}"
        elif server_available:
            st.sidebar.warning(f"Local AI: Server connected, but model failed to load ({ollama_model})")
        else:
            st.sidebar.warning("Local AI: Not available, using rule-based fallback")
    elif provider_choice == "OpenAI API":
        env_key_available = bool(os.getenv("OPENAI_API_KEY"))
        api_key_input = st.sidebar.text_input(
            "OpenAI API key",
            value="",
            type="password",
            placeholder="Uses OPENAI_API_KEY if blank",
        )
        selected_openai_model = st.sidebar.selectbox("OpenAI model", OPENAI_MODELS, index=0)
        custom_openai_model = ""
        if selected_openai_model == "custom":
            custom_openai_model = st.sidebar.text_input("Custom OpenAI model name", value="")
        openai_model = (
            custom_openai_model.strip()
            if selected_openai_model == "custom" and custom_openai_model.strip()
            else selected_openai_model
        )
        api_key = api_key_input.strip() or os.getenv("OPENAI_API_KEY", "")
        if selected_openai_model == "custom" and not custom_openai_model.strip():
            st.sidebar.warning("OpenAI API: Enter a custom model name")
        elif not api_key:
            st.sidebar.warning("OpenAI API: No API key found, using rule-based fallback")
        elif importlib.util.find_spec("openai") is None:
            st.sidebar.warning("OpenAI API: Python SDK not installed, using rule-based fallback")
        else:
            try:
                ai_provider = OpenAIProvider(model=openai_model, api_key=api_key)
                provider_available = True
                provider_id = f"openai:{openai_model}:{abs(hash(api_key))}"
                source = "sidebar key" if api_key_input.strip() else "OPENAI_API_KEY"
                st.sidebar.success(f"OpenAI API: Configured via {source}")
                if env_key_available and api_key_input.strip():
                    st.sidebar.caption("Sidebar key is used for this session only.")
            except Exception:
                st.sidebar.warning("OpenAI API: SDK unavailable, using rule-based fallback")
    upload_col, compare_col = st.columns([1, 1])
    with upload_col:
        uploaded_file = st.file_uploader("1. Upload baseline / original CSV", type=["csv"])
    with compare_col:
        comparison_file = st.file_uploader(
            "2. Upload new / changed CSV for comparison",
            type=["csv"],
            key="comparison_csv",
        )
    if uploaded_file is None:
        st.info("Upload a CSV file to start profiling.")
        return

    if uploaded_file.size > MAX_UPLOAD_SIZE_BYTES:
        st.error("File exceeds the 200 MB limit. Downsample or split the file before uploading.")
        return

    upload_fingerprint = f"{uploaded_file.name}:{uploaded_file.size}"
    if st.session_state.get("active_upload_fingerprint") != upload_fingerprint:
        st.session_state["active_upload_fingerprint"] = upload_fingerprint
        st.session_state.pop("cleaning_comparison", None)
        st.session_state.pop("working_df", None)
        st.session_state["applied_fixes"] = []

    try:
        original_df = _read_csv(uploaded_file)
    except ValueError as exc:
        st.error(str(exc))
        return

    if original_df.empty:
        st.warning("The CSV loaded successfully, but it has no data rows. The debugger can still report the empty dataset issue.")

    monitor_name = st.text_input(
        "Dataset / monitor name",
        value=_default_monitor_name(uploaded_file.name),
        help="Runs with the same monitor name and schema are grouped in History trends.",
    ).strip() or _default_monitor_name(uploaded_file.name)

    if "working_df" not in st.session_state:
        st.session_state["working_df"] = original_df.copy()
    df = st.session_state["working_df"]
    roles = _render_role_setup(df)
    original_score = _score_uploaded_df(original_df, roles=roles)
    schema_fingerprint = schema_fingerprint_for_frame(df)

    with st.spinner("Analyzing dataset quality signals..."):
        overview = profile_dataframe(df)
        dataset_context = _build_dataset_context(df)
        deterministic_issues = enrich_issues(run_quality_checks(df, roles=roles))
        issues = deterministic_issues
        if use_ai and provider_available and issues:
            issues = _apply_ai_provider_insights(issues, dataset_context, ai_provider, provider_id)
        health_score = calculate_health_score(issues)
        severity_counts = count_by_severity(issues)
        recommendations = generate_recommendations(issues)
        obs_metrics = observability_metrics(df, issues, health_score)
        cleaning_state = st.session_state.get("cleaning_comparison")
        if use_ai and provider_available and deterministic_issues:
            summary_cache_key = (
                f"summary:{provider_id}:"
                f"{json.dumps(deterministic_issues, sort_keys=True)}:"
                f"{health_score}:{json.dumps(dataset_context, sort_keys=True)}"
            )
            explanation = _ai_cache_get(summary_cache_key)
            if explanation is None:
                explanation = _ai_cache_set(
                    summary_cache_key,
                    ai_provider.generate_executive_summary(deterministic_issues, health_score, dataset_context),
                )
        else:
            explanation = _cached_explanation(
                json.dumps(issues, sort_keys=True),
                json.dumps(overview, sort_keys=True),
                health_score,
            )
    comparison_df = None
    comparison_error = None
    if comparison_file is not None:
        try:
            comparison_df = _read_csv(comparison_file)
        except ValueError as exc:
            comparison_error = str(exc)
    run_signature = json.dumps(
        {
            "upload": upload_fingerprint,
            "shape": df.shape,
            "score": health_score,
            "issues": len(issues),
            "provider": provider_id if use_ai and provider_available else "rule-based",
            "roles": roles,
            "monitor_name": monitor_name,
            "applied_fixes": st.session_state.get("applied_fixes", []),
        },
        sort_keys=True,
        default=str,
    )
    if st.session_state.get("last_saved_run_signature") != run_signature:
        st.session_state["last_saved_run_id"] = save_analysis_run(
            dataset_name=monitor_name,
            row_count=overview["row_count"],
            column_count=overview["column_count"],
            health_score=health_score,
            issues=issues,
            severity_counts=severity_counts,
            provider_used=provider_choice if use_ai and provider_available else "Rule-based only",
            schema_fingerprint=schema_fingerprint,
            config_summary={
                "roles": roles,
                "schema_fingerprint": schema_fingerprint,
                "provider": provider_choice if use_ai else "Rule-based only",
                "working_copy": bool(st.session_state.get("applied_fixes")),
            },
        )
        st.session_state["last_saved_run_signature"] = run_signature
    if comparison_df is not None:
        comparison_issues = enrich_issues(run_quality_checks(comparison_df, roles=roles))
        comparison_score = calculate_health_score(comparison_issues)
        comparison_counts = count_by_severity(comparison_issues)
        comparison_signature = json.dumps(
            {
                "comparison": f"{comparison_file.name}:{comparison_file.size}",
                "shape": comparison_df.shape,
                "score": comparison_score,
                "issues": len(comparison_issues),
                "roles": roles,
                "monitor_name": monitor_name,
            },
            sort_keys=True,
            default=str,
        )
        if st.session_state.get("last_saved_comparison_signature") != comparison_signature:
            save_analysis_run(
                dataset_name=monitor_name,
                row_count=comparison_df.shape[0],
                column_count=comparison_df.shape[1],
                health_score=comparison_score,
                issues=comparison_issues,
                severity_counts=comparison_counts,
                provider_used=f"Comparison batch ({provider_choice if use_ai and provider_available else 'Rule-based only'})",
                schema_fingerprint=schema_fingerprint_for_frame(comparison_df),
                config_summary={
                    "roles": roles,
                    "schema_fingerprint": schema_fingerprint_for_frame(comparison_df),
                    "source_file": comparison_file.name,
                    "comparison_to": uploaded_file.name,
                },
            )
            st.session_state["last_saved_comparison_signature"] = comparison_signature
    markdown_report = generate_markdown_report(overview, health_score, issues, explanation, cleaning_state)
    drift_summary_for_ai = None
    if comparison_df is not None:
        drift_summary_for_ai = compare_datasets(original_df, comparison_df)

    score_col, rows_col, cols_col, issues_col = st.columns(4)
    score_col.metric("Health score", f"{health_score}/100", _health_label(health_score), delta_color="off")
    rows_col.metric("Rows", overview["row_count"])
    cols_col.metric("Columns", overview["column_count"])
    issues_col.metric("Issues", len(issues))

    obs_col_1, obs_col_2, obs_col_3, obs_col_4 = st.columns(4)
    obs_col_1.metric("Issue density", obs_metrics["issue_density"])
    obs_col_2.metric("Problematic columns", f"{obs_metrics['problematic_columns_pct']}%")
    obs_col_3.metric("Estimated ML risk", obs_metrics["estimated_ml_risk_level"])
    obs_col_4.metric("Cleanliness trend", obs_metrics["cleanliness_trend"])
    obs_col_4.caption(obs_metrics["cleanliness_trend_detail"])

    if comparison_error:
        st.error(f"Could not load comparison dataset: {comparison_error}")
    elif comparison_df is not None and drift_summary_for_ai is not None:
        st.success(f"Comparison active: `{uploaded_file.name}` vs `{comparison_file.name}`")
        cmp_col_1, cmp_col_2, cmp_col_3, cmp_col_4 = st.columns(4)
        cmp_col_1.metric("Baseline health", f"{health_score}/100")
        cmp_col_2.metric("New batch health", f"{comparison_score}/100", comparison_score - health_score)
        cmp_col_3.metric("Drift risk score", f"{drift_summary_for_ai['drift_score']}/100", drift_summary_for_ai["risk_label"], delta_color="off")
        cmp_col_4.metric("Drift issues", len(drift_summary_for_ai["issues"]))
        st.caption("Open the Data Drift tab for missing-value drift, distribution drift, categorical changes, and schema changes.")
    else:
        st.info("Upload a second CSV in the comparison box to compare the baseline against a new or changed dataset.")

    if st.session_state.get("applied_fixes"):
        delta = health_score - original_score
        st.success(f"Working dataset score moved from {original_score} to {health_score} ({delta:+d}).")
        if st.button("Reset to uploaded baseline"):
            st.session_state["working_df"] = original_df.copy()
            st.session_state["applied_fixes"] = []
            st.session_state.pop("cleaning_comparison", None)
            st.rerun()

    (
        tab_overview,
        tab_drift,
        tab_recommendations,
        tab_cleaning,
        tab_report,
        tab_history,
        tab_investigate,
        tab_visuals,
        tab_ai,
        tab_methodology,
    ) = st.tabs(
        [
            "Overview",
            "Data Drift",
            "Recommendations",
            "Cleaning simulation",
            "Report",
            "History",
            "Investigate",
            "Visuals",
            "AI Copilot",
            "Methodology",
        ]
    )

    with tab_overview:
        st.subheader("Dataset Snapshot")
        st.write(f"Shape: **{overview['shape'][0]} rows x {overview['shape'][1]} columns**")
        st.dataframe(dtype_summary(df), use_container_width=True)

        if comparison_df is not None and drift_summary_for_ai is not None:
            st.subheader("Baseline vs New Dataset")
            st.write(f"Comparing **{uploaded_file.name}** against **{comparison_file.name}**.")
            compare_shape_col_1, compare_shape_col_2, compare_shape_col_3 = st.columns(3)
            compare_shape_col_1.metric("Baseline shape", f"{original_df.shape[0]} x {original_df.shape[1]}")
            compare_shape_col_2.metric("New shape", f"{comparison_df.shape[0]} x {comparison_df.shape[1]}")
            compare_shape_col_3.metric("Drift risk", drift_summary_for_ai["risk_label"])

        st.subheader("Top Risks")
        risks = top_risks(issues, limit=5)
        if risks:
            for issue in risks:
                st.markdown(
                    f"- **{issue['display_name']}** in `{issue['column']}` ({issue['severity']}): {issue['ml_impact']}"
                )
        else:
            st.success("No common ML or analytics data risks were detected by the current scan.")

        st.subheader("First 10 Rows")
        st.dataframe(df.head(10), use_container_width=True)

        if st.session_state.get("applied_fixes"):
            st.subheader("Applied Fixes")
            for fix in st.session_state["applied_fixes"]:
                st.markdown(f"- {fix}")

    with tab_investigate:
        st.subheader("Interactive Issue Investigation")
        critical_col, warning_col, minor_col = st.columns(3)
        critical_col.metric("Critical", severity_counts["critical"])
        warning_col.metric("Warnings", severity_counts["warning"])
        minor_col.metric("Minor", severity_counts["minor"])

        critical_tab, warning_tab, minor_tab, raw_tab = st.tabs(["Critical", "Warnings", "Minor", "Raw issue list"])
        with critical_tab:
            _severity_table(issues, "critical", df, roles=roles)
        with warning_tab:
            _severity_table(issues, "warning", df, roles=roles)
        with minor_tab:
            _severity_table(issues, "minor", df, roles=roles)
        with raw_tab:
            st.dataframe(pd.DataFrame(issues), use_container_width=True)

    with tab_recommendations:
        st.subheader("AI-Generated Cleaning Recommendations")
        if not recommendations:
            st.success("No remediation recommendations are needed from the current scan.")
        for recommendation in recommendations:
            with st.container(border=True):
                st.markdown(f"**Priority {recommendation['priority']}: {recommendation['title']}**")
                st.markdown(severity_badge(recommendation["severity"]), unsafe_allow_html=True)
                st.write(f"Column: `{recommendation['column']}`")
                st.write(f"Expected impact: **+{recommendation['expected_impact']} score**")
                st.write(f"Downstream ML risk: {recommendation['downstream_ml_risk']}")
                st.write(f"Recommended action: {recommendation['recommended_action']}")

    with tab_visuals:
        st.subheader("Dataset Quality Signals")
        left_col, right_col = st.columns(2)
        with left_col:
            st.plotly_chart(missing_values_chart(df), use_container_width=True)
        with right_col:
            st.plotly_chart(severity_pie_chart(issues), use_container_width=True)

        heatmap_col, severity_col = st.columns(2)
        with heatmap_col:
            st.subheader("Missing Value Heatmap")
            st.plotly_chart(missing_value_heatmap(df), use_container_width=True)
        with severity_col:
            st.subheader("Severity Distribution")
            st.plotly_chart(severity_distribution_chart(issues), use_container_width=True)

        numeric_columns = list(df.select_dtypes(include="number").columns)
        if numeric_columns:
            selected_numeric = st.selectbox("Numeric feature distribution", numeric_columns)
            st.plotly_chart(numeric_distribution_histogram(df, selected_numeric), use_container_width=True)
        else:
            st.info("No numeric columns are available for distribution histograms.")

        st.subheader("Correlation Heatmap")
        st.plotly_chart(correlation_heatmap(df), use_container_width=True)

        card_col, breakdown_col = st.columns(2)
        with card_col:
            st.subheader("Cardinality Comparison")
            st.plotly_chart(cardinality_comparison_chart(df), use_container_width=True)
        with breakdown_col:
            st.subheader("Issue Breakdown by Category")
            st.plotly_chart(issue_breakdown_by_category(issues), use_container_width=True)

    with tab_cleaning:
        st.subheader("Before/After Cleaning Simulation")
        st.caption("Apply conservative cleaning actions to estimate measurable impact before changing source data.")

        action_col_1, action_col_2, action_col_3 = st.columns(3)
        with action_col_1:
            drop_duplicates = st.checkbox("Drop duplicates", value=True)
        with action_col_2:
            fill_numeric = st.checkbox("Fill numeric missing values with median", value=True)
        with action_col_3:
            remove_constants = st.checkbox("Remove constant columns", value=True)

        if st.button("Apply suggested cleaning and rerun scan", type="primary"):
            cleaned_df, cleaning_actions = simulate_cleaning(
                df,
                drop_duplicates=drop_duplicates,
                fill_numeric_median=fill_numeric,
                remove_constant_columns=remove_constants,
            )
            comparison = compare_cleaning_results(df, cleaned_df, roles=roles)
            comparison["actions"] = cleaning_actions
            comparison["cleaned_shape"] = cleaned_df.shape
            comparison["cleaned_csv"] = cleaned_df.to_csv(index=False)
            st.session_state["cleaning_comparison"] = comparison
            cleaning_state = comparison

        if cleaning_state:
            original_col, cleaned_col, improvement_col = st.columns(3)
            original_col.metric("Original score", cleaning_state["original_score"])
            cleaned_col.metric("Cleaned score", cleaning_state["cleaned_score"])
            improvement_col.metric("Improvement", f"{cleaning_state['improvement']:+d}", delta_color="normal")

            st.markdown("**Cleaning actions applied:**")
            for action in cleaning_state["actions"]:
                st.markdown(f"- {action}")

            _render_cleaned_export(
                pd.read_csv(BytesIO(cleaning_state["cleaned_csv"].encode("utf-8"))),
                "Cleaned dataset preview",
                df.shape[0],
                df.shape[1],
                cleaning_state["original_score"],
                cleaning_state["original_issue_count"],
                cleaning_state["cleaned_score"],
                cleaning_state["cleaned_issue_count"],
                "simulation_cleaned",
            )

            st.markdown("**Remaining cleaned-data issues:**")
            if cleaning_state["cleaned_issues"]:
                st.dataframe(pd.DataFrame(cleaning_state["cleaned_issues"]), use_container_width=True)
            else:
                st.success("No issues remain after the selected conservative cleaning actions.")
        else:
            st.info("Run the simulation to compare original and cleaned dataset health.")

    with tab_drift:
        st.subheader("Dataset Drift Comparison")
        st.caption("Baseline dataset vs new dataset for ML observability and data reliability monitoring.")
        if comparison_file is None:
            st.info("Upload a new CSV in the header to activate drift comparison.")
        elif comparison_error:
            st.error(comparison_error)
        else:
            try:
                new_df = comparison_df
                threshold_col_1, threshold_col_2 = st.columns(2)
                with threshold_col_1:
                    missing_warning_threshold = st.slider(
                        "Missing drift warning threshold",
                        min_value=0.01,
                        max_value=0.50,
                        value=0.05,
                        step=0.01,
                        format="%.2f",
                    )
                with threshold_col_2:
                    missing_critical_threshold = st.slider(
                        "Missing drift critical threshold",
                        min_value=0.05,
                        max_value=0.80,
                        value=0.20,
                        step=0.01,
                        format="%.2f",
                    )
                if missing_critical_threshold <= missing_warning_threshold:
                    st.warning("Critical threshold should be greater than warning threshold. Using the warning threshold as the effective lower bound.")
                    missing_critical_threshold = missing_warning_threshold
                drift_summary = compare_datasets(
                    original_df,
                    new_df,
                    missing_warning_threshold=missing_warning_threshold,
                    missing_critical_threshold=missing_critical_threshold,
                )

                drift_col_1, drift_col_2, drift_col_3, drift_col_4 = st.columns(4)
                drift_col_1.metric("Drift Risk Score", f"{drift_summary['drift_score']}/100", drift_summary["risk_label"], delta_color="off")
                drift_col_2.metric("Critical drift", drift_summary["severity_summary"]["critical"])
                drift_col_3.metric("Warnings", drift_summary["severity_summary"]["warning"])
                drift_col_4.metric("Shared columns", drift_summary["shared_columns"])

                shape_col_1, shape_col_2 = st.columns(2)
                shape_col_1.metric("Baseline dataset", str(drift_summary["baseline_shape"]))
                shape_col_2.metric("New dataset", str(drift_summary["new_shape"]))

                st.subheader("AI Drift Explanation")
                compact_drift = _compact_drift_context(drift_summary)
                if use_ai and provider_available:
                    drift_explanation_cache_key = f"drift_inline:{provider_id}:{json.dumps(compact_drift, sort_keys=True)}"
                    local_drift_explanation = _ai_cache_get(drift_explanation_cache_key)
                    if local_drift_explanation is None:
                        local_drift_explanation = _ai_cache_set(
                            drift_explanation_cache_key,
                            ai_provider.generate_drift_explanation(compact_drift),
                        )
                else:
                    local_drift_explanation = None
                st.info(local_drift_explanation or drift_summary["explanation"])

                missing_drift_alerts = [
                    issue for issue in drift_summary["issues"] if issue["drift_type"] == "missing_value_drift"
                ]
                if missing_drift_alerts:
                    st.subheader("Missing Drift Alerts")
                    for issue in missing_drift_alerts:
                        alert_text = (
                            f"Column `{issue['column']}`: {issue['metric']} | "
                            f"Severity: {issue['severity']}"
                        )
                        if issue["severity"] == "critical":
                            st.error(alert_text)
                        else:
                            st.warning(alert_text)

                drift_overview_tab, drift_schema_tab, drift_tables_tab, drift_visuals_tab = st.tabs(
                    ["Drift issues", "Schema drift", "Column breakdown", "Visuals"]
                )

                with drift_overview_tab:
                    st.plotly_chart(drift_issue_severity_chart(drift_summary["issue_table"]), use_container_width=True)
                    if drift_summary["issue_table"].empty:
                        st.success("No material drift issues detected.")
                    else:
                        st.dataframe(drift_summary["issue_table"], use_container_width=True, hide_index=True)

                with drift_schema_tab:
                    schema_col_1, schema_col_2 = st.columns(2)
                    with schema_col_1:
                        st.markdown("**Added columns**")
                        st.write(drift_summary["added_columns"] or "None")
                    with schema_col_2:
                        st.markdown("**Removed columns**")
                        st.write(drift_summary["removed_columns"] or "None")
                    st.markdown("**Dtype changes**")
                    if drift_summary["dtype_changes"]:
                        st.dataframe(pd.DataFrame(drift_summary["dtype_changes"]), use_container_width=True, hide_index=True)
                    else:
                        st.write("None")

                with drift_tables_tab:
                    st.markdown("**Missing value drift**")
                    st.dataframe(drift_summary["missing_display_table"], use_container_width=True, hide_index=True)
                    with st.expander("Missing value drift detail"):
                        st.dataframe(
                            drift_summary["missing_table"][
                                [
                                    "column",
                                    "baseline_missing_count",
                                    "new_missing_count",
                                    "missing_count_delta",
                                    "baseline_missing_rate",
                                    "new_missing_rate",
                                    "change",
                                    "severity",
                                ]
                            ],
                            use_container_width=True,
                            hide_index=True,
                        )
                    st.markdown("**Numeric distribution drift**")
                    st.dataframe(drift_summary["numeric_table"], use_container_width=True, hide_index=True)
                    st.markdown("**Categorical distribution drift**")
                    st.dataframe(drift_summary["categorical_table"], use_container_width=True, hide_index=True)
                    st.markdown("**Cardinality drift**")
                    st.dataframe(drift_summary["cardinality_table"], use_container_width=True, hide_index=True)

                with drift_visuals_tab:
                    psi_col, missing_col = st.columns(2)
                    with psi_col:
                        st.subheader("PSI by Numeric Feature")
                        st.plotly_chart(psi_bar_chart(drift_summary["numeric_table"]), use_container_width=True)
                    with missing_col:
                        st.subheader("Missing Rate Drift")
                        st.plotly_chart(missing_drift_chart(drift_summary["missing_table"]), use_container_width=True)

                    drift_baseline_df = drift_summary["baseline_normalized"]
                    drift_new_df = drift_summary["new_normalized"]
                    numeric_shared = [
                        column
                        for column in sorted(set(drift_baseline_df.columns) & set(drift_new_df.columns))
                        if pd.api.types.is_numeric_dtype(drift_baseline_df[column]) and pd.api.types.is_numeric_dtype(drift_new_df[column])
                    ]
                    categorical_shared = [
                        column
                        for column in sorted(set(drift_baseline_df.columns) & set(drift_new_df.columns))
                        if column not in numeric_shared
                    ]
                    if numeric_shared:
                        selected_drift_numeric = st.selectbox("Numeric distribution overlay", numeric_shared)
                        st.plotly_chart(
                            numeric_distribution_overlay(drift_baseline_df, drift_new_df, selected_drift_numeric),
                            use_container_width=True,
                        )
                    if categorical_shared:
                        selected_drift_category = st.selectbox("Category frequency comparison", categorical_shared)
                        st.plotly_chart(
                            category_frequency_comparison(drift_baseline_df, drift_new_df, selected_drift_category),
                            use_container_width=True,
                        )
            except ValueError as exc:
                st.error(str(exc))

    with tab_history:
        st.subheader("Run History")
        st.caption("Local SQLite history stores run metadata and issue summaries only. Raw uploaded data is not stored.")
        recent_runs = load_recent_runs(limit=100)
        dataset_runs = load_runs_for_dataset(monitor_name, limit=50, schema_fingerprint=schema_fingerprint)

        if recent_runs.empty:
            st.info("No saved runs yet.")
        else:
            current_run_id = st.session_state.get("last_saved_run_id")
            st.write(f"Current saved run: `{current_run_id}`")
            st.dataframe(
                recent_runs[
                    [
                        "timestamp",
                        "dataset_name",
                        "health_score",
                        "issue_count",
                        "critical_count",
                        "warning_count",
                        "minor_count",
                        "provider_used",
                    ]
                ],
                use_container_width=True,
                hide_index=True,
            )

        if len(dataset_runs) >= 2:
            st.subheader(f"Trend for {monitor_name}")
            st.caption(f"Grouped by matching schema fingerprint: `{schema_fingerprint}`")
            trend_col_1, trend_col_2 = st.columns(2)
            with trend_col_1:
                st.plotly_chart(
                    px.line(
                        dataset_runs,
                        x="timestamp",
                        y="health_score",
                        markers=True,
                        labels={"timestamp": "Run time", "health_score": "Health score"},
                    ),
                    use_container_width=True,
                )
            with trend_col_2:
                st.plotly_chart(
                    px.line(
                        dataset_runs,
                        x="timestamp",
                        y="issue_count",
                        markers=True,
                        labels={"timestamp": "Run time", "issue_count": "Issue count"},
                    ),
                    use_container_width=True,
                )
            severity_trend = dataset_runs.melt(
                id_vars=["timestamp"],
                value_vars=["critical_count", "warning_count", "minor_count"],
                var_name="severity",
                value_name="count",
            )
            st.plotly_chart(
                px.line(
                    severity_trend,
                    x="timestamp",
                    y="count",
                    color="severity",
                    markers=True,
                    labels={"timestamp": "Run time", "count": "Issues"},
                ),
                use_container_width=True,
            )

            previous_runs = dataset_runs[dataset_runs["run_id"] != st.session_state.get("last_saved_run_id")]
            if not previous_runs.empty:
                previous = previous_runs.iloc[-1]
                st.subheader("Current vs Previous Run")
                compare_col_1, compare_col_2, compare_col_3 = st.columns(3)
                compare_col_1.metric(
                    "Health score",
                    health_score,
                    health_score - int(previous["health_score"]),
                )
                compare_col_2.metric(
                    "Issue count",
                    len(issues),
                    len(issues) - int(previous["issue_count"]),
                    delta_color="inverse",
                )
                compare_col_3.metric(
                    "Critical issues",
                    severity_counts["critical"],
                    severity_counts["critical"] - int(previous["critical_count"]),
                    delta_color="inverse",
                )
        else:
            st.info("Run this dataset more than once to see health and issue trends.")

    with tab_ai:
        st.subheader("AI Copilot")
        st.caption(
            "AI providers are used only after deterministic analysis is complete. They receive metadata, issue objects, "
            "summary statistics, and drift summaries, never full raw dataset rows."
        )
        context_json = json.dumps(dataset_context, sort_keys=True)
        deterministic_issues_json = json.dumps(deterministic_issues, sort_keys=True)
        drift_context = _compact_drift_context(drift_summary_for_ai)
        drift_json = json.dumps(drift_context, sort_keys=True)

        if use_ai and provider_available:
            st.success(f"AI provider active: {provider_choice}")
        else:
            st.warning("AI provider is unavailable or off. Showing rule-based fallback outputs.")

        qa_tab, plan_tab, code_tab, contract_tab, drift_ai_tab = st.tabs(
            ["Dataset Q&A", "Fix Plan", "Cleaning Code", "Data Contract", "Drift Explanation"]
        )

        with qa_tab:
            question = st.text_input(
                "Ask about the analysis",
                placeholder="Which columns are risky for ML training and why?",
            )
            if st.button("Ask AI", disabled=not question):
                if use_ai and provider_available:
                    cache_key = f"qa:{provider_id}:{question}:{deterministic_issues_json}:{health_score}:{context_json}:{drift_json}"
                    answer = _ai_cache_get(cache_key)
                    if answer is None:
                        answer = _ai_cache_set(
                            cache_key,
                            ai_provider.answer_dataset_question(
                                question,
                                deterministic_issues,
                                health_score,
                                dataset_context,
                                drift_context,
                            ),
                        )
                else:
                    answer = None
                st.markdown(answer or RuleBasedProvider().answer_dataset_question(question, issues, health_score, dataset_context, drift_context))

        with plan_tab:
            if not issues:
                st.success("No deterministic issues were detected, so no fix plan is needed.")
            elif st.button("Generate Fix Plan"):
                if use_ai and provider_available:
                    cache_key = f"fix_plan:{provider_id}:{deterministic_issues_json}:{health_score}:{context_json}"
                    plan = _ai_cache_get(cache_key)
                    if plan is None:
                        plan = _ai_cache_set(
                            cache_key,
                            ai_provider.generate_fix_plan(deterministic_issues, health_score, dataset_context),
                        )
                else:
                    plan = None
                st.markdown(plan or RuleBasedProvider().generate_fix_plan(issues, health_score, dataset_context))

        with code_tab:
            if issues:
                issue_labels = [
                    f"{index + 1}. {issue['display_name']} - {issue['column']} - {issue['severity']}"
                    for index, issue in enumerate(issues)
                ]
                selected_issue_label = st.selectbox("Choose an issue", issue_labels)
                selected_issue = issues[issue_labels.index(selected_issue_label)]
                st.markdown("AI-generated code is preview-only and is never executed automatically.")
                if st.button("Generate Cleaning Code Preview"):
                    if use_ai and provider_available:
                        selected_issue_json = json.dumps(selected_issue, sort_keys=True)
                        cache_key = f"code:{provider_id}:{selected_issue_json}:{context_json}"
                        code = _ai_cache_get(cache_key)
                        if code is None:
                            code = _ai_cache_set(
                                cache_key,
                                ai_provider.generate_cleaning_code(selected_issue, dataset_context),
                            )
                    else:
                        code = None
                    st.markdown(code or f"```python\n{selected_issue.get('example_cleaning_code', '# No code available.')}\n```")
            else:
                st.success("No detected issues need cleaning code.")

        with contract_tab:
            if st.button("Generate Data Contract Draft"):
                if use_ai and provider_available:
                    cache_key = f"contract:{provider_id}:{context_json}:{deterministic_issues_json}"
                    contract = _ai_cache_get(cache_key)
                    if contract is None:
                        contract = _ai_cache_set(
                            cache_key,
                            ai_provider.generate_data_contract(dataset_context, deterministic_issues),
                        )
                else:
                    contract = None
                st.markdown(contract or RuleBasedProvider().generate_data_contract(dataset_context, issues))

        with drift_ai_tab:
            if not drift_context:
                st.info("Upload both baseline and new datasets to generate a drift explanation.")
            elif st.button("Generate Drift Explanation"):
                if use_ai and provider_available:
                    cache_key = f"drift:{provider_id}:{drift_json}"
                    drift_ai_explanation = _ai_cache_get(cache_key)
                    if drift_ai_explanation is None:
                        drift_ai_explanation = _ai_cache_set(
                            cache_key,
                            ai_provider.generate_drift_explanation(drift_context),
                        )
                else:
                    drift_ai_explanation = None
                st.markdown(
                    drift_ai_explanation
                    or drift_context.get("rule_based_explanation")
                    or drift_context.get("risk_label", "No drift explanation available.")
                )

    with tab_methodology:
        st.subheader("Scoring Methodology")
        st.write(
            "The health score starts at 100. Critical issues deduct 15 points, warnings deduct 5 points, "
            "and minor concerns deduct 2 points. The final score is clamped between 0 and 100."
        )
        st.dataframe(pd.DataFrame(scoring_rules()), use_container_width=True, hide_index=True)

        st.subheader("Score Deduction Breakdown")
        breakdown = score_breakdown(issues)
        if breakdown:
            st.dataframe(pd.DataFrame(breakdown), use_container_width=True, hide_index=True)
        else:
            st.success("No deductions were applied.")

        st.subheader("Why Severity Matters")
        st.markdown(
            """
            - **Critical** issues can materially distort modeling, reporting, or downstream decisions.
            - **Warning** issues are likely to reduce generalization, interpretability, or operational trust.
            - **Minor** issues are lower-risk cleanup opportunities that still deserve review before production use.
            """
        )

    with tab_report:
        st.subheader("LLM-Ready Explanation")
        st.markdown(explanation)

        st.subheader("Data Contract Export")
        validation_contract = generate_validation_contract(df, roles=roles)
        contract_col_1, contract_col_2 = st.columns(2)
        with contract_col_1:
            st.download_button(
                "Download validation contract (YAML)",
                data=contract_to_yaml(validation_contract).encode("utf-8"),
                file_name="data_contract.yaml",
                mime="text/yaml",
            )
        with contract_col_2:
            st.download_button(
                "Download validation contract (JSON)",
                data=contract_to_json(validation_contract).encode("utf-8"),
                file_name="data_contract.json",
                mime="application/json",
            )
        with st.expander("Preview validation contract"):
            st.code(contract_to_yaml(validation_contract), language="yaml")

        st.subheader("Download Markdown Report")
        st.download_button(
            label="Download report",
            data=markdown_report.encode("utf-8"),
            file_name="data_debugger_report.md",
            mime="text/markdown",
        )
        st.markdown(markdown_report)


if __name__ == "__main__":
    main()
