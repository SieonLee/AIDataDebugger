"""Local SQLite storage for repeatable dataset reliability runs."""

from __future__ import annotations

import json
import hashlib
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


DEFAULT_HISTORY_DB = Path(".data_debugger") / "history.sqlite3"


def init_history_db(db_path: Path = DEFAULT_HISTORY_DB) -> None:
    """Create the local run history table if it does not exist."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS analysis_runs (
                run_id TEXT PRIMARY KEY,
                dataset_name TEXT NOT NULL,
                schema_fingerprint TEXT NOT NULL DEFAULT '',
                timestamp TEXT NOT NULL,
                row_count INTEGER NOT NULL,
                column_count INTEGER NOT NULL,
                health_score INTEGER NOT NULL,
                issue_count INTEGER NOT NULL,
                critical_count INTEGER NOT NULL,
                warning_count INTEGER NOT NULL,
                minor_count INTEGER NOT NULL,
                provider_used TEXT NOT NULL,
                config_summary TEXT NOT NULL,
                issue_summaries TEXT NOT NULL
            )
            """
        )
        _ensure_column(conn, "analysis_runs", "schema_fingerprint", "TEXT NOT NULL DEFAULT ''")
        conn.commit()


def save_analysis_run(
    dataset_name: str,
    row_count: int,
    column_count: int,
    health_score: int,
    issues: list[dict[str, Any]],
    severity_counts: dict[str, int],
    provider_used: str,
    config_summary: dict[str, Any],
    schema_fingerprint: str = "",
    db_path: Path = DEFAULT_HISTORY_DB,
) -> str:
    """Persist one analysis run without storing raw uploaded data."""
    init_history_db(db_path)
    run_id = str(uuid.uuid4())
    timestamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
    issue_summaries = [
        {
            "issue_type": issue.get("issue_type"),
            "display_name": issue.get("display_name"),
            "column": issue.get("column"),
            "severity": issue.get("severity"),
            "metric": issue.get("metric"),
        }
        for issue in issues
    ]

    with sqlite3.connect(db_path) as conn:
        conn.execute(
            """
            INSERT INTO analysis_runs (
                run_id, dataset_name, timestamp, row_count, column_count,
                health_score, issue_count, critical_count, warning_count, minor_count, schema_fingerprint,
                provider_used, config_summary, issue_summaries
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                run_id,
                dataset_name,
                timestamp,
                row_count,
                column_count,
                health_score,
                len(issues),
                int(severity_counts.get("critical", 0)),
                int(severity_counts.get("warning", 0)),
                int(severity_counts.get("minor", 0)),
                schema_fingerprint,
                provider_used,
                json.dumps(config_summary, sort_keys=True),
                json.dumps(issue_summaries, sort_keys=True),
            ),
        )
        conn.commit()
    return run_id


def load_recent_runs(limit: int = 100, db_path: Path = DEFAULT_HISTORY_DB) -> pd.DataFrame:
    """Load recent run metadata as a DataFrame."""
    init_history_db(db_path)
    with sqlite3.connect(db_path) as conn:
        return pd.read_sql_query(
            """
            SELECT
                run_id, dataset_name, timestamp, row_count, column_count,
                health_score, issue_count, critical_count, warning_count, minor_count,
                schema_fingerprint, provider_used, config_summary, issue_summaries
            FROM analysis_runs
            ORDER BY timestamp DESC
            LIMIT ?
            """,
            conn,
            params=(limit,),
        )


def load_runs_for_dataset(
    dataset_name: str,
    limit: int = 50,
    schema_fingerprint: str | None = None,
    db_path: Path = DEFAULT_HISTORY_DB,
) -> pd.DataFrame:
    """Load recent runs for one dataset name in chronological order."""
    init_history_db(db_path)
    with sqlite3.connect(db_path) as conn:
        where = "WHERE dataset_name = ?"
        params: tuple[Any, ...] = (dataset_name,)
        if schema_fingerprint:
            where += " AND schema_fingerprint = ?"
            params = (dataset_name, schema_fingerprint)
        runs = pd.read_sql_query(
            f"""
            SELECT
                run_id, dataset_name, timestamp, row_count, column_count,
                health_score, issue_count, critical_count, warning_count, minor_count,
                schema_fingerprint, provider_used, config_summary, issue_summaries
            FROM analysis_runs
            {where}
            ORDER BY timestamp DESC
            LIMIT ?
            """,
            conn,
            params=(*params, limit),
        )
    if not runs.empty:
        runs = runs.sort_values("timestamp")
    return runs


def schema_fingerprint_for_frame(df: pd.DataFrame) -> str:
    """Hash ordered column names for apples-to-apples monitor grouping.

    Dtype changes are still tracked as drift signals, but they should not split
    one production dataset monitor into separate history timelines.
    """
    schema = list(map(str, df.columns))
    return hashlib.sha256(json.dumps(schema, sort_keys=True).encode("utf-8")).hexdigest()[:16]


def _ensure_column(conn: sqlite3.Connection, table: str, column: str, definition: str) -> None:
    existing = {row[1] for row in conn.execute(f"PRAGMA table_info({table})").fetchall()}
    if column not in existing:
        conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {definition}")
