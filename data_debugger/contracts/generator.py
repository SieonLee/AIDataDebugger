"""Validation contract generation for dataset reliability workflows."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any

import pandas as pd


def generate_data_contract(df: pd.DataFrame, roles: dict[str, Any] | None = None) -> dict[str, Any]:
    """Generate a deterministic validation contract without sending raw rows anywhere."""
    roles = roles or {}
    contract: dict[str, Any] = {
        "contract_version": "0.1",
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "required_columns": list(map(str, df.columns)),
        "roles": {
            "target_column": roles.get("target_column") or None,
            "timestamp_column": roles.get("timestamp_column") or None,
            "entity_id_column": roles.get("entity_id_column") or None,
            "protected_columns": roles.get("protected_columns", []),
            "excluded_from_ml_checks": roles.get("exclude_ml_columns", []),
        },
        "columns": {},
    }
    for column in df.columns:
        series = df[column]
        column_rules: dict[str, Any] = {
            "dtype": str(series.dtype),
            "max_missing_rate": 0.05,
            "nullable": bool(series.isna().any()),
            "is_inferred": True,
        }
        non_null = series.dropna()
        if pd.api.types.is_numeric_dtype(series) and not non_null.empty:
            column_rules["min_value"] = _json_safe(non_null.min())
            column_rules["max_value"] = _json_safe(non_null.max())
        unique_count = int(non_null.nunique(dropna=True))
        if 0 < unique_count <= 20:
            column_rules["allowed_values"] = [_json_safe(value) for value in sorted(non_null.unique(), key=str)]
        if roles.get("entity_id_column") == column:
            column_rules["unique"] = True
        contract["columns"][str(column)] = column_rules
    return contract


def contract_to_json(contract: dict[str, Any]) -> str:
    return json.dumps(contract, indent=2, ensure_ascii=False)


def contract_to_yaml(contract: dict[str, Any]) -> str:
    """Render a small YAML subset without requiring PyYAML."""
    return _to_yaml(contract)


def _to_yaml(value: Any, indent: int = 0) -> str:
    space = "  " * indent
    if isinstance(value, dict):
        lines = []
        for key, item in value.items():
            if isinstance(item, (dict, list)):
                lines.append(f"{space}{key}:")
                lines.append(_to_yaml(item, indent + 1))
            else:
                lines.append(f"{space}{key}: {_yaml_scalar(item)}")
        return "\n".join(lines)
    if isinstance(value, list):
        if not value:
            return f"{space}[]"
        lines = []
        for item in value:
            if isinstance(item, (dict, list)):
                lines.append(f"{space}-")
                lines.append(_to_yaml(item, indent + 1))
            else:
                lines.append(f"{space}- {_yaml_scalar(item)}")
        return "\n".join(lines)
    return f"{space}{_yaml_scalar(value)}"


def _yaml_scalar(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value)
    text = str(value).replace('"', '\\"')
    return f'"{text}"'


def _json_safe(value: Any) -> Any:
    if hasattr(value, "item"):
        return value.item()
    return value
