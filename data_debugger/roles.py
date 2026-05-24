"""Dataset role helpers for ML-aware analysis."""

from __future__ import annotations

from typing import Any


def normalize_roles(roles: dict[str, Any] | None) -> dict[str, Any]:
    roles = roles or {}
    return {
        "target_column": roles.get("target_column") or "",
        "timestamp_column": roles.get("timestamp_column") or "",
        "entity_id_column": roles.get("entity_id_column") or "",
        "protected_columns": list(roles.get("protected_columns") or []),
        "exclude_ml_columns": list(roles.get("exclude_ml_columns") or []),
    }


def columns_excluded_from_ml_checks(roles: dict[str, Any] | None) -> set[str]:
    normalized = normalize_roles(roles)
    excluded = set(normalized["protected_columns"])
    excluded.update(normalized["exclude_ml_columns"])
    for role_key in ("timestamp_column", "entity_id_column"):
        if normalized[role_key]:
            excluded.add(normalized[role_key])
    return excluded
