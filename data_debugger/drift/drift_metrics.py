"""Core drift metrics for baseline-vs-new dataset comparison."""

from __future__ import annotations

import numpy as np
import pandas as pd


EPSILON = 1e-6


def population_stability_index(
    baseline: pd.Series,
    current: pd.Series,
    bins: int = 10,
) -> float:
    """Calculate PSI for numeric distributions using baseline quantile bins."""
    base = pd.to_numeric(baseline, errors="coerce").dropna()
    new = pd.to_numeric(current, errors="coerce").dropna()
    if base.empty or new.empty:
        return 0.0

    quantiles = np.linspace(0, 1, bins + 1)
    edges = np.unique(base.quantile(quantiles).to_numpy())
    if len(edges) < 3:
        min_value = min(float(base.min()), float(new.min()))
        max_value = max(float(base.max()), float(new.max()))
        if min_value == max_value:
            return 0.0
        edges = np.linspace(min_value, max_value, bins + 1)

    edges[0] = -np.inf
    edges[-1] = np.inf
    base_counts = pd.cut(base, bins=edges, include_lowest=True).value_counts(sort=False)
    new_counts = pd.cut(new, bins=edges, include_lowest=True).value_counts(sort=False)

    base_pct = (base_counts / max(1, base_counts.sum())).clip(lower=EPSILON)
    new_pct = (new_counts / max(1, new_counts.sum())).clip(lower=EPSILON)
    psi = ((new_pct - base_pct) * np.log(new_pct / base_pct)).sum()
    return round(float(psi), 4)


def categorical_distribution_shift(
    baseline: pd.Series,
    current: pd.Series,
) -> dict:
    """Compare categorical frequency distributions and dominant labels."""
    base = baseline.dropna().astype(str)
    new = current.dropna().astype(str)
    base_freq = base.value_counts(normalize=True)
    new_freq = new.value_counts(normalize=True)
    all_categories = sorted(set(base_freq.index) | set(new_freq.index))

    max_frequency_change = 0.0
    for category in all_categories:
        change = abs(float(new_freq.get(category, 0.0) - base_freq.get(category, 0.0)))
        max_frequency_change = max(max_frequency_change, change)

    base_dominant = str(base_freq.index[0]) if not base_freq.empty else None
    new_dominant = str(new_freq.index[0]) if not new_freq.empty else None

    return {
        "new_categories": sorted(set(new_freq.index) - set(base_freq.index)),
        "removed_categories": sorted(set(base_freq.index) - set(new_freq.index)),
        "baseline_dominant": base_dominant,
        "new_dominant": new_dominant,
        "baseline_dominant_share": round(float(base_freq.iloc[0]), 4) if not base_freq.empty else 0.0,
        "new_dominant_share": round(float(new_freq.iloc[0]), 4) if not new_freq.empty else 0.0,
        "max_frequency_change": round(max_frequency_change, 4),
    }


def numeric_stats(series: pd.Series) -> dict:
    numeric = pd.to_numeric(series, errors="coerce").dropna()
    if numeric.empty:
        return {
            "mean": None,
            "std": None,
            "min": None,
            "p25": None,
            "median": None,
            "p75": None,
            "max": None,
        }
    return {
        "mean": round(float(numeric.mean()), 4),
        "std": round(float(numeric.std(ddof=0)), 4),
        "min": round(float(numeric.min()), 4),
        "p25": round(float(numeric.quantile(0.25)), 4),
        "median": round(float(numeric.median()), 4),
        "p75": round(float(numeric.quantile(0.75)), 4),
        "max": round(float(numeric.max()), 4),
    }
