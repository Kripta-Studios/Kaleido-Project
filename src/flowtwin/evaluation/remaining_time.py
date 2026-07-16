from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import mean_absolute_error, median_absolute_error


def pinball_loss(target: np.ndarray, prediction: np.ndarray, quantile: float) -> float:
    error = np.asarray(target, dtype=float) - np.asarray(prediction, dtype=float)
    return float(np.mean(np.maximum(quantile * error, (quantile - 1) * error)))


def interval_coverage(
    target: np.ndarray, lower: np.ndarray, upper: np.ndarray
) -> tuple[float, float]:
    target = np.asarray(target, dtype=float)
    lower = np.asarray(lower, dtype=float)
    upper = np.asarray(upper, dtype=float)
    if np.any(upper < lower):
        raise ValueError("interval upper bound is below lower bound")
    coverage = float(np.mean((target >= lower) & (target <= upper)))
    width = float(np.mean(upper - lower))
    return coverage, width


def remaining_time_metrics(
    target: np.ndarray,
    p50: np.ndarray,
    p90: np.ndarray | None = None,
    interval50: tuple[np.ndarray, np.ndarray] | None = None,
    interval90: tuple[np.ndarray, np.ndarray] | None = None,
) -> dict[str, float]:
    target = np.asarray(target, dtype=float)
    p50 = np.maximum(0.0, np.asarray(p50, dtype=float))
    metrics = {
        "mae_minutes": float(mean_absolute_error(target, p50)),
        "median_ae_minutes": float(median_absolute_error(target, p50)),
        "pinball_p50_minutes": pinball_loss(target, p50, 0.5),
    }
    if p90 is not None:
        p90_values = np.maximum(p50, np.asarray(p90, dtype=float))
        metrics["pinball_p90_minutes"] = pinball_loss(target, p90_values, 0.9)
        metrics["p90_quantile_coverage"] = float(np.mean(target <= p90_values))
        metrics["p50_to_p90_width_minutes"] = float(np.mean(p90_values - p50))
    if interval50 is not None:
        coverage, width = interval_coverage(target, *interval50)
        metrics["p50_interval_coverage"] = coverage
        metrics["p50_interval_width_minutes"] = width
    if interval90 is not None:
        coverage, width = interval_coverage(target, *interval90)
        metrics["p90_interval_coverage"] = coverage
        metrics["p90_interval_width_minutes"] = width
    return metrics


def grouped_remaining_time_metrics(
    target: np.ndarray,
    prediction: np.ndarray,
    groups: np.ndarray,
    min_rows: int = 10,
) -> dict[str, Any]:
    target = np.asarray(target, dtype=float)
    prediction = np.asarray(prediction, dtype=float)
    groups = np.asarray(groups)
    rows: dict[str, dict[str, float]] = {}
    for group in sorted(set(map(str, groups))):
        mask = groups.astype(str) == group
        if int(mask.sum()) < min_rows:
            continue
        rows[group] = remaining_time_metrics(target[mask], prediction[mask])
    worst = max(rows, key=lambda key: rows[key]["mae_minutes"]) if rows else None
    return {"groups": rows, "worst_group": worst}


def cluster_bootstrap_mae(
    target: np.ndarray,
    prediction: np.ndarray,
    operation_ids: np.ndarray,
    *,
    samples: int = 500,
    seed: int = 42,
) -> dict[str, float]:
    target = np.asarray(target, dtype=float)
    prediction = np.asarray(prediction, dtype=float)
    operation_ids = np.asarray(operation_ids)
    unique = np.unique(operation_ids)
    if unique.size < 2:
        raise ValueError("cluster bootstrap requires at least two operations")
    indices_by_operation = {key: np.flatnonzero(operation_ids == key) for key in unique}
    rng = np.random.default_rng(seed)
    estimates: list[float] = []
    for _ in range(samples):
        sampled = rng.choice(unique, size=unique.size, replace=True)
        indices = np.concatenate([indices_by_operation[key] for key in sampled])
        estimates.append(float(mean_absolute_error(target[indices], prediction[indices])))
    low, high = np.quantile(estimates, [0.025, 0.975])
    return {
        "estimate": float(mean_absolute_error(target, prediction)),
        "bootstrap_95_low": float(low),
        "bootstrap_95_high": float(high),
        "bootstrap_samples": float(samples),
    }
