from __future__ import annotations

from typing import Any

import numpy as np


def regression_diagnostics(
    target: np.ndarray,
    prediction: np.ndarray,
    *,
    tolerances: tuple[float, ...],
) -> dict[str, Any]:
    target_values = np.asarray(target, dtype=float)
    prediction_values = np.maximum(0.0, np.asarray(prediction, dtype=float))
    if target_values.ndim != 1 or prediction_values.ndim != 1:
        raise ValueError("target and prediction must be one-dimensional")
    if target_values.size == 0 or target_values.size != prediction_values.size:
        raise ValueError("target and prediction must have the same non-zero length")
    if np.any(~np.isfinite(target_values)) or np.any(~np.isfinite(prediction_values)):
        raise ValueError("target and prediction must contain finite values")
    absolute_error = np.abs(target_values - prediction_values)
    return {
        "rows": int(target_values.size),
        "mae": float(np.mean(absolute_error)),
        "median_absolute_error": float(np.median(absolute_error)),
        "p90_absolute_error": float(np.quantile(absolute_error, 0.90)),
        "target_mean": float(np.mean(target_values)),
        "target_median": float(np.median(target_values)),
        "weighted_absolute_percentage_error": (
            float(absolute_error.sum() / target_values.sum())
            if float(target_values.sum()) > 0
            else None
        ),
        "within_tolerance": {
            f"within_{tolerance:g}": float(np.mean(absolute_error <= tolerance))
            for tolerance in tolerances
        },
    }


def grouped_bootstrap_mae(
    target: np.ndarray,
    prediction: np.ndarray,
    groups: np.ndarray,
    *,
    seed: int,
    samples: int = 500,
) -> dict[str, float]:
    target_values = np.asarray(target, dtype=float)
    prediction_values = np.asarray(prediction, dtype=float)
    group_values = np.asarray(groups).astype(str)
    unique = np.unique(group_values)
    if unique.size < 2:
        raise ValueError("grouped bootstrap requires at least two groups")
    indices = {group: np.flatnonzero(group_values == group) for group in unique}
    rng = np.random.default_rng(seed)
    estimates: list[float] = []
    for _ in range(samples):
        sampled = rng.choice(unique, size=unique.size, replace=True)
        rows = np.concatenate([indices[group] for group in sampled])
        estimates.append(float(np.mean(np.abs(target_values[rows] - prediction_values[rows]))))
    low, high = np.quantile(estimates, [0.025, 0.975])
    return {
        "estimate": float(np.mean(np.abs(target_values - prediction_values))),
        "bootstrap_95_low": float(low),
        "bootstrap_95_high": float(high),
        "bootstrap_samples": float(samples),
    }


def chronological_group_partitions(
    group_ids: np.ndarray,
    group_times: np.ndarray,
    *,
    train_fraction: float = 0.70,
    validation_fraction: float = 0.15,
) -> dict[str, str]:
    ids = np.asarray(group_ids).astype(str)
    times = np.asarray(group_times)
    if ids.size == 0 or ids.size != times.size:
        raise ValueError("group ids and times must have the same non-zero length")
    ordered = sorted(zip(ids.tolist(), times.tolist(), strict=True), key=lambda item: item[1])
    unique_ordered: list[str] = []
    seen: set[str] = set()
    for group, _ in ordered:
        if group not in seen:
            seen.add(group)
            unique_ordered.append(group)
    if len(unique_ordered) < 7:
        raise ValueError("chronological split requires at least seven groups")
    train_end = max(1, int(len(unique_ordered) * train_fraction))
    validation_end = max(
        train_end + 1,
        int(len(unique_ordered) * (train_fraction + validation_fraction)),
    )
    validation_end = min(validation_end, len(unique_ordered) - 1)
    partitions: dict[str, str] = {}
    for index, group in enumerate(unique_ordered):
        if index < train_end:
            partitions[group] = "train"
        elif index < validation_end:
            partitions[group] = "validation"
        else:
            partitions[group] = "test"
    return partitions
