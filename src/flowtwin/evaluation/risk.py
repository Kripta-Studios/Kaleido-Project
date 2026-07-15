from __future__ import annotations

from typing import Any, Literal

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    precision_score,
    recall_score,
)

from flowtwin.evaluation.calibration import reliability_report


def select_threshold_validation(
    target: np.ndarray,
    probability: np.ndarray,
    *,
    method: Literal["f1", "cost"] = "f1",
    false_alert_cost: float = 1.0,
    missed_event_cost: float = 5.0,
) -> dict[str, float | str]:
    target = np.asarray(target, dtype=int)
    probability = np.asarray(probability, dtype=float)
    candidates = np.unique(np.concatenate(([0.0], probability, [1.0])))
    best_threshold = 0.5
    best_value = -float("inf")
    for threshold in candidates:
        predicted = probability >= threshold
        true_positive = int(np.sum(predicted & (target == 1)))
        false_positive = int(np.sum(predicted & (target == 0)))
        false_negative = int(np.sum(~predicted & (target == 1)))
        if method == "cost":
            value = -(false_positive * false_alert_cost + false_negative * missed_event_cost)
        else:
            denominator = 2 * true_positive + false_positive + false_negative
            value = 2 * true_positive / denominator if denominator else 0.0
        if value > best_value:
            best_value = value
            best_threshold = float(threshold)
    return {
        "threshold": best_threshold,
        "selection_method": f"validation_{method}",
        "objective_value": float(best_value),
    }


def risk_metrics(
    target: np.ndarray,
    probability: np.ndarray,
    *,
    threshold: float,
    operation_ids: np.ndarray | None = None,
) -> dict[str, Any]:
    target = np.asarray(target, dtype=int)
    probability = np.clip(np.asarray(probability, dtype=float), 0, 1)
    predicted = probability >= threshold
    false_alerts = int(np.sum(predicted & (target == 0)))
    missed = int(np.sum(~predicted & (target == 1)))
    denominator = len(np.unique(operation_ids)) if operation_ids is not None else len(target)
    return {
        "auprc": float(average_precision_score(target, probability)),
        "precision": float(precision_score(target, predicted, zero_division=0)),
        "recall": float(recall_score(target, predicted, zero_division=0)),
        "brier": float(brier_score_loss(target, probability)),
        "false_alerts": false_alerts,
        "false_alerts_per_100_operations": 100 * false_alerts / max(denominator, 1),
        "missed_events": missed,
        "threshold": threshold,
        "calibration": reliability_report(target, probability),
    }
