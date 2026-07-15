from __future__ import annotations

from itertools import pairwise
from typing import Any

import numpy as np


def expected_calibration_error(
    target: np.ndarray, probability: np.ndarray, bins: int = 10
) -> tuple[float, list[dict[str, Any]]]:
    target = np.asarray(target, dtype=float)
    probability = np.clip(np.asarray(probability, dtype=float), 0, 1)
    edges = np.linspace(0, 1, bins + 1)
    rows: list[dict[str, Any]] = []
    ece = 0.0
    for index, (left, right) in enumerate(pairwise(edges)):
        mask = (probability >= left) & (
            probability <= right if index == bins - 1 else probability < right
        )
        count = int(mask.sum())
        if not count:
            continue
        confidence = float(probability[mask].mean())
        frequency = float(target[mask].mean())
        ece += count / len(target) * abs(confidence - frequency)
        rows.append(
            {
                "bin_left": float(left),
                "bin_right": float(right),
                "count": count,
                "mean_probability": confidence,
                "event_frequency": frequency,
            }
        )
    return float(ece), rows


def reliability_report(
    target: np.ndarray, probability: np.ndarray, bins: int = 10
) -> dict[str, Any]:
    ece, rows = expected_calibration_error(target, probability, bins=bins)
    return {"ece": ece, "bins": rows}
