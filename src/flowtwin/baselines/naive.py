from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class MedianRemainingTime:
    group_medians: dict[str, float] = field(default_factory=dict)
    global_median: float = 0.0

    def fit(self, remaining: np.ndarray, groups: np.ndarray | None = None) -> MedianRemainingTime:
        values = np.asarray(remaining, dtype=float)
        if values.size == 0:
            raise ValueError("cannot fit median baseline on an empty target")
        self.global_median = float(np.median(values))
        if groups is not None:
            group_values: dict[str, list[float]] = {}
            for group, value in zip(groups, values, strict=True):
                group_values.setdefault(str(group), []).append(float(value))
            self.group_medians = {
                key: float(np.median(items)) for key, items in group_values.items()
            }
        return self

    def predict(self, groups: np.ndarray | None, size: int | None = None) -> np.ndarray:
        if groups is None:
            if size is None:
                raise ValueError("size is required when groups are omitted")
            return np.full(size, self.global_median, dtype=float)
        result: np.ndarray = np.asarray(
            [self.group_medians.get(str(group), self.global_median) for group in groups],
            dtype=float,
        )
        return result


def persistence_risk_rule(
    elapsed_minutes: np.ndarray,
    planned_duration_minutes: np.ndarray,
    completion_fraction: np.ndarray,
) -> np.ndarray:
    elapsed = np.asarray(elapsed_minutes, dtype=float)
    planned = np.maximum(np.asarray(planned_duration_minutes, dtype=float), 1e-9)
    progress = np.clip(np.asarray(completion_fraction, dtype=float), 1e-3, 1)
    projected_total = elapsed / progress
    result: np.ndarray = np.clip((projected_total / planned - 0.8) / 0.4, 0, 1)
    return result
