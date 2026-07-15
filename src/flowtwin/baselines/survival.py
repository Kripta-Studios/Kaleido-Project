from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class KaplanMeierRemainingTime:
    event_times_: np.ndarray | None = None
    survival_: np.ndarray | None = None

    def fit(self, durations: np.ndarray, observed: np.ndarray) -> KaplanMeierRemainingTime:
        durations = np.asarray(durations, dtype=float)
        observed = np.asarray(observed, dtype=bool)
        if durations.size == 0 or durations.shape != observed.shape:
            raise ValueError("durations and observed must be non-empty with equal shape")
        unique_times = np.unique(durations)
        survival_values: list[float] = []
        survival = 1.0
        for time in unique_times:
            at_risk = int(np.sum(durations >= time))
            events = int(np.sum((durations == time) & observed))
            if at_risk:
                survival *= 1 - events / at_risk
            survival_values.append(survival)
        self.event_times_ = unique_times
        self.survival_ = np.asarray(survival_values)
        return self

    def _check(self) -> tuple[np.ndarray, np.ndarray]:
        if self.event_times_ is None or self.survival_ is None:
            raise RuntimeError("KaplanMeierRemainingTime is not fitted")
        return self.event_times_, self.survival_

    def survival_at(self, times: np.ndarray) -> np.ndarray:
        event_times, survival = self._check()
        query = np.asarray(times, dtype=float)
        indices = np.searchsorted(event_times, query, side="right") - 1
        return np.where(indices >= 0, survival[np.maximum(indices, 0)], 1.0)

    def remaining_quantile(self, elapsed: np.ndarray, quantile: float = 0.5) -> np.ndarray:
        if not 0 < quantile < 1:
            raise ValueError("quantile must be in (0, 1)")
        event_times, survival = self._check()
        elapsed_values = np.asarray(elapsed, dtype=float)
        predictions: list[float] = []
        for start in elapsed_values:
            survival_start = float(self.survival_at(np.asarray([start]))[0])
            if survival_start <= 0:
                predictions.append(0.0)
                continue
            threshold = survival_start * (1 - quantile)
            candidates = event_times[(event_times >= start) & (survival <= threshold)]
            total_time = float(candidates[0]) if candidates.size else float(event_times[-1])
            predictions.append(max(0.0, total_time - start))
        return np.asarray(predictions)

    def predict(self, elapsed: np.ndarray) -> np.ndarray:
        return self.remaining_quantile(elapsed, 0.5)
