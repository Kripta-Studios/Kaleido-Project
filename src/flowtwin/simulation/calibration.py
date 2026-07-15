from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable
from statistics import median


def calibrate_task_durations(
    observed: Iterable[tuple[str, float]], min_observations: int = 5
) -> dict[str, dict[str, float | int]]:
    values: dict[str, list[float]] = defaultdict(list)
    for activity, duration_minutes in observed:
        if duration_minutes >= 0:
            values[activity].append(duration_minutes)
    result: dict[str, dict[str, float | int]] = {}
    for activity, durations in values.items():
        if len(durations) >= min_observations:
            ordered = sorted(durations)
            result[activity] = {
                "observations": len(durations),
                "median_minutes": median(durations),
                "p90_minutes": ordered[min(len(ordered) - 1, int(0.9 * len(ordered)))],
            }
    return result
