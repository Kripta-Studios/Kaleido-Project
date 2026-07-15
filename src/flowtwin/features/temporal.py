from __future__ import annotations

import math
from datetime import datetime


def cyclical_time_features(value: datetime) -> dict[str, float]:
    hour_angle = 2 * math.pi * (value.hour + value.minute / 60) / 24
    weekday_angle = 2 * math.pi * value.weekday() / 7
    return {
        "hour_sin": math.sin(hour_angle),
        "hour_cos": math.cos(hour_angle),
        "weekday_sin": math.sin(weekday_angle),
        "weekday_cos": math.cos(weekday_angle),
    }


def minutes_between(start: datetime, end: datetime) -> float:
    if start.tzinfo is None or end.tzinfo is None:
        raise ValueError("temporal features require timezone-aware values")
    return (end - start).total_seconds() / 60
