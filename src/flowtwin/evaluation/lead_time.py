from __future__ import annotations

from datetime import datetime
from typing import Any


def lead_time_report(
    alerts: list[tuple[str, datetime]],
    outcomes: dict[str, datetime],
) -> dict[str, Any]:
    first_alert: dict[str, datetime] = {}
    for operation_id, timestamp in alerts:
        first_alert[operation_id] = min(first_alert.get(operation_id, timestamp), timestamp)
    lead_minutes = {
        operation_id: (outcome_time - first_alert[operation_id]).total_seconds() / 60
        for operation_id, outcome_time in outcomes.items()
        if operation_id in first_alert and first_alert[operation_id] <= outcome_time
    }
    values = sorted(lead_minutes.values())
    middle = values[len(values) // 2] if values else None
    return {
        "detected_operations": len(values),
        "outcome_operations": len(outcomes),
        "missed_operations": len(set(outcomes) - set(first_alert)),
        "median_lead_minutes": middle,
        "lead_minutes_by_operation": lead_minutes,
    }
