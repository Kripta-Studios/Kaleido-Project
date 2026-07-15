from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable
from itertools import pairwise
from statistics import median
from typing import Any

from flowtwin.data.contracts import OperationEvent
from flowtwin.process.discovery import ordered_traces


def bottleneck_report(events: Iterable[OperationEvent]) -> dict[str, Any]:
    waits: dict[tuple[str, str], list[float]] = defaultdict(list)
    rework: dict[str, int] = defaultdict(int)
    traces = ordered_traces(events)
    for trace in traces.values():
        seen: set[str] = set()
        for left, right in pairwise(trace):
            minutes = max(0.0, (right.event_time_utc - left.event_time_utc).total_seconds() / 60)
            waits[(left.event_type, right.event_type)].append(minutes)
        for event in trace:
            if event.event_type in seen:
                rework[event.event_type] += 1
            seen.add(event.event_type)
    rows = [
        {
            "source": edge[0],
            "target": edge[1],
            "observations": len(values),
            "median_wait_minutes": median(values),
            "total_wait_minutes": sum(values),
        }
        for edge, values in waits.items()
    ]
    rows.sort(key=lambda row: row["total_wait_minutes"], reverse=True)
    return {
        "bottlenecks": rows,
        "rework_repetitions": dict(sorted(rework.items(), key=lambda item: item[1], reverse=True)),
        "limitations": [
            "inter-event duration combines work and waiting unless activity semantics "
            "distinguish them"
        ],
    }
