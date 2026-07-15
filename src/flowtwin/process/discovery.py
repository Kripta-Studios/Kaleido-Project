from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Iterable
from itertools import pairwise
from typing import Any

from flowtwin.data.contracts import OperationEvent


def ordered_traces(events: Iterable[OperationEvent]) -> dict[str, list[OperationEvent]]:
    traces: dict[str, list[OperationEvent]] = defaultdict(list)
    for event in events:
        if event.operation_id is not None:
            traces[event.operation_id].append(event)
    for trace in traces.values():
        trace.sort(key=lambda item: (item.event_time_utc, item.event_id))
    return dict(traces)


def discover_process(events: Iterable[OperationEvent]) -> dict[str, Any]:
    traces = ordered_traces(events)
    activities: Counter[str] = Counter()
    directly_follows: Counter[tuple[str, str]] = Counter()
    starts: Counter[str] = Counter()
    ends: Counter[str] = Counter()
    for trace in traces.values():
        if not trace:
            continue
        starts[trace[0].event_type] += 1
        ends[trace[-1].event_type] += 1
        activities.update(event.event_type for event in trace)
        directly_follows.update(
            (left.event_type, right.event_type) for left, right in pairwise(trace)
        )
    return {
        "operations": len(traces),
        "events": sum(activities.values()),
        "activities": dict(activities.most_common()),
        "start_activities": dict(starts.most_common()),
        "end_activities": dict(ends.most_common()),
        "directly_follows": [
            {"source": source, "target": target, "count": count}
            for (source, target), count in directly_follows.most_common()
        ],
    }
