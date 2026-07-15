from __future__ import annotations

from collections.abc import Iterable, Sequence
from itertools import pairwise
from typing import Any

from flowtwin.data.contracts import OperationEvent
from flowtwin.process.discovery import ordered_traces


def conformance_report(
    events: Iterable[OperationEvent],
    allowed_edges: Iterable[tuple[str, str]],
    required_milestones: Sequence[str] = (),
) -> dict[str, Any]:
    allowed = set(allowed_edges)
    traces = ordered_traces(events)
    deviations: list[dict[str, Any]] = []
    conforming = 0
    for operation_id, trace in traces.items():
        observed = [event.event_type for event in trace]
        illegal = [
            [left, right] for left, right in pairwise(observed) if (left, right) not in allowed
        ]
        missing = [milestone for milestone in required_milestones if milestone not in observed]
        if illegal or missing:
            deviations.append(
                {
                    "operation_id": operation_id,
                    "illegal_transitions": illegal,
                    "missing_milestones": missing,
                }
            )
        else:
            conforming += 1
    total = len(traces)
    return {
        "operations": total,
        "conforming_operations": conforming,
        "fitness": conforming / total if total else 0.0,
        "deviations": deviations,
    }
