from __future__ import annotations

from collections import Counter
from collections.abc import Iterable
from typing import Any

from flowtwin.data.contracts import OperationEvent
from flowtwin.process.discovery import ordered_traces


def variant_report(events: Iterable[OperationEvent], top_n: int = 20) -> dict[str, Any]:
    traces = ordered_traces(events)
    variants = Counter(tuple(event.event_type for event in trace) for trace in traces.values())
    total = sum(variants.values())
    top = variants.most_common(top_n)
    covered = sum(count for _, count in top)
    return {
        "variant_count": len(variants),
        "operation_count": total,
        "top_variant_coverage": covered / total if total else 0.0,
        "variants": [
            {"activities": list(activities), "count": count, "share": count / total}
            for activities, count in top
        ],
    }
