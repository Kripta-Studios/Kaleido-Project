from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable
from datetime import datetime

from pydantic import BaseModel, Field

from flowtwin.data.contracts import OperationEvent, PlanRevision


class TimestampFinding(BaseModel):
    code: str
    severity: str
    message: str
    event_ids: list[str] = Field(default_factory=list)


class TimestampReport(BaseModel):
    passed: bool
    findings: list[TimestampFinding]


def validate_timestamps(events: Iterable[OperationEvent]) -> TimestampReport:
    findings: list[TimestampFinding] = []
    seen_ids: set[str] = set()
    last_by_operation: dict[str, datetime] = {}
    source_ids: dict[tuple[str, str], list[str]] = defaultdict(list)

    for event in events:
        if event.event_id in seen_ids:
            findings.append(
                TimestampFinding(
                    code="duplicate_event_id",
                    severity="error",
                    message=f"duplicate canonical event id: {event.event_id}",
                    event_ids=[event.event_id],
                )
            )
        seen_ids.add(event.event_id)
        source_ids[(event.source_system, event.source_record_id)].append(event.event_id)
        if event.operation_id:
            previous = last_by_operation.get(event.operation_id)
            if previous is not None and event.event_time_utc < previous:
                findings.append(
                    TimestampFinding(
                        code="out_of_order_source_row",
                        severity="warning",
                        message=f"operation {event.operation_id} is out of temporal order",
                        event_ids=[event.event_id],
                    )
                )
            last_by_operation[event.operation_id] = max(
                previous or event.event_time_utc, event.event_time_utc
            )

    for source_key, event_ids in source_ids.items():
        if len(event_ids) > 1:
            findings.append(
                TimestampFinding(
                    code="duplicate_source_record",
                    severity="warning",
                    message=f"source record {source_key} appears {len(event_ids)} times",
                    event_ids=event_ids,
                )
            )
    return TimestampReport(
        passed=not any(finding.severity == "error" for finding in findings),
        findings=findings,
    )


def plan_at_cutoff(
    revisions: Iterable[PlanRevision], operation_id: str, cutoff: datetime
) -> PlanRevision | None:
    if cutoff.tzinfo is None or cutoff.utcoffset() is None:
        raise ValueError("cutoff must be timezone-aware")
    eligible = [
        revision
        for revision in revisions
        if revision.operation_id == operation_id and revision.valid_from_utc <= cutoff
    ]
    if not eligible:
        return None
    return max(eligible, key=lambda revision: (revision.valid_from_utc, revision.revision))


def validate_plan_history(revisions: Iterable[PlanRevision]) -> list[TimestampFinding]:
    by_plan: dict[str, list[PlanRevision]] = defaultdict(list)
    for revision in revisions:
        by_plan[revision.plan_id].append(revision)
    findings: list[TimestampFinding] = []
    for plan_id, history in by_plan.items():
        ordered = sorted(history, key=lambda item: item.revision)
        revision_numbers = [item.revision for item in ordered]
        if len(revision_numbers) != len(set(revision_numbers)):
            findings.append(
                TimestampFinding(
                    code="duplicate_plan_revision",
                    severity="error",
                    message=f"plan {plan_id} contains duplicate revision numbers",
                )
            )
        valid_times = [item.valid_from_utc for item in ordered]
        if valid_times != sorted(valid_times):
            findings.append(
                TimestampFinding(
                    code="plan_revision_time_rollback",
                    severity="error",
                    message=f"plan {plan_id} revisions are not immutable chronological events",
                )
            )
    return findings
