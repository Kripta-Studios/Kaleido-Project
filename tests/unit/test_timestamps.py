from __future__ import annotations

from datetime import UTC, datetime, timedelta
from zoneinfo import ZoneInfo

from flowtwin.data.contracts import OperationEvent, PlanRevision
from flowtwin.data.timestamps import plan_at_cutoff, validate_plan_history, validate_timestamps


def _event(event_id: str, time: datetime) -> OperationEvent:
    return OperationEvent(
        event_id=event_id,
        source_system="fixture",
        source_record_id=event_id,
        event_type="event",
        event_time_utc=time,
        ingested_at_utc=time,
        operation_id="op",
    )


def test_out_of_order_source_rows_are_reported() -> None:
    first = datetime(2025, 1, 1, 12, tzinfo=UTC)
    report = validate_timestamps([_event("later", first), _event("earlier", first - timedelta(1))])
    assert any(item.code == "out_of_order_source_row" for item in report.findings)
    assert report.passed


def test_dst_offsets_map_to_distinct_instants() -> None:
    madrid = ZoneInfo("Europe/Madrid")
    summer = datetime(2025, 7, 1, 12, tzinfo=madrid).astimezone(UTC)
    winter = datetime(2025, 1, 1, 12, tzinfo=madrid).astimezone(UTC)
    assert summer.hour == 10
    assert winter.hour == 11


def test_plan_cutoff_never_sees_future_revision() -> None:
    start = datetime(2025, 1, 1, tzinfo=UTC)
    revisions = [
        PlanRevision(
            plan_id="p",
            revision=0,
            valid_from_utc=start,
            operation_id="op",
            milestone="complete",
            planned_end_utc=start + timedelta(hours=8),
        ),
        PlanRevision(
            plan_id="p",
            revision=1,
            valid_from_utc=start + timedelta(hours=3),
            operation_id="op",
            milestone="complete",
            planned_end_utc=start + timedelta(hours=10),
        ),
    ]
    before_revision = plan_at_cutoff(revisions, "op", start + timedelta(hours=2))
    after_revision = plan_at_cutoff(revisions, "op", start + timedelta(hours=4))
    assert before_revision is not None
    assert after_revision is not None
    assert before_revision.revision == 0
    assert after_revision.revision == 1


def test_overwritten_plan_history_is_rejected() -> None:
    start = datetime(2025, 1, 1, tzinfo=UTC)
    revisions = [
        PlanRevision(
            plan_id="p",
            revision=0,
            valid_from_utc=start + timedelta(hours=2),
            operation_id="op",
            milestone="complete",
        ),
        PlanRevision(
            plan_id="p",
            revision=1,
            valid_from_utc=start,
            operation_id="op",
            milestone="complete",
        ),
    ]
    assert any(
        item.code == "plan_revision_time_rollback" for item in validate_plan_history(revisions)
    )
