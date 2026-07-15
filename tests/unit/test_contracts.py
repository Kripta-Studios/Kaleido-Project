from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest
from pydantic import ValidationError

from flowtwin.data.contracts import (
    CompletionStatus,
    OperationEvent,
    OperationOutcome,
    PlanRevision,
)


def test_operation_event_requires_aware_timestamps() -> None:
    with pytest.raises(ValidationError, match="timezone-aware"):
        OperationEvent(
            event_id="e1",
            source_system="fixture",
            source_record_id="1",
            event_type="start",
            event_time_utc=datetime(2025, 1, 1),
            ingested_at_utc=datetime(2025, 1, 1),
        )


def test_clock_skew_must_be_explicit() -> None:
    event_time = datetime(2025, 1, 1, tzinfo=UTC)
    with pytest.raises(ValidationError, match="clock_skew"):
        OperationEvent(
            event_id="e1",
            source_system="fixture",
            source_record_id="1",
            event_type="start",
            event_time_utc=event_time,
            ingested_at_utc=event_time - timedelta(seconds=1),
        )
    valid = OperationEvent(
        event_id="e1",
        source_system="fixture",
        source_record_id="1",
        event_type="start",
        event_time_utc=event_time,
        ingested_at_utc=event_time - timedelta(seconds=1),
        data_quality_flags=("clock_skew",),
    )
    assert valid.data_quality_flags == ("clock_skew",)


def test_censored_outcome_never_has_exact_completion() -> None:
    with pytest.raises(ValidationError, match="censored"):
        OperationOutcome(
            operation_id="op",
            completed_at_utc=datetime(2025, 1, 1, tzinfo=UTC),
            completion_status=CompletionStatus.INCOMPLETE,
            censored=True,
        )


def test_unknown_incident_status_stays_unknown() -> None:
    with pytest.raises(ValidationError, match="incident"):
        OperationOutcome(
            operation_id="op",
            completion_status=CompletionStatus.UNKNOWN,
            incident_types=("damage",),
            incident_status_known=False,
            censored=True,
        )


def test_plan_interval_cannot_run_backwards() -> None:
    start = datetime(2025, 1, 1, 12, tzinfo=UTC)
    with pytest.raises(ValidationError, match="precedes"):
        PlanRevision(
            plan_id="p",
            revision=0,
            valid_from_utc=start - timedelta(days=1),
            operation_id="op",
            milestone="complete",
            planned_start_utc=start,
            planned_end_utc=start - timedelta(minutes=1),
        )
