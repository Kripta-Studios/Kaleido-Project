from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import polars as pl
from fastapi.testclient import TestClient

from flowtwin.data.adapters.trace_port import TracePortAdapter
from flowtwin.data.roles import ColumnRole
from flowtwin.serving.api import create_app


def test_customer_and_project_ids_are_identifiers_not_actions(tmp_path: Path) -> None:
    path = tmp_path / "events.csv"
    pl.DataFrame(
        [
            {
                "event_id": "e1",
                "event_type": "start",
                "event_time": "2025-01-01T00:00:00+00:00",
                "operation_id": "op1",
                "project_id": "p1",
                "customer_id": "c1",
            },
            {
                "event_id": "e2",
                "event_type": "end",
                "event_time": "2025-01-01T01:00:00+00:00",
                "operation_id": "op1",
                "project_id": "p1",
                "customer_id": "c1",
            },
        ]
    ).write_csv(path)
    roles = TracePortAdapter(path).classify_fields()
    assert roles.roles["project_id"] == ColumnRole.IDENTIFIER
    assert roles.roles["customer_id"] == ColumnRole.IDENTIFIER


def test_photo_after_cutoff_is_rejected_by_score_contract(tmp_path: Path) -> None:
    client = TestClient(create_app(tmp_path))
    cutoff = datetime(2025, 1, 1, 12, tzinfo=UTC)
    event = {
        "event_id": "photo-future",
        "source_system": "fixture",
        "source_record_id": "photo-future",
        "event_type": "photo_added",
        "event_time_utc": (cutoff + timedelta(minutes=1)).isoformat(),
        "ingested_at_utc": (cutoff + timedelta(minutes=2)).isoformat(),
        "operation_id": "op",
        "payload_ref": "redacted://photo",
    }
    response = client.post(
        "/v1/score/operation-prefix",
        json={
            "operation_id": "op",
            "events": [event],
            "prediction_time": cutoff.isoformat(),
        },
    )
    assert response.status_code == 422
    assert "after prediction_time" in response.text


def test_duplicate_retry_is_visible_not_silently_deleted(tmp_path: Path) -> None:
    path = tmp_path / "events.csv"
    pl.DataFrame(
        [
            {
                "event_id": "e1",
                "source_record_id": "retry-1",
                "event_type": "start",
                "event_time": "2025-01-01T00:00:00+00:00",
                "operation_id": "op",
            },
            {
                "event_id": "e2",
                "source_record_id": "retry-1",
                "event_type": "start",
                "event_time": "2025-01-01T00:00:01+00:00",
                "operation_id": "op",
            },
        ]
    ).write_csv(path)
    report = TracePortAdapter(path).validate_timestamps()
    assert any(item.code == "duplicate_source_record" for item in report.findings)
