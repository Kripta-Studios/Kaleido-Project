from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

from fastapi.testclient import TestClient

from flowtwin.data.contracts import OperationEvent, PlanRevision
from flowtwin.serving.api import create_app


def _event(event_id: str, minute: int) -> OperationEvent:
    time = datetime(2025, 1, 1, tzinfo=UTC) + timedelta(minutes=minute)
    return OperationEvent(
        event_id=event_id,
        source_system="fixture",
        source_record_id=event_id,
        event_type="handling" if minute else "start",
        event_time_utc=time,
        ingested_at_utc=time,
        operation_id="op-1",
    )


def test_health_and_demo_ui_are_read_only(tmp_path: Path) -> None:
    client = TestClient(create_app(tmp_path))
    health = client.get("/health")
    assert health.status_code == 200
    assert health.json()["source_write_capability"] is False
    page = client.get("/")
    assert page.status_code == 200
    assert "Synthetic shadow replay" in page.text


def test_batch_validation_does_not_persist(tmp_path: Path) -> None:
    client = TestClient(create_app(tmp_path))
    response = client.post(
        "/v1/events/batch",
        json={"events": [_event("e1", 0).model_dump(mode="json")]},
    )
    assert response.status_code == 200
    assert response.json()["persisted"] is False


def test_score_names_cutoff_plan_revision_and_sources(tmp_path: Path) -> None:
    client = TestClient(create_app(tmp_path))
    start = datetime(2025, 1, 1, tzinfo=UTC)
    plans = [
        PlanRevision(
            plan_id="p",
            revision=0,
            valid_from_utc=start - timedelta(days=1),
            operation_id="op-1",
            milestone="complete",
            planned_end_utc=start + timedelta(hours=6),
        ),
        PlanRevision(
            plan_id="p",
            revision=1,
            valid_from_utc=start + timedelta(hours=2),
            operation_id="op-1",
            milestone="complete",
            planned_end_utc=start + timedelta(hours=8),
        ),
    ]
    response = client.post(
        "/v1/score/operation-prefix",
        json={
            "operation_id": "op-1",
            "events": [
                _event("e1", 0).model_dump(mode="json"),
                _event("e2", 60).model_dump(mode="json"),
                _event("e3", 90).model_dump(mode="json"),
            ],
            "plan_revisions": [plan.model_dump(mode="json") for plan in plans],
            "prediction_time": (start + timedelta(minutes=100)).isoformat(),
            "estimated_progress": 0.4,
            "horizon_hours": 8,
        },
    )
    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["plan_revision"] == 0
    assert payload["source_event_ids"] == ["e1", "e2", "e3"]
    assert payload["writes_to_source_system"] is False
    assert payload["claim_state"] == "smoke_only"


def test_demo_operation_is_watermarked(tmp_path: Path) -> None:
    client = TestClient(create_app(tmp_path))
    overview = client.get("/v1/demo/overview").json()
    assert "SYNTHETIC" in overview["watermark"]
    detail = client.get("/v1/demo/operations/TP-2407").json()
    assert detail["synthetic"] is True
    assert detail["claim_state"] == "smoke_only"
