from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Literal

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from flowtwin import __version__
from flowtwin.data.timestamps import plan_at_cutoff, validate_timestamps
from flowtwin.serving.model_registry import ModelRegistry
from flowtwin.serving.schemas import (
    EventBatchRequest,
    ModelCardResponse,
    PredictionInterval,
    ScenarioRequest,
    ScorePrefixRequest,
    ScoreResponse,
    ValidationResponse,
)


def _audit_id(payload: dict[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, default=str).encode()
    return hashlib.sha256(encoded).hexdigest()[:20]


def _demo_operations() -> list[dict[str, Any]]:
    now = datetime.now(UTC).replace(second=0, microsecond=0)
    definitions = [
        ("TP-2407", "MV Atlantic Cedar", "Breakbulk discharge", 0.78, 86, "critical", -32),
        ("TP-2411", "MV Boreal Wind", "Wind component loading", 0.54, 64, "watch", 18),
        ("TP-2418", "MV Senda", "Container handling", 0.31, 27, "stable", 42),
        ("TP-2420", "MV North Star", "Breakbulk loading", 0.18, 19, "stable", 64),
    ]
    rows = []
    for index, (operation_id, vessel, operation_type, progress, risk, status, delta) in enumerate(
        definitions
    ):
        rows.append(
            {
                "operation_id": operation_id,
                "vessel": vessel,
                "operation_type": operation_type,
                "berth": f"Berth {2 + index}",
                "shift": "Day A" if index < 3 else "Night B",
                "progress": progress,
                "risk": risk,
                "status": status,
                "plan_delta_minutes": delta,
                "last_event": (now - timedelta(minutes=7 + 9 * index)).isoformat(),
                "remaining_p50_minutes": [312, 426, 188, 520][index],
                "remaining_p90_minutes": [488, 650, 314, 780][index],
            }
        )
    return rows


def _demo_detail(operation_id: str) -> dict[str, Any]:
    operation = next(
        (row for row in _demo_operations() if row["operation_id"] == operation_id), None
    )
    if operation is None:
        raise KeyError(operation_id)
    now = datetime.now(UTC).replace(second=0, microsecond=0)
    labels = [
        ("Shift started", -390, "complete"),
        ("Cargo ready", -348, "complete"),
        ("Resource assigned", -302, "complete"),
        ("Handling started", -266, "complete"),
        ("Lift sequence 08", -96, "complete"),
        ("Weather hold", -44, "warning" if operation["risk"] >= 60 else "complete"),
        ("Current cutoff", 0, "current"),
        ("Operation complete P50", operation["remaining_p50_minutes"], "future"),
    ]
    return {
        **operation,
        "data_cutoff": now.isoformat(),
        "plan_revision": 1,
        "model_version": "demo-rule-v1",
        "claim_state": "smoke_only",
        "synthetic": True,
        "timeline": [
            {
                "label": label,
                "time": (now + timedelta(minutes=offset)).isoformat(),
                "state": state,
            }
            for label, offset, state in labels
        ],
        "reason_codes": (
            ["WAITING_TIME_ACCELERATION", "RESOURCE_UTILIZATION_HIGH", "PLAN_BUFFER_CONSUMED"]
            if operation["risk"] >= 60
            else ["PREFIX_WITHIN_REFERENCE_RANGE", "PLAN_BUFFER_AVAILABLE"]
        ),
        "risk_horizons": [
            {"hours": 2, "risk": max(5, operation["risk"] - 22)},
            {"hours": 4, "risk": max(8, operation["risk"] - 8)},
            {"hours": 8, "risk": operation["risk"]},
        ],
        "bottleneck": {
            "object": "Crane 02" if operation["risk"] >= 60 else "Cargo staging",
            "median_wait_minutes": 41 if operation["risk"] >= 60 else 18,
            "evidence": "synthetic shadow replay",
        },
    }


def create_app(artifact_root: Path | None = None) -> FastAPI:
    root = artifact_root or Path("outputs")
    registry = ModelRegistry(root)
    app = FastAPI(
        title="Kaleido FlowTwin",
        version=__version__,
        description="Read-only predictive operations MVP. No source-system write capability.",
    )
    static_dir = Path(__file__).parents[1] / "dashboard" / "static"
    app.mount("/static", StaticFiles(directory=static_dir), name="static")

    @app.get("/", include_in_schema=False)
    def index() -> FileResponse:
        return FileResponse(static_dir / "index.html")

    @app.get("/health")
    def health() -> dict[str, Any]:
        latest = registry.latest()
        return {
            "status": "ok",
            "read_only": True,
            "source_write_capability": False,
            "version": __version__,
            "model_loaded": latest is not None,
            "claim_state": latest.metrics.get("claim_state") if latest else "smoke_only",
        }

    @app.post("/v1/events/batch", response_model=ValidationResponse)
    def validate_batch(request: EventBatchRequest) -> ValidationResponse:
        report = validate_timestamps(request.events)
        digest = _audit_id({"events": [event.model_dump(mode="json") for event in request.events]})
        return ValidationResponse(
            accepted=report.passed,
            event_count=len(request.events),
            batch_sha256=digest,
            findings=[finding.model_dump(mode="json") for finding in report.findings],
        )

    @app.post("/v1/score/operation-prefix", response_model=ScoreResponse)
    def score_prefix(request: ScorePrefixRequest) -> ScoreResponse:
        ordered = sorted(request.events, key=lambda event: event.event_time_utc)
        first, last = ordered[0], ordered[-1]
        elapsed = max(0.0, (last.event_time_utc - first.event_time_utc).total_seconds() / 60)
        since_previous = (
            max(0.0, (last.event_time_utc - ordered[-2].event_time_utc).total_seconds() / 60)
            if len(ordered) > 1
            else 0.0
        )
        progress = request.estimated_progress or min(0.9, max(0.05, len(ordered) / 12))
        active_plan = plan_at_cutoff(
            request.plan_revisions, request.operation_id, request.prediction_time
        )
        registered = registry.latest()
        reason_codes: list[str] = []
        if registered is None:
            remaining_p50 = max(30.0, elapsed * (1 - progress) / max(progress, 0.05))
            remaining_p90 = remaining_p50 * 1.55
            risk = min(0.95, max(0.05, 0.18 + 0.72 * (1 - progress)))
            interval_lower = max(0.0, remaining_p50 * 0.62)
            interval_upper = remaining_p50 * 1.65
            model_version = "demo-rule-v1"
            claim_state = "smoke_only"
            reason_codes.append("NO_TRAINED_ARTIFACT_RULE_FALLBACK")
        else:
            features = registry.feature_row(
                activity=last.event_type,
                elapsed_minutes=elapsed,
                since_previous_minutes=since_previous,
                prefix_events=len(ordered),
                hour_utc=request.prediction_time.hour,
                weekday_utc=request.prediction_time.weekday() + 1,
            )
            remaining_p50 = float(max(0, registered.remaining_model.predict(features)[0]))
            risk = float(registered.risk_model.predict_proba(features)[0, 1])
            interval_lower_array, interval_upper_array = registered.conformal.interval(
                np.asarray([remaining_p50]), 0.9
            )
            interval_lower = float(interval_lower_array[0])
            interval_upper = float(interval_upper_array[0])
            remaining_p90 = max(remaining_p50, interval_upper)
            model_version = registered.version
            claim_state = str(registered.metrics["claim_state"])
            reason_codes.append("PUBLIC_WAREHOUSE_TRANSFER_NOT_KALEIDO_VALIDATED")
        if active_plan and active_plan.planned_end_utc:
            projected_end = request.prediction_time + timedelta(minutes=remaining_p50)
            buffer_minutes = (active_plan.planned_end_utc - projected_end).total_seconds() / 60
            reason_codes.append(
                "PROJECTED_AFTER_VISIBLE_PLAN" if buffer_minutes < 0 else "VISIBLE_PLAN_BUFFER"
            )
        if len(ordered) < 3:
            reason_codes.append("SPARSE_PREFIX")
        abstained = len(ordered) < 2
        confidence: Literal["low", "medium", "high"] = (
            "low" if abstained or registered is None else "medium"
        )
        audit_payload = {
            "operation_id": request.operation_id,
            "cutoff": request.prediction_time,
            "model_version": model_version,
            "source_event_ids": [event.event_id for event in ordered],
            "plan_revision": active_plan.revision if active_plan else None,
        }
        return ScoreResponse(
            model_version=model_version,
            claim_state=claim_state,
            data_cutoff=last.event_time_utc,
            plan_revision=active_plan.revision if active_plan else None,
            prediction_time=request.prediction_time,
            horizon_hours=request.horizon_hours,
            remaining_time_p50_minutes=remaining_p50,
            remaining_time_p90_minutes=remaining_p90,
            interval=PredictionInterval(
                lower_minutes=interval_lower,
                upper_minutes=interval_upper,
                nominal_coverage=0.9,
            ),
            deviation_risk=risk,
            confidence=confidence,
            abstained=abstained,
            reason_codes=reason_codes,
            source_event_ids=[event.event_id for event in ordered],
            audit_id=_audit_id(audit_payload),
        )

    @app.get("/v1/operations/{operation_id}/risk")
    def operation_risk(operation_id: str) -> dict[str, Any]:
        try:
            return _demo_detail(operation_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="operation not found") from exc

    @app.post("/v1/scenarios/rank")
    def scenarios(request: ScenarioRequest) -> dict[str, Any]:
        rows = [
            {
                "action": action,
                "rank": index + 1,
                "estimated_p50_delta_minutes": -max(5, 28 - index * 7),
                "support": "synthetic_scenario_only",
                "causal_claim": False,
            }
            for index, action in enumerate(request.approved_actions)
        ]
        return {
            "operation_id": request.operation_id,
            "scenarios": rows,
            "evidence_type": "simulation_not_realized_saving",
            "advisory_only": True,
        }

    @app.get("/v1/models/{version}/card", response_model=ModelCardResponse)
    def model_card(version: str) -> ModelCardResponse:
        registered = registry.latest()
        if registered is None or version not in {registered.version, "latest"}:
            return ModelCardResponse(
                model_version="demo-rule-v1",
                claim_state="smoke_only",
                dataset_id="synthetic_ui_fixture",
                split_protocol="none_demo_only",
                metrics={},
                limitations=[
                    "No fitted artifact is loaded",
                    "Synthetic UI values are not business evidence",
                ],
            )
        return ModelCardResponse(
            model_version=registered.version,
            claim_state=str(registered.metrics["claim_state"]),
            dataset_id=str(registered.metrics["dataset_id"]),
            split_protocol=str(registered.metrics["split_protocol"]),
            metrics=registered.metrics["selected_model_test"],
            limitations=[
                "Public non-port warehouse dataset",
                "Not Kaleido accuracy or value evidence",
                "Long-duration risk is a proxy, not plan deviation",
            ],
        )

    @app.get("/v1/demo/overview")
    def demo_overview() -> dict[str, Any]:
        operations = _demo_operations()
        return {
            "watermark": "SYNTHETIC SHADOW REPLAY · SMOKE_ONLY",
            "operations_active": len(operations),
            "operations_at_risk": sum(item["risk"] >= 60 for item in operations),
            "mean_plan_delta_minutes": round(
                sum(item["plan_delta_minutes"] for item in operations) / len(operations)
            ),
            "p90_coverage": None,
            "false_alerts_per_shift": None,
            "operations": operations,
        }

    @app.get("/v1/demo/operations/{operation_id}")
    def demo_operation(operation_id: str) -> dict[str, Any]:
        try:
            return _demo_detail(operation_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="operation not found") from exc

    return app
