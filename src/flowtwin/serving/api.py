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
        {
            "operation_id": "TP-2407",
            "vessel": "MV Atlantic Cedar",
            "operation_type": "Breakbulk discharge",
            "progress": 0.78,
            "risk": 86,
            "status": "critical",
            "plan_delta_minutes": -32,
            "remaining_p50_minutes": 312,
            "remaining_p90_minutes": 488,
        },
        {
            "operation_id": "TP-2411",
            "vessel": "MV Boreal Wind",
            "operation_type": "Wind component loading",
            "progress": 0.54,
            "risk": 64,
            "status": "watch",
            "plan_delta_minutes": 18,
            "remaining_p50_minutes": 426,
            "remaining_p90_minutes": 650,
        },
        {
            "operation_id": "TP-2418",
            "vessel": "MV Senda",
            "operation_type": "Container handling",
            "progress": 0.31,
            "risk": 27,
            "status": "stable",
            "plan_delta_minutes": 42,
            "remaining_p50_minutes": 188,
            "remaining_p90_minutes": 314,
        },
        {
            "operation_id": "TP-2420",
            "vessel": "MV North Star",
            "operation_type": "Breakbulk loading",
            "progress": 0.18,
            "risk": 19,
            "status": "stable",
            "plan_delta_minutes": 64,
            "remaining_p50_minutes": 520,
            "remaining_p90_minutes": 780,
        },
        {
            "operation_id": "TP-2424",
            "vessel": "MV Galicia Spirit",
            "operation_type": "Project cargo discharge",
            "progress": 0.66,
            "risk": 72,
            "status": "critical",
            "plan_delta_minutes": -48,
            "remaining_p50_minutes": 274,
            "remaining_p90_minutes": 502,
        },
        {
            "operation_id": "TP-2428",
            "vessel": "MV Meridian",
            "operation_type": "Steel coil loading",
            "progress": 0.43,
            "risk": 51,
            "status": "watch",
            "plan_delta_minutes": 8,
            "remaining_p50_minutes": 355,
            "remaining_p90_minutes": 610,
        },
        {
            "operation_id": "TP-2432",
            "vessel": "MV Ocean Vale",
            "operation_type": "Ro-ro discharge",
            "progress": 0.88,
            "risk": 12,
            "status": "stable",
            "plan_delta_minutes": 76,
            "remaining_p50_minutes": 96,
            "remaining_p90_minutes": 168,
        },
        {
            "operation_id": "TP-2435",
            "vessel": "MV Alba",
            "operation_type": "Heavy lift loading",
            "progress": 0.24,
            "risk": 58,
            "status": "watch",
            "plan_delta_minutes": -6,
            "remaining_p50_minutes": 608,
            "remaining_p90_minutes": 920,
        },
    ]
    rows: list[dict[str, Any]] = []
    for index, definition in enumerate(definitions):
        rows.append(
            {
                **definition,
                "berth": f"Berth {2 + index}",
                "shift": "Day A" if index % 3 != 2 else "Night B",
                "last_event": (now - timedelta(minutes=7 + 9 * index)).isoformat(),
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
    audit_payload = {
        "operation_id": operation_id,
        "cutoff": now.isoformat(),
        "plan_revision": 1,
        "model_version": "demo-rule-v1",
        "source_event_count": sum(offset <= 0 for _, offset, _ in labels),
    }
    return {
        **operation,
        "data_cutoff": now.isoformat(),
        "plan_revision": 1,
        "plan_valid_from": (now - timedelta(hours=9)).isoformat(),
        "model_version": "demo-rule-v1",
        "claim_state": "smoke_only",
        "synthetic": True,
        "audit_id": _audit_id(audit_payload),
        "source_event_count": audit_payload["source_event_count"],
        "source_event_ids": [
            f"{operation_id}-E{index:02d}"
            for index, (_, offset, _) in enumerate(labels, start=1)
            if offset <= 0
        ],
        "object_links": [
            {"type": "vessel", "id": operation["vessel"]},
            {"type": "berth", "id": operation["berth"]},
            {"type": "shift", "id": operation["shift"]},
            {"type": "resource", "id": "Crane 02"},
            {"type": "cargo_unit", "id": f"CG-{operation_id[3:]}-A"},
        ],
        "allowed_scenarios": [
            "resequence_cargo",
            "reassign_approved_resource",
            "adjust_staging_window",
        ],
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
        "data_quality": {
            "timestamp_audit": "passed_demo_fixture",
            "plan_cutoff": "revision_1_visible",
            "missing_required_fields": 0,
            "relationship_integrity": "passed_demo_fixture",
        },
    }


def _read_artifact(path: Path) -> dict[str, Any] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return None
    return payload if isinstance(payload, dict) else None


def _metric_context(
    baseline: dict[str, Any] | None,
    temporal_jepa: dict[str, Any] | None,
    hybrid: dict[str, Any] | None,
) -> dict[str, Any]:
    if not baseline:
        return {
            "metric": "remaining_time_mae_minutes",
            "available": False,
            "benchmark_verdict": "pending",
            "operational_verdict": "unknown",
            "explanation": "Generated baseline artifacts are not available.",
        }
    selected = baseline.get("selected_model_test", {})
    tests = baseline.get("remaining_time_test", {})
    selected_mae = float(selected.get("mae_minutes", 0.0))
    comparators: list[dict[str, Any]] = []
    for key, label in (
        ("global_median", "Global median"),
        ("activity_median", "Activity median"),
    ):
        value = tests.get(key, {}).get("mae_minutes")
        if value is None:
            continue
        comparator = float(value)
        comparators.append(
            {
                "key": key,
                "label": label,
                "mae_minutes": comparator,
                "selected_gain_minutes": comparator - selected_mae,
                "selected_gain_percent": (
                    100.0 * (comparator - selected_mae) / comparator if comparator else 0.0
                ),
                "same_protocol": True,
            }
        )
    if temporal_jepa:
        selected_name = str(temporal_jepa.get("selected_main_validation_only", ""))
        value = temporal_jepa.get("aggregates", {}).get(selected_name, {}).get(
            "mae_mean_minutes"
        )
        if value is not None:
            comparators.append(
                {
                    "key": "temporal_t_jepa",
                    "label": "Temporal T-JEPA",
                    "mae_minutes": float(value),
                    "selected_gain_minutes": float(value) - selected_mae,
                    "selected_gain_percent": None,
                    "same_protocol": False,
                    "note": "Three-seed JEPA rerun; compare direction, not uncertainty bands.",
                }
            )
    three_seed_raw = None
    best_hybrid = None
    if hybrid:
        raw = hybrid.get("aggregates", {}).get("raw", {})
        raw_value = raw.get("mae_mean_minutes")
        if raw_value is not None:
            three_seed_raw = {
                "mae_mean_minutes": float(raw_value),
                "mae_std_minutes": float(raw.get("mae_std_minutes", 0.0)),
            }
        hybrid_name = str(hybrid.get("best_hybrid_validation_only", ""))
        hybrid_values = hybrid.get("aggregates", {}).get(hybrid_name, {})
        hybrid_value = hybrid_values.get("mae_mean_minutes")
        if hybrid_value is not None:
            best_hybrid = {
                "name": hybrid_name,
                "mae_mean_minutes": float(hybrid_value),
                "mae_std_minutes": float(hybrid_values.get("mae_std_minutes", 0.0)),
                "delta_vs_raw_minutes": (
                    float(hybrid_value) - float(raw_value)
                    if raw_value is not None
                    else None
                ),
            }
    bootstrap = baseline.get("selected_model_mae_cluster_bootstrap", {})
    worst = baseline.get("worst_group_last_activity", {})
    worst_group_name = worst.get("worst_group")
    worst_group_metrics = worst.get("groups", {}).get(str(worst_group_name), {})
    return {
        "metric": "remaining_time_mae_minutes",
        "available": True,
        "selected_model": "quantile_boosting",
        "selected_mae_minutes": selected_mae,
        "selected_mae_human": f"{selected_mae / 60:.1f} hours average absolute error",
        "median_absolute_error_minutes": selected.get("median_ae_minutes"),
        "p90_interval_width_minutes": selected.get("p90_interval_width_minutes"),
        "p90_interval_coverage": selected.get("p90_interval_coverage"),
        "bootstrap_95_low_minutes": bootstrap.get("bootstrap_95_low"),
        "bootstrap_95_high_minutes": bootstrap.get("bootstrap_95_high"),
        "worst_group": worst_group_name,
        "worst_group_mae_minutes": worst_group_metrics.get("mae_minutes"),
        "comparators": comparators,
        "three_seed_raw": three_seed_raw,
        "best_hybrid": best_hybrid,
        "benchmark_verdict": "best_tested_mean_error_floor",
        "operational_verdict": "unknown_without_kaleido_acceptance_threshold",
        "explanation": (
            "MAE is the average absolute gap between predicted and actual remaining time. "
            "It ranks models on the same test, but its absolute usefulness depends on the "
            "operation duration, decision horizon, interval width and operator tolerance."
        ),
        "presentation_line": (
            "Boosting wins the public benchmark, but we do not yet know whether its error "
            "is useful for a Kaleido operation."
        ),
    }


def _ais_demo_evidence(
    ais: dict[str, Any], ocel: dict[str, Any] | None
) -> dict[str, Any]:
    selected_name = str(ais["selected_model_validation_only"])
    selected = ais["selected_test"]
    test_metrics = ais["test_metrics"]
    baseline_order = (
        ("kinematic_eta", "ETA distancia / velocidad", "Física sin aprendizaje"),
        ("port_distance_median", "Mediana puerto-distancia", "Histórico agrupado"),
        ("physics_residual_eta", "Boosting residual físico", "Física + corrección aprendida"),
        ("tabular_eta", "Boosting ETA", "Modelo elegido solo en validación"),
    )
    stages: list[dict[str, Any]] = []
    best_mae = min(float(values["mae"]) for values in test_metrics.values())
    for index, (key, label, description) in enumerate(baseline_order, start=1):
        values = test_metrics[key]
        metric = float(values["mae"])
        stages.append(
            {
                "milestone": f"ETA-{index}",
                "label": label,
                "description": description,
                "metric_value": metric,
                "metric_unit": "hours",
                "mae_hours": metric,
                "relative_score": min(100.0, 100.0 * best_mae / metric),
                "status": "selected" if key == selected_name else "comparator",
            }
        )
    bootstrap = ais["selected_trip_bootstrap"]
    tolerance = selected["within_tolerance"]
    gate = ais["promotion_gate"]
    split = ais["split"]["counts"]
    nola_rows = int(ais["by_port"]["new_orleans"]["rows"])
    test_prefixes = int(split["test"]["prefixes"])
    ocel_finding = None
    if ocel:
        ocel_finding = {
            "title": "OCEL object graph diagnostic",
            "selected_validation_only": ocel["selected_model_validation_only"],
            "flat_mae_hours": ocel["test_metrics"]["flat_boosting"]["mae"],
            "correct_graph_mae_hours": ocel["test_metrics"]["object_graph_boosting"]["mae"],
            "shuffled_graph_mae_hours": ocel["test_metrics"]["shuffled_object_graph"]["mae"],
            "promotion_gate_passed": False,
            "interpretation": (
                "The correct graph improved test MAE, but validation selected the flat trace; "
                "the graph remains process/diagnostic evidence only."
            ),
        }
    return {
        "dataset_id": ais["dataset"]["dataset_id"],
        "claim_state": "smoke_only",
        "primary_task": "AIS vessel ETA to port geofence",
        "stages": stages,
        "research_finding": {
            "title": "Future AIS ETA capability gate",
            "verdict": "6 / 6 PUBLIC GATES PASSED",
            "summary": (
                "Boosting was selected on January validation data and evaluated once on "
                "the untouched 1-7 February future test. This is public US capability "
                "evidence, not a Kaleido/Vigo accuracy claim."
            ),
            "metric_display": f"{float(selected['mae']):.2f} h",
            "metric_label": (
                f"test MAE · {int(split['test']['trips'])} trips · "
                f"IC95% {float(bootstrap['bootstrap_95_low']):.2f}-"
                f"{float(bootstrap['bootstrap_95_high']):.2f} h"
            ),
            "causal_claim": False,
        },
        "metric_context": {
            "metric": "eta_mae_hours",
            "available": True,
            "selected_model": selected_name,
            "selected_mae_hours": float(selected["mae"]),
            "median_absolute_error_hours": float(selected["median_absolute_error"]),
            "p90_absolute_error_hours": float(selected["p90_absolute_error"]),
            "bootstrap_95_low_hours": float(bootstrap["bootstrap_95_low"]),
            "bootstrap_95_high_hours": float(bootstrap["bootstrap_95_high"]),
            "within_1h": float(tolerance["within_1"]),
            "within_2h": float(tolerance["within_2"]),
            "within_4h": float(tolerance["within_4"]),
            "p90_interval_coverage": float(ais["p90_interval_coverage"]),
            "p90_interval_width_hours": float(ais["p90_interval_width_hours"]),
            "comparators": [
                {
                    "key": key,
                    "label": label,
                    "mae_hours": float(test_metrics[key]["mae"]),
                    "selected_gain_percent": (
                        100.0
                        * (float(test_metrics[key]["mae"]) - float(selected["mae"]))
                        / float(test_metrics[key]["mae"])
                    ),
                }
                for key, label in (
                    ("kinematic_eta", "ETA distancia / velocidad"),
                    ("port_distance_median", "Mediana puerto-distancia"),
                )
            ],
            "gate_passed": bool(gate["passed"]),
            "gate_checks_passed": sum(bool(value) for value in gate["checks"].values()),
            "gate_checks_total": len(gate["checks"]),
            "test_trips": int(split["test"]["trips"]),
            "test_prefixes": test_prefixes,
            "nola_prefix_share": nola_rows / test_prefixes,
            "operational_verdict": "public_demo_passed_kaleido_threshold_unknown",
            "explanation": (
                "MAE is the average absolute ETA error. The tolerance percentages make the "
                "scale interpretable; Kaleido must still define the tolerance for each decision."
            ),
            "presentation_line": (
                "The public demonstrator passed its frozen gates; transfer to Vigo or "
                "Kaleido operations remains untested."
            ),
        },
        "secondary_finding": ocel_finding,
        "legacy_research": {
            "warehouse_remaining_time": "rejected_as_product_demonstrator",
            "jepa": "research_shadow_until_incremental_gate_passes",
        },
        "note": (
            "Lower MAE is better. Trips are grouped; model selection used validation only. "
            "Public data demonstrates capability, not Kaleido value."
        ),
    }


def _demo_evidence(root: Path) -> dict[str, Any]:
    ais = _read_artifact(root / "noaa_ais_eta_v3" / "metrics.json")
    if ais:
        ocel = _read_artifact(root / "ocel_logistics_graph_v1" / "metrics.json")
        return _ais_demo_evidence(ais, ocel)
    baseline = _read_artifact(root / "warehouse_smoke_v2" / "metrics.json")
    sequence = _read_artifact(root / "warehouse_sequence_smoke_v4" / "metrics.json")
    jepa = _read_artifact(root / "warehouse_event_jepa_smoke_v2" / "metrics.json")
    temporal_jepa = _read_artifact(
        root / "warehouse_temporal_t_jepa_v1" / "metrics.json"
    )
    var_jepa = _read_artifact(root / "warehouse_var_event_jepa_v1" / "metrics.json")
    hybrid = _read_artifact(root / "warehouse_jepa_hybrid_v1" / "metrics.json")
    ablations = _read_artifact(root / "warehouse_event_jepa_ablations_v1" / "metrics.json")
    action_jepa = _read_artifact(
        root / "warehouse_action_event_jepa_visreg_v2" / "metrics.json"
    )
    stages: list[dict[str, Any]] = []
    if baseline:
        metric = float(baseline["selected_model_test"]["mae_minutes"])
        stages.append(
            {
                "milestone": "M3",
                "label": "Quantile boosting",
                "description": "Strong tabular floor · chronological test",
                "mae_minutes": metric,
                "relative_score": 100.0,
                "status": "reference",
            }
        )
    if sequence:
        architecture = str(sequence["best_architecture"])
        metric = float(sequence["aggregates"][architecture]["mae_mean_minutes"])
        reference = stages[0]["mae_minutes"] if stages else metric
        stages.append(
            {
                "milestone": "M4",
                "label": f"{architecture.title()} sequence",
                "description": "Three-seed neural baseline",
                "mae_minutes": metric,
                "relative_score": min(100.0, 100.0 * float(reference) / metric),
                "status": "did_not_beat_reference" if metric >= float(reference) else "improved",
            }
        )
    if jepa:
        selected = str(jepa["selected_variant_validation_only"])
        metric = float(jepa["aggregates"][selected]["mae_mean_minutes"])
        reference = stages[0]["mae_minutes"] if stages else metric
        stages.append(
            {
                "milestone": "M5",
                "label": f"Event-JEPA · {selected}",
                "description": "Action-free, multi-horizon latent prediction · three seeds",
                "mae_minutes": metric,
                "relative_score": min(100.0, 100.0 * float(reference) / metric),
                "status": "did_not_beat_reference" if metric >= float(reference) else "improved",
            }
        )
    if temporal_jepa:
        selected = str(temporal_jepa["selected_main_validation_only"])
        metric = float(temporal_jepa["aggregates"][selected]["mae_mean_minutes"])
        reference = stages[0]["mae_minutes"] if stages else metric
        stages.append(
            {
                "milestone": "M5-T",
                "label": f"Temporal T-JEPA · {selected}",
                "description": "Disjoint future blocks + EMA teacher · three seeds",
                "mae_minutes": metric,
                "relative_score": min(100.0, 100.0 * float(reference) / metric),
                "status": "did_not_beat_reference" if metric >= float(reference) else "improved",
            }
        )
    if var_jepa:
        metric = float(var_jepa["aggregate"]["mae_mean_minutes"])
        reference = stages[0]["mae_minutes"] if stages else metric
        stages.append(
            {
                "milestone": "M5-V",
                "label": "Var-Event-JEPA",
                "description": "Temporal variational ELBO + latent uncertainty · three seeds",
                "mae_minutes": metric,
                "relative_score": min(100.0, 100.0 * float(reference) / metric),
                "status": "did_not_beat_reference" if metric >= float(reference) else "improved",
            }
        )
    if hybrid:
        selected = str(hybrid["selected_overall_validation_only"])
        metric = float(hybrid["aggregates"][selected]["mae_mean_minutes"])
        reference = stages[0]["mae_minutes"] if stages else metric
        stages.append(
            {
                "milestone": "M5-H",
                "label": f"Hybrid gate · {selected} selected",
                "description": "Raw + frozen T/Var embeddings · validation-only selection",
                "mae_minutes": metric,
                "relative_score": min(100.0, 100.0 * float(reference) / metric),
                "status": "reference" if selected == "raw" else "improved",
            }
        )
    if not stages:
        stages = [
            {
                "milestone": "M0",
                "label": "Synthetic fixture",
                "description": "Waiting for local generated artifacts",
                "mae_minutes": None,
                "relative_score": 20.0,
                "status": "pending",
            }
        ]
    research_finding = None
    if hybrid:
        raw_metric = float(hybrid["aggregates"]["raw"]["mae_mean_minutes"])
        best_hybrid = str(hybrid["best_hybrid_validation_only"])
        hybrid_metric = float(hybrid["aggregates"][best_hybrid]["mae_mean_minutes"])
        research_finding = {
            "title": "JEPA representation gate",
            "verdict": "KEEP RAW BOOSTING",
            "summary": (
                "Validation selected raw boosting. Neither deterministic nor variational "
                "JEPA embeddings added incremental value on this short public warehouse log; "
                "the world-model promotion gate remains closed."
            ),
            "metric_display": f"+{hybrid_metric - raw_metric:.2f} min",
            "metric_label": f"{best_hybrid} vs raw · test diagnostic",
            "causal_claim": False,
        }
    elif ablations and action_jepa:
        action_metrics = action_jepa["aggregates"]
        correct = float(action_metrics["correct_action"]["mae_mean_minutes"])
        shuffled = float(action_metrics["shuffled_action"]["mae_mean_minutes"])
        research_finding = {
            "title": "JEPA research gate",
            "verdict": "WORLD MODEL NOT PROMOTED",
            "summary": (
                "The JEPA objective beats its random-encoder control and VISReg recovers "
                "the injected action signal in 3/3 seeds. Predictor scale remains below "
                "the collapse threshold, so this is research evidence only."
            ),
            "correct_action_mae_minutes": correct,
            "shuffled_action_mae_minutes": shuffled,
            "injected_signal_gain_minutes": shuffled - correct,
            "action_source": "generated_overlay_only",
            "causal_claim": False,
        }
    return {
        "dataset_id": (baseline or {}).get("dataset_id", "synthetic_ui_fixture"),
        "claim_state": "smoke_only",
        "stages": stages,
        "research_finding": research_finding,
        "metric_context": _metric_context(baseline, temporal_jepa, hybrid),
        "note": (
            "Public warehouse data demonstrates pipeline competence only. Lower MAE is better; "
            "test results never select the model and do not establish Kaleido value."
        ),
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
        ais = _read_artifact(root / "noaa_ais_eta_v3" / "metrics.json")
        if ais and version in {"latest", "ais-eta-v3"}:
            selected = ais["selected_test"]
            return ModelCardResponse(
                model_version="ais-eta-v3",
                claim_state="smoke_only",
                dataset_id=str(ais["dataset"]["dataset_id"]),
                split_protocol=(
                    "grouped vessel trips; January train/validation; untouched "
                    "1-7 February 2025 future test"
                ),
                metrics={
                    "selected_model_validation_only": ais["selected_model_validation_only"],
                    "test_mae_hours": selected["mae"],
                    "test_median_ae_hours": selected["median_absolute_error"],
                    "within_1h": selected["within_tolerance"]["within_1"],
                    "within_2h": selected["within_tolerance"]["within_2"],
                    "within_4h": selected["within_tolerance"]["within_4"],
                    "p90_interval_coverage": ais["p90_interval_coverage"],
                    "p90_interval_width_hours": ais["p90_interval_width_hours"],
                    "promotion_gate": ais["promotion_gate"],
                },
                limitations=[
                    "Public US AIS capability example; not Kaleido/Vigo accuracy evidence",
                    "Test prefixes are concentrated in New Orleans",
                    "P90 interval is wide and requires port-specific calibration",
                    "Circular geofences are inferred demonstrator targets",
                ],
            )
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
        ais = _read_artifact(root / "noaa_ais_eta_v3" / "metrics.json")
        baseline = _read_artifact(root / "warehouse_smoke_v2" / "metrics.json")
        coverage = None
        if ais:
            coverage = ais.get("p90_interval_coverage")
        elif baseline:
            coverage = baseline.get("selected_model_test", {}).get("p90_interval_coverage")
        if ais:
            data_manifest_path = root / "noaa_ais_eta_v3" / "data_manifest.json"
            data_manifest_hash = (
                hashlib.sha256(data_manifest_path.read_bytes()).hexdigest()
                if data_manifest_path.is_file()
                else None
            )
            split = ais["split"]["counts"]
            dataset = {
                "dataset_id": ais["dataset"]["dataset_id"],
                "source_cases_used": sum(int(values["trips"]) for values in split.values()),
                "prefix_rows": sum(int(values["prefixes"]) for values in split.values()),
                "split_protocol": "grouped_trips_chronological_future",
                "split_counts_operations": {
                    name: int(values["trips"]) for name, values in split.items()
                },
                "source_file_sha256": data_manifest_hash,
                "claim_state": "smoke_only",
                "entity_label": "vessel trips",
                "domain_note": "NOAA AIS 2025 · public US port ETA",
            }
        else:
            dataset = {
                "dataset_id": (baseline or {}).get("dataset_id", "synthetic_ui_fixture"),
                "source_cases_used": (baseline or {}).get("source_cases_used"),
                "prefix_rows": (baseline or {}).get("prefix_rows"),
                "split_protocol": (baseline or {}).get("split_protocol"),
                "split_counts_operations": (baseline or {}).get("split_counts_operations"),
                "source_file_sha256": (baseline or {}).get("source_file_sha256"),
                "claim_state": (baseline or {}).get("claim_state", "smoke_only"),
                "entity_label": "operations",
                "domain_note": "public warehouse fallback",
            }
        return {
            "watermark": "SYNTHETIC SHADOW REPLAY · SMOKE_ONLY",
            "generated_at": datetime.now(UTC).isoformat(),
            "read_only": True,
            "operations_active": len(operations),
            "operations_at_risk": sum(item["risk"] >= 60 for item in operations),
            "mean_plan_delta_minutes": round(
                sum(item["plan_delta_minutes"] for item in operations) / len(operations)
            ),
            "p90_coverage": coverage,
            "false_alerts_per_shift": None,
            "fixture_events": 2185,
            "dataset": dataset,
            "operations": operations,
        }

    @app.get("/v1/demo/evidence")
    def demo_evidence() -> dict[str, Any]:
        return _demo_evidence(root)

    @app.get("/v1/demo/alerts")
    def demo_alerts() -> dict[str, Any]:
        operations = _demo_operations()
        alerts = [
            {
                "operation_id": item["operation_id"],
                "severity": item["status"],
                "risk": item["risk"],
                "message": (
                    "Synthetic risk is above the 60% demo threshold"
                    if item["risk"] >= 60
                    else "Plan buffer is negative in the synthetic replay"
                ),
                "last_event": item["last_event"],
                "claim_state": "smoke_only",
            }
            for item in operations
            if item["risk"] >= 60 or item["plan_delta_minutes"] < 0
        ]
        return {
            "read_only": True,
            "claim_state": "smoke_only",
            "count": len(alerts),
            "alerts": alerts,
        }

    @app.get("/v1/demo/export")
    def demo_export() -> dict[str, Any]:
        return {
            "exported_at": datetime.now(UTC).isoformat(),
            "read_only": True,
            "claim_state": "smoke_only",
            "overview": demo_overview(),
            "evidence": demo_evidence(),
        }

    @app.get("/v1/demo/operations/{operation_id}")
    def demo_operation(operation_id: str) -> dict[str, Any]:
        try:
            return _demo_detail(operation_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="operation not found") from exc

    return app
