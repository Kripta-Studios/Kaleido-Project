from __future__ import annotations

import json
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
    assert "Synthetic operational replay" in page.text


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
    assert overview["read_only"] is True
    assert len(overview["operations"]) == 8
    detail = client.get("/v1/demo/operations/TP-2407").json()
    assert detail["synthetic"] is True
    assert detail["claim_state"] == "smoke_only"
    assert detail["audit_id"]
    assert detail["source_event_count"] == len(detail["source_event_ids"])
    assert detail["allowed_scenarios"]


def test_demo_dashboard_controls_have_read_only_backends(tmp_path: Path) -> None:
    client = TestClient(create_app(tmp_path))
    page = client.get("/").text
    for control_id in (
        "refresh-button",
        "alerts-button",
        "profile-button",
        "filters-toggle",
        "export-operations",
        "export-evidence",
        "audit-open",
    ):
        assert f'id="{control_id}"' in page
    alerts = client.get("/v1/demo/alerts")
    assert alerts.status_code == 200
    assert alerts.json()["read_only"] is True
    assert alerts.json()["count"] >= 1
    export = client.get("/v1/demo/export")
    assert export.status_code == 200
    assert export.json()["claim_state"] == "smoke_only"
    scenario = client.post(
        "/v1/scenarios/rank",
        json={
            "operation_id": "TP-2407",
            "approved_actions": ["resequence_cargo", "adjust_staging_window"],
        },
    )
    assert scenario.status_code == 200
    assert scenario.json()["advisory_only"] is True
    assert scenario.json()["evidence_type"] == "simulation_not_realized_saving"


def test_demo_evidence_exposes_jepa_gate_from_generated_artifacts(
    tmp_path: Path,
) -> None:
    artifacts = {
        "warehouse_smoke_v2": {
            "dataset_id": "public-fixture",
            "selected_model_test": {"mae_minutes": 734.5},
        },
        "warehouse_temporal_t_jepa_v1": {
            "selected_main_validation_only": "multi_visreg",
            "aggregates": {"multi_visreg": {"mae_mean_minutes": 759.4}},
        },
        "warehouse_var_event_jepa_v1": {
            "aggregate": {"mae_mean_minutes": 760.4},
        },
        "warehouse_jepa_hybrid_v1": {
            "selected_overall_validation_only": "raw",
            "best_hybrid_validation_only": "raw_var_jepa",
            "aggregates": {
                "raw": {"mae_mean_minutes": 734.4},
                "raw_var_jepa": {"mae_mean_minutes": 736.6},
            },
        },
    }
    for run_name, metrics in artifacts.items():
        run = tmp_path / run_name
        run.mkdir()
        (run / "metrics.json").write_text(json.dumps(metrics), encoding="utf-8")

    payload = TestClient(create_app(tmp_path)).get("/v1/demo/evidence").json()
    assert payload["claim_state"] == "smoke_only"
    assert [stage["milestone"] for stage in payload["stages"]] == [
        "M3",
        "M5-T",
        "M5-V",
        "M5-H",
    ]
    assert payload["research_finding"]["verdict"] == "KEEP RAW BOOSTING"
    assert payload["research_finding"]["metric_display"] == "+2.20 min"
    assert payload["metric_context"]["benchmark_verdict"] == ("best_tested_mean_error_floor")
    assert payload["metric_context"]["operational_verdict"] == (
        "unknown_without_kaleido_acceptance_threshold"
    )


def test_demo_evidence_prefers_future_ais_eta_artifacts(tmp_path: Path) -> None:
    run = tmp_path / "noaa_ais_eta_v3"
    run.mkdir()
    selected = {
        "mae": 1.88,
        "median_absolute_error": 1.37,
        "p90_absolute_error": 4.28,
        "within_tolerance": {"within_1": 0.42, "within_2": 0.61, "within_4": 0.87},
    }
    metrics = {
        "dataset": {"dataset_id": "noaa-ais-test"},
        "selected_model_validation_only": "tabular_eta",
        "selected_test": selected,
        "test_metrics": {
            "kinematic_eta": {**selected, "mae": 7.79},
            "port_distance_median": {**selected, "mae": 2.73},
            "physics_residual_eta": {**selected, "mae": 1.89},
            "tabular_eta": selected,
        },
        "selected_trip_bootstrap": {
            "bootstrap_95_low": 1.70,
            "bootstrap_95_high": 2.08,
        },
        "promotion_gate": {
            "passed": True,
            "checks": {f"gate_{index}": True for index in range(6)},
        },
        "split": {
            "counts": {
                "train": {"trips": 303, "prefixes": 5893},
                "validation": {"trips": 73, "prefixes": 1381},
                "test": {"trips": 85, "prefixes": 1780},
            }
        },
        "by_port": {"new_orleans": {"rows": 1559}},
        "p90_interval_coverage": 0.945,
        "p90_interval_width_hours": 9.04,
    }
    (run / "metrics.json").write_text(json.dumps(metrics), encoding="utf-8")
    (run / "data_manifest.json").write_text("{}", encoding="utf-8")

    client = TestClient(create_app(tmp_path))
    evidence = client.get("/v1/demo/evidence").json()
    assert evidence["dataset_id"] == "noaa-ais-test"
    assert evidence["research_finding"]["verdict"] == "6 / 6 PUBLIC GATES PASSED"
    assert evidence["metric_context"]["selected_mae_hours"] == 1.88
    assert evidence["metric_context"]["test_trips"] == 85
    assert [stage["milestone"] for stage in evidence["stages"]] == [
        "ETA-1",
        "ETA-2",
        "ETA-3",
        "ETA-4",
    ]
    overview = client.get("/v1/demo/overview").json()
    assert overview["dataset"]["split_counts_operations"]["test"] == 85
    card = client.get("/v1/models/latest/card").json()
    assert card["model_version"] == "ais-eta-v3"
    assert card["metrics"]["test_mae_hours"] == 1.88


def test_demo_evidence_prefers_clean_phys_jepa_core_artifact(tmp_path: Path) -> None:
    run = tmp_path / "noaa_ais_phys_jepa_clean_test_v2"
    run.mkdir()
    result_rows = [
        {
            "embedding_diagnostics_test": {
                "effective_rank": rank,
                "collapsed": False,
            },
            "full_label_trajectory": {
                "hybrid_conformal": {
                    "test_coverage": coverage,
                    "mean_interval_width_km": width,
                }
            },
        }
        for rank, coverage, width in (
            (11.4, 0.893, 11.9),
            (12.5, 0.898, 11.9),
            (12.2, 0.903, 12.2),
        )
    ]
    downstream = {
        "full_trajectory_test_hybrid_mae_mean_km": 2.587,
        "full_trajectory_test_hybrid_mae_std_km": 0.053,
        "full_trajectory_test_raw_deviation_auprc_mean": 0.880,
        "full_trajectory_test_hybrid_deviation_auprc_mean": 0.904,
        "sparse_eta_test_relative_improvement_percent": 0.587,
        "sparse_delay_test_raw_auprc_mean": 0.619,
        "sparse_delay_test_hybrid_auprc_mean": 0.606,
    }
    metrics = {
        "claim_state": "claim_eligible",
        "dataset_export_version": 1,
        "dataset_id": "noaa-ais-phys-clean",
        "split_protocol": "chronological_future_grouped_vessel_trip",
        "split_counts": {
            "train": {"trips": 341, "samples": 3778},
            "validation": {"trips": 83, "samples": 976},
            "test": {"trips": 57, "samples": 750},
        },
        "number_of_seeds": 3,
        "test_influenced_choice": False,
        "threshold_selection": "fixed physical 2h shortfall definition",
        "models_and_baselines": {
            "kinematic": {"distance_mae_km": 4.018},
            "trajectory_boosting": {"test": {"distance_mae_km": 2.635}},
            "supervised": {
                "gru": {"aggregate": {"test_distance_mae_mean_km": 2.798}},
                "transformer": {"aggregate": {"test_distance_mae_mean_km": 2.830}},
            },
            "jepa": {
                "phys_vicreg": {
                    "aggregate": {"test_distance_mae_mean_km": 3.036},
                    "results": result_rows,
                    "downstream_heads": {"results": result_rows},
                }
            },
        },
        "product_candidate": {
            "model": "trajectory_boosting_plus_phys_vicreg",
            "gate": {"passed": False},
            "selected_downstream": downstream,
            "paired_test_uncertainty": {
                "raw_mae_km": 2.635,
                "hybrid_mae_km": 2.326,
                "relative_improvement_percent": 11.722,
                "relative_improvement_ci95_percent": [5.901, 17.130],
                "bootstrap_probability_improvement": 0.9995,
                "samples": 2000,
            },
        },
    }
    (run / "metrics.json").write_text(json.dumps(metrics), encoding="utf-8")
    (run / "data_manifest.json").write_text("{}", encoding="utf-8")

    client = TestClient(create_app(tmp_path))
    evidence = client.get("/v1/demo/evidence").json()
    assert evidence["claim_state"] == "claim_eligible"
    assert evidence["research_finding"]["verdict"] == ("CORE WORLD MODEL PASSED · FULL GATE CLOSED")
    assert evidence["metric_context"]["hybrid_ensemble_mae_km"] == 2.326
    assert evidence["metric_context"]["collapsed_seeds"] == 0
    assert [stage["milestone"] for stage in evidence["stages"]] == [
        "WM-1",
        "WM-2",
        "WM-3",
        "WM-4",
        "WM-5",
        "WM-6",
    ]
    overview = client.get("/v1/demo/overview").json()
    assert overview["dataset"]["split_counts_operations"]["test"] == 57
    assert overview["p90_coverage"] == 0.898
    card = client.get("/v1/models/latest/card").json()
    assert card["model_version"] == "ais-phys-jepa-v1"
    assert card["claim_state"] == "claim_eligible"
    assert card["metrics"]["full_product_gate"] is False
