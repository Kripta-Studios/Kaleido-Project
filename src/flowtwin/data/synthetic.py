from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import polars as pl


@dataclass(frozen=True)
class SyntheticFixtureSummary:
    operations: int
    events: int
    completed: int
    censored: int
    plan_revisions: int
    output_dir: Path


def generate_trace_port_fixture(
    output_dir: Path,
    *,
    operations: int = 240,
    seed: int = 42,
) -> SyntheticFixtureSummary:
    """Create a deterministic, explicitly synthetic Trace Port-like export."""
    if operations < 10:
        raise ValueError("fixture requires at least ten operations")
    rng = np.random.default_rng(seed)
    base = datetime(2025, 1, 1, 6, tzinfo=UTC)
    events: list[dict[str, object]] = []
    plans: list[dict[str, object]] = []
    outcomes: list[dict[str, object]] = []
    event_number = 0
    revision_count = 0
    censored_count = 0

    for index in range(operations):
        operation_id = f"SYN-OP-{index:05d}"
        project_id = f"SYN-PROJECT-{index // 12:03d}"
        shift_id = f"SYN-SHIFT-{index:05d}"
        cargo_type = ("breakbulk", "container", "wind_component")[index % 3]
        vessel_id = f"SYN-VESSEL-{index % 17:03d}"
        resource_id = f"SYN-CRANE-{index % 5:02d}"
        start = base + timedelta(hours=6 * index + float(rng.uniform(0, 1)))
        planned_duration = float(rng.normal(9 * 60, 75))
        planned_duration = max(240.0, planned_duration)
        planned_end = start + timedelta(minutes=planned_duration)
        plans.append(
            {
                "plan_id": f"SYN-PLAN-{index:05d}",
                "revision": 0,
                "valid_from_utc": start - timedelta(days=2),
                "operation_id": operation_id,
                "milestone": "operation_complete",
                "planned_start_utc": start,
                "planned_end_utc": planned_end,
                "reason": "synthetic_initial_plan",
            }
        )
        revision_count += 1
        has_revision = bool(rng.random() < 0.18)
        if has_revision:
            revision_time = start + timedelta(minutes=planned_duration * 0.35)
            planned_end = planned_end + timedelta(minutes=float(rng.normal(50, 25)))
            plans.append(
                {
                    "plan_id": f"SYN-PLAN-{index:05d}",
                    "revision": 1,
                    "valid_from_utc": revision_time,
                    "operation_id": operation_id,
                    "milestone": "operation_complete",
                    "planned_start_utc": start,
                    "planned_end_utc": planned_end,
                    "reason": "synthetic_weather_replan",
                }
            )
            revision_count += 1

        congestion = 1 + 0.18 * (index % 7 == 0) + float(rng.normal(0, 0.08))
        cargo_factor = {"breakbulk": 1.0, "container": 0.82, "wind_component": 1.25}[cargo_type]
        actual_duration = max(180.0, planned_duration * congestion * cargo_factor)
        incident = bool(rng.random() < 0.1)
        if incident:
            actual_duration += float(rng.uniform(80, 220))
        censored = bool(rng.random() < 0.05)
        if censored:
            censored_count += 1

        sequence = [
            ("operation_created", 0.0),
            ("shift_started", 0.02),
            ("cargo_ready", 0.1),
            ("resource_assigned", 0.14),
            ("handling_started", 0.2),
            ("cargo_progress", 0.38),
            ("cargo_progress", 0.58),
            ("cargo_progress", 0.78),
        ]
        if incident:
            sequence.extend([("incident_opened", 0.5), ("incident_resolved", 0.68)])
        sequence.sort(key=lambda item: item[1])
        if not censored:
            sequence.append(("operation_completed", 1.0))
        else:
            sequence = [item for item in sequence if item[1] <= 0.62]

        for event_type, fraction in sequence:
            event_time = start + timedelta(minutes=actual_duration * fraction)
            ingested_at = event_time + timedelta(minutes=float(rng.uniform(0, 8)))
            event_number += 1
            events.append(
                {
                    "event_id": f"SYN-EVT-{event_number:07d}",
                    "source_record_id": f"SYN-ROW-{event_number:07d}",
                    "event_type": event_type,
                    "event_time": event_time,
                    "ingested_at": ingested_at,
                    "project_id": project_id,
                    "operation_id": operation_id,
                    "shift_id": shift_id,
                    "cargo_unit_id": f"SYN-CARGO-{index:05d}",
                    "resource_id": resource_id,
                    "vessel_id": vessel_id,
                    "location_id": f"SYN-BERTH-{index % 4:02d}",
                    "cargo_type": cargo_type,
                    "numeric_value": min(100.0, fraction * 100),
                    "unit": "percent",
                    "operator_action": (
                        "assign_resource" if event_type == "resource_assigned" else None
                    ),
                    "synthetic": True,
                }
            )
        outcomes.append(
            {
                "operation_id": operation_id,
                "completed_at_utc": None
                if censored
                else start + timedelta(minutes=actual_duration),
                "completion_status": "incomplete" if censored else "complete",
                "deviation_minutes": None if censored else actual_duration - planned_duration,
                "incident_status_known": not censored,
                "incident_types": ["synthetic_handling_incident"]
                if incident and not censored
                else [],
                "censored": censored,
                "synthetic": True,
            }
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(events).write_parquet(output_dir / "events.parquet")
    pl.DataFrame(plans).write_parquet(output_dir / "plan_revisions.parquet")
    pl.DataFrame(outcomes).write_parquet(output_dir / "outcomes.parquet")
    return SyntheticFixtureSummary(
        operations=operations,
        events=len(events),
        completed=operations - censored_count,
        censored=censored_count,
        plan_revisions=revision_count,
        output_dir=output_dir,
    )
