from datetime import UTC, datetime, timedelta
from pathlib import Path

import polars as pl

from flowtwin.features.prefix import build_prefix_dataset
from flowtwin.simulation.discrete_event import OperationSimulator, TaskSpec
from flowtwin.simulation.synthetic_actions import generate_synthetic_action_overlay


def test_discrete_event_simulation_is_seeded_and_labeled() -> None:
    simulator = OperationSimulator(
        [
            TaskSpec("stage", "team", 20),
            TaskSpec("lift", "crane", 40),
        ],
        {"team": 1, "crane": 1},
    )
    left = simulator.run(replications=10, seed=7)
    right = simulator.run(replications=10, seed=7)
    assert left.operation_completion_minutes == right.operation_completion_minutes
    assert left.p90_completion_minutes >= left.p50_completion_minutes
    assert left.evidence_type == "discrete_event_simulation_not_realized_saving"


def test_synthetic_actions_are_separate_seeded_post_cutoff_events(tmp_path: Path) -> None:
    source = tmp_path / "log.csv"
    start = datetime(2025, 1, 1, tzinfo=UTC)
    pl.DataFrame(
        [
            {
                "case:concept:name": f"case-{case}",
                "concept:name": f"A{event}",
                "time:timestamp": (
                    start + timedelta(days=case, minutes=event * (10 + case))
                ).isoformat(),
            }
            for case in range(30)
            for event in range(5)
        ]
    ).write_csv(source)
    prefixes = build_prefix_dataset(source).frame
    prefix_path = tmp_path / "prefixes.parquet"
    left_path = tmp_path / "left.parquet"
    right_path = tmp_path / "right.parquet"
    prefixes.write_parquet(prefix_path)
    summary = generate_synthetic_action_overlay(prefix_path, left_path, seed=7)
    generate_synthetic_action_overlay(prefix_path, right_path, seed=7)
    left = pl.read_parquet(left_path)
    right = pl.read_parquet(right_path)
    assert left.equals(right)
    assert summary.evidence_type == "synthetic_injected_action_signal_only"
    assert (left["action_time"] == left["prediction_cutoff"]).all()
    assert (left["behavior_propensity"] > 0).all()
    assert not left["source_action_claim"].any()
