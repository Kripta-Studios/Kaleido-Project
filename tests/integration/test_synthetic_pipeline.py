from __future__ import annotations

from pathlib import Path

import polars as pl

from flowtwin.data.adapters.trace_port import TracePortAdapter
from flowtwin.data.synthetic import generate_trace_port_fixture
from flowtwin.process.bottlenecks import bottleneck_report
from flowtwin.process.discovery import discover_process
from flowtwin.process.variants import variant_report


def test_synthetic_ocel_like_audit_and_process_report(tmp_path: Path) -> None:
    summary = generate_trace_port_fixture(tmp_path / "fixture", operations=30, seed=2)
    adapter = TracePortAdapter(summary.output_dir / "events.parquet")
    audit = adapter.audit()
    assert audit["manifest"]["operations"] == 30
    assert audit["timestamp_report"]["passed"]
    assert audit["leakage_report"]["passed"]
    assert audit["object_graph"]["passed"]
    events = adapter.events()
    assert discover_process(events)["operations"] == 30
    assert variant_report(events)["variant_count"] >= 2
    assert bottleneck_report(events)["bottlenecks"]


def test_censoring_is_not_encoded_as_exact_completion(tmp_path: Path) -> None:
    summary = generate_trace_port_fixture(tmp_path / "fixture", operations=40, seed=3)
    outcomes = pl.read_parquet(summary.output_dir / "outcomes.parquet")
    censored = outcomes.filter(pl.col("censored"))
    assert censored.height > 0
    assert censored["completed_at_utc"].null_count() == censored.height
    assert censored["deviation_minutes"].null_count() == censored.height


def test_plan_revisions_are_separate_immutable_rows(tmp_path: Path) -> None:
    summary = generate_trace_port_fixture(tmp_path / "fixture", operations=40, seed=4)
    plans = pl.read_parquet(summary.output_dir / "plan_revisions.parquet")
    revised = plans.group_by("plan_id").len().filter(pl.col("len") > 1)
    assert revised.height > 0
    for plan_id in revised["plan_id"].to_list():
        history = plans.filter(pl.col("plan_id") == plan_id).sort("revision")
        assert history["valid_from_utc"].is_sorted()
