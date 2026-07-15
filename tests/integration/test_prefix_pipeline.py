from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import polars as pl

from flowtwin.features.prefix import build_prefix_dataset


def _write_log(path: Path, cases: int = 20) -> None:
    start = datetime(2025, 1, 1, tzinfo=UTC)
    rows = []
    for case in range(cases):
        for event in range(5):
            rows.append(
                {
                    "case:concept:name": f"case-{case:02d}",
                    "concept:name": f"A{event}",
                    "time:timestamp": (
                        start + timedelta(days=case, minutes=event * (10 + case))
                    ).isoformat(),
                }
            )
    pl.DataFrame(rows).write_csv(path)


def test_prefix_dataset_is_cutoff_safe_and_grouped(tmp_path: Path) -> None:
    source = tmp_path / "log.csv"
    _write_log(source)
    dataset = build_prefix_dataset(source, prediction_points=(0.25, 0.5, 0.75))
    frame = dataset.frame
    assert frame.height == 60
    assert (frame["remaining_minutes"] >= 0).all()
    assert (frame["remaining_minutes"] > 0).all()
    assert "progress_ratio" not in frame.columns
    assignments = frame.group_by("operation_id").agg(pl.col("partition").n_unique())
    assert assignments["partition"].max() == 1
    assert set(frame["partition"].unique()) == {"train", "validation", "test"}


def test_prefix_future_end_is_target_not_feature(tmp_path: Path) -> None:
    source = tmp_path / "log.csv"
    _write_log(source)
    frame = build_prefix_dataset(source).frame
    assert "_case_end" not in frame.columns
    assert "remaining_minutes" in frame.columns
