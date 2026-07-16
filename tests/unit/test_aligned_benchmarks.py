from __future__ import annotations

import numpy as np
import pandas as pd

from flowtwin.benchmarks.ais_eta import _candidate_arrivals, _trip_rows
from flowtwin.benchmarks.common import (
    chronological_group_partitions,
    regression_diagnostics,
)


def test_regression_diagnostics_reports_operational_windows() -> None:
    result = regression_diagnostics(
        np.asarray([1.0, 2.0, 4.0]),
        np.asarray([1.5, 1.0, 7.0]),
        tolerances=(1.0, 2.0),
    )

    assert result["mae"] == 1.5
    assert result["median_absolute_error"] == 1.0
    assert result["within_tolerance"]["within_1"] == 2 / 3


def test_chronological_partitions_keep_groups_ordered() -> None:
    groups = np.asarray([f"trip-{index}" for index in range(10)])
    times = np.asarray(pd.date_range("2025-01-01", periods=10, freq="D"))
    partitions = chronological_group_partitions(groups, times)

    assert [partitions[group] for group in groups[:7]] == ["train"] * 7
    assert partitions["trip-7"] == "validation"
    assert partitions["trip-9"] == "test"


def test_ais_arrival_requires_crossing_and_cooldown() -> None:
    timestamps = pd.date_range("2025-01-01", periods=8, freq="2h", tz="UTC")
    frame = pd.DataFrame(
        {
            "timestamp": timestamps,
            "distance_km": [20.0, 10.0, 7.0, 12.0, 7.0, 20.0, 10.0, 7.0],
        }
    )

    assert _candidate_arrivals(frame, radius_km=8.0) == [2, 7]


def test_ais_trip_features_use_only_pre_arrival_rows() -> None:
    timestamps = pd.date_range("2025-01-01", periods=8, freq="1h", tz="UTC")
    frame = pd.DataFrame(
        {
            "mmsi": [123456789] * 8,
            "timestamp": timestamps,
            "longitude": np.linspace(-91.0, -90.05, 8),
            "latitude": np.linspace(29.5, 29.95, 8),
            "distance_km": [100.0, 80.0, 60.0, 45.0, 30.0, 20.0, 10.0, 5.0],
            "bearing_to_port": [90.0] * 8,
            "sog": [12.0] * 8,
            "cog": [90.0] * 8,
            "vessel_type": [70] * 8,
            "length": [180.0] * 8,
            "width": [28.0] * 8,
            "draft": [9.0] * 8,
        }
    )
    rows = _trip_rows(frame, 7, "new_orleans", None)

    assert rows
    assert all(row["prediction_cutoff"] < row["arrival_time"] for row in rows)
    assert all(0.25 <= row["remaining_hours"] <= 12.0 for row in rows)
    assert all("mmsi" in row for row in rows)
