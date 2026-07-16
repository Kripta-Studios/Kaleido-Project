from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import polars as pl
import pytest

from flowtwin.benchmarks.ais_world_model import (
    _paired_trip_bootstrap_improvement,
    build_ais_trajectory_data,
)
from flowtwin.models.ais_phys_jepa import (
    AISWorldModelConfig,
    ais_jepa_loss,
    build_ais_phys_jepa,
    build_ais_supervised_forecaster,
)


def test_ais_phys_jepa_forward_and_loss() -> None:
    torch = pytest.importorskip("torch")
    config = AISWorldModelConfig(
        input_size=7,
        forecast_size=5,
        max_length=8,
        horizon_count=3,
        hidden_size=16,
        latent_size=8,
        layers=1,
        attention_heads=2,
        regularizer_slices=4,
    )
    model = build_ais_phys_jepa(config, use_physics=True, ema_momentum=0.99)
    batch = 4
    context = torch.randn(batch, 8, 7)
    lengths = torch.tensor([8, 7, 6, 5])
    ports = torch.tensor([1, 2, 3, 4])
    vessels = torch.tensor([1, 2, 1, 2])
    physics = torch.randn(batch, 3, 5)
    target_context = torch.randn(batch, 3, 8, 7)
    target_lengths = torch.full((batch, 3), 8)
    state, target, predicted, forecast = model(
        context,
        lengths,
        ports,
        vessels,
        physics,
        target_context,
        target_lengths,
    )
    loss, diagnostics = ais_jepa_loss(
        state,
        target,
        predicted,
        forecast,
        torch.randn(batch, 3, 5),
        config=config,
        regularizer="visreg",
        step=1,
    )
    assert tuple(predicted.shape) == (batch, 3, 8)
    assert tuple(forecast.shape) == (batch, 3, 5)
    assert torch.isfinite(loss)
    assert diagnostics["latent"] >= 0
    model.update_target()


@pytest.mark.parametrize("kind", ["gru", "transformer"])
def test_ais_supervised_forecaster(kind: str) -> None:
    torch = pytest.importorskip("torch")
    config = AISWorldModelConfig(
        hidden_size=16,
        latent_size=8,
        layers=1,
        attention_heads=2,
    )
    model = build_ais_supervised_forecaster(
        config, kind=kind, use_physics=True
    )
    prediction = model(
        torch.randn(2, 8, 7),
        torch.tensor([8, 6]),
        torch.tensor([1, 2]),
        torch.tensor([1, 2]),
        torch.randn(2, 3, 5),
    )
    assert tuple(prediction.shape) == (2, 3, 5)
    assert torch.isfinite(prediction).all()


def test_ais_trajectory_builder_keeps_future_out_of_context(tmp_path: Path) -> None:
    rows = []
    start = datetime(2025, 1, 1, tzinfo=UTC)
    for partition_index, partition in enumerate(("train", "validation", "test")):
        trip_id = f"trip-{partition}"
        for step in range(10):
            rows.append(
                {
                    "trip_id": trip_id,
                    "prediction_cutoff": start
                    + timedelta(days=partition_index, minutes=30 * step),
                    "partition": partition,
                    "port": "new_orleans",
                    "vessel_group": "cargo",
                    "distance_km": 100.0 - 5.0 * step,
                    "sog_knots": 10.0,
                    "course_error_degrees": 0.0,
                    "approach_speed_kmh": 10.0,
                    "minutes_since_previous": 30.0,
                    "direct_eta_hours": 5.0,
                    "remaining_hours": 5.0 - 0.5 * step,
                }
            )
    path = tmp_path / "prefixes.parquet"
    pl.DataFrame(rows).write_parquet(path)
    data = build_ais_trajectory_data(
        path,
        {
            "data": {
                "horizons_hours": [0.5, 1.0, 2.0],
                "max_sequence_length": 8,
                "min_context_events": 4,
                "target_tolerance_hours": 0.01,
            }
        },
    )
    train = data.partitions["train"]
    assert set(data.partitions) == {"train", "validation", "test"}
    assert set(train.trip_ids) == {"trip-train"}
    assert train.context_numeric.shape[1:] == (8, 7)
    assert train.target_context_numeric.shape[1:] == (3, 8, 7)
    assert train.target_state_original[0, 0, 0] < train.current_state_original[0, 0]


def test_paired_trip_bootstrap_preserves_grouped_improvement() -> None:
    target = np.zeros((4, 3, 5), dtype=np.float32)
    raw = np.zeros_like(target)
    hybrid = np.zeros_like(target)
    raw[..., 0] = 2.0
    hybrid[..., 0] = 1.0
    partition = SimpleNamespace(
        target_state_original=target,
        trip_ids=["trip-a", "trip-a", "trip-b", "trip-b"],
    )
    result = _paired_trip_bootstrap_improvement(
        partition, raw, hybrid, seed=42, samples=100
    )
    assert result["trips"] == 2
    assert result["relative_improvement_percent"] == pytest.approx(50.0)
    assert result["relative_improvement_ci95_percent"] == pytest.approx([50.0, 50.0])
