from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from flowtwin.event_jepa_training import build_event_jepa_data
from flowtwin.features.prefix import build_prefix_dataset
from flowtwin.models.action_event_jepa import build_action_event_jepa
from flowtwin.models.contracts import EventJEPAConfig
from flowtwin.models.event_jepa import (
    build_event_jepa,
    event_jepa_loss,
    sigreg_loss,
    visreg_loss,
)
from flowtwin.models.uncertainty import embedding_diagnostics

torch = pytest.importorskip("torch")


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


def test_event_jepa_targets_advance_beyond_context(tmp_path: Path) -> None:
    source = tmp_path / "log.csv"
    run = tmp_path / "baseline"
    run.mkdir()
    _write_log(source)
    build_prefix_dataset(source).frame.write_parquet(run / "prefixes.parquet")
    data = build_event_jepa_data(source, run, max_length=8)
    train = data.partitions["train"]
    assert train.target_tokens.shape[1:] == (3, 8)
    assert np.all(train.target_lengths[:, 0] > train.context.lengths)
    assert np.all(train.target_lengths[:, 2] >= train.target_lengths[:, 0])


def test_sequence_context_uses_ordinal_when_timestamps_tie(tmp_path: Path) -> None:
    source = tmp_path / "ties.csv"
    run = tmp_path / "baseline"
    run.mkdir()
    start = datetime(2025, 1, 1, tzinfo=UTC)
    rows = []
    for case in range(20):
        for event in range(5):
            rows.append(
                {
                    "case:concept:name": f"case-{case:02d}",
                    "concept:name": f"A{event}",
                    "time:timestamp": (
                        start + timedelta(days=case, minutes=0 if event < 4 else 10)
                    ).isoformat(),
                }
            )
    pl.DataFrame(rows).write_csv(source)
    prefixes = build_prefix_dataset(source).frame.sort(
        ["operation_id", "prediction_cutoff"]
    )
    prefixes.write_parquet(run / "prefixes.parquet")
    data = build_event_jepa_data(source, run, max_length=8)
    for partition_name, partition in data.partitions.items():
        expected = prefixes.filter(pl.col("partition") == partition_name)[
            "prefix_events"
        ].to_numpy()
        assert np.array_equal(partition.context.lengths, expected)
        assert np.all(partition.target_lengths[:, 0] > partition.context.lengths)


def test_event_jepa_forward_loss_and_diagnostics_are_finite() -> None:
    config = EventJEPAConfig(vocabulary_size=12, max_length=8, latent_size=16, hidden_size=16)
    model = build_event_jepa(config)
    context_tokens = torch.randint(2, 12, (6, 8))
    context_lengths = torch.full((6,), 8)
    context_numeric = torch.rand(6, 3)
    target_tokens = torch.randint(2, 12, (6, 3, 8))
    target_lengths = torch.full((6, 3), 8)
    target_numeric = torch.rand(6, 3, 3)
    context, targets, predictions = model(
        context_tokens,
        context_lengths,
        context_numeric,
        target_tokens,
        target_lengths,
        target_numeric,
    )
    loss, parts = event_jepa_loss(
        context,
        targets,
        predictions,
        config=config,
        step=1,
    )
    assert torch.isfinite(loss)
    assert parts["alignment"] >= 0
    assert context.shape == (6, 16)
    diagnostics = embedding_diagnostics(context.detach().numpy())
    assert diagnostics["effective_rank"] > 0


def test_sigreg_penalizes_constant_embeddings() -> None:
    constant = torch.zeros(128, 16)
    gaussian = torch.randn(128, 16)
    assert float(sigreg_loss(constant, num_slices=32, seed=1)) > float(
        sigreg_loss(gaussian, num_slices=32, seed=1)
    )


def test_visreg_has_finite_nonzero_loss_and_gradient_at_collapse() -> None:
    collapsed = torch.zeros(128, 16, requires_grad=True)
    loss = visreg_loss(collapsed, num_slices=32, seed=1)
    loss.backward()
    assert torch.isfinite(loss)
    assert float(loss.detach()) > 0
    assert collapsed.grad is not None
    assert torch.isfinite(collapsed.grad).all()
    assert float(collapsed.grad.norm()) > 0


@pytest.mark.parametrize(
    "mode",
    ["correct_action", "current_prefix_only", "context_only", "action_only"],
)
def test_action_event_jepa_forward_is_finite_for_each_channel_ablation(mode: str) -> None:
    config = EventJEPAConfig(vocabulary_size=12, max_length=8, latent_size=16, hidden_size=16)
    model = build_action_event_jepa(config, action_count=6)
    batch = 6
    context, target, prediction = model(
        torch.randint(2, 12, (batch, 8)),
        torch.full((batch,), 8),
        torch.rand(batch, 3),
        torch.randint(0, 6, (batch,)),
        torch.rand(batch, 2),
        torch.randint(2, 12, (batch, 8)),
        torch.full((batch,), 8),
        torch.rand(batch, 3),
        mode,
    )
    assert context.shape == target.shape == prediction.shape == (batch, 16)
    assert torch.isfinite(prediction).all()
