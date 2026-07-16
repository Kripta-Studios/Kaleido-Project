from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from flowtwin.features.prefix import build_prefix_dataset
from flowtwin.jepa_hybrid_training import _joined_frame
from flowtwin.models.contracts import EventJEPAConfig, VarEventJEPAConfig
from flowtwin.models.temporal_t_jepa import (
    build_temporal_t_jepa,
    temporal_t_jepa_loss,
)
from flowtwin.models.var_event_jepa import build_var_event_jepa, var_event_jepa_loss
from flowtwin.temporal_jepa_data import build_disjoint_event_jepa_data

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


def test_disjoint_targets_contain_only_unseen_future_events(tmp_path: Path) -> None:
    source = tmp_path / "log.csv"
    baseline = tmp_path / "baseline"
    baseline.mkdir()
    _write_log(source)
    build_prefix_dataset(source).frame.write_parquet(baseline / "prefixes.parquet")

    data = build_disjoint_event_jepa_data(source, baseline, max_length=8)
    for partition in data.partitions.values():
        assert np.all(partition.target_lengths[:, 0] == 1)
        assert np.all(partition.target_lengths[:, 1] <= 2)
        for row in range(len(partition.context.tokens)):
            context = set(partition.context.tokens[row]) - {0}
            for horizon in range(3):
                target = set(partition.target_tokens[row, horizon]) - {0}
                assert target
                assert context.isdisjoint(target)


def test_temporal_t_jepa_has_stopped_ema_teacher_and_finite_loss() -> None:
    config = EventJEPAConfig(
        vocabulary_size=12,
        max_length=8,
        hidden_size=16,
        latent_size=16,
        layers=1,
        attention_heads=4,
        dropout=0.0,
    )
    model = build_temporal_t_jepa(
        config,
        ema_momentum=0.5,
        register_token=False,
    )
    assert all(not parameter.requires_grad for parameter in model.target_encoder.parameters())
    context_parameter = next(model.context_encoder.parameters())
    target_parameter = next(model.target_encoder.parameters())
    target_before = target_parameter.detach().clone()
    with torch.no_grad():
        context_parameter.add_(1.0)
    model.update_target()
    assert torch.allclose(target_parameter, target_before + 0.5)

    batch = 6
    context, targets, predictions = model(
        torch.randint(2, 12, (batch, 8)),
        torch.full((batch,), 8),
        torch.rand(batch, 3),
        torch.randint(2, 12, (batch, 3, 8)),
        torch.full((batch, 3), 8),
        torch.rand(batch, 3, 3),
    )
    loss, parts = temporal_t_jepa_loss(
        context,
        targets,
        predictions,
        config=config,
        step=1,
        regularizer="sigreg",
    )
    assert torch.isfinite(loss)
    assert parts["alignment"] >= 0
    assert not targets.requires_grad


def test_var_event_jepa_loss_and_uncertainties_are_finite() -> None:
    config = VarEventJEPAConfig(
        vocabulary_size=12,
        max_length=8,
        hidden_size=16,
        latent_size=8,
        auxiliary_size=4,
        layers=1,
        attention_heads=4,
    )
    model = build_var_event_jepa(config)
    batch = 6
    context_tokens = torch.randint(2, 12, (batch, 8))
    context_lengths = torch.full((batch,), 8)
    context_numeric = torch.rand(batch, 3)
    target_tokens = torch.randint(2, 12, (batch, 3, 8))
    target_lengths = torch.full((batch, 3), 8)
    target_numeric = torch.rand(batch, 3, 3)
    outputs = model(
        context_tokens,
        context_lengths,
        context_numeric,
        target_tokens,
        target_lengths,
        target_numeric,
    )
    loss, parts = var_event_jepa_loss(
        outputs,
        context_tokens,
        context_numeric,
        target_tokens,
        target_numeric,
        config=config,
        kl_scale=1.0,
    )
    embedding, context_uncertainty, predictive_uncertainty = model.inference_embedding(
        context_tokens,
        context_lengths,
        context_numeric,
    )
    assert torch.isfinite(loss)
    assert parts["target_kl"] >= 0
    assert embedding.shape == (batch, config.latent_size)
    assert torch.all(context_uncertainty > 0)
    assert torch.all(predictive_uncertainty > 0)


def test_hybrid_join_is_one_to_one_and_preserves_embedding_features(
    tmp_path: Path,
) -> None:
    cutoffs = [datetime(2025, 1, 1, tzinfo=UTC), datetime(2025, 1, 2, tzinfo=UTC)]
    prefixes = pl.DataFrame(
        {
            "operation_id": ["a", "b"],
            "prediction_cutoff": cutoffs,
            "partition": ["train", "test"],
            "remaining_minutes": [10.0, 20.0],
        }
    )
    shared = {
        "operation_id": ["a", "b"],
        "prediction_cutoff": cutoffs,
        "partition": ["train", "test"],
        "remaining_minutes": [10.0, 20.0],
    }
    t_path = tmp_path / "t.parquet"
    v_path = tmp_path / "v.parquet"
    pl.DataFrame({**shared, "t_000": [0.1, 0.2]}).write_parquet(t_path)
    pl.DataFrame({**shared, "v_000": [0.3, 0.4]}).write_parquet(v_path)

    joined, t_features, v_features = _joined_frame(prefixes, t_path, v_path)
    assert joined.height == prefixes.height
    assert t_features == ["t_000"]
    assert v_features == ["v_000"]
