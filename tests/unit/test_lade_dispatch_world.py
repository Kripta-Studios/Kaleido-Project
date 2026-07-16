from __future__ import annotations

import csv
from pathlib import Path

import pytest

from flowtwin.benchmarks.lade_dispatch import (
    DispatchModalitySpec,
    _grouped_label_mask,
    _jepa_loader,
    build_lade_dispatch_data,
    mask_dispatch_modalities,
)
from flowtwin.models.dispatch_world_jepa import (
    DispatchWorldJEPAConfig,
    build_dispatch_world_jepa,
    dispatch_world_jepa_loss,
)


def _write_lade_fixture(path: Path) -> None:
    columns = [
        "order_id",
        "region_id",
        "city",
        "courier_id",
        "lng",
        "lat",
        "aoi_id",
        "aoi_type",
        "accept_time",
        "accept_gps_time",
        "accept_gps_lng",
        "accept_gps_lat",
        "delivery_time",
        "delivery_gps_time",
        "delivery_gps_lng",
        "delivery_gps_lat",
        "ds",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for courier, date in ((1, "08-31"), (2, "09-15"), (3, "10-15")):
            for index in range(8):
                hour = 9 + index // 2
                minute = 30 * (index % 2)
                timestamp = f"{date} {hour:02d}:{minute:02d}:00"
                writer.writerow(
                    {
                        "order_id": courier * 100 + index,
                        "region_id": 31,
                        "city": "Jilin",
                        "courier_id": courier,
                        "lng": 126.5 + index * 0.001,
                        "lat": 43.8 + index * 0.001,
                        "aoi_id": 10 + index,
                        "aoi_type": 14,
                        "accept_time": f"{date} 08:00:00",
                        "accept_gps_time": f"{date} 08:00:00",
                        "accept_gps_lng": 126.5,
                        "accept_gps_lat": 43.8,
                        "delivery_time": timestamp,
                        "delivery_gps_time": timestamp,
                        "delivery_gps_lng": 126.5 + index * 0.001,
                        "delivery_gps_lat": 43.8 + index * 0.001,
                        "ds": date.replace("-", ""),
                    }
                )


def _config() -> dict[str, object]:
    return {
        "split": {
            "train_end_ds": 831,
            "validation_start_ds": 901,
            "validation_end_ds": 930,
            "test_start_ds": 1001,
        },
        "data": {
            "min_route_tasks": 6,
            "max_route_minutes": 900,
            "max_sequence_length": 8,
            "horizons": [1, 2, 4],
        },
    }


def test_lade_dispatch_builder_keeps_routes_grouped_and_actions_visible(tmp_path: Path) -> None:
    source = tmp_path / "lade.csv"
    _write_lade_fixture(source)
    data = build_lade_dispatch_data(source, _config())
    assert all(data.partitions[name].route_ids for name in ("train", "validation", "test"))
    route_sets = {
        name: set(partition.route_ids) for name, partition in data.partitions.items()
    }
    assert route_sets["train"].isdisjoint(route_sets["validation"])
    assert route_sets["train"].isdisjoint(route_sets["test"])
    assert "courier_id" not in data.raw_feature_names
    assert (data.partitions["test"].action_lengths[:, -1] == 4).all()
    assert data.action_policy == "accepted_pending_fifo"


def test_modality_mask_removes_coordinates_without_mutating_source(tmp_path: Path) -> None:
    source = tmp_path / "lade.csv"
    _write_lade_fixture(source)
    data = build_lade_dispatch_data(source, _config())
    original = data.partitions["train"]
    masked = mask_dispatch_modalities(
        data,
        DispatchModalitySpec(
            continuous_coordinates=False,
            aoi_identity=False,
            absolute_clock=False,
        ),
    ).partitions["train"]
    assert (masked.context_numeric[..., :2] == 0).all()
    assert (masked.action_numeric[..., :3] == 0).all()
    assert set(masked.context_tokens.reshape(-1)) <= {0, 1}
    for feature in (
        "travelled_km",
        "current_lat_z",
        "current_lng_z",
        "hour_sin",
        "hour_cos",
    ):
        index = data.raw_feature_names.index(feature)
        assert (masked.raw_features[:, index] == 0).all()
    assert not (original.context_numeric[..., :2] == 0).all()


def test_sparse_label_mask_selects_whole_routes_deterministically() -> None:
    route_ids = ["a", "a", "b", "b", "c", "c", "d", "d"]
    first = _grouped_label_mask(route_ids, fraction=0.5, seed=42)
    second = _grouped_label_mask(route_ids, fraction=0.5, seed=42)
    assert (first == second).all()
    assert int(first.sum()) == 4
    assert all(first[index] == first[index + 1] for index in range(0, 8, 2))


def test_jepa_extraction_loader_keeps_prefix_order(tmp_path: Path) -> None:
    pytest.importorskip("torch")
    source = tmp_path / "lade.csv"
    _write_lade_fixture(source)
    data = build_lade_dispatch_data(source, _config())
    partition = data.partitions["train"]
    actions = (
        partition.action_tokens,
        partition.action_type_tokens,
        partition.action_numeric,
        partition.action_lengths,
    )
    loader = _jepa_loader(
        partition,
        actions,
        batch_size=2,
        shuffle=False,
        seed=42,
    )
    first_batch = next(iter(loader))
    assert first_batch[0].numpy().tolist() == partition.context_tokens[:2].tolist()
    assert first_batch[-1].numpy().tolist() == pytest.approx(
        (partition.target_minutes[:2] / 60.0).tolist()
    )


def test_dispatch_world_jepa_forward_and_loss(tmp_path: Path) -> None:
    pytest.importorskip("torch")
    source = tmp_path / "lade.csv"
    _write_lade_fixture(source)
    data = build_lade_dispatch_data(source, _config())
    partition = data.partitions["train"]
    config = DispatchWorldJEPAConfig(
        vocabulary_size=data.vocabulary_size,
        type_vocabulary_size=data.type_vocabulary_size,
        max_length=8,
        max_action_length=4,
        hidden_size=16,
        latent_size=8,
        attention_heads=2,
        horizon_count=3,
        regularizer_slices=4,
    )
    model = build_dispatch_world_jepa(config, ema_momentum=0.99)
    import torch

    batch = slice(0, 2)
    state, targets, predictions = model(
        torch.from_numpy(partition.context_tokens[batch]),
        torch.from_numpy(partition.context_type_tokens[batch]),
        torch.from_numpy(partition.context_numeric[batch]),
        torch.from_numpy(partition.context_lengths[batch]),
        torch.from_numpy(partition.action_tokens[batch]),
        torch.from_numpy(partition.action_type_tokens[batch]),
        torch.from_numpy(partition.action_numeric[batch]),
        torch.from_numpy(partition.action_lengths[batch]),
        torch.from_numpy(partition.target_tokens[batch]),
        torch.from_numpy(partition.target_type_tokens[batch]),
        torch.from_numpy(partition.target_numeric[batch]),
        torch.from_numpy(partition.target_lengths[batch]),
    )
    loss, diagnostics = dispatch_world_jepa_loss(
        state,
        targets,
        predictions,
        config=config,
        step=1,
        regularizer="visreg",
    )
    assert tuple(predictions.shape) == (2, 3, 8)
    assert torch.isfinite(loss)
    assert diagnostics["total"] >= 0
    model.update_target()
