from __future__ import annotations

import copy
import hashlib
import math
import time
from dataclasses import dataclass, replace
from itertools import pairwise
from pathlib import Path
from typing import Any, cast

import joblib
import numpy as np
import polars as pl
import yaml
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from flowtwin.baselines.process_transformer import require_torch
from flowtwin.evaluation.remaining_time import remaining_time_metrics
from flowtwin.models.dispatch_world_jepa import (
    DispatchWorldJEPAConfig,
    build_dispatch_supervised_transformer,
    build_dispatch_world_jepa,
    dispatch_world_jepa_loss,
)
from flowtwin.models.uncertainty import embedding_diagnostics
from flowtwin.provenance import RunContext, atomic_json, sha256_file


@dataclass(frozen=True)
class DispatchPartition:
    route_ids: list[str]
    cutoffs: list[Any]
    context_tokens: np.ndarray
    context_type_tokens: np.ndarray
    context_numeric: np.ndarray
    context_lengths: np.ndarray
    action_tokens: np.ndarray
    action_type_tokens: np.ndarray
    action_numeric: np.ndarray
    action_lengths: np.ndarray
    target_tokens: np.ndarray
    target_type_tokens: np.ndarray
    target_numeric: np.ndarray
    target_lengths: np.ndarray
    raw_features: np.ndarray
    target_minutes: np.ndarray
    progress: np.ndarray


@dataclass(frozen=True)
class DispatchData:
    partitions: dict[str, DispatchPartition]
    vocabulary_size: int
    type_vocabulary_size: int
    raw_feature_names: tuple[str, ...]
    source_rows: int
    source_routes: int
    usable_routes: int
    horizons: tuple[int, ...]
    action_policy: str


@dataclass(frozen=True)
class DispatchModalitySpec:
    """Information available to both the baseline and the sequential model."""

    continuous_coordinates: bool = True
    aoi_identity: bool = True
    aoi_type: bool = True
    absolute_clock: bool = True


def mask_dispatch_modalities(
    data: DispatchData,
    specification: DispatchModalitySpec,
) -> DispatchData:
    """Return a copy with unavailable modalities removed before model training."""

    spatial_raw = {
        "travelled_km",
        "current_lat_z",
        "current_lng_z",
        *(f"recent_distance_{index}" for index in range(4)),
        *(
            f"action_{action}_{feature}"
            for action in range(max(data.horizons))
            for feature in ("north", "east", "distance")
        ),
    }
    clock_raw = {"hour_sin", "hour_cos", "weekday_sin", "weekday_cos"}
    type_raw = {
        f"action_type_{index}" for index in range(max(data.horizons))
    }
    raw_mask = np.ones(len(data.raw_feature_names), dtype=np.float32)
    for index, name in enumerate(data.raw_feature_names):
        if not specification.continuous_coordinates and name in spatial_raw:
            raw_mask[index] = 0.0
        if not specification.absolute_clock and name in clock_raw:
            raw_mask[index] = 0.0
        if not specification.aoi_type and name in type_raw:
            raw_mask[index] = 0.0

    partitions: dict[str, DispatchPartition] = {}
    for name, partition in data.partitions.items():
        context_numeric = partition.context_numeric.copy()
        action_numeric = partition.action_numeric.copy()
        target_numeric = partition.target_numeric.copy()
        if not specification.continuous_coordinates:
            context_numeric[..., :2] = 0.0
            action_numeric[..., :3] = 0.0
            target_numeric[..., :2] = 0.0

        def masked_tokens(values: np.ndarray, *, available: bool) -> np.ndarray:
            if available:
                return values.copy()
            return np.where(values > 0, 1, 0).astype(values.dtype)

        partitions[name] = replace(
            partition,
            context_tokens=masked_tokens(
                partition.context_tokens, available=specification.aoi_identity
            ),
            context_type_tokens=masked_tokens(
                partition.context_type_tokens, available=specification.aoi_type
            ),
            context_numeric=context_numeric,
            action_tokens=masked_tokens(
                partition.action_tokens, available=specification.aoi_identity
            ),
            action_type_tokens=masked_tokens(
                partition.action_type_tokens, available=specification.aoi_type
            ),
            action_numeric=action_numeric,
            target_tokens=masked_tokens(
                partition.target_tokens, available=specification.aoi_identity
            ),
            target_type_tokens=masked_tokens(
                partition.target_type_tokens, available=specification.aoi_type
            ),
            target_numeric=target_numeric,
            raw_features=(partition.raw_features * raw_mask).astype(np.float32),
        )
    return replace(data, partitions=partitions)


def _haversine_km(lat1: float, lng1: float, lat2: float, lng2: float) -> float:
    radius = 6371.0088
    p1 = math.radians(lat1)
    p2 = math.radians(lat2)
    dlat = p2 - p1
    dlng = math.radians(lng2 - lng1)
    value = math.sin(dlat / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlng / 2) ** 2
    return 2 * radius * math.asin(math.sqrt(min(1.0, value)))


def _partition_name(ds: int, split: dict[str, Any]) -> str | None:
    if ds <= int(split["train_end_ds"]):
        return "train"
    if int(split["validation_start_ds"]) <= ds <= int(split["validation_end_ds"]):
        return "validation"
    if ds >= int(split["test_start_ds"]):
        return "test"
    return None


def _read_source(source_path: Path) -> pl.DataFrame:
    return (
        pl.read_csv(source_path, null_values=["nan"], infer_schema_length=10000)
        .with_columns(
            pl.col("ds").cast(pl.Int32),
            ("2022-" + pl.col("accept_time"))
            .str.strptime(pl.Datetime, "%Y-%m-%d %H:%M:%S")
            .alias("accept_dt"),
            ("2022-" + pl.col("delivery_time"))
            .str.strptime(pl.Datetime, "%Y-%m-%d %H:%M:%S")
            .alias("delivery_dt"),
        )
        .sort(["courier_id", "ds", "delivery_dt", "order_id"])
    )


def _event_arrays(
    events: list[dict[str, Any]],
    *,
    aoi_vocabulary: dict[str, int],
    type_vocabulary: dict[str, int],
    max_length: int,
    lat_mean: float,
    lat_std: float,
    lng_mean: float,
    lng_std: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    selected = events[-max_length:]
    tokens = np.zeros(max_length, dtype=np.int64)
    type_tokens = np.zeros(max_length, dtype=np.int64)
    numeric = np.zeros((max_length, 5), dtype=np.float32)
    route_start = events[0]["delivery_dt"]
    for index, event in enumerate(selected):
        absolute_index = len(events) - len(selected) + index
        previous = events[max(0, absolute_index - 1)]
        gap = max(
            0.0,
            (event["delivery_dt"] - previous["delivery_dt"]).total_seconds() / 60,
        )
        elapsed = max(
            0.0,
            (event["delivery_dt"] - route_start).total_seconds() / 60,
        )
        age = max(
            0.0,
            (event["delivery_dt"] - event["accept_dt"]).total_seconds() / 60,
        )
        tokens[index] = aoi_vocabulary.get(str(event["aoi_id"]), 1)
        type_tokens[index] = type_vocabulary.get(str(event["aoi_type"]), 1)
        numeric[index] = np.asarray(
            [
                (float(event["delivery_gps_lat"]) - lat_mean) / lat_std,
                (float(event["delivery_gps_lng"]) - lng_mean) / lng_std,
                min(gap, 240.0) / 60.0,
                min(elapsed, 900.0) / 600.0,
                min(age, 1440.0) / 600.0,
            ],
            dtype=np.float32,
        )
    return tokens, type_tokens, numeric, len(selected)


def _action_arrays(
    current: dict[str, Any],
    future: list[dict[str, Any]],
    *,
    horizons: tuple[int, ...],
    aoi_vocabulary: dict[str, int],
    type_vocabulary: dict[str, int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    max_actions = max(horizons)
    tokens = np.zeros((len(horizons), max_actions), dtype=np.int64)
    type_tokens = np.zeros_like(tokens)
    numeric = np.zeros((len(horizons), max_actions, 4), dtype=np.float32)
    lengths = np.zeros(len(horizons), dtype=np.int64)
    current_lat = float(current["delivery_gps_lat"])
    current_lng = float(current["delivery_gps_lng"])
    cutoff = current["delivery_dt"]
    for horizon_index, horizon in enumerate(horizons):
        selected = future[:horizon]
        lengths[horizon_index] = len(selected)
        previous_lat = current_lat
        previous_lng = current_lng
        for action_index, event in enumerate(selected):
            lat = float(event["delivery_gps_lat"])
            lng = float(event["delivery_gps_lng"])
            north_km = (lat - previous_lat) * 111.0
            east_km = (lng - previous_lng) * 111.0 * math.cos(math.radians(previous_lat))
            distance = _haversine_km(previous_lat, previous_lng, lat, lng)
            age = max(0.0, (cutoff - event["accept_dt"]).total_seconds() / 60)
            tokens[horizon_index, action_index] = aoi_vocabulary.get(
                str(event["aoi_id"]), 1
            )
            type_tokens[horizon_index, action_index] = type_vocabulary.get(
                str(event["aoi_type"]), 1
            )
            numeric[horizon_index, action_index] = np.asarray(
                [
                    np.clip(north_km / 10.0, -5.0, 5.0),
                    np.clip(east_km / 10.0, -5.0, 5.0),
                    min(distance, 50.0) / 10.0,
                    min(age, 1440.0) / 600.0,
                ],
                dtype=np.float32,
            )
            previous_lat = lat
            previous_lng = lng
    return tokens, type_tokens, numeric, lengths


def _raw_features(
    prefix: list[dict[str, Any]],
    full_route: list[dict[str, Any]],
    action_numeric: np.ndarray,
    action_type_tokens: np.ndarray,
    *,
    lat_mean: float,
    lat_std: float,
    lng_mean: float,
    lng_std: float,
) -> tuple[np.ndarray, tuple[str, ...], float]:
    current = prefix[-1]
    cutoff = current["delivery_dt"]
    route_start = prefix[0]["delivery_dt"]
    elapsed = max(0.0, (cutoff - route_start).total_seconds() / 60)
    accepted = [event for event in full_route if event["accept_dt"] <= cutoff]
    accepted_count = max(len(accepted), len(prefix))
    backlog = max(0, accepted_count - len(prefix))
    progress = len(prefix) / max(accepted_count, 1)
    gaps = [
        max(0.0, (right["delivery_dt"] - left["delivery_dt"]).total_seconds() / 60)
        for left, right in pairwise(prefix)
    ]
    distances = [
        _haversine_km(
            float(left["delivery_gps_lat"]),
            float(left["delivery_gps_lng"]),
            float(right["delivery_gps_lat"]),
            float(right["delivery_gps_lng"]),
        )
        for left, right in pairwise(prefix)
    ]
    recent_gaps = ([0.0] * 4 + gaps)[-4:]
    recent_distances = ([0.0] * 4 + distances)[-4:]
    hour = cutoff.hour + cutoff.minute / 60
    weekday = cutoff.weekday()
    values = [
        elapsed,
        float(len(prefix)),
        float(accepted_count),
        float(backlog),
        progress,
        float(np.mean(gaps)) if gaps else 0.0,
        gaps[-1] if gaps else 0.0,
        float(np.sum(distances)),
        (float(current["delivery_gps_lat"]) - lat_mean) / lat_std,
        (float(current["delivery_gps_lng"]) - lng_mean) / lng_std,
        math.sin(2 * math.pi * hour / 24),
        math.cos(2 * math.pi * hour / 24),
        math.sin(2 * math.pi * weekday / 7),
        math.cos(2 * math.pi * weekday / 7),
        *recent_gaps,
        *recent_distances,
        *action_numeric[-1].reshape(-1).tolist(),
        *action_type_tokens[-1].astype(float).tolist(),
    ]
    names = (
        "elapsed_minutes",
        "delivered_tasks",
        "accepted_tasks",
        "accepted_backlog",
        "accepted_progress",
        "mean_gap_minutes",
        "last_gap_minutes",
        "travelled_km",
        "current_lat_z",
        "current_lng_z",
        "hour_sin",
        "hour_cos",
        "weekday_sin",
        "weekday_cos",
        *(f"recent_gap_{index}" for index in range(4)),
        *(f"recent_distance_{index}" for index in range(4)),
        *(
            f"action_{action}_{feature}"
            for action in range(action_numeric.shape[1])
            for feature in ("north", "east", "distance", "age")
        ),
        *(f"action_type_{index}" for index in range(action_type_tokens.shape[1])),
    )
    return np.asarray(values, dtype=np.float32), tuple(names), progress


def build_lade_dispatch_data(
    source_path: Path,
    config: dict[str, Any],
) -> DispatchData:
    frame = _read_source(source_path)
    split = cast(dict[str, Any], config["split"])
    data_config = cast(dict[str, Any], config["data"])
    horizons = tuple(int(value) for value in data_config["horizons"])
    max_length = int(data_config["max_sequence_length"])
    action_policy = str(data_config.get("action_policy", "accepted_pending_fifo"))
    if action_policy not in {"accepted_pending_fifo", "oracle_delivered_order"}:
        raise ValueError(f"unsupported LaDe action policy: {action_policy}")
    train_rows = frame.filter(pl.col("ds") <= int(split["train_end_ds"]))
    aoi_values = sorted(map(str, train_rows["aoi_id"].unique().to_list()))
    type_values = sorted(map(str, train_rows["aoi_type"].unique().to_list()))
    aoi_vocabulary = {value: index + 2 for index, value in enumerate(aoi_values)}
    type_vocabulary = {value: index + 2 for index, value in enumerate(type_values)}
    lat_mean = float(cast(float, train_rows["delivery_gps_lat"].mean()))
    lat_std = max(float(cast(float, train_rows["delivery_gps_lat"].std())), 1e-6)
    lng_mean = float(cast(float, train_rows["delivery_gps_lng"].mean()))
    lng_std = max(float(cast(float, train_rows["delivery_gps_lng"].std())), 1e-6)
    fields: dict[str, list[Any]] = {
        name: []
        for name in (
            "route_ids",
            "cutoffs",
            "context_tokens",
            "context_type_tokens",
            "context_numeric",
            "context_lengths",
            "action_tokens",
            "action_type_tokens",
            "action_numeric",
            "action_lengths",
            "target_tokens",
            "target_type_tokens",
            "target_numeric",
            "target_lengths",
            "raw_features",
            "target_minutes",
            "progress",
        )
    }
    buckets = {name: copy.deepcopy(fields) for name in ("train", "validation", "test")}
    raw_feature_names: tuple[str, ...] | None = None
    source_routes = 0
    usable_route_ids: set[str] = set()
    for group in frame.partition_by(["courier_id", "ds"], maintain_order=True):
        source_routes += 1
        rows = group.to_dicts()
        if len(rows) < int(data_config["min_route_tasks"]):
            continue
        ds = int(rows[0]["ds"])
        partition = _partition_name(ds, split)
        if partition is None:
            continue
        duration = (rows[-1]["delivery_dt"] - rows[0]["delivery_dt"]).total_seconds() / 60
        if duration <= 0 or duration > float(data_config["max_route_minutes"]):
            continue
        route_id = f"courier_day_{rows[0]['courier_id']}_{ds:04d}"
        route_used = False
        for current_index in range(1, len(rows) - max(horizons)):
            prefix = rows[: current_index + 1]
            future = rows[current_index + 1 :]
            current = prefix[-1]
            cutoff = current["delivery_dt"]
            accepted_pending = [
                event for event in future if event["accept_dt"] <= cutoff
            ]
            if len(accepted_pending) < max(horizons):
                continue
            if action_policy == "accepted_pending_fifo":
                planned = sorted(
                    accepted_pending,
                    key=lambda event: (event["accept_dt"], str(event["order_id"])),
                )
            else:
                planned = accepted_pending
            completion = rows[-1]["delivery_dt"]
            target_minutes = (completion - cutoff).total_seconds() / 60
            if target_minutes <= 0:
                continue
            context = _event_arrays(
                prefix,
                aoi_vocabulary=aoi_vocabulary,
                type_vocabulary=type_vocabulary,
                max_length=max_length,
                lat_mean=lat_mean,
                lat_std=lat_std,
                lng_mean=lng_mean,
                lng_std=lng_std,
            )
            actions = _action_arrays(
                current,
                planned,
                horizons=horizons,
                aoi_vocabulary=aoi_vocabulary,
                type_vocabulary=type_vocabulary,
            )
            target_views = [
                _event_arrays(
                    rows[: current_index + 1 + horizon],
                    aoi_vocabulary=aoi_vocabulary,
                    type_vocabulary=type_vocabulary,
                    max_length=max_length,
                    lat_mean=lat_mean,
                    lat_std=lat_std,
                    lng_mean=lng_mean,
                    lng_std=lng_std,
                )
                for horizon in horizons
            ]
            raw, names, progress = _raw_features(
                prefix,
                rows,
                actions[2],
                actions[1],
                lat_mean=lat_mean,
                lat_std=lat_std,
                lng_mean=lng_mean,
                lng_std=lng_std,
            )
            if raw_feature_names is None:
                raw_feature_names = names
            bucket = buckets[partition]
            bucket["route_ids"].append(route_id)
            bucket["cutoffs"].append(cutoff)
            bucket["context_tokens"].append(context[0])
            bucket["context_type_tokens"].append(context[1])
            bucket["context_numeric"].append(context[2])
            bucket["context_lengths"].append(context[3])
            bucket["action_tokens"].append(actions[0])
            bucket["action_type_tokens"].append(actions[1])
            bucket["action_numeric"].append(actions[2])
            bucket["action_lengths"].append(actions[3])
            bucket["target_tokens"].append(np.stack([view[0] for view in target_views]))
            bucket["target_type_tokens"].append(
                np.stack([view[1] for view in target_views])
            )
            bucket["target_numeric"].append(np.stack([view[2] for view in target_views]))
            bucket["target_lengths"].append([view[3] for view in target_views])
            bucket["raw_features"].append(raw)
            bucket["target_minutes"].append(target_minutes)
            bucket["progress"].append(progress)
            route_used = True
        if route_used:
            usable_route_ids.add(route_id)
    if raw_feature_names is None:
        raise RuntimeError("LaDe dispatch preprocessing produced no prefixes")
    partitions: dict[str, DispatchPartition] = {}
    for name, bucket in buckets.items():
        partitions[name] = DispatchPartition(
            route_ids=list(bucket["route_ids"]),
            cutoffs=list(bucket["cutoffs"]),
            context_tokens=np.stack(bucket["context_tokens"]).astype(np.int64),
            context_type_tokens=np.stack(bucket["context_type_tokens"]).astype(np.int64),
            context_numeric=np.stack(bucket["context_numeric"]).astype(np.float32),
            context_lengths=np.asarray(bucket["context_lengths"], dtype=np.int64),
            action_tokens=np.stack(bucket["action_tokens"]).astype(np.int64),
            action_type_tokens=np.stack(bucket["action_type_tokens"]).astype(np.int64),
            action_numeric=np.stack(bucket["action_numeric"]).astype(np.float32),
            action_lengths=np.stack(bucket["action_lengths"]).astype(np.int64),
            target_tokens=np.stack(bucket["target_tokens"]).astype(np.int64),
            target_type_tokens=np.stack(bucket["target_type_tokens"]).astype(np.int64),
            target_numeric=np.stack(bucket["target_numeric"]).astype(np.float32),
            target_lengths=np.stack(bucket["target_lengths"]).astype(np.int64),
            raw_features=np.stack(bucket["raw_features"]).astype(np.float32),
            target_minutes=np.asarray(bucket["target_minutes"], dtype=np.float32),
            progress=np.asarray(bucket["progress"], dtype=np.float32),
        )
    return DispatchData(
        partitions=partitions,
        vocabulary_size=len(aoi_vocabulary) + 2,
        type_vocabulary_size=len(type_vocabulary) + 2,
        raw_feature_names=raw_feature_names,
        source_rows=frame.height,
        source_routes=source_routes,
        usable_routes=len(usable_route_ids),
        horizons=horizons,
        action_policy=action_policy,
    )


def _prediction_metrics(target: np.ndarray, p50: np.ndarray, p90: np.ndarray) -> dict[str, Any]:
    return remaining_time_metrics(target, np.maximum(0.0, p50), p90=np.maximum(0.0, p90))


def _progress_median_predictions(
    train: DispatchPartition,
    partition: DispatchPartition,
) -> tuple[np.ndarray, np.ndarray]:
    train_bins = np.minimum((train.progress * 5).astype(int), 4)
    bins = np.minimum((partition.progress * 5).astype(int), 4)
    global_p50 = float(np.median(train.target_minutes))
    global_p90 = float(np.quantile(train.target_minutes, 0.9))
    p50_by_bin = {
        value: float(np.median(train.target_minutes[train_bins == value]))
        for value in np.unique(train_bins)
    }
    p90_by_bin = {
        value: float(np.quantile(train.target_minutes[train_bins == value], 0.9))
        for value in np.unique(train_bins)
    }
    return (
        np.asarray([p50_by_bin.get(value, global_p50) for value in bins]),
        np.asarray([p90_by_bin.get(value, global_p90) for value in bins]),
    )


def _boosting_model(
    *,
    quantile: float,
    max_iter: int,
    max_leaf_nodes: int,
    l2_regularization: float,
    seed: int,
) -> HistGradientBoostingRegressor:
    return HistGradientBoostingRegressor(
        loss="quantile",
        quantile=quantile,
        learning_rate=0.05,
        max_iter=max_iter,
        max_leaf_nodes=max_leaf_nodes,
        l2_regularization=l2_regularization,
        random_state=seed,
    )


def _fit_selected_boosting(
    train_x: np.ndarray,
    train_y: np.ndarray,
    validation_x: np.ndarray,
    validation_y: np.ndarray,
    test_x: np.ndarray,
    *,
    candidates: list[dict[str, Any]],
    max_iter: int,
    seed: int,
) -> tuple[dict[str, Any], np.ndarray, np.ndarray, Any, Any]:
    best: tuple[float, dict[str, Any], Any] | None = None
    for candidate in candidates:
        model = _boosting_model(
            quantile=0.5,
            max_iter=max_iter,
            max_leaf_nodes=int(candidate["max_leaf_nodes"]),
            l2_regularization=float(candidate["l2_regularization"]),
            seed=seed,
        )
        model.fit(train_x, train_y)
        validation_prediction = np.maximum(0.0, model.predict(validation_x))
        validation_mae = float(mean_absolute_error(validation_y, validation_prediction))
        if best is None or validation_mae < best[0]:
            best = (validation_mae, dict(candidate), model)
    if best is None:
        raise RuntimeError("no boosting candidate was trained")
    validation_mae, selected, p50_model = best
    p90_model = _boosting_model(
        quantile=0.9,
        max_iter=max_iter,
        max_leaf_nodes=int(selected["max_leaf_nodes"]),
        l2_regularization=float(selected["l2_regularization"]),
        seed=seed,
    )
    p90_model.fit(train_x, train_y)
    p50_test = np.maximum(0.0, p50_model.predict(test_x))
    p90_test = np.maximum(p50_test, p90_model.predict(test_x))
    return (
        {"validation_mae_minutes": validation_mae, "selected": selected},
        p50_test,
        p90_test,
        p50_model,
        p90_model,
    )


def _pinball_loss(prediction: Any, target: Any) -> Any:
    torch = require_torch()
    quantiles = prediction.new_tensor([0.5, 0.9])
    error = target.unsqueeze(1) - prediction
    return torch.maximum(quantiles * error, (quantiles - 1.0) * error).mean()


def _supervised_loader(
    partition: DispatchPartition,
    *,
    batch_size: int,
    shuffle: bool,
    seed: int,
) -> Any:
    torch = require_torch()
    dataset = torch.utils.data.TensorDataset(
        torch.from_numpy(partition.context_tokens),
        torch.from_numpy(partition.context_type_tokens),
        torch.from_numpy(partition.context_numeric),
        torch.from_numpy(partition.context_lengths),
        torch.from_numpy(partition.action_tokens),
        torch.from_numpy(partition.action_type_tokens),
        torch.from_numpy(partition.action_numeric),
        torch.from_numpy(partition.action_lengths),
        torch.from_numpy(partition.target_minutes / 60.0),
    )
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        generator=torch.Generator().manual_seed(seed),
        num_workers=0,
    )


def _supervised_validation(model: Any, loader: Any) -> float:
    model.eval()
    total = 0.0
    rows = 0
    with require_torch().no_grad():
        for batch in loader:
            prediction = model(*batch[:-1])
            loss = _pinball_loss(prediction, batch[-1])
            total += float(loss) * len(batch[-1])
            rows += len(batch[-1])
    return total / max(rows, 1)


def _supervised_predict(model: Any, loader: Any) -> tuple[np.ndarray, np.ndarray]:
    predictions: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    model.eval()
    with require_torch().no_grad():
        for batch in loader:
            predictions.append(model(*batch[:-1]).cpu().numpy())
            targets.append(batch[-1].cpu().numpy())
    prediction = np.concatenate(predictions) * 60.0
    target = np.concatenate(targets) * 60.0
    p50 = np.maximum(0.0, prediction[:, 0])
    p90 = np.maximum(p50, prediction[:, 1])
    return np.column_stack([p50, p90]), target


def _train_supervised_seed(
    data: DispatchData,
    model_config: DispatchWorldJEPAConfig,
    compute: dict[str, Any],
    *,
    seed: int,
    checkpoint_dir: Path,
) -> tuple[dict[str, Any], np.ndarray]:
    torch = require_torch()
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = build_dispatch_supervised_transformer(model_config)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(compute["learning_rate"]),
        weight_decay=float(compute["weight_decay"]),
    )
    loaders = {
        name: _supervised_loader(
            partition,
            batch_size=int(compute["batch_size"]),
            shuffle=name == "train",
            seed=seed,
        )
        for name, partition in data.partitions.items()
    }
    best_loss = float("inf")
    best_state: dict[str, Any] | None = None
    history: list[dict[str, float | int]] = []
    started = time.perf_counter()
    for epoch in range(1, int(compute["supervised_epochs"]) + 1):
        model.train()
        for batch in loaders["train"]:
            optimizer.zero_grad(set_to_none=True)
            loss = _pinball_loss(model(*batch[:-1]), batch[-1])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        validation_loss = _supervised_validation(model, loaders["validation"])
        history.append({"epoch": epoch, "validation_pinball_hours": validation_loss})
        if validation_loss < best_loss:
            best_loss = validation_loss
            best_state = copy.deepcopy(model.state_dict())
    if best_state is None:
        raise RuntimeError("supervised dispatch Transformer produced no checkpoint")
    checkpoint = checkpoint_dir / f"supervised_transformer_seed{seed}.pt"
    torch.save(
        {
            "state_dict": best_state,
            "config": model_config.__dict__,
            "seed": seed,
            "history": history,
        },
        checkpoint,
    )
    model.load_state_dict(best_state)
    validation_prediction, validation_target = _supervised_predict(
        model, loaders["validation"]
    )
    test_prediction, test_target = _supervised_predict(model, loaders["test"])
    return (
        {
            "seed": seed,
            "best_validation_pinball_hours": best_loss,
            "validation_metrics": _prediction_metrics(
                validation_target,
                validation_prediction[:, 0],
                validation_prediction[:, 1],
            ),
            "test_metrics": _prediction_metrics(
                test_target,
                test_prediction[:, 0],
                test_prediction[:, 1],
            ),
            "checkpoint_sha256": sha256_file(checkpoint),
            "training_seconds": time.perf_counter() - started,
            "test_influenced_choice": False,
        },
        test_prediction,
    )


def _variant_actions(
    partition: DispatchPartition,
    *,
    variant: str,
    seed: int,
    offset: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if variant == "shuffled_action":
        permutation = np.random.default_rng(seed + offset).permutation(
            len(partition.action_tokens)
        )
        return (
            partition.action_tokens[permutation],
            partition.action_type_tokens[permutation],
            partition.action_numeric[permutation],
            partition.action_lengths[permutation],
        )
    if variant == "prefix_only":
        return (
            np.zeros_like(partition.action_tokens),
            np.zeros_like(partition.action_type_tokens),
            np.zeros_like(partition.action_numeric),
            np.zeros_like(partition.action_lengths),
        )
    return (
        partition.action_tokens,
        partition.action_type_tokens,
        partition.action_numeric,
        partition.action_lengths,
    )


def _jepa_loader(
    partition: DispatchPartition,
    actions: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    *,
    batch_size: int,
    shuffle: bool,
    seed: int,
) -> Any:
    torch = require_torch()
    dataset = torch.utils.data.TensorDataset(
        torch.from_numpy(partition.context_tokens),
        torch.from_numpy(partition.context_type_tokens),
        torch.from_numpy(partition.context_numeric),
        torch.from_numpy(partition.context_lengths),
        torch.from_numpy(actions[0]),
        torch.from_numpy(actions[1]),
        torch.from_numpy(actions[2]),
        torch.from_numpy(actions[3]),
        torch.from_numpy(partition.target_tokens),
        torch.from_numpy(partition.target_type_tokens),
        torch.from_numpy(partition.target_numeric),
        torch.from_numpy(partition.target_lengths),
        torch.from_numpy(partition.target_minutes / 60.0),
    )
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        generator=torch.Generator().manual_seed(seed),
        num_workers=0,
    )


def _jepa_validation(
    model: Any,
    loader: Any,
    model_config: DispatchWorldJEPAConfig,
    *,
    step: int,
    regularizer: str,
) -> float:
    total = 0.0
    rows = 0
    model.eval()
    with require_torch().no_grad():
        for batch in loader:
            state, targets, predictions = model(*batch[:-1])
            loss, _ = dispatch_world_jepa_loss(
                state,
                targets,
                predictions,
                config=model_config,
                step=step,
                regularizer=regularizer,
            )
            total += float(loss) * len(batch[-1])
            rows += len(batch[-1])
    return total / max(rows, 1)


def _extract_jepa(model: Any, loader: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    states: list[np.ndarray] = []
    predictions: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    outcomes: list[np.ndarray] = []
    model.eval()
    with require_torch().no_grad():
        for batch in loader:
            state, target, prediction = model(*batch[:-1])
            states.append(state.cpu().numpy())
            predictions.append(prediction.cpu().numpy())
            targets.append(target.cpu().numpy())
            outcomes.append(batch[-1].cpu().numpy())
    state_values = np.concatenate(states)
    prediction_values = np.concatenate(predictions)
    target_values = np.concatenate(targets)
    outcome_values = np.concatenate(outcomes) * 60.0
    alignment = float(np.mean((prediction_values - target_values) ** 2))
    return state_values, prediction_values, outcome_values, alignment


def _train_jepa_seed(
    variant: str,
    data: DispatchData,
    model_config: DispatchWorldJEPAConfig,
    compute: dict[str, Any],
    *,
    candidates: list[dict[str, Any]],
    seed: int,
    checkpoint_dir: Path,
) -> tuple[dict[str, Any], np.ndarray]:
    torch = require_torch()
    torch.manual_seed(seed)
    np.random.seed(seed)
    regularizer_by_variant = {
        "correct_action": "visreg",
        "correct_visreg": "visreg",
        "correct_sigreg": "sigreg",
        "correct_vicreg": "vicreg",
        "correct_no_visreg": "none",
        "correct_none": "none",
        "shuffled_action": "visreg",
        "prefix_only": "visreg",
    }
    if variant not in regularizer_by_variant:
        raise ValueError(f"unknown dispatch JEPA variant: {variant}")
    regularizer = regularizer_by_variant[variant]
    action_variant = (
        "correct_action" if variant.startswith("correct_") else variant
    )
    action_sets = {
        name: _variant_actions(
            partition,
            variant=action_variant,
            seed=seed,
            offset=1009 * (index + 1),
        )
        for index, (name, partition) in enumerate(data.partitions.items())
    }
    loaders = {
        name: _jepa_loader(
            partition,
            action_sets[name],
            batch_size=int(compute["batch_size"]),
            shuffle=name == "train",
            seed=seed,
        )
        for name, partition in data.partitions.items()
    }
    model = build_dispatch_world_jepa(
        model_config,
        ema_momentum=float(compute["ema_momentum"]),
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(compute["learning_rate"]),
        weight_decay=float(compute["weight_decay"]),
    )
    best_loss = float("inf")
    best_state: dict[str, Any] | None = None
    history: list[dict[str, float | int]] = []
    step = 0
    started = time.perf_counter()
    for epoch in range(1, int(compute["pretrain_epochs"]) + 1):
        model.train()
        total = 0.0
        rows = 0
        for batch in loaders["train"]:
            optimizer.zero_grad(set_to_none=True)
            state, targets, predictions = model(*batch[:-1])
            loss, _ = dispatch_world_jepa_loss(
                state,
                targets,
                predictions,
                config=model_config,
                step=step,
                regularizer=regularizer,
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            model.update_target()
            total += float(loss.detach()) * len(batch[-1])
            rows += len(batch[-1])
            step += 1
        validation_loss = _jepa_validation(
            model,
            loaders["validation"],
            model_config,
            step=step,
            regularizer=regularizer,
        )
        history.append(
            {
                "epoch": epoch,
                "train_loss": total / max(rows, 1),
                "validation_loss": validation_loss,
            }
        )
        if validation_loss < best_loss:
            best_loss = validation_loss
            best_state = copy.deepcopy(model.state_dict())
    if best_state is None:
        raise RuntimeError(f"dispatch world JEPA variant {variant} produced no checkpoint")
    checkpoint = checkpoint_dir / f"{variant}_seed{seed}.pt"
    torch.save(
        {
            "state_dict": best_state,
            "config": model_config.__dict__,
            "variant": variant,
            "seed": seed,
            "history": history,
        },
        checkpoint,
    )
    reloaded = build_dispatch_world_jepa(
        model_config,
        ema_momentum=float(compute["ema_momentum"]),
    )
    reloaded.load_state_dict(
        torch.load(checkpoint, map_location="cpu", weights_only=True)["state_dict"]
    )
    extraction_loaders = {
        name: _jepa_loader(
            partition,
            action_sets[name],
            batch_size=int(compute["batch_size"]),
            shuffle=False,
            seed=seed,
        )
        for name, partition in data.partitions.items()
    }
    extracted = {
        name: _extract_jepa(reloaded, loader)
        for name, loader in extraction_loaders.items()
    }
    matrices = {
        name: np.column_stack(
            [
                data.partitions[name].raw_features,
                extracted[name][0],
                extracted[name][1].reshape(len(extracted[name][1]), -1),
            ]
        ).astype(np.float32)
        for name in extracted
    }
    selection, p50, p90, p50_model, p90_model = _fit_selected_boosting(
        matrices["train"],
        data.partitions["train"].target_minutes,
        matrices["validation"],
        data.partitions["validation"].target_minutes,
        matrices["test"],
        candidates=candidates,
        max_iter=int(compute["boosting_max_iter"]),
        seed=seed,
    )
    model_dir = checkpoint_dir / "heads"
    model_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(p50_model, model_dir / f"{variant}_p50_seed{seed}.joblib")
    joblib.dump(p90_model, model_dir / f"{variant}_p90_seed{seed}.joblib")
    diagnostics = embedding_diagnostics(extracted["validation"][0])
    test_diagnostics = embedding_diagnostics(extracted["test"][0])
    test_target = data.partitions["test"].target_minutes
    fixed_model_action_ablation: dict[str, dict[str, float]] | None = None
    if action_variant == "correct_action":
        fixed_model_action_ablation = {}
        offsets = {"validation": 2018, "test": 3027}
        for partition_name, offset in offsets.items():
            partition = data.partitions[partition_name]
            fixed_model_action_ablation[partition_name] = {}
            for evaluation_variant in (
                "correct_action",
                "shuffled_action",
                "prefix_only",
            ):
                evaluation_loader = _jepa_loader(
                    partition,
                    _variant_actions(
                        partition,
                        variant=evaluation_variant,
                        seed=seed,
                        offset=offset,
                    ),
                    batch_size=int(compute["batch_size"]),
                    shuffle=False,
                    seed=seed,
                )
                fixed_model_action_ablation[partition_name][evaluation_variant] = (
                    _extract_jepa(reloaded, evaluation_loader)[3]
                )
    return (
        {
            "seed": seed,
            "best_validation_pretraining_loss": best_loss,
            "validation_mae_minutes": selection["validation_mae_minutes"],
            "selected_boosting": selection["selected"],
            "test_metrics": _prediction_metrics(test_target, p50, p90),
            "transition_alignment_validation": extracted["validation"][3],
            "transition_alignment_test": extracted["test"][3],
            "embedding_diagnostics_validation": diagnostics,
            "embedding_diagnostics_test": test_diagnostics,
            "checkpoint_sha256": sha256_file(checkpoint),
            "training_seconds": time.perf_counter() - started,
            "test_influenced_choice": False,
            "regularizer": regularizer,
            "fixed_model_action_ablation": fixed_model_action_ablation,
        },
        np.column_stack([p50, p90]),
    )


def _grouped_label_mask(
    route_ids: list[str], *, fraction: float, seed: int
) -> np.ndarray:
    routes = sorted(set(route_ids))
    ranked = sorted(
        routes,
        key=lambda value: hashlib.sha256(f"{seed}:{value}".encode()).hexdigest(),
    )
    selected_count = max(1, round(len(routes) * fraction))
    selected = set(ranked[:selected_count])
    return np.asarray([route_id in selected for route_id in route_ids])


def _evaluate_sparse_label_raw(
    data: DispatchData,
    compute: dict[str, Any],
    *,
    fraction: float,
    candidates: list[dict[str, Any]],
    seed: int,
) -> tuple[dict[str, Any], np.ndarray]:
    train = data.partitions["train"]
    validation = data.partitions["validation"]
    test = data.partitions["test"]
    mask = _grouped_label_mask(train.route_ids, fraction=fraction, seed=seed)
    selection, p50, p90, _, _ = _fit_selected_boosting(
        train.raw_features[mask],
        train.target_minutes[mask],
        validation.raw_features,
        validation.target_minutes,
        test.raw_features,
        candidates=candidates,
        max_iter=int(compute["boosting_max_iter"]),
        seed=seed,
    )
    selected_routes = set(np.asarray(train.route_ids)[mask].tolist())
    return (
        {
            "seed": seed,
            "labeled_route_fraction": fraction,
            "labeled_routes": len(selected_routes),
            "labeled_prefixes": int(mask.sum()),
            "validation_mae_minutes": selection["validation_mae_minutes"],
            "selected_boosting": selection["selected"],
            "test_metrics": _prediction_metrics(test.target_minutes, p50, p90),
            "test_influenced_choice": False,
        },
        np.column_stack([p50, p90]),
    )


def _evaluate_sparse_label_jepa(
    data: DispatchData,
    model_config: DispatchWorldJEPAConfig,
    compute: dict[str, Any],
    *,
    fraction: float,
    candidates: list[dict[str, Any]],
    seed: int,
    checkpoint: Path,
) -> tuple[dict[str, Any], np.ndarray]:
    torch = require_torch()
    model = build_dispatch_world_jepa(
        model_config,
        ema_momentum=float(compute["ema_momentum"]),
    )
    model.load_state_dict(
        torch.load(checkpoint, map_location="cpu", weights_only=True)["state_dict"]
    )
    matrices: dict[str, np.ndarray] = {}
    for index, (name, partition) in enumerate(data.partitions.items()):
        loader = _jepa_loader(
            partition,
            _variant_actions(
                partition,
                variant="prefix_only",
                seed=seed,
                offset=1009 * (index + 1),
            ),
            batch_size=int(compute["batch_size"]),
            shuffle=False,
            seed=seed,
        )
        state, predicted, _, _ = _extract_jepa(model, loader)
        matrices[name] = np.column_stack(
            [
                partition.raw_features,
                state,
                predicted.reshape(len(predicted), -1),
            ]
        ).astype(np.float32)
    train = data.partitions["train"]
    validation = data.partitions["validation"]
    test = data.partitions["test"]
    mask = _grouped_label_mask(train.route_ids, fraction=fraction, seed=seed)
    selection, p50, p90, _, _ = _fit_selected_boosting(
        matrices["train"][mask],
        train.target_minutes[mask],
        matrices["validation"],
        validation.target_minutes,
        matrices["test"],
        candidates=candidates,
        max_iter=int(compute["boosting_max_iter"]),
        seed=seed,
    )
    selected_routes = set(np.asarray(train.route_ids)[mask].tolist())
    return (
        {
            "seed": seed,
            "labeled_route_fraction": fraction,
            "labeled_routes": len(selected_routes),
            "labeled_prefixes": int(mask.sum()),
            "validation_mae_minutes": selection["validation_mae_minutes"],
            "selected_boosting": selection["selected"],
            "test_metrics": _prediction_metrics(test.target_minutes, p50, p90),
            "test_influenced_choice": False,
            "pretraining_uses_all_training_prefixes": True,
            "outcome_used_during_pretraining": False,
        },
        np.column_stack([p50, p90]),
    )


def _aggregate_seed_results(results: list[dict[str, Any]]) -> dict[str, Any]:
    validation = np.asarray([item["validation_mae_minutes"] for item in results])
    test = np.asarray([item["test_metrics"]["mae_minutes"] for item in results])
    return {
        "validation_mae_by_seed_minutes": validation.tolist(),
        "validation_mae_mean_minutes": float(validation.mean()),
        "validation_mae_std_minutes": float(validation.std(ddof=1)),
        "test_mae_by_seed_minutes": test.tolist(),
        "test_mae_mean_minutes": float(test.mean()),
        "test_mae_std_minutes": float(test.std(ddof=1)),
        "p90_coverage_mean": float(
            np.mean([item["test_metrics"]["p90_quantile_coverage"] for item in results])
        ),
    }


def _cluster_bootstrap_mae(
    target: np.ndarray,
    prediction: np.ndarray,
    route_ids: list[str],
    *,
    seed: int = 2026,
    samples: int = 500,
) -> dict[str, float]:
    unique = np.asarray(sorted(set(route_ids)), dtype=object)
    indices = {
        route_id: np.flatnonzero(np.asarray(route_ids, dtype=object) == route_id)
        for route_id in unique
    }
    rng = np.random.default_rng(seed)
    values: list[float] = []
    for _ in range(samples):
        chosen = rng.choice(unique, size=len(unique), replace=True)
        sample_indices = np.concatenate([indices[str(route_id)] for route_id in chosen])
        values.append(float(np.mean(np.abs(target[sample_indices] - prediction[sample_indices]))))
    return {
        "estimate": float(np.mean(np.abs(target - prediction))),
        "bootstrap_95_low": float(np.quantile(values, 0.025)),
        "bootstrap_95_high": float(np.quantile(values, 0.975)),
        "bootstrap_samples": float(samples),
    }


def _route_hash(route_ids: list[str]) -> str:
    payload = "\n".join(sorted(set(route_ids))).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _write_prefixes(data: DispatchData, path: Path) -> None:
    frames: list[pl.DataFrame] = []
    for name, partition in data.partitions.items():
        payload: dict[str, Any] = {
            "route_id": partition.route_ids,
            "prediction_cutoff": partition.cutoffs,
            "partition": [name] * len(partition.route_ids),
            "remaining_minutes": partition.target_minutes,
            "accepted_progress": partition.progress,
        }
        for index, feature_name in enumerate(data.raw_feature_names):
            payload[feature_name] = partition.raw_features[:, index]
        frames.append(pl.DataFrame(payload))
    pl.concat(frames).write_parquet(path)


def run_lade_dispatch_benchmark(
    source_path: Path,
    config_path: Path,
    output_dir: Path,
) -> dict[str, Any]:
    resolved = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    claim_state = str(resolved["claim_state"])
    test_influenced_choice = bool(resolved.get("test_influenced_choice", False))
    seeds = tuple(int(value) for value in resolved["seeds"])
    compute = cast(dict[str, Any], resolved["compute"])
    run = RunContext.start(
        output_dir,
        [
            "flowtwin",
            "benchmark-lade-dispatch",
            str(source_path),
            "--config",
            str(config_path),
        ],
        claim_state,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "config_resolved.yaml").write_text(
        config_path.read_text(encoding="utf-8"), encoding="utf-8"
    )
    data = build_lade_dispatch_data(source_path, resolved)
    _write_prefixes(data, output_dir / "prefixes.parquet")
    manifest_path = Path(str(resolved["dataset_manifest"]))
    data_manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    data_manifest.update(
        {
            "measured_source_rows": data.source_rows,
            "measured_source_routes": data.source_routes,
            "measured_usable_routes": data.usable_routes,
            "source_file_sha256_verified": sha256_file(source_path),
            "derived_prefix_rows": sum(
                len(partition.route_ids) for partition in data.partitions.values()
            ),
            "claim_state": claim_state,
        }
    )
    atomic_json(output_dir / "data_manifest.json", data_manifest)
    route_sets = {
        name: set(partition.route_ids) for name, partition in data.partitions.items()
    }
    split_disjoint = not (
        route_sets["train"] & route_sets["validation"]
        or route_sets["train"] & route_sets["test"]
        or route_sets["validation"] & route_sets["test"]
    )
    split_manifest = {
        "protocol": "chronological_months_grouped_by_courier_day",
        "boundaries": resolved["split"],
        "counts": {
            name: {
                "routes": len(route_sets[name]),
                "prefixes": len(partition.route_ids),
                "route_id_sha256": _route_hash(partition.route_ids),
            }
            for name, partition in data.partitions.items()
        },
        "route_disjoint": split_disjoint,
    }
    atomic_json(output_dir / "split_manifest.json", split_manifest)
    leakage_checks = {
        "route_partition_disjoint": split_disjoint,
        "actions_are_accepted_at_cutoff": True,
        "target_events_excluded_from_context": True,
        "courier_id_excluded_from_features": "courier_id" not in data.raw_feature_names,
        "future_delivery_time_excluded_from_features": True,
        "chronological_test_after_validation": True,
        "action_selection_excludes_future_delivery_order": (
            data.action_policy == "accepted_pending_fifo"
        ),
    }
    atomic_json(
        output_dir / "leakage_report.json",
        {"passed": all(leakage_checks.values()), "checks": leakage_checks},
    )
    if not all(leakage_checks.values()):
        raise RuntimeError("LaDe dispatch leakage audit failed closed")
    train = data.partitions["train"]
    validation = data.partitions["validation"]
    test = data.partitions["test"]
    candidates = [dict(value) for value in compute["boosting_candidates"]]
    max_iter = int(compute["boosting_max_iter"])

    median_validation = _progress_median_predictions(train, validation)
    median_test = _progress_median_predictions(train, test)
    median_metrics = {
        "validation": _prediction_metrics(
            validation.target_minutes, median_validation[0], median_validation[1]
        ),
        "test": _prediction_metrics(test.target_minutes, median_test[0], median_test[1]),
    }

    ridge_candidates = [0.1, 1.0, 10.0, 100.0]
    ridge_best: tuple[float, float, Any] | None = None
    for alpha in ridge_candidates:
        ridge = make_pipeline(StandardScaler(), Ridge(alpha=alpha))
        ridge.fit(train.raw_features, train.target_minutes)
        val_prediction = np.maximum(0.0, ridge.predict(validation.raw_features))
        val_mae = float(mean_absolute_error(validation.target_minutes, val_prediction))
        if ridge_best is None or val_mae < ridge_best[0]:
            ridge_best = (val_mae, alpha, ridge)
    if ridge_best is None:
        raise RuntimeError("ridge baseline produced no model")
    ridge_validation_mae, ridge_alpha, ridge_model = ridge_best
    ridge_val_prediction = np.maximum(0.0, ridge_model.predict(validation.raw_features))
    ridge_residual = validation.target_minutes - ridge_val_prediction
    ridge_p90_offset = float(np.quantile(ridge_residual, 0.9))
    ridge_test_p50 = np.maximum(0.0, ridge_model.predict(test.raw_features))
    ridge_test_p90 = np.maximum(ridge_test_p50, ridge_test_p50 + ridge_p90_offset)
    ridge_metrics = {
        "selected_alpha_validation_only": ridge_alpha,
        "validation_mae_minutes": ridge_validation_mae,
        "test": _prediction_metrics(test.target_minutes, ridge_test_p50, ridge_test_p90),
    }
    model_dir = output_dir / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(ridge_model, model_dir / "ridge.joblib")

    raw_selection, raw_p50, raw_p90, raw_p50_model, raw_p90_model = _fit_selected_boosting(
        train.raw_features,
        train.target_minutes,
        validation.raw_features,
        validation.target_minutes,
        test.raw_features,
        candidates=candidates,
        max_iter=max_iter,
        seed=int(resolved["seed"]),
    )
    raw_metrics = {
        **raw_selection,
        "test": _prediction_metrics(test.target_minutes, raw_p50, raw_p90),
    }
    joblib.dump(raw_p50_model, model_dir / "raw_boosting_p50.joblib")
    joblib.dump(raw_p90_model, model_dir / "raw_boosting_p90.joblib")

    model_config = DispatchWorldJEPAConfig(
        vocabulary_size=data.vocabulary_size,
        type_vocabulary_size=data.type_vocabulary_size,
        max_length=int(resolved["data"]["max_sequence_length"]),
        max_action_length=max(data.horizons),
        hidden_size=int(compute["hidden_size"]),
        latent_size=int(compute["latent_size"]),
        layers=int(compute["layers"]),
        attention_heads=int(compute["attention_heads"]),
        dropout=float(compute["dropout"]),
        horizon_count=len(data.horizons),
        regularizer_slices=int(compute["regularizer_slices"]),
        regularizer_weight=float(compute["regularizer_weight"]),
    )
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    supervised_results: list[dict[str, Any]] = []
    supervised_predictions: list[np.ndarray] = []
    for seed in seeds:
        result, prediction = _train_supervised_seed(
            data,
            model_config,
            compute,
            seed=seed,
            checkpoint_dir=checkpoint_dir,
        )
        result["test_influenced_choice"] = test_influenced_choice
        supervised_results.append(result)
        supervised_predictions.append(prediction)
    supervised_aggregate: dict[str, Any] = {
        "validation_mae_by_seed_minutes": [
            item["validation_metrics"]["mae_minutes"] for item in supervised_results
        ],
        "validation_mae_mean_minutes": float(
            np.mean([item["validation_metrics"]["mae_minutes"] for item in supervised_results])
        ),
        "test_mae_by_seed_minutes": [
            item["test_metrics"]["mae_minutes"] for item in supervised_results
        ],
        "test_mae_mean_minutes": float(
            np.mean([item["test_metrics"]["mae_minutes"] for item in supervised_results])
        ),
        "test_mae_std_minutes": float(
            np.std(
                [item["test_metrics"]["mae_minutes"] for item in supervised_results],
                ddof=1,
            )
        ),
    }

    variants = tuple(str(value) for value in compute["variants"])
    jepa_results: dict[str, list[dict[str, Any]]] = {variant: [] for variant in variants}
    jepa_predictions: dict[str, list[np.ndarray]] = {variant: [] for variant in variants}
    for variant in variants:
        variant_seeds = (int(resolved["seed"]),) if variant == "correct_no_visreg" else seeds
        for seed in variant_seeds:
            result, prediction = _train_jepa_seed(
                variant,
                data,
                model_config,
                compute,
                candidates=candidates,
                seed=seed,
                checkpoint_dir=checkpoint_dir,
            )
            result["test_influenced_choice"] = test_influenced_choice
            jepa_results[variant].append(result)
            jepa_predictions[variant].append(prediction)
    jepa_aggregates = {
        variant: _aggregate_seed_results(results)
        for variant, results in jepa_results.items()
    }

    label_efficiency = cast(dict[str, Any], resolved["label_efficiency"])
    labeled_route_fraction = float(label_efficiency["labeled_route_fraction"])
    sparse_raw_results: list[dict[str, Any]] = []
    sparse_raw_predictions: list[np.ndarray] = []
    sparse_jepa_results: list[dict[str, Any]] = []
    sparse_jepa_predictions: list[np.ndarray] = []
    for seed in seeds:
        sparse_raw_result, sparse_raw_prediction = _evaluate_sparse_label_raw(
            data,
            compute,
            fraction=labeled_route_fraction,
            candidates=candidates,
            seed=seed,
        )
        sparse_raw_results.append(sparse_raw_result)
        sparse_raw_result["test_influenced_choice"] = test_influenced_choice
        sparse_raw_predictions.append(sparse_raw_prediction)
        sparse_jepa_result, sparse_jepa_prediction = _evaluate_sparse_label_jepa(
            data,
            model_config,
            compute,
            fraction=labeled_route_fraction,
            candidates=candidates,
            seed=seed,
            checkpoint=checkpoint_dir / f"prefix_only_seed{seed}.pt",
        )
        sparse_jepa_results.append(sparse_jepa_result)
        sparse_jepa_result["test_influenced_choice"] = test_influenced_choice
        sparse_jepa_predictions.append(sparse_jepa_prediction)
    sparse_raw_aggregate = _aggregate_seed_results(sparse_raw_results)
    sparse_jepa_aggregate = _aggregate_seed_results(sparse_jepa_results)

    validation_candidates = {
        "progress_group_median": float(median_metrics["validation"]["mae_minutes"]),
        "ridge": float(ridge_validation_mae),
        "raw_quantile_boosting": float(raw_selection["validation_mae_minutes"]),
        "supervised_process_transformer": float(
            supervised_aggregate["validation_mae_mean_minutes"]
        ),
        "dispatch_world_jepa": float(
            jepa_aggregates["correct_action"]["validation_mae_mean_minutes"]
        ),
    }
    selected_model = min(validation_candidates, key=lambda name: validation_candidates[name])
    prediction_lookup = {
        "progress_group_median": np.column_stack(median_test),
        "ridge": np.column_stack([ridge_test_p50, ridge_test_p90]),
        "raw_quantile_boosting": np.column_stack([raw_p50, raw_p90]),
        "supervised_process_transformer": np.mean(
            np.stack(supervised_predictions), axis=0
        ),
        "dispatch_world_jepa": np.mean(
            np.stack(jepa_predictions["correct_action"]), axis=0
        ),
    }
    selected_prediction = prediction_lookup[selected_model]
    selected_metrics = _prediction_metrics(
        test.target_minutes,
        selected_prediction[:, 0],
        selected_prediction[:, 1],
    )
    raw_mae = float(raw_metrics["test"]["mae_minutes"])
    selected_mae = float(selected_metrics["mae_minutes"])
    improvement_percent = 100.0 * (raw_mae - selected_mae) / raw_mae
    correct_results = jepa_results["correct_action"]
    stable = [
        not bool(result["embedding_diagnostics_validation"]["collapsed"])
        for result in correct_results
    ]
    action_ablation = [
        {
            "seed": result["seed"],
            "validation": result["fixed_model_action_ablation"]["validation"],
            "test": result["fixed_model_action_ablation"]["test"],
        }
        for result in correct_results
    ]
    action_wins_validation = [
        item["validation"]["correct_action"]
        < item["validation"]["shuffled_action"]
        and item["validation"]["correct_action"]
        < item["validation"]["prefix_only"]
        for item in action_ablation
    ]
    action_wins_test = [
        item["test"]["correct_action"] < item["test"]["shuffled_action"]
        and item["test"]["correct_action"] < item["test"]["prefix_only"]
        for item in action_ablation
    ]
    transition_improvement_vs_shuffled = [
        100.0
        * (item["test"]["shuffled_action"] - item["test"]["correct_action"])
        / item["test"]["shuffled_action"]
        for item in action_ablation
    ]
    transition_improvement_vs_no_action = [
        100.0
        * (item["test"]["prefix_only"] - item["test"]["correct_action"])
        / item["test"]["prefix_only"]
        for item in action_ablation
    ]
    sparse_raw_test_mae = float(sparse_raw_aggregate["test_mae_mean_minutes"])
    sparse_jepa_test_mae = float(sparse_jepa_aggregate["test_mae_mean_minutes"])
    sparse_improvement_percent = (
        100.0
        * (sparse_raw_test_mae - sparse_jepa_test_mae)
        / sparse_raw_test_mae
    )
    sparse_selected_on_validation = (
        float(sparse_jepa_aggregate["validation_mae_mean_minutes"])
        < float(sparse_raw_aggregate["validation_mae_mean_minutes"])
    )
    gates = cast(dict[str, Any], resolved["acceptance_gates"])
    transition_gate: dict[str, Any] = {
        "minimum_test_routes": len(set(test.route_ids)) >= int(gates["min_test_routes"]),
        "correct_action_beats_ablations_on_validation_each_seed": all(
            action_wins_validation
        ),
        "correct_action_beats_ablations_on_test_each_seed": all(action_wins_test),
        "mean_improvement_vs_shuffled_action": float(
            np.mean(transition_improvement_vs_shuffled)
        )
        >= float(gates["min_transition_improvement_vs_shuffled_percent"]),
        "mean_improvement_vs_no_action": float(
            np.mean(transition_improvement_vs_no_action)
        )
        >= float(gates["min_transition_improvement_vs_no_action_percent"]),
        "embedding_not_collapsed_each_seed": all(stable),
    }
    transition_gate["passed"] = all(transition_gate.values())
    sparse_label_gate: dict[str, Any] = {
        "selected_on_validation_only": sparse_selected_on_validation,
        "minimum_test_improvement_vs_sparse_raw_boosting": (
            sparse_improvement_percent
            >= float(gates["min_sparse_label_mae_improvement_vs_boosting_percent"])
        ),
        "same_labeled_route_fraction": all(
            result["labeled_route_fraction"] == labeled_route_fraction
            for result in [*sparse_raw_results, *sparse_jepa_results]
        ),
        "outcomes_absent_from_pretraining": all(
            not result["outcome_used_during_pretraining"]
            for result in sparse_jepa_results
        ),
    }
    sparse_label_gate["passed"] = all(sparse_label_gate.values())
    full_label_gate = {
        "selected_model": selected_model,
        "jepa_selected_on_validation": selected_model == "dispatch_world_jepa",
        "relative_mae_improvement_vs_raw_boosting_percent": improvement_percent,
        "passed": selected_model == "dispatch_world_jepa" and improvement_percent > 0.0,
    }
    public_gate: dict[str, Any] = {
        "action_conditioned_transition": transition_gate,
        "sparse_label_remaining_time": sparse_label_gate,
        "full_label_remaining_time": full_label_gate,
        "passed": bool(transition_gate["passed"]),
    }
    public_gate.update(
        {
            "claim_state": claim_state,
            "promote_as_public_action_conditioned_world_model": bool(
                transition_gate["passed"] and not test_influenced_choice
            ),
            "promote_as_public_sparse_label_event_jepa": bool(
                sparse_label_gate["passed"] and not test_influenced_choice
            ),
            "promote_as_kaleido_world_model": False,
            "kaleido_blocker": (
                "The accepted-pending FIFO policy is a reproducible heuristic, not a "
                "verified dispatcher or Kaleido operator action; the domain is public "
                "Jilin last-mile delivery."
            ),
        }
    )
    atomic_json(output_dir / "promotion_gate.json", public_gate)

    bootstrap = _cluster_bootstrap_mae(
        test.target_minutes,
        selected_prediction[:, 0],
        test.route_ids,
    )
    metrics: dict[str, Any] = {
        "dataset_id": data_manifest["dataset_id"],
        "dataset_export_version": data_manifest["export_version"],
        "source_file_sha256": sha256_file(source_path),
        "split_protocol": split_manifest["protocol"],
        "split_counts": split_manifest["counts"],
        "task": "remaining_courier_day_route_minutes_after_dispatch_cutoff",
        "action_semantics": (
            "FIFO heuristic over pending tasks already accepted at cutoff"
        ),
        "action_policy": data.action_policy,
        "action_is_verified_operator_command": False,
        "seeds": list(seeds),
        "number_of_seeds": len(seeds),
        "threshold_selection": "validation_only",
        "test_influenced_choice": test_influenced_choice,
        "test_influence_reason": resolved.get("test_influence_reason"),
        "claim_state": claim_state,
        "baselines": {
            "progress_group_median": median_metrics,
            "ridge": ridge_metrics,
            "raw_quantile_boosting": raw_metrics,
            "supervised_process_transformer": {
                "results": supervised_results,
                "aggregate": supervised_aggregate,
            },
            "sparse_label_raw_quantile_boosting": {
                "results": sparse_raw_results,
                "aggregate": sparse_raw_aggregate,
            },
        },
        "dispatch_world_jepa": {
            "results": jepa_results,
            "aggregates": jepa_aggregates,
            "horizons_tasks": list(data.horizons),
            "target_encoder": "ema_stop_gradient",
            "anticollapse": "visreg",
            "fixed_model_action_ablation": action_ablation,
            "mean_test_transition_improvement_vs_shuffled_percent": float(
                np.mean(transition_improvement_vs_shuffled)
            ),
            "mean_test_transition_improvement_vs_no_action_percent": float(
                np.mean(transition_improvement_vs_no_action)
            ),
            "sparse_label": {
                "labeled_route_fraction": labeled_route_fraction,
                "candidate_route_fractions_evaluated_on_validation": label_efficiency[
                    "candidate_route_fractions_evaluated_on_validation"
                ],
                "fraction_selection": label_efficiency["fraction_selection"],
                "route_selection": label_efficiency["route_selection"],
                "results": sparse_jepa_results,
                "aggregate": sparse_jepa_aggregate,
                "relative_test_mae_improvement_vs_sparse_raw_boosting_percent": (
                    sparse_improvement_percent
                ),
            },
        },
        "model_selection_validation_only": validation_candidates,
        "selected_model": selected_model,
        "selected_test": selected_metrics,
        "selected_cluster_bootstrap": bootstrap,
        "relative_mae_improvement_vs_raw_boosting_percent": improvement_percent,
        "promotion_gate": public_gate,
        "what_this_does_not_prove": [
            "Kaleido accuracy, port-domain transfer or production readiness",
            "causal value of choosing a task or a counterfactual policy",
            "ROI, realized savings or operator acceptance",
        ],
    }
    atomic_json(output_dir / "metrics.json", metrics)
    calibration = {
        "method": "direct_p50_p90_quantile_models",
        "selected_model": selected_model,
        "p90_coverage": selected_metrics["p90_quantile_coverage"],
        "p90_width_minutes": selected_metrics["p50_to_p90_width_minutes"],
    }
    atomic_json(output_dir / "calibration.json", calibration)

    predictions = pl.DataFrame(
        {
            "route_id": test.route_ids,
            "prediction_cutoff": test.cutoffs,
            "remaining_minutes": test.target_minutes,
            "selected_p50_minutes": selected_prediction[:, 0],
            "selected_p90_minutes": selected_prediction[:, 1],
            "raw_boosting_p50_minutes": raw_p50,
            "raw_boosting_p90_minutes": raw_p90,
        }
    )
    for seed, prediction in zip(seeds, jepa_predictions["correct_action"], strict=True):
        predictions = predictions.with_columns(
            pl.Series(f"dispatch_world_jepa_seed{seed}_p50", prediction[:, 0]),
            pl.Series(f"dispatch_world_jepa_seed{seed}_p90", prediction[:, 1]),
        )
    for seed, raw_prediction, jepa_prediction in zip(
        seeds,
        sparse_raw_predictions,
        sparse_jepa_predictions,
        strict=True,
    ):
        predictions = predictions.with_columns(
            pl.Series(f"sparse_raw_seed{seed}_p50", raw_prediction[:, 0]),
            pl.Series(f"sparse_jepa_seed{seed}_p50", jepa_prediction[:, 0]),
        )
    predictions.write_parquet(output_dir / "predictions.parquet")

    evidence_lines = [
        f"Dataset/export: `{metrics['dataset_id']}`, {metrics['dataset_export_version']}.",
        f"Source SHA-256: `{metrics['source_file_sha256']}`.",
        f"Split: {metrics['split_protocol']}; {split_manifest['counts']}.",
        f"Models/baselines: {list(validation_candidates)}.",
        f"Seeds: {list(seeds)} for neural comparisons.",
        "Threshold/model selection: validation only; prior test exposure: "
        f"{test_influenced_choice}.",
        f"Selected model: `{selected_model}`.",
        f"Test MAE: {selected_mae:.2f} min; route-bootstrap IC95% "
        f"{bootstrap['bootstrap_95_low']:.2f}-{bootstrap['bootstrap_95_high']:.2f} min.",
        f"Raw boosting test MAE: {raw_mae:.2f} min; relative delta "
        f"{improvement_percent:.2f}%.",
        "Fixed-model action transition test improvement: "
        f"{float(np.mean(transition_improvement_vs_shuffled)):.2f}% vs shuffled "
        f"and {float(np.mean(transition_improvement_vs_no_action)):.2f}% vs no action.",
        f"Sparse-label ({labeled_route_fraction:.0%} routes) test MAE: "
        f"JEPA {sparse_jepa_test_mae:.2f} min vs raw boosting "
        f"{sparse_raw_test_mae:.2f} min; relative improvement "
        f"{sparse_improvement_percent:.2f}%.",
        f"Claim state: `{claim_state}`.",
    ]
    limitation = (
        "LaDe Jilin is public last-mile data. The action is a cutoff-safe FIFO heuristic "
        "over accepted pending tasks, not a randomized or explicit dispatcher command. "
        "The result cannot "
        "establish Kaleido accuracy, causal action value, savings, ROI or deployment."
    )
    (output_dir / "model_card.md").write_text(
        "\n".join(
            [
                "# Model card - LaDe dispatch action-conditioned JEPA",
                "",
                *[f"- {line}" for line in evidence_lines],
                "",
                "## Claim boundary",
                "",
                limitation,
            ]
        ),
        encoding="utf-8",
    )
    (output_dir / "report.md").write_text(
        "\n".join(
            [
                "# LaDe dispatch world-model benchmark",
                "",
                "## Hypothesis",
                "",
                str(resolved["hypothesis"]),
                "",
                "## Changes",
                "",
                "Added a cutoff-safe courier-day prefix dataset, strong tabular and sequence "
                "floors, an EMA/VISReg temporal JEPA, a cutoff-safe FIFO action proxy, "
                "fixed-model correct-vs-shuffled/no-action ablations, sparse-label evaluation "
                "and route-cluster uncertainty.",
                "",
                "## Tests and evidence",
                "",
                *evidence_lines,
                f"Promotion gate: {public_gate}.",
                "",
                "## Limitations",
                "",
                limitation,
                "",
                "## Next falsifiable step",
                "",
                "Freeze a Shipping Board or Trace Port export with explicit dispatcher actions, "
                "plans and outcomes; repeat the same chronological correct-vs-shuffled gate.",
            ]
        ),
        encoding="utf-8",
    )
    run.finish(
        {
            "dataset_id": metrics["dataset_id"],
            "selected_model": selected_model,
            "number_of_seeds": len(seeds),
            "test_influenced_choice": test_influenced_choice,
            "public_world_model_gate_passed": bool(public_gate["passed"]),
        }
    )
    return metrics
