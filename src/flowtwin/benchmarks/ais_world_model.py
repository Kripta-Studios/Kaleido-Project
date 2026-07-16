from __future__ import annotations

import copy
import hashlib
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import joblib
import numpy as np
import polars as pl
import yaml
from sklearn.decomposition import PCA
from sklearn.ensemble import (
    ExtraTreesRegressor,
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
)
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import average_precision_score, mean_absolute_error
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from flowtwin.baselines.process_transformer import require_torch
from flowtwin.models.ais_phys_jepa import (
    AISWorldModelConfig,
    ais_jepa_loss,
    build_ais_phys_jepa,
    build_ais_supervised_forecaster,
)
from flowtwin.models.uncertainty import embedding_diagnostics
from flowtwin.provenance import RunContext, atomic_json, sha256_file

PORT_TOKENS = {"new_york": 1, "houston": 2, "los_angeles": 3, "new_orleans": 4}
VESSEL_TOKENS = {"cargo": 1, "tanker": 2}
CONTEXT_FEATURES = (
    "distance_km",
    "sog_knots",
    "course_cos",
    "course_sin",
    "approach_speed_kmh",
    "log_gap_minutes",
    "direct_eta_hours",
)
FORECAST_FEATURES = CONTEXT_FEATURES[:5]


@dataclass(frozen=True)
class AISTrajectoryPartition:
    trip_ids: list[str]
    cutoffs: list[Any]
    context_numeric: np.ndarray
    context_lengths: np.ndarray
    port_tokens: np.ndarray
    vessel_tokens: np.ndarray
    physics_forecast: np.ndarray
    target_context_numeric: np.ndarray
    target_context_lengths: np.ndarray
    target_state: np.ndarray
    target_state_original: np.ndarray
    physics_original: np.ndarray
    current_state_original: np.ndarray
    raw_features: np.ndarray
    remaining_hours: np.ndarray
    direct_eta_hours: np.ndarray


@dataclass(frozen=True)
class AISTrajectoryData:
    partitions: dict[str, AISTrajectoryPartition]
    horizons_hours: tuple[float, ...]
    context_mean: np.ndarray
    context_std: np.ndarray
    source_rows: int
    source_trips: int
    sample_trips: int


def _event_vector(row: dict[str, Any]) -> np.ndarray:
    angle = math.radians(float(row.get("course_error_degrees") or 180.0))
    return np.asarray(
        [
            float(row.get("distance_km") or 0.0),
            float(row.get("sog_knots") or 0.0),
            math.cos(angle),
            math.sin(angle),
            float(np.clip(float(row.get("approach_speed_kmh") or 0.0), -60.0, 60.0)),
            math.log1p(max(0.0, float(row.get("minutes_since_previous") or 0.0))),
            float(np.clip(float(row.get("direct_eta_hours") or 0.0), 0.0, 24.0)),
        ],
        dtype=np.float32,
    )


def _pad_sequence(values: list[np.ndarray], max_length: int) -> tuple[np.ndarray, int]:
    selected = values[-max_length:]
    output = np.zeros((max_length, len(CONTEXT_FEATURES)), dtype=np.float32)
    output[: len(selected)] = np.stack(selected)
    return output, len(selected)


def _target_indices(
    rows: list[dict[str, Any]],
    index: int,
    horizons: tuple[float, ...],
    tolerance_hours: float,
) -> list[int] | None:
    cutoff = rows[index]["prediction_cutoff"]
    selected: list[int] = []
    for horizon in horizons:
        candidates = [
            (
                abs((row["prediction_cutoff"] - cutoff).total_seconds() / 3600.0 - horizon),
                future_index,
            )
            for future_index, row in enumerate(rows[index + 1 :], index + 1)
        ]
        if not candidates:
            return None
        error, future_index = min(candidates)
        if error > tolerance_hours:
            return None
        selected.append(future_index)
    return selected if len(set(selected)) == len(selected) else None


def _physics_forecast(
    current: np.ndarray,
    horizons: tuple[float, ...],
) -> np.ndarray:
    distance, sog, course_cos, course_sin, approach_speed = current[:5]
    radial_speed = max(0.0, float(sog) * 1.852 * float(course_cos))
    return np.asarray(
        [
            [
                max(0.0, float(distance) - radial_speed * horizon),
                sog,
                course_cos,
                course_sin,
                approach_speed,
            ]
            for horizon in horizons
        ],
        dtype=np.float32,
    )


def build_ais_trajectory_data(
    prefix_path: Path,
    config: dict[str, Any],
) -> AISTrajectoryData:
    frame = pl.read_parquet(prefix_path).sort("trip_id", "prediction_cutoff")
    data_config = cast(dict[str, Any], config["data"])
    horizons = tuple(float(value) for value in data_config["horizons_hours"])
    max_length = int(data_config["max_sequence_length"])
    min_context = int(data_config["min_context_events"])
    tolerance = float(data_config["target_tolerance_hours"])
    raw_samples: dict[str, list[dict[str, Any]]] = {
        name: [] for name in ("train", "validation", "test")
    }
    sample_trip_ids: set[str] = set()
    for trip in frame.partition_by("trip_id", maintain_order=True):
        rows = trip.to_dicts()
        vectors = [_event_vector(row) for row in rows]
        for index in range(min_context - 1, len(rows)):
            target_indices = _target_indices(rows, index, horizons, tolerance)
            if target_indices is None:
                continue
            partition = str(rows[index]["partition"])
            if partition not in raw_samples:
                continue
            context, context_length = _pad_sequence(vectors[: index + 1], max_length)
            target_contexts: list[np.ndarray] = []
            target_lengths: list[int] = []
            target_states: list[np.ndarray] = []
            for target_index in target_indices:
                target_context, target_length = _pad_sequence(
                    vectors[: target_index + 1], max_length
                )
                target_contexts.append(target_context)
                target_lengths.append(target_length)
                target_states.append(vectors[target_index][:5])
            current = vectors[index]
            physics = _physics_forecast(current, horizons)
            trip_id = str(rows[index]["trip_id"])
            raw_samples[partition].append(
                {
                    "trip_id": trip_id,
                    "cutoff": rows[index]["prediction_cutoff"],
                    "context": context,
                    "context_length": context_length,
                    "port_token": PORT_TOKENS[str(rows[index]["port"])],
                    "vessel_token": VESSEL_TOKENS[str(rows[index]["vessel_group"])],
                    "physics": physics,
                    "target_context": np.stack(target_contexts),
                    "target_lengths": np.asarray(target_lengths),
                    "target_state": np.stack(target_states),
                    "current_state": current[:5],
                    "remaining_hours": float(rows[index]["remaining_hours"]),
                    "direct_eta_hours": float(rows[index]["direct_eta_hours"]),
                }
            )
            sample_trip_ids.add(trip_id)
    if not all(raw_samples.values()):
        raise RuntimeError("AIS world-model preprocessing produced an empty partition")

    train_values = np.concatenate(
        [sample["context"][: sample["context_length"]] for sample in raw_samples["train"]]
    )
    context_mean = train_values.mean(axis=0).astype(np.float32)
    context_std = np.maximum(train_values.std(axis=0), 1e-4).astype(np.float32)
    forecast_mean = context_mean[:5]
    forecast_std = context_std[:5]
    partitions: dict[str, AISTrajectoryPartition] = {}
    for name, samples in raw_samples.items():
        contexts = np.stack(
            [(sample["context"] - context_mean) / context_std for sample in samples]
        ).astype(np.float32)
        normalized_target_contexts = np.stack(
            [(sample["target_context"] - context_mean) / context_std for sample in samples]
        ).astype(np.float32)
        physics_original = np.stack([sample["physics"] for sample in samples]).astype(np.float32)
        target_original = np.stack([sample["target_state"] for sample in samples]).astype(
            np.float32
        )
        physics = ((physics_original - forecast_mean) / forecast_std).astype(np.float32)
        targets = ((target_original - forecast_mean) / forecast_std).astype(np.float32)
        port_tokens = np.asarray([sample["port_token"] for sample in samples], dtype=np.int64)
        vessel_tokens = np.asarray([sample["vessel_token"] for sample in samples], dtype=np.int64)
        raw_features = np.column_stack(
            [
                contexts.reshape(len(samples), -1),
                physics.reshape(len(samples), -1),
                np.eye(len(PORT_TOKENS) + 1, dtype=np.float32)[port_tokens, 1:],
                np.eye(len(VESSEL_TOKENS) + 1, dtype=np.float32)[vessel_tokens, 1:],
            ]
        ).astype(np.float32)
        partitions[name] = AISTrajectoryPartition(
            trip_ids=[sample["trip_id"] for sample in samples],
            cutoffs=[sample["cutoff"] for sample in samples],
            context_numeric=contexts,
            context_lengths=np.asarray(
                [sample["context_length"] for sample in samples], dtype=np.int64
            ),
            port_tokens=port_tokens,
            vessel_tokens=vessel_tokens,
            physics_forecast=physics,
            target_context_numeric=normalized_target_contexts,
            target_context_lengths=np.stack(
                [sample["target_lengths"] for sample in samples]
            ).astype(np.int64),
            target_state=targets,
            target_state_original=target_original,
            physics_original=physics_original,
            current_state_original=np.stack([sample["current_state"] for sample in samples]).astype(
                np.float32
            ),
            raw_features=raw_features,
            remaining_hours=np.asarray(
                [sample["remaining_hours"] for sample in samples], dtype=np.float32
            ),
            direct_eta_hours=np.asarray(
                [sample["direct_eta_hours"] for sample in samples], dtype=np.float32
            ),
        )
    return AISTrajectoryData(
        partitions=partitions,
        horizons_hours=horizons,
        context_mean=context_mean,
        context_std=context_std,
        source_rows=frame.height,
        source_trips=frame["trip_id"].n_unique(),
        sample_trips=len(sample_trip_ids),
    )


def _loader(
    partition: AISTrajectoryPartition,
    *,
    batch_size: int,
    shuffle: bool,
    seed: int,
) -> Any:
    torch = require_torch()
    dataset = torch.utils.data.TensorDataset(
        torch.from_numpy(partition.context_numeric),
        torch.from_numpy(partition.context_lengths),
        torch.from_numpy(partition.port_tokens),
        torch.from_numpy(partition.vessel_tokens),
        torch.from_numpy(partition.physics_forecast),
        torch.from_numpy(partition.target_context_numeric),
        torch.from_numpy(partition.target_context_lengths),
        torch.from_numpy(partition.target_state),
    )
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        generator=torch.Generator().manual_seed(seed),
        num_workers=0,
    )


def _denormalize(values: np.ndarray, data: AISTrajectoryData) -> np.ndarray:
    return np.asarray(values * data.context_std[:5] + data.context_mean[:5], dtype=np.float32)


def _forecast_metrics(
    partition: AISTrajectoryPartition,
    prediction: np.ndarray,
    horizons: tuple[float, ...],
    *,
    deviation_km: float,
) -> dict[str, Any]:
    target = partition.target_state_original
    distance_by_horizon = [
        float(mean_absolute_error(target[:, index, 0], prediction[:, index, 0]))
        for index in range(len(horizons))
    ]
    speed_by_horizon = [
        float(mean_absolute_error(target[:, index, 1], prediction[:, index, 1]))
        for index in range(len(horizons))
    ]
    last = len(horizons) - 1
    deviation_target = (
        target[:, last, 0] - partition.physics_original[:, last, 0] > deviation_km
    ).astype(int)
    deviation_score = prediction[:, last, 0] - partition.physics_original[:, last, 0]
    return {
        "distance_mae_km": float(np.mean(np.abs(target[..., 0] - prediction[..., 0]))),
        "distance_median_ae_km": float(np.median(np.abs(target[..., 0] - prediction[..., 0]))),
        "distance_mae_by_horizon_km": {
            str(horizon): value
            for horizon, value in zip(horizons, distance_by_horizon, strict=True)
        },
        "speed_mae_knots": float(np.mean(np.abs(target[..., 1] - prediction[..., 1]))),
        "speed_mae_by_horizon_knots": {
            str(horizon): value for horizon, value in zip(horizons, speed_by_horizon, strict=True)
        },
        "deviation_auprc": float(average_precision_score(deviation_target, deviation_score)),
        "deviation_prevalence": float(deviation_target.mean()),
        "rows": len(target),
    }


def _distance_conformal_metrics(
    data: AISTrajectoryData,
    predictions: dict[str, np.ndarray],
    *,
    coverage: float = 0.9,
) -> dict[str, Any]:
    validation_target = data.partitions["validation"].target_state_original[..., 0]
    test_target = data.partitions["test"].target_state_original[..., 0]
    residual = np.abs(validation_target - predictions["validation"][..., 0])
    quantiles = np.quantile(residual, coverage, axis=0, method="higher")
    test_error = np.abs(test_target - predictions["test"][..., 0])
    return {
        "method": "split_conformal_absolute_residual_validation",
        "nominal_coverage": coverage,
        "radius_by_horizon_km": {
            str(horizon): float(radius)
            for horizon, radius in zip(data.horizons_hours, quantiles, strict=True)
        },
        "test_coverage": float(np.mean(test_error <= quantiles)),
        "test_coverage_by_horizon": {
            str(horizon): float(np.mean(test_error[:, index] <= quantiles[index]))
            for index, horizon in enumerate(data.horizons_hours)
        },
        "mean_interval_width_km": float(np.mean(2.0 * quantiles)),
    }


def _paired_trip_bootstrap_improvement(
    partition: AISTrajectoryPartition,
    raw_prediction: np.ndarray,
    hybrid_prediction: np.ndarray,
    *,
    seed: int,
    samples: int = 2000,
) -> dict[str, Any]:
    target = partition.target_state_original[..., 0]
    raw_error = np.abs(target - raw_prediction[..., 0]).mean(axis=1)
    hybrid_error = np.abs(target - hybrid_prediction[..., 0]).mean(axis=1)
    routes = np.asarray(partition.trip_ids)
    unique_routes = np.asarray(sorted(set(partition.trip_ids)))
    raw_sums = np.asarray([raw_error[routes == route].sum() for route in unique_routes])
    hybrid_sums = np.asarray([hybrid_error[routes == route].sum() for route in unique_routes])
    counts = np.asarray([(routes == route).sum() for route in unique_routes])
    generator = np.random.default_rng(seed)
    improvements = np.empty(samples, dtype=float)
    for index in range(samples):
        selected = generator.integers(0, len(unique_routes), size=len(unique_routes))
        raw_mae = raw_sums[selected].sum() / counts[selected].sum()
        hybrid_mae = hybrid_sums[selected].sum() / counts[selected].sum()
        improvements[index] = 100.0 * (raw_mae - hybrid_mae) / raw_mae
    point_raw = float(raw_error.mean())
    point_hybrid = float(hybrid_error.mean())
    return {
        "method": "paired_trip_bootstrap",
        "samples": samples,
        "seed": seed,
        "trips": len(unique_routes),
        "raw_mae_km": point_raw,
        "hybrid_mae_km": point_hybrid,
        "relative_improvement_percent": 100.0 * (point_raw - point_hybrid) / point_raw,
        "relative_improvement_ci95_percent": np.quantile(improvements, [0.025, 0.975]).tolist(),
        "bootstrap_probability_improvement": float((improvements > 0.0).mean()),
    }


def _fit_boosting(
    data: AISTrajectoryData,
    compute: dict[str, Any],
    *,
    seed: int,
    output_dir: Path,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    train = data.partitions["train"]
    validation = data.partitions["validation"]
    candidates = [dict(value) for value in compute["boosting_candidates"]]
    best: tuple[float, dict[str, Any]] | None = None
    for candidate in candidates:
        values: list[float] = []
        for horizon in range(len(data.horizons_hours)):
            model = HistGradientBoostingRegressor(
                loss="absolute_error",
                learning_rate=0.05,
                max_iter=int(compute["boosting_max_iter"]),
                max_leaf_nodes=int(candidate["max_leaf_nodes"]),
                l2_regularization=float(candidate["l2_regularization"]),
                random_state=seed,
            ).fit(train.raw_features, train.target_state_original[:, horizon, 0])
            values.append(
                float(
                    mean_absolute_error(
                        validation.target_state_original[:, horizon, 0],
                        model.predict(validation.raw_features),
                    )
                )
            )
        score = float(np.mean(values))
        if best is None or score < best[0]:
            best = (score, candidate)
    if best is None:
        raise RuntimeError("AIS boosting selection produced no model")
    selected = best[1]
    predictions = {
        name: np.zeros((len(partition.trip_ids), len(data.horizons_hours), 5), dtype=np.float32)
        for name, partition in data.partitions.items()
    }
    models: list[Any] = []
    for horizon in range(len(data.horizons_hours)):
        for feature in range(5):
            model = HistGradientBoostingRegressor(
                loss="absolute_error",
                learning_rate=0.05,
                max_iter=int(compute["boosting_max_iter"]),
                max_leaf_nodes=int(selected["max_leaf_nodes"]),
                l2_regularization=float(selected["l2_regularization"]),
                random_state=seed,
            ).fit(
                train.raw_features,
                train.target_state_original[:, horizon, feature],
            )
            models.append(model)
            for name, partition in data.partitions.items():
                predictions[name][:, horizon, feature] = model.predict(partition.raw_features)
    output_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(models, output_dir / "trajectory_boosting.joblib")
    return {
        "selected_validation_only": selected,
        "validation_distance_mae_km": best[0],
    }, predictions


def _predict_supervised(model: Any, loader: Any, data: AISTrajectoryData) -> np.ndarray:
    values: list[np.ndarray] = []
    model.eval()
    with require_torch().no_grad():
        for batch in loader:
            values.append(model(*batch[:5]).cpu().numpy())
    return _denormalize(np.concatenate(values), data)


def _train_supervised(
    kind: str,
    data: AISTrajectoryData,
    model_config: AISWorldModelConfig,
    compute: dict[str, Any],
    *,
    seed: int,
    checkpoint_dir: Path,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    torch = require_torch()
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = build_ais_supervised_forecaster(model_config, kind=kind, use_physics=True)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(compute["learning_rate"]),
        weight_decay=float(compute["weight_decay"]),
    )
    loaders = {
        name: _loader(
            partition,
            batch_size=int(compute["batch_size"]),
            shuffle=name == "train",
            seed=seed,
        )
        for name, partition in data.partitions.items()
    }
    extraction_loaders = {
        name: _loader(
            partition,
            batch_size=int(compute["batch_size"]),
            shuffle=False,
            seed=seed,
        )
        for name, partition in data.partitions.items()
    }
    best = float("inf")
    best_state: dict[str, Any] | None = None
    started = time.perf_counter()
    for _ in range(int(compute["supervised_epochs"])):
        model.train()
        for batch in loaders["train"]:
            optimizer.zero_grad(set_to_none=True)
            prediction = model(*batch[:5])
            loss = torch.nn.functional.smooth_l1_loss(prediction, batch[-1])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        validation = _predict_supervised(model, extraction_loaders["validation"], data)
        score = float(
            np.mean(
                np.abs(
                    data.partitions["validation"].target_state_original[..., 0] - validation[..., 0]
                )
            )
        )
        if score < best:
            best = score
            best_state = copy.deepcopy(model.state_dict())
    if best_state is None:
        raise RuntimeError(f"AIS {kind} produced no checkpoint")
    model.load_state_dict(best_state)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = checkpoint_dir / f"supervised_{kind}_seed{seed}.pt"
    torch.save({"state_dict": best_state, "config": model_config.__dict__}, checkpoint)
    predictions = {
        name: _predict_supervised(model, loader, data)
        for name, loader in extraction_loaders.items()
    }
    return {
        "seed": seed,
        "validation_distance_mae_km": best,
        "checkpoint_sha256": sha256_file(checkpoint),
        "training_seconds": time.perf_counter() - started,
    }, predictions


def _extract_jepa(
    model: Any,
    loader: Any,
    data: AISTrajectoryData,
) -> tuple[np.ndarray, np.ndarray, float]:
    states: list[np.ndarray] = []
    forecasts: list[np.ndarray] = []
    alignments: list[np.ndarray] = []
    model.eval()
    with require_torch().no_grad():
        for batch in loader:
            state, target, predicted, forecast = model(*batch[:-1])
            states.append(state.cpu().numpy())
            forecasts.append(forecast.cpu().numpy())
            alignments.append(((predicted - target) ** 2).mean(dim=(1, 2)).cpu().numpy())
    return (
        np.concatenate(states),
        _denormalize(np.concatenate(forecasts), data),
        float(np.concatenate(alignments).mean()),
    )


def _train_jepa(
    variant: str,
    data: AISTrajectoryData,
    model_config: AISWorldModelConfig,
    compute: dict[str, Any],
    *,
    seed: int,
    checkpoint_dir: Path,
) -> tuple[dict[str, Any], dict[str, np.ndarray], dict[str, np.ndarray]]:
    torch = require_torch()
    torch.manual_seed(seed)
    np.random.seed(seed)
    use_physics = variant.startswith("phys_")
    regularizer = variant.split("_", maxsplit=1)[1]
    model = build_ais_phys_jepa(
        model_config,
        use_physics=use_physics,
        ema_momentum=float(compute["ema_momentum"]),
    )
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=float(compute["learning_rate"]),
        weight_decay=float(compute["weight_decay"]),
    )
    loaders = {
        name: _loader(
            partition,
            batch_size=int(compute["batch_size"]),
            shuffle=name == "train",
            seed=seed,
        )
        for name, partition in data.partitions.items()
    }
    extraction_loaders = {
        name: _loader(
            partition,
            batch_size=int(compute["batch_size"]),
            shuffle=False,
            seed=seed,
        )
        for name, partition in data.partitions.items()
    }
    best = float("inf")
    best_state: dict[str, Any] | None = None
    started = time.perf_counter()
    step = 0
    for _ in range(int(compute["pretrain_epochs"])):
        model.train()
        for batch in loaders["train"]:
            optimizer.zero_grad(set_to_none=True)
            state, target, predicted, forecast = model(*batch[:-1])
            loss, _ = ais_jepa_loss(
                state,
                target,
                predicted,
                forecast,
                batch[-1],
                config=model_config,
                regularizer=regularizer,
                step=step,
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            model.update_target()
            step += 1
        _, validation, _ = _extract_jepa(model, extraction_loaders["validation"], data)
        score = float(
            np.mean(
                np.abs(
                    data.partitions["validation"].target_state_original[..., 0] - validation[..., 0]
                )
            )
        )
        if score < best:
            best = score
            best_state = copy.deepcopy(model.state_dict())
    if best_state is None:
        raise RuntimeError(f"AIS JEPA variant {variant} produced no checkpoint")
    model.load_state_dict(best_state)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = checkpoint_dir / f"{variant}_seed{seed}.pt"
    torch.save(
        {
            "state_dict": best_state,
            "config": model_config.__dict__,
            "variant": variant,
        },
        checkpoint,
    )
    extracted = {
        name: _extract_jepa(model, loader, data) for name, loader in extraction_loaders.items()
    }
    result = {
        "seed": seed,
        "variant": variant,
        "regularizer": regularizer,
        "use_physics": use_physics,
        "validation_distance_mae_km": best,
        "transition_alignment_validation": extracted["validation"][2],
        "transition_alignment_test": extracted["test"][2],
        "embedding_diagnostics_validation": embedding_diagnostics(extracted["validation"][0]),
        "embedding_diagnostics_test": embedding_diagnostics(extracted["test"][0]),
        "checkpoint_sha256": sha256_file(checkpoint),
        "training_seconds": time.perf_counter() - started,
    }
    forecasts = {name: value[1] for name, value in extracted.items()}
    representations = {
        name: np.column_stack([value[0], value[1].reshape(len(value[0]), -1)]).astype(np.float32)
        for name, value in extracted.items()
    }
    return result, forecasts, representations


def _grouped_label_mask(
    route_ids: list[str],
    *,
    fraction: float,
    seed: int,
) -> np.ndarray:
    routes = sorted(set(route_ids))
    ranked = sorted(
        routes,
        key=lambda value: hashlib.sha256(f"{seed}:{value}".encode()).hexdigest(),
    )
    selected = set(ranked[: max(1, round(len(routes) * fraction))])
    return np.asarray([route_id in selected for route_id in route_ids])


def _fit_distance_head(
    data: AISTrajectoryData,
    matrices: dict[str, np.ndarray],
    compute: dict[str, Any],
    *,
    train_mask: np.ndarray,
    seed: int,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    train = data.partitions["train"]
    validation = data.partitions["validation"]
    candidates = [dict(value) for value in compute["boosting_candidates"]]
    best: tuple[float, dict[str, Any]] | None = None
    for candidate in candidates:
        values: list[float] = []
        for horizon in range(len(data.horizons_hours)):
            model = HistGradientBoostingRegressor(
                loss="absolute_error",
                learning_rate=0.05,
                max_iter=int(compute["boosting_max_iter"]),
                max_leaf_nodes=int(candidate["max_leaf_nodes"]),
                l2_regularization=float(candidate["l2_regularization"]),
                random_state=seed,
            ).fit(
                matrices["train"][train_mask],
                train.target_state_original[train_mask, horizon, 0],
            )
            values.append(
                float(
                    mean_absolute_error(
                        validation.target_state_original[:, horizon, 0],
                        model.predict(matrices["validation"]),
                    )
                )
            )
        score = float(np.mean(values))
        if best is None or score < best[0]:
            best = (score, candidate)
    if best is None:
        raise RuntimeError("AIS downstream distance head produced no model")
    predictions = {
        name: partition.physics_original.copy() for name, partition in data.partitions.items()
    }
    for horizon in range(len(data.horizons_hours)):
        model = HistGradientBoostingRegressor(
            loss="absolute_error",
            learning_rate=0.05,
            max_iter=int(compute["boosting_max_iter"]),
            max_leaf_nodes=int(best[1]["max_leaf_nodes"]),
            l2_regularization=float(best[1]["l2_regularization"]),
            random_state=seed,
        ).fit(
            matrices["train"][train_mask],
            train.target_state_original[train_mask, horizon, 0],
        )
        for name in predictions:
            predictions[name][:, horizon, 0] = model.predict(matrices[name])
    return {
        "selected_validation_only": best[1],
        "validation_distance_mae_km": best[0],
    }, predictions


def _fit_sparse_outcome_heads(
    data: AISTrajectoryData,
    eta_matrices: dict[str, np.ndarray],
    delay_matrices: dict[str, np.ndarray],
    compute: dict[str, Any],
    *,
    train_mask: np.ndarray,
    seed: int,
    delay_hours: float,
) -> dict[str, Any]:
    train = data.partitions["train"]
    validation = data.partitions["validation"]
    test = data.partitions["test"]
    sparse_config = cast(dict[str, Any], compute["sparse_heads"])
    eta_config = cast(dict[str, Any], sparse_config["eta"])
    delay_config = cast(dict[str, Any], sparse_config["delay"])
    if eta_config["family"] != "extra_trees":
        raise ValueError("the frozen sparse ETA head must be extra_trees")
    eta_model = ExtraTreesRegressor(
        n_estimators=int(eta_config["n_estimators"]),
        min_samples_leaf=int(eta_config["min_samples_leaf"]),
        max_features=float(eta_config["max_features"]),
        n_jobs=-1,
        random_state=seed,
    ).fit(eta_matrices["train"][train_mask], train.remaining_hours[train_mask])
    delay_train = (train.remaining_hours - train.direct_eta_hours > delay_hours).astype(int)
    if delay_config["family"] != "hist_gradient_boosting":
        raise ValueError("the frozen sparse delay head must be hist_gradient_boosting")
    classifier = HistGradientBoostingClassifier(
        loss="log_loss",
        learning_rate=float(delay_config["learning_rate"]),
        max_iter=int(delay_config["max_iter"]),
        max_leaf_nodes=int(delay_config["max_leaf_nodes"]),
        l2_regularization=float(delay_config["l2_regularization"]),
        random_state=seed,
    ).fit(delay_matrices["train"][train_mask], delay_train[train_mask])

    ridge = make_pipeline(StandardScaler(), Ridge(alpha=float(sparse_config["ridge_alpha"]))).fit(
        eta_matrices["train"][train_mask], train.remaining_hours[train_mask]
    )
    logistic = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            C=float(sparse_config["logistic_c"]),
            max_iter=2000,
            random_state=seed,
        ),
    ).fit(delay_matrices["train"][train_mask], delay_train[train_mask])

    def metrics(name: str, partition: AISTrajectoryPartition) -> dict[str, float]:
        eta_prediction = np.maximum(0.0, eta_model.predict(eta_matrices[name]))
        delay_target = (
            partition.remaining_hours - partition.direct_eta_hours > delay_hours
        ).astype(int)
        delay_probability = classifier.predict_proba(delay_matrices[name])[:, 1]
        return {
            "eta_mae_hours": float(mean_absolute_error(partition.remaining_hours, eta_prediction)),
            "delay_auprc": float(average_precision_score(delay_target, delay_probability)),
            "delay_prevalence": float(delay_target.mean()),
            "ridge_eta_mae_hours": float(
                mean_absolute_error(
                    partition.remaining_hours,
                    np.maximum(0.0, ridge.predict(eta_matrices[name])),
                )
            ),
            "logistic_delay_auprc": float(
                average_precision_score(
                    delay_target, logistic.predict_proba(delay_matrices[name])[:, 1]
                )
            ),
        }

    selected_routes = set(np.asarray(train.trip_ids)[train_mask].tolist())
    return {
        "head_policy_selected_on_development_validation": {
            "eta": eta_config,
            "delay": delay_config,
            "ridge_alpha": sparse_config["ridge_alpha"],
            "logistic_c": sparse_config["logistic_c"],
        },
        "labeled_routes": len(selected_routes),
        "labeled_samples": int(train_mask.sum()),
        "validation": metrics("validation", validation),
        "test": metrics("test", test),
    }


def _evaluate_jepa_downstream(
    data: AISTrajectoryData,
    representations: list[dict[str, np.ndarray]],
    compute: dict[str, Any],
    *,
    seeds: tuple[int, ...],
    labeled_route_fraction: float,
    delay_hours: float,
    deviation_km: float,
) -> dict[str, Any]:
    results: list[dict[str, Any]] = []
    raw_test_predictions: list[np.ndarray] = []
    hybrid_test_predictions: list[np.ndarray] = []
    for seed, per_partition in zip(seeds, representations, strict=True):
        raw = {name: partition.raw_features for name, partition in data.partitions.items()}
        hybrid = {
            name: np.column_stack([raw[name], per_partition[name]]).astype(np.float32)
            for name in raw
        }
        latent_size = int(compute["latent_size"])
        horizon_count = len(data.horizons_hours)
        sparse_config = cast(dict[str, Any], compute["sparse_heads"])
        pca_components = min(
            int(sparse_config["state_pca_components"]),
            latent_size,
            len(per_partition["train"]) - 1,
        )
        pca = PCA(n_components=pca_components, random_state=seed).fit(
            per_partition["train"][:, :latent_size]
        )
        eta_hybrid: dict[str, np.ndarray] = {}
        delay_hybrid: dict[str, np.ndarray] = {}
        eta_one_horizon: dict[str, np.ndarray] = {}
        delay_one_horizon: dict[str, np.ndarray] = {}
        for name, values in per_partition.items():
            state = values[:, :latent_size]
            forecast = values[:, latent_size:].reshape(
                len(values), horizon_count, len(FORECAST_FEATURES)
            )
            eta_hybrid[name] = np.column_stack(
                [raw[name], forecast[:, :, :2].reshape(len(values), -1)]
            ).astype(np.float32)
            delay_hybrid[name] = np.column_stack(
                [raw[name], pca.transform(state), forecast[:, :, 0]]
            ).astype(np.float32)
            eta_one_horizon[name] = np.column_stack([raw[name], forecast[:, -1, :2]]).astype(
                np.float32
            )
            delay_one_horizon[name] = np.column_stack(
                [raw[name], pca.transform(state), forecast[:, -1, 0]]
            ).astype(np.float32)
        full_mask = np.ones(len(data.partitions["train"].trip_ids), dtype=bool)
        label_mask = _grouped_label_mask(
            data.partitions["train"].trip_ids,
            fraction=labeled_route_fraction,
            seed=seed,
        )
        raw_selection, raw_predictions = _fit_distance_head(
            data, raw, compute, train_mask=full_mask, seed=seed
        )
        hybrid_selection, hybrid_predictions = _fit_distance_head(
            data, hybrid, compute, train_mask=full_mask, seed=seed
        )
        raw_sparse = _fit_sparse_outcome_heads(
            data,
            raw,
            raw,
            compute,
            train_mask=label_mask,
            seed=seed,
            delay_hours=delay_hours,
        )
        hybrid_sparse = _fit_sparse_outcome_heads(
            data,
            eta_hybrid,
            delay_hybrid,
            compute,
            train_mask=label_mask,
            seed=seed,
            delay_hours=delay_hours,
        )
        one_horizon_sparse = _fit_sparse_outcome_heads(
            data,
            eta_one_horizon,
            delay_one_horizon,
            compute,
            train_mask=label_mask,
            seed=seed,
            delay_hours=delay_hours,
        )
        raw_test = _forecast_metrics(
            data.partitions["test"],
            raw_predictions["test"],
            data.horizons_hours,
            deviation_km=deviation_km,
        )
        hybrid_test = _forecast_metrics(
            data.partitions["test"],
            hybrid_predictions["test"],
            data.horizons_hours,
            deviation_km=deviation_km,
        )
        raw_test_predictions.append(raw_predictions["test"])
        hybrid_test_predictions.append(hybrid_predictions["test"])
        results.append(
            {
                "seed": seed,
                "full_label_trajectory": {
                    "raw_validation_distance_mae_km": raw_selection["validation_distance_mae_km"],
                    "hybrid_validation_distance_mae_km": hybrid_selection[
                        "validation_distance_mae_km"
                    ],
                    "raw_test": raw_test,
                    "hybrid_test": hybrid_test,
                    "raw_conformal": _distance_conformal_metrics(data, raw_predictions),
                    "hybrid_conformal": _distance_conformal_metrics(data, hybrid_predictions),
                },
                "sparse_outcomes": {
                    "labeled_route_fraction": labeled_route_fraction,
                    "raw": raw_sparse,
                    "hybrid": hybrid_sparse,
                    "one_horizon_hybrid_ablation": one_horizon_sparse,
                    "pretraining_uses_remaining_hours": False,
                    "pretraining_uses_delay_label": False,
                    "feature_policy_selected_on_development_validation": {
                        "eta": "raw_plus_jepa_forecast_distance_and_speed",
                        "delay": "raw_plus_train_pca8_state_plus_jepa_forecast_distance",
                        "pca_fit": "all_unlabelled_training_states_only",
                    },
                },
            }
        )
    full_validation_raw = np.asarray(
        [item["full_label_trajectory"]["raw_validation_distance_mae_km"] for item in results]
    )
    full_validation_hybrid = np.asarray(
        [item["full_label_trajectory"]["hybrid_validation_distance_mae_km"] for item in results]
    )
    sparse_validation_raw = np.asarray(
        [item["sparse_outcomes"]["raw"]["validation"]["eta_mae_hours"] for item in results]
    )
    sparse_validation_hybrid = np.asarray(
        [item["sparse_outcomes"]["hybrid"]["validation"]["eta_mae_hours"] for item in results]
    )
    full_test_raw = np.asarray(
        [item["full_label_trajectory"]["raw_test"]["distance_mae_km"] for item in results]
    )
    full_test_hybrid = np.asarray(
        [item["full_label_trajectory"]["hybrid_test"]["distance_mae_km"] for item in results]
    )
    sparse_test_raw = np.asarray(
        [item["sparse_outcomes"]["raw"]["test"]["eta_mae_hours"] for item in results]
    )
    sparse_test_hybrid = np.asarray(
        [item["sparse_outcomes"]["hybrid"]["test"]["eta_mae_hours"] for item in results]
    )
    sparse_delay_validation_raw = np.asarray(
        [item["sparse_outcomes"]["raw"]["validation"]["delay_auprc"] for item in results]
    )
    sparse_delay_validation_hybrid = np.asarray(
        [item["sparse_outcomes"]["hybrid"]["validation"]["delay_auprc"] for item in results]
    )
    sparse_delay_test_raw = np.asarray(
        [item["sparse_outcomes"]["raw"]["test"]["delay_auprc"] for item in results]
    )
    sparse_delay_test_hybrid = np.asarray(
        [item["sparse_outcomes"]["hybrid"]["test"]["delay_auprc"] for item in results]
    )

    def sample_std(values: np.ndarray) -> float:
        return float(values.std(ddof=1) if len(values) > 1 else 0.0)

    return {
        "results": results,
        "_test_predictions": {
            "raw": raw_test_predictions,
            "hybrid": hybrid_test_predictions,
        },
        "aggregate": {
            "full_trajectory_validation_raw_mae_mean_km": float(full_validation_raw.mean()),
            "full_trajectory_validation_hybrid_mae_mean_km": float(full_validation_hybrid.mean()),
            "full_trajectory_validation_hybrid_mae_std_km": sample_std(full_validation_hybrid),
            "full_trajectory_validation_relative_improvement_percent": float(
                100.0
                * (full_validation_raw.mean() - full_validation_hybrid.mean())
                / full_validation_raw.mean()
            ),
            "full_trajectory_test_raw_mae_mean_km": float(full_test_raw.mean()),
            "full_trajectory_test_hybrid_mae_mean_km": float(full_test_hybrid.mean()),
            "full_trajectory_test_hybrid_mae_std_km": sample_std(full_test_hybrid),
            "full_trajectory_test_relative_improvement_percent": float(
                100.0 * (full_test_raw.mean() - full_test_hybrid.mean()) / full_test_raw.mean()
            ),
            "full_trajectory_test_raw_deviation_auprc_mean": float(
                np.mean(
                    [
                        item["full_label_trajectory"]["raw_test"]["deviation_auprc"]
                        for item in results
                    ]
                )
            ),
            "full_trajectory_test_hybrid_deviation_auprc_mean": float(
                np.mean(
                    [
                        item["full_label_trajectory"]["hybrid_test"]["deviation_auprc"]
                        for item in results
                    ]
                )
            ),
            "sparse_eta_validation_raw_mae_mean_hours": float(sparse_validation_raw.mean()),
            "sparse_eta_validation_hybrid_mae_mean_hours": float(sparse_validation_hybrid.mean()),
            "sparse_eta_validation_hybrid_mae_std_hours": sample_std(sparse_validation_hybrid),
            "sparse_eta_validation_relative_improvement_percent": float(
                100.0
                * (sparse_validation_raw.mean() - sparse_validation_hybrid.mean())
                / sparse_validation_raw.mean()
            ),
            "sparse_eta_test_raw_mae_mean_hours": float(sparse_test_raw.mean()),
            "sparse_eta_test_hybrid_mae_mean_hours": float(sparse_test_hybrid.mean()),
            "sparse_eta_test_hybrid_mae_std_hours": sample_std(sparse_test_hybrid),
            "sparse_eta_test_relative_improvement_percent": float(
                100.0
                * (sparse_test_raw.mean() - sparse_test_hybrid.mean())
                / sparse_test_raw.mean()
            ),
            "sparse_delay_validation_raw_auprc_mean": float(sparse_delay_validation_raw.mean()),
            "sparse_delay_validation_hybrid_auprc_mean": float(
                sparse_delay_validation_hybrid.mean()
            ),
            "sparse_delay_validation_hybrid_auprc_std": sample_std(sparse_delay_validation_hybrid),
            "sparse_delay_test_raw_auprc_mean": float(sparse_delay_test_raw.mean()),
            "sparse_delay_test_hybrid_auprc_mean": float(sparse_delay_test_hybrid.mean()),
            "sparse_delay_test_hybrid_auprc_std": sample_std(sparse_delay_test_hybrid),
        },
    }


def _route_hash(route_ids: list[str]) -> str:
    return hashlib.sha256("\n".join(sorted(set(route_ids))).encode()).hexdigest()


def run_ais_world_model_benchmark(
    prefix_path: Path,
    config_path: Path,
    output_dir: Path,
) -> dict[str, Any]:
    resolved = cast(dict[str, Any], yaml.safe_load(config_path.read_text(encoding="utf-8")))
    claim_state = str(resolved["claim_state"])
    run = RunContext.start(
        output_dir,
        [
            "flowtwin",
            "benchmark-ais-world-model",
            "--prefixes",
            str(prefix_path),
            "--config",
            str(config_path),
        ],
        claim_state,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "config_resolved.yaml").write_text(
        config_path.read_text(encoding="utf-8"), encoding="utf-8"
    )
    data = build_ais_trajectory_data(prefix_path, resolved)
    manifest_path = Path(str(resolved["dataset_manifest"]))
    manifest = cast(dict[str, Any], yaml.safe_load(manifest_path.read_text(encoding="utf-8")))
    atomic_json(
        output_dir / "data_manifest.json",
        {
            **manifest,
            "prefix_cache": str(prefix_path),
            "prefix_cache_sha256": sha256_file(prefix_path),
            "source_rows": data.source_rows,
            "source_trips": data.source_trips,
            "sample_trips": data.sample_trips,
            "claim_state": claim_state,
        },
    )
    route_sets = {name: set(partition.trip_ids) for name, partition in data.partitions.items()}
    split_disjoint = not (
        route_sets["train"] & route_sets["validation"]
        or route_sets["train"] & route_sets["test"]
        or route_sets["validation"] & route_sets["test"]
    )
    split_manifest = {
        "protocol": "fixed_chronological_future_grouped_by_arrival_trip",
        "counts": {
            name: {
                "trips": len(route_sets[name]),
                "samples": len(partition.trip_ids),
                "trip_id_sha256": _route_hash(partition.trip_ids),
            }
            for name, partition in data.partitions.items()
        },
        "trip_disjoint": split_disjoint,
    }
    atomic_json(output_dir / "split_manifest.json", split_manifest)
    test_influenced_choice = bool(resolved["test_influenced_choice"])
    test_influence_reason = resolved.get("test_influence_reason")
    test_influence_disclosure_consistent = (
        test_influenced_choice and bool(test_influence_reason)
    ) or (not test_influenced_choice and test_influence_reason in (None, ""))
    leakage = {
        "trip_partition_disjoint": split_disjoint,
        "context_ends_at_prediction_cutoff": True,
        "targets_are_future_states_only": True,
        "mmsi_excluded_from_features": True,
        "arrival_time_excluded_from_features": True,
        "test_influence_disclosure_consistent": test_influence_disclosure_consistent,
    }
    atomic_json(
        output_dir / "leakage_report.json",
        {"passed": all(leakage.values()), "checks": leakage},
    )
    if not all(leakage.values()):
        raise RuntimeError("AIS world-model leakage audit failed closed")

    compute = cast(dict[str, Any], resolved["compute"])
    model_config = AISWorldModelConfig(
        max_length=int(resolved["data"]["max_sequence_length"]),
        horizon_count=len(data.horizons_hours),
        hidden_size=int(compute["hidden_size"]),
        latent_size=int(compute["latent_size"]),
        layers=int(compute["layers"]),
        attention_heads=int(compute["attention_heads"]),
        dropout=float(compute["dropout"]),
        regularizer_slices=int(compute["regularizer_slices"]),
        regularizer_weight=float(compute["regularizer_weight"]),
        forecast_weight=float(compute["forecast_weight"]),
    )
    deviation_km = float(resolved["deviation_definition"]["shortfall_km"])
    test = data.partitions["test"]
    baseline_predictions = {
        "persistence": np.repeat(
            test.current_state_original[:, None, :], len(data.horizons_hours), axis=1
        ),
        "kinematic": test.physics_original,
    }
    boosting_selection, boosting_predictions = _fit_boosting(
        data,
        compute,
        seed=int(resolved["seed"]),
        output_dir=output_dir / "models",
    )
    baseline_predictions["trajectory_boosting"] = boosting_predictions["test"]

    seeds = tuple(int(value) for value in resolved["seeds"])
    checkpoint_dir = output_dir / "checkpoints"
    supervised: dict[str, list[dict[str, Any]]] = {"gru": [], "transformer": []}
    supervised_predictions: dict[str, list[np.ndarray]] = {"gru": [], "transformer": []}
    for kind in supervised:
        for seed in seeds:
            result, predictions = _train_supervised(
                kind,
                data,
                model_config,
                compute,
                seed=seed,
                checkpoint_dir=checkpoint_dir,
            )
            result["test_metrics"] = _forecast_metrics(
                test,
                predictions["test"],
                data.horizons_hours,
                deviation_km=deviation_km,
            )
            supervised[kind].append(result)
            supervised_predictions[kind].append(predictions["test"])

    variants = tuple(str(value) for value in compute["jepa_variants"])
    jepa: dict[str, list[dict[str, Any]]] = {variant: [] for variant in variants}
    jepa_predictions: dict[str, list[np.ndarray]] = {variant: [] for variant in variants}
    jepa_representations: dict[str, list[dict[str, np.ndarray]]] = {
        variant: [] for variant in variants
    }
    for variant in variants:
        for seed in seeds:
            result, predictions, representations = _train_jepa(
                variant,
                data,
                model_config,
                compute,
                seed=seed,
                checkpoint_dir=checkpoint_dir,
            )
            result["test_metrics"] = _forecast_metrics(
                test,
                predictions["test"],
                data.horizons_hours,
                deviation_km=deviation_km,
            )
            jepa[variant].append(result)
            jepa_predictions[variant].append(predictions["test"])
            jepa_representations[variant].append(representations)

    label_efficiency = cast(dict[str, Any], resolved["label_efficiency"])
    downstream = {
        variant: _evaluate_jepa_downstream(
            data,
            jepa_representations[variant],
            compute,
            seeds=seeds,
            labeled_route_fraction=float(label_efficiency["labeled_route_fraction"]),
            delay_hours=float(label_efficiency["delay_vs_direct_eta_hours"]),
            deviation_km=deviation_km,
        )
        for variant in variants
    }
    downstream_test_predictions = {
        variant: cast(dict[str, list[np.ndarray]], result.pop("_test_predictions"))
        for variant, result in downstream.items()
    }

    def aggregate(results: list[dict[str, Any]]) -> dict[str, Any]:
        validation = np.asarray([result["validation_distance_mae_km"] for result in results])
        test_values = np.asarray([result["test_metrics"]["distance_mae_km"] for result in results])
        return {
            "validation_distance_mae_by_seed_km": validation.tolist(),
            "validation_distance_mae_mean_km": float(validation.mean()),
            "test_distance_mae_by_seed_km": test_values.tolist(),
            "test_distance_mae_mean_km": float(test_values.mean()),
            "test_distance_mae_std_km": float(
                test_values.std(ddof=1) if len(test_values) > 1 else 0.0
            ),
            "test_deviation_auprc_mean": float(
                np.mean([result["test_metrics"]["deviation_auprc"] for result in results])
            ),
        }

    supervised_aggregates = {name: aggregate(results) for name, results in supervised.items()}
    jepa_aggregates = {name: aggregate(results) for name, results in jepa.items()}
    hybrid_validation_candidates = {
        variant: float(
            downstream[variant]["aggregate"]["full_trajectory_validation_hybrid_mae_mean_km"]
        )
        for variant in variants
    }
    selected_hybrid_variant = min(
        hybrid_validation_candidates, key=hybrid_validation_candidates.__getitem__
    )
    selected_downstream = downstream[selected_hybrid_variant]["aggregate"]
    selected_test_predictions = downstream_test_predictions[selected_hybrid_variant]
    paired_test_uncertainty = _paired_trip_bootstrap_improvement(
        test,
        np.mean(np.stack(selected_test_predictions["raw"]), axis=0),
        np.mean(np.stack(selected_test_predictions["hybrid"]), axis=0),
        seed=int(resolved["seed"]),
    )
    raw_validation_mae = float(selected_downstream["full_trajectory_validation_raw_mae_mean_km"])
    hybrid_validation_mae = float(
        selected_downstream["full_trajectory_validation_hybrid_mae_mean_km"]
    )
    product_gate = {
        "selected_variant_validation_only": selected_hybrid_variant,
        "minimum_test_trips": len(set(data.partitions["test"].trip_ids))
        >= int(resolved["acceptance_gates"].get("min_test_trips", 1)),
        "hybrid_improves_full_trajectory_validation": (hybrid_validation_mae < raw_validation_mae),
        "minimum_full_trajectory_validation_improvement": float(
            selected_downstream["full_trajectory_validation_relative_improvement_percent"]
        )
        >= float(
            resolved["acceptance_gates"][
                "min_relative_distance_mae_improvement_vs_boosting_percent"
            ]
        ),
        "minimum_sparse_eta_validation_improvement": float(
            selected_downstream["sparse_eta_validation_relative_improvement_percent"]
        )
        >= float(resolved["acceptance_gates"]["min_sparse_eta_improvement_percent"]),
        "hybrid_improves_sparse_delay_validation_auprc": float(
            selected_downstream["sparse_delay_validation_hybrid_auprc_mean"]
        )
        > float(selected_downstream["sparse_delay_validation_raw_auprc_mean"]),
        "representations_not_collapsed_each_seed": all(
            not bool(result["embedding_diagnostics_validation"]["collapsed"])
            for result in jepa[selected_hybrid_variant]
        ),
        "hybrid_improves_full_trajectory_validation_each_seed": all(
            float(item["full_label_trajectory"]["hybrid_validation_distance_mae_km"])
            < float(item["full_label_trajectory"]["raw_validation_distance_mae_km"])
            for item in downstream[selected_hybrid_variant]["results"]
        ),
        "hybrid_improves_full_trajectory_test_each_seed": all(
            float(item["full_label_trajectory"]["hybrid_test"]["distance_mae_km"])
            < float(item["full_label_trajectory"]["raw_test"]["distance_mae_km"])
            for item in downstream[selected_hybrid_variant]["results"]
        ),
        "minimum_full_trajectory_test_improvement": float(
            selected_downstream["full_trajectory_test_relative_improvement_percent"]
        )
        >= float(
            resolved["acceptance_gates"][
                "min_relative_distance_mae_improvement_vs_boosting_percent"
            ]
        ),
        "minimum_sparse_eta_test_improvement": float(
            selected_downstream["sparse_eta_test_relative_improvement_percent"]
        )
        >= float(resolved["acceptance_gates"]["min_sparse_eta_improvement_percent"]),
        "hybrid_improves_sparse_delay_test_auprc": float(
            selected_downstream["sparse_delay_test_hybrid_auprc_mean"]
        )
        > float(selected_downstream["sparse_delay_test_raw_auprc_mean"]),
    }
    product_gate["passed"] = all(
        value for key, value in product_gate.items() if key != "selected_variant_validation_only"
    )
    validation_candidates = {
        "trajectory_boosting": float(boosting_selection["validation_distance_mae_km"]),
        **{
            f"supervised_{name}": float(summary["validation_distance_mae_mean_km"])
            for name, summary in supervised_aggregates.items()
        },
        **{
            name: float(summary["validation_distance_mae_mean_km"])
            for name, summary in jepa_aggregates.items()
        },
    }
    selected = min(validation_candidates, key=validation_candidates.__getitem__)
    test_metrics = {
        name: _forecast_metrics(
            test,
            prediction,
            data.horizons_hours,
            deviation_km=deviation_km,
        )
        for name, prediction in baseline_predictions.items()
    }
    for name, ensemble_values in supervised_predictions.items():
        test_metrics[f"supervised_{name}"] = _forecast_metrics(
            test,
            np.mean(np.stack(ensemble_values), axis=0),
            data.horizons_hours,
            deviation_km=deviation_km,
        )
    for name, ensemble_values in jepa_predictions.items():
        test_metrics[name] = _forecast_metrics(
            test,
            np.mean(np.stack(ensemble_values), axis=0),
            data.horizons_hours,
            deviation_km=deviation_km,
        )
    metrics: dict[str, Any] = {
        "dataset_id": manifest["dataset_id"],
        "dataset_export_version": manifest["export_version"],
        "prefix_cache_sha256": sha256_file(prefix_path),
        "split_protocol": split_manifest["protocol"],
        "split_counts": split_manifest["counts"],
        "task": "multi_horizon_port_approach_state_and_deviation_prediction",
        "horizons_hours": list(data.horizons_hours),
        "deviation_definition": resolved["deviation_definition"],
        "models_and_baselines": {
            "persistence": test_metrics["persistence"],
            "kinematic": test_metrics["kinematic"],
            "trajectory_boosting": {
                **boosting_selection,
                "test": test_metrics["trajectory_boosting"],
            },
            "supervised": {
                name: {
                    "results": supervised[name],
                    "aggregate": supervised_aggregates[name],
                    "ensemble_test": test_metrics[f"supervised_{name}"],
                }
                for name in supervised
            },
            "jepa": {
                name: {
                    "results": jepa[name],
                    "aggregate": jepa_aggregates[name],
                    "ensemble_test": test_metrics[name],
                    "downstream_heads": downstream[name],
                }
                for name in jepa
            },
        },
        "model_selection_validation_only": validation_candidates,
        "selected_model": selected,
        "selected_test": test_metrics[selected],
        "product_candidate": {
            "model": f"trajectory_boosting_plus_{selected_hybrid_variant}",
            "selection_metric": "full_trajectory_validation_distance_mae_km",
            "validation_candidates": hybrid_validation_candidates,
            "selected_downstream": selected_downstream,
            "paired_test_uncertainty": paired_test_uncertainty,
            "gate": product_gate,
        },
        "seeds": list(seeds),
        "number_of_seeds": len(seeds),
        "label_efficiency": label_efficiency,
        "threshold_selection": "fixed physical 2h shortfall definition",
        "test_influenced_choice": test_influenced_choice,
        "test_influence_reason": test_influence_reason,
        "claim_state": claim_state,
        "promotion": {
            "kaleido": False,
            "clean_public_test": bool(product_gate["passed"] and not test_influenced_choice),
            "development_candidate": selected,
            "product_candidate": f"trajectory_boosting_plus_{selected_hybrid_variant}",
        },
        "what_this_does_not_prove": [
            "Kaleido or European port accuracy",
            (
                "a clean held-out result because February 1-7 was previously opened"
                if test_influenced_choice
                else "transfer beyond the frozen NOAA future holdout"
            ),
            "causal action value, ROI, savings or production readiness",
        ],
    }
    atomic_json(output_dir / "metrics.json", metrics)
    atomic_json(output_dir / "promotion_gate.json", product_gate)
    atomic_json(
        output_dir / "metric_uncertainty.json",
        {"paired_test": paired_test_uncertainty},
    )
    selected_conformal = [
        item["full_label_trajectory"]["hybrid_conformal"]
        for item in downstream[selected_hybrid_variant]["results"]
    ]
    atomic_json(
        output_dir / "calibration.json",
        {
            "method": "split_conformal_absolute_residual_validation",
            "nominal_coverage": 0.9,
            "selected_product_candidate": (f"trajectory_boosting_plus_{selected_hybrid_variant}"),
            "by_seed": selected_conformal,
            "mean_test_coverage": float(
                np.mean([item["test_coverage"] for item in selected_conformal])
            ),
            "mean_interval_width_km": float(
                np.mean([item["mean_interval_width_km"] for item in selected_conformal])
            ),
        },
    )
    prediction_frame = pl.DataFrame(
        {
            "trip_id": test.trip_ids,
            "prediction_cutoff": test.cutoffs,
            "actual_distance_h2_km": test.target_state_original[:, -1, 0],
            "physics_distance_h2_km": test.physics_original[:, -1, 0],
        }
    )
    selected_hybrid_prediction = np.mean(
        np.stack(downstream_test_predictions[selected_hybrid_variant]["hybrid"]),
        axis=0,
    )
    selected_raw_prediction = np.mean(
        np.stack(downstream_test_predictions[selected_hybrid_variant]["raw"]),
        axis=0,
    )
    prediction_frame = prediction_frame.with_columns(
        pl.Series("product_raw_distance_h2_km", selected_raw_prediction[:, -1, 0]),
        pl.Series(
            "product_hybrid_distance_h2_km",
            selected_hybrid_prediction[:, -1, 0],
        ),
    )
    for name in test_metrics:
        output_values = (
            baseline_predictions[name]
            if name in baseline_predictions
            else np.mean(
                np.stack(
                    supervised_predictions[name.removeprefix("supervised_")]
                    if name.startswith("supervised_")
                    else jepa_predictions[name]
                ),
                axis=0,
            )
        )
        prediction_frame = prediction_frame.with_columns(
            pl.Series(f"{name}_distance_h2_km", output_values[:, -1, 0])
        )
    prediction_frame.write_parquet(output_dir / "predictions.parquet")
    run_kind = "development" if test_influenced_choice else "clean future holdout"
    exposure_evidence = (
        "February 1-7 was previously opened; this run cannot support a clean-test claim."
        if test_influenced_choice
        else "The future test interval was opened once after validation-only selection; "
        "no model, threshold or regularizer was selected from test."
    )
    limitation = (
        "Development data is US AIS and its February partition was already examined "
        "by the ETA work."
        if test_influenced_choice
        else "The clean holdout is public US AIS; it does not establish accuracy or value "
        "on Kaleido or European-port data."
    )
    next_step = (
        "Freeze the validation-selected architecture and code commit, then download and "
        "open a new NOAA calendar interval once."
        if test_influenced_choice
        else "Replay the frozen hybrid read-only on a pseudonymized Kaleido Shipping "
        "Board/AIS export under pre-agreed shadow gates."
    )
    evidence = [
        f"Dataset/export: {manifest['dataset_id']} / {manifest['export_version']}.",
        f"Prefix SHA-256: {sha256_file(prefix_path)}.",
        f"Split: {split_manifest['protocol']}; {split_manifest['counts']}.",
        f"Models: {list(validation_candidates)}; seeds {list(seeds)}.",
        f"Selected on validation: {selected}.",
        f"{run_kind.title()} distance MAE: {test_metrics[selected]['distance_mae_km']:.3f} km.",
        "Product candidate selected on validation: trajectory boosting plus "
        f"{selected_hybrid_variant}; hybrid validation improvement "
        f"{selected_downstream['full_trajectory_validation_relative_improvement_percent']:.2f}%.",
        f"Product {run_kind} trajectory MAE: "
        f"{selected_downstream['full_trajectory_test_raw_mae_mean_km']:.3f} -> "
        f"{selected_downstream['full_trajectory_test_hybrid_mae_mean_km']:.3f} km "
        f"({selected_downstream['full_trajectory_test_relative_improvement_percent']:.2f}%).",
        "Paired trip-bootstrap improvement CI95: "
        f"{paired_test_uncertainty['relative_improvement_ci95_percent']}%; "
        f"P(improvement)={paired_test_uncertainty['bootstrap_probability_improvement']:.3f}.",
        f"Sparse ETA {run_kind}: "
        f"{selected_downstream['sparse_eta_test_raw_mae_mean_hours']:.3f} -> "
        f"{selected_downstream['sparse_eta_test_hybrid_mae_mean_hours']:.3f} h; "
        f"delay AUPRC {selected_downstream['sparse_delay_test_raw_auprc_mean']:.3f} -> "
        f"{selected_downstream['sparse_delay_test_hybrid_auprc_mean']:.3f}.",
        f"Predeclared product gate passed: {product_gate['passed']}.",
        exposure_evidence,
    ]
    (output_dir / "model_card.md").write_text(
        "\n".join(
            [
                f"# Model card - AIS Port Call Deviation Twin {run_kind}",
                "",
                *[f"- {line}" for line in evidence],
                "",
                "## Claim boundary",
                "",
                "Public US AIS diagnostic only. No Kaleido, causal, deployment, "
                "savings or ROI claim.",
            ]
        ),
        encoding="utf-8",
    )
    (output_dir / "report.md").write_text(
        "\n".join(
            [
                f"# AIS Port Call Deviation Twin {run_kind} report",
                "",
                "## Hypothesis",
                "",
                str(resolved["hypothesis"]),
                "",
                "## Changes",
                "",
                "Built cutoff-safe AIS sequences and compared persistence, kinematics, "
                "GBT, GRU, Transformer, plain JEPA and physics-informed JEPA at "
                "0.5/1/2 hour horizons.",
                "",
                "## Tests and evidence",
                "",
                *[f"- {line}" for line in evidence],
                "",
                "## Limitations",
                "",
                limitation,
                "",
                "## Next falsifiable step",
                "",
                next_step,
            ]
        ),
        encoding="utf-8",
    )
    run.finish(
        {
            "dataset_id": metrics["dataset_id"],
            "selected_model": selected,
            "number_of_seeds": len(seeds),
            "test_influenced_choice": test_influenced_choice,
            "claim_state": claim_state,
        }
    )
    return metrics
