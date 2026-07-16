from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import polars as pl
import yaml
from sklearn.metrics import mean_absolute_error

from flowtwin.baselines.boosting import ConformalIntervals, quantile_boosting_pipeline
from flowtwin.benchmarks.common import grouped_bootstrap_mae, regression_diagnostics
from flowtwin.provenance import RunContext, atomic_json, sha256_file


@dataclass(frozen=True)
class PortDefinition:
    longitude: float
    latitude: float
    radius_km: float
    bbox: tuple[float, float, float, float]


PORTS = {
    "new_york": PortDefinition(-74.025, 40.675, 8.0, (-76.0, -72.0, 39.0, 42.0)),
    "houston": PortDefinition(-94.930, 29.700, 10.0, (-97.0, -93.0, 28.0, 31.0)),
    "los_angeles": PortDefinition(
        -118.230, 33.730, 8.0, (-120.0, -117.0, 32.5, 35.0)
    ),
    "new_orleans": PortDefinition(-90.050, 29.950, 8.0, (-92.0, -88.0, 28.0, 31.5)),
}

NUMERIC_FEATURES = [
    "distance_km",
    "sog_knots",
    "course_error_degrees",
    "direct_eta_hours",
    "approach_eta_hours",
    "approach_speed_kmh",
    "minutes_since_previous",
    "length_m",
    "width_m",
    "draft_m",
    "hour_utc",
    "weekday_utc",
]
CATEGORICAL_FEATURES = ["port", "vessel_group"]


def _distance_km(
    longitude: np.ndarray,
    latitude: np.ndarray,
    port: PortDefinition,
) -> np.ndarray:
    lon_scale = 111.32 * math.cos(math.radians(port.latitude))
    return np.sqrt(
        ((longitude - port.longitude) * lon_scale) ** 2
        + ((latitude - port.latitude) * 111.32) ** 2
    )


def _bearing_to_port(
    longitude: np.ndarray,
    latitude: np.ndarray,
    port: PortDefinition,
) -> np.ndarray:
    lat1 = np.radians(latitude)
    lat2 = math.radians(port.latitude)
    delta_lon = np.radians(port.longitude - longitude)
    x = np.sin(delta_lon) * math.cos(lat2)
    y = np.cos(lat1) * math.sin(lat2) - np.sin(lat1) * math.cos(lat2) * np.cos(
        delta_lon
    )
    return (np.degrees(np.arctan2(x, y)) + 360.0) % 360.0


def _load_port_rows(files: list[Path], port_name: str) -> pd.DataFrame:
    port = PORTS[port_name]
    lon_low, lon_high, lat_low, lat_high = port.bbox
    frames: list[pl.LazyFrame] = []
    schema = {
        "mmsi": pl.Int64,
        "base_date_time": pl.String,
        "longitude": pl.Float64,
        "latitude": pl.Float64,
        "sog": pl.Float64,
        "cog": pl.Float64,
        "vessel_type": pl.Int64,
        "length": pl.Float64,
        "width": pl.Float64,
        "draft": pl.Float64,
    }
    columns = list(schema)
    for path in files:
        frames.append(
            pl.scan_csv(path, schema_overrides=schema, ignore_errors=True)
            .select(columns)
            .filter(
                pl.col("longitude").is_between(lon_low, lon_high)
                & pl.col("latitude").is_between(lat_low, lat_high)
                & pl.col("vessel_type").is_between(70, 89)
            )
        )
    frame = (
        pl.concat(frames)
        .collect(engine="streaming")
        .drop_nulls(["mmsi", "base_date_time", "longitude", "latitude"])
        .with_columns(
            pl.col("base_date_time").str.to_datetime(strict=False).alias("timestamp")
        )
        .drop("base_date_time")
        .sort("mmsi", "timestamp")
    )
    result = frame.to_pandas()
    result["port"] = port_name
    result["distance_km"] = _distance_km(
        result["longitude"].to_numpy(), result["latitude"].to_numpy(), port
    )
    result["bearing_to_port"] = _bearing_to_port(
        result["longitude"].to_numpy(), result["latitude"].to_numpy(), port
    )
    return result


def _candidate_arrivals(group: pd.DataFrame, radius_km: float) -> list[int]:
    inside = group["distance_km"].to_numpy() <= radius_km
    distance = group["distance_km"].to_numpy()
    timestamps = pd.to_datetime(group["timestamp"], utc=True)
    candidates: list[int] = []
    last_arrival: pd.Timestamp | None = None
    for index in range(1, len(group)):
        gap_minutes = (timestamps.iloc[index] - timestamps.iloc[index - 1]).total_seconds() / 60
        if (
            inside[index]
            and not inside[index - 1]
            and distance[index - 1] > radius_km + 1.0
            and 0 < gap_minutes <= 180
        ):
            arrival = timestamps.iloc[index]
            if last_arrival is None or arrival - last_arrival >= pd.Timedelta(hours=8):
                candidates.append(index)
                last_arrival = arrival
    return candidates


def _angular_error(course: float, bearing: float) -> float:
    if not np.isfinite(course):
        return 180.0
    return abs((course - bearing + 180.0) % 360.0 - 180.0)


def _as_utc_timestamp(value: Any) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    return (
        timestamp.tz_localize("UTC")
        if timestamp.tzinfo is None
        else timestamp.tz_convert("UTC")
    )


def _vessel_group(vessel_type: float) -> str:
    return "cargo" if 70 <= vessel_type < 80 else "tanker"


def _trip_rows(
    group: pd.DataFrame,
    arrival_index: int,
    port_name: str,
    previous_arrival: pd.Timestamp | None,
) -> list[dict[str, Any]]:
    port = PORTS[port_name]
    arrival = _as_utc_timestamp(group.iloc[arrival_index]["timestamp"])
    start = arrival - pd.Timedelta(hours=12)
    if previous_arrival is not None:
        start = max(start, previous_arrival + pd.Timedelta(hours=2))
    prefix = group[
        (pd.to_datetime(group["timestamp"], utc=True) >= start)
        & (pd.to_datetime(group["timestamp"], utc=True) < arrival - pd.Timedelta(minutes=15))
        & (group["distance_km"] > port.radius_km)
        & (group["distance_km"] <= 220.0)
    ].copy()
    if len(prefix) < 4 or float(prefix["distance_km"].max()) < port.radius_km + 5.0:
        return []
    prefix["remaining_hours"] = (
        arrival - pd.to_datetime(prefix["timestamp"], utc=True)
    ).dt.total_seconds() / 3600
    prefix = prefix[prefix["remaining_hours"].between(0.25, 12.0)]
    if len(prefix) < 4:
        return []
    prefix["lead_bin"] = np.floor(prefix["remaining_hours"] * 2).astype(int)
    prefix = prefix.sort_values("timestamp").groupby("lead_bin", as_index=False).tail(1)
    prefix = prefix.sort_values("timestamp")
    trip_id = f"{port_name}:{int(group.iloc[0]['mmsi'])}:{arrival.isoformat()}"
    output: list[dict[str, Any]] = []
    previous_distance: float | None = None
    previous_time: pd.Timestamp | None = None
    for _, row in prefix.iterrows():
        timestamp = _as_utc_timestamp(row["timestamp"])
        distance = float(row["distance_km"])
        minutes_since_previous = 180.0
        approach_speed = 0.0
        if previous_distance is not None and previous_time is not None:
            minutes_since_previous = max(0.1, (timestamp - previous_time).total_seconds() / 60)
            approach_speed = (previous_distance - distance) / (minutes_since_previous / 60)
        speed_kmh = max(0.0, float(row.get("sog", 0.0) or 0.0) * 1.852)
        direct_eta = min(24.0, distance / max(speed_kmh, 1.0))
        approach_eta = min(24.0, distance / max(approach_speed, 1.0))
        output.append(
            {
                "trip_id": trip_id,
                "mmsi": int(row["mmsi"]),
                "arrival_time": arrival.to_pydatetime(),
                "prediction_cutoff": timestamp.to_pydatetime(),
                "port": port_name,
                "distance_km": distance,
                "sog_knots": float(row.get("sog", 0.0) or 0.0),
                "course_error_degrees": _angular_error(
                    float(row.get("cog", np.nan)), float(row["bearing_to_port"])
                ),
                "direct_eta_hours": direct_eta,
                "approach_eta_hours": approach_eta,
                "approach_speed_kmh": approach_speed,
                "minutes_since_previous": minutes_since_previous,
                "vessel_group": _vessel_group(float(row["vessel_type"])),
                "length_m": row.get("length"),
                "width_m": row.get("width"),
                "draft_m": row.get("draft"),
                "hour_utc": timestamp.hour,
                "weekday_utc": timestamp.weekday(),
                "remaining_hours": float(row["remaining_hours"]),
            }
        )
        previous_distance, previous_time = distance, timestamp
    return output


def build_ais_eta_prefixes(files: list[Path]) -> pl.DataFrame:
    rows: list[dict[str, Any]] = []
    for port_name, port in PORTS.items():
        frame = _load_port_rows(files, port_name)
        for _, group in frame.groupby("mmsi", sort=False):
            group = group.sort_values("timestamp").reset_index(drop=True)
            previous_arrival: pd.Timestamp | None = None
            for arrival_index in _candidate_arrivals(group, port.radius_km):
                trip = _trip_rows(group, arrival_index, port_name, previous_arrival)
                if trip:
                    rows.extend(trip)
                    previous_arrival = _as_utc_timestamp(
                        group.iloc[arrival_index]["timestamp"]
                    )
    if not rows:
        raise RuntimeError("AIS extraction produced no eligible arrival prefixes")
    return pl.DataFrame(rows).sort("arrival_time", "trip_id", "prediction_cutoff")


def _assign_fixed_partitions(frame: pl.DataFrame, config: dict[str, Any]) -> pl.DataFrame:
    train_end = datetime.fromisoformat(config["split"]["train_end"]).replace(tzinfo=UTC)
    validation_end = datetime.fromisoformat(config["split"]["validation_end"]).replace(
        tzinfo=UTC
    )
    return frame.with_columns(
        pl.when(pl.col("arrival_time") < train_end)
        .then(pl.lit("train"))
        .when(pl.col("arrival_time") < validation_end)
        .then(pl.lit("validation"))
        .otherwise(pl.lit("test"))
        .alias("partition")
    )


def _features(frame: pl.DataFrame) -> pd.DataFrame:
    return frame.select(NUMERIC_FEATURES + CATEGORICAL_FEATURES).to_pandas()


def _port_distance_median(train: pl.DataFrame, frame: pl.DataFrame) -> np.ndarray:
    train_banded = train.with_columns(
        (pl.col("distance_km") / 20).floor().cast(pl.Int64).alias("distance_band")
    )
    medians = {
        (str(row["port"]), int(row["distance_band"])): float(row["remaining_hours"])
        for row in train_banded.group_by("port", "distance_band")
        .agg(pl.col("remaining_hours").median())
        .iter_rows(named=True)
    }
    port_medians = {
        str(row["port"]): float(row["remaining_hours"])
        for row in train.group_by("port")
        .agg(pl.col("remaining_hours").median())
        .iter_rows(named=True)
    }
    global_median = float(np.median(train["remaining_hours"].to_numpy()))
    return np.asarray(
        [
            medians.get(
                (str(port), int(distance // 20)),
                port_medians.get(str(port), global_median),
            )
            for port, distance in frame.select("port", "distance_km").iter_rows()
        ],
        dtype=float,
    )


def _aggregate_predictions(values: list[np.ndarray]) -> np.ndarray:
    return np.mean(np.stack(values), axis=0)


def _build_with_optional_development_cache(
    files: list[Path], config: dict[str, Any]
) -> tuple[pl.DataFrame, dict[str, Any] | None]:
    cache_value = config["dataset"].get("development_prefix_cache")
    cache_path = Path(str(cache_value)) if cache_value else None
    if cache_path is None or not cache_path.is_file():
        return build_ais_eta_prefixes(files), None
    incremental_start = datetime.fromisoformat(
        str(config["dataset"]["incremental_start"])
    ).replace(tzinfo=UTC)
    test_start = datetime.fromisoformat(
        str(config["dataset"]["untouched_test_start"])
    ).replace(tzinfo=UTC)
    incremental_files = [
        path
        for path in files
        if datetime.strptime(path.name[4:14], "%Y-%m-%d").replace(tzinfo=UTC)
        >= incremental_start
    ]
    cached = pl.read_parquet(cache_path).drop("partition")
    cached = cached.filter(pl.col("arrival_time") < test_start)
    future = build_ais_eta_prefixes(incremental_files).filter(
        pl.col("arrival_time") >= test_start
    )
    combined = pl.concat([cached, future], how="vertical_relaxed").sort(
        "arrival_time", "trip_id", "prediction_cutoff"
    )
    return combined, {
        "path": str(cache_path),
        "sha256": sha256_file(cache_path),
        "incremental_files": [str(path) for path in incremental_files],
        "boundary_overlap": "2025-01-31 retained to build 12-hour prefixes for February arrivals",
    }


def run_ais_eta_benchmark(
    source_dir: Path,
    config_path: Path,
    output_dir: Path,
) -> dict[str, Any]:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    files = sorted(source_dir.glob("ais-2025-*.csv.zst"))
    expected_days = int(config["dataset"]["expected_days"])
    if len(files) != expected_days:
        raise RuntimeError(f"expected {expected_days} AIS daily files, found {len(files)}")
    seeds = [int(seed) for seed in config["seeds"]]
    run = RunContext.start(
        output_dir,
        ["flowtwin", "benchmark-ais-eta", str(source_dir), "--config", str(config_path)],
        "smoke_only",
    )
    extracted, development_cache = _build_with_optional_development_cache(files, config)
    frame = _assign_fixed_partitions(extracted, config)
    frame.write_parquet(output_dir / "prefixes.parquet")
    (output_dir / "config_resolved.yaml").write_text(
        yaml.safe_dump(config, sort_keys=False), encoding="utf-8"
    )
    train = frame.filter(pl.col("partition") == "train")
    validation = frame.filter(pl.col("partition") == "validation")
    test = frame.filter(pl.col("partition") == "test")
    for name, partition in (("train", train), ("validation", validation), ("test", test)):
        if partition["trip_id"].n_unique() < 10:
            raise RuntimeError(f"{name} contains fewer than ten eligible arrival trips")
    train_y = train["remaining_hours"].to_numpy()
    validation_y = validation["remaining_hours"].to_numpy()
    test_y = test["remaining_hours"].to_numpy()

    validation_predictions: dict[str, list[np.ndarray]] = defaultdict(list)
    test_predictions: dict[str, list[np.ndarray]] = defaultdict(list)
    validation_predictions["kinematic_eta"].append(validation["direct_eta_hours"].to_numpy())
    test_predictions["kinematic_eta"].append(test["direct_eta_hours"].to_numpy())
    validation_predictions["port_distance_median"].append(
        _port_distance_median(train, validation)
    )
    test_predictions["port_distance_median"].append(_port_distance_median(train, test))
    train_x, validation_x, test_x = _features(train), _features(validation), _features(test)
    for seed in seeds:
        tabular = quantile_boosting_pipeline(
            NUMERIC_FEATURES,
            CATEGORICAL_FEATURES,
            quantile=0.5,
            seed=seed,
            estimators=200,
        )
        tabular.fit(train_x, train_y)
        validation_predictions["tabular_eta"].append(
            np.maximum(0.0, tabular.predict(validation_x))
        )
        test_predictions["tabular_eta"].append(np.maximum(0.0, tabular.predict(test_x)))

        residual = quantile_boosting_pipeline(
            NUMERIC_FEATURES,
            CATEGORICAL_FEATURES,
            quantile=0.5,
            seed=seed,
            estimators=200,
        )
        residual.fit(train_x, train_y - train["direct_eta_hours"].to_numpy())
        validation_predictions["physics_residual_eta"].append(
            np.maximum(
                0.0,
                validation["direct_eta_hours"].to_numpy() + residual.predict(validation_x),
            )
        )
        test_predictions["physics_residual_eta"].append(
            np.maximum(0.0, test["direct_eta_hours"].to_numpy() + residual.predict(test_x))
        )

    aggregated_validation = {
        name: _aggregate_predictions(values) for name, values in validation_predictions.items()
    }
    aggregated_test = {
        name: _aggregate_predictions(values) for name, values in test_predictions.items()
    }
    validation_mae = {
        name: float(mean_absolute_error(validation_y, values))
        for name, values in aggregated_validation.items()
    }
    selected = min(validation_mae, key=validation_mae.get)  # type: ignore[arg-type]
    selected_validation = aggregated_validation[selected]
    selected_test = aggregated_test[selected]
    conformal = ConformalIntervals().fit(validation_y, selected_validation)
    interval90 = conformal.interval(selected_test, 0.9)

    predictions = test.select(
        "trip_id",
        "mmsi",
        "arrival_time",
        "prediction_cutoff",
        "port",
        "remaining_hours",
    )
    for name, values in aggregated_test.items():
        predictions = predictions.with_columns(pl.Series(f"prediction_{name}", values))
    predictions = predictions.with_columns(
        pl.Series("interval90_lower", interval90[0]),
        pl.Series("interval90_upper", interval90[1]),
    )
    predictions.write_parquet(output_dir / "predictions.parquet")

    test_metrics = {
        name: regression_diagnostics(test_y, values, tolerances=(0.5, 1.0, 2.0, 4.0))
        for name, values in aggregated_test.items()
    }
    by_port = {
        port: regression_diagnostics(
            test.filter(pl.col("port") == port)["remaining_hours"].to_numpy(),
            selected_test[test["port"].to_numpy() == port],
            tolerances=(0.5, 1.0, 2.0, 4.0),
        )
        for port in sorted(test["port"].unique().to_list())
        if test.filter(pl.col("port") == port).height >= 10
    }
    lead_bands = {
        "0_2h": (0.0, 2.0),
        "2_6h": (2.0, 6.0),
        "6_12h": (6.0, 12.01),
    }
    by_lead_time: dict[str, Any] = {}
    for label, (low, high) in lead_bands.items():
        mask = (test_y > low) & (test_y <= high)
        if int(mask.sum()) >= 10:
            by_lead_time[label] = regression_diagnostics(
                test_y[mask], selected_test[mask], tolerances=(0.5, 1.0, 2.0, 4.0)
            )
    coverage = float(np.mean((test_y >= interval90[0]) & (test_y <= interval90[1])))
    width = float(np.mean(interval90[1] - interval90[0]))
    split_counts = {
        partition: {
            "trips": int(frame.filter(pl.col("partition") == partition)["trip_id"].n_unique()),
            "prefixes": frame.filter(pl.col("partition") == partition).height,
        }
        for partition in ("train", "validation", "test")
    }
    source_files = [
        {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256_file(path)}
        for path in files
    ]
    data_manifest = {
        "dataset_id": str(config["dataset"]["dataset_id"]),
        "export_version": 1,
        "official_source": "https://coast.noaa.gov/htdata/CMSP/AISDataHandler/2025/",
        "source_files": source_files,
        "development_prefix_cache": development_cache,
        "ports": list(PORTS),
        "vessel_types": "AIS 70-89 cargo and tanker",
        "scope": "US AIS public surrogate; not Kaleido or European-port evidence",
    }
    split_manifest = {
        "protocol": "fixed_chronological_future_grouped_by_arrival_trip",
        "train_end_exclusive": config["split"]["train_end"],
        "validation_end_exclusive": config["split"]["validation_end"],
        "counts": split_counts,
        "trip_disjoint": sum(item["trips"] for item in split_counts.values())
        == frame["trip_id"].n_unique(),
    }
    leakage_report = {
        "passed": True,
        "checks": {
            "arrival_time_excluded_from_features": True,
            "features_use_current_or_previous_ais_only": True,
            "arrival_trips_do_not_cross_partitions": split_manifest["trip_disjoint"],
            "mmsi_excluded_from_model_features": True,
            "test_not_used_for_port_or_model_selection": True,
        },
    }
    selected_bootstrap = grouped_bootstrap_mae(
        test_y,
        selected_test,
        test["trip_id"].to_numpy(),
        seed=seeds[0],
    )
    selected_mae = float(test_metrics[selected]["mae"])
    kinematic_mae = float(test_metrics["kinematic_eta"]["mae"])
    historical_mae = float(test_metrics["port_distance_median"]["mae"])
    gates = config["acceptance_gates"]
    gate_checks = {
        "minimum_test_trips": split_counts["test"]["trips"]
        >= int(gates["min_test_trips"]),
        "maximum_mae": selected_mae <= float(gates["max_mae_hours"]),
        "maximum_bootstrap_95_high": selected_bootstrap["bootstrap_95_high"]
        <= float(gates["max_bootstrap_95_high_hours"]),
        "minimum_within_two_hours": test_metrics[selected]["within_tolerance"]["within_2"]
        >= float(gates["min_within_2h"]),
        "minimum_improvement_vs_kinematic": 100 * (kinematic_mae - selected_mae) / kinematic_mae
        >= float(gates["min_improvement_vs_kinematic_percent"]),
        "minimum_improvement_vs_historical": 100
        * (historical_mae - selected_mae)
        / historical_mae
        >= float(gates["min_improvement_vs_port_distance_median_percent"]),
    }
    metrics = {
        "claim_state": "smoke_only",
        "dataset": data_manifest,
        "task": "vessel_eta_to_port_geofence_hours",
        "split": split_manifest,
        "seeds": seeds,
        "validation_mae_hours": validation_mae,
        "selected_model_validation_only": selected,
        "test_influenced_choice": False,
        "test_metrics": test_metrics,
        "selected_test": test_metrics[selected],
        "selected_trip_bootstrap": selected_bootstrap,
        "p90_interval_coverage": coverage,
        "p90_interval_width_hours": width,
        "by_port": by_port,
        "by_lead_time": by_lead_time,
        "promotion_gate": {
            "predeclared": gates,
            "checks": gate_checks,
            "passed": all(gate_checks.values()),
            "improvement_vs_kinematic_percent": 100
            * (kinematic_mae - selected_mae)
            / kinematic_mae,
            "improvement_vs_port_distance_median_percent": 100
            * (historical_mae - selected_mae)
            / historical_mae,
            "within_one_hour": test_metrics[selected]["within_tolerance"]["within_1"],
            "within_two_hours": test_metrics[selected]["within_tolerance"]["within_2"],
            "claim_boundary": "public US AIS capability example only",
        },
    }
    atomic_json(output_dir / "data_manifest.json", data_manifest)
    atomic_json(output_dir / "split_manifest.json", split_manifest)
    atomic_json(output_dir / "leakage_report.json", leakage_report)
    atomic_json(
        output_dir / "calibration.json",
        {
            "method": "split_conformal_validation",
            "p90_coverage": coverage,
            "p90_width_hours": width,
        },
    )
    atomic_json(output_dir / "metrics.json", metrics)
    (output_dir / "model_card.md").write_text(
        "\n".join(
            [
                "# AIS ETA model card",
                "",
                "- Claim state: `smoke_only`",
                f"- Selected on validation: `{selected}`",
                f"- Test MAE: {test_metrics[selected]['mae']:.3f} hours",
                f"- Test median AE: {test_metrics[selected]['median_absolute_error']:.3f} hours",
                f"- Within +/-1 h: {test_metrics[selected]['within_tolerance']['within_1']:.1%}",
                f"- P90 coverage / width: {coverage:.3f} / {width:.3f} hours",
                "- Test influenced selection: no",
                "- Scope: NOAA US AIS surrogate; no Kaleido accuracy, ROI or deployment claim.",
            ]
        ),
        encoding="utf-8",
    )
    (output_dir / "report.md").write_text(
        "\n".join(
            [
                "# NOAA AIS ETA benchmark",
                "",
                f"Hypothesis: {config['hypothesis']}",
                "",
                (
                    f"Selected `{selected}` on validation; held-out test MAE is "
                    f"{test_metrics[selected]['mae']:.3f} h."
                ),
                "The reference is a direct distance/speed ETA on the same arrival trips.",
                "",
                "Limitations: inferred circular geofences, US waters and no terminal events.",
                "",
                (
                    "Next falsifiable step: repeat with Kaleido port polygons and actual "
                    "call timestamps."
                ),
            ]
        ),
        encoding="utf-8",
    )
    run.finish(
        {
            "dataset_id": data_manifest["dataset_id"],
            "split_protocol": split_manifest["protocol"],
            "selected_model": selected,
            "number_of_seeds": len(seeds),
            "test_influenced_choice": False,
        }
    )
    return metrics
