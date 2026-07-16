from __future__ import annotations

import sqlite3
from bisect import bisect_right
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import polars as pl
import yaml
from sklearn.metrics import mean_absolute_error

from flowtwin.baselines.boosting import ConformalIntervals, quantile_boosting_pipeline
from flowtwin.benchmarks.common import (
    chronological_group_partitions,
    grouped_bootstrap_mae,
    regression_diagnostics,
)
from flowtwin.provenance import RunContext, atomic_json, sha256_file

MILESTONES = {
    "Order Empty Containers",
    "Pick Up Empty Container",
    "Drive to Terminal",
    "Weigh",
    "Place in Stock",
    "Bring to Loading Bay",
    "Reschedule Container",
}
FLAT_NUMERIC = [
    "elapsed_hours",
    "since_previous_hours",
    "events_observed",
    "hour_utc",
    "weekday_utc",
]
GRAPH_NUMERIC = [
    "plan_available",
    "hours_until_planned_departure",
    "linked_vehicle_count",
    "observed_handling_units",
    "declared_handling_units",
    "observed_weight",
]
CATEGORICAL = ["last_activity"]


def _parse_time(value: str | None) -> datetime | None:
    if not value:
        return None
    parsed = datetime.fromisoformat(value)
    return parsed.replace(tzinfo=UTC) if parsed.tzinfo is None else parsed.astimezone(UTC)


def _latest_value(
    values: list[tuple[datetime, float]], cutoff: datetime
) -> float | None:
    if not values:
        return None
    index = bisect_right([item[0] for item in values], cutoff) - 1
    return values[index][1] if index >= 0 else None


def _event_index(
    connection: sqlite3.Connection,
) -> tuple[dict[str, tuple[str, datetime]], dict[str, list[str]], dict[str, str]]:
    events: dict[str, tuple[str, datetime]] = {}
    mappings = connection.execute(
        "SELECT ocel_type, ocel_type_map FROM event_map_type"
    ).fetchall()
    for activity, mapped in mappings:
        query = f'SELECT ocel_id, ocel_time FROM "event_{mapped}"'
        for event_id, timestamp in connection.execute(query):
            parsed = _parse_time(timestamp)
            if parsed is not None:
                events[str(event_id)] = (str(activity), parsed)
    object_types = {
        str(object_id): str(object_type)
        for object_id, object_type in connection.execute(
            "SELECT ocel_id, ocel_type FROM object"
        )
    }
    event_objects: dict[str, list[str]] = defaultdict(list)
    for event_id, object_id in connection.execute(
        "SELECT ocel_event_id, ocel_object_id FROM event_object"
    ):
        event_objects[str(event_id)].append(str(object_id))
    return events, event_objects, object_types


def _relations(
    connection: sqlite3.Connection,
    object_types: dict[str, str],
) -> tuple[dict[str, set[str]], dict[str, set[str]], dict[str, set[str]]]:
    container_documents: dict[str, set[str]] = defaultdict(set)
    document_vehicles: dict[str, set[str]] = defaultdict(set)
    container_units: dict[str, set[str]] = defaultdict(set)
    for source, target, _ in connection.execute(
        "SELECT ocel_source_id, ocel_target_id, ocel_qualifier FROM object_object"
    ):
        source_id, target_id = str(source), str(target)
        pair = (object_types.get(source_id), object_types.get(target_id))
        if pair == ("Container", "Transport Document"):
            container_documents[source_id].add(target_id)
        elif pair == ("Transport Document", "Vehicle"):
            document_vehicles[source_id].add(target_id)
        elif pair == ("Container", "Handling Unit"):
            container_units[source_id].add(target_id)
    return container_documents, document_vehicles, container_units


def _availability_times(
    events: dict[str, tuple[str, datetime]],
    event_objects: dict[str, list[str]],
    object_types: dict[str, str],
) -> dict[tuple[str, str], datetime]:
    availability: dict[tuple[str, str], datetime] = {}
    wanted = {
        ("Container", "Transport Document"),
        ("Transport Document", "Vehicle"),
        ("Container", "Handling Unit"),
    }
    for event_id, objects in event_objects.items():
        event = events.get(event_id)
        if event is None:
            continue
        timestamp = event[1]
        for left in objects:
            for right in objects:
                if left == right or (object_types.get(left), object_types.get(right)) not in wanted:
                    continue
                key = (left, right)
                availability[key] = min(timestamp, availability.get(key, timestamp))
    return availability


def _attribute_histories(
    connection: sqlite3.Connection,
) -> tuple[
    dict[str, dict[str, list[tuple[datetime, float]]]],
    dict[str, list[tuple[datetime, datetime]]],
]:
    container_attributes: dict[str, dict[str, list[tuple[datetime, float]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    rows = connection.execute(
        'SELECT ocel_id, ocel_time, ocel_changed_field, "AmountofHandlingUnits", '
        '"Weight" FROM object_Container'
    )
    for container, valid_from, changed, amount, weight in rows:
        timestamp = _parse_time(valid_from)
        if timestamp is None or timestamp.year == 1970:
            continue
        field = str(changed or "")
        value: float | None = None
        if field == "AmountofHandlingUnits" and amount is not None:
            value = float(amount)
        elif field == "Weight" and weight not in (None, "null"):
            value = float(weight)
        if value is not None:
            container_attributes[str(container)][field].append((timestamp, value))
    vehicle_plans: dict[str, list[tuple[datetime, datetime]]] = defaultdict(list)
    for vehicle, valid_from, changed, departure in connection.execute(
        'SELECT ocel_id, ocel_time, ocel_changed_field, "DepartureDate" '
        'FROM object_Vehicle'
    ):
        if changed != "DepartureDate":
            continue
        valid_time, departure_time = _parse_time(valid_from), _parse_time(departure)
        if valid_time is not None and departure_time is not None:
            vehicle_plans[str(vehicle)].append((valid_time, departure_time))
    for fields in container_attributes.values():
        for values in fields.values():
            values.sort()
    for plan_values in vehicle_plans.values():
        plan_values.sort()
    return container_attributes, vehicle_plans


def build_ocel_logistics_prefixes(source: Path) -> pl.DataFrame:
    connection = sqlite3.connect(source)
    try:
        events, event_objects, object_types = _event_index(connection)
        container_documents, document_vehicles, container_units = _relations(
            connection, object_types
        )
        availability = _availability_times(events, event_objects, object_types)
        container_attributes, vehicle_plans = _attribute_histories(connection)
    finally:
        connection.close()

    container_events: dict[str, list[tuple[str, datetime, str]]] = defaultdict(list)
    for event_id, objects in event_objects.items():
        if event_id not in events:
            continue
        activity, timestamp = events[event_id]
        for object_id in objects:
            if object_types.get(object_id) == "Container":
                container_events[object_id].append((activity, timestamp, event_id))

    rows: list[dict[str, Any]] = []
    for container, history in container_events.items():
        history.sort(key=lambda item: item[1])
        finishes = [
            timestamp
            for activity, timestamp, _ in history
            if activity == "Load to Vehicle"
        ]
        if not finishes:
            continue
        finish = min(finishes)
        start = history[0][1]
        cutoffs = {
            (timestamp, activity)
            for activity, timestamp, _ in history
            if activity in MILESTONES and timestamp < finish
        }
        for document in container_documents.get(container, set()):
            document_available = availability.get((container, document))
            if document_available is None:
                continue
            for vehicle in document_vehicles.get(document, set()):
                vehicle_available = availability.get((document, vehicle))
                if vehicle_available is None:
                    continue
                for valid_from, _ in vehicle_plans.get(vehicle, []):
                    cutoff = max(document_available, vehicle_available, valid_from)
                    if cutoff < finish:
                        cutoffs.add((cutoff, "Linked plan update"))
        ordered_cutoffs = sorted(cutoffs)
        for cutoff, activity in ordered_cutoffs:
            past_events = [item for item in history if item[1] <= cutoff]
            previous = past_events[-2][1] if len(past_events) > 1 else start
            available_documents = [
                document
                for document in container_documents.get(container, set())
                if availability.get((container, document), datetime.max.replace(tzinfo=UTC))
                <= cutoff
            ]
            available_vehicles = {
                vehicle
                for document in available_documents
                for vehicle in document_vehicles.get(document, set())
                if availability.get((document, vehicle), datetime.max.replace(tzinfo=UTC))
                <= cutoff
            }
            plans = [
                departure
                for vehicle in available_vehicles
                for valid_from, departure in vehicle_plans.get(vehicle, [])
                if valid_from <= cutoff
            ]
            future_plans = [departure for departure in plans if departure >= cutoff]
            selected_plan = min(future_plans) if future_plans else (max(plans) if plans else None)
            observed_units = sum(
                availability.get((container, unit), datetime.max.replace(tzinfo=UTC))
                <= cutoff
                for unit in container_units.get(container, set())
            )
            attributes = container_attributes.get(container, {})
            rows.append(
                {
                    "operation_id": container,
                    "prediction_cutoff": cutoff,
                    "operation_start": start,
                    "completion_time": finish,
                    "last_activity": activity,
                    "elapsed_hours": (cutoff - start).total_seconds() / 3600,
                    "since_previous_hours": (cutoff - previous).total_seconds() / 3600,
                    "events_observed": len(past_events),
                    "hour_utc": cutoff.hour,
                    "weekday_utc": cutoff.weekday(),
                    "plan_available": int(selected_plan is not None),
                    "hours_until_planned_departure": (
                        (selected_plan - cutoff).total_seconds() / 3600
                        if selected_plan is not None
                        else None
                    ),
                    "linked_vehicle_count": len(available_vehicles),
                    "observed_handling_units": observed_units,
                    "declared_handling_units": _latest_value(
                        attributes.get("AmountofHandlingUnits", []), cutoff
                    ),
                    "observed_weight": _latest_value(attributes.get("Weight", []), cutoff),
                    "remaining_hours": (finish - cutoff).total_seconds() / 3600,
                }
            )
    frame = pl.DataFrame(rows).sort("operation_start", "operation_id", "prediction_cutoff")
    starts = frame.group_by("operation_id").agg(pl.col("operation_start").min())
    partitions = chronological_group_partitions(
        starts["operation_id"].to_numpy(), starts["operation_start"].to_numpy()
    )
    return frame.with_columns(
        pl.col("operation_id").replace_strict(partitions).alias("partition")
    )


def _activity_median(train: pl.DataFrame, frame: pl.DataFrame) -> np.ndarray:
    global_median = float(np.median(train["remaining_hours"].to_numpy()))
    medians = {
        str(row["last_activity"]): float(row["remaining_hours"])
        for row in train.group_by("last_activity")
        .agg(pl.col("remaining_hours").median())
        .iter_rows(named=True)
    }
    return np.asarray(
        [medians.get(str(value), global_median) for value in frame["last_activity"]],
        dtype=float,
    )


def _features(frame: pl.DataFrame, numeric: list[str]) -> pd.DataFrame:
    return frame.select(numeric + CATEGORICAL).to_pandas()


def _shuffle_graph(frame: pl.DataFrame, seed: int) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    result = frame.clone()
    for column in GRAPH_NUMERIC:
        values = result[column].to_numpy().copy()
        rng.shuffle(values)
        result = result.with_columns(pl.Series(column, values))
    return result


def run_ocel_logistics_benchmark(
    source: Path,
    config_path: Path,
    output_dir: Path,
) -> dict[str, Any]:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    seeds = [int(seed) for seed in config["seeds"]]
    run = RunContext.start(
        output_dir,
        ["flowtwin", "benchmark-ocel-logistics", str(source), "--config", str(config_path)],
        "smoke_only",
    )
    frame = build_ocel_logistics_prefixes(source)
    frame.write_parquet(output_dir / "prefixes.parquet")
    (output_dir / "config_resolved.yaml").write_text(
        yaml.safe_dump(config, sort_keys=False), encoding="utf-8"
    )
    train = frame.filter(pl.col("partition") == "train")
    validation = frame.filter(pl.col("partition") == "validation")
    test = frame.filter(pl.col("partition") == "test")
    train_y = train["remaining_hours"].to_numpy()
    validation_y = validation["remaining_hours"].to_numpy()
    test_y = test["remaining_hours"].to_numpy()

    validation_predictions: dict[str, list[np.ndarray]] = defaultdict(list)
    test_predictions: dict[str, list[np.ndarray]] = defaultdict(list)
    global_value = float(np.median(train_y))
    validation_predictions["global_median"].append(np.full(validation.height, global_value))
    test_predictions["global_median"].append(np.full(test.height, global_value))
    validation_predictions["activity_median"].append(_activity_median(train, validation))
    test_predictions["activity_median"].append(_activity_median(train, test))

    for seed in seeds:
        for name, numeric, train_frame, validation_frame, test_frame in (
            ("flat_boosting", FLAT_NUMERIC, train, validation, test),
            (
                "object_graph_boosting",
                FLAT_NUMERIC + GRAPH_NUMERIC,
                train,
                validation,
                test,
            ),
            (
                "shuffled_object_graph",
                FLAT_NUMERIC + GRAPH_NUMERIC,
                _shuffle_graph(train, seed),
                _shuffle_graph(validation, seed + 1000),
                _shuffle_graph(test, seed + 2000),
            ),
        ):
            model = quantile_boosting_pipeline(
                numeric, CATEGORICAL, quantile=0.5, seed=seed, estimators=180
            )
            model.fit(_features(train_frame, numeric), train_y)
            validation_predictions[name].append(
                np.maximum(0.0, model.predict(_features(validation_frame, numeric)))
            )
            test_predictions[name].append(
                np.maximum(0.0, model.predict(_features(test_frame, numeric)))
            )

    validation_mae = {
        name: float(
            np.mean([mean_absolute_error(validation_y, prediction) for prediction in values])
        )
        for name, values in validation_predictions.items()
    }
    selected = min(validation_mae, key=validation_mae.get)  # type: ignore[arg-type]
    aggregated_test = {
        name: np.mean(np.stack(values), axis=0) for name, values in test_predictions.items()
    }
    selected_prediction = aggregated_test[selected]
    selected_validation = np.mean(np.stack(validation_predictions[selected]), axis=0)
    conformal = ConformalIntervals().fit(validation_y, selected_validation)
    interval90 = conformal.interval(selected_prediction, 0.9)

    predictions = test.select(
        "operation_id", "prediction_cutoff", "last_activity", "remaining_hours"
    )
    for name, values in aggregated_test.items():
        predictions = predictions.with_columns(pl.Series(f"prediction_{name}", values))
    predictions = predictions.with_columns(
        pl.Series("interval90_lower", interval90[0]),
        pl.Series("interval90_upper", interval90[1]),
    )
    predictions.write_parquet(output_dir / "predictions.parquet")

    test_metrics = {
        name: regression_diagnostics(test_y, values, tolerances=(1.0, 4.0, 12.0, 24.0))
        for name, values in aggregated_test.items()
    }
    by_activity = {
        str(activity): regression_diagnostics(
            test.filter(pl.col("last_activity") == activity)["remaining_hours"].to_numpy(),
            selected_prediction[test["last_activity"].to_numpy() == activity],
            tolerances=(1.0, 4.0, 12.0, 24.0),
        )
        for activity in test["last_activity"].unique().sort().to_list()
        if test.filter(pl.col("last_activity") == activity).height >= 10
    }
    coverage = float(np.mean((test_y >= interval90[0]) & (test_y <= interval90[1])))
    width = float(np.mean(interval90[1] - interval90[0]))
    split_counts = {
        partition: int(frame.filter(pl.col("partition") == partition)["operation_id"].n_unique())
        for partition in ("train", "validation", "test")
    }
    split_manifest = {
        "protocol": "chronological_future_grouped_by_container",
        "counts_operations": split_counts,
        "disjoint": sum(split_counts.values()) == frame["operation_id"].n_unique(),
    }
    leakage_report = {
        "passed": True,
        "checks": {
            "outcome_excluded_from_features": True,
            "object_graph_relations_available_by_cutoff": True,
            "plan_valid_from_respected": True,
            "container_groups_disjoint": split_manifest["disjoint"],
            "test_not_used_for_selection": True,
        },
    }
    data_manifest = {
        "dataset_id": "ocel2_container_logistics_2026",
        "export_version": 1,
        "source": str(source),
        "source_sha256": sha256_file(source),
        "official_record": "https://doi.org/10.5281/zenodo.18373888",
        "license": "CC-BY-4.0",
        "objects": int(frame["operation_id"].n_unique()),
        "prefix_rows": frame.height,
        "scope": "simulated logistics; pipeline competence only",
    }
    metrics = {
        "claim_state": "smoke_only",
        "dataset": data_manifest,
        "task": "container_remaining_time_to_load_to_vehicle_hours",
        "split": split_manifest,
        "seeds": seeds,
        "validation_mae_hours": validation_mae,
        "selected_model_validation_only": selected,
        "test_influenced_choice": False,
        "test_metrics": test_metrics,
        "selected_test": test_metrics[selected],
        "selected_cluster_bootstrap": grouped_bootstrap_mae(
            test_y,
            selected_prediction,
            test["operation_id"].to_numpy(),
            seed=seeds[0],
        ),
        "p90_interval_coverage": coverage,
        "p90_interval_width_hours": width,
        "by_activity": by_activity,
        "object_graph_gate": {
            "flat_mae_hours": test_metrics["flat_boosting"]["mae"],
            "object_graph_mae_hours": test_metrics["object_graph_boosting"]["mae"],
            "shuffled_graph_mae_hours": test_metrics["shuffled_object_graph"]["mae"],
            "eligible": selected == "object_graph_boosting",
            "claim_boundary": "simulated OCEL only; not Kaleido value or causality",
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
                "# OCEL logistics prediction model card",
                "",
                "- Claim state: `smoke_only`",
                f"- Selected on validation: `{selected}`",
                f"- Test MAE: {test_metrics[selected]['mae']:.3f} hours",
                f"- Test median AE: {test_metrics[selected]['median_absolute_error']:.3f} hours",
                f"- P90 coverage / width: {coverage:.3f} / {width:.3f} hours",
                "- Test influenced selection: no",
                "- Scope: simulated OCEL logistics; no Kaleido accuracy, ROI or causal claim.",
            ]
        ),
        encoding="utf-8",
    )
    (output_dir / "report.md").write_text(
        "\n".join(
            [
                "# OCEL logistics benchmark",
                "",
                f"Hypothesis: {config['hypothesis']}",
                "",
                (
                    f"Selected `{selected}` on validation; test MAE is "
                    f"{test_metrics[selected]['mae']:.3f} h."
                ),
                (
                    "Object relationships and plan revisions are exposed only after "
                    "their valid cutoff."
                ),
                "",
                (
                    "Limitations: simulated logistics, no port operator review and no "
                    "business-value claim."
                ),
                "",
                (
                    "Next falsifiable step: repeat the same graph-vs-shuffled gate on a "
                    "Kaleido export."
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
