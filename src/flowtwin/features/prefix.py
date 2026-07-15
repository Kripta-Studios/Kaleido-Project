from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import polars as pl

from flowtwin.data.splits import (
    OperationSummary,
    Partition,
    SplitManifest,
    chronological_grouped_split,
)


@dataclass(frozen=True)
class PrefixDataset:
    frame: pl.DataFrame
    split: SplitManifest
    source_rows: int
    source_cases: int


def _scan_events(path: Path) -> pl.LazyFrame:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pl.scan_csv(path, infer_schema_length=10000)
    if suffix in {".parquet", ".pq"}:
        return pl.scan_parquet(path)
    raise ValueError(f"unsupported prefix source: {suffix}")


def build_prefix_dataset(
    path: Path,
    *,
    case_column: str = "case:concept:name",
    activity_column: str = "concept:name",
    timestamp_column: str = "time:timestamp",
    prediction_points: tuple[float, ...] = (0.25, 0.5, 0.75),
    max_cases: int | None = None,
    seed: int = 42,
) -> PrefixDataset:
    if not prediction_points or any(point <= 0 or point >= 1 for point in prediction_points):
        raise ValueError("prediction points must be strictly between zero and one")
    source = (
        _scan_events(path)
        .select(case_column, activity_column, timestamp_column)
        .with_columns(
            pl.col(case_column).cast(pl.String),
            pl.col(activity_column).cast(pl.String),
            pl.col(timestamp_column)
            .cast(pl.String)
            .str.to_datetime(strict=False, time_zone="UTC")
            .alias("_event_time"),
        )
        .drop_nulls([case_column, activity_column, "_event_time"])
    )
    case_stats = (
        source.group_by(case_column)
        .agg(
            pl.col("_event_time").min().alias("_case_start"),
            pl.col("_event_time").max().alias("_case_end"),
            pl.len().alias("_case_events"),
        )
        .sort(["_case_start", case_column])
    )
    if max_cases is not None:
        case_stats = case_stats.head(max_cases)
    selected_cases = case_stats.select(case_column)
    events = (
        source.join(selected_cases, on=case_column, how="inner")
        .sort([case_column, "_event_time", activity_column])
        .with_columns(
            pl.col("_event_time").shift(1).over(case_column).alias("_previous_time"),
            pl.int_range(1, pl.len() + 1).over(case_column).alias("_event_index"),
        )
        .join(case_stats, on=case_column, how="inner")
    )
    conditions = [
        pl.col("_event_index")
        == (pl.col("_case_events").cast(pl.Float64) * point).ceil().cast(pl.Int64)
        for point in prediction_points
    ]
    selected = conditions[0]
    for condition in conditions[1:]:
        selected = selected | condition
    prefixes = (
        events.filter(selected & (pl.col("_event_index") < pl.col("_case_events")))
        .with_columns(
            ((pl.col("_event_time") - pl.col("_case_start")).dt.total_seconds() / 60)
            .cast(pl.Float64)
            .alias("elapsed_minutes"),
            ((pl.col("_case_end") - pl.col("_event_time")).dt.total_seconds() / 60)
            .cast(pl.Float64)
            .alias("remaining_minutes"),
            (
                (pl.col("_event_time") - pl.col("_previous_time")).dt.total_seconds().fill_null(0)
                / 60
            )
            .cast(pl.Float64)
            .alias("since_previous_minutes"),
            pl.col("_event_index").cast(pl.Float64).alias("prefix_events"),
            pl.col("_event_time").dt.hour().alias("hour_utc"),
            pl.col("_event_time").dt.weekday().alias("weekday_utc"),
            pl.col(activity_column).alias("last_activity"),
            pl.col(case_column).alias("operation_id"),
            pl.col("_event_time").alias("prediction_cutoff"),
        )
        .select(
            "operation_id",
            "prediction_cutoff",
            "_case_start",
            "last_activity",
            "elapsed_minutes",
            "since_previous_minutes",
            "prefix_events",
            "hour_utc",
            "weekday_utc",
            "remaining_minutes",
        )
        .collect()
    )
    cases = case_stats.select(case_column, "_case_start").collect()
    summaries = [
        OperationSummary(operation_id=row[case_column], start_time=row["_case_start"])
        for row in cases.iter_rows(named=True)
    ]
    split = chronological_grouped_split(summaries, seed=seed)
    partition_map = {
        operation_id: partition.value for operation_id, partition in split.assignments.items()
    }
    prefixes = prefixes.with_columns(
        pl.col("operation_id")
        .replace_strict(partition_map, default=Partition.TEST.value)
        .alias("partition")
    )
    return PrefixDataset(
        frame=prefixes,
        split=split,
        source_rows=int(source.select(pl.len()).collect().item()),
        source_cases=cases.height,
    )


def prefix_feature_columns() -> list[str]:
    return [
        "last_activity",
        "elapsed_minutes",
        "since_previous_minutes",
        "prefix_events",
        "hour_utc",
        "weekday_utc",
    ]


def prediction_point_index(event_count: int, fraction: float) -> int:
    return max(1, math.ceil(event_count * fraction))
