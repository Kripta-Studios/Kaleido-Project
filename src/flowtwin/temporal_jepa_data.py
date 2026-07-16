from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from flowtwin.event_jepa_training import EventJEPAData, EventJEPAPartition
from flowtwin.sequence_training import _load_events, build_sequence_data


def _disjoint_target_view(
    trace: list[tuple[Any, str]],
    observed_count: int,
    target_count: int,
    sequence_data: Any,
) -> tuple[np.ndarray, int, np.ndarray]:
    """Encode only future events while retaining future-state numeric semantics."""

    future = trace[observed_count:target_count]
    if not future:
        raise ValueError("a disjoint target view must contain at least one future event")
    encoded = [sequence_data.vocabulary.encode(activity) for _, activity in future]
    encoded = encoded[: sequence_data.max_length]
    tokens = np.zeros(sequence_data.max_length, dtype=np.int64)
    tokens[: len(encoded)] = encoded
    future_state = trace[:target_count]
    first_time = future_state[0][0]
    last_time = future_state[-1][0]
    previous_time = future_state[-2][0] if len(future_state) > 1 else last_time
    numeric = np.asarray(
        [
            max(0.0, (last_time - first_time).total_seconds() / 86400),
            np.log1p(len(future_state)),
            max(0.0, (last_time - previous_time).total_seconds() / 3600),
        ],
        dtype=np.float32,
    )
    return tokens, len(encoded), numeric


def build_disjoint_event_jepa_data(
    source_path: Path,
    baseline_run_dir: Path,
    *,
    max_length: int,
    horizon_offsets: tuple[int, ...] = (1, 2, -1),
    horizon_labels: tuple[str, ...] = ("next_event", "two_events", "completion_suffix"),
) -> EventJEPAData:
    """Build causal context prefixes and non-overlapping future event blocks."""

    sequence_data = build_sequence_data(
        source_path,
        baseline_run_dir,
        max_length=max_length,
    )
    prefixes = pl.read_parquet(baseline_run_dir / "prefixes.parquet").sort(
        ["operation_id", "prediction_cutoff", "prefix_events"]
    )
    event_groups = _load_events(
        source_path,
        set(prefixes["operation_id"].to_list()),
        case_column="case:concept:name",
        activity_column="concept:name",
        timestamp_column="time:timestamp",
    )
    buckets: dict[str, dict[str, list[Any]]] = {
        name: {"tokens": [], "lengths": [], "numeric": [], "operations": [], "cutoffs": []}
        for name in ("train", "validation", "test")
    }
    for row in prefixes.iter_rows(named=True):
        operation_id = str(row["operation_id"])
        trace = event_groups[operation_id]
        observed_count = int(row["prefix_events"])
        if observed_count < 1 or observed_count >= len(trace):
            continue
        views = [
            _disjoint_target_view(
                trace,
                observed_count,
                len(trace) if offset < 0 else min(len(trace), observed_count + offset),
                sequence_data,
            )
            for offset in horizon_offsets
        ]
        partition = str(row["partition"])
        buckets[partition]["tokens"].append(np.stack([view[0] for view in views]))
        buckets[partition]["lengths"].append([view[1] for view in views])
        buckets[partition]["numeric"].append(np.stack([view[2] for view in views]))
        buckets[partition]["operations"].append(operation_id)
        buckets[partition]["cutoffs"].append(row["prediction_cutoff"])
    partitions: dict[str, EventJEPAPartition] = {}
    for name, bucket in buckets.items():
        context = sequence_data.partitions[name]
        if context.operation_ids != bucket["operations"] or context.cutoffs != bucket["cutoffs"]:
            raise RuntimeError("disjoint Event-JEPA targets are not aligned with causal prefixes")
        partitions[name] = EventJEPAPartition(
            context=context,
            target_tokens=np.stack(bucket["tokens"]).astype(np.int64),
            target_lengths=np.asarray(bucket["lengths"], dtype=np.int64),
            target_numeric=np.stack(bucket["numeric"]).astype(np.float32),
        )
    return EventJEPAData(
        partitions=partitions,
        vocabulary_size=sequence_data.vocabulary_size,
        max_length=max_length,
        horizon_labels=horizon_labels,
    )


def write_embeddings(
    path: Path,
    data: EventJEPAData,
    embeddings: dict[str, np.ndarray],
    *,
    column_prefix: str,
    uncertainty: dict[str, tuple[np.ndarray, np.ndarray]] | None = None,
) -> None:
    frames: list[pl.DataFrame] = []
    for partition_name in ("train", "validation", "test"):
        partition = data.partitions[partition_name]
        values = embeddings[partition_name]
        payload: dict[str, Any] = {
            "operation_id": partition.context.operation_ids,
            "prediction_cutoff": partition.context.cutoffs,
            "partition": [partition_name] * len(values),
            "remaining_minutes": partition.context.target_hours * 60,
        }
        for index in range(values.shape[1]):
            payload[f"{column_prefix}_{index:03d}"] = values[:, index]
        if uncertainty is not None:
            context_uncertainty, predictive_uncertainty = uncertainty[partition_name]
            payload[f"{column_prefix}_context_uncertainty"] = context_uncertainty
            payload[f"{column_prefix}_predictive_uncertainty"] = predictive_uncertainty
        frames.append(pl.DataFrame(payload))
    path.parent.mkdir(parents=True, exist_ok=True)
    pl.concat(frames).write_parquet(path)
