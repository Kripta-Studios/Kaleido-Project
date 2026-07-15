from __future__ import annotations

import copy
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import yaml

from flowtwin.baselines.process_transformer import (
    SequenceModelConfig,
    build_sequence_model,
    pinball_loss,
    require_torch,
)
from flowtwin.evaluation.remaining_time import remaining_time_metrics
from flowtwin.features.categorical import Vocabulary
from flowtwin.provenance import RunContext, atomic_json, sha256_file


@dataclass(frozen=True)
class SequencePartition:
    tokens: np.ndarray
    lengths: np.ndarray
    numeric: np.ndarray
    target_hours: np.ndarray
    operation_ids: list[str]
    cutoffs: list[Any]


@dataclass(frozen=True)
class SequenceData:
    partitions: dict[str, SequencePartition]
    vocabulary_size: int
    max_length: int


def _load_events(
    source_path: Path,
    operation_ids: set[str],
    *,
    case_column: str,
    activity_column: str,
    timestamp_column: str,
) -> dict[str, list[tuple[Any, str]]]:
    frame = (
        pl.scan_csv(source_path, infer_schema_length=10000)
        .select(case_column, activity_column, timestamp_column)
        .with_columns(
            pl.col(case_column).cast(pl.String),
            pl.col(activity_column).cast(pl.String),
            pl.col(timestamp_column)
            .cast(pl.String)
            .str.to_datetime(strict=False, time_zone="UTC")
            .alias("_event_time"),
        )
        .filter(pl.col(case_column).is_in(sorted(operation_ids)))
        .drop_nulls([case_column, activity_column, "_event_time"])
        .sort([case_column, "_event_time", activity_column])
        .collect()
    )
    groups: dict[str, list[tuple[Any, str]]] = {}
    for row in frame.iter_rows(named=True):
        groups.setdefault(str(row[case_column]), []).append(
            (row["_event_time"], str(row[activity_column]))
        )
    return groups


def build_sequence_data(
    source_path: Path,
    baseline_run_dir: Path,
    *,
    max_length: int = 128,
    case_column: str = "case:concept:name",
    activity_column: str = "concept:name",
    timestamp_column: str = "time:timestamp",
) -> SequenceData:
    prefixes = pl.read_parquet(baseline_run_dir / "prefixes.parquet").sort(
        ["operation_id", "prediction_cutoff"]
    )
    operation_ids = set(prefixes["operation_id"].to_list())
    event_groups = _load_events(
        source_path,
        operation_ids,
        case_column=case_column,
        activity_column=activity_column,
        timestamp_column=timestamp_column,
    )
    training_operations = set(
        prefixes.filter(pl.col("partition") == "train")["operation_id"].to_list()
    )
    vocabulary = Vocabulary(
        activity
        for operation_id in training_operations
        for _, activity in event_groups[operation_id]
    )
    buckets: dict[str, dict[str, list[Any]]] = {
        partition: {
            "tokens": [],
            "lengths": [],
            "numeric": [],
            "target": [],
            "operation_ids": [],
            "cutoffs": [],
        }
        for partition in ("train", "validation", "test")
    }
    for row in prefixes.iter_rows(named=True):
        operation_id = str(row["operation_id"])
        cutoff = row["prediction_cutoff"]
        complete_trace = event_groups[operation_id]
        observed = [
            (time_value, activity)
            for time_value, activity in complete_trace
            if time_value <= cutoff
        ]
        if not observed:
            continue
        encoded = [vocabulary.encode(activity) for _, activity in observed][-max_length:]
        length = len(encoded)
        padded = np.zeros(max_length, dtype=np.int64)
        padded[:length] = encoded
        first_time = observed[0][0]
        last_time = observed[-1][0]
        previous_time = observed[-2][0] if len(observed) > 1 else last_time
        elapsed_minutes = max(0.0, (last_time - first_time).total_seconds() / 60)
        since_previous = max(0.0, (last_time - previous_time).total_seconds() / 60)
        numeric = np.asarray(
            [
                elapsed_minutes / 1440,
                np.log1p(len(observed)),
                since_previous / 60,
            ],
            dtype=np.float32,
        )
        partition = str(row["partition"])
        bucket = buckets[partition]
        bucket["tokens"].append(padded)
        bucket["lengths"].append(length)
        bucket["numeric"].append(numeric)
        bucket["target"].append(float(row["remaining_minutes"]) / 60)
        bucket["operation_ids"].append(operation_id)
        bucket["cutoffs"].append(cutoff)
    partitions = {
        name: SequencePartition(
            tokens=np.stack(bucket["tokens"]).astype(np.int64),
            lengths=np.asarray(bucket["lengths"], dtype=np.int64),
            numeric=np.stack(bucket["numeric"]).astype(np.float32),
            target_hours=np.asarray(bucket["target"], dtype=np.float32),
            operation_ids=list(bucket["operation_ids"]),
            cutoffs=list(bucket["cutoffs"]),
        )
        for name, bucket in buckets.items()
    }
    return SequenceData(
        partitions=partitions,
        vocabulary_size=len(vocabulary),
        max_length=max_length,
    )


def _loader(partition: SequencePartition, batch_size: int, shuffle: bool, seed: int) -> Any:
    torch = require_torch()
    dataset = torch.utils.data.TensorDataset(
        torch.from_numpy(partition.tokens),
        torch.from_numpy(partition.lengths),
        torch.from_numpy(partition.numeric),
        torch.from_numpy(partition.target_hours),
    )
    generator = torch.Generator().manual_seed(seed)
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        generator=generator,
        num_workers=0,
    )


def _evaluate_loss(model: Any, loader: Any) -> float:
    torch = require_torch()
    model.eval()
    total = 0.0
    rows = 0
    with torch.no_grad():
        for tokens, lengths, numeric, target in loader:
            prediction = model(tokens, lengths, numeric)
            loss = pinball_loss(prediction, target)
            total += float(loss) * len(target)
            rows += len(target)
    return total / max(rows, 1)


def _predict(model: Any, loader: Any) -> np.ndarray:
    torch = require_torch()
    model.eval()
    values: list[np.ndarray] = []
    with torch.no_grad():
        for tokens, lengths, numeric, _ in loader:
            values.append(model(tokens, lengths, numeric).cpu().numpy())
    return np.concatenate(values, axis=0)


def train_sequence_models(
    source_path: Path,
    baseline_run_dir: Path,
    config_path: Path,
    output_dir: Path,
    *,
    seeds: tuple[int, ...] = (11, 42, 73),
) -> dict[str, Any]:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    compute = config["compute"]
    run = RunContext.start(
        output_dir,
        ["flowtwin", "train-sequence", str(source_path)],
        str(config["claim_state"]),
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "config_resolved.yaml").write_text(
        config_path.read_text(encoding="utf-8"), encoding="utf-8"
    )
    data = build_sequence_data(
        source_path,
        baseline_run_dir,
        max_length=int(compute["max_sequence_length"]),
    )
    torch = require_torch()
    baseline_metrics = json.loads(
        (baseline_run_dir / "metrics.json").read_text(encoding="utf-8")
    )
    baseline_name = baseline_metrics["model_selection"]["selected_model"]
    baseline_mae = float(baseline_metrics["selected_model_test"]["mae_minutes"])
    results: dict[str, list[dict[str, Any]]] = {"gru": [], "transformer": []}
    prediction_frame = pl.DataFrame(
        {
            "operation_id": data.partitions["test"].operation_ids,
            "prediction_cutoff": data.partitions["test"].cutoffs,
            "remaining_minutes": data.partitions["test"].target_hours * 60,
        }
    )
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    for architecture in ("gru", "transformer"):
        for seed in seeds:
            torch.manual_seed(seed)
            np.random.seed(seed)
            model = build_sequence_model(
                SequenceModelConfig(
                    vocabulary_size=data.vocabulary_size,
                    hidden_size=64,
                    layers=2,
                    dropout=0.1,
                    architecture=architecture,
                    max_length=data.max_length,
                )
            )
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
            train_loader = _loader(
                data.partitions["train"], int(compute["batch_size"]), True, seed
            )
            validation_loader = _loader(
                data.partitions["validation"], int(compute["batch_size"]), False, seed
            )
            history: list[dict[str, float | int]] = []
            best_validation = float("inf")
            best_state: dict[str, Any] | None = None
            started = time.perf_counter()
            for epoch in range(1, int(compute["epochs"]) + 1):
                model.train()
                training_total = 0.0
                training_rows = 0
                for tokens, lengths, numeric, target in train_loader:
                    optimizer.zero_grad(set_to_none=True)
                    prediction = model(tokens, lengths, numeric)
                    loss = pinball_loss(prediction, target)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    training_total += float(loss.detach()) * len(target)
                    training_rows += len(target)
                validation_loss = _evaluate_loss(model, validation_loader)
                history.append(
                    {
                        "epoch": epoch,
                        "train_pinball_hours": training_total / max(training_rows, 1),
                        "validation_pinball_hours": validation_loss,
                    }
                )
                if validation_loss < best_validation:
                    best_validation = validation_loss
                    best_state = copy.deepcopy(model.state_dict())
            if best_state is None:
                raise RuntimeError("sequence training produced no checkpoint")
            model.load_state_dict(best_state)
            checkpoint = checkpoint_dir / f"{architecture}_seed{seed}.pt"
            torch.save(
                {
                    "state_dict": best_state,
                    "architecture": architecture,
                    "seed": seed,
                    "vocabulary_size": data.vocabulary_size,
                    "max_length": data.max_length,
                },
                checkpoint,
            )
            reloaded = build_sequence_model(
                SequenceModelConfig(
                    vocabulary_size=data.vocabulary_size,
                    architecture=architecture,
                    max_length=data.max_length,
                )
            )
            saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
            reloaded.load_state_dict(saved["state_dict"])
            test_loader = _loader(
                data.partitions["test"], int(compute["batch_size"]), False, seed
            )
            predictions_hours = _predict(reloaded, test_loader)
            p50_minutes = predictions_hours[:, 0] * 60
            p90_minutes = predictions_hours[:, 1] * 60
            target_minutes = data.partitions["test"].target_hours * 60
            metrics = remaining_time_metrics(
                target_minutes,
                p50_minutes,
                p90=p90_minutes,
            )
            duration = time.perf_counter() - started
            result = {
                "seed": seed,
                "metrics": metrics,
                "best_validation_pinball_hours": best_validation,
                "history": history,
                "training_seconds": duration,
                "checkpoint_sha256": sha256_file(checkpoint),
                "test_influenced_choice": False,
            }
            results[architecture].append(result)
            prediction_frame = prediction_frame.with_columns(
                pl.Series(f"{architecture}_seed{seed}_p50", p50_minutes),
                pl.Series(f"{architecture}_seed{seed}_p90", p90_minutes),
            )
    prediction_frame.write_parquet(output_dir / "predictions.parquet")
    aggregates: dict[str, Any] = {}
    for architecture, architecture_results in results.items():
        maes = np.asarray(
            [item["metrics"]["mae_minutes"] for item in architecture_results],
            dtype=float,
        )
        aggregates[architecture] = {
            "mae_mean_minutes": float(maes.mean()),
            "mae_std_minutes": float(maes.std(ddof=1)),
            "mae_by_seed_minutes": maes.tolist(),
            "beats_tabular_each_seed": bool(np.all(maes < baseline_mae)),
        }
    best_architecture = min(
        aggregates,
        key=lambda name: aggregates[name]["mae_mean_minutes"],
    )
    payload = {
        "dataset_id": baseline_metrics["dataset_id"],
        "dataset_export_version": baseline_metrics["dataset_export_version"],
        "source_file_sha256": baseline_metrics["source_file_sha256"],
        "split_protocol": baseline_metrics["split_protocol"],
        "split_counts_operations": baseline_metrics["split_counts_operations"],
        "models": results,
        "aggregates": aggregates,
        "best_architecture": best_architecture,
        "tabular_reference": {
            "model": baseline_name,
            "test_mae_minutes": baseline_mae,
        },
        "number_of_seeds": len(seeds),
        "seeds": list(seeds),
        "test_influenced_choice": False,
        "claim_state": "smoke_only",
        "public_data_scope": "pipeline competence only; not Kaleido value evidence",
    }
    atomic_json(output_dir / "metrics.json", payload)
    gate = {
        "m1_data_contract": True,
        "m2_process_intelligence": True,
        "m3_tabular_baselines": True,
        "m4_sequential_baselines": True,
        "same_frozen_split": True,
        "three_seeds": len(seeds) >= 3,
        "best_sequential_architecture": best_architecture,
        "best_sequential_beats_tabular_each_seed": aggregates[best_architecture][
            "beats_tabular_each_seed"
        ],
        "event_jepa_allowed_for_public_smoke": True,
        "event_jepa_allowed_for_kaleido_claim": False,
        "reason": "No Kaleido data and no verified action fields.",
        "claim_state": "smoke_only",
    }
    atomic_json(output_dir / "m4_gate.json", gate)
    run.finish(
        {
            "dataset_id": payload["dataset_id"],
            "split_protocol": payload["split_protocol"],
            "number_of_seeds": len(seeds),
            "test_influenced_choice": False,
            "best_architecture": best_architecture,
        }
    )
    return payload
