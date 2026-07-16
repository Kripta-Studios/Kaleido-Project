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

from flowtwin.baselines.process_transformer import pinball_loss, require_torch
from flowtwin.evaluation.remaining_time import remaining_time_metrics
from flowtwin.models.contracts import EventJEPAConfig
from flowtwin.models.event_jepa import build_event_jepa, event_jepa_loss
from flowtwin.models.heads import build_quantile_head
from flowtwin.models.uncertainty import embedding_diagnostics
from flowtwin.provenance import RunContext, atomic_json, sha256_file
from flowtwin.sequence_training import (
    SequenceData,
    SequencePartition,
    _load_events,
    build_sequence_data,
)


@dataclass(frozen=True)
class EventJEPAPartition:
    context: SequencePartition
    target_tokens: np.ndarray
    target_lengths: np.ndarray
    target_numeric: np.ndarray


@dataclass(frozen=True)
class EventJEPAData:
    partitions: dict[str, EventJEPAPartition]
    vocabulary_size: int
    max_length: int
    horizon_labels: tuple[str, ...]


def _target_view(
    trace: list[tuple[Any, str]],
    target_count: int,
    sequence_data: SequenceData,
) -> tuple[np.ndarray, int, np.ndarray]:
    target_trace = trace[:target_count]
    encoded = [
        sequence_data.vocabulary.encode(activity) for _, activity in target_trace
    ][-sequence_data.max_length :]
    tokens = np.zeros(sequence_data.max_length, dtype=np.int64)
    tokens[: len(encoded)] = encoded
    first_time = target_trace[0][0]
    last_time = target_trace[-1][0]
    previous_time = target_trace[-2][0] if len(target_trace) > 1 else last_time
    numeric = np.asarray(
        [
            max(0.0, (last_time - first_time).total_seconds() / 86400),
            np.log1p(len(target_trace)),
            max(0.0, (last_time - previous_time).total_seconds() / 3600),
        ],
        dtype=np.float32,
    )
    return tokens, len(encoded), numeric


def build_event_jepa_data(
    source_path: Path,
    baseline_run_dir: Path,
    *,
    max_length: int,
    horizon_offsets: tuple[int, ...] = (1, 2, -1),
    horizon_labels: tuple[str, ...] = ("next_event", "two_events", "completion"),
) -> EventJEPAData:
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
        cutoff = row["prediction_cutoff"]
        trace = event_groups[operation_id]
        observed_count = int(row["prefix_events"])
        if observed_count < 1 or observed_count >= len(trace):
            continue
        views = [
            _target_view(
                trace,
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
        buckets[partition]["cutoffs"].append(cutoff)
    partitions: dict[str, EventJEPAPartition] = {}
    for name, bucket in buckets.items():
        context = sequence_data.partitions[name]
        if context.operation_ids != bucket["operations"] or context.cutoffs != bucket["cutoffs"]:
            raise RuntimeError("Event-JEPA targets are not aligned with causal prefixes")
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


def _loader(
    partition: EventJEPAPartition,
    batch_size: int,
    *,
    shuffle: bool,
    seed: int,
) -> Any:
    torch = require_torch()
    dataset = torch.utils.data.TensorDataset(
        torch.from_numpy(partition.context.tokens),
        torch.from_numpy(partition.context.lengths),
        torch.from_numpy(partition.context.numeric),
        torch.from_numpy(partition.target_tokens),
        torch.from_numpy(partition.target_lengths),
        torch.from_numpy(partition.target_numeric),
        torch.from_numpy(partition.context.target_hours),
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
    config: EventJEPAConfig,
    step: int,
    *,
    use_sigreg: bool = True,
) -> float:
    torch = require_torch()
    total = 0.0
    rows = 0
    model.eval()
    with torch.no_grad():
        for tokens, lengths, numeric, target_tokens, target_lengths, target_numeric, _ in loader:
            context, targets, predictions = model(
                tokens, lengths, numeric, target_tokens, target_lengths, target_numeric
            )
            loss, _ = event_jepa_loss(
                context,
                targets,
                predictions,
                config=config,
                step=step,
                use_sigreg=use_sigreg,
            )
            total += float(loss) * len(tokens)
            rows += len(tokens)
    return total / max(rows, 1)


def _extract(model: Any, loader: Any) -> tuple[np.ndarray, np.ndarray]:
    torch = require_torch()
    values: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for tokens, lengths, numeric, _, _, _, target in loader:
            values.append(model.encode(tokens, lengths, numeric).cpu().numpy())
            targets.append(target.cpu().numpy())
    return np.concatenate(values), np.concatenate(targets)


def _train_frozen_probe(
    extracted: dict[str, tuple[np.ndarray, np.ndarray]],
    *,
    latent_size: int,
    seed: int,
    epochs: int,
    batch_size: int,
) -> tuple[Any, np.ndarray, float]:
    torch = require_torch()
    torch.manual_seed(seed)
    head = build_quantile_head(latent_size)
    optimizer = torch.optim.AdamW(head.parameters(), lr=3e-3, weight_decay=1e-4)
    train_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(
            torch.from_numpy(extracted["train"][0]).float(),
            torch.from_numpy(extracted["train"][1]).float(),
        ),
        batch_size=batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),
    )
    validation_x = torch.from_numpy(extracted["validation"][0]).float()
    validation_y = torch.from_numpy(extracted["validation"][1]).float()
    best_loss = float("inf")
    best_state: dict[str, Any] | None = None
    for _ in range(epochs):
        head.train()
        for latent, target in train_loader:
            optimizer.zero_grad(set_to_none=True)
            loss = pinball_loss(head(latent), target)
            loss.backward()
            optimizer.step()
        head.eval()
        with torch.no_grad():
            validation_loss = float(pinball_loss(head(validation_x), validation_y))
        if validation_loss < best_loss:
            best_loss = validation_loss
            best_state = copy.deepcopy(head.state_dict())
    if best_state is None:
        raise RuntimeError("frozen Event-JEPA probe produced no checkpoint")
    head.load_state_dict(best_state)
    with torch.no_grad():
        prediction = head(torch.from_numpy(extracted["test"][0]).float()).cpu().numpy()
    return best_state, prediction, best_loss


def _train_finetuned_probe(
    reference: Any,
    data: EventJEPAData,
    *,
    latent_size: int,
    seed: int,
    epochs: int,
    batch_size: int,
) -> tuple[dict[str, Any], np.ndarray, float]:
    torch = require_torch()
    model = copy.deepcopy(reference)
    head = build_quantile_head(latent_size)
    optimizer = torch.optim.AdamW(
        [
            {"params": model.encoder.parameters(), "lr": 1e-4},
            {"params": head.parameters(), "lr": 1e-3},
        ],
        weight_decay=1e-4,
    )
    loaders = {
        name: _loader(
            partition,
            batch_size,
            shuffle=name == "train",
            seed=seed,
        )
        for name, partition in data.partitions.items()
    }
    best_loss = float("inf")
    best_state: dict[str, Any] | None = None
    for _ in range(epochs):
        model.train()
        head.train()
        for tokens, lengths, numeric, _, _, _, target in loaders["train"]:
            optimizer.zero_grad(set_to_none=True)
            loss = pinball_loss(head(model.encode(tokens, lengths, numeric)), target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.encoder.parameters(), 1.0)
            optimizer.step()
        model.eval()
        head.eval()
        total = 0.0
        rows = 0
        with torch.no_grad():
            for tokens, lengths, numeric, _, _, _, target in loaders["validation"]:
                loss = pinball_loss(head(model.encode(tokens, lengths, numeric)), target)
                total += float(loss) * len(target)
                rows += len(target)
        validation_loss = total / max(rows, 1)
        if validation_loss < best_loss:
            best_loss = validation_loss
            best_state = {
                "encoder": copy.deepcopy(model.encoder.state_dict()),
                "head": copy.deepcopy(head.state_dict()),
            }
    if best_state is None:
        raise RuntimeError("fine-tuned Event-JEPA probe produced no checkpoint")
    model.encoder.load_state_dict(best_state["encoder"])
    head.load_state_dict(best_state["head"])
    predictions: list[np.ndarray] = []
    model.eval()
    head.eval()
    with torch.no_grad():
        for tokens, lengths, numeric, _, _, _, _ in loaders["test"]:
            predictions.append(head(model.encode(tokens, lengths, numeric)).cpu().numpy())
    return best_state, np.concatenate(predictions), best_loss


def train_event_jepa(
    source_path: Path,
    baseline_run_dir: Path,
    sequence_run_dir: Path,
    config_path: Path,
    output_dir: Path,
) -> dict[str, Any]:
    for label, run_dir in (
        ("baseline", baseline_run_dir),
        ("sequence", sequence_run_dir),
    ):
        if (run_dir / "INVALIDATED.md").is_file():
            raise RuntimeError(f"refusing invalidated {label} run: {run_dir}")
    resolved = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    compute = resolved["compute"]
    seeds = tuple(int(seed) for seed in compute["seeds"])
    baseline = json.loads((baseline_run_dir / "metrics.json").read_text(encoding="utf-8"))
    sequence = json.loads((sequence_run_dir / "metrics.json").read_text(encoding="utf-8"))
    gate = json.loads((sequence_run_dir / "m4_gate.json").read_text(encoding="utf-8"))
    if not gate.get("event_jepa_allowed_for_public_smoke"):
        raise RuntimeError("M4 gate does not allow the Event-JEPA public smoke")
    run = RunContext.start(
        output_dir,
        ["flowtwin", "train-event-jepa", str(source_path)],
        str(resolved["claim_state"]),
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "config_resolved.yaml").write_text(
        config_path.read_text(encoding="utf-8"), encoding="utf-8"
    )
    for name in ("data_manifest.json", "split_manifest.json", "leakage_report.json"):
        (output_dir / name).write_bytes((baseline_run_dir / name).read_bytes())
    data = build_event_jepa_data(
        source_path,
        baseline_run_dir,
        max_length=int(compute["max_sequence_length"]),
    )
    model_config = EventJEPAConfig(
        vocabulary_size=data.vocabulary_size,
        max_length=data.max_length,
        hidden_size=int(compute["hidden_size"]),
        latent_size=int(compute["latent_size"]),
        layers=int(compute["layers"]),
        attention_heads=int(compute["attention_heads"]),
        horizon_count=len(data.horizon_labels),
        sigreg_slices=int(compute["sigreg_slices"]),
        sigreg_weight=float(compute["sigreg_weight"]),
    )
    torch = require_torch()
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    runs: list[dict[str, Any]] = []
    predictions = pl.DataFrame(
        {
            "operation_id": data.partitions["test"].context.operation_ids,
            "prediction_cutoff": data.partitions["test"].context.cutoffs,
            "remaining_minutes": data.partitions["test"].context.target_hours * 60,
        }
    )
    for seed in seeds:
        torch.manual_seed(seed)
        np.random.seed(seed)
        model = build_event_jepa(model_config)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
        train_loader = _loader(
            data.partitions["train"], int(compute["batch_size"]), shuffle=True, seed=seed
        )
        validation_loader = _loader(
            data.partitions["validation"],
            int(compute["batch_size"]),
            shuffle=False,
            seed=seed,
        )
        best_loss = float("inf")
        best_state: dict[str, Any] | None = None
        history: list[dict[str, float | int]] = []
        step = 0
        started = time.perf_counter()
        checkpoint = checkpoint_dir / f"event_jepa_seed{seed}.pt"
        resumed_checkpoint = False
        if checkpoint.is_file():
            candidate = torch.load(checkpoint, map_location="cpu", weights_only=True)
            resumed_checkpoint = bool(
                candidate.get("config") == model_config.__dict__
                and tuple(candidate.get("horizons", ())) == data.horizon_labels
                and candidate.get("seed") == seed
            )
            if resumed_checkpoint:
                best_state = candidate["state_dict"]
                best_loss = float(candidate["best_validation_loss"])
                history = list(candidate["pretraining_history"])
        if not resumed_checkpoint:
            for epoch in range(1, int(compute["pretrain_epochs"]) + 1):
                model.train()
                total = 0.0
                rows = 0
                for (
                    tokens,
                    lengths,
                    numeric,
                    target_tokens,
                    target_lengths,
                    target_numeric,
                    _,
                ) in train_loader:
                    optimizer.zero_grad(set_to_none=True)
                    context, targets, projected = model(
                        tokens, lengths, numeric, target_tokens, target_lengths, target_numeric
                    )
                    loss, _ = event_jepa_loss(
                        context, targets, projected, config=model_config, step=step
                    )
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    total += float(loss.detach()) * len(tokens)
                    rows += len(tokens)
                    step += 1
                validation_loss = _jepa_validation(
                    model, validation_loader, model_config, step
                )
                history.append(
                    {
                        "epoch": epoch,
                        "train_loss": total / max(rows, 1),
                        "validation_loss": validation_loss,
                    }
                )
                print(
                    f"event_jepa seed={seed} epoch={epoch} "
                    f"validation_loss={validation_loss:.6f}",
                    flush=True,
                )
                if validation_loss < best_loss:
                    best_loss = validation_loss
                    best_state = copy.deepcopy(model.state_dict())
        if best_state is None:
            raise RuntimeError("Event-JEPA pretraining produced no checkpoint")
        model.load_state_dict(best_state)
        if not resumed_checkpoint:
            torch.save(
                {
                    "state_dict": best_state,
                    "config": model_config.__dict__,
                    "horizons": data.horizon_labels,
                    "seed": seed,
                    "best_validation_loss": best_loss,
                    "pretraining_history": history,
                },
                checkpoint,
            )
        reloaded = build_event_jepa(model_config)
        reloaded.load_state_dict(
            torch.load(checkpoint, map_location="cpu", weights_only=True)["state_dict"]
        )
        extracted = {
            name: _extract(
                reloaded,
                _loader(partition, int(compute["batch_size"]), shuffle=False, seed=seed),
            )
            for name, partition in data.partitions.items()
        }
        diagnostics = embedding_diagnostics(extracted["validation"][0])
        frozen_state, frozen_prediction, frozen_validation = _train_frozen_probe(
            extracted,
            latent_size=model_config.latent_size,
            seed=seed,
            epochs=int(compute["frozen_head_epochs"]),
            batch_size=int(compute["batch_size"]),
        )
        finetuned_state, finetuned_prediction, finetuned_validation = _train_finetuned_probe(
            reloaded,
            data,
            latent_size=model_config.latent_size,
            seed=seed,
            epochs=int(compute["finetune_epochs"]),
            batch_size=int(compute["batch_size"]),
        )
        torch.save(frozen_state, checkpoint_dir / f"frozen_head_seed{seed}.pt")
        torch.save(finetuned_state, checkpoint_dir / f"finetuned_head_seed{seed}.pt")
        target_minutes = extracted["test"][1] * 60
        variants: dict[str, Any] = {}
        for name, prediction, validation_loss in (
            ("frozen", frozen_prediction, frozen_validation),
            ("finetuned", finetuned_prediction, finetuned_validation),
        ):
            p50 = prediction[:, 0] * 60
            p90 = prediction[:, 1] * 60
            variants[name] = {
                "best_validation_pinball_hours": validation_loss,
                "metrics": remaining_time_metrics(target_minutes, p50, p90=p90),
            }
            predictions = predictions.with_columns(
                pl.Series(f"{name}_seed{seed}_p50", p50),
                pl.Series(f"{name}_seed{seed}_p90", p90),
            )
        runs.append(
            {
                "seed": seed,
                "pretraining_best_validation_loss": best_loss,
                "pretraining_history": history,
                "embedding_diagnostics_validation": diagnostics,
                "variants": variants,
                "checkpoint_sha256": sha256_file(checkpoint),
                "training_seconds": time.perf_counter() - started,
                "resumed_checkpoint": resumed_checkpoint,
                "test_influenced_choice": False,
            }
        )
    predictions.write_parquet(output_dir / "predictions.parquet")
    aggregates: dict[str, Any] = {}
    for variant in ("frozen", "finetuned"):
        maes = np.asarray(
            [item["variants"][variant]["metrics"]["mae_minutes"] for item in runs]
        )
        validation_losses = np.asarray(
            [item["variants"][variant]["best_validation_pinball_hours"] for item in runs]
        )
        aggregates[variant] = {
            "mae_mean_minutes": float(maes.mean()),
            "mae_std_minutes": float(maes.std(ddof=1)),
            "mae_by_seed_minutes": maes.tolist(),
            "validation_pinball_mean_hours": float(validation_losses.mean()),
        }
    selected = min(
        aggregates,
        key=lambda name: aggregates[name]["validation_pinball_mean_hours"],
    )
    baseline_mae = float(baseline["selected_model_test"]["mae_minutes"])
    sequence_name = str(sequence["best_architecture"])
    sequence_mae = float(sequence["aggregates"][sequence_name]["mae_mean_minutes"])
    payload: dict[str, Any] = {
        "dataset_id": baseline["dataset_id"],
        "dataset_export_version": baseline["dataset_export_version"],
        "source_file_sha256": baseline["source_file_sha256"],
        "split_protocol": baseline["split_protocol"],
        "split_counts_operations": baseline["split_counts_operations"],
        "horizons": list(data.horizon_labels),
        "action_conditioned": False,
        "world_model_claim": False,
        "seeds": list(seeds),
        "number_of_seeds": len(seeds),
        "runs": runs,
        "aggregates": aggregates,
        "selected_variant_validation_only": selected,
        "references": {
            "tabular_model": baseline["model_selection"]["selected_model"],
            "tabular_test_mae_minutes": baseline_mae,
            "sequential_model": sequence_name,
            "sequential_mean_test_mae_minutes": sequence_mae,
        },
        "test_influenced_choice": False,
        "claim_state": "smoke_only",
        "public_data_scope": "technical MVP only; not Kaleido value or causal evidence",
    }
    atomic_json(output_dir / "metrics.json", payload)
    stable = all(not item["embedding_diagnostics_validation"]["collapsed"] for item in runs)
    selected_mean = float(aggregates[selected]["mae_mean_minutes"])
    m5_gate = {
        "m4_gate_passed_for_public_smoke": True,
        "three_seeds": len(seeds) >= 3,
        "embedding_stable_each_seed": stable,
        "selected_variant": selected,
        "selected_mean_mae_minutes": selected_mean,
        "beats_tabular_mean_mae": selected_mean < baseline_mae,
        "beats_best_sequential_mean_mae": selected_mean < sequence_mae,
        "promote_as_public_representation_model": len(seeds) >= 3
        and stable
        and selected_mean < min(baseline_mae, sequence_mae),
        "promote_as_kaleido_world_model": False,
        "reason_kaleido_blocked": "No Kaleido trajectories or verified operational actions.",
        "claim_state": "smoke_only",
    }
    atomic_json(output_dir / "m5_gate.json", m5_gate)
    atomic_json(
        output_dir / "calibration.json",
        {
            "method": "direct_quantile_heads_no_posthoc_calibration",
            "variants": {
                variant: [
                    item["variants"][variant]["metrics"].get("p90_quantile_coverage")
                    for item in runs
                ]
                for variant in ("frozen", "finetuned")
            },
        },
    )
    _write_reports(output_dir, payload, m5_gate)
    run.finish(
        {
            "dataset_id": payload["dataset_id"],
            "split_protocol": payload["split_protocol"],
            "number_of_seeds": len(seeds),
            "selected_variant": selected,
            "test_influenced_choice": False,
        }
    )
    return payload


def _write_reports(output_dir: Path, metrics: dict[str, Any], gate: dict[str, Any]) -> None:
    selected = metrics["selected_variant_validation_only"]
    aggregate = metrics["aggregates"][selected]
    limitation = (
        "Public obfuscated non-port data; action-free representation experiment; no plan "
        "revisions, Kaleido outcomes, action value, ROI or deployment evidence."
    )
    common = [
        f"Dataset/export: `{metrics['dataset_id']}`, {metrics['dataset_export_version']}.",
        f"Source SHA-256: `{metrics['source_file_sha256']}`.",
        f"Split: {metrics['split_protocol']}.",
        f"Seeds: {metrics['seeds']}.",
        f"Variant selected on validation: `{selected}`.",
        "Test influenced a choice: no.",
        f"Mean test MAE: {aggregate['mae_mean_minutes']:.2f} minutes.",
        f"Between-seed MAE SD: {aggregate['mae_std_minutes']:.2f} minutes.",
        "Action conditioned: no. World-model claim: no.",
    ]
    (output_dir / "model_card.md").write_text(
        "\n".join(
            [
                "# Model card — action-free Event-JEPA public smoke",
                "",
                *[f"- {line}" for line in common],
                "",
                "## Limitations",
                "",
                limitation,
            ]
        ),
        encoding="utf-8",
    )
    (output_dir / "report.md").write_text(
        "\n".join(
            [
                "# Event-JEPA experiment report",
                "",
                "## Hypothesis",
                "",
                "Future-latent self-supervision improves remaining-time prediction over "
                "supervised tabular and sequence baselines.",
                "",
                "## Changes",
                "",
                "Added multi-horizon action-free Event-JEPA with SIGReg, frozen and fine-tuned "
                "quantile probes, collapse diagnostics and three seeds.",
                "",
                "## Tests and evidence",
                "",
                *common,
                f"Public promotion gate: {gate['promote_as_public_representation_model']}.",
                "",
                "## Limitations",
                "",
                limitation,
                "",
                "## Next falsifiable step",
                "",
                "Run no-SIGReg, one-horizon and shuffled-temporal-pair ablations, then the "
                "separate synthetic-action recovery benchmark.",
            ]
        ),
        encoding="utf-8",
    )
