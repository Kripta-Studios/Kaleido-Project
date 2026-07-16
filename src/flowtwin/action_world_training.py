from __future__ import annotations

import copy
import json
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import yaml

from flowtwin.baselines.process_transformer import require_torch
from flowtwin.evaluation.remaining_time import remaining_time_metrics
from flowtwin.event_jepa_training import (
    _train_frozen_probe,
    build_event_jepa_data,
)
from flowtwin.models.action_event_jepa import build_action_event_jepa
from flowtwin.models.contracts import EventJEPAConfig
from flowtwin.models.event_jepa import event_jepa_loss
from flowtwin.models.uncertainty import embedding_diagnostics
from flowtwin.provenance import RunContext, atomic_json, sha256_file
from flowtwin.sequence_training import SequencePartition
from flowtwin.simulation.synthetic_actions import ACTION_NAMES

MODES = (
    "correct_action",
    "shuffled_action",
    "current_prefix_only",
    "context_only",
    "action_only",
)


@dataclass(frozen=True)
class ActionWorldPartition:
    context: SequencePartition
    target_tokens: np.ndarray
    target_lengths: np.ndarray
    target_numeric: np.ndarray
    action_codes: np.ndarray
    action_context: np.ndarray
    target_hours: np.ndarray


@dataclass(frozen=True)
class ActionWorldData:
    partitions: dict[str, ActionWorldPartition]
    vocabulary_size: int
    max_length: int


def build_action_world_data(
    source_path: Path,
    baseline_run_dir: Path,
    overlay_path: Path,
    *,
    max_length: int,
) -> ActionWorldData:
    completion = build_event_jepa_data(
        source_path,
        baseline_run_dir,
        max_length=max_length,
        horizon_offsets=(-1,),
        horizon_labels=("synthetic_completion",),
    )
    overlay = pl.read_parquet(overlay_path).sort(
        ["operation_id", "prediction_cutoff", "prefix_events"]
    )
    partitions: dict[str, ActionWorldPartition] = {}
    for name, completion_partition in completion.partitions.items():
        rows = overlay.filter(pl.col("partition") == name)
        operations = list(map(str, rows["operation_id"].to_list()))
        cutoffs = rows["prediction_cutoff"].to_list()
        if (
            completion_partition.context.operation_ids != operations
            or completion_partition.context.cutoffs != cutoffs
        ):
            raise RuntimeError("synthetic action overlay is not aligned to causal prefixes")
        target_numeric = completion_partition.target_numeric[:, 0, :].copy()
        synthetic_remaining = rows["synthetic_remaining_minutes"].to_numpy().astype(float)
        elapsed = rows["elapsed_minutes"].to_numpy().astype(float)
        target_numeric[:, 0] = (elapsed + synthetic_remaining) / 1440
        target_numeric[:, 2] = synthetic_remaining / 1440
        action_codes = np.asarray(
            [ACTION_NAMES.index(str(value)) for value in rows["synthetic_action"]],
            dtype=np.int64,
        )
        action_context = np.column_stack(
            [
                rows["parallel_station_available"].cast(pl.Float32).to_numpy(),
                rows["constraint_active"].cast(pl.Float32).to_numpy(),
            ]
        ).astype(np.float32)
        partitions[name] = ActionWorldPartition(
            context=completion_partition.context,
            target_tokens=completion_partition.target_tokens[:, 0, :],
            target_lengths=completion_partition.target_lengths[:, 0],
            target_numeric=target_numeric,
            action_codes=action_codes,
            action_context=action_context,
            target_hours=(synthetic_remaining / 60).astype(np.float32),
        )
    return ActionWorldData(
        partitions=partitions,
        vocabulary_size=completion.vocabulary_size,
        max_length=completion.max_length,
    )


def _shuffle_actions(data: ActionWorldData, seed: int) -> ActionWorldData:
    partitions: dict[str, ActionWorldPartition] = {}
    for offset, (name, partition) in enumerate(data.partitions.items()):
        rng = np.random.default_rng(seed + 1009 * (offset + 1))
        partitions[name] = replace(
            partition,
            action_codes=rng.permutation(partition.action_codes),
        )
    return replace(data, partitions=partitions)


def _loader(
    partition: ActionWorldPartition,
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
        torch.from_numpy(partition.action_codes),
        torch.from_numpy(partition.action_context),
        torch.from_numpy(partition.target_tokens),
        torch.from_numpy(partition.target_lengths),
        torch.from_numpy(partition.target_numeric),
        torch.from_numpy(partition.target_hours),
    )
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        generator=torch.Generator().manual_seed(seed),
        num_workers=0,
    )


def _validation(
    model: Any,
    loader: Any,
    config: EventJEPAConfig,
    mode: str,
    step: int,
    regularizer: str,
) -> float:
    torch = require_torch()
    total = 0.0
    rows = 0
    model.eval()
    with torch.no_grad():
        for tokens, lengths, numeric, actions, context, tt, tl, tn, _ in loader:
            state, target, prediction = model(
                tokens, lengths, numeric, actions, context, tt, tl, tn, mode
            )
            loss, _ = event_jepa_loss(
                state,
                target.unsqueeze(1),
                prediction.unsqueeze(1),
                config=config,
                step=step,
                use_sigreg=regularizer == "sigreg",
                use_visreg=regularizer == "visreg",
            )
            total += float(loss) * len(tokens)
            rows += len(tokens)
    return total / max(rows, 1)


def _extract(model: Any, loader: Any, mode: str) -> tuple[np.ndarray, np.ndarray]:
    torch = require_torch()
    values: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for tokens, lengths, numeric, actions, context, _, _, _, target in loader:
            values.append(
                model.predict_state(
                    tokens, lengths, numeric, actions, context, mode
                ).cpu().numpy()
            )
            targets.append(target.cpu().numpy())
    return np.concatenate(values), np.concatenate(targets)


def _run_seed(
    mode: str,
    data: ActionWorldData,
    compute: dict[str, Any],
    seed: int,
    checkpoint_dir: Path,
) -> tuple[dict[str, Any], np.ndarray]:
    torch = require_torch()
    torch.manual_seed(seed)
    np.random.seed(seed)
    effective_data = _shuffle_actions(data, seed) if mode == "shuffled_action" else data
    model_mode = "correct_action" if mode == "shuffled_action" else mode
    regularizer = str(compute.get("regularizer", "sigreg"))
    if regularizer not in {"sigreg", "visreg"}:
        raise ValueError(f"unsupported action Event-JEPA regularizer: {regularizer}")
    config = EventJEPAConfig(
        vocabulary_size=data.vocabulary_size,
        max_length=data.max_length,
        hidden_size=int(compute["hidden_size"]),
        latent_size=int(compute["latent_size"]),
        layers=int(compute["layers"]),
        attention_heads=int(compute["attention_heads"]),
        horizon_count=1,
        sigreg_slices=int(compute["sigreg_slices"]),
        sigreg_weight=float(compute["sigreg_weight"]),
    )
    model = build_action_event_jepa(config, len(ACTION_NAMES))
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    loaders = {
        name: _loader(
            partition,
            int(compute["batch_size"]),
            shuffle=name == "train",
            seed=seed,
        )
        for name, partition in effective_data.partitions.items()
    }
    best_loss = float("inf")
    best_state: dict[str, Any] | None = None
    history: list[dict[str, float | int]] = []
    step = 0
    started = time.perf_counter()
    for epoch in range(1, int(compute["pretrain_epochs"]) + 1):
        model.train()
        total = 0.0
        rows = 0
        for tokens, lengths, numeric, actions, context, tt, tl, tn, _ in loaders["train"]:
            optimizer.zero_grad(set_to_none=True)
            state, target, prediction = model(
                tokens, lengths, numeric, actions, context, tt, tl, tn, model_mode
            )
            loss, _ = event_jepa_loss(
                state,
                target.unsqueeze(1),
                prediction.unsqueeze(1),
                config=config,
                step=step,
                use_sigreg=regularizer == "sigreg",
                use_visreg=regularizer == "visreg",
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total += float(loss.detach()) * len(tokens)
            rows += len(tokens)
            step += 1
        validation_loss = _validation(
            model,
            loaders["validation"],
            config,
            model_mode,
            step,
            regularizer,
        )
        history.append(
            {
                "epoch": epoch,
                "train_loss": total / max(rows, 1),
                "validation_loss": validation_loss,
            }
        )
        print(
            f"action_event_jepa mode={mode} seed={seed} epoch={epoch} "
            f"validation_loss={validation_loss:.6f}",
            flush=True,
        )
        if validation_loss < best_loss:
            best_loss = validation_loss
            best_state = copy.deepcopy(model.state_dict())
    if best_state is None:
        raise RuntimeError(f"action Event-JEPA mode {mode} produced no checkpoint")
    checkpoint = checkpoint_dir / f"{mode}_seed{seed}.pt"
    torch.save(
        {
            "state_dict": best_state,
            "config": config.__dict__,
            "mode": mode,
            "seed": seed,
            "best_validation_loss": best_loss,
            "history": history,
        },
        checkpoint,
    )
    reloaded = build_action_event_jepa(config, len(ACTION_NAMES))
    reloaded.load_state_dict(
        torch.load(checkpoint, map_location="cpu", weights_only=True)["state_dict"]
    )
    extracted = {
        name: _extract(reloaded, loader, model_mode) for name, loader in loaders.items()
    }
    diagnostics = embedding_diagnostics(extracted["validation"][0])
    probe_state, prediction, probe_validation = _train_frozen_probe(
        extracted,
        latent_size=config.latent_size,
        seed=seed,
        epochs=int(compute["frozen_head_epochs"]),
        batch_size=int(compute["batch_size"]),
    )
    torch.save(probe_state, checkpoint_dir / f"{mode}_probe_seed{seed}.pt")
    target_minutes = extracted["test"][1] * 60
    p50 = prediction[:, 0] * 60
    p90 = prediction[:, 1] * 60
    return (
        {
            "seed": seed,
            "pretraining_best_validation_loss": best_loss,
            "probe_best_validation_pinball_hours": probe_validation,
            "pretraining_history": history,
            "embedding_diagnostics_validation": diagnostics,
            "metrics": remaining_time_metrics(target_minutes, p50, p90=p90),
            "checkpoint_sha256": sha256_file(checkpoint),
            "training_seconds": time.perf_counter() - started,
            "test_influenced_choice": False,
            "regularizer": regularizer,
        },
        prediction,
    )


def train_action_event_jepa(
    source_path: Path,
    baseline_run_dir: Path,
    overlay_path: Path,
    config_path: Path,
    output_dir: Path,
) -> dict[str, Any]:
    if (baseline_run_dir / "INVALIDATED.md").is_file():
        raise RuntimeError(f"refusing invalidated baseline run: {baseline_run_dir}")
    resolved = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    compute = resolved["compute"]
    seeds = tuple(int(seed) for seed in compute["seeds"])
    modes = tuple(str(mode) for mode in resolved.get("modes", MODES))
    if not {"correct_action", "shuffled_action"}.issubset(modes):
        raise ValueError("action benchmark requires correct_action and shuffled_action")
    baseline = json.loads((baseline_run_dir / "metrics.json").read_text(encoding="utf-8"))
    run = RunContext.start(
        output_dir,
        ["flowtwin", "train-action-event-jepa", str(source_path), str(overlay_path)],
        str(resolved["claim_state"]),
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "config_resolved.yaml").write_text(
        config_path.read_text(encoding="utf-8"), encoding="utf-8"
    )
    data_manifest = json.loads((baseline_run_dir / "data_manifest.json").read_text())
    data_manifest["synthetic_action_overlay_sha256"] = sha256_file(overlay_path)
    data_manifest["synthetic_action_evidence_only"] = True
    atomic_json(output_dir / "data_manifest.json", data_manifest)
    for name in ("split_manifest.json", "leakage_report.json"):
        (output_dir / name).write_bytes((baseline_run_dir / name).read_bytes())
    data = build_action_world_data(
        source_path,
        baseline_run_dir,
        overlay_path,
        max_length=int(compute["max_sequence_length"]),
    )
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    predictions = pl.DataFrame(
        {
            "operation_id": data.partitions["test"].context.operation_ids,
            "prediction_cutoff": data.partitions["test"].context.cutoffs,
            "synthetic_remaining_minutes": data.partitions["test"].target_hours * 60,
        }
    )
    results: dict[str, list[dict[str, Any]]] = {mode: [] for mode in modes}
    for mode in modes:
        for seed in seeds:
            result, prediction = _run_seed(mode, data, compute, seed, checkpoint_dir)
            results[mode].append(result)
            predictions = predictions.with_columns(
                pl.Series(f"{mode}_seed{seed}_p50", prediction[:, 0] * 60),
                pl.Series(f"{mode}_seed{seed}_p90", prediction[:, 1] * 60),
            )
    predictions.write_parquet(output_dir / "predictions.parquet")
    aggregates: dict[str, Any] = {}
    for mode, mode_results in results.items():
        maes = np.asarray([item["metrics"]["mae_minutes"] for item in mode_results])
        aggregates[mode] = {
            "mae_mean_minutes": float(maes.mean()),
            "mae_std_minutes": float(maes.std(ddof=1)),
            "mae_by_seed_minutes": maes.tolist(),
            "probe_validation_pinball_mean_hours": float(
                np.mean(
                    [item["probe_best_validation_pinball_hours"] for item in mode_results]
                )
            ),
            "p90_coverage_mean": float(
                np.mean(
                    [item["metrics"]["p90_quantile_coverage"] for item in mode_results]
                )
            ),
        }
    paired_wins = [
        correct["metrics"]["mae_minutes"] < shuffled["metrics"]["mae_minutes"]
        for correct, shuffled in zip(
            results["correct_action"], results["shuffled_action"], strict=True
        )
    ]
    correct = aggregates["correct_action"]["mae_mean_minutes"]
    shuffled = aggregates["shuffled_action"]["mae_mean_minutes"]
    payload: dict[str, Any] = {
        "dataset_id": baseline["dataset_id"],
        "dataset_export_version": baseline["dataset_export_version"],
        "source_file_sha256": baseline["source_file_sha256"],
        "synthetic_action_overlay_sha256": sha256_file(overlay_path),
        "split_protocol": baseline["split_protocol"],
        "split_counts_operations": baseline["split_counts_operations"],
        "seeds": list(seeds),
        "number_of_seeds": len(seeds),
        "fixed_modes": list(modes),
        "results": results,
        "aggregates": aggregates,
        "correct_minus_shuffled_mean_mae_minutes": float(correct - shuffled),
        "correct_actions_beat_shuffled_mean": bool(correct < shuffled),
        "correct_actions_beat_shuffled_each_seed": bool(all(paired_wins)),
        "paired_seed_wins": paired_wins,
        "action_source": "generated_at_prediction_cutoff_with_logged_propensity",
        "regularizer": str(compute.get("regularizer", "sigreg")),
        "world_model_scope": "recovery_of_injected_synthetic_transition_only",
        "kaleido_action_claim": False,
        "test_influenced_choice": False,
        "claim_state": "smoke_only",
    }
    atomic_json(output_dir / "metrics.json", payload)
    gate = {
        "three_seeds": len(seeds) >= 3,
        "correct_actions_beat_shuffled_mean": bool(correct < shuffled),
        "correct_actions_beat_shuffled_each_seed": bool(all(paired_wins)),
        "synthetic_action_signal_recovered": bool(len(seeds) >= 3 and all(paired_wins)),
        "promote_as_kaleido_world_model": False,
        "claim_state": "smoke_only",
    }
    atomic_json(output_dir / "action_world_gate.json", gate)
    atomic_json(
        output_dir / "calibration.json",
        {
            "method": "direct_quantile_heads_no_posthoc_calibration",
            "p90_coverage_mean_by_mode": {
                mode: values["p90_coverage_mean"] for mode, values in aggregates.items()
            },
        },
    )
    _write_reports(output_dir, payload, gate)
    run.finish(
        {
            "dataset_id": payload["dataset_id"],
            "number_of_seeds": len(seeds),
            "synthetic_action_signal_recovered": gate["synthetic_action_signal_recovered"],
            "test_influenced_choice": False,
        }
    )
    return payload


def _write_reports(output_dir: Path, metrics: dict[str, Any], gate: dict[str, Any]) -> None:
    evidence = [
        f"Dataset/export: `{metrics['dataset_id']}`, {metrics['dataset_export_version']}.",
        f"Source SHA-256: `{metrics['source_file_sha256']}`.",
        f"Overlay SHA-256: `{metrics['synthetic_action_overlay_sha256']}`.",
        f"Split: {metrics['split_protocol']}.",
        f"Seeds: {metrics['seeds']}.",
        f"Modes: {metrics['fixed_modes']}.",
        "Test influenced a choice: no.",
        f"Correct-action MAE: "
        f"{metrics['aggregates']['correct_action']['mae_mean_minutes']:.2f} minutes.",
        f"Shuffled-action MAE: "
        f"{metrics['aggregates']['shuffled_action']['mae_mean_minutes']:.2f} minutes.",
        f"Correct beats shuffled in every seed: {gate['correct_actions_beat_shuffled_each_seed']}.",
    ]
    limitation = (
        "Actions, propensities and structural effects are generated. This tests only recovery "
        "of an injected transition and cannot prove Kaleido action value, causality, savings, "
        "ROI or deployment readiness."
    )
    (output_dir / "report.md").write_text(
        "\n".join(
            [
                "# Synthetic action-conditioned Event-JEPA report",
                "",
                "## Hypothesis",
                "",
                "A latent transition predictor with the correct generated action stream "
                "beats the same architecture with shuffled actions across seeds.",
                "",
                "## Changes",
                "",
                "Added separate state, action and context channels and a synthetic future-state "
                "target whose timing follows the logged structural action effect.",
                "",
                "## Tests and evidence",
                "",
                *evidence,
                "",
                "## Limitations",
                "",
                limitation,
                "",
                "## Next falsifiable step",
                "",
                "Repeat only after Kaleido supplies timestamped controllable actions and "
                "immutable outcomes; require the same correct-vs-shuffled gate on holdout.",
            ]
        ),
        encoding="utf-8",
    )
    (output_dir / "model_card.md").write_text(
        "\n".join(
            [
                "# Model card - synthetic action-conditioned Event-JEPA",
                "",
                *[f"- {line}" for line in evidence],
                "- Threshold selection: not applicable; fixed modes.",
                "- Claim state: `smoke_only`.",
                "",
                "## What this does not prove",
                "",
                limitation,
            ]
        ),
        encoding="utf-8",
    )
