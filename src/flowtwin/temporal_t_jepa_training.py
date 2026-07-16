from __future__ import annotations

import copy
import json
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import yaml

from flowtwin.baselines.process_transformer import require_torch
from flowtwin.evaluation.remaining_time import remaining_time_metrics
from flowtwin.event_jepa_training import (
    EventJEPAData,
    EventJEPAPartition,
    _loader,
    _train_frozen_probe,
)
from flowtwin.models.contracts import EventJEPAConfig
from flowtwin.models.temporal_t_jepa import (
    build_temporal_t_jepa,
    temporal_t_jepa_loss,
)
from flowtwin.models.uncertainty import embedding_diagnostics
from flowtwin.provenance import RunContext, atomic_json, sha256_file
from flowtwin.temporal_jepa_data import build_disjoint_event_jepa_data, write_embeddings


def _shuffled_training_targets(data: EventJEPAData, seed: int) -> EventJEPAData:
    training = data.partitions["train"]
    permutation = np.random.default_rng(seed).permutation(len(training.context.operation_ids))
    shuffled = EventJEPAPartition(
        context=training.context,
        target_tokens=training.target_tokens[permutation],
        target_lengths=training.target_lengths[permutation],
        target_numeric=training.target_numeric[permutation],
    )
    return replace(data, partitions={**data.partitions, "train": shuffled})


def _extract(model: Any, data: EventJEPAData, batch_size: int, seed: int) -> dict[str, Any]:
    torch = require_torch()
    extracted: dict[str, Any] = {}
    model.eval()
    with torch.no_grad():
        for name, partition in data.partitions.items():
            values: list[np.ndarray] = []
            targets: list[np.ndarray] = []
            for tokens, lengths, numeric, _, _, _, target in _loader(
                partition,
                batch_size,
                shuffle=False,
                seed=seed,
            ):
                values.append(model.encode(tokens, lengths, numeric).cpu().numpy())
                targets.append(target.cpu().numpy())
            extracted[name] = (np.concatenate(values), np.concatenate(targets))
    return extracted


def _validation_loss(
    model: Any,
    data: EventJEPAData,
    config: EventJEPAConfig,
    batch_size: int,
    seed: int,
    step: int,
    regularizer: str,
) -> float:
    torch = require_torch()
    total = 0.0
    rows = 0
    model.eval()
    with torch.no_grad():
        for tokens, lengths, numeric, target_tokens, target_lengths, target_numeric, _ in _loader(
            data.partitions["validation"],
            batch_size,
            shuffle=False,
            seed=seed,
        ):
            context, targets, predictions = model(
                tokens,
                lengths,
                numeric,
                target_tokens,
                target_lengths,
                target_numeric,
            )
            loss, _ = temporal_t_jepa_loss(
                context,
                targets,
                predictions,
                config=config,
                step=step,
                regularizer=regularizer,
            )
            total += float(loss) * len(tokens)
            rows += len(tokens)
    return total / max(rows, 1)


def _run_variant(
    variant: str,
    data: EventJEPAData,
    compute: dict[str, Any],
    seed: int,
    checkpoint_dir: Path,
    embedding_dir: Path,
    *,
    regularizer: str,
    register_token: bool,
    save_embeddings: bool,
) -> tuple[dict[str, Any], np.ndarray]:
    torch = require_torch()
    torch.manual_seed(seed)
    np.random.seed(seed)
    config = EventJEPAConfig(
        vocabulary_size=data.vocabulary_size,
        max_length=data.max_length,
        hidden_size=int(compute["hidden_size"]),
        latent_size=int(compute["latent_size"]),
        layers=int(compute["layers"]),
        attention_heads=int(compute["attention_heads"]),
        dropout=float(compute.get("dropout", 0.0)),
        horizon_count=len(data.horizon_labels),
        sigreg_slices=int(compute["sigreg_slices"]),
        sigreg_weight=float(compute["regularizer_weight"]),
    )
    momentum = float(compute["ema_momentum"])
    model = build_temporal_t_jepa(
        config,
        ema_momentum=momentum,
        register_token=register_token,
    )
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=float(compute["learning_rate"]),
        weight_decay=float(compute["weight_decay"]),
    )
    batch_size = int(compute["batch_size"])
    checkpoint = checkpoint_dir / f"{variant}_seed{seed}.pt"
    history: list[dict[str, float | int]] = []
    best_loss = float("inf")
    best_state: dict[str, Any] | None = None
    step = 0
    started = time.perf_counter()
    train_loader = _loader(data.partitions["train"], batch_size, shuffle=True, seed=seed)
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
            context, targets, predictions = model(
                tokens,
                lengths,
                numeric,
                target_tokens,
                target_lengths,
                target_numeric,
            )
            loss, _ = temporal_t_jepa_loss(
                context,
                targets,
                predictions,
                config=config,
                step=step,
                regularizer=regularizer,
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [parameter for parameter in model.parameters() if parameter.requires_grad],
                1.0,
            )
            optimizer.step()
            model.update_target()
            total += float(loss.detach()) * len(tokens)
            rows += len(tokens)
            step += 1
        validation = _validation_loss(
            model,
            data,
            config,
            batch_size,
            seed,
            step,
            regularizer,
        )
        history.append(
            {
                "epoch": epoch,
                "train_loss": total / max(rows, 1),
                "validation_loss": validation,
            }
        )
        print(
            f"temporal_t_jepa variant={variant} seed={seed} epoch={epoch} "
            f"validation_loss={validation:.6f}",
            flush=True,
        )
        if validation < best_loss:
            best_loss = validation
            best_state = copy.deepcopy(model.state_dict())
    if best_state is None:
        raise RuntimeError(f"Temporal T-JEPA variant {variant} produced no checkpoint")
    model.load_state_dict(best_state)
    torch.save(
        {
            "state_dict": best_state,
            "config": config.__dict__,
            "variant": variant,
            "seed": seed,
            "horizons": data.horizon_labels,
            "regularizer": regularizer,
            "register_token": register_token,
            "ema_momentum": momentum,
            "best_validation_loss": best_loss,
            "history": history,
        },
        checkpoint,
    )
    extracted = _extract(model, data, batch_size, seed)
    diagnostics = embedding_diagnostics(extracted["validation"][0])
    probe_state, prediction, probe_validation = _train_frozen_probe(
        extracted,
        latent_size=config.latent_size,
        seed=seed,
        epochs=int(compute["frozen_head_epochs"]),
        batch_size=batch_size,
    )
    probe_checkpoint = checkpoint_dir / f"{variant}_probe_seed{seed}.pt"
    torch.save(probe_state, probe_checkpoint)
    if save_embeddings:
        write_embeddings(
            embedding_dir / f"{variant}_seed{seed}.parquet",
            data,
            {name: values[0] for name, values in extracted.items()},
            column_prefix="t",
        )
    target_minutes = extracted["test"][1] * 60
    p50 = prediction[:, 0] * 60
    p90 = prediction[:, 1] * 60
    result = {
        "seed": seed,
        "variant": variant,
        "regularizer": regularizer,
        "register_token": register_token,
        "pretraining_best_validation_loss": best_loss,
        "probe_best_validation_pinball_hours": probe_validation,
        "pretraining_history": history,
        "embedding_diagnostics_validation": diagnostics,
        "metrics": remaining_time_metrics(target_minutes, p50, p90=p90),
        "checkpoint_sha256": sha256_file(checkpoint),
        "probe_checkpoint_sha256": sha256_file(probe_checkpoint),
        "training_seconds": time.perf_counter() - started,
        "test_influenced_choice": False,
    }
    return result, prediction


def _aggregate(results: list[dict[str, Any]]) -> dict[str, Any]:
    maes = np.asarray([item["metrics"]["mae_minutes"] for item in results], dtype=float)
    validation = np.asarray(
        [item["probe_best_validation_pinball_hours"] for item in results],
        dtype=float,
    )
    return {
        "mae_mean_minutes": float(maes.mean()),
        "mae_std_minutes": float(maes.std(ddof=1)),
        "mae_by_seed_minutes": maes.tolist(),
        "validation_pinball_mean_hours": float(validation.mean()),
        "effective_rank_mean": float(
            np.mean(
                [
                    item["embedding_diagnostics_validation"]["effective_rank"]
                    for item in results
                ]
            )
        ),
        "collapsed_any_seed": any(
            item["embedding_diagnostics_validation"]["collapsed"] for item in results
        ),
        "p90_coverage_mean": float(
            np.mean([item["metrics"]["p90_quantile_coverage"] for item in results])
        ),
    }


def train_temporal_t_jepa(
    source_path: Path,
    baseline_run_dir: Path,
    sequence_run_dir: Path,
    current_jepa_run_dir: Path,
    config_path: Path,
    output_dir: Path,
) -> dict[str, Any]:
    for label, directory in (
        ("baseline", baseline_run_dir),
        ("sequence", sequence_run_dir),
        ("current_jepa", current_jepa_run_dir),
    ):
        if (directory / "INVALIDATED.md").is_file():
            raise RuntimeError(f"refusing invalidated {label} run: {directory}")
    resolved = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    compute = resolved["compute"]
    seeds = tuple(int(seed) for seed in compute["seeds"])
    regularizers = tuple(str(value) for value in resolved["regularizer_candidates"])
    baseline = json.loads((baseline_run_dir / "metrics.json").read_text(encoding="utf-8"))
    sequence = json.loads((sequence_run_dir / "metrics.json").read_text(encoding="utf-8"))
    current = json.loads((current_jepa_run_dir / "metrics.json").read_text(encoding="utf-8"))
    run = RunContext.start(
        output_dir,
        ["flowtwin", "train-temporal-t-jepa", str(source_path)],
        str(resolved["claim_state"]),
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "config_resolved.yaml").write_text(
        config_path.read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    for name in ("data_manifest.json", "split_manifest.json", "leakage_report.json"):
        (output_dir / name).write_bytes((baseline_run_dir / name).read_bytes())
    multi_data = build_disjoint_event_jepa_data(
        source_path,
        baseline_run_dir,
        max_length=int(compute["max_sequence_length"]),
    )
    completion_data = build_disjoint_event_jepa_data(
        source_path,
        baseline_run_dir,
        max_length=int(compute["max_sequence_length"]),
        horizon_offsets=(-1,),
        horizon_labels=("completion_suffix",),
    )
    checkpoint_dir = output_dir / "checkpoints"
    embedding_dir = output_dir / "embeddings"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    embedding_dir.mkdir(parents=True, exist_ok=True)
    predictions = pl.DataFrame(
        {
            "operation_id": multi_data.partitions["test"].context.operation_ids,
            "prediction_cutoff": multi_data.partitions["test"].context.cutoffs,
            "remaining_minutes": multi_data.partitions["test"].context.target_hours * 60,
        }
    )
    results: dict[str, list[dict[str, Any]]] = {}
    for regularizer_name in regularizers:
        register_token = regularizer_name == "reg_token"
        regularizer = "none" if register_token else regularizer_name
        variant = f"multi_{regularizer_name}"
        results[variant] = []
        for seed in seeds:
            result, prediction = _run_variant(
                variant,
                multi_data,
                compute,
                seed,
                checkpoint_dir,
                embedding_dir,
                regularizer=regularizer,
                register_token=register_token,
                save_embeddings=True,
            )
            results[variant].append(result)
            predictions = predictions.with_columns(
                pl.Series(f"{variant}_seed{seed}_p50", prediction[:, 0] * 60),
                pl.Series(f"{variant}_seed{seed}_p90", prediction[:, 1] * 60),
            )
    initial_aggregates = {name: _aggregate(values) for name, values in results.items()}
    selected_main = min(
        initial_aggregates,
        key=lambda name: initial_aggregates[name]["validation_pinball_mean_hours"],
    )
    selected_regularizer_name = selected_main.removeprefix("multi_")
    selected_register = selected_regularizer_name == "reg_token"
    selected_regularizer = "none" if selected_register else selected_regularizer_name
    for kind, data in (
        ("completion", completion_data),
        ("shuffled", _shuffled_training_targets(multi_data, int(compute["shuffle_seed"]))),
    ):
        variant = f"{kind}_{selected_regularizer_name}"
        results[variant] = []
        for seed in seeds:
            result, prediction = _run_variant(
                variant,
                data,
                compute,
                seed,
                checkpoint_dir,
                embedding_dir,
                regularizer=selected_regularizer,
                register_token=selected_register,
                save_embeddings=False,
            )
            results[variant].append(result)
            predictions = predictions.with_columns(
                pl.Series(f"{variant}_seed{seed}_p50", prediction[:, 0] * 60),
                pl.Series(f"{variant}_seed{seed}_p90", prediction[:, 1] * 60),
            )
    predictions.write_parquet(output_dir / "predictions.parquet")
    aggregates = {name: _aggregate(values) for name, values in results.items()}
    completion_variant = f"completion_{selected_regularizer_name}"
    shuffled_variant = f"shuffled_{selected_regularizer_name}"
    baseline_mae = float(baseline["selected_model_test"]["mae_minutes"])
    current_selected = str(current["selected_variant_validation_only"])
    current_mae = float(current["aggregates"][current_selected]["mae_mean_minutes"])
    sequence_selected = str(sequence["best_architecture"])
    sequence_mae = float(sequence["aggregates"][sequence_selected]["mae_mean_minutes"])
    payload: dict[str, Any] = {
        "dataset_id": baseline["dataset_id"],
        "dataset_export_version": baseline["dataset_export_version"],
        "source_file_sha256": baseline["source_file_sha256"],
        "split_protocol": baseline["split_protocol"],
        "split_counts_operations": baseline["split_counts_operations"],
        "seeds": list(seeds),
        "number_of_seeds": len(seeds),
        "masking": "causal_prefix_to_non_overlapping_future_event_block",
        "target_encoder": "ema_stop_gradient",
        "ema_momentum": float(compute["ema_momentum"]),
        "regularizer_candidates": list(regularizers),
        "selected_main_validation_only": selected_main,
        "completion_variant": completion_variant,
        "shuffled_variant": shuffled_variant,
        "results": results,
        "aggregates": aggregates,
        "references": {
            "raw_boosting_mae_minutes": baseline_mae,
            "current_event_jepa_mae_minutes": current_mae,
            "sequence_model": sequence_selected,
            "sequence_mae_minutes": sequence_mae,
        },
        "test_influenced_choice": False,
        "claim_state": "smoke_only",
        "world_model_claim": False,
    }
    atomic_json(output_dir / "metrics.json", payload)
    selected_results = results[selected_main]
    completion_results = results[completion_variant]
    shuffled_results = results[shuffled_variant]
    main_maes = [item["metrics"]["mae_minutes"] for item in selected_results]
    completion_maes = [item["metrics"]["mae_minutes"] for item in completion_results]
    shuffled_maes = [item["metrics"]["mae_minutes"] for item in shuffled_results]
    gate = {
        "three_seeds": len(seeds) >= 3,
        "selected_on_validation_only": True,
        "embedding_stable_each_seed": all(
            not item["embedding_diagnostics_validation"]["collapsed"]
            for item in selected_results
        ),
        "correct_future_beats_shuffled_each_seed": all(
            main < shuffled for main, shuffled in zip(main_maes, shuffled_maes, strict=True)
        ),
        "multi_horizon_beats_completion_each_seed": all(
            main < completion
            for main, completion in zip(main_maes, completion_maes, strict=True)
        ),
        "beats_raw_boosting_mean": aggregates[selected_main]["mae_mean_minutes"]
        < baseline_mae,
        "beats_current_event_jepa_mean": aggregates[selected_main]["mae_mean_minutes"]
        < current_mae,
        "beats_sequence_mean": aggregates[selected_main]["mae_mean_minutes"] < sequence_mae,
        "promote_as_public_representation_model": False,
        "promote_as_kaleido_world_model": False,
        "claim_state": "smoke_only",
    }
    gate["promote_as_public_representation_model"] = bool(
        gate["three_seeds"]
        and gate["embedding_stable_each_seed"]
        and gate["correct_future_beats_shuffled_each_seed"]
        and gate["multi_horizon_beats_completion_each_seed"]
        and gate["beats_raw_boosting_mean"]
    )
    atomic_json(output_dir / "promotion_gate.json", gate)
    atomic_json(
        output_dir / "calibration.json",
        {
            "method": "direct_quantile_probe_no_posthoc_calibration",
            "p90_coverage_mean_by_variant": {
                name: aggregate["p90_coverage_mean"] for name, aggregate in aggregates.items()
            },
        },
    )
    _write_reports(output_dir, payload, gate)
    run.finish(
        {
            "dataset_id": payload["dataset_id"],
            "number_of_seeds": len(seeds),
            "selected_main": selected_main,
            "test_influenced_choice": False,
        }
    )
    return payload


def _write_reports(output_dir: Path, metrics: dict[str, Any], gate: dict[str, Any]) -> None:
    selected = str(metrics["selected_main_validation_only"])
    aggregate = metrics["aggregates"][selected]
    evidence = [
        f"Dataset/export: `{metrics['dataset_id']}`, {metrics['dataset_export_version']}.",
        f"Source SHA-256: `{metrics['source_file_sha256']}`.",
        f"Split: {metrics['split_protocol']}.",
        f"Seeds: {metrics['seeds']}.",
        f"Regularizer selected on validation: `{selected}`.",
        "Test influenced a choice: no.",
        f"Mean test MAE: {aggregate['mae_mean_minutes']:.2f} minutes.",
        f"Between-seed MAE SD: {aggregate['mae_std_minutes']:.2f} minutes.",
        f"Correct future beats shuffled each seed: "
        f"{gate['correct_future_beats_shuffled_each_seed']}.",
        f"Multi-horizon beats completion each seed: "
        f"{gate['multi_horizon_beats_completion_each_seed']}.",
    ]
    limitation = (
        "Public obfuscated non-port event log with short traces and no verified actions, "
        "object graph, immutable plans or Kaleido outcomes. Technical smoke evidence only."
    )
    (output_dir / "model_card.md").write_text(
        "\n".join(
            [
                "# Model card - Temporal T-JEPA public smoke",
                "",
                *[f"- {line}" for line in evidence],
                "- Claim state: `smoke_only`.",
                "",
                "## What this does not prove",
                "",
                limitation,
            ]
        ),
        encoding="utf-8",
    )
    (output_dir / "report.md").write_text(
        "\n".join(
            [
                "# Temporal T-JEPA experiment report",
                "",
                "## Hypothesis",
                "",
                "EMA target encoding and non-overlapping future blocks improve representation "
                "value over the current shared-encoder Event-JEPA.",
                "",
                "## Changes",
                "",
                "Added stopped-gradient EMA targets, disjoint temporal masking, validation-only "
                "regularizer selection, and completion/shuffled controls.",
                "",
                "## Tests and evidence",
                "",
                *evidence,
                f"Public promotion gate: {gate['promote_as_public_representation_model']}.",
                "",
                "## Limitations",
                "",
                limitation,
                "",
                "## Next falsifiable step",
                "",
                "Compare a variational temporal JEPA and raw-plus-embedding boosting on the same "
                "frozen split without changing this test protocol.",
            ]
        ),
        encoding="utf-8",
    )
