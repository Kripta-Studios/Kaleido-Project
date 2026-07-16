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
    _extract,
    _jepa_validation,
    _loader,
    _train_frozen_probe,
    build_event_jepa_data,
)
from flowtwin.models.contracts import EventJEPAConfig
from flowtwin.models.event_jepa import build_event_jepa, event_jepa_loss
from flowtwin.models.uncertainty import embedding_diagnostics
from flowtwin.provenance import RunContext, atomic_json, sha256_file

ABLATIONS = (
    "random_encoder_no_jepa",
    "completion_only_sigreg",
    "multi_horizon_no_sigreg",
    "shuffled_temporal_pairs",
)


def _shuffle_training_targets(data: EventJEPAData, seed: int) -> EventJEPAData:
    rng = np.random.default_rng(seed)
    training = data.partitions["train"]
    permutation = rng.permutation(len(training.context.operation_ids))
    shuffled = EventJEPAPartition(
        context=training.context,
        target_tokens=training.target_tokens[permutation],
        target_lengths=training.target_lengths[permutation],
        target_numeric=training.target_numeric[permutation],
    )
    return replace(data, partitions={**data.partitions, "train": shuffled})


def _model_config(data: EventJEPAData, compute: dict[str, Any]) -> EventJEPAConfig:
    return EventJEPAConfig(
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


def _run_seed(
    ablation: str,
    data: EventJEPAData,
    compute: dict[str, Any],
    seed: int,
    checkpoint_dir: Path,
) -> tuple[dict[str, Any], np.ndarray]:
    torch = require_torch()
    torch.manual_seed(seed)
    np.random.seed(seed)
    use_sigreg = ablation != "multi_horizon_no_sigreg"
    train_data = (
        _shuffle_training_targets(data, seed)
        if ablation == "shuffled_temporal_pairs"
        else data
    )
    config = _model_config(data, compute)
    model = build_event_jepa(config)
    train_loader = _loader(
        train_data.partitions["train"],
        int(compute["batch_size"]),
        shuffle=True,
        seed=seed,
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
    if ablation == "random_encoder_no_jepa":
        best_state = copy.deepcopy(model.state_dict())
        best_loss = _jepa_validation(
            model,
            validation_loader,
            config,
            step,
            use_sigreg=use_sigreg,
        )
        history.append({"epoch": 0, "validation_loss": best_loss})
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
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
                loss, _ = event_jepa_loss(
                    context,
                    targets,
                    predictions,
                    config=config,
                    step=step,
                    use_sigreg=use_sigreg,
                )
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                total += float(loss.detach()) * len(tokens)
                rows += len(tokens)
                step += 1
            validation_loss = _jepa_validation(
                model,
                validation_loader,
                config,
                step,
                use_sigreg=use_sigreg,
            )
            history.append(
                {
                    "epoch": epoch,
                    "train_loss": total / max(rows, 1),
                    "validation_loss": validation_loss,
                }
            )
            print(
                f"event_jepa_ablation={ablation} seed={seed} epoch={epoch} "
                f"validation_loss={validation_loss:.6f}",
                flush=True,
            )
            if validation_loss < best_loss:
                best_loss = validation_loss
                best_state = copy.deepcopy(model.state_dict())
    if best_state is None:
        raise RuntimeError(f"{ablation} produced no checkpoint")
    checkpoint = checkpoint_dir / f"{ablation}_seed{seed}.pt"
    torch.save(
        {
            "state_dict": best_state,
            "config": config.__dict__,
            "ablation": ablation,
            "seed": seed,
            "horizons": data.horizon_labels,
            "best_validation_loss": best_loss,
            "pretraining_history": history,
        },
        checkpoint,
    )
    reloaded = build_event_jepa(config)
    reloaded.load_state_dict(
        torch.load(checkpoint, map_location="cpu", weights_only=True)["state_dict"]
    )
    extracted = {
        name: _extract(
            reloaded,
            _loader(
                partition,
                int(compute["batch_size"]),
                shuffle=False,
                seed=seed,
            ),
        )
        for name, partition in data.partitions.items()
    }
    diagnostics = embedding_diagnostics(extracted["validation"][0])
    probe_state, prediction, probe_validation = _train_frozen_probe(
        extracted,
        latent_size=config.latent_size,
        seed=seed,
        epochs=int(compute["frozen_head_epochs"]),
        batch_size=int(compute["batch_size"]),
    )
    torch.save(probe_state, checkpoint_dir / f"{ablation}_probe_seed{seed}.pt")
    target_minutes = extracted["test"][1] * 60
    p50 = prediction[:, 0] * 60
    p90 = prediction[:, 1] * 60
    result = {
        "seed": seed,
        "pretraining_best_validation_loss": best_loss,
        "probe_best_validation_pinball_hours": probe_validation,
        "pretraining_history": history,
        "embedding_diagnostics_validation": diagnostics,
        "metrics": remaining_time_metrics(target_minutes, p50, p90=p90),
        "checkpoint_sha256": sha256_file(checkpoint),
        "training_seconds": time.perf_counter() - started,
        "test_influenced_choice": False,
    }
    return result, prediction


def train_event_jepa_ablations(
    source_path: Path,
    baseline_run_dir: Path,
    main_jepa_run_dir: Path,
    config_path: Path,
    output_dir: Path,
) -> dict[str, Any]:
    for label, run_dir in (("baseline", baseline_run_dir), ("main_jepa", main_jepa_run_dir)):
        if (run_dir / "INVALIDATED.md").is_file():
            raise RuntimeError(f"refusing invalidated {label} run: {run_dir}")
    resolved = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    compute = resolved["compute"]
    seeds = tuple(int(seed) for seed in compute["seeds"])
    baseline = json.loads((baseline_run_dir / "metrics.json").read_text(encoding="utf-8"))
    main = json.loads((main_jepa_run_dir / "metrics.json").read_text(encoding="utf-8"))
    run = RunContext.start(
        output_dir,
        ["flowtwin", "train-event-jepa-ablations", str(source_path)],
        str(resolved["claim_state"]),
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "config_resolved.yaml").write_text(
        config_path.read_text(encoding="utf-8"), encoding="utf-8"
    )
    for name in ("data_manifest.json", "split_manifest.json", "leakage_report.json"):
        (output_dir / name).write_bytes((baseline_run_dir / name).read_bytes())
    multi_data = build_event_jepa_data(
        source_path,
        baseline_run_dir,
        max_length=int(compute["max_sequence_length"]),
    )
    completion_data = build_event_jepa_data(
        source_path,
        baseline_run_dir,
        max_length=int(compute["max_sequence_length"]),
        horizon_offsets=(-1,),
        horizon_labels=("completion",),
    )
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    predictions = pl.DataFrame(
        {
            "operation_id": multi_data.partitions["test"].context.operation_ids,
            "prediction_cutoff": multi_data.partitions["test"].context.cutoffs,
            "remaining_minutes": multi_data.partitions["test"].context.target_hours * 60,
        }
    )
    results: dict[str, list[dict[str, Any]]] = {name: [] for name in ABLATIONS}
    for ablation in ABLATIONS:
        data = completion_data if ablation == "completion_only_sigreg" else multi_data
        for seed in seeds:
            result, prediction = _run_seed(ablation, data, compute, seed, checkpoint_dir)
            results[ablation].append(result)
            predictions = predictions.with_columns(
                pl.Series(f"{ablation}_seed{seed}_p50", prediction[:, 0] * 60),
                pl.Series(f"{ablation}_seed{seed}_p90", prediction[:, 1] * 60),
            )
    predictions.write_parquet(output_dir / "predictions.parquet")
    aggregates: dict[str, Any] = {}
    for ablation, ablation_results in results.items():
        maes = np.asarray([item["metrics"]["mae_minutes"] for item in ablation_results])
        aggregates[ablation] = {
            "mae_mean_minutes": float(maes.mean()),
            "mae_std_minutes": float(maes.std(ddof=1)),
            "mae_by_seed_minutes": maes.tolist(),
            "probe_validation_pinball_mean_hours": float(
                np.mean(
                    [
                        item["probe_best_validation_pinball_hours"]
                        for item in ablation_results
                    ]
                )
            ),
            "p90_coverage_mean": float(
                np.mean(
                    [item["metrics"]["p90_quantile_coverage"] for item in ablation_results]
                )
            ),
            "effective_rank_mean": float(
                np.mean(
                    [
                        item["embedding_diagnostics_validation"]["effective_rank"]
                        for item in ablation_results
                    ]
                )
            ),
            "collapsed_any_seed": any(
                item["embedding_diagnostics_validation"]["collapsed"]
                for item in ablation_results
            ),
        }
    main_variant = str(main["selected_variant_validation_only"])
    main_aggregate = main["aggregates"][main_variant]
    main_mae = float(main_aggregate["mae_mean_minutes"])
    comparisons = {
        name: {
            "ablation_minus_main_mae_minutes": float(values["mae_mean_minutes"] - main_mae),
            "main_has_lower_mae": bool(main_mae < values["mae_mean_minutes"]),
        }
        for name, values in aggregates.items()
    }
    payload: dict[str, Any] = {
        "dataset_id": baseline["dataset_id"],
        "dataset_export_version": baseline["dataset_export_version"],
        "source_file_sha256": baseline["source_file_sha256"],
        "split_protocol": baseline["split_protocol"],
        "split_counts_operations": baseline["split_counts_operations"],
        "seeds": list(seeds),
        "number_of_seeds": len(seeds),
        "fixed_ablations": list(ABLATIONS),
        "main_reference": {
            "variant": main_variant,
            "mae_mean_minutes": main_mae,
            "mae_std_minutes": main_aggregate["mae_std_minutes"],
        },
        "results": results,
        "aggregates": aggregates,
        "comparisons_to_main": comparisons,
        "test_influenced_choice": False,
        "claim_state": "smoke_only",
    }
    atomic_json(output_dir / "metrics.json", payload)
    gate = {
        "main_beats_random_encoder": comparisons["random_encoder_no_jepa"][
            "main_has_lower_mae"
        ],
        "sigreg_beats_no_sigreg": comparisons["multi_horizon_no_sigreg"][
            "main_has_lower_mae"
        ],
        "multi_horizon_beats_completion_only": comparisons["completion_only_sigreg"][
            "main_has_lower_mae"
        ],
        "correct_temporal_pairs_beat_shuffled": comparisons["shuffled_temporal_pairs"][
            "main_has_lower_mae"
        ],
        "test_influenced_choice": False,
        "claim_state": "smoke_only",
    }
    atomic_json(output_dir / "ablation_gate.json", gate)
    atomic_json(
        output_dir / "calibration.json",
        {
            "method": "direct_quantile_heads_no_posthoc_calibration",
            "p90_coverage_mean_by_ablation": {
                name: values["p90_coverage_mean"] for name, values in aggregates.items()
            },
        },
    )
    _write_reports(output_dir, payload, gate)
    run.finish(
        {
            "dataset_id": payload["dataset_id"],
            "number_of_seeds": len(seeds),
            "fixed_ablations": list(ABLATIONS),
            "test_influenced_choice": False,
        }
    )
    return payload


def _write_reports(output_dir: Path, metrics: dict[str, Any], gate: dict[str, Any]) -> None:
    lines = [
        "# Event-JEPA ablation report",
        "",
        "## Hypothesis",
        "",
        "The future-latent objective, correct temporal pairing, SIGReg and direct "
        "multi-horizon targets each add measurable representation value.",
        "",
        "## Evidence",
        "",
        f"Dataset/export: `{metrics['dataset_id']}`, {metrics['dataset_export_version']}.",
        f"Source SHA-256: `{metrics['source_file_sha256']}`.",
        f"Split: {metrics['split_protocol']}.",
        f"Seeds: {metrics['seeds']}.",
        "Ablations were fixed before this run; test influenced a choice: no.",
        f"Main reference MAE: {metrics['main_reference']['mae_mean_minutes']:.2f} minutes.",
    ]
    for name, aggregate in metrics["aggregates"].items():
        lines.append(
            f"{name}: {aggregate['mae_mean_minutes']:.2f} +/- "
            f"{aggregate['mae_std_minutes']:.2f} minutes."
        )
    lines.extend(
        [
            "",
            "## Gate",
            "",
            *[f"- {name}: {value}" for name, value in gate.items()],
            "",
            "## Limitations",
            "",
            "One public obfuscated non-port event log. These ablations test technical "
            "representation behavior only; they do not prove Kaleido accuracy, action value, "
            "causality, ROI, savings or deployment readiness.",
            "",
            "## Next falsifiable step",
            "",
            "Condition a separate latent transition model on explicit generated actions and "
            "require correct actions to beat shuffled actions across seeds.",
        ]
    )
    (output_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")
    (output_dir / "model_card.md").write_text(
        "\n".join(
            [
                "# Model card - Event-JEPA ablations",
                "",
                f"- Dataset/export: `{metrics['dataset_id']}`, "
                f"{metrics['dataset_export_version']}.",
                f"- Source SHA-256: `{metrics['source_file_sha256']}`.",
                f"- Split: {metrics['split_protocol']}.",
                f"- Seeds: {metrics['seeds']}.",
                f"- Fixed variants: {metrics['fixed_ablations']}.",
                "- Threshold selection: not applicable.",
                "- Test influenced a choice: no.",
                "- Claim state: `smoke_only`.",
                "",
                "## What this does not prove",
                "",
                "No Kaleido, causal action, ROI, savings, production or deployment claim.",
            ]
        ),
        encoding="utf-8",
    )
