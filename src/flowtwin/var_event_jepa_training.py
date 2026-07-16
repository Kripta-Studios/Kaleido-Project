from __future__ import annotations

import copy
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import yaml

from flowtwin.baselines.process_transformer import require_torch
from flowtwin.evaluation.remaining_time import remaining_time_metrics
from flowtwin.event_jepa_training import _loader, _train_frozen_probe
from flowtwin.models.contracts import VarEventJEPAConfig
from flowtwin.models.uncertainty import embedding_diagnostics
from flowtwin.models.var_event_jepa import (
    build_var_event_jepa,
    var_event_jepa_loss,
)
from flowtwin.provenance import RunContext, atomic_json, sha256_file
from flowtwin.temporal_jepa_data import build_disjoint_event_jepa_data, write_embeddings


def _validation_loss(
    model: Any,
    data: Any,
    config: VarEventJEPAConfig,
    batch_size: int,
    seed: int,
) -> float:
    torch = require_torch()
    model.eval()
    total = 0.0
    rows = 0
    with torch.no_grad():
        for tokens, lengths, numeric, target_tokens, target_lengths, target_numeric, _ in _loader(
            data.partitions["validation"],
            batch_size,
            shuffle=False,
            seed=seed,
        ):
            outputs = model(
                tokens,
                lengths,
                numeric,
                target_tokens,
                target_lengths,
                target_numeric,
            )
            loss, _ = var_event_jepa_loss(
                outputs,
                tokens,
                numeric,
                target_tokens,
                target_numeric,
                config=config,
                kl_scale=1.0,
            )
            total += float(loss) * len(tokens)
            rows += len(tokens)
    return total / max(rows, 1)


def _extract(
    model: Any,
    data: Any,
    batch_size: int,
    seed: int,
) -> tuple[
    dict[str, tuple[np.ndarray, np.ndarray]],
    dict[str, tuple[np.ndarray, np.ndarray]],
]:
    model.eval()
    extracted: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    uncertainties: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for name, partition in data.partitions.items():
        values: list[np.ndarray] = []
        targets: list[np.ndarray] = []
        context_uncertainty: list[np.ndarray] = []
        predictive_uncertainty: list[np.ndarray] = []
        for tokens, lengths, numeric, _, _, _, target in _loader(
            partition,
            batch_size,
            shuffle=False,
            seed=seed,
        ):
            mean, context_std, predictive_std = model.inference_embedding(
                tokens,
                lengths,
                numeric,
            )
            values.append(mean.cpu().numpy())
            targets.append(target.cpu().numpy())
            context_uncertainty.append(context_std.cpu().numpy())
            predictive_uncertainty.append(predictive_std.cpu().numpy())
        extracted[name] = (np.concatenate(values), np.concatenate(targets))
        uncertainties[name] = (
            np.concatenate(context_uncertainty),
            np.concatenate(predictive_uncertainty),
        )
    return extracted, uncertainties


def _rank_correlation(left: np.ndarray, right: np.ndarray) -> float:
    left_rank = np.argsort(np.argsort(np.asarray(left, dtype=float)))
    right_rank = np.argsort(np.argsort(np.asarray(right, dtype=float)))
    if left_rank.std() == 0 or right_rank.std() == 0:
        return 0.0
    return float(np.corrcoef(left_rank, right_rank)[0, 1])


def _model_config(data: Any, compute: dict[str, Any]) -> VarEventJEPAConfig:
    return VarEventJEPAConfig(
        vocabulary_size=data.vocabulary_size,
        max_length=data.max_length,
        hidden_size=int(compute["hidden_size"]),
        latent_size=int(compute["latent_size"]),
        auxiliary_size=int(compute["auxiliary_size"]),
        layers=int(compute["layers"]),
        attention_heads=int(compute["attention_heads"]),
        dropout=float(compute.get("dropout", 0.0)),
        horizon_count=len(data.horizon_labels),
        reconstruction_weight=float(compute["reconstruction_weight"]),
        generation_weight=float(compute["generation_weight"]),
        context_kl_weight=float(compute["context_kl_weight"]),
        auxiliary_kl_weight=float(compute["auxiliary_kl_weight"]),
        target_kl_weight=float(compute["target_kl_weight"]),
    )


def train_var_event_jepa(
    source_path: Path,
    baseline_run_dir: Path,
    temporal_t_jepa_run_dir: Path,
    config_path: Path,
    output_dir: Path,
) -> dict[str, Any]:
    for label, directory in (
        ("baseline", baseline_run_dir),
        ("temporal_t_jepa", temporal_t_jepa_run_dir),
    ):
        if (directory / "INVALIDATED.md").is_file():
            raise RuntimeError(f"refusing invalidated {label} run: {directory}")
    resolved = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    compute = resolved["compute"]
    seeds = tuple(int(seed) for seed in compute["seeds"])
    baseline = json.loads((baseline_run_dir / "metrics.json").read_text(encoding="utf-8"))
    temporal = json.loads(
        (temporal_t_jepa_run_dir / "metrics.json").read_text(encoding="utf-8")
    )
    run = RunContext.start(
        output_dir,
        ["flowtwin", "train-var-event-jepa", str(source_path)],
        str(resolved["claim_state"]),
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "config_resolved.yaml").write_text(
        config_path.read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    for name in ("data_manifest.json", "split_manifest.json", "leakage_report.json"):
        (output_dir / name).write_bytes((baseline_run_dir / name).read_bytes())
    data = build_disjoint_event_jepa_data(
        source_path,
        baseline_run_dir,
        max_length=int(compute["max_sequence_length"]),
    )
    model_config = _model_config(data, compute)
    torch = require_torch()
    checkpoint_dir = output_dir / "checkpoints"
    embedding_dir = output_dir / "embeddings"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    embedding_dir.mkdir(parents=True, exist_ok=True)
    predictions = pl.DataFrame(
        {
            "operation_id": data.partitions["test"].context.operation_ids,
            "prediction_cutoff": data.partitions["test"].context.cutoffs,
            "remaining_minutes": data.partitions["test"].context.target_hours * 60,
        }
    )
    results: list[dict[str, Any]] = []
    for seed in seeds:
        torch.manual_seed(seed)
        np.random.seed(seed)
        model = build_var_event_jepa(model_config)
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=float(compute["learning_rate"]),
            weight_decay=float(compute["weight_decay"]),
        )
        batch_size = int(compute["batch_size"])
        train_loader = _loader(
            data.partitions["train"],
            batch_size,
            shuffle=True,
            seed=seed,
        )
        best_loss = float("inf")
        best_state: dict[str, Any] | None = None
        history: list[dict[str, float | int]] = []
        started = time.perf_counter()
        epochs = int(compute["pretrain_epochs"])
        anneal_epochs = max(1, int(compute["kl_anneal_epochs"]))
        for epoch in range(1, epochs + 1):
            model.train()
            total = 0.0
            rows = 0
            kl_scale = min(1.0, epoch / anneal_epochs)
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
                outputs = model(
                    tokens,
                    lengths,
                    numeric,
                    target_tokens,
                    target_lengths,
                    target_numeric,
                )
                loss, _ = var_event_jepa_loss(
                    outputs,
                    tokens,
                    numeric,
                    target_tokens,
                    target_numeric,
                    config=model_config,
                    kl_scale=kl_scale,
                )
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                total += float(loss.detach()) * len(tokens)
                rows += len(tokens)
            validation = _validation_loss(
                model,
                data,
                model_config,
                batch_size,
                seed,
            )
            history.append(
                {
                    "epoch": epoch,
                    "kl_scale": kl_scale,
                    "train_loss": total / max(rows, 1),
                    "validation_loss": validation,
                }
            )
            print(
                f"var_event_jepa seed={seed} epoch={epoch} "
                f"validation_loss={validation:.6f}",
                flush=True,
            )
            if validation < best_loss:
                best_loss = validation
                best_state = copy.deepcopy(model.state_dict())
        if best_state is None:
            raise RuntimeError("Var-Event-JEPA pretraining produced no checkpoint")
        model.load_state_dict(best_state)
        checkpoint = checkpoint_dir / f"var_event_jepa_seed{seed}.pt"
        torch.save(
            {
                "state_dict": best_state,
                "config": model_config.__dict__,
                "seed": seed,
                "horizons": data.horizon_labels,
                "best_validation_loss": best_loss,
                "history": history,
            },
            checkpoint,
        )
        extracted, uncertainties = _extract(model, data, batch_size, seed)
        write_embeddings(
            embedding_dir / f"var_event_jepa_seed{seed}.parquet",
            data,
            {name: values[0] for name, values in extracted.items()},
            column_prefix="v",
            uncertainty=uncertainties,
        )
        diagnostics = embedding_diagnostics(extracted["validation"][0])
        probe_state, prediction, probe_validation = _train_frozen_probe(
            extracted,
            latent_size=model_config.latent_size,
            seed=seed,
            epochs=int(compute["frozen_head_epochs"]),
            batch_size=batch_size,
        )
        probe_checkpoint = checkpoint_dir / f"var_event_jepa_probe_seed{seed}.pt"
        torch.save(probe_state, probe_checkpoint)
        target_minutes = extracted["test"][1] * 60
        p50 = prediction[:, 0] * 60
        p90 = prediction[:, 1] * 60
        absolute_error = np.abs(target_minutes - p50)
        predictive_uncertainty = uncertainties["test"][1]
        uncertainty_correlation = _rank_correlation(
            predictive_uncertainty,
            absolute_error,
        )
        metrics = remaining_time_metrics(target_minutes, p50, p90=p90)
        predictions = predictions.with_columns(
            pl.Series(f"var_seed{seed}_p50", p50),
            pl.Series(f"var_seed{seed}_p90", p90),
            pl.Series(
                f"var_seed{seed}_predictive_uncertainty",
                predictive_uncertainty,
            ),
        )
        results.append(
            {
                "seed": seed,
                "pretraining_best_validation_loss": best_loss,
                "probe_best_validation_pinball_hours": probe_validation,
                "pretraining_history": history,
                "embedding_diagnostics_validation": diagnostics,
                "metrics": metrics,
                "predictive_uncertainty_mean": float(predictive_uncertainty.mean()),
                "predictive_uncertainty_error_spearman": uncertainty_correlation,
                "checkpoint_sha256": sha256_file(checkpoint),
                "probe_checkpoint_sha256": sha256_file(probe_checkpoint),
                "training_seconds": time.perf_counter() - started,
                "test_influenced_choice": False,
            }
        )
    predictions.write_parquet(output_dir / "predictions.parquet")
    maes = np.asarray([item["metrics"]["mae_minutes"] for item in results], dtype=float)
    temporal_selected = str(temporal["selected_main_validation_only"])
    temporal_mae = float(temporal["aggregates"][temporal_selected]["mae_mean_minutes"])
    baseline_mae = float(baseline["selected_model_test"]["mae_minutes"])
    aggregate = {
        "mae_mean_minutes": float(maes.mean()),
        "mae_std_minutes": float(maes.std(ddof=1)),
        "mae_by_seed_minutes": maes.tolist(),
        "p90_coverage_mean": float(
            np.mean([item["metrics"]["p90_quantile_coverage"] for item in results])
        ),
        "effective_rank_mean": float(
            np.mean(
                [
                    item["embedding_diagnostics_validation"]["effective_rank"]
                    for item in results
                ]
            )
        ),
        "uncertainty_error_spearman_mean": float(
            np.mean(
                [item["predictive_uncertainty_error_spearman"] for item in results]
            )
        ),
    }
    payload: dict[str, Any] = {
        "dataset_id": baseline["dataset_id"],
        "dataset_export_version": baseline["dataset_export_version"],
        "source_file_sha256": baseline["source_file_sha256"],
        "split_protocol": baseline["split_protocol"],
        "split_counts_operations": baseline["split_counts_operations"],
        "seeds": list(seeds),
        "number_of_seeds": len(seeds),
        "objective": "temporal_variational_elbo",
        "masking": "causal_prefix_to_non_overlapping_future_event_block",
        "horizons": list(data.horizon_labels),
        "results": results,
        "aggregate": aggregate,
        "references": {
            "raw_boosting_mae_minutes": baseline_mae,
            "temporal_t_jepa_variant": temporal_selected,
            "temporal_t_jepa_mae_minutes": temporal_mae,
        },
        "test_influenced_choice": False,
        "claim_state": "smoke_only",
        "world_model_claim": False,
    }
    atomic_json(output_dir / "metrics.json", payload)
    gate = {
        "three_seeds": len(seeds) >= 3,
        "embedding_stable_each_seed": all(
            not item["embedding_diagnostics_validation"]["collapsed"] for item in results
        ),
        "finite_positive_uncertainty_each_seed": all(
            np.isfinite(item["predictive_uncertainty_mean"])
            and item["predictive_uncertainty_mean"] > 0
            for item in results
        ),
        "uncertainty_tracks_error_each_seed": all(
            item["predictive_uncertainty_error_spearman"] > 0 for item in results
        ),
        "beats_temporal_t_jepa_mean": aggregate["mae_mean_minutes"] < temporal_mae,
        "beats_raw_boosting_each_seed": all(value < baseline_mae for value in maes),
        "promote_as_public_representation_model": False,
        "promote_as_kaleido_world_model": False,
        "claim_state": "smoke_only",
    }
    gate["promote_as_public_representation_model"] = bool(
        gate["three_seeds"]
        and gate["embedding_stable_each_seed"]
        and gate["finite_positive_uncertainty_each_seed"]
        and gate["uncertainty_tracks_error_each_seed"]
        and gate["beats_raw_boosting_each_seed"]
    )
    atomic_json(output_dir / "promotion_gate.json", gate)
    atomic_json(
        output_dir / "calibration.json",
        {
            "method": "latent_predictive_uncertainty_diagnostic_and_direct_quantile_probe",
            "p90_coverage_mean": aggregate["p90_coverage_mean"],
            "predictive_uncertainty_error_spearman_by_seed": [
                item["predictive_uncertainty_error_spearman"] for item in results
            ],
            "time_uncertainty_calibrated": False,
        },
    )
    _write_reports(output_dir, payload, gate)
    run.finish(
        {
            "dataset_id": payload["dataset_id"],
            "number_of_seeds": len(seeds),
            "test_influenced_choice": False,
        }
    )
    return payload


def _write_reports(output_dir: Path, metrics: dict[str, Any], gate: dict[str, Any]) -> None:
    aggregate = metrics["aggregate"]
    evidence = [
        f"Dataset/export: `{metrics['dataset_id']}`, {metrics['dataset_export_version']}.",
        f"Source SHA-256: `{metrics['source_file_sha256']}`.",
        f"Split: {metrics['split_protocol']}.",
        f"Seeds: {metrics['seeds']}.",
        "Threshold selection: not applicable.",
        "Test influenced a choice: no.",
        f"Mean test MAE: {aggregate['mae_mean_minutes']:.2f} minutes.",
        f"Between-seed MAE SD: {aggregate['mae_std_minutes']:.2f} minutes.",
        f"Mean latent uncertainty/error Spearman: "
        f"{aggregate['uncertainty_error_spearman_mean']:.3f}.",
    ]
    limitation = (
        "The Gaussian temporal ELBO is a development adaptation, not a reproduction of the "
        "Var-T-JEPA paper. Latent uncertainty is not calibrated in minutes. Public non-port "
        "data cannot establish Kaleido value or operational world-model validity."
    )
    (output_dir / "model_card.md").write_text(
        "\n".join(
            [
                "# Model card - Var-Event-JEPA public smoke",
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
                "# Var-Event-JEPA experiment report",
                "",
                "## Hypothesis",
                "",
                "A temporal ELBO improves representation stability and exposes useful "
                "per-prefix uncertainty without heuristic distribution regularization.",
                "",
                "## Changes",
                "",
                "Added Gaussian context, auxiliary and future latents, a learned conditional "
                "future prior, observation decoders and KL annealing.",
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
                "Test whether raw features plus frozen deterministic/variational embeddings "
                "beat raw quantile boosting across the same seeds.",
            ]
        ),
        encoding="utf-8",
    )
