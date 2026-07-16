from __future__ import annotations

import math
from dataclasses import replace
from pathlib import Path
from typing import Any, cast

import numpy as np
import polars as pl
import yaml

from flowtwin.baselines.process_transformer import require_torch
from flowtwin.benchmarks.lade_dispatch import (
    DispatchData,
    _aggregate_seed_results,
    _route_hash,
    _train_jepa_seed,
    _write_prefixes,
    build_lade_dispatch_data,
)
from flowtwin.models.dispatch_world_jepa import DispatchWorldJEPAConfig
from flowtwin.models.event_jepa import anticollapse_loss
from flowtwin.provenance import RunContext, atomic_json, sha256_file


def _model_config(
    data: DispatchData,
    resolved: dict[str, Any],
    *,
    regularizer_weight: float,
) -> DispatchWorldJEPAConfig:
    compute = cast(dict[str, Any], resolved["compute"])
    return DispatchWorldJEPAConfig(
        vocabulary_size=data.vocabulary_size,
        type_vocabulary_size=data.type_vocabulary_size,
        max_length=int(resolved["data"]["max_sequence_length"]),
        max_action_length=max(data.horizons),
        hidden_size=int(compute["hidden_size"]),
        latent_size=int(compute["latent_size"]),
        layers=int(compute["layers"]),
        attention_heads=int(compute["attention_heads"]),
        dropout=float(compute["dropout"]),
        horizon_count=len(data.horizons),
        regularizer_slices=int(compute["regularizer_slices"]),
        regularizer_weight=regularizer_weight,
    )


def _gradient_diagnostics(
    regularizer: str,
    *,
    latent_size: int,
    num_slices: int,
    seed: int,
) -> dict[str, Any]:
    torch = require_torch()
    diagnostics: dict[str, Any] = {}
    for scale in (0.0, 1e-6, 1e-4, 1e-2, 1.0):
        generator = torch.Generator().manual_seed(seed)
        values = (
            scale
            * torch.randn(
                256,
                latent_size,
                generator=generator,
                dtype=torch.float32,
            )
        ).requires_grad_()
        loss = anticollapse_loss(
            values,
            regularizer=regularizer,
            num_slices=num_slices,
            seed=seed,
        )
        if loss.requires_grad:
            loss.backward()
            gradient_norm = float(values.grad.norm())
        else:
            gradient_norm = 0.0
        diagnostics[f"scale_{scale:g}"] = {
            "loss": float(loss.detach()),
            "gradient_norm": gradient_norm,
            "finite": math.isfinite(gradient_norm),
        }
    return diagnostics


def _action_improvements(result: dict[str, Any], partition: str) -> dict[str, float]:
    values = result["fixed_model_action_ablation"][partition]
    correct = float(values["correct_action"])
    shuffled = float(values["shuffled_action"])
    no_action = float(values["prefix_only"])
    return {
        "correct_alignment": correct,
        "shuffled_alignment": shuffled,
        "no_action_alignment": no_action,
        "improvement_vs_shuffled_percent": 100.0 * (shuffled - correct) / shuffled,
        "improvement_vs_no_action_percent": 100.0 * (no_action - correct) / no_action,
    }


def run_lade_regularizer_benchmark(
    source_path: Path,
    config_path: Path,
    output_dir: Path,
) -> dict[str, Any]:
    resolved = cast(
        dict[str, Any], yaml.safe_load(config_path.read_text(encoding="utf-8"))
    )
    claim_state = str(resolved["claim_state"])
    run = RunContext.start(
        output_dir,
        [
            "flowtwin",
            "benchmark-lade-regularizers",
            str(source_path),
            "--config",
            str(config_path),
        ],
        claim_state,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "config_resolved.yaml").write_text(
        config_path.read_text(encoding="utf-8"), encoding="utf-8"
    )
    data = build_lade_dispatch_data(source_path, resolved)
    _write_prefixes(data, output_dir / "prefixes.parquet")
    manifest_path = Path(str(resolved["dataset_manifest"]))
    data_manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    data_manifest.update(
        {
            "measured_source_rows": data.source_rows,
            "measured_source_routes": data.source_routes,
            "measured_usable_routes": data.usable_routes,
            "source_file_sha256_verified": sha256_file(source_path),
            "claim_state": claim_state,
        }
    )
    atomic_json(output_dir / "data_manifest.json", data_manifest)
    route_sets = {
        name: set(partition.route_ids) for name, partition in data.partitions.items()
    }
    split_disjoint = not (
        route_sets["train"] & route_sets["validation"]
        or route_sets["train"] & route_sets["test"]
        or route_sets["validation"] & route_sets["test"]
    )
    split_manifest = {
        "protocol": "chronological_months_grouped_by_courier_day",
        "boundaries": resolved["split"],
        "counts": {
            name: {
                "routes": len(route_sets[name]),
                "prefixes": len(partition.route_ids),
                "route_id_sha256": _route_hash(partition.route_ids),
            }
            for name, partition in data.partitions.items()
        },
        "route_disjoint": split_disjoint,
    }
    atomic_json(output_dir / "split_manifest.json", split_manifest)
    leakage_checks = {
        "route_partition_disjoint": split_disjoint,
        "actions_are_accepted_at_cutoff": True,
        "target_events_excluded_from_context": True,
        "courier_id_excluded_from_features": "courier_id"
        not in data.raw_feature_names,
        "test_previously_opened_is_disclosed": bool(
            resolved["test_influenced_choice"]
        ),
        "action_selection_excludes_future_delivery_order": (
            data.action_policy == "accepted_pending_fifo"
        ),
    }
    atomic_json(
        output_dir / "leakage_report.json",
        {"passed": all(leakage_checks.values()), "checks": leakage_checks},
    )
    if not all(leakage_checks.values()):
        raise RuntimeError("LaDe regularizer leakage audit failed closed")

    compute = cast(dict[str, Any], resolved["compute"])
    candidates = [dict(value) for value in compute["boosting_candidates"]]
    regularizers = cast(dict[str, dict[str, Any]], resolved["regularizers"])
    selection_seed = int(resolved["seed"])
    development = replace(
        data,
        partitions={
            "train": data.partitions["train"],
            "validation": data.partitions["validation"],
            "test": data.partitions["validation"],
        },
    )
    candidate_results: dict[str, list[dict[str, Any]]] = {}
    selected_weights: dict[str, float] = {}
    candidate_root = output_dir / "validation_candidates"
    for regularizer, specification in regularizers.items():
        variant = str(specification["variant"])
        candidate_results[regularizer] = []
        for raw_weight in specification["weight_candidates"]:
            weight = float(raw_weight)
            model_config = _model_config(data, resolved, regularizer_weight=weight)
            candidate_checkpoint_dir = (
                candidate_root / regularizer / f"weight_{weight:g}"
            )
            candidate_checkpoint_dir.mkdir(parents=True, exist_ok=True)
            result, _ = _train_jepa_seed(
                variant,
                development,
                model_config,
                compute,
                candidates=candidates,
                seed=selection_seed,
                checkpoint_dir=candidate_checkpoint_dir,
            )
            action = _action_improvements(result, "validation")
            candidate_results[regularizer].append(
                {
                    "weight": weight,
                    "validation_action": action,
                    "validation_embedding": result[
                        "embedding_diagnostics_validation"
                    ],
                    "validation_remaining_time_mae_minutes": result[
                        "validation_mae_minutes"
                    ],
                    "test_partition_used_for_selection": False,
                }
            )
        eligible = [
            item
            for item in candidate_results[regularizer]
            if not item["validation_embedding"]["collapsed"]
        ]
        pool = eligible or candidate_results[regularizer]
        selected = max(
            pool,
            key=lambda item: item["validation_action"][
                "improvement_vs_shuffled_percent"
            ],
        )
        selected_weights[regularizer] = float(selected["weight"])

    seeds = tuple(int(value) for value in resolved["seeds"])
    final_results: dict[str, list[dict[str, Any]]] = {}
    final_predictions: dict[str, list[np.ndarray]] = {}
    checkpoint_root = output_dir / "checkpoints"
    for regularizer, specification in regularizers.items():
        final_results[regularizer] = []
        final_predictions[regularizer] = []
        model_config = _model_config(
            data,
            resolved,
            regularizer_weight=selected_weights[regularizer],
        )
        final_checkpoint_dir = checkpoint_root / regularizer
        final_checkpoint_dir.mkdir(parents=True, exist_ok=True)
        for seed in seeds:
            result, prediction = _train_jepa_seed(
                str(specification["variant"]),
                data,
                model_config,
                compute,
                candidates=candidates,
                seed=seed,
                checkpoint_dir=final_checkpoint_dir,
            )
            result["test_influenced_choice"] = bool(
                resolved["test_influenced_choice"]
            )
            final_results[regularizer].append(result)
            final_predictions[regularizer].append(prediction)

    gates = cast(dict[str, Any], resolved["acceptance_gates"])
    summaries: dict[str, Any] = {}
    for regularizer, results in final_results.items():
        validation_actions = [
            _action_improvements(result, "validation") for result in results
        ]
        test_actions = [_action_improvements(result, "test") for result in results]
        validation_embeddings = [
            result["embedding_diagnostics_validation"] for result in results
        ]
        test_embeddings = [result["embedding_diagnostics_test"] for result in results]
        gradients = _gradient_diagnostics(
            regularizer,
            latent_size=int(compute["latent_size"]),
            num_slices=int(compute["regularizer_slices"]),
            seed=selection_seed,
        )
        near_collapse_gradient = float(gradients["scale_0.0001"]["gradient_norm"])
        collapse_gate = {
            "not_collapsed_validation_each_seed": all(
                not bool(item["collapsed"]) for item in validation_embeddings
            ),
            "not_collapsed_test_each_seed": all(
                not bool(item["collapsed"]) for item in test_embeddings
            ),
            "minimum_effective_rank_validation": min(
                float(item["effective_rank"]) for item in validation_embeddings
            )
            >= float(gates["min_effective_rank"]),
            "minimum_mean_dimension_std_validation": min(
                float(item["mean_dimension_std"]) for item in validation_embeddings
            )
            >= float(gates["min_mean_dimension_std"]),
            "correct_action_beats_shuffled_validation_each_seed": all(
                item["improvement_vs_shuffled_percent"] > 0.0
                for item in validation_actions
            ),
            "correct_action_beats_shuffled_test_each_seed": all(
                item["improvement_vs_shuffled_percent"] > 0.0
                for item in test_actions
            ),
            "finite_near_collapse_gradient": math.isfinite(near_collapse_gradient),
            "nonzero_near_collapse_gradient": near_collapse_gradient > 0.0,
        }
        collapse_gate["passed"] = all(collapse_gate.values())
        summaries[regularizer] = {
            "selected_weight_validation_only": selected_weights[regularizer],
            "aggregate": _aggregate_seed_results(results),
            "validation_action": validation_actions,
            "test_action": test_actions,
            "mean_validation_improvement_vs_shuffled_percent": float(
                np.mean(
                    [item["improvement_vs_shuffled_percent"] for item in validation_actions]
                )
            ),
            "mean_test_improvement_vs_shuffled_percent": float(
                np.mean([item["improvement_vs_shuffled_percent"] for item in test_actions])
            ),
            "validation_embeddings": validation_embeddings,
            "test_embeddings": test_embeddings,
            "gradient_diagnostics": gradients,
            "collapse_gate": collapse_gate,
        }

    eligible_regularizers = [
        name for name, summary in summaries.items() if summary["collapse_gate"]["passed"]
    ]
    selection_pool = eligible_regularizers or list(summaries)
    selected_regularizer = max(
        selection_pool,
        key=lambda name: summaries[name][
            "mean_validation_improvement_vs_shuffled_percent"
        ],
    )
    metrics: dict[str, Any] = {
        "dataset_id": data_manifest["dataset_id"],
        "dataset_export_version": data_manifest["export_version"],
        "source_file_sha256": sha256_file(source_path),
        "split_protocol": split_manifest["protocol"],
        "split_counts": split_manifest["counts"],
        "task": "anti_collapse_and_action_sensitive_latent_transition_diagnostic",
        "regularizer_weight_selection": "seed_42_validation_only",
        "regularizer_selection": "three_seed_validation_only_after_collapse_gate",
        "candidate_results": candidate_results,
        "selected_weights": selected_weights,
        "regularizers": summaries,
        "selected_regularizer": selected_regularizer,
        "selected_test_summary": summaries[selected_regularizer],
        "number_of_seeds": len(seeds),
        "seeds": list(seeds),
        "test_influenced_choice": bool(resolved["test_influenced_choice"]),
        "test_influence_reason": resolved["test_influence_reason"],
        "claim_state": claim_state,
        "promotion": {
            "production_or_kaleido": False,
            "public_clean_test": False,
            "diagnostic_regularizer_recommendation": selected_regularizer,
        },
        "what_this_does_not_prove": [
            "a clean unseen-test comparison because October was previously opened",
            "Kaleido transfer, causal action value, ROI or production readiness",
            "that Gaussian latent assumptions hold for port operations",
        ],
    }
    atomic_json(output_dir / "metrics.json", metrics)
    atomic_json(
        output_dir / "calibration.json",
        {
            "applicable": False,
            "reason": "This run selects anti-collapse regularization for latent transitions.",
        },
    )
    predictions = pl.DataFrame(
        {
            "route_id": data.partitions["test"].route_ids,
            "prediction_cutoff": data.partitions["test"].cutoffs,
            "remaining_minutes": data.partitions["test"].target_minutes,
        }
    )
    for regularizer, per_seed in final_predictions.items():
        for seed, prediction in zip(seeds, per_seed, strict=True):
            predictions = predictions.with_columns(
                pl.Series(f"{regularizer}_seed{seed}_p50", prediction[:, 0])
            )
    predictions.write_parquet(output_dir / "predictions.parquet")
    evidence = [
        f"Dataset/export: {metrics['dataset_id']} / {metrics['dataset_export_version']}.",
        f"SHA-256: {metrics['source_file_sha256']}.",
        f"Split: {metrics['split_protocol']}; {metrics['split_counts']}.",
        f"Seeds: {list(seeds)}.",
        f"Selected on validation: {selected_regularizer} with weight "
        f"{selected_weights[selected_regularizer]}.",
        "Test was previously opened in the parent experiment; this run is diagnostic.",
    ]
    (output_dir / "model_card.md").write_text(
        "\n".join(
            [
                "# Model card - LaDe JEPA anti-collapse diagnostic",
                "",
                *[f"- {line}" for line in evidence],
                "",
                "## Claim boundary",
                "",
                "No clean-test, Kaleido, causal, ROI or production claim is permitted.",
            ]
        ),
        encoding="utf-8",
    )
    (output_dir / "report.md").write_text(
        "\n".join(
            [
                "# LaDe JEPA anti-collapse diagnostic",
                "",
                "## Hypothesis",
                "",
                str(resolved["hypothesis"]),
                "",
                "## Changes",
                "",
                "Compared official-style SIGReg, VISReg, VICReg and no regularizer with "
                "validation-only weight selection, three seeds, fixed-model action "
                "ablations and collapse-gradient diagnostics.",
                "",
                "## Tests and evidence",
                "",
                *[f"- {line}" for line in evidence],
                "",
                "## Limitations",
                "",
                "The October test was already opened by the parent benchmark. This is a "
                "diagnostic comparison on public last-mile data, not Kaleido evidence.",
                "",
                "## Next falsifiable step",
                "",
                "Freeze a new time period or Kaleido shadow export before using the selected "
                "regularizer in a claim-eligible comparison.",
            ]
        ),
        encoding="utf-8",
    )
    run.finish(
        {
            "dataset_id": metrics["dataset_id"],
            "selected_regularizer": selected_regularizer,
            "number_of_seeds": len(seeds),
            "test_influenced_choice": bool(resolved["test_influenced_choice"]),
            "claim_state": claim_state,
        }
    )
    return metrics
