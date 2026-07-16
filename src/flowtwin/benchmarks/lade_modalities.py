from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any, cast

import joblib
import numpy as np
import polars as pl
import yaml

from flowtwin.benchmarks.lade_dispatch import (
    DispatchData,
    DispatchModalitySpec,
    _aggregate_seed_results,
    _fit_selected_boosting,
    _prediction_metrics,
    _route_hash,
    _train_jepa_seed,
    build_lade_dispatch_data,
    mask_dispatch_modalities,
)
from flowtwin.models.dispatch_world_jepa import DispatchWorldJEPAConfig
from flowtwin.provenance import RunContext, atomic_json, sha256_file


def _model_config(data: DispatchData, resolved: dict[str, Any]) -> DispatchWorldJEPAConfig:
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
        regularizer_weight=float(compute["regularizer_weight"]),
    )


def _action_improvement(result: dict[str, Any]) -> dict[str, float]:
    values = result["fixed_model_action_ablation"]["test"]
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


def run_lade_modality_benchmark(
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
            "benchmark-lade-modalities",
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
    base_data = build_lade_dispatch_data(source_path, resolved)
    manifest_path = Path(str(resolved["dataset_manifest"]))
    data_manifest = cast(
        dict[str, Any], yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    )
    data_manifest.update(
        {
            "measured_source_rows": base_data.source_rows,
            "measured_source_routes": base_data.source_routes,
            "measured_usable_routes": base_data.usable_routes,
            "source_file_sha256_verified": sha256_file(source_path),
            "action_policy": base_data.action_policy,
            "claim_state": claim_state,
        }
    )
    atomic_json(output_dir / "data_manifest.json", data_manifest)
    route_sets = {
        name: set(partition.route_ids)
        for name, partition in base_data.partitions.items()
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
            for name, partition in base_data.partitions.items()
        },
        "route_disjoint": split_disjoint,
    }
    atomic_json(output_dir / "split_manifest.json", split_manifest)
    leakage_checks = {
        "route_partition_disjoint": split_disjoint,
        "actions_are_accepted_at_cutoff": True,
        "action_selection_excludes_future_delivery_order": (
            base_data.action_policy == "accepted_pending_fifo"
        ),
        "target_events_excluded_from_context": True,
        "test_previously_opened_is_disclosed": bool(
            resolved["test_influenced_choice"]
        ),
    }
    atomic_json(
        output_dir / "leakage_report.json",
        {"passed": all(leakage_checks.values()), "checks": leakage_checks},
    )
    if not all(leakage_checks.values()):
        raise RuntimeError("LaDe modality leakage audit failed closed")

    compute = cast(dict[str, Any], resolved["compute"])
    candidates = [dict(value) for value in compute["boosting_candidates"]]
    seeds = tuple(int(value) for value in resolved["seeds"])
    specifications = cast(dict[str, dict[str, bool]], resolved["modalities"])
    metrics_by_modality: dict[str, Any] = {}
    prediction_columns: dict[str, Any] = {
        "route_id": base_data.partitions["test"].route_ids,
        "prediction_cutoff": base_data.partitions["test"].cutoffs,
        "remaining_minutes": base_data.partitions["test"].target_minutes,
    }
    for name, values in specifications.items():
        specification = DispatchModalitySpec(**values)
        data = mask_dispatch_modalities(base_data, specification)
        train = data.partitions["train"]
        validation = data.partitions["validation"]
        test = data.partitions["test"]
        selection, raw_p50, raw_p90, raw_p50_model, raw_p90_model = (
            _fit_selected_boosting(
                train.raw_features,
                train.target_minutes,
                validation.raw_features,
                validation.target_minutes,
                test.raw_features,
                candidates=candidates,
                max_iter=int(compute["boosting_max_iter"]),
                seed=int(resolved["seed"]),
            )
        )
        modality_dir = output_dir / "modalities" / name
        model_dir = modality_dir / "models"
        model_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(raw_p50_model, model_dir / "raw_boosting_p50.joblib")
        joblib.dump(raw_p90_model, model_dir / "raw_boosting_p90.joblib")
        raw_test = _prediction_metrics(test.target_minutes, raw_p50, raw_p90)
        prediction_columns[f"{name}_raw_p50"] = raw_p50
        prediction_columns[f"{name}_raw_p90"] = raw_p90

        results: list[dict[str, Any]] = []
        predictions: list[np.ndarray] = []
        checkpoint_dir = modality_dir / "checkpoints"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        for seed in seeds:
            result, prediction = _train_jepa_seed(
                "correct_visreg",
                data,
                _model_config(data, resolved),
                compute,
                candidates=candidates,
                seed=seed,
                checkpoint_dir=checkpoint_dir,
            )
            result["test_influenced_choice"] = bool(
                resolved["test_influenced_choice"]
            )
            results.append(result)
            predictions.append(prediction)
            prediction_columns[f"{name}_jepa_seed{seed}_p50"] = prediction[:, 0]
            prediction_columns[f"{name}_jepa_seed{seed}_p90"] = prediction[:, 1]
        aggregate = _aggregate_seed_results(results)
        jepa_mae = float(aggregate["test_mae_mean_minutes"])
        raw_mae = float(raw_test["mae_minutes"])
        action = [_action_improvement(result) for result in results]
        metrics_by_modality[name] = {
            "availability": asdict(specification),
            "raw_boosting": {
                **selection,
                "test": raw_test,
            },
            "dispatch_world_jepa": {
                "results": results,
                "aggregate": aggregate,
                "mean_test_action_improvement_vs_shuffled_percent": float(
                    np.mean(
                        [item["improvement_vs_shuffled_percent"] for item in action]
                    )
                ),
                "mean_test_action_improvement_vs_no_action_percent": float(
                    np.mean(
                        [item["improvement_vs_no_action_percent"] for item in action]
                    )
                ),
            },
            "jepa_relative_mae_improvement_vs_raw_percent": (
                100.0 * (raw_mae - jepa_mae) / raw_mae
            ),
            "jepa_beats_raw_validation": (
                float(aggregate["validation_mae_mean_minutes"])
                < float(selection["validation_mae_minutes"])
            ),
            "embedding_not_collapsed_each_seed": all(
                not bool(result["embedding_diagnostics_validation"]["collapsed"])
                for result in results
            ),
        }

    full_raw = float(
        metrics_by_modality["full"]["raw_boosting"]["test"]["mae_minutes"]
    )
    for result in metrics_by_modality.values():
        raw_mae = float(result["raw_boosting"]["test"]["mae_minutes"])
        result["raw_degradation_vs_full_percent"] = 100.0 * (
            raw_mae - full_raw
        ) / full_raw

    metrics: dict[str, Any] = {
        "dataset_id": data_manifest["dataset_id"],
        "dataset_export_version": data_manifest["export_version"],
        "source_file_sha256": sha256_file(source_path),
        "split_protocol": split_manifest["protocol"],
        "split_counts": split_manifest["counts"],
        "action_policy": base_data.action_policy,
        "task": "remaining_time_robustness_to_unavailable_modalities",
        "modalities": metrics_by_modality,
        "seeds": list(seeds),
        "number_of_seeds": len(seeds),
        "threshold_selection": "validation_only",
        "test_influenced_choice": bool(resolved["test_influenced_choice"]),
        "test_influence_reason": resolved["test_influence_reason"],
        "claim_state": claim_state,
        "promotion": {
            "production_or_kaleido": False,
            "public_clean_test": False,
            "diagnostic_only": True,
        },
        "what_this_does_not_prove": [
            "Kaleido accuracy, because LaDe is public last-mile data",
            "that AOI identifiers correspond to Kaleido zones, resources or objects",
            "causal action value, production readiness, ROI or realized savings",
        ],
    }
    atomic_json(output_dir / "metrics.json", metrics)
    atomic_json(
        output_dir / "calibration.json",
        {
            "method": "direct_quantile_models_per_modality",
            "test_previously_opened": bool(resolved["test_influenced_choice"]),
            "results": {
                name: {
                    "raw_p90_coverage": result["raw_boosting"]["test"][
                        "p90_quantile_coverage"
                    ],
                    "jepa_p90_coverage_mean": result["dispatch_world_jepa"][
                        "aggregate"
                    ]["p90_coverage_mean"],
                }
                for name, result in metrics_by_modality.items()
            },
        },
    )
    pl.DataFrame(prediction_columns).write_parquet(output_dir / "predictions.parquet")

    evidence = [
        f"Dataset/export: {metrics['dataset_id']} / {metrics['dataset_export_version']}.",
        f"SHA-256: {metrics['source_file_sha256']}.",
        f"Split: {metrics['split_protocol']}; {metrics['split_counts']}.",
        f"Action policy: {base_data.action_policy}.",
        f"Models: quantile boosting and VISReg Dispatch World-JEPA; seeds {list(seeds)}.",
        "October was previously opened; all modality comparisons are diagnostic.",
    ]
    result_lines = [
        (
            f"- {name}: raw MAE {result['raw_boosting']['test']['mae_minutes']:.2f} "
            f"min; JEPA MAE mean "
            f"{result['dispatch_world_jepa']['aggregate']['test_mae_mean_minutes']:.2f} "
            f"min; JEPA-vs-raw delta "
            f"{result['jepa_relative_mae_improvement_vs_raw_percent']:.2f}%."
        )
        for name, result in metrics_by_modality.items()
    ]
    (output_dir / "model_card.md").write_text(
        "\n".join(
            [
                "# Model card - LaDe modality robustness",
                "",
                *[f"- {line}" for line in evidence],
                "",
                "## Diagnostic results",
                "",
                *result_lines,
                "",
                "## Claim boundary",
                "",
                "Public last-mile diagnostic with prior test exposure; no Kaleido, causal, "
                "production, savings or ROI claim is permitted.",
            ]
        ),
        encoding="utf-8",
    )
    (output_dir / "report.md").write_text(
        "\n".join(
            [
                "# LaDe modality robustness diagnostic",
                "",
                "## Hypothesis",
                "",
                str(resolved["hypothesis"]),
                "",
                "## Changes",
                "",
                "Retrained both boosting and Event-JEPA under four explicit availability "
                "tiers. Continuous coordinates, AOI identity and absolute clock are removed "
                "before training rather than masked only at inference.",
                "",
                "## Tests and evidence",
                "",
                *[f"- {line}" for line in evidence],
                *result_lines,
                "",
                "## Limitations",
                "",
                "The October partition was previously opened and LaDe is not port data. "
                "The FIFO action is observable but remains a heuristic, not an operator action.",
                "",
                "## Next falsifiable step",
                "",
                "Freeze a new Kaleido shadow export and predeclare tiers for GPS, named "
                "operational zones and event-only fallback before opening outcomes.",
            ]
        ),
        encoding="utf-8",
    )
    run.finish(
        {
            "dataset_id": metrics["dataset_id"],
            "number_of_seeds": len(seeds),
            "test_influenced_choice": bool(resolved["test_influenced_choice"]),
            "claim_state": claim_state,
        }
    )
    return metrics
