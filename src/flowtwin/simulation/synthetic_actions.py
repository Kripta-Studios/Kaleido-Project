from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
import polars as pl
import yaml
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, median_absolute_error

from flowtwin.provenance import RunContext, atomic_json, sha256_file

ACTION_NAMES = (
    "no_action",
    "expedite_release",
    "add_temporary_capacity",
    "priority_dispatch",
    "reroute_parallel_station",
    "planned_hold",
)

ACTION_MULTIPLIERS = {
    "no_action": 1.0,
    "expedite_release": 0.85,
    "add_temporary_capacity": 0.78,
    "priority_dispatch": 0.90,
    "reroute_parallel_station": 0.82,
    "planned_hold": 1.12,
}

ACTION_COST_UNITS = {
    "no_action": 0.0,
    "expedite_release": 2.0,
    "add_temporary_capacity": 4.0,
    "priority_dispatch": 1.0,
    "reroute_parallel_station": 2.5,
    "planned_hold": 0.5,
}


@dataclass(frozen=True)
class SyntheticActionSummary:
    rows: int
    operations: int
    action_counts: dict[str, int]
    seed: int
    evidence_type: str = "synthetic_injected_action_signal_only"


def _eligible_actions(
    row: dict[str, Any],
    *,
    elapsed_q75: float,
    wait_median: float,
    parallel_available: bool,
    constraint_active: bool,
) -> list[str]:
    eligible = ["no_action"]
    if float(row["since_previous_minutes"]) >= wait_median:
        eligible.append("expedite_release")
    if float(row["elapsed_minutes"]) >= elapsed_q75:
        eligible.append("add_temporary_capacity")
    if float(row["prefix_events"]) >= 3:
        eligible.append("priority_dispatch")
    if parallel_available:
        eligible.append("reroute_parallel_station")
    if constraint_active:
        eligible.append("planned_hold")
    return eligible


def generate_synthetic_action_overlay(
    prefixes_path: Path,
    output_path: Path,
    *,
    seed: int = 42,
) -> SyntheticActionSummary:
    """Overlay explicit generated actions without mutating or relabeling source events."""

    frame = pl.read_parquet(prefixes_path).sort(["operation_id", "prediction_cutoff"])
    train = frame.filter(pl.col("partition") == "train")
    elapsed_q75 = float(cast(float, train["elapsed_minutes"].quantile(0.75)))
    wait_median = float(cast(float, train["since_previous_minutes"].median()))
    rng = np.random.default_rng(seed)
    rows: list[dict[str, Any]] = []
    for row in frame.iter_rows(named=True):
        parallel_available = bool(rng.random() < 0.35)
        constraint_active = bool(rng.random() < 0.15)
        eligible = _eligible_actions(
            row,
            elapsed_q75=elapsed_q75,
            wait_median=wait_median,
            parallel_available=parallel_available,
            constraint_active=constraint_active,
        )
        weights = np.ones(len(eligible), dtype=float)
        state_pressure = min(2.0, float(row["elapsed_minutes"]) / max(elapsed_q75, 1.0))
        for index, action in enumerate(eligible):
            if action != "no_action":
                weights[index] += state_pressure
        probabilities = weights / weights.sum()
        selected_index = int(rng.choice(len(eligible), p=probabilities))
        action = eligible[selected_index]
        multiplier = ACTION_MULTIPLIERS[action]
        stochastic_multiplier = float(rng.lognormal(mean=0.0, sigma=0.03))
        untreated = float(row["remaining_minutes"])
        synthetic_remaining = max(0.0, untreated * multiplier * stochastic_multiplier)
        rows.append(
            {
                **row,
                "synthetic_action": action,
                "action_time": row["prediction_cutoff"],
                "action_eligible": True,
                "eligible_actions": json.dumps(eligible),
                "behavior_propensity": float(probabilities[selected_index]),
                "parallel_station_available": parallel_available,
                "constraint_active": constraint_active,
                "untreated_remaining_minutes": untreated,
                "structural_multiplier": multiplier,
                "structural_effect_minutes": synthetic_remaining - untreated,
                "action_cost_units": ACTION_COST_UNITS[action],
                "synthetic_remaining_minutes": synthetic_remaining,
                "generator_seed": seed,
                "causal_scope": "injected_structural_equation_only",
                "source_action_claim": False,
            }
        )
    result = pl.DataFrame(rows)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.write_parquet(output_path)
    counts = {
        str(row["synthetic_action"]): int(row["len"])
        for row in result.group_by("synthetic_action").len().iter_rows(named=True)
    }
    return SyntheticActionSummary(
        rows=result.height,
        operations=result["operation_id"].n_unique(),
        action_counts=counts,
        seed=seed,
    )


def _activity_vocabulary(train: pl.DataFrame) -> dict[str, int]:
    return {
        activity: index + 1
        for index, activity in enumerate(sorted(map(str, train["last_activity"].unique())))
    }


def _matrix(
    frame: pl.DataFrame,
    activity_vocabulary: dict[str, int],
    *,
    action_mode: str,
    rng: np.random.Generator,
) -> np.ndarray:
    base = np.column_stack(
        [
            frame["elapsed_minutes"].to_numpy(),
            frame["since_previous_minutes"].to_numpy(),
            frame["prefix_events"].to_numpy(),
            frame["hour_utc"].to_numpy(),
            frame["weekday_utc"].to_numpy(),
            np.asarray(
                [activity_vocabulary.get(str(value), 0) for value in frame["last_activity"]],
                dtype=float,
            ),
            frame["parallel_station_available"].cast(pl.Int8).to_numpy(),
            frame["constraint_active"].cast(pl.Int8).to_numpy(),
        ]
    ).astype(float)
    action_codes = np.asarray(
        [ACTION_NAMES.index(str(value)) for value in frame["synthetic_action"]], dtype=float
    )
    if action_mode == "shuffled_action":
        action_codes = rng.permutation(action_codes)
    if action_mode == "current_prefix_only":
        return base
    if action_mode == "context_only":
        return base[:, -2:]
    if action_mode == "action_only":
        return action_codes.reshape(-1, 1)
    return np.column_stack([base, action_codes])


def train_synthetic_action_benchmark(
    overlay_path: Path,
    config_path: Path,
    output_dir: Path,
) -> dict[str, Any]:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    seeds = tuple(int(seed) for seed in config["compute"]["seeds"])
    frame = pl.read_parquet(overlay_path)
    train = frame.filter(pl.col("partition") == "train")
    validation = frame.filter(pl.col("partition") == "validation")
    test = frame.filter(pl.col("partition") == "test")
    vocabulary = _activity_vocabulary(train)
    run = RunContext.start(
        output_dir,
        ["flowtwin", "train-synthetic-actions", str(overlay_path)],
        str(config["claim_state"]),
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "config_resolved.yaml").write_text(
        config_path.read_text(encoding="utf-8"), encoding="utf-8"
    )
    source_manifest_path = Path(str(config["dataset_manifest"]))
    data_manifest = yaml.safe_load(source_manifest_path.read_text(encoding="utf-8"))
    data_manifest["derived_overlay_path"] = str(overlay_path)
    data_manifest["derived_overlay_sha256"] = sha256_file(overlay_path)
    data_manifest["derived_evidence_type"] = "synthetic_injected_action_signal_only"
    atomic_json(output_dir / "data_manifest.json", data_manifest)
    split_manifest_path = Path(str(config["split_manifest"]))
    (output_dir / "split_manifest.json").write_bytes(split_manifest_path.read_bytes())
    operation_partition_counts = frame.group_by("operation_id").agg(
        pl.col("partition").n_unique().alias("partitions")
    )
    leakage_checks = {
        "operation_partition_disjoint": bool(
            operation_partition_counts["partitions"].max() == 1
        ),
        "action_at_prediction_cutoff": bool(
            (frame["action_time"] == frame["prediction_cutoff"]).all()
        ),
        "positive_logged_propensity": bool((frame["behavior_propensity"] > 0).all()),
        "no_source_action_claim": bool(not frame["source_action_claim"].any()),
    }
    atomic_json(
        output_dir / "leakage_report.json",
        {"passed": all(leakage_checks.values()), "checks": leakage_checks},
    )
    modes = (
        "correct_action",
        "shuffled_action",
        "current_prefix_only",
        "context_only",
        "action_only",
    )
    results: dict[str, list[dict[str, float | int]]] = {mode: [] for mode in modes}
    prediction_frame = test.select(
        "operation_id", "prediction_cutoff", "synthetic_action", "synthetic_remaining_minutes"
    )
    for seed in seeds:
        rng = np.random.default_rng(seed)
        target_train = train["synthetic_remaining_minutes"].to_numpy()
        target_validation = validation["synthetic_remaining_minutes"].to_numpy()
        target_test = test["synthetic_remaining_minutes"].to_numpy()
        for mode in modes:
            model = HistGradientBoostingRegressor(
                loss="absolute_error",
                max_iter=int(config["compute"]["max_iter"]),
                random_state=seed,
            )
            train_x = _matrix(train, vocabulary, action_mode=mode, rng=rng)
            validation_x = _matrix(validation, vocabulary, action_mode=mode, rng=rng)
            test_x = _matrix(test, vocabulary, action_mode=mode, rng=rng)
            model.fit(train_x, target_train)
            validation_prediction = np.maximum(0.0, model.predict(validation_x))
            test_prediction = np.maximum(0.0, model.predict(test_x))
            results[mode].append(
                {
                    "seed": seed,
                    "validation_mae_minutes": float(
                        mean_absolute_error(target_validation, validation_prediction)
                    ),
                    "test_mae_minutes": float(mean_absolute_error(target_test, test_prediction)),
                    "test_median_ae_minutes": float(
                        median_absolute_error(target_test, test_prediction)
                    ),
                }
            )
            prediction_frame = prediction_frame.with_columns(
                pl.Series(f"{mode}_seed{seed}", test_prediction)
            )
    prediction_frame.write_parquet(output_dir / "predictions.parquet")
    aggregates = {
        mode: {
            "test_mae_mean_minutes": float(
                np.mean([item["test_mae_minutes"] for item in mode_results])
            ),
            "test_mae_std_minutes": float(
                np.std([item["test_mae_minutes"] for item in mode_results], ddof=1)
            ),
        }
        for mode, mode_results in results.items()
    }
    correct = aggregates["correct_action"]["test_mae_mean_minutes"]
    shuffled = aggregates["shuffled_action"]["test_mae_mean_minutes"]
    paired_wins = [
        correct_seed["test_mae_minutes"] < shuffled_seed["test_mae_minutes"]
        for correct_seed, shuffled_seed in zip(
            results["correct_action"], results["shuffled_action"], strict=True
        )
    ]
    action_counts = {
        str(row["synthetic_action"]): int(row["len"])
        for row in frame.group_by("synthetic_action").len().iter_rows(named=True)
    }
    payload: dict[str, Any] = {
        "dataset_id": data_manifest["dataset_id"],
        "dataset_export_version": data_manifest["export_version"],
        "source_file_sha256": data_manifest["files"][0]["sha256"],
        "overlay_sha256": data_manifest["derived_overlay_sha256"],
        "overlay_path": str(overlay_path),
        "rows": frame.height,
        "operations": frame["operation_id"].n_unique(),
        "split_protocol": "inherited_chronological_future_grouped_by_operation",
        "seeds": list(seeds),
        "number_of_seeds": len(seeds),
        "models": results,
        "aggregates": aggregates,
        "correct_actions_beat_shuffled_mean_mae": bool(correct < shuffled),
        "correct_actions_beat_shuffled_each_seed": bool(all(paired_wins)),
        "paired_seed_wins": paired_wins,
        "mean_mae_improvement_vs_shuffled_minutes": float(shuffled - correct),
        "support_diagnostics": {
            "action_counts": action_counts,
            "minimum_behavior_propensity": float(
                cast(float, frame["behavior_propensity"].min())
            ),
            "maximum_behavior_propensity": float(
                cast(float, frame["behavior_propensity"].max())
            ),
            "all_rows_in_logged_support": bool(frame["action_eligible"].all()),
        },
        "threshold_selection": "none_fixed_protocol",
        "test_influenced_choice": False,
        "evidence_type": "synthetic_injected_action_signal_only",
        "source_action_claim": False,
        "kaleido_action_claim": False,
        "claim_state": "smoke_only",
    }
    atomic_json(output_dir / "metrics.json", payload)
    atomic_json(
        output_dir / "calibration.json",
        {"method": "not_applicable_point_prediction_action_recovery"},
    )
    atomic_json(
        output_dir / "action_manifest.json",
        {
            "actions": ACTION_MULTIPLIERS,
            "cost_units": ACTION_COST_UNITS,
            "generator": "public_remaining_time_as_untreated_anchor_plus_seeded_structural_effect",
            "not_observed_actions": True,
        },
    )
    (output_dir / "report.md").write_text(
        "\n".join(
            [
                "# Synthetic action recovery report",
                "",
                "## Hypothesis",
                "",
                "A learner using correct generated actions recovers an explicitly injected "
                "effect better than the same learner with shuffled actions.",
                "",
                "## Evidence",
                "",
                f"Dataset/export: `{payload['dataset_id']}`, "
                f"{payload['dataset_export_version']}.",
                f"Source SHA-256: `{payload['source_file_sha256']}`.",
                f"Overlay SHA-256: `{payload['overlay_sha256']}`.",
                f"Split: {payload['split_protocol']}.",
                f"Seeds: {payload['seeds']}.",
                f"Correct-action mean test MAE: {correct:.2f} minutes.",
                f"Shuffled-action mean test MAE: {shuffled:.2f} minutes.",
                f"Correct beats shuffled: {correct < shuffled}.",
                f"Correct beats shuffled in every seed: {all(paired_wins)}.",
                "Test did not select a model or threshold.",
                "",
                "## Limitations",
                "",
                "Actions and effects are generated, not observed. The public activities remain "
                "observations. This proves only recovery of injected signal and cannot support a "
                "Kaleido causal, savings or deployment claim.",
                "",
                "## Next falsifiable step",
                "",
                "Condition the Event-JEPA latent predictor on the same action stream and repeat "
                "correct/shuffled/action-only ablations before requesting Kaleido action fields.",
            ]
        ),
        encoding="utf-8",
    )
    (output_dir / "model_card.md").write_text(
        "\n".join(
            [
                "# Model card - synthetic action recovery smoke",
                "",
                f"- Dataset/export: `{payload['dataset_id']}`, "
                f"{payload['dataset_export_version']}.",
                f"- Source SHA-256: `{payload['source_file_sha256']}`.",
                f"- Overlay SHA-256: `{payload['overlay_sha256']}`.",
                f"- Split: {payload['split_protocol']}.",
                f"- Models: {list(modes)}.",
                f"- Seeds: {payload['seeds']}.",
                "- Threshold selection: none; fixed protocol.",
                "- Test influenced a choice: no.",
                "- Claim state: `smoke_only`.",
                "",
                "## What this does not prove",
                "",
                "The actions and their effects are generated. This does not prove an "
                "observed Kaleido action effect, causal validity, savings, ROI or deployment.",
            ]
        ),
        encoding="utf-8",
    )
    run.finish(
        {
            "number_of_seeds": len(seeds),
            "correct_actions_beat_shuffled": bool(correct < shuffled),
            "test_influenced_choice": False,
        }
    )
    return payload
