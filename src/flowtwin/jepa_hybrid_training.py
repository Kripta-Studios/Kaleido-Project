from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import polars as pl
import yaml
from sklearn.metrics import mean_absolute_error

from flowtwin.baselines.boosting import quantile_boosting_pipeline
from flowtwin.evaluation.remaining_time import remaining_time_metrics
from flowtwin.provenance import RunContext, atomic_json, sha256_file
from flowtwin.training import CATEGORICAL_FEATURES, NUMERIC_FEATURES

VARIANTS = ("raw", "raw_t_jepa", "raw_var_jepa", "raw_t_var_jepa")


def _embedding_features(frame: pl.DataFrame, prefix: str) -> list[str]:
    return sorted(column for column in frame.columns if column.startswith(f"{prefix}_"))


def _load_embedding(path: Path, prefix: str) -> tuple[pl.DataFrame, list[str]]:
    frame = pl.read_parquet(path)
    features = _embedding_features(frame, prefix)
    required = {"operation_id", "prediction_cutoff", "partition", "remaining_minutes"}
    if not required.issubset(frame.columns):
        raise RuntimeError(f"embedding artifact lacks join/provenance fields: {path}")
    if not features:
        raise RuntimeError(f"embedding artifact has no {prefix}_ columns: {path}")
    return frame.select("operation_id", "prediction_cutoff", *features), features


def _joined_frame(
    prefixes: pl.DataFrame,
    t_path: Path,
    v_path: Path,
) -> tuple[pl.DataFrame, list[str], list[str]]:
    t_frame, t_features = _load_embedding(t_path, "t")
    v_frame, v_features = _load_embedding(v_path, "v")
    joined = prefixes.join(
        t_frame,
        on=["operation_id", "prediction_cutoff"],
        how="left",
        validate="1:1",
    ).join(
        v_frame,
        on=["operation_id", "prediction_cutoff"],
        how="left",
        validate="1:1",
    )
    if joined.select(
        pl.any_horizontal(pl.col(t_features + v_features).is_null()).any()
    ).item():
        raise RuntimeError("hybrid join produced missing embedding values")
    return joined, t_features, v_features


def _numeric_for_variant(
    variant: str,
    t_features: list[str],
    v_features: list[str],
) -> list[str]:
    values = list(NUMERIC_FEATURES)
    if variant in {"raw_t_jepa", "raw_t_var_jepa"}:
        values.extend(t_features)
    if variant in {"raw_var_jepa", "raw_t_var_jepa"}:
        values.extend(v_features)
    return values


def _aggregate(results: list[dict[str, Any]]) -> dict[str, Any]:
    maes = np.asarray([item["metrics"]["mae_minutes"] for item in results], dtype=float)
    validation = np.asarray([item["validation_mae_minutes"] for item in results], dtype=float)
    return {
        "mae_mean_minutes": float(maes.mean()),
        "mae_std_minutes": float(maes.std(ddof=1)),
        "mae_by_seed_minutes": maes.tolist(),
        "validation_mae_mean_minutes": float(validation.mean()),
        "p90_coverage_mean": float(
            np.mean([item["metrics"]["p90_quantile_coverage"] for item in results])
        ),
    }


def train_jepa_hybrid_boosting(
    baseline_run_dir: Path,
    temporal_t_jepa_run_dir: Path,
    var_event_jepa_run_dir: Path,
    config_path: Path,
    output_dir: Path,
) -> dict[str, Any]:
    for label, directory in (
        ("baseline", baseline_run_dir),
        ("temporal_t_jepa", temporal_t_jepa_run_dir),
        ("var_event_jepa", var_event_jepa_run_dir),
    ):
        if (directory / "INVALIDATED.md").is_file():
            raise RuntimeError(f"refusing invalidated {label} run: {directory}")
    resolved = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    compute = resolved["compute"]
    seeds = tuple(int(seed) for seed in compute["seeds"])
    variants = tuple(str(value) for value in resolved.get("variants", VARIANTS))
    if not set(variants).issubset(VARIANTS) or "raw" not in variants:
        raise ValueError("hybrid variants must be known and include raw")
    baseline = json.loads((baseline_run_dir / "metrics.json").read_text(encoding="utf-8"))
    temporal = json.loads(
        (temporal_t_jepa_run_dir / "metrics.json").read_text(encoding="utf-8")
    )
    variational = json.loads(
        (var_event_jepa_run_dir / "metrics.json").read_text(encoding="utf-8")
    )
    selected_t = str(temporal["selected_main_validation_only"])
    prefixes = pl.read_parquet(baseline_run_dir / "prefixes.parquet").sort(
        ["operation_id", "prediction_cutoff", "prefix_events"]
    )
    run = RunContext.start(
        output_dir,
        ["flowtwin", "train-jepa-hybrid"],
        str(resolved["claim_state"]),
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "config_resolved.yaml").write_text(
        config_path.read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    for name in ("data_manifest.json", "split_manifest.json", "leakage_report.json"):
        (output_dir / name).write_bytes((baseline_run_dir / name).read_bytes())
    model_dir = output_dir / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    prediction_base = prefixes.filter(pl.col("partition") == "test").select(
        "operation_id",
        "prediction_cutoff",
        "last_activity",
        "remaining_minutes",
    )
    predictions = prediction_base
    results: dict[str, list[dict[str, Any]]] = {variant: [] for variant in variants}
    feature_counts: dict[str, int] = {}
    for seed in seeds:
        frame, t_features, v_features = _joined_frame(
            prefixes,
            temporal_t_jepa_run_dir / "embeddings" / f"{selected_t}_seed{seed}.parquet",
            var_event_jepa_run_dir / "embeddings" / f"var_event_jepa_seed{seed}.parquet",
        )
        for variant in variants:
            started = time.perf_counter()
            numeric = _numeric_for_variant(variant, t_features, v_features)
            feature_counts[variant] = len(numeric) + len(CATEGORICAL_FEATURES)
            train = frame.filter(pl.col("partition") == "train")
            validation = frame.filter(pl.col("partition") == "validation")
            test = frame.filter(pl.col("partition") == "test")
            train_x = train.select(numeric + CATEGORICAL_FEATURES).to_pandas()
            validation_x = validation.select(numeric + CATEGORICAL_FEATURES).to_pandas()
            test_x = test.select(numeric + CATEGORICAL_FEATURES).to_pandas()
            train_y = train["remaining_minutes"].to_numpy()
            validation_y = validation["remaining_minutes"].to_numpy()
            test_y = test["remaining_minutes"].to_numpy()
            p50_model = quantile_boosting_pipeline(
                numeric,
                CATEGORICAL_FEATURES,
                quantile=0.5,
                seed=seed,
                estimators=int(compute["estimators"]),
            )
            p90_model = quantile_boosting_pipeline(
                numeric,
                CATEGORICAL_FEATURES,
                quantile=0.9,
                seed=seed,
                estimators=int(compute["estimators"]),
            )
            p50_model.fit(train_x, train_y)
            p90_model.fit(train_x, train_y)
            validation_prediction = np.maximum(0.0, p50_model.predict(validation_x))
            p50 = np.maximum(0.0, p50_model.predict(test_x))
            p90 = np.maximum(p50, p90_model.predict(test_x))
            validation_mae = float(mean_absolute_error(validation_y, validation_prediction))
            metrics = remaining_time_metrics(test_y, p50, p90=p90)
            p50_path = model_dir / f"{variant}_p50_seed{seed}.joblib"
            p90_path = model_dir / f"{variant}_p90_seed{seed}.joblib"
            joblib.dump(p50_model, p50_path)
            joblib.dump(p90_model, p90_path)
            predictions = predictions.with_columns(
                pl.Series(f"{variant}_seed{seed}_p50", p50),
                pl.Series(f"{variant}_seed{seed}_p90", p90),
            )
            results[variant].append(
                {
                    "seed": seed,
                    "validation_mae_minutes": validation_mae,
                    "metrics": metrics,
                    "feature_count": feature_counts[variant],
                    "p50_model_sha256": sha256_file(p50_path),
                    "p90_model_sha256": sha256_file(p90_path),
                    "training_seconds": time.perf_counter() - started,
                    "test_influenced_choice": False,
                }
            )
            print(
                f"hybrid_boosting variant={variant} seed={seed} "
                f"validation_mae={validation_mae:.3f}",
                flush=True,
            )
    predictions.write_parquet(output_dir / "predictions.parquet")
    aggregates = {variant: _aggregate(values) for variant, values in results.items()}
    best_hybrid = min(
        (variant for variant in variants if variant != "raw"),
        key=lambda variant: aggregates[variant]["validation_mae_mean_minutes"],
    )
    selected_overall = min(
        variants,
        key=lambda variant: aggregates[variant]["validation_mae_mean_minutes"],
    )
    payload: dict[str, Any] = {
        "dataset_id": baseline["dataset_id"],
        "dataset_export_version": baseline["dataset_export_version"],
        "source_file_sha256": baseline["source_file_sha256"],
        "split_protocol": baseline["split_protocol"],
        "split_counts_operations": baseline["split_counts_operations"],
        "seeds": list(seeds),
        "number_of_seeds": len(seeds),
        "variants": list(variants),
        "temporal_embedding_variant": selected_t,
        "feature_counts": feature_counts,
        "results": results,
        "aggregates": aggregates,
        "best_hybrid_validation_only": best_hybrid,
        "selected_overall_validation_only": selected_overall,
        "references": {
            "existing_raw_boosting_mae_minutes": baseline["selected_model_test"][
                "mae_minutes"
            ],
            "temporal_t_jepa_mae_minutes": temporal["aggregates"][selected_t][
                "mae_mean_minutes"
            ],
            "var_event_jepa_mae_minutes": variational["aggregate"]["mae_mean_minutes"],
        },
        "test_influenced_choice": False,
        "claim_state": "smoke_only",
        "world_model_claim": False,
    }
    atomic_json(output_dir / "metrics.json", payload)
    raw_by_seed = aggregates["raw"]["mae_by_seed_minutes"]
    hybrid_by_seed = aggregates[best_hybrid]["mae_by_seed_minutes"]
    gate = {
        "three_seeds": len(seeds) >= 3,
        "best_hybrid_selected_on_validation_only": True,
        "best_hybrid": best_hybrid,
        "best_hybrid_beats_raw_each_seed": all(
            hybrid < raw
            for hybrid, raw in zip(hybrid_by_seed, raw_by_seed, strict=True)
        ),
        "best_hybrid_beats_raw_mean": aggregates[best_hybrid]["mae_mean_minutes"]
        < aggregates["raw"]["mae_mean_minutes"],
        "promote_hybrid_on_public_data": False,
        "promote_as_kaleido_world_model": False,
        "claim_state": "smoke_only",
    }
    gate["promote_hybrid_on_public_data"] = bool(
        gate["three_seeds"]
        and gate["best_hybrid_beats_raw_each_seed"]
        and gate["best_hybrid_beats_raw_mean"]
    )
    atomic_json(output_dir / "promotion_gate.json", gate)
    atomic_json(
        output_dir / "calibration.json",
        {
            "method": "direct_quantile_gradient_boosting",
            "p90_coverage_mean_by_variant": {
                variant: aggregate["p90_coverage_mean"]
                for variant, aggregate in aggregates.items()
            },
        },
    )
    _write_reports(output_dir, payload, gate)
    run.finish(
        {
            "dataset_id": payload["dataset_id"],
            "number_of_seeds": len(seeds),
            "best_hybrid": best_hybrid,
            "test_influenced_choice": False,
        }
    )
    return payload


def _write_reports(output_dir: Path, metrics: dict[str, Any], gate: dict[str, Any]) -> None:
    selected = str(metrics["best_hybrid_validation_only"])
    aggregate = metrics["aggregates"][selected]
    raw = metrics["aggregates"]["raw"]
    evidence = [
        f"Dataset/export: `{metrics['dataset_id']}`, {metrics['dataset_export_version']}.",
        f"Source SHA-256: `{metrics['source_file_sha256']}`.",
        f"Split: {metrics['split_protocol']}.",
        f"Seeds: {metrics['seeds']}.",
        f"Best hybrid selected on validation: `{selected}`.",
        "Test influenced a choice: no.",
        f"Raw mean test MAE: {raw['mae_mean_minutes']:.2f} minutes.",
        f"Hybrid mean test MAE: {aggregate['mae_mean_minutes']:.2f} minutes.",
        f"Hybrid between-seed MAE SD: {aggregate['mae_std_minutes']:.2f} minutes.",
        f"Hybrid beats raw each seed: {gate['best_hybrid_beats_raw_each_seed']}.",
    ]
    limitation = (
        "The learned features come from the same public non-port log and do not establish "
        "transfer, Kaleido value, causality, savings or deployment readiness."
    )
    (output_dir / "model_card.md").write_text(
        "\n".join(
            [
                "# Model card - JEPA hybrid quantile boosting",
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
                "# JEPA hybrid boosting comparison",
                "",
                "## Hypothesis",
                "",
                "Raw operational features plus frozen temporal embeddings beat raw-only "
                "quantile boosting on the same split.",
                "",
                "## Changes",
                "",
                "Trained raw, raw+T-JEPA, raw+Var-JEPA and raw+both quantile boosting "
                "variants across three representation seeds.",
                "",
                "## Tests and evidence",
                "",
                *evidence,
                f"Public hybrid promotion gate: {gate['promote_hybrid_on_public_data']}.",
                "",
                "## Limitations",
                "",
                limitation,
                "",
                "## Next falsifiable step",
                "",
                "Repeat the frozen protocol on a versioned Kaleido export with operator-reviewed "
                "roles, immutable plan revisions and actual outcomes.",
            ]
        ),
        encoding="utf-8",
    )
