from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import polars as pl
import yaml
from sklearn.metrics import mean_absolute_error

from flowtwin.baselines.boosting import (
    ConformalIntervals,
    logistic_risk_pipeline,
    quantile_boosting_pipeline,
    ridge_remaining_pipeline,
)
from flowtwin.baselines.naive import MedianRemainingTime
from flowtwin.baselines.survival import KaplanMeierRemainingTime
from flowtwin.config import ExperimentConfig, load_experiment
from flowtwin.data.leakage import run_leakage_audit
from flowtwin.data.roles import ColumnRole, FieldClassification
from flowtwin.data.splits import Partition
from flowtwin.evaluation.remaining_time import (
    cluster_bootstrap_mae,
    grouped_remaining_time_metrics,
    remaining_time_metrics,
)
from flowtwin.evaluation.risk import risk_metrics, select_threshold_validation
from flowtwin.features.prefix import build_prefix_dataset, prefix_feature_columns
from flowtwin.provenance import RunContext, atomic_json, sha256_file

NUMERIC_FEATURES = [
    "elapsed_minutes",
    "since_previous_minutes",
    "prefix_events",
    "hour_utc",
    "weekday_utc",
]
CATEGORICAL_FEATURES = ["last_activity"]


def _classification() -> FieldClassification:
    roles = {
        "operation_id": ColumnRole.IDENTIFIER,
        "prediction_cutoff": ColumnRole.TIMESTAMP,
        "_case_start": ColumnRole.TIMESTAMP,
        "last_activity": ColumnRole.OBSERVATION,
        "elapsed_minutes": ColumnRole.OBSERVATION,
        "since_previous_minutes": ColumnRole.OBSERVATION,
        "prefix_events": ColumnRole.OBSERVATION,
        "hour_utc": ColumnRole.CONTEXT,
        "weekday_utc": ColumnRole.CONTEXT,
        "remaining_minutes": ColumnRole.OUTCOME,
        "partition": ColumnRole.IDENTIFIER,
    }
    return FieldClassification(
        roles=roles,
        rationale={field: "prefix dataset contract v1" for field in roles},
    )


def _as_xy(frame: pl.DataFrame) -> tuple[Any, np.ndarray]:
    features = frame.select(NUMERIC_FEATURES + CATEGORICAL_FEATURES).to_pandas()
    target = frame["remaining_minutes"].to_numpy()
    return features, target


def train_warehouse_baselines(
    source_path: Path,
    config_path: Path,
    output_dir: Path,
) -> dict[str, Any]:
    config: ExperimentConfig = load_experiment(config_path)
    run = RunContext.start(
        output_dir,
        ["flowtwin", "train-baselines", str(source_path), "--config", str(config_path)],
        config.claim_state,
    )
    dataset = build_prefix_dataset(
        source_path,
        prediction_points=tuple(config.prediction_points),
        max_cases=config.compute.max_cases,
        seed=config.seed,
    )
    frame = dataset.frame
    frame.write_parquet(output_dir / "prefixes.parquet")
    shutil.copyfile(config_path, output_dir / "config_resolved.yaml")
    split_payload = dataset.split.model_dump(mode="json")
    atomic_json(output_dir / "split_manifest.json", split_payload)

    partitions = [
        (row["operation_id"], Partition(row["partition"]))
        for row in frame.select("operation_id", "partition").iter_rows(named=True)
    ]
    leakage = run_leakage_audit(
        _classification(),
        prefix_feature_columns(),
        operation_partitions=partitions,
        prefix_cutoffs=[
            (row["operation_id"], row["prediction_cutoff"], row["prediction_cutoff"])
            for row in frame.select("operation_id", "prediction_cutoff").iter_rows(named=True)
        ],
    )
    leakage.require_passed()
    atomic_json(output_dir / "leakage_report.json", leakage.model_dump(mode="json"))

    train = frame.filter(pl.col("partition") == Partition.TRAIN.value)
    validation = frame.filter(pl.col("partition") == Partition.VALIDATION.value)
    test = frame.filter(pl.col("partition") == Partition.TEST.value)
    train_x, train_y = _as_xy(train)
    validation_x, validation_y = _as_xy(validation)
    test_x, test_y = _as_xy(test)

    models: dict[str, Any] = {}
    validation_predictions: dict[str, np.ndarray] = {}
    test_predictions: dict[str, np.ndarray] = {}

    global_median = MedianRemainingTime().fit(train_y)
    models["global_median"] = global_median
    validation_predictions["global_median"] = global_median.predict(None, size=len(validation_y))
    test_predictions["global_median"] = global_median.predict(None, size=len(test_y))

    activity_median = MedianRemainingTime().fit(train_y, train["last_activity"].to_numpy())
    models["activity_median"] = activity_median
    validation_predictions["activity_median"] = activity_median.predict(
        validation["last_activity"].to_numpy()
    )
    test_predictions["activity_median"] = activity_median.predict(test["last_activity"].to_numpy())

    operation_durations = (
        train.with_columns(
            (pl.col("elapsed_minutes") + pl.col("remaining_minutes")).alias("duration")
        )
        .group_by("operation_id")
        .agg(pl.col("duration").max())
    )
    survival = KaplanMeierRemainingTime().fit(
        operation_durations["duration"].to_numpy(),
        np.ones(operation_durations.height, dtype=bool),
    )
    models["kaplan_meier"] = survival
    validation_predictions["kaplan_meier"] = survival.predict(
        validation["elapsed_minutes"].to_numpy()
    )
    test_predictions["kaplan_meier"] = survival.predict(test["elapsed_minutes"].to_numpy())

    ridge = ridge_remaining_pipeline(NUMERIC_FEATURES, CATEGORICAL_FEATURES)
    ridge.fit(train_x, train_y)
    models["ridge"] = ridge
    validation_predictions["ridge"] = np.maximum(0, ridge.predict(validation_x))
    test_predictions["ridge"] = np.maximum(0, ridge.predict(test_x))

    quantile50 = quantile_boosting_pipeline(
        NUMERIC_FEATURES,
        CATEGORICAL_FEATURES,
        quantile=0.5,
        seed=config.seed,
    )
    quantile90 = quantile_boosting_pipeline(
        NUMERIC_FEATURES,
        CATEGORICAL_FEATURES,
        quantile=0.9,
        seed=config.seed,
    )
    quantile50.fit(train_x, train_y)
    quantile90.fit(train_x, train_y)
    models["quantile_p50"] = quantile50
    models["quantile_p90"] = quantile90
    validation_predictions["quantile_boosting"] = np.maximum(0, quantile50.predict(validation_x))
    test_predictions["quantile_boosting"] = np.maximum(0, quantile50.predict(test_x))
    test_p90 = np.maximum(test_predictions["quantile_boosting"], quantile90.predict(test_x))

    validation_mae = {
        name: float(mean_absolute_error(validation_y, values))
        for name, values in validation_predictions.items()
    }
    selected_name = min(validation_mae, key=lambda name: validation_mae[name])
    selected_prediction = test_predictions[selected_name]
    conformal = ConformalIntervals().fit(validation_y, validation_predictions[selected_name])
    interval50 = conformal.interval(selected_prediction, 0.5)
    interval90 = conformal.interval(selected_prediction, 0.9)

    total_train_duration = train["elapsed_minutes"].to_numpy() + train_y
    duration_threshold = float(np.quantile(total_train_duration, 0.75))
    train_risk = (total_train_duration > duration_threshold).astype(int)
    validation_risk = (
        validation["elapsed_minutes"].to_numpy() + validation_y > duration_threshold
    ).astype(int)
    test_risk = (test["elapsed_minutes"].to_numpy() + test_y > duration_threshold).astype(int)
    risk_model = logistic_risk_pipeline(NUMERIC_FEATURES, CATEGORICAL_FEATURES)
    risk_model.fit(train_x, train_risk)
    validation_probability = risk_model.predict_proba(validation_x)[:, 1]
    threshold = select_threshold_validation(validation_risk, validation_probability, method="f1")
    test_probability = risk_model.predict_proba(test_x)[:, 1]

    model_dir = output_dir / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    for name, model in models.items():
        joblib.dump(model, model_dir / f"{name}.joblib")
    joblib.dump(risk_model, model_dir / "long_duration_risk_logistic.joblib")
    joblib.dump(conformal, model_dir / "conformal.joblib")

    predictions = test.select("operation_id", "prediction_cutoff", "last_activity")
    for name, values in test_predictions.items():
        predictions = predictions.with_columns(pl.Series(f"prediction_{name}", values))
    predictions = predictions.with_columns(
        pl.Series("remaining_minutes", test_y),
        pl.Series("selected_p50", selected_prediction),
        pl.Series("quantile_p90", test_p90),
        pl.Series("interval50_lower", interval50[0]),
        pl.Series("interval50_upper", interval50[1]),
        pl.Series("interval90_lower", interval90[0]),
        pl.Series("interval90_upper", interval90[1]),
        pl.Series("long_duration_proxy_target", test_risk),
        pl.Series("long_duration_proxy_probability", test_probability),
    )
    predictions.write_parquet(output_dir / "predictions.parquet")

    test_metrics = {
        name: remaining_time_metrics(test_y, values) for name, values in test_predictions.items()
    }
    selected_metrics = remaining_time_metrics(
        test_y,
        selected_prediction,
        p90=test_p90,
        interval50=interval50,
        interval90=interval90,
    )
    uncertainty = cluster_bootstrap_mae(
        test_y,
        selected_prediction,
        test["operation_id"].to_numpy(),
        seed=config.seed,
    )
    grouped = grouped_remaining_time_metrics(
        test_y,
        selected_prediction,
        test["last_activity"].to_numpy(),
    )
    risk = risk_metrics(
        test_risk,
        test_probability,
        threshold=float(threshold["threshold"]),
        operation_ids=test["operation_id"].to_numpy(),
    )
    dataset_manifest_path = Path(config.dataset_manifest)
    dataset_manifest_payload = yaml.safe_load(dataset_manifest_path.read_text(encoding="utf-8"))
    atomic_json(output_dir / "data_manifest.json", dataset_manifest_payload)
    calibration_payload = {
        "method": "split_conformal_validation",
        "residual_q50": conformal.residual_q50,
        "residual_q90": conformal.residual_q90,
        "risk_reliability": risk["calibration"],
    }
    atomic_json(output_dir / "calibration.json", calibration_payload)

    metrics: dict[str, Any] = {
        "dataset_id": dataset_manifest_payload["dataset_id"],
        "dataset_export_version": dataset_manifest_payload["export_version"],
        "source_file": str(source_path),
        "source_file_sha256": sha256_file(source_path),
        "split_protocol": dataset.split.protocol,
        "split_counts_operations": dataset.split.counts(),
        "model_selection": {
            "validation_mae_minutes": validation_mae,
            "selected_model": selected_name,
            "test_influenced_choice": False,
        },
        "remaining_time_test": test_metrics,
        "selected_model_test": selected_metrics,
        "selected_model_mae_cluster_bootstrap": uncertainty,
        "worst_group_last_activity": grouped,
        "risk_test": risk,
        "risk_target": {
            "name": "long_duration_proxy",
            "training_duration_p75_minutes": duration_threshold,
            "not_equivalent_to": "Kaleido material plan deviation",
        },
        "threshold_selection": threshold,
        "number_of_seeds": 1,
        "seed": config.seed,
        "claim_state": config.claim_state,
        "public_data_scope": "pipeline competence only; not Kaleido value evidence",
        "source_rows_scanned": dataset.source_rows,
        "source_cases_used": dataset.source_cases,
        "prefix_rows": frame.height,
    }
    atomic_json(output_dir / "metrics.json", metrics)
    _write_model_card(output_dir / "model_card.md", config, metrics)
    _write_report(output_dir / "report.md", config, metrics)
    run.finish(
        {
            "dataset_id": metrics["dataset_id"],
            "split_protocol": metrics["split_protocol"],
            "selected_model": selected_name,
            "number_of_seeds": 1,
            "threshold_selection": threshold["selection_method"],
            "test_influenced_choice": False,
        }
    )
    return metrics


def _write_model_card(path: Path, config: ExperimentConfig, metrics: dict[str, Any]) -> None:
    selected = metrics["model_selection"]["selected_model"]
    value = metrics["selected_model_test"]
    uncertainty = metrics["selected_model_mae_cluster_bootstrap"]
    path.write_text(
        "\n".join(
            [
                "# Model card — warehouse remaining-time smoke",
                "",
                f"- Claim state: `{config.claim_state}`",
                f"- Dataset: `{metrics['dataset_id']}`, export {metrics['dataset_export_version']}",
                f"- Source SHA-256: `{metrics['source_file_sha256']}`",
                f"- Split: {metrics['split_protocol']}",
                f"- Selected on validation: `{selected}`",
                "- Test influenced a choice: no",
                f"- Seeds: {metrics['number_of_seeds']}",
                f"- Test MAE: {value['mae_minutes']:.2f} minutes",
                (
                    "- Cluster-bootstrap 95% interval for MAE: "
                    f"{uncertainty['bootstrap_95_low']:.2f}-"
                    f"{uncertainty['bootstrap_95_high']:.2f} minutes"
                ),
                f"- Threshold selection: {metrics['threshold_selection']['selection_method']}",
                "",
                "## Intended use",
                "",
                "Development validation of prefix construction, chronological grouped splits, "
                "baselines, calibration and artifact generation.",
                "",
                "## What this does not prove",
                "",
                "This public warehouse log is not a port or Kaleido dataset. The long-duration "
                "proxy is not material plan deviation. These results do not prove Kaleido "
                "accuracy, early warning, ROI, savings or production readiness.",
            ]
        ),
        encoding="utf-8",
    )


def _write_report(path: Path, config: ExperimentConfig, metrics: dict[str, Any]) -> None:
    selected = metrics["model_selection"]["selected_model"]
    selected_metrics = metrics["selected_model_test"]
    path.write_text(
        "\n".join(
            [
                "# Experiment report",
                "",
                f"## Hypothesis\n\n{config.hypothesis}",
                "",
                "## Changes",
                "",
                "Built causal prefixes, chronological operation-level splits, required simple "
                "baselines, validation-only selection and conformal intervals.",
                "",
                "## Tests and evidence",
                "",
                f"Selected model: `{selected}`.",
                f"Test MAE: {selected_metrics['mae_minutes']:.2f} minutes.",
                f"P90 interval coverage: {selected_metrics['p90_interval_coverage']:.3f}.",
                f"Claim state: `{config.claim_state}`.",
                "",
                "## Limitations",
                "",
                "One public non-port dataset, one seed, obfuscated context, no plan revisions and "
                "a proxy risk label. Test did not influence model or threshold selection.",
                "",
                "## Next falsifiable step",
                "",
                "Run the same frozen protocol for three seeds and compare GRU/ProcessTransformer "
                "against the selected tabular model before opening the Event-JEPA gate.",
            ]
        ),
        encoding="utf-8",
    )
