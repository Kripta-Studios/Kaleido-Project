from __future__ import annotations

import numpy as np

from flowtwin.baselines.boosting import ConformalIntervals
from flowtwin.baselines.naive import MedianRemainingTime
from flowtwin.baselines.survival import KaplanMeierRemainingTime
from flowtwin.evaluation.calibration import expected_calibration_error
from flowtwin.evaluation.remaining_time import remaining_time_metrics
from flowtwin.evaluation.risk import risk_metrics, select_threshold_validation


def test_median_and_survival_baselines() -> None:
    target = np.asarray([10, 20, 30, 40], dtype=float)
    median = MedianRemainingTime().fit(target)
    assert median.predict(None, size=2).tolist() == [25.0, 25.0]
    km = KaplanMeierRemainingTime().fit(target, np.ones(4, dtype=bool))
    prediction = km.predict(np.asarray([0.0]))
    assert 20 <= prediction[0] <= 30


def test_conformal_intervals_and_coverage() -> None:
    target = np.asarray([10, 20, 30, 40], dtype=float)
    prediction = np.asarray([12, 18, 33, 37], dtype=float)
    conformal = ConformalIntervals().fit(target, prediction)
    interval90 = conformal.interval(prediction, 0.9)
    metrics = remaining_time_metrics(target, prediction, interval90=interval90)
    assert metrics["p90_interval_coverage"] >= 0.75
    assert metrics["mae_minutes"] == 2.5


def test_risk_threshold_is_selected_on_validation_input() -> None:
    target = np.asarray([0, 0, 1, 1])
    probability = np.asarray([0.1, 0.2, 0.7, 0.9])
    selected = select_threshold_validation(target, probability)
    metrics = risk_metrics(target, probability, threshold=float(selected["threshold"]))
    assert selected["selection_method"] == "validation_f1"
    assert metrics["auprc"] == 1.0
    assert metrics["brier"] < 0.1


def test_ece_is_zero_for_balanced_bins() -> None:
    target = np.asarray([0, 1])
    probability = np.asarray([0.0, 1.0])
    ece, _ = expected_calibration_error(target, probability, bins=2)
    assert ece == 0
