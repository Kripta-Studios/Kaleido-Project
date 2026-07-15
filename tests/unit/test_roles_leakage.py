from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from flowtwin.data.leakage import run_leakage_audit
from flowtwin.data.roles import ColumnRole, FieldClassification, classify_fields
from flowtwin.data.splits import Partition


def _classification() -> FieldClassification:
    roles = {
        "elapsed": ColumnRole.OBSERVATION,
        "cargo": ColumnRole.CONTEXT,
        "operator_action": ColumnRole.ACTION,
        "completed_at": ColumnRole.OUTCOME,
        "post_cutoff_notes": ColumnRole.FORBIDDEN,
    }
    return FieldClassification(
        roles=roles,
        rationale={key: "test contract" for key in roles},
    )


def test_action_context_and_outcome_are_disjoint() -> None:
    classification = _classification()
    assert classification.fields(ColumnRole.ACTION) == {"operator_action"}
    assert not (
        classification.fields(ColumnRole.ACTION) & classification.fields(ColumnRole.CONTEXT)
    )


def test_future_outcome_hidden_in_column_name_fails_closed() -> None:
    report = run_leakage_audit(_classification(), ["elapsed", "post_cutoff_notes"])
    assert not report.passed
    with pytest.raises(RuntimeError, match="failed closed"):
        report.require_passed()


def test_unsafe_debug_watermarks_but_does_not_pass() -> None:
    report = run_leakage_audit(_classification(), ["completed_at"], unsafe_debug=True)
    assert not report.passed
    assert report.watermark == "UNSAFE_DEBUG_NOT_CLAIM_ELIGIBLE"


def test_operation_crossing_partitions_is_detected() -> None:
    report = run_leakage_audit(
        _classification(),
        ["elapsed"],
        operation_partitions=[("op", Partition.TRAIN), ("op", Partition.TEST)],
    )
    assert any(item.code == "operation_crosses_split" for item in report.findings)


def test_event_after_cutoff_is_detected() -> None:
    cutoff = datetime(2025, 1, 1, tzinfo=UTC)
    report = run_leakage_audit(
        _classification(),
        ["elapsed"],
        prefix_cutoffs=[("op", cutoff + timedelta(seconds=1), cutoff)],
    )
    assert any(item.code == "event_after_prediction_cutoff" for item in report.findings)


def test_name_classifier_is_conservative() -> None:
    result = classify_fields(["remaining_time", "customer", "resource_assignment"])
    assert result.roles["remaining_time"] == ColumnRole.FORBIDDEN
    assert result.roles["customer"] == ColumnRole.CONTEXT
    assert result.roles["resource_assignment"] == ColumnRole.ACTION
