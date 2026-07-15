from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from flowtwin.data.contracts import EventObjectRelation, ObjectReference
from flowtwin.data.object_graph import ObjectGraph
from flowtwin.data.splits import (
    OperationSummary,
    Partition,
    assert_group_disjoint,
    chronological_grouped_split,
)


def test_object_graph_reports_orphans_and_missing_objects() -> None:
    graph = ObjectGraph(
        [ObjectReference(object_id="o1", object_type="operation")],
        ["e1", "e2"],
        [
            EventObjectRelation(event_id="e1", object_id="o1"),
            EventObjectRelation(event_id="e1", object_id="missing"),
        ],
    )
    report = graph.validate()
    assert not report.passed
    assert report.orphan_event_ids == ["e2"]
    assert report.missing_object_ids == ["missing"]


def test_chronological_split_keeps_operations_disjoint() -> None:
    start = datetime(2025, 1, 1, tzinfo=UTC)
    operations = [
        OperationSummary(operation_id=f"op-{index}", start_time=start + timedelta(days=index))
        for index in range(20)
    ]
    split = chronological_grouped_split(operations)
    assert split.assignments["op-0"] == Partition.TRAIN
    assert split.assignments["op-19"] == Partition.TEST
    assert split.counts() == {"train": 14, "validation": 3, "test": 3}


def test_row_split_mismatch_fails() -> None:
    with pytest.raises(ValueError, match="multiple partitions"):
        assert_group_disjoint(
            {"op": Partition.TRAIN},
            [("op", Partition.TRAIN), ("op", Partition.TEST)],
        )
