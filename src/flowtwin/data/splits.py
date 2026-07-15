from __future__ import annotations

from collections import Counter
from collections.abc import Iterable
from datetime import datetime
from enum import StrEnum

from pydantic import BaseModel, Field, model_validator


class Partition(StrEnum):
    TRAIN = "train"
    VALIDATION = "validation"
    TEST = "test"


class OperationSummary(BaseModel):
    operation_id: str
    start_time: datetime
    group_id: str | None = None


class SplitManifest(BaseModel):
    protocol: str
    seed: int
    assignments: dict[str, Partition]
    cutoffs: dict[str, datetime | None] = Field(default_factory=dict)

    @model_validator(mode="after")
    def has_all_partitions(self) -> SplitManifest:
        counts = Counter(self.assignments.values())
        missing = set(Partition) - set(counts)
        if missing:
            raise ValueError(f"split is missing partitions: {sorted(missing)}")
        return self

    def counts(self) -> dict[str, int]:
        return {
            partition.value: count
            for partition, count in Counter(self.assignments.values()).items()
        }


def chronological_grouped_split(
    operations: Iterable[OperationSummary],
    train_fraction: float = 0.7,
    validation_fraction: float = 0.15,
    seed: int = 42,
) -> SplitManifest:
    if not 0 < train_fraction < 1:
        raise ValueError("train_fraction must be in (0, 1)")
    if not 0 < validation_fraction < 1 - train_fraction:
        raise ValueError("validation_fraction leaves no test partition")
    values = sorted(operations, key=lambda item: (item.start_time, item.operation_id))
    if len(values) < 3:
        raise ValueError("at least three operations are required for a three-way split")
    train_end = max(1, min(len(values) - 2, int(len(values) * train_fraction)))
    validation_end = max(
        train_end + 1,
        min(len(values) - 1, int(len(values) * (train_fraction + validation_fraction))),
    )
    assignments: dict[str, Partition] = {}
    for index, operation in enumerate(values):
        if index < train_end:
            partition = Partition.TRAIN
        elif index < validation_end:
            partition = Partition.VALIDATION
        else:
            partition = Partition.TEST
        assignments[operation.operation_id] = partition
    return SplitManifest(
        protocol="chronological_future_grouped_by_operation",
        seed=seed,
        assignments=assignments,
        cutoffs={
            "train_end": values[train_end - 1].start_time,
            "validation_end": values[validation_end - 1].start_time,
        },
    )


def assert_group_disjoint(
    assignments: dict[str, Partition], rows: Iterable[tuple[str, Partition]]
) -> None:
    observed: dict[str, set[Partition]] = {}
    for operation_id, partition in rows:
        observed.setdefault(operation_id, set()).add(partition)
    leaking = {key: value for key, value in observed.items() if len(value) > 1}
    if leaking:
        raise ValueError(f"operations cross multiple partitions: {sorted(leaking)}")
    mismatched = {
        key: next(iter(parts))
        for key, parts in observed.items()
        if key in assignments and next(iter(parts)) != assignments[key]
    }
    if mismatched:
        raise ValueError(f"row partitions do not match split manifest: {sorted(mismatched)}")
