from __future__ import annotations

import re
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, model_validator


class ColumnRole(StrEnum):
    IDENTIFIER = "identifier"
    TIMESTAMP = "timestamp"
    ACTION = "action"
    CONTEXT = "context"
    OBSERVATION = "observation"
    OUTCOME = "outcome"
    FORBIDDEN = "forbidden"
    UNKNOWN = "unknown"


FUTURE_PATTERNS = tuple(
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        r"future",
        r"final[_ -]?plan",
        r"time[_ -]?to[_ -]?(finish|end|complete)",
        r"remaining[_ -]?time",
        r"post[_ -]?cutoff",
        r"actual[_ -]?(end|finish|complete)",
        r"outcome",
        r"target",
        r"label",
    )
)


class FieldClassification(BaseModel):
    model_config = ConfigDict(extra="forbid")

    roles: dict[str, ColumnRole] = Field(default_factory=dict)
    rationale: dict[str, str] = Field(default_factory=dict)

    @model_validator(mode="after")
    def every_role_has_a_rationale(self) -> FieldClassification:
        missing = set(self.roles) - set(self.rationale)
        if missing:
            raise ValueError(f"missing field-role rationale: {sorted(missing)}")
        return self

    def fields(self, role: ColumnRole) -> set[str]:
        return {field for field, assigned in self.roles.items() if assigned == role}


def suggest_role(column: str) -> ColumnRole:
    normalized = column.lower()
    if any(pattern.search(normalized) for pattern in FUTURE_PATTERNS):
        return ColumnRole.FORBIDDEN
    if normalized.endswith("_id") or normalized in {"id", "case:concept:name"}:
        return ColumnRole.IDENTIFIER
    if "timestamp" in normalized or normalized.endswith("_at") or normalized.endswith("_time"):
        return ColumnRole.TIMESTAMP
    if any(token in normalized for token in ("assign", "approve", "sequence_change", "replan")):
        return ColumnRole.ACTION
    if any(
        token in normalized
        for token in ("cargo", "vessel", "client", "customer", "weather", "shift")
    ):
        return ColumnRole.CONTEXT
    if any(token in normalized for token in ("event", "status", "quantity", "activity")):
        return ColumnRole.OBSERVATION
    return ColumnRole.UNKNOWN


def classify_fields(
    columns: list[str], overrides: dict[str, ColumnRole] | None = None
) -> FieldClassification:
    overrides = overrides or {}
    roles: dict[str, ColumnRole] = {}
    rationale: dict[str, str] = {}
    for column in columns:
        if column in overrides:
            roles[column] = overrides[column]
            rationale[column] = "explicit adapter mapping"
        else:
            roles[column] = suggest_role(column)
            rationale[column] = "conservative name-based suggestion; requires owner review"
    return FieldClassification(roles=roles, rationale=rationale)
