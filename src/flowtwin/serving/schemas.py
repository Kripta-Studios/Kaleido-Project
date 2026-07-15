from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from flowtwin.data.contracts import OperationEvent, PlanRevision


class EventBatchRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    events: list[OperationEvent] = Field(min_length=1, max_length=10000)


class ValidationResponse(BaseModel):
    accepted: bool
    event_count: int
    batch_sha256: str
    persisted: Literal[False] = False
    findings: list[dict[str, Any]]


class ScorePrefixRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    operation_id: str
    events: list[OperationEvent] = Field(min_length=1, max_length=10000)
    plan_revisions: list[PlanRevision] = Field(default_factory=list)
    prediction_time: datetime
    estimated_progress: float | None = Field(default=None, ge=0, le=1)
    horizon_hours: int = Field(default=8, ge=1, le=168)

    @model_validator(mode="after")
    def events_match_operation_and_cutoff(self) -> ScorePrefixRequest:
        if self.prediction_time.tzinfo is None:
            raise ValueError("prediction_time must be timezone-aware")
        mismatched = [
            event.event_id
            for event in self.events
            if event.operation_id not in {None, self.operation_id}
        ]
        if mismatched:
            raise ValueError(f"events belong to another operation: {mismatched[:5]}")
        future = [
            event.event_id for event in self.events if event.event_time_utc > self.prediction_time
        ]
        if future:
            raise ValueError(f"events occur after prediction_time: {future[:5]}")
        return self


class PredictionInterval(BaseModel):
    lower_minutes: float = Field(ge=0)
    upper_minutes: float = Field(ge=0)
    nominal_coverage: float = Field(gt=0, lt=1)


class ScoreResponse(BaseModel):
    model_version: str
    claim_state: str
    data_cutoff: datetime
    plan_revision: int | None
    prediction_time: datetime
    horizon_hours: int
    remaining_time_p50_minutes: float = Field(ge=0)
    remaining_time_p90_minutes: float = Field(ge=0)
    interval: PredictionInterval
    deviation_risk: float = Field(ge=0, le=1)
    confidence: Literal["low", "medium", "high"]
    abstained: bool
    reason_codes: list[str]
    source_event_ids: list[str]
    audit_id: str
    writes_to_source_system: Literal[False] = False


class ScenarioRequest(BaseModel):
    operation_id: str
    approved_actions: list[str] = Field(min_length=1, max_length=10)


class ModelCardResponse(BaseModel):
    model_version: str
    claim_state: str
    dataset_id: str
    split_protocol: str
    metrics: dict[str, Any]
    limitations: list[str]
