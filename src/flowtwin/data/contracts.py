from __future__ import annotations

from datetime import datetime
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class ClaimState(StrEnum):
    PLANNED = "planned"
    SMOKE_ONLY = "smoke_only"
    DIAGNOSTIC = "diagnostic"
    CLAIM_ELIGIBLE = "claim_eligible"
    PILOT_SHADOW = "pilot_shadow"
    VALIDATED_PILOT = "validated_pilot"


class CompletionStatus(StrEnum):
    COMPLETE = "complete"
    INCOMPLETE = "incomplete"
    CANCELLED = "cancelled"
    UNKNOWN = "unknown"


def _aware(value: datetime, field: str) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{field} must be timezone-aware")
    return value


class OperationEvent(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    event_id: str = Field(min_length=1)
    source_system: str = Field(min_length=1)
    source_record_id: str = Field(min_length=1)
    event_type: str = Field(min_length=1)
    event_time_utc: datetime
    ingested_at_utc: datetime
    project_id: str | None = None
    operation_id: str | None = None
    shift_id: str | None = None
    cargo_unit_id: str | None = None
    resource_id: str | None = None
    vessel_id: str | None = None
    location_id: str | None = None
    actor_role: str | None = None
    status_from: str | None = None
    status_to: str | None = None
    numeric_value: float | None = None
    unit: str | None = None
    data_quality_flags: tuple[str, ...] = ()
    payload_ref: str | None = None

    @field_validator("event_time_utc", "ingested_at_utc")
    @classmethod
    def timestamps_are_aware(cls, value: datetime, info: object) -> datetime:
        field_name = getattr(info, "field_name", "timestamp")
        return _aware(value, field_name)

    @model_validator(mode="after")
    def ingestion_is_not_silently_before_event(self) -> OperationEvent:
        if (
            self.ingested_at_utc < self.event_time_utc
            and "clock_skew" not in self.data_quality_flags
        ):
            raise ValueError("ingested_at_utc precedes event_time_utc without clock_skew flag")
        return self


class PlanRevision(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    plan_id: str = Field(min_length=1)
    revision: int = Field(ge=0)
    valid_from_utc: datetime
    operation_id: str = Field(min_length=1)
    milestone: str = Field(min_length=1)
    planned_start_utc: datetime | None = None
    planned_end_utc: datetime | None = None
    planned_quantity: float | None = None
    planned_resources: tuple[str, ...] = ()
    reason: str | None = None

    @field_validator("valid_from_utc", "planned_start_utc", "planned_end_utc")
    @classmethod
    def timestamps_are_aware(cls, value: datetime | None, info: object) -> datetime | None:
        if value is None:
            return None
        return _aware(value, getattr(info, "field_name", "timestamp"))

    @model_validator(mode="after")
    def interval_is_valid(self) -> PlanRevision:
        if (
            self.planned_start_utc is not None
            and self.planned_end_utc is not None
            and self.planned_end_utc < self.planned_start_utc
        ):
            raise ValueError("planned_end_utc precedes planned_start_utc")
        return self


class OperationOutcome(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    operation_id: str = Field(min_length=1)
    completed_at_utc: datetime | None = None
    completion_status: CompletionStatus
    deviation_minutes: float | None = None
    incident_types: tuple[str, ...] = ()
    incident_status_known: bool = False
    cost_eur: float | None = Field(default=None, ge=0)
    censored: bool

    @field_validator("completed_at_utc")
    @classmethod
    def completed_at_is_aware(cls, value: datetime | None) -> datetime | None:
        if value is None:
            return None
        return _aware(value, "completed_at_utc")

    @model_validator(mode="after")
    def censoring_is_consistent(self) -> OperationOutcome:
        if self.censored and self.completed_at_utc is not None:
            raise ValueError("censored outcomes cannot have an exact completion timestamp")
        if (
            not self.censored
            and self.completion_status == CompletionStatus.COMPLETE
            and self.completed_at_utc is None
        ):
            raise ValueError("completed outcome requires completed_at_utc")
        if not self.incident_status_known and self.incident_types:
            raise ValueError("incident types cannot be asserted when incident status is unknown")
        return self


class ObjectReference(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    object_id: str
    object_type: str


class EventObjectRelation(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    event_id: str
    object_id: str
    qualifier: str = ""


class ObjectRelation(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    source_object_id: str
    target_object_id: str
    qualifier: str = ""
