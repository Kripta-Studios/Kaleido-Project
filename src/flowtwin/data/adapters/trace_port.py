from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import polars as pl

from flowtwin.data.contracts import (
    EventObjectRelation,
    ObjectReference,
    OperationEvent,
)
from flowtwin.data.leakage import LeakageReport, run_leakage_audit
from flowtwin.data.object_graph import ObjectGraph
from flowtwin.data.roles import ColumnRole, FieldClassification, classify_fields
from flowtwin.data.splits import OperationSummary, SplitManifest, chronological_grouped_split
from flowtwin.data.timestamps import TimestampReport, validate_timestamps
from flowtwin.provenance import sha256_file

DEFAULT_ALIASES: dict[str, tuple[str, ...]] = {
    "event_id": ("event_id", "id", "eventId"),
    "source_record_id": ("source_record_id", "record_id", "id", "eventId"),
    "event_type": ("event_type", "activity", "concept:name", "eventName"),
    "event_time": ("event_time", "timestamp", "time:timestamp", "occurred_at"),
    "ingested_at": ("ingested_at", "recorded_at", "created_at"),
    "operation_id": ("operation_id", "operation", "case:concept:name", "case_id"),
    "project_id": ("project_id", "project"),
    "shift_id": ("shift_id", "shift"),
    "cargo_unit_id": ("cargo_unit_id", "cargo_id", "packing_list_line_id"),
    "resource_id": ("resource_id", "equipment_id", "org:resource"),
    "vessel_id": ("vessel_id", "vessel"),
    "location_id": ("location_id", "berth_id", "location"),
    "actor_role": ("actor_role", "role"),
    "status_from": ("status_from",),
    "status_to": ("status_to", "status"),
    "numeric_value": ("numeric_value", "quantity"),
    "unit": ("unit",),
    "payload_ref": ("payload_ref", "photo_ref", "document_ref"),
}


class TracePortAdapter:
    """Conservative read-only adapter for CSV, JSON/JSONL and Parquet exports."""

    def __init__(
        self,
        path: Path,
        *,
        source_system: str = "trace_port",
        timezone: str = "Europe/Madrid",
        aliases: dict[str, tuple[str, ...]] | None = None,
        role_overrides: dict[str, ColumnRole] | None = None,
    ) -> None:
        self.path = path
        self.source_system = source_system
        self.timezone = ZoneInfo(timezone)
        self.aliases = aliases or DEFAULT_ALIASES
        self.role_overrides = role_overrides or {}
        self._frame: pl.DataFrame | None = None
        self._events: list[OperationEvent] | None = None

    def _load(self) -> pl.DataFrame:
        if self._frame is not None:
            return self._frame
        suffix = self.path.suffix.lower()
        if suffix == ".csv":
            frame = pl.read_csv(self.path, infer_schema_length=10000, try_parse_dates=False)
        elif suffix in {".json", ".jsonl", ".ndjson"}:
            frame = pl.read_ndjson(self.path)
        elif suffix in {".parquet", ".pq"}:
            frame = pl.read_parquet(self.path)
        else:
            raise ValueError(f"unsupported Trace Port export format: {suffix}")
        self._frame = frame
        return frame

    def _column(self, canonical: str, required: bool = False) -> str | None:
        columns = set(self._load().columns)
        for candidate in self.aliases.get(canonical, (canonical,)):
            if candidate in columns:
                return candidate
        if required:
            raise ValueError(
                f"required canonical field {canonical!r} not found; candidates="
                f"{self.aliases.get(canonical, ())}"
            )
        return None

    def _parse_timestamp(self, value: Any) -> datetime:
        if isinstance(value, datetime):
            parsed = value
        elif isinstance(value, str):
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        else:
            raise ValueError(f"invalid timestamp value: {value!r}")
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=self.timezone)
        return parsed.astimezone(UTC)

    def events(self) -> list[OperationEvent]:
        if self._events is not None:
            return self._events
        frame = self._load()
        event_id_col = self._column("event_id")
        source_id_col = self._column("source_record_id")
        event_type_col = self._column("event_type", required=True)
        event_time_col = self._column("event_time", required=True)
        assert event_type_col is not None
        assert event_time_col is not None
        ingested_col = self._column("ingested_at")
        optional = {
            key: self._column(key)
            for key in (
                "operation_id",
                "project_id",
                "shift_id",
                "cargo_unit_id",
                "resource_id",
                "vessel_id",
                "location_id",
                "actor_role",
                "status_from",
                "status_to",
                "numeric_value",
                "unit",
                "payload_ref",
            )
        }
        events: list[OperationEvent] = []

        def string_value(row_mapping: dict[str, Any], name: str) -> str | None:
            column = optional[name]
            raw = row_mapping.get(column) if column else None
            return None if raw is None else str(raw)

        for row_index, row in enumerate(frame.iter_rows(named=True)):
            event_time = self._parse_timestamp(row[event_time_col])
            ingested_at = (
                self._parse_timestamp(row[ingested_col])
                if ingested_col and row.get(ingested_col) is not None
                else event_time
            )
            flags: tuple[str, ...] = ("clock_skew",) if ingested_at < event_time else ()
            source_record_id = (
                str(row[source_id_col])
                if source_id_col and row.get(source_id_col) is not None
                else str(row_index)
            )
            event_id = (
                str(row[event_id_col])
                if event_id_col and row.get(event_id_col) is not None
                else f"{self.source_system}:{source_record_id}"
            )

            numeric_column = optional["numeric_value"]
            numeric_raw = row.get(numeric_column) if numeric_column else None
            events.append(
                OperationEvent(
                    event_id=event_id,
                    source_system=self.source_system,
                    source_record_id=source_record_id,
                    event_type=str(row[event_type_col]),
                    event_time_utc=event_time,
                    ingested_at_utc=ingested_at,
                    project_id=string_value(row, "project_id"),
                    operation_id=string_value(row, "operation_id"),
                    shift_id=string_value(row, "shift_id"),
                    cargo_unit_id=string_value(row, "cargo_unit_id"),
                    resource_id=string_value(row, "resource_id"),
                    vessel_id=string_value(row, "vessel_id"),
                    location_id=string_value(row, "location_id"),
                    actor_role=string_value(row, "actor_role"),
                    status_from=string_value(row, "status_from"),
                    status_to=string_value(row, "status_to"),
                    numeric_value=float(numeric_raw) if numeric_raw is not None else None,
                    unit=string_value(row, "unit"),
                    data_quality_flags=flags,
                    payload_ref=string_value(row, "payload_ref"),
                )
            )
        self._events = events
        return events

    def build_manifest(self) -> dict[str, Any]:
        frame = self._load()
        events = self.events()
        operations = {event.operation_id for event in events if event.operation_id}
        projects = {event.project_id for event in events if event.project_id}
        timestamps = [event.event_time_utc for event in events]
        return {
            "dataset_id": f"{self.source_system}_export",
            "owner": "unknown",
            "source_system": self.source_system,
            "export_version": "unknown",
            "access_date": datetime.now(UTC).date().isoformat(),
            "license_or_agreement": "pending",
            "timezone_source": str(self.timezone),
            "rows": frame.height,
            "operations": len(operations),
            "projects": len(projects),
            "date_min": min(timestamps).isoformat() if timestamps else None,
            "date_max": max(timestamps).isoformat() if timestamps else None,
            "sha256": sha256_file(self.path),
            "contains_personal_data": "unknown",
            "contains_photos": self._column("payload_ref") is not None,
            "plan_revisions_available": "unknown",
            "outcomes_available": "unknown",
            "columns": frame.columns,
        }

    def validate_timestamps(self) -> TimestampReport:
        return validate_timestamps(self.events())

    def classify_fields(self) -> FieldClassification:
        overrides = dict(self.role_overrides)
        for canonical in ("event_id", "operation_id", "project_id", "shift_id"):
            column = self._column(canonical)
            if column:
                overrides.setdefault(column, ColumnRole.IDENTIFIER)
        for canonical in ("event_time", "ingested_at"):
            column = self._column(canonical)
            if column:
                overrides.setdefault(column, ColumnRole.TIMESTAMP)
        return classify_fields(self._load().columns, overrides)

    def build_object_graph(self) -> ObjectGraph:
        objects: dict[str, ObjectReference] = {}
        relations: list[EventObjectRelation] = []
        type_fields = {
            "project": "project_id",
            "operation": "operation_id",
            "shift": "shift_id",
            "cargo_unit": "cargo_unit_id",
            "resource": "resource_id",
            "vessel": "vessel_id",
            "location": "location_id",
        }
        for event in self.events():
            for object_type, field in type_fields.items():
                object_id = getattr(event, field)
                if object_id is None:
                    continue
                canonical_id = f"{object_type}:{object_id}"
                objects[canonical_id] = ObjectReference(
                    object_id=canonical_id, object_type=object_type
                )
                relations.append(
                    EventObjectRelation(
                        event_id=event.event_id,
                        object_id=canonical_id,
                        qualifier="participates",
                    )
                )
        return ObjectGraph(objects.values(), (event.event_id for event in self.events()), relations)

    def build_grouped_splits(self) -> SplitManifest:
        starts: dict[str, datetime] = {}
        for event in self.events():
            if event.operation_id is None:
                continue
            current = starts.get(event.operation_id)
            starts[event.operation_id] = min(current or event.event_time_utc, event.event_time_utc)
        if len(starts) < 3:
            raise ValueError(
                "at least three linked operations are required to build grouped splits"
            )
        return chronological_grouped_split(
            OperationSummary(operation_id=key, start_time=value) for key, value in starts.items()
        )

    def run_leakage_audit(self, unsafe_debug: bool = False) -> LeakageReport:
        classification = self.classify_fields()
        feature_fields = [
            field
            for field, role in classification.roles.items()
            if role in {ColumnRole.ACTION, ColumnRole.CONTEXT, ColumnRole.OBSERVATION}
        ]
        report = run_leakage_audit(
            classification=classification,
            feature_fields=feature_fields,
            unsafe_debug=unsafe_debug,
        )
        if not unsafe_debug:
            report.require_passed()
        return report

    def audit(self, unsafe_debug: bool = False) -> dict[str, Any]:
        leakage = self.run_leakage_audit(unsafe_debug=unsafe_debug)
        return {
            "manifest": self.build_manifest(),
            "timestamp_report": self.validate_timestamps().model_dump(mode="json"),
            "field_classification": self.classify_fields().model_dump(mode="json"),
            "object_graph": self.build_object_graph().validate().model_dump(mode="json"),
            "split_manifest": self.build_grouped_splits().model_dump(mode="json"),
            "leakage_report": leakage.model_dump(mode="json"),
            "watermark": leakage.watermark,
        }


def dump_audit(path: Path, audit: dict[str, Any]) -> None:
    path.write_text(json.dumps(audit, indent=2, sort_keys=True), encoding="utf-8")
