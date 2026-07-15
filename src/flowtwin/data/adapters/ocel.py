from __future__ import annotations

import sqlite3
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from flowtwin.data.contracts import (
    EventObjectRelation,
    ObjectReference,
    ObjectRelation,
    OperationEvent,
)
from flowtwin.data.leakage import LeakageReport, run_leakage_audit
from flowtwin.data.object_graph import ObjectGraph
from flowtwin.data.roles import ColumnRole, FieldClassification, classify_fields
from flowtwin.data.splits import OperationSummary, SplitManifest, chronological_grouped_split
from flowtwin.data.timestamps import TimestampReport, validate_timestamps
from flowtwin.provenance import sha256_file


def _parse_utc(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError("OCEL timestamp is timezone-naive; explicit source mapping is required")
    return parsed.astimezone(UTC)


class OcelSQLiteAdapter:
    def __init__(self, path: Path, *, primary_object_type: str = "Container") -> None:
        self.path = path
        self.primary_object_type = primary_object_type
        self._events: list[OperationEvent] | None = None

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(f"file:{self.path.as_posix()}?mode=ro", uri=True)
        connection.row_factory = sqlite3.Row
        return connection

    def _event_times(self, connection: sqlite3.Connection) -> dict[str, datetime]:
        result: dict[str, datetime] = {}
        tables = [
            row[0]
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name LIKE 'event_%'"
            )
            if row[0] != "event_map_type" and row[0] != "event_object"
        ]
        for table in tables:
            columns = {row[1] for row in connection.execute(f'PRAGMA table_info("{table}")')}
            if "ocel_time" not in columns:
                continue
            for row in connection.execute(f'SELECT ocel_id, ocel_time FROM "{table}"'):
                result[str(row["ocel_id"])] = _parse_utc(str(row["ocel_time"]))
        return result

    def events(self) -> list[OperationEvent]:
        if self._events is not None:
            return self._events
        with self._connect() as connection:
            times = self._event_times(connection)
            object_types = {
                str(row["ocel_id"]): str(row["ocel_type"])
                for row in connection.execute("SELECT ocel_id, ocel_type FROM object")
            }
            event_objects: dict[str, list[str]] = {}
            for row in connection.execute(
                "SELECT ocel_event_id, ocel_object_id FROM event_object ORDER BY rowid"
            ):
                event_objects.setdefault(str(row["ocel_event_id"]), []).append(
                    str(row["ocel_object_id"])
                )
            values: list[OperationEvent] = []
            for row in connection.execute("SELECT ocel_id, ocel_type FROM event ORDER BY rowid"):
                event_id = str(row["ocel_id"])
                if event_id not in times:
                    continue
                linked = event_objects.get(event_id, [])
                primary = next(
                    (
                        object_id
                        for object_id in linked
                        if object_types.get(object_id) == self.primary_object_type
                    ),
                    None,
                )
                operation_id = primary or (linked[0] if linked else None)
                values.append(
                    OperationEvent(
                        event_id=event_id,
                        source_system="ocel_sqlite",
                        source_record_id=event_id,
                        event_type=str(row["ocel_type"]),
                        event_time_utc=times[event_id],
                        ingested_at_utc=times[event_id],
                        operation_id=operation_id,
                    )
                )
        self._events = values
        return values

    def build_manifest(self) -> dict[str, Any]:
        with self._connect() as connection:
            event_count = int(connection.execute("SELECT COUNT(*) FROM event").fetchone()[0])
            object_count = int(connection.execute("SELECT COUNT(*) FROM object").fetchone()[0])
            relation_count = int(
                connection.execute("SELECT COUNT(*) FROM event_object").fetchone()[0]
            )
        parsed = self.events()
        timestamps = [event.event_time_utc for event in parsed]
        return {
            "dataset_id": self.path.stem,
            "source_system": "ocel_sqlite",
            "sha256": sha256_file(self.path),
            "event_rows": event_count,
            "parsed_events_with_timestamp": len(parsed),
            "objects": object_count,
            "event_object_relations": relation_count,
            "date_min": min(timestamps).isoformat() if timestamps else None,
            "date_max": max(timestamps).isoformat() if timestamps else None,
            "primary_object_type": self.primary_object_type,
        }

    def validate_timestamps(self) -> TimestampReport:
        return validate_timestamps(self.events())

    def classify_fields(self) -> FieldClassification:
        return classify_fields(
            ["ocel_id", "ocel_type", "ocel_time", "ocel_object_id", "ocel_qualifier"],
            {
                "ocel_id": ColumnRole.IDENTIFIER,
                "ocel_type": ColumnRole.OBSERVATION,
                "ocel_time": ColumnRole.TIMESTAMP,
                "ocel_object_id": ColumnRole.IDENTIFIER,
                "ocel_qualifier": ColumnRole.OBSERVATION,
            },
        )

    def build_object_graph(self) -> ObjectGraph:
        with self._connect() as connection:
            objects = [
                ObjectReference(object_id=str(row["ocel_id"]), object_type=str(row["ocel_type"]))
                for row in connection.execute("SELECT ocel_id, ocel_type FROM object")
            ]
            event_object = [
                EventObjectRelation(
                    event_id=str(row["ocel_event_id"]),
                    object_id=str(row["ocel_object_id"]),
                    qualifier=str(row["ocel_qualifier"] or ""),
                )
                for row in connection.execute(
                    "SELECT ocel_event_id, ocel_object_id, ocel_qualifier FROM event_object"
                )
            ]
            object_object = [
                ObjectRelation(
                    source_object_id=str(row["ocel_source_id"]),
                    target_object_id=str(row["ocel_target_id"]),
                    qualifier=str(row["ocel_qualifier"] or ""),
                )
                for row in connection.execute(
                    "SELECT ocel_source_id, ocel_target_id, ocel_qualifier FROM object_object"
                )
            ]
            event_ids = [
                str(row["ocel_id"]) for row in connection.execute("SELECT ocel_id FROM event")
            ]
        return ObjectGraph(objects, event_ids, event_object, object_object)

    def build_grouped_splits(self) -> SplitManifest:
        starts: dict[str, datetime] = {}
        for event in self.events():
            if event.operation_id is None:
                continue
            starts[event.operation_id] = min(
                starts.get(event.operation_id, event.event_time_utc), event.event_time_utc
            )
        return chronological_grouped_split(
            OperationSummary(operation_id=operation_id, start_time=start)
            for operation_id, start in starts.items()
        )

    def run_leakage_audit(self, unsafe_debug: bool = False) -> LeakageReport:
        report = run_leakage_audit(
            self.classify_fields(),
            feature_fields=["ocel_type", "ocel_qualifier"],
            unsafe_debug=unsafe_debug,
        )
        if not unsafe_debug:
            report.require_passed()
        return report
