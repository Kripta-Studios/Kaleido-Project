from __future__ import annotations

from typing import Any, Protocol

from flowtwin.data.contracts import OperationEvent
from flowtwin.data.leakage import LeakageReport
from flowtwin.data.object_graph import ObjectGraph
from flowtwin.data.roles import FieldClassification
from flowtwin.data.splits import SplitManifest
from flowtwin.data.timestamps import TimestampReport


class OperationalAdapter(Protocol):
    def build_manifest(self) -> dict[str, Any]: ...

    def validate_timestamps(self) -> TimestampReport: ...

    def classify_fields(self) -> FieldClassification: ...

    def build_object_graph(self) -> ObjectGraph: ...

    def build_grouped_splits(self) -> SplitManifest: ...

    def run_leakage_audit(self, unsafe_debug: bool = False) -> LeakageReport: ...

    def events(self) -> list[OperationEvent]: ...
