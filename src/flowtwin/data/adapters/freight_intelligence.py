from __future__ import annotations

from pathlib import Path

from flowtwin.data.adapters.trace_port import TracePortAdapter


class FreightIntelligenceAdapter(TracePortAdapter):
    def __init__(self, path: Path, *, timezone: str = "UTC") -> None:
        super().__init__(path, source_system="freight_intelligence", timezone=timezone)
