from __future__ import annotations

from pathlib import Path

from flowtwin.data.adapters.trace_port import TracePortAdapter


class ShippingBoardAdapter(TracePortAdapter):
    def __init__(self, path: Path, *, timezone: str = "Europe/Madrid") -> None:
        super().__init__(path, source_system="shipping_board", timezone=timezone)
