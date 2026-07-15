from __future__ import annotations

from pathlib import Path

import pytest

from flowtwin.data.adapters.ocel import OcelSQLiteAdapter
from flowtwin.process.discovery import discover_process

PUBLIC_OCEL = Path("data/raw/public/ocel_container_logistics_v3/container_logistics.sqlite")


@pytest.mark.slow
@pytest.mark.skipif(not PUBLIC_OCEL.is_file(), reason="public OCEL is not downloaded")
def test_public_container_ocel_round_trip() -> None:
    adapter = OcelSQLiteAdapter(PUBLIC_OCEL)
    manifest = adapter.build_manifest()
    graph = adapter.build_object_graph().validate()
    process = discover_process(adapter.events())
    assert manifest["event_rows"] > 35_000
    assert manifest["objects"] > 13_000
    assert graph.passed
    assert process["events"] > 35_000
    assert adapter.run_leakage_audit().passed
