from __future__ import annotations

import json
from pathlib import Path

from flowtwin.provenance import RunContext


def test_run_manifest_never_hashes_itself_when_directory_is_reused(tmp_path: Path) -> None:
    output = tmp_path / "run"
    output.mkdir()
    (output / "artifact.txt").write_text("evidence", encoding="utf-8")
    (output / "run_manifest.json").write_text("stale", encoding="utf-8")

    manifest = RunContext.start(output, ["flowtwin", "test"], "smoke_only").finish()

    persisted = json.loads((output / "run_manifest.json").read_text(encoding="utf-8"))
    assert "artifact.txt" in manifest["artifacts"]
    assert "run_manifest.json" not in manifest["artifacts"]
    assert persisted["artifacts"] == manifest["artifacts"]
