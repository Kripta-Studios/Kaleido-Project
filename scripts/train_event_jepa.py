from __future__ import annotations

import json
from pathlib import Path

import typer


def main(
    evidence_path: Path = Path("outputs/warehouse_sequence_smoke_v1/m4_gate.json"),
) -> None:
    if not evidence_path.is_file():
        typer.echo(
            "Event-JEPA gate is closed: M1-M4 evidence is missing. "
            "Run audit, process discovery, tabular and sequential baselines first."
        )
        raise typer.Exit(2)
    evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
    if not evidence.get("event_jepa_allowed_for_public_smoke", False):
        typer.echo("Event-JEPA gate is closed by the recorded M1-M4 evidence.")
        raise typer.Exit(2)
    typer.echo("Gate open. Event-JEPA implementation is not yet promoted.")


if __name__ == "__main__":
    typer.run(main)
