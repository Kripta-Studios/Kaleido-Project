from __future__ import annotations

import json
from pathlib import Path

import typer


def main(metrics: Path) -> None:
    payload = json.loads(metrics.read_text(encoding="utf-8"))
    typer.echo(json.dumps(payload.get("selected_model_test", {}), indent=2))


if __name__ == "__main__":
    typer.run(main)
