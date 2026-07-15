"""OCEL is built by source-specific adapters; this entry validates an OCEL SQLite file."""

from pathlib import Path

import typer

from flowtwin.data.adapters.ocel import OcelSQLiteAdapter


def main(source: Path) -> None:
    report = OcelSQLiteAdapter(source).build_object_graph().validate()
    typer.echo(report.model_dump_json(indent=2))
    if not report.passed:
        raise typer.Exit(1)


if __name__ == "__main__":
    typer.run(main)
