from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated, Any

import typer
import uvicorn
from rich.console import Console
from rich.table import Table

from flowtwin.data.adapters.ocel import OcelSQLiteAdapter
from flowtwin.data.adapters.trace_port import TracePortAdapter
from flowtwin.data.manifests import load_manifest, verify_manifest_files
from flowtwin.data.synthetic import generate_trace_port_fixture
from flowtwin.process.bottlenecks import bottleneck_report
from flowtwin.process.discovery import discover_process
from flowtwin.process.variants import variant_report
from flowtwin.provenance import RunContext, atomic_json
from flowtwin.reporting import process_report_html, write_evidence_pdf
from flowtwin.sequence_training import train_sequence_models
from flowtwin.serving.api import create_app
from flowtwin.training import train_warehouse_baselines

app = typer.Typer(
    name="flowtwin",
    help="Kaleido FlowTwin read-only predictive operations toolkit.",
    no_args_is_help=True,
)
console = Console()


@app.command("demo")
def demo(
    output: Annotated[Path, typer.Option("--output", "-o")] = Path(
        "data/processed/demo_trace_port"
    ),
    operations: Annotated[int, typer.Option(min=10)] = 240,
    seed: int = 42,
) -> None:
    summary = generate_trace_port_fixture(output, operations=operations, seed=seed)
    console.print(
        f"[green]Synthetic fixture created[/green]: {summary.operations} operations, "
        f"{summary.events} events, {summary.censored} censored. "
        "[yellow]SMOKE_ONLY — not Kaleido evidence.[/yellow]"
    )


@app.command("audit")
def audit(
    source: Path,
    output: Annotated[Path, typer.Option("--output", "-o")] = Path("outputs/audit"),
    timezone: str = "Europe/Madrid",
    unsafe_debug: Annotated[bool, typer.Option("--unsafe-debug")] = False,
) -> None:
    run = RunContext.start(
        output,
        ["flowtwin", "audit", str(source)],
        "smoke_only" if unsafe_debug else "diagnostic",
    )
    adapter = TracePortAdapter(source, timezone=timezone)
    payload = adapter.audit(unsafe_debug=unsafe_debug)
    atomic_json(output / "audit_report.json", payload)
    atomic_json(output / "data_manifest.json", payload["manifest"])
    atomic_json(output / "split_manifest.json", payload["split_manifest"])
    atomic_json(output / "leakage_report.json", payload["leakage_report"])
    run.finish(
        {
            "dataset_id": payload["manifest"]["dataset_id"],
            "split_protocol": payload["split_manifest"]["protocol"],
            "test_influenced_choice": False,
        }
    )
    console.print(
        f"[green]Audit complete[/green] — {payload['manifest']['rows']} rows; "
        f"leakage passed={payload['leakage_report']['passed']}"
    )


@app.command("discover")
def discover(
    source: Path,
    output: Annotated[Path, typer.Option("--output", "-o")] = Path("outputs/process"),
    primary_object_type: str = "Container",
) -> None:
    run = RunContext.start(
        output,
        ["flowtwin", "discover", str(source)],
        "smoke_only",
    )
    adapter = OcelSQLiteAdapter(source, primary_object_type=primary_object_type)
    events = adapter.events()
    payload: dict[str, Any] = {
        "manifest": adapter.build_manifest(),
        "timestamp_report": adapter.validate_timestamps().model_dump(mode="json"),
        "object_graph": adapter.build_object_graph().validate().model_dump(mode="json"),
        "leakage_report": adapter.run_leakage_audit().model_dump(mode="json"),
        "split_manifest": adapter.build_grouped_splits().model_dump(mode="json"),
        "discovery": discover_process(events),
        "variants": variant_report(events),
        "bottlenecks": bottleneck_report(events),
        "claim_state": "smoke_only",
    }
    atomic_json(output / "process_report.json", payload)
    (output / "process_report.html").write_text(
        process_report_html(
            payload,
            title="Container logistics process report",
            watermark="SMOKE_ONLY",
        ),
        encoding="utf-8",
    )
    run.finish(
        {
            "dataset_id": payload["manifest"]["dataset_id"],
            "split_protocol": payload["split_manifest"]["protocol"],
            "test_influenced_choice": False,
        }
    )
    console.print(
        f"[green]Process report complete[/green] — "
        f"{payload['discovery']['events']} events, "
        f"{payload['variants']['variant_count']} variants"
    )


@app.command("verify-data")
def verify_data(
    manifest: Path,
    repository_root: Annotated[Path, typer.Option("--repository-root")] = Path("."),
) -> None:
    loaded = load_manifest(manifest)
    results = verify_manifest_files(loaded, repository_root)
    table = Table("File", "Exists", "Size", "SHA-256")
    for result in results:
        table.add_row(
            result.path,
            "yes" if result.exists else "no",
            "ok" if result.size_matches else "mismatch",
            "ok" if result.sha256_matches else "mismatch",
        )
    console.print(table)
    if not all(item.exists and item.size_matches and item.sha256_matches for item in results):
        raise typer.Exit(code=1)


@app.command("train-baselines")
def train_baselines(
    source: Path,
    config: Annotated[Path, typer.Option("--config", "-c")] = Path(
        "configs/experiment/warehouse_smoke.yaml"
    ),
    output: Annotated[Path, typer.Option("--output", "-o")] = Path("outputs/warehouse_smoke_v2"),
) -> None:
    metrics = train_warehouse_baselines(source, config, output)
    selected = metrics["model_selection"]["selected_model"]
    result = metrics["selected_model_test"]
    console.print(
        f"[green]Training complete[/green] — selected={selected}; "
        f"test MAE={result['mae_minutes']:.2f} min; "
        "[yellow]claim_state=smoke_only, public non-port data[/yellow]"
    )


@app.command("build-report")
def build_report(
    metrics_path: Path,
    output: Annotated[Path, typer.Option("--output", "-o")] = Path("outputs/report.pdf"),
) -> None:
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    write_evidence_pdf(output, "Kaleido FlowTwin evidence report", metrics)
    console.print(f"[green]Report written[/green] to {output}")


@app.command("train-sequence")
def train_sequence(
    source: Path,
    baseline_run: Annotated[Path, typer.Option("--baseline-run")] = Path(
        "outputs/warehouse_smoke_v2"
    ),
    config: Annotated[Path, typer.Option("--config", "-c")] = Path(
        "configs/experiment/warehouse_sequence_smoke.yaml"
    ),
    output: Annotated[Path, typer.Option("--output", "-o")] = Path(
        "outputs/warehouse_sequence_smoke_v2"
    ),
) -> None:
    metrics = train_sequence_models(source, baseline_run, config, output)
    best = metrics["best_architecture"]
    aggregate = metrics["aggregates"][best]
    console.print(
        f"[green]Sequential training complete[/green] — best={best}; "
        f"mean test MAE={aggregate['mae_mean_minutes']:.2f} min across "
        f"{metrics['number_of_seeds']} seeds; "
        "[yellow]claim_state=smoke_only[/yellow]"
    )


@app.command("serve")
def serve(
    host: str = "127.0.0.1",
    port: int = 8000,
    artifact_root: Path = Path("outputs"),
) -> None:
    console.print(
        f"Serving read-only dashboard on http://{host}:{port} "
        "[yellow](synthetic values remain watermarked)[/yellow]"
    )
    uvicorn.run(create_app(artifact_root), host=host, port=port)


if __name__ == "__main__":
    app()
