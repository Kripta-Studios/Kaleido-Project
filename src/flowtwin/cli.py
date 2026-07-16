from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated, Any

import typer
import uvicorn
from rich.console import Console
from rich.table import Table

from flowtwin.action_world_training import train_action_event_jepa
from flowtwin.benchmarks.ais_eta import run_ais_eta_benchmark
from flowtwin.benchmarks.ais_world_model import run_ais_world_model_benchmark
from flowtwin.benchmarks.lade_dispatch import run_lade_dispatch_benchmark
from flowtwin.benchmarks.lade_modalities import run_lade_modality_benchmark
from flowtwin.benchmarks.lade_regularizers import run_lade_regularizer_benchmark
from flowtwin.benchmarks.ocel_logistics import run_ocel_logistics_benchmark
from flowtwin.data.adapters.ocel import OcelSQLiteAdapter
from flowtwin.data.adapters.trace_port import TracePortAdapter
from flowtwin.data.manifests import load_manifest, verify_manifest_files
from flowtwin.data.synthetic import generate_trace_port_fixture
from flowtwin.event_jepa_ablations import train_event_jepa_ablations
from flowtwin.event_jepa_training import train_event_jepa
from flowtwin.final_package import build_final_package, finalize_final_package
from flowtwin.jepa_hybrid_training import train_jepa_hybrid_boosting
from flowtwin.process.bottlenecks import bottleneck_report
from flowtwin.process.discovery import discover_process
from flowtwin.process.variants import variant_report
from flowtwin.provenance import RunContext, atomic_json
from flowtwin.reporting import process_report_html, write_evidence_pdf
from flowtwin.sequence_training import train_sequence_models
from flowtwin.serving.api import create_app
from flowtwin.simulation.synthetic_actions import (
    generate_synthetic_action_overlay,
    train_synthetic_action_benchmark,
)
from flowtwin.temporal_t_jepa_training import train_temporal_t_jepa
from flowtwin.training import train_warehouse_baselines
from flowtwin.var_event_jepa_training import train_var_event_jepa

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


@app.command("benchmark-ocel-logistics")
def benchmark_ocel_logistics(
    source: Path,
    config: Annotated[Path, typer.Option("--config", "-c")] = Path(
        "configs/experiment/ocel_logistics_graph_smoke.yaml"
    ),
    output: Annotated[Path, typer.Option("--output", "-o")] = Path(
        "outputs/ocel_logistics_graph_v1"
    ),
) -> None:
    metrics = run_ocel_logistics_benchmark(source, config, output)
    selected = metrics["selected_model_validation_only"]
    result = metrics["selected_test"]
    console.print(
        f"[green]OCEL logistics benchmark complete[/green] — selected={selected}; "
        f"test MAE={result['mae']:.2f} h; "
        "[yellow]simulated public data, smoke_only[/yellow]"
    )


@app.command("benchmark-ais-eta")
def benchmark_ais_eta(
    source: Path,
    config: Annotated[Path, typer.Option("--config", "-c")] = Path(
        "configs/experiment/noaa_ais_eta_smoke.yaml"
    ),
    output: Annotated[Path, typer.Option("--output", "-o")] = Path(
        "outputs/noaa_ais_eta_v3"
    ),
) -> None:
    metrics = run_ais_eta_benchmark(source, config, output)
    selected = metrics["selected_model_validation_only"]
    result = metrics["selected_test"]
    console.print(
        f"[green]NOAA AIS ETA benchmark complete[/green] — selected={selected}; "
        f"test MAE={result['mae']:.2f} h; "
        f"within +/-1 h={result['within_tolerance']['within_1']:.1%}; "
        "[yellow]public US AIS, smoke_only[/yellow]"
    )


@app.command("benchmark-lade-dispatch")
def benchmark_lade_dispatch(
    source: Path,
    config: Annotated[Path, typer.Option("--config", "-c")] = Path(
        "configs/experiment/lade_dispatch_world_jepa_smoke.yaml"
    ),
    output: Annotated[Path, typer.Option("--output", "-o")] = Path(
        "outputs/lade_dispatch_world_jepa_v3_fifo"
    ),
) -> None:
    metrics = run_lade_dispatch_benchmark(source, config, output)
    selected = metrics["selected_model"]
    result = metrics["selected_test"]
    gate = metrics["promotion_gate"]
    console.print(
        f"[green]LaDe dispatch benchmark complete[/green] — selected={selected}; "
        f"test MAE={result['mae_minutes']:.2f} min; "
        f"public world-model gate={gate['passed']}; "
        f"[yellow]public last-mile proxy, {metrics['claim_state']}[/yellow]"
    )


@app.command("benchmark-ais-world-model")
def benchmark_ais_world_model(
    prefixes: Path = Path("outputs/noaa_ais_eta_v3/prefixes.parquet"),
    config: Annotated[Path, typer.Option("--config", "-c")] = Path(
        "configs/experiment/noaa_ais_phys_jepa_development.yaml"
    ),
    output: Annotated[Path, typer.Option("--output", "-o")] = Path(
        "outputs/noaa_ais_phys_jepa_development_v1"
    ),
) -> None:
    metrics = run_ais_world_model_benchmark(prefixes, config, output)
    selected = metrics["selected_model"]
    result = metrics["selected_test"]
    product = metrics["product_candidate"]
    exposure = (
        "previously opened development period only"
        if metrics["test_influenced_choice"]
        else "frozen future holdout opened once"
    )
    console.print(
        "[green]AIS world-model benchmark complete[/green] — "
        f"selected={selected}; distance MAE={result['distance_mae_km']:.3f} km; "
        f"deviation AUPRC={result['deviation_auprc']:.3f}; "
        f"hybrid={product['model']}; gate={product['gate']['passed']}; "
        f"[yellow]{exposure}; {metrics['claim_state']}[/yellow]"
    )


@app.command("benchmark-lade-regularizers")
def benchmark_lade_regularizers(
    source: Path,
    config: Annotated[Path, typer.Option("--config", "-c")] = Path(
        "configs/experiment/lade_dispatch_regularizers_diagnostic.yaml"
    ),
    output: Annotated[Path, typer.Option("--output", "-o")] = Path(
        "outputs/lade_dispatch_regularizers_v3_fifo"
    ),
) -> None:
    metrics = run_lade_regularizer_benchmark(source, config, output)
    selected = metrics["selected_regularizer"]
    summary = metrics["regularizers"][selected]
    console.print(
        f"[green]LaDe anti-collapse diagnostic complete[/green] — "
        f"selected={selected}; "
        f"test action delta={summary['mean_test_improvement_vs_shuffled_percent']:.2f}%; "
        "[yellow]test previously opened, diagnostic only[/yellow]"
    )


@app.command("benchmark-lade-modalities")
def benchmark_lade_modalities(
    source: Path,
    config: Annotated[Path, typer.Option("--config", "-c")] = Path(
        "configs/experiment/lade_dispatch_modalities_diagnostic.yaml"
    ),
    output: Annotated[Path, typer.Option("--output", "-o")] = Path(
        "outputs/lade_dispatch_modalities_v1_fifo"
    ),
) -> None:
    metrics = run_lade_modality_benchmark(source, config, output)
    full = metrics["modalities"]["full"]
    no_coordinates = metrics["modalities"]["no_continuous_coordinates"]
    console.print(
        "[green]LaDe modality diagnostic complete[/green] — "
        f"full raw MAE={full['raw_boosting']['test']['mae_minutes']:.2f} min; "
        "no-coordinate raw MAE="
        f"{no_coordinates['raw_boosting']['test']['mae_minutes']:.2f} min; "
        "[yellow]test previously opened, diagnostic only[/yellow]"
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
        "outputs/warehouse_sequence_smoke_v4"
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


@app.command("train-event-jepa")
def train_event_jepa_command(
    source: Path,
    baseline_run: Annotated[Path, typer.Option("--baseline-run")] = Path(
        "outputs/warehouse_smoke_v2"
    ),
    sequence_run: Annotated[Path, typer.Option("--sequence-run")] = Path(
        "outputs/warehouse_sequence_smoke_v4"
    ),
    config: Annotated[Path, typer.Option("--config", "-c")] = Path(
        "configs/experiment/warehouse_event_jepa_smoke.yaml"
    ),
    output: Annotated[Path, typer.Option("--output", "-o")] = Path(
        "outputs/warehouse_event_jepa_smoke_v2"
    ),
) -> None:
    metrics = train_event_jepa(source, baseline_run, sequence_run, config, output)
    selected = metrics["selected_variant_validation_only"]
    aggregate = metrics["aggregates"][selected]
    console.print(
        f"[green]Event-JEPA training complete[/green] — variant={selected}; "
        f"mean test MAE={aggregate['mae_mean_minutes']:.2f} min across "
        f"{metrics['number_of_seeds']} seeds; "
        "[yellow]action_free=true, world_model_claim=false, smoke_only[/yellow]"
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


@app.command("train-temporal-t-jepa")
def train_temporal_t_jepa_command(
    source: Path,
    baseline_run: Annotated[Path, typer.Option("--baseline-run")] = Path(
        "outputs/warehouse_smoke_v2"
    ),
    sequence_run: Annotated[Path, typer.Option("--sequence-run")] = Path(
        "outputs/warehouse_sequence_smoke_v4"
    ),
    current_jepa_run: Annotated[Path, typer.Option("--current-jepa-run")] = Path(
        "outputs/warehouse_event_jepa_smoke_v2"
    ),
    config: Annotated[Path, typer.Option("--config", "-c")] = Path(
        "configs/experiment/warehouse_temporal_t_jepa_smoke.yaml"
    ),
    output: Annotated[Path, typer.Option("--output", "-o")] = Path(
        "outputs/warehouse_temporal_t_jepa_v1"
    ),
) -> None:
    metrics = train_temporal_t_jepa(
        source,
        baseline_run,
        sequence_run,
        current_jepa_run,
        config,
        output,
    )
    selected = metrics["selected_main_validation_only"]
    aggregate = metrics["aggregates"][selected]
    console.print(
        f"[green]Temporal T-JEPA complete[/green] — regularizer={selected}; "
        f"mean test MAE={aggregate['mae_mean_minutes']:.2f} min across "
        f"{metrics['number_of_seeds']} seeds; "
        "[yellow]disjoint future + EMA teacher, smoke_only[/yellow]"
    )


@app.command("train-var-event-jepa")
def train_var_event_jepa_command(
    source: Path,
    baseline_run: Annotated[Path, typer.Option("--baseline-run")] = Path(
        "outputs/warehouse_smoke_v2"
    ),
    temporal_t_jepa_run: Annotated[
        Path, typer.Option("--temporal-t-jepa-run")
    ] = Path("outputs/warehouse_temporal_t_jepa_v1"),
    config: Annotated[Path, typer.Option("--config", "-c")] = Path(
        "configs/experiment/warehouse_var_event_jepa_smoke.yaml"
    ),
    output: Annotated[Path, typer.Option("--output", "-o")] = Path(
        "outputs/warehouse_var_event_jepa_v1"
    ),
) -> None:
    metrics = train_var_event_jepa(
        source,
        baseline_run,
        temporal_t_jepa_run,
        config,
        output,
    )
    aggregate = metrics["aggregate"]
    console.print(
        "[green]Var-Event-JEPA complete[/green] — "
        f"mean test MAE={aggregate['mae_mean_minutes']:.2f} min across "
        f"{metrics['number_of_seeds']} seeds; "
        "[yellow]variational diagnostic, smoke_only[/yellow]"
    )


@app.command("train-jepa-hybrid")
def train_jepa_hybrid_command(
    baseline_run: Annotated[Path, typer.Option("--baseline-run")] = Path(
        "outputs/warehouse_smoke_v2"
    ),
    temporal_t_jepa_run: Annotated[
        Path, typer.Option("--temporal-t-jepa-run")
    ] = Path("outputs/warehouse_temporal_t_jepa_v1"),
    var_event_jepa_run: Annotated[
        Path, typer.Option("--var-event-jepa-run")
    ] = Path("outputs/warehouse_var_event_jepa_v1"),
    config: Annotated[Path, typer.Option("--config", "-c")] = Path(
        "configs/experiment/warehouse_jepa_hybrid_smoke.yaml"
    ),
    output: Annotated[Path, typer.Option("--output", "-o")] = Path(
        "outputs/warehouse_jepa_hybrid_v1"
    ),
) -> None:
    metrics = train_jepa_hybrid_boosting(
        baseline_run,
        temporal_t_jepa_run,
        var_event_jepa_run,
        config,
        output,
    )
    selected = metrics["selected_overall_validation_only"]
    aggregate = metrics["aggregates"][selected]
    console.print(
        f"[green]JEPA hybrid boosting complete[/green] — selected={selected}; "
        f"mean test MAE={aggregate['mae_mean_minutes']:.2f} min across "
        f"{metrics['number_of_seeds']} seeds; "
        "[yellow]validation-only selection, smoke_only[/yellow]"
    )


@app.command("train-event-jepa-ablations")
def train_event_jepa_ablations_command(
    source: Path,
    baseline_run: Annotated[Path, typer.Option("--baseline-run")] = Path(
        "outputs/warehouse_smoke_v2"
    ),
    main_jepa_run: Annotated[Path, typer.Option("--main-jepa-run")] = Path(
        "outputs/warehouse_event_jepa_smoke_v2"
    ),
    config: Annotated[Path, typer.Option("--config", "-c")] = Path(
        "configs/experiment/warehouse_event_jepa_ablations_smoke.yaml"
    ),
    output: Annotated[Path, typer.Option("--output", "-o")] = Path(
        "outputs/warehouse_event_jepa_ablations_v1"
    ),
) -> None:
    metrics = train_event_jepa_ablations(
        source,
        baseline_run,
        main_jepa_run,
        config,
        output,
    )
    console.print(
        f"[green]Event-JEPA ablations complete[/green] — "
        f"{len(metrics['fixed_ablations'])} variants across "
        f"{metrics['number_of_seeds']} seeds; "
        "[yellow]action_free=true, smoke_only[/yellow]"
    )


@app.command("train-action-event-jepa")
def train_action_event_jepa_command(
    source: Path,
    baseline_run: Annotated[Path, typer.Option("--baseline-run")] = Path(
        "outputs/warehouse_smoke_v2"
    ),
    overlay: Annotated[Path, typer.Option("--overlay")] = Path(
        "outputs/warehouse_synthetic_actions_v1/overlay.parquet"
    ),
    config: Annotated[Path, typer.Option("--config", "-c")] = Path(
        "configs/experiment/warehouse_action_event_jepa_smoke.yaml"
    ),
    output: Annotated[Path, typer.Option("--output", "-o")] = Path(
        "outputs/warehouse_action_event_jepa_v1"
    ),
) -> None:
    metrics = train_action_event_jepa(source, baseline_run, overlay, config, output)
    gate = metrics["correct_actions_beat_shuffled_each_seed"]
    console.print(
        f"[green]Synthetic action Event-JEPA complete[/green] — "
        f"correct actions beat shuffled in every seed={gate}; "
        "[yellow]injected transition only, no Kaleido action claim[/yellow]"
    )


@app.command("generate-synthetic-actions")
def generate_synthetic_actions(
    prefixes: Path = Path("outputs/warehouse_smoke_v2/prefixes.parquet"),
    output: Annotated[Path, typer.Option("--output", "-o")] = Path(
        "outputs/warehouse_synthetic_actions_v1/overlay.parquet"
    ),
    seed: int = 42,
) -> None:
    summary = generate_synthetic_action_overlay(prefixes, output, seed=seed)
    console.print(
        f"[green]Synthetic action overlay written[/green] — rows={summary.rows}; "
        f"operations={summary.operations}; [yellow]{summary.evidence_type}[/yellow]"
    )


@app.command("train-synthetic-actions")
def train_synthetic_actions(
    overlay: Path = Path("outputs/warehouse_synthetic_actions_v1/overlay.parquet"),
    config: Annotated[Path, typer.Option("--config", "-c")] = Path(
        "configs/experiment/warehouse_synthetic_actions_smoke.yaml"
    ),
    output: Annotated[Path, typer.Option("--output", "-o")] = Path(
        "outputs/warehouse_synthetic_actions_v1"
    ),
) -> None:
    metrics = train_synthetic_action_benchmark(overlay, config, output)
    improvement = metrics["mean_mae_improvement_vs_shuffled_minutes"]
    console.print(
        f"[green]Synthetic action benchmark complete[/green] — "
        f"correct-vs-shuffled MAE improvement={improvement:.2f} min; "
        "[yellow]injected signal only, no source/Kaleido action claim[/yellow]"
    )


@app.command("build-final-package")
def build_final_package_command(
    repository_root: Annotated[Path, typer.Option("--repository-root")] = Path("."),
) -> None:
    summary = build_final_package(repository_root)
    console.print(
        "[green]Final evidence package generated[/green] — "
        f"claim_state={summary['claim_state']}; "
        f"verified_runs={len(summary['provenance'])}; "
        "[yellow]public/synthetic evidence only[/yellow]"
    )


@app.command("finalize-final-package")
def finalize_final_package_command(
    repository_root: Annotated[Path, typer.Option("--repository-root")] = Path("."),
) -> None:
    summary = finalize_final_package(repository_root)
    console.print(
        "[green]Compiled final package verified and hashed[/green] — "
        f"claim_state={summary['claim_state']}; "
        "[yellow]PDFs are current with generated TeX[/yellow]"
    )


if __name__ == "__main__":
    app()
