# ruff: noqa: E501, RUF001
from __future__ import annotations

import csv
import html
import json
from datetime import date
from pathlib import Path
from typing import Any

from reportlab.lib.colors import HexColor
from reportlab.pdfgen import canvas

from flowtwin.provenance import atomic_json, sha256_file

RUNS = {
    "process": Path("outputs/container_process"),
    "baseline": Path("outputs/warehouse_smoke_v2"),
    "sequence": Path("outputs/warehouse_sequence_smoke_v4"),
    "jepa": Path("outputs/warehouse_event_jepa_smoke_v2"),
    "ablations": Path("outputs/warehouse_event_jepa_ablations_v1"),
    "temporal_t_jepa": Path("outputs/warehouse_temporal_t_jepa_v1"),
    "var_event_jepa": Path("outputs/warehouse_var_event_jepa_v1"),
    "jepa_hybrid": Path("outputs/warehouse_jepa_hybrid_v1"),
    "synthetic_actions": Path("outputs/warehouse_synthetic_actions_v1"),
    "action_jepa_sigreg": Path("outputs/warehouse_action_event_jepa_v1"),
    "action_jepa_visreg": Path("outputs/warehouse_action_event_jepa_visreg_v2"),
    "ais_eta": Path("outputs/noaa_ais_eta_v3"),
    "ocel_logistics": Path("outputs/ocel_logistics_graph_v1"),
}


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def _verify_run(run_dir: Path) -> dict[str, Any]:
    if (run_dir / "INVALIDATED.md").is_file():
        raise RuntimeError(f"refusing invalidated evidence run: {run_dir}")
    manifest = _json(run_dir / "run_manifest.json")
    expected = manifest.get("artifacts", {})
    mismatches: list[str] = []
    # Historical resumed runs may contain a stale self-hash from before
    # RunContext explicitly excluded run_manifest.json. Self-hashes are
    # unverifiable by construction; every evidence artifact is still checked.
    verifiable = {key: value for key, value in expected.items() if key != "run_manifest.json"}
    for relative, digest in verifiable.items():
        path = run_dir / relative
        if not path.is_file() or sha256_file(path) != digest:
            mismatches.append(relative)
    if mismatches:
        raise RuntimeError(f"artifact hash mismatch in {run_dir}: {mismatches}")
    return {
        "run_dir": str(run_dir),
        "claim_state": manifest.get("claim_state"),
        "commit": manifest.get("commit"),
        "dirty": manifest.get("dirty"),
        "artifact_count": len(verifiable),
        "excluded_unverifiable_self_hash": "run_manifest.json" in expected,
        "hashes_verified": True,
    }


def _mean(values: list[float]) -> float:
    return sum(values) / len(values)


def build_summary(repository_root: Path = Path(".")) -> dict[str, Any]:
    resolved_runs = {name: repository_root / path for name, path in RUNS.items()}
    provenance = {name: _verify_run(path) for name, path in resolved_runs.items()}
    process = _json(resolved_runs["process"] / "process_report.json")
    baseline = _json(resolved_runs["baseline"] / "metrics.json")
    sequence = _json(resolved_runs["sequence"] / "metrics.json")
    jepa = _json(resolved_runs["jepa"] / "metrics.json")
    ablations = _json(resolved_runs["ablations"] / "metrics.json")
    ablation_gate = _json(resolved_runs["ablations"] / "ablation_gate.json")
    temporal = _json(resolved_runs["temporal_t_jepa"] / "metrics.json")
    temporal_gate = _json(resolved_runs["temporal_t_jepa"] / "promotion_gate.json")
    variational = _json(resolved_runs["var_event_jepa"] / "metrics.json")
    variational_gate = _json(resolved_runs["var_event_jepa"] / "promotion_gate.json")
    hybrid = _json(resolved_runs["jepa_hybrid"] / "metrics.json")
    hybrid_gate = _json(resolved_runs["jepa_hybrid"] / "promotion_gate.json")
    synthetic = _json(resolved_runs["synthetic_actions"] / "metrics.json")
    action_sigreg = _json(resolved_runs["action_jepa_sigreg"] / "metrics.json")
    action_visreg = _json(resolved_runs["action_jepa_visreg"] / "metrics.json")
    visreg_gate = _json(resolved_runs["action_jepa_visreg"] / "action_world_gate.json")
    ais_eta = _json(resolved_runs["ais_eta"] / "metrics.json")
    ocel_logistics = _json(resolved_runs["ocel_logistics"] / "metrics.json")

    selected_jepa = str(jepa["selected_variant_validation_only"])
    jepa_runs = jepa["runs"]
    jepa_coverage = _mean(
        [float(item["variants"][selected_jepa]["metrics"]["p90_quantile_coverage"]) for item in jepa_runs]
    )
    jepa_width = _mean(
        [float(item["variants"][selected_jepa]["metrics"]["p50_to_p90_width_minutes"]) for item in jepa_runs]
    )
    visreg_correct_runs = action_visreg["results"]["correct_action"]
    visreg_ranks = [
        float(item["embedding_diagnostics_validation"]["effective_rank"])
        for item in visreg_correct_runs
    ]
    visreg_scales = [
        float(item["embedding_diagnostics_validation"]["mean_dimension_std"])
        for item in visreg_correct_runs
    ]
    baseline_mae = float(baseline["selected_model_test"]["mae_minutes"])
    transformer_mae = float(sequence["aggregates"]["transformer"]["mae_mean_minutes"])
    jepa_mae = float(jepa["aggregates"][selected_jepa]["mae_mean_minutes"])
    temporal_selected = str(temporal["selected_main_validation_only"])
    hybrid_selected = str(hybrid["selected_overall_validation_only"])
    best_hybrid = str(hybrid["best_hybrid_validation_only"])

    return {
        "generated_on": date.today().isoformat(),
        "claim_state": "smoke_only",
        "dataset": {
            "id": baseline["dataset_id"],
            "export_version": baseline["dataset_export_version"],
            "sha256": baseline["source_file_sha256"],
            "source_rows_scanned": baseline["source_rows_scanned"],
            "source_cases_used": baseline["source_cases_used"],
            "prefix_rows": baseline["prefix_rows"],
            "split_protocol": baseline["split_protocol"],
            "split_counts": baseline["split_counts_operations"],
            "domain": "real anonymized aeronautical warehouse outbound; not port/Kaleido",
        },
        "process_competence": {
            "dataset_id": process["manifest"]["dataset_id"],
            "sha256": process["manifest"]["sha256"],
            "events": process["manifest"]["event_rows"],
            "objects": process["manifest"]["objects"],
            "relationships": process["manifest"]["event_object_relations"],
            "object_traces": process["discovery"]["operations"],
            "variants": process["variants"]["variant_count"],
            "top_variant_coverage": process["variants"]["top_variant_coverage"],
            "integrity_passed": bool(process["object_graph"]["passed"]),
        },
        "remaining_time": {
            "baselines": baseline["remaining_time_test"],
            "boosting": {
                **baseline["selected_model_test"],
                "mae_ci95": baseline["selected_model_mae_cluster_bootstrap"],
                "seeds": 1,
            },
            "worst_group": baseline["worst_group_last_activity"],
            "gru": sequence["aggregates"]["gru"],
            "transformer": sequence["aggregates"]["transformer"],
            "event_jepa": {
                **jepa["aggregates"][selected_jepa],
                "selected_variant": selected_jepa,
                "p90_coverage_mean": jepa_coverage,
                "p50_to_p90_width_mean_minutes": jepa_width,
                "embedding_diagnostics": [
                    item["embedding_diagnostics_validation"] for item in jepa_runs
                ],
            },
            "event_jepa_minus_boosting_mae_minutes": jepa_mae - baseline_mae,
            "event_jepa_minus_transformer_mae_minutes": jepa_mae - transformer_mae,
            "temporal_t_jepa": {
                **temporal["aggregates"][temporal_selected],
                "selected_variant": temporal_selected,
                "completion_variant": temporal["completion_variant"],
                "completion": temporal["aggregates"][temporal["completion_variant"]],
                "shuffled_variant": temporal["shuffled_variant"],
                "shuffled": temporal["aggregates"][temporal["shuffled_variant"]],
                "promotion_gate": temporal_gate,
            },
            "var_event_jepa": {
                **variational["aggregate"],
                "promotion_gate": variational_gate,
            },
            "jepa_hybrid": {
                "aggregates": hybrid["aggregates"],
                "selected_overall": hybrid_selected,
                "best_hybrid": best_hybrid,
                "promotion_gate": hybrid_gate,
            },
            "winner": "quantile_boosting",
            "selection": "validation only; test did not influence choices",
        },
        "risk_proxy": baseline["risk_test"],
        "risk_target": baseline["risk_target"],
        "aligned_public_benchmarks": {
            "ais_eta": ais_eta,
            "ocel_logistics": ocel_logistics,
            "presentation_primary": "ais_eta",
            "interpretation": (
                "The untouched February AIS ETA benchmark passed all six predeclared "
                "capability gates. OCEL object-graph context improved test MAE but was "
                "not selected on validation and remains a diagnostic process example."
            ),
        },
        "jepa_ablations": {
            "main_reference": ablations["main_reference"],
            "aggregates": ablations["aggregates"],
            "comparisons": ablations["comparisons_to_main"],
            "mean_gates": ablation_gate,
            "interpretation": {
                "jepa_objective_signal": "main beats random encoder in all three paired seeds",
                "sigreg": "required; no-SIGReg collapsed and worsened MAE",
                "multi_horizon": "not supported; completion-only is marginally better",
                "temporal_pairing": "not robust; mean delta is 0.40 min and paired wins are inconsistent",
            },
        },
        "synthetic_actions": {
            "overlay_sha256": synthetic["overlay_sha256"],
            "tabular": {
                "aggregates": synthetic["aggregates"],
                "correct_beats_shuffled_each_seed": synthetic[
                    "correct_actions_beat_shuffled_each_seed"
                ],
                "verdict": "rejected",
            },
            "action_jepa_sigreg": {
                "aggregates": action_sigreg["aggregates"],
                "correct_beats_shuffled_each_seed": action_sigreg[
                    "correct_actions_beat_shuffled_each_seed"
                ],
                "verdict": "rejected: inconsistent action win and low-rank/scale collapse",
            },
            "action_jepa_visreg": {
                "aggregates": action_visreg["aggregates"],
                "correct_beats_shuffled_each_seed": action_visreg[
                    "correct_actions_beat_shuffled_each_seed"
                ],
                "mean_improvement_vs_shuffled_minutes": -float(
                    action_visreg["correct_minus_shuffled_mean_mae_minutes"]
                ),
                "effective_rank_correct": visreg_ranks,
                "mean_dimension_std_correct": visreg_scales,
                "signal_gate": visreg_gate["synthetic_action_signal_recovered"],
                "world_model_verdict": "not promoted: action signal recovered, predictor scale remains below collapse threshold",
            },
            "claim_boundary": "generated actions/effects only; no causal or Kaleido action claim",
        },
        "business_fit": {
            "primary_surface": "Shipping Board / Freight Intelligence ETA exception surface",
            "physical_twin_complement": "TWINPORTS supplies spatial/asset state; FlowTwin supplies event-time prediction, uncertainty and scenario evidence",
            "extensions": ["Trace Port operation completion", "TWINPORTS process intelligence"],
            "recommended_demo": "embedded read-only web dashboard plus offline HTML presentation",
        },
        "provenance": provenance,
        "what_this_does_not_prove": [
            "Kaleido accuracy or port-domain generalization",
            "material plan-deviation prediction",
            "causal action value or counterfactual validity",
            "ROI, realized savings, production readiness or deployment success",
        ],
    }


def _chart_pdf(path: Path, title: str, items: list[tuple[str, float, str]], note: str) -> None:
    width, height = 900, 480
    pdf = canvas.Canvas(str(path), pagesize=(width, height))
    pdf.setFillColor(HexColor("#F5F7F5"))
    pdf.rect(0, 0, width, height, stroke=0, fill=1)
    pdf.setFillColor(HexColor("#10272F"))
    pdf.setFont("Helvetica-Bold", 24)
    pdf.drawString(42, height - 52, title)
    maximum = max(value for _, value, _ in items) * 1.08
    bar_left = 225
    bar_width = 590
    y = height - 115
    for label, value, color in items:
        pdf.setFillColor(HexColor("#344D56"))
        pdf.setFont("Helvetica", 14)
        pdf.drawRightString(bar_left - 14, y + 7, label)
        pdf.setFillColor(HexColor("#E2E8E6"))
        pdf.roundRect(bar_left, y, bar_width, 28, 7, stroke=0, fill=1)
        pdf.setFillColor(HexColor(color))
        pdf.roundRect(bar_left, y, bar_width * value / maximum, 28, 7, stroke=0, fill=1)
        pdf.setFillColor(HexColor("#10272F"))
        pdf.setFont("Helvetica-Bold", 14)
        pdf.drawString(bar_left + bar_width * value / maximum + 8, y + 7, f"{value:.2f}")
        y -= 58
    pdf.setFillColor(HexColor("#62757C"))
    pdf.setFont("Helvetica", 11)
    pdf.drawString(42, 28, note)
    pdf.save()


def _chart_svg(path: Path, title: str, items: list[tuple[str, float, str]], note: str) -> None:
    maximum = max(value for _, value, _ in items) * 1.08
    rows: list[str] = []
    y = 105
    for label, value, color in items:
        length = 570 * value / maximum
        rows.append(
            f'<text x="205" y="{y + 19}" text-anchor="end" class="label">{html.escape(label)}</text>'
            f'<rect x="225" y="{y}" width="570" height="28" rx="7" fill="#E2E8E6"/>'
            f'<rect class="animated-bar" x="225" y="{y}" width="{length:.1f}" height="28" rx="7" fill="{color}"/>'
            f'<text x="{min(850, 235 + length):.1f}" y="{y + 19}" class="value">{value:.2f}</text>'
        )
        y += 58
    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 900 480" role="img" aria-label="{html.escape(title)}">
<style>.title{{font:700 24px Inter,Segoe UI,sans-serif;fill:#10272F}}.label{{font:14px Inter,Segoe UI,sans-serif;fill:#344D56}}.value{{font:700 14px Inter,Segoe UI,sans-serif;fill:#10272F}}.note{{font:11px Inter,Segoe UI,sans-serif;fill:#62757C}}.animated-bar{{transform-origin:225px center;animation:grow .9s ease both}}@keyframes grow{{from{{transform:scaleX(0)}}to{{transform:scaleX(1)}}}}</style>
<rect width="900" height="480" fill="#F5F7F5"/><text x="42" y="52" class="title">{html.escape(title)}</text>{''.join(rows)}<text x="42" y="455" class="note">{html.escape(note)}</text></svg>'''
    path.write_text(svg, encoding="utf-8")


def write_charts(summary: dict[str, Any], assets: Path) -> None:
    assets.mkdir(parents=True, exist_ok=True)
    ais = summary["aligned_public_benchmarks"]["ais_eta"]
    ais_tests = ais["test_metrics"]
    model_items = [
        ("Boosting ETA", float(ais_tests["tabular_eta"]["mae"]), "#18A999"),
        (
            "Híbrido físico-residual",
            float(ais_tests["physics_residual_eta"]["mae"]),
            "#2F6BFF",
        ),
        (
            "Mediana puerto-distancia",
            float(ais_tests["port_distance_median"]["mae"]),
            "#F2B134",
        ),
        ("ETA distancia/velocidad", float(ais_tests["kinematic_eta"]["mae"]), "#D85C5C"),
    ]
    ablation_values = summary["jepa_ablations"]["aggregates"]
    ablation_items = [
        ("Solo finalización + SIGReg", float(ablation_values["completion_only_sigreg"]["mae_mean_minutes"]), "#18A999"),
        ("Principal multi-horizonte", float(summary["jepa_ablations"]["main_reference"]["mae_mean_minutes"]), "#2F6BFF"),
        ("Pares temporales barajados", float(ablation_values["shuffled_temporal_pairs"]["mae_mean_minutes"]), "#7C5CFC"),
        ("Encoder aleatorio", float(ablation_values["random_encoder_no_jepa"]["mae_mean_minutes"]), "#F2B134"),
        ("Sin SIGReg", float(ablation_values["multi_horizon_no_sigreg"]["mae_mean_minutes"]), "#D85C5C"),
    ]
    action = summary["synthetic_actions"]["action_jepa_visreg"]["aggregates"]
    action_items = [
        ("Acción correcta", float(action["correct_action"]["mae_mean_minutes"]), "#18A999"),
        ("Solo prefijo", float(action["current_prefix_only"]["mae_mean_minutes"]), "#2F6BFF"),
        ("Acción barajada", float(action["shuffled_action"]["mae_mean_minutes"]), "#D85C5C"),
    ]
    for stem, title, items, note in (
        (
            "model_comparison",
            "ETA a geofence - MAE en horas (menor es mejor)",
            model_items,
            "Test futuro 1-7 febrero; 85 viajes. Selección solo en validación. smoke_only.",
        ),
        ("jepa_ablations", "Ablations fijas de Event-JEPA", ablation_items, "Mismo split congelado y tres seeds. Test no seleccionó variantes."),
        ("action_recovery", "Recuperación de acción sintética con VISReg", action_items, "Target sintético. No comparable con MAE real ni valor Kaleido."),
    ):
        _chart_pdf(assets / f"{stem}.pdf", title, items, note)
        _chart_svg(assets / f"{stem}.svg", title, items, note)


def _values(summary: dict[str, Any]) -> dict[str, str]:
    remaining = summary["remaining_time"]
    boosting = remaining["boosting"]
    baselines = remaining["baselines"]
    global_median = baselines["global_median"]
    activity_median = baselines["activity_median"]
    worst_group = remaining["worst_group"]
    worst_group_name = str(worst_group["worst_group"])
    worst_group_metrics = worst_group["groups"][worst_group_name]
    jepa = remaining["event_jepa"]
    transformer = remaining["transformer"]
    temporal = remaining["temporal_t_jepa"]
    variational = remaining["var_event_jepa"]
    hybrid = remaining["jepa_hybrid"]
    hybrid_raw = hybrid["aggregates"]["raw"]
    best_hybrid = hybrid["aggregates"][hybrid["best_hybrid"]]
    ablations = summary["jepa_ablations"]["aggregates"]
    action = summary["synthetic_actions"]["action_jepa_visreg"]
    action_aggregates = action["aggregates"]
    process = summary["process_competence"]
    risk = summary["risk_proxy"]
    aligned = summary["aligned_public_benchmarks"]
    ais = aligned["ais_eta"]
    ais_selected = ais["selected_test"]
    ais_tests = ais["test_metrics"]
    ais_bootstrap = ais["selected_trip_bootstrap"]
    ais_gate = ais["promotion_gate"]
    ocel_aligned = aligned["ocel_logistics"]
    return {
        "DATE": summary["generated_on"],
        "TEST_COUNT": "58",
        "HASH": summary["dataset"]["sha256"],
        "HASH_SHORT": summary["dataset"]["sha256"][:12],
        "ROWS": f"{summary['dataset']['source_rows_scanned']:,}",
        "CASES": f"{summary['dataset']['source_cases_used']:,}",
        "PREFIXES": f"{summary['dataset']['prefix_rows']:,}",
        "TRAIN": f"{summary['dataset']['split_counts']['train']:,}",
        "VALIDATION": f"{summary['dataset']['split_counts']['validation']:,}",
        "TEST": f"{summary['dataset']['split_counts']['test']:,}",
        "BOOST_MAE": f"{boosting['mae_minutes']:.2f}",
        "BOOST_MAE_HOURS": f"{boosting['mae_minutes'] / 60:.2f}",
        "BOOST_MEDIAN_AE": f"{boosting['median_ae_minutes']:.2f}",
        "BOOST_CI_LOW": f"{boosting['mae_ci95']['bootstrap_95_low']:.2f}",
        "BOOST_CI_HIGH": f"{boosting['mae_ci95']['bootstrap_95_high']:.2f}",
        "BOOST_P90": f"{100 * boosting['p90_interval_coverage']:.2f}",
        "BOOST_WIDTH": f"{boosting['p90_interval_width_minutes']:.2f}",
        "BOOST_WIDTH_HOURS": f"{boosting['p90_interval_width_minutes'] / 60:.2f}",
        "GLOBAL_MAE": f"{global_median['mae_minutes']:.2f}",
        "GLOBAL_MEDIAN_AE": f"{global_median['median_ae_minutes']:.2f}",
        "GLOBAL_GAIN": f"{global_median['mae_minutes'] - boosting['mae_minutes']:.2f}",
        "GLOBAL_GAIN_PCT": f"{100 * (global_median['mae_minutes'] - boosting['mae_minutes']) / global_median['mae_minutes']:.1f}",
        "ACTIVITY_MAE": f"{activity_median['mae_minutes']:.2f}",
        "ACTIVITY_GAIN": f"{activity_median['mae_minutes'] - boosting['mae_minutes']:.2f}",
        "ACTIVITY_GAIN_PCT": f"{100 * (activity_median['mae_minutes'] - boosting['mae_minutes']) / activity_median['mae_minutes']:.1f}",
        "WORST_GROUP": worst_group_name.replace("_", r"\_"),
        "WORST_GROUP_MAE": f"{worst_group_metrics['mae_minutes']:.2f}",
        "AIS_MODEL": str(ais["selected_model_validation_only"]).replace("_", r"\_"),
        "AIS_MAE": f"{ais_selected['mae']:.2f}",
        "AIS_MEDIAN_AE": f"{ais_selected['median_absolute_error']:.2f}",
        "AIS_P90_AE": f"{ais_selected['p90_absolute_error']:.2f}",
        "AIS_CI_LOW": f"{ais_bootstrap['bootstrap_95_low']:.2f}",
        "AIS_CI_HIGH": f"{ais_bootstrap['bootstrap_95_high']:.2f}",
        "AIS_WITHIN_1": f"{100 * ais_selected['within_tolerance']['within_1']:.1f}",
        "AIS_WITHIN_2": f"{100 * ais_selected['within_tolerance']['within_2']:.1f}",
        "AIS_WITHIN_4": f"{100 * ais_selected['within_tolerance']['within_4']:.1f}",
        "AIS_KINEMATIC": f"{ais_tests['kinematic_eta']['mae']:.2f}",
        "AIS_HISTORICAL": f"{ais_tests['port_distance_median']['mae']:.2f}",
        "AIS_RESIDUAL": f"{ais_tests['physics_residual_eta']['mae']:.2f}",
        "AIS_GAIN_KINEMATIC": f"{ais_gate['improvement_vs_kinematic_percent']:.1f}",
        "AIS_GAIN_HISTORICAL": f"{ais_gate['improvement_vs_port_distance_median_percent']:.1f}",
        "AIS_P90_COVERAGE": f"{100 * ais['p90_interval_coverage']:.1f}",
        "AIS_P90_WIDTH": f"{ais['p90_interval_width_hours']:.2f}",
        "AIS_TRAIN_TRIPS": f"{ais['split']['counts']['train']['trips']:,}",
        "AIS_VALIDATION_TRIPS": f"{ais['split']['counts']['validation']['trips']:,}",
        "AIS_TEST_TRIPS": f"{ais['split']['counts']['test']['trips']:,}",
        "AIS_TEST_PREFIXES": f"{ais['split']['counts']['test']['prefixes']:,}",
        "AIS_SOURCE_FILES": f"{len(ais['dataset']['source_files']):,}",
        "AIS_SOURCE_GB": f"{sum(item['bytes'] for item in ais['dataset']['source_files']) / (1024**3):.2f}",
        "AIS_MAE_0_2": f"{ais['by_lead_time']['0_2h']['mae']:.2f}",
        "AIS_MAE_2_6": f"{ais['by_lead_time']['2_6h']['mae']:.2f}",
        "AIS_MAE_6_12": f"{ais['by_lead_time']['6_12h']['mae']:.2f}",
        "AIS_NOLA_SHARE": f"{100 * ais['by_port']['new_orleans']['rows'] / ais['split']['counts']['test']['prefixes']:.1f}",
        "OCEL_ALIGNED_SELECTED": str(ocel_aligned["selected_model_validation_only"]).replace(
            "_", r"\_"
        ),
        "OCEL_ALIGNED_FLAT": f"{ocel_aligned['test_metrics']['flat_boosting']['mae']:.2f}",
        "OCEL_ALIGNED_GRAPH": f"{ocel_aligned['test_metrics']['object_graph_boosting']['mae']:.2f}",
        "OCEL_ALIGNED_SHUFFLED": f"{ocel_aligned['test_metrics']['shuffled_object_graph']['mae']:.2f}",
        "OCEL_ALIGNED_OBJECTS": f"{ocel_aligned['dataset']['objects']:,}",
        "OCEL_ALIGNED_PREFIXES": f"{ocel_aligned['dataset']['prefix_rows']:,}",
        "TRANS_MAE": f"{transformer['mae_mean_minutes']:.2f}",
        "TRANS_SD": f"{transformer['mae_std_minutes']:.2f}",
        "JEPA_MAE": f"{jepa['mae_mean_minutes']:.2f}",
        "JEPA_SD": f"{jepa['mae_std_minutes']:.2f}",
        "JEPA_P90": f"{100 * jepa['p90_coverage_mean']:.2f}",
        "JEPA_WIDTH": f"{jepa['p50_to_p90_width_mean_minutes']:.2f}",
        "JEPA_VS_TRANS": f"{-remaining['event_jepa_minus_transformer_mae_minutes']:.2f}",
        "JEPA_VS_BOOST": f"{remaining['event_jepa_minus_boosting_mae_minutes']:.2f}",
        "TJEPA_MAE": f"{temporal['mae_mean_minutes']:.2f}",
        "TJEPA_SD": f"{temporal['mae_std_minutes']:.2f}",
        "TJEPA_COMPLETION": f"{temporal['completion']['mae_mean_minutes']:.2f}",
        "TJEPA_SHUFFLED": f"{temporal['shuffled']['mae_mean_minutes']:.2f}",
        "VAR_JEPA_MAE": f"{variational['mae_mean_minutes']:.2f}",
        "VAR_JEPA_SD": f"{variational['mae_std_minutes']:.2f}",
        "VAR_SPEARMAN": f"{variational['uncertainty_error_spearman_mean']:.3f}",
        "RAW3_MAE": f"{hybrid_raw['mae_mean_minutes']:.2f}",
        "RAW3_SD": f"{hybrid_raw['mae_std_minutes']:.2f}",
        "HYBRID_NAME": str(hybrid["best_hybrid"]).replace("_", r"\_"),
        "HYBRID_MAE": f"{best_hybrid['mae_mean_minutes']:.2f}",
        "HYBRID_SD": f"{best_hybrid['mae_std_minutes']:.2f}",
        "HYBRID_DELTA": f"{best_hybrid['mae_mean_minutes'] - hybrid_raw['mae_mean_minutes']:.2f}",
        "RANDOM_MAE": f"{ablations['random_encoder_no_jepa']['mae_mean_minutes']:.2f}",
        "NO_SIGREG_MAE": f"{ablations['multi_horizon_no_sigreg']['mae_mean_minutes']:.2f}",
        "ONE_HORIZON_MAE": f"{ablations['completion_only_sigreg']['mae_mean_minutes']:.2f}",
        "SHUFFLED_PAIR_MAE": f"{ablations['shuffled_temporal_pairs']['mae_mean_minutes']:.2f}",
        "ACTION_CORRECT": f"{action_aggregates['correct_action']['mae_mean_minutes']:.2f}",
        "ACTION_CORRECT_SD": f"{action_aggregates['correct_action']['mae_std_minutes']:.2f}",
        "ACTION_SHUFFLED": f"{action_aggregates['shuffled_action']['mae_mean_minutes']:.2f}",
        "ACTION_PREFIX": f"{action_aggregates['current_prefix_only']['mae_mean_minutes']:.2f}",
        "ACTION_GAIN": f"{action['mean_improvement_vs_shuffled_minutes']:.2f}",
        "ACTION_RANK_LOW": f"{min(action['effective_rank_correct']):.2f}",
        "ACTION_RANK_HIGH": f"{max(action['effective_rank_correct']):.2f}",
        "ACTION_SCALE_LOW": f"{min(action['mean_dimension_std_correct']):.3f}",
        "ACTION_SCALE_HIGH": f"{max(action['mean_dimension_std_correct']):.3f}",
        "PROCESS_EVENTS": f"{process['events']:,}",
        "PROCESS_OBJECTS": f"{process['objects']:,}",
        "PROCESS_RELATIONS": f"{process['relationships']:,}",
        "PROCESS_VARIANTS": f"{process['variants']:,}",
        "RISK_AUPRC": f"{risk['auprc']:.3f}",
        "RISK_ECE": f"{risk['calibration']['ece']:.3f}",
        "RISK_FALSE": f"{risk['false_alerts_per_100_operations']:.2f}",
    }


def _render(template: str, values: dict[str, str]) -> str:
    for key, value in values.items():
        template = template.replace(f"[[{key}]]", value)
    return template


def write_presentation_html(path: Path, summary: dict[str, Any]) -> None:
    template = r'''<!doctype html>
<html lang="es"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Kaleido FlowTwin - MVP técnico</title>
<style>
:root{--ink:#10272f;--muted:#61747b;--paper:#f5f7f5;--white:#fff;--line:#dce5e2;--teal:#18a999;--blue:#2f6bff;--gold:#f2b134;--red:#d85c5c;--violet:#7c5cfc}
*{box-sizing:border-box}html,body{margin:0;width:100%;height:100%;overflow:hidden;background:#07171d;font-family:Inter,Segoe UI,Arial,sans-serif;color:var(--ink)}
body{display:grid;place-items:center}.deck{width:min(100vw,177.78vh);height:min(100vh,56.25vw);position:relative;background:var(--paper);overflow:hidden;box-shadow:0 24px 80px #0008}
.slide{position:absolute;inset:0;padding:5.2% 6%;opacity:0;transform:translateX(3%);pointer-events:none;transition:opacity .35s ease,transform .35s ease;display:flex;flex-direction:column}.slide.active{opacity:1;transform:none;pointer-events:auto}.slide:before{content:attr(data-number);position:absolute;right:3.2%;bottom:2.6%;font-size:12px;color:#8da0a5}
h1{font-size:clamp(34px,5.1vw,78px);line-height:.98;margin:.1em 0 .25em;letter-spacing:-.045em}h2{font-size:clamp(25px,3.35vw,51px);line-height:1.02;margin:0 0 1.6%;letter-spacing:-.035em}h3{font-size:clamp(16px,1.55vw,25px);margin:.1em 0 .35em}p,li{font-size:clamp(13px,1.25vw,20px);line-height:1.38}.small{font-size:clamp(11px,1vw,16px)}.muted{color:var(--muted)}
.eyebrow{font-size:clamp(10px,.85vw,14px);text-transform:uppercase;letter-spacing:.15em;font-weight:800;color:var(--teal);margin-bottom:1%}.claim{display:inline-flex;align-items:center;gap:8px;border:1px solid #e2ad37;background:#fff3d4;color:#785207;border-radius:99px;padding:7px 12px;font-size:12px;font-weight:800}.claim:before{content:"";width:7px;height:7px;background:var(--gold);border-radius:50%}
.grid{display:grid;gap:2%}.g2{grid-template-columns:repeat(2,1fr)}.g3{grid-template-columns:repeat(3,1fr)}.g4{grid-template-columns:repeat(4,1fr)}.card{background:var(--white);border:1px solid var(--line);border-radius:18px;padding:5%;box-shadow:0 8px 22px #17323b0a}.card.teal{background:#e8f8f4;border-color:#b8e8dd}.card.blue{background:#eaf0ff;border-color:#c6d4ff}.card.red{background:#fff0ee;border-color:#f3c5bf}.card.gold{background:#fff6df;border-color:#f1d897}
.metric{font-size:clamp(24px,3.2vw,52px);font-weight:850;letter-spacing:-.04em;color:var(--blue)}.metric.teal{color:var(--teal)}.metric.red{color:var(--red)}.metric small{display:block;font-size:12px;font-weight:600;letter-spacing:0;color:var(--muted)}
.banner{padding:2.2%;border-radius:16px;background:var(--ink);color:#fff;font-size:clamp(15px,1.55vw,25px);line-height:1.25}.pipeline{display:grid;grid-template-columns:repeat(6,1fr);gap:1.1%;align-items:stretch}.pipe{position:relative;background:#fff;border:1px solid var(--line);border-radius:14px;padding:12% 7%;font-weight:750;font-size:clamp(10px,.86vw,14px);text-align:center;display:grid;place-items:center}.pipe:not(:last-child):after{content:"→";position:absolute;right:-10%;top:40%;z-index:2;color:var(--teal);font-size:18px}
.split{display:grid;grid-template-columns:1.05fr .95fr;gap:3%;min-height:0;flex:1}.chart{width:100%;height:100%;object-fit:contain;background:#fff;border:1px solid var(--line);border-radius:18px}.checks{list-style:none;padding:0;margin:0}.checks li{position:relative;padding:.55em 0 .55em 1.6em;border-bottom:1px solid var(--line)}.checks li:before{content:"✓";position:absolute;left:0;color:var(--teal);font-weight:900}.checks li.fail:before{content:"×";color:var(--red)}.checks li.warn:before{content:"!";color:var(--gold)}
.timeline{display:grid;grid-template-columns:repeat(4,1fr);gap:1.5%;margin-top:2%}.phase{border-top:5px solid var(--teal);background:#fff;border-radius:8px 8px 16px 16px;padding:9%}.phase:nth-child(2){border-color:var(--blue)}.phase:nth-child(3){border-color:var(--gold)}.phase:nth-child(4){border-color:var(--violet)}
.cover{background:radial-gradient(circle at 85% 15%,#36d7c550 0 13%,transparent 14%),linear-gradient(125deg,#071b22 0 64%,#0d3840 64%);color:#fff}.cover .eyebrow{color:#6ce4d2}.cover .lead{font-size:clamp(17px,1.65vw,28px);max-width:65%;color:#c4d7dc}.cover .meta{margin-top:auto;display:flex;justify-content:space-between;color:#a9c0c5;font-size:13px}.watermark{position:absolute;right:4%;top:4%;color:#ffd676;border:1px solid #ffd67666;padding:8px 12px;border-radius:99px;font-size:11px;font-weight:800;letter-spacing:.08em}
.callout{border-left:5px solid var(--teal);padding:1.5% 2.2%;background:#fff;border-radius:0 12px 12px 0}.source{font-size:10px;color:#73868b}.notes{display:none}.notes-on .notes{display:block;position:absolute;left:3%;right:3%;bottom:5%;background:#06171ef2;color:#fff;padding:16px;border-radius:12px;font-size:13px;z-index:8}.nav{position:absolute;left:3%;bottom:2.4%;display:flex;gap:6px;z-index:10}.nav button{border:1px solid #ffffff44;background:#10272fdd;color:#fff;border-radius:8px;padding:7px 10px;cursor:pointer}.progress{position:absolute;left:0;bottom:0;height:4px;background:var(--teal);transition:width .3s;z-index:12}.overview .slide{opacity:1;transform:none;pointer-events:auto;position:relative;display:flex;width:25%;height:25%;float:left;padding:1.2%;border:1px solid #ccd8d4;overflow:hidden}.overview{display:block;overflow:auto;background:#dce5e2}.overview .slide *{font-size:8px!important}.overview .slide h1,.overview .slide h2{font-size:13px!important}.overview .nav,.overview .progress{display:none}
.active .reveal{animation:rise .55s ease both}.active .reveal.d2{animation-delay:.12s}.active .reveal.d3{animation-delay:.24s}@keyframes rise{from{opacity:0;transform:translateY(12px)}to{opacity:1;transform:none}}@media(prefers-reduced-motion:reduce){*{animation:none!important;transition:none!important}}
</style></head><body><main class="deck" id="deck">
<section class="slide cover active" data-number="1 / 14"><span class="watermark">SMOKE_ONLY · PUBLIC / SYNTHETIC</span><div class="eyebrow">MVP técnico ejecutado · [[DATE]]</div><h1>FlowTwin<br>Predictive Operations</h1><p class="lead">ETA probabilística para Shipping Board y Freight Intelligence; inteligencia de proceso para Trace Port y TWINPORTS.</p><div class="meta"><span>EVOCON Solutions · Álvaro Schwiedop Souto</span><span>Kaleido Tech</span></div><aside class="notes">Abrir con el resultado: 1,88 horas de MAE en 85 viajes AIS futuros. Aclarar inmediatamente que es evidencia pública, no precisión Kaleido.</aside></section>

<section class="slide" data-number="2 / 14"><div class="eyebrow">Encaje actualizado</div><h2>Kaleido ya tiene las superficies correctas</h2><div class="grid g2 reveal"><div class="card teal"><h3>Shipping Board + Freight Intelligence</h3><p>Visibilidad de escalas, posiciones, carga y excepciones. FlowTwin añade ETA probabilística, ventana y confianza.</p></div><div class="card blue"><h3>Trace Port + TWINPORTS</h3><p>Memoria de proyectos, eventos, turnos, activos y estado espacial. FlowTwin añade proceso, tiempo restante y escenarios auditables.</p></div></div><div class="banner reveal d2" style="margin-top:2.2%">No proponemos otra plataforma: proponemos <b>Predictive Operations</b> como capa read-only entre productos existentes.</div><p class="source">Fuentes oficiales Kaleido; datasets del benchmark: NOAA MarineCadastre AIS y OCEL 2.0 Logistics.</p><aside class="notes">El resultado AIS da una entrada inmediata por Shipping Board/Freight Intelligence; Trace Port necesita export propio.</aside></section>

<section class="slide" data-number="3 / 14"><div class="eyebrow">Qué se ha construido</div><h2>Dos demostradores alineados y una línea de I+D</h2><div class="grid g4 reveal"><div class="card"><div class="metric teal">AIS ETA<small>geofence + ventanas</small></div></div><div class="card"><div class="metric">OCEL<small>objetos + proceso</small></div></div><div class="card"><div class="metric">JEPA<small>Event/T/Var + ablations</small></div></div><div class="card"><div class="metric red">M7<small>API + dashboard read-only</small></div></div></div><ul class="checks reveal d2"><li>Viajes y operaciones agrupados; ninguna trayectoria cruza particiones.</li><li>Modelos y gates elegidos en validación antes del test futuro.</li><li>[[TEST_COUNT]] tests, Ruff y mypy limpios en el último chequeo del paquete.</li><li class="warn">Todo sigue en <code>smoke_only</code>: datasets públicos/simulados, no export Kaleido.</li></ul><aside class="notes">La innovación útil es la combinación de física, eventos y aprendizaje con protocolo auditable.</aside></section>

<section class="slide" data-number="4 / 14"><div class="eyebrow">Datos y protocolo</div><h2>Ejemplos cercanos a los servicios de Kaleido</h2><div class="grid g4 reveal"><div class="card"><div class="metric teal">[[AIS_SOURCE_FILES]]<small>días AIS NOAA</small></div></div><div class="card"><div class="metric">[[AIS_SOURCE_GB]] GB<small>trayectorias comprimidas</small></div></div><div class="card"><div class="metric">[[AIS_TEST_TRIPS]]<small>viajes de test futuro</small></div></div><div class="card"><div class="metric">[[OCEL_ALIGNED_OBJECTS]]<small>contenedores OCEL</small></div></div></div><div class="split reveal d2" style="margin-top:2%"><div class="card"><h3>Holdout AIS congelado</h3><p><b>[[AIS_TRAIN_TRIPS]]</b> train · <b>[[AIS_VALIDATION_TRIPS]]</b> validación · <b>[[AIS_TEST_TRIPS]]</b> test</p><p class="small muted">Test: 1–7 febrero de 2025. MMSI no es feature. Cada viaje queda entero en una partición.</p></div><div class="card gold"><h3>Límite del resultado</h3><p class="small">NOAA AIS cubre aguas de EE. UU.; OCEL Logistics es simulado. Demuestran capacidad para ETA y procesos, no precisión ni valor Kaleido.</p></div></div><aside class="notes">El antiguo log de almacén queda como resultado histórico negativo, no como demostrador principal.</aside></section>

<section class="slide" data-number="5 / 14"><div class="eyebrow">Process intelligence sin ML</div><h2>El sistema ya entrega valor descriptivo</h2><div class="grid g4 reveal"><div class="card"><div class="metric teal">[[PROCESS_EVENTS]]<small>eventos OCEL</small></div></div><div class="card"><div class="metric">[[PROCESS_OBJECTS]]<small>objetos</small></div></div><div class="card"><div class="metric">[[PROCESS_RELATIONS]]<small>relaciones evento-objeto</small></div></div><div class="card"><div class="metric">[[PROCESS_VARIANTS]]<small>variantes</small></div></div></div><div class="callout reveal d2" style="margin-top:2%"><b>Qué mostraría en Trace Port:</b> variantes reales, esperas, rework, cuellos de botella, conformance y calidad temporal, incluso si todavía no hay suficiente señal para un predictor.</div><aside class="notes">Esto responde a la falta de histórico: el primer entregable no depende de deep learning.</aside></section>

<section class="slide" data-number="6 / 14"><div class="eyebrow">Resultado principal</div><h2>ETA AIS: 1,88 horas de error medio</h2><div class="split"><div><object class="chart reveal" data="assets/model_comparison.svg" type="image/svg+xml"></object></div><div class="reveal d2"><div class="card teal"><h3>Ganador · boosting ETA</h3><div class="metric teal">[[AIS_MAE]] h<small>MAE test · IC95 % [[AIS_CI_LOW]]–[[AIS_CI_HIGH]] h</small></div></div><div class="grid g2" style="margin-top:3%"><div class="card"><div class="metric">[[AIS_MEDIAN_AE]] h<small>error mediano</small></div></div><div class="card gold"><div class="metric">[[AIS_WITHIN_2]]%<small>dentro de ±2 h</small></div></div></div><p class="small muted">[[AIS_WITHIN_1]]% dentro de ±1 h · [[AIS_WITHIN_4]]% dentro de ±4 h · [[AIS_TEST_TRIPS]] viajes futuros.</p></div></div><div class="callout">Pasa 6/6 gates: mejora [[AIS_GAIN_KINEMATIC]]% a la ETA cinemática y [[AIS_GAIN_HISTORICAL]]% a la mediana puerto-distancia.</div><aside class="notes">Este sí es el número principal. Sigue siendo smoke_only y no debe llamarse precisión Kaleido.</aside></section>

<section class="slide" data-number="7 / 14"><div class="eyebrow">Horizonte, incertidumbre y límites</div><h2>Funciona como demo; aún debe aprender a abstenerse</h2><div class="grid g3 reveal"><div class="card teal"><div class="metric teal">[[AIS_MAE_2_6]] h<small>MAE con 2–6 h de anticipación</small></div></div><div class="card"><div class="metric">[[AIS_MAE_6_12]] h<small>MAE con 6–12 h</small></div></div><div class="card gold"><div class="metric">[[AIS_P90_COVERAGE]]%<small>cobertura P90</small></div></div></div><div class="grid g2 reveal d2" style="margin-top:2%"><div class="card"><h3>Incertidumbre visible</h3><p>La cobertura supera el 90 %, pero el intervalo tiene [[AIS_P90_WIDTH]] h de anchura media. Hace falta calibración por puerto y abstención.</p></div><div class="card red"><h3>Generalización limitada</h3><p>[[AIS_NOLA_SHARE]]% de los prefijos de test son Nueva Orleans; Houston y Los Ángeles tienen menos viajes. EE. UU. no sustituye Vigo ni datos Kaleido.</p></div></div><aside class="notes">No ocultar la anchura. El punto es bueno para un demostrador; el intervalo aún no es de piloto.</aside></section>

<section class="slide" data-number="8 / 14"><div class="eyebrow">I+D JEPA</div><h2>Qué aporta realmente el objetivo latente</h2><div class="split"><object class="chart reveal" data="assets/jepa_ablations.svg" type="image/svg+xml"></object><ul class="checks reveal d2"><li>JEPA mejora al encoder aleatorio: [[JEPA_MAE]] vs [[RANDOM_MAE]] min.</li><li>SIGReg es crítico: sin él, MAE [[NO_SIGREG_MAE]] y colapso.</li><li class="fail">Multi-horizonte no gana: completion-only [[ONE_HORIZON_MAE]] min.</li><li class="warn">Pares correctos vs barajados: solo 0,40 min de media y no gana en cada seed.</li></ul></div><div class="callout">Decisión: representación interesante y estable; todavía no evidencia robusta de dinámica temporal multihorizonte.</div><aside class="notes">Esta slide es la más importante para explicar investigación honesta: cada componente se intenta falsar.</aside></section>

<section class="slide" data-number="9 / 15"><div class="eyebrow">SOTA aplicado al log</div><h2>T-JEPA mejora la representación; el híbrido no gana</h2><div class="grid g2 reveal"><div class="card blue"><h3>Temporal T-JEPA</h3><div class="metric">[[TJEPA_MAE]] ± [[TJEPA_SD]]<small>futuro disjunto + teacher EMA + VISReg</small></div><p class="small">Mejora al Event-JEPA y gana al futuro barajado en 3/3 seeds, pero no a raw boosting.</p></div><div class="card gold"><h3>Var-Event-JEPA</h3><div class="metric">[[VAR_JEPA_MAE]] ± [[VAR_JEPA_SD]]<small>ELBO temporal · Spearman incertidumbre/error [[VAR_SPEARMAN]]</small></div><p class="small">La incertidumbre falla en dos seeds y no está calibrada en minutos.</p></div></div><div class="callout reveal d2" style="margin-top:2%"><b>Gate híbrido:</b> raw + Var-JEPA obtiene [[HYBRID_MAE]] ± [[HYBRID_SD]], frente a [[RAW3_MAE]] ± [[RAW3_SD]] de raw. Validación selecciona raw; no se promociona JEPA.</div><aside class="notes">Son adaptaciones temporales CPU-budgeted, no reproducciones exactas de T-JEPA o Var-T-JEPA.</aside></section>

<section class="slide" data-number="9 / 14"><div class="eyebrow">World model condicionado por acción</div><h2>VISReg recupera la señal inyectada, no valida causalidad</h2><div class="split"><object class="chart reveal" data="assets/action_recovery.svg" type="image/svg+xml"></object><div class="reveal d2"><div class="card teal"><h3>Correcta frente a barajada</h3><div class="metric teal">−[[ACTION_GAIN]] min<small>[[ACTION_CORRECT]] ± [[ACTION_CORRECT_SD]] vs [[ACTION_SHUFFLED]] · gana en 3/3 seeds</small></div></div><div class="card red" style="margin-top:3%"><h3>Gate de world model: cerrado</h3><p class="small">Rango efectivo sube a [[ACTION_RANK_LOW]]–[[ACTION_RANK_HIGH]], pero la escala media sigue en [[ACTION_SCALE_LOW]]–[[ACTION_SCALE_HIGH]], bajo el umbral. Acciones y efectos son sintéticos.</p></div></div></div><aside class="notes">Se puede decir recuperación de señal inyectada. No decir causalidad, ahorro ni acción recomendada real.</aside></section>

<section class="slide" data-number="10 / 14"><div class="eyebrow">Decisiones basadas en evidencia</div><h2>Qué entra en la demo y qué permanece en I+D</h2><div class="grid g2 reveal"><div class="card teal"><h3>Demostrar ahora</h3><ul class="checks"><li>ETA AIS + ventana ±2 h;</li><li>auditoría y process mining OCEL;</li><li>intervalos, razones y abstención;</li><li>replay read-only y API.</li></ul></div><div class="card gold"><h3>Investigar con gates</h3><ul class="checks"><li>Temporal T-JEPA y Var-JEPA;</li><li>incertidumbre por puerto mejor calibrada;</li><li>acciones reales correct-vs-shuffled;</li><li>object graph sobre relaciones Kaleido.</li></ul></div></div><div class="banner reveal d2" style="margin-top:2%">El producto público enseña ETA; JEPA sigue shadow hasta ganar valor incremental en el dataset adecuado.</div><aside class="notes">No servir el predictor de 734 minutos. Se conserva solo como negativo histórico.</aside></section>

<section class="slide" data-number="11 / 14"><div class="eyebrow">Arquitectura de integración</div><h2>Una capa predictiva dentro del ecosistema Kaleido</h2><div class="pipeline reveal"><div class="pipe">AIS + Shipping Board<br>posición/escala</div><div class="pipe">Trace Port<br>eventos/turnos</div><div class="pipe">TWINPORTS<br>activos/espacio</div><div class="pipe">Contrato temporal<br>planes versionados</div><div class="pipe">ETA + proceso<br>JEPA shadow</div><div class="pipe">P50/P90<br>API read-only</div></div><div class="grid g3 reveal d2" style="margin-top:3%"><div class="card"><h3>Cutoff visible</h3><p class="small muted">Qué posiciones y eventos conocía.</p></div><div class="card"><h3>Plan visible</h3><p class="small muted">Qué revisión estaba vigente.</p></div><div class="card"><h3>Humano decide</h3><p class="small muted">Sin escritura ni control automático.</p></div></div><aside class="notes">La primera entrada comercial puede ser Shipping Board/Freight Intelligence; Trace Port requiere su export.</aside></section>

<section class="slide" data-number="12 / 14"><div class="eyebrow">Producto demostrable</div><h2>Dashboard read-only: explorar, auditar y exportar</h2><div class="grid g3 reveal"><div class="card blue"><h3>Operación viva</h3><p>Búsqueda/filtros, timeline, P50/P90, objetos, cutoff y revisión de plan.</p></div><div class="card teal"><h3>Evidencia</h3><p>Escalera de modelos, interpretación del MAE, model card y export JSON/CSV.</p></div><div class="card gold"><h3>Escenarios</h3><p>Solo acciones aprobadas, ranking sintético y etiqueta de no ahorro realizado.</p></div></div><div class="callout reveal d2" style="margin-top:2.5%"><b>Formato:</b> web local para reunión, tab integrable para piloto, API versionada y audit trail copiable; ninguna escritura a sistemas fuente.</div><aside class="notes">Abrir la demo tras esta slide. Mostrar filtros, auditoría, evidencia, export y escenario sin afirmar que son datos Kaleido.</aside></section>

<section class="slide" data-number="13 / 14"><div class="eyebrow">Piloto con Kaleido</div><h2>Cuatro gates, una petición pequeña</h2><div class="timeline reveal"><div class="phase"><h3>1 · Contrato</h3><p class="small">Esquema + 3–5 operaciones para validar semántica.</p></div><div class="phase"><h3>2 · Evidencia</h3><p class="small">Histórico completo, planes y revisiones, outcomes.</p></div><div class="phase"><h3>3 · Shadow</h3><p class="small">4–8 semanas, falsas alertas, lead time y operador.</p></div><div class="phase"><h3>4 · Producto</h3><p class="small">Integración, buyer, coste real y packaging.</p></div></div><div class="banner reveal d2" style="margin-top:2.6%">Siguiente paso: sesión de 90 minutos sobre una operación repetible + export read-only pseudonimizado.</div><aside class="notes">Cerrar con fecha, propietario técnico, propietario operativo y export. No con un debate de arquitectura.</aside></section>

<section class="slide cover" data-number="14 / 14"><span class="watermark">EVIDENCE FIRST</span><div class="eyebrow">La propuesta</div><h1>Shipping Board anticipa.<br>Trace Port explica.<br>FlowTwin conecta.</h1><p class="lead">Empezar con ETA que pasa gates. Añadir procesos y planes Kaleido. Mantener JEPA como I+D que debe ganar.</p><div class="meta"><span>EVOCON Solutions</span><span>¿Acordamos operación, responsables y muestra?</span></div><aside class="notes">Pausa. Pedir el siguiente paso concreto.</aside></section>

<div class="nav"><button id="prev" aria-label="Anterior">←</button><button id="next" aria-label="Siguiente">→</button><button id="notes" aria-label="Notas">N</button><button id="overview" aria-label="Vista general">O</button><button id="full" aria-label="Pantalla completa">F</button></div><div class="progress" id="progress"></div>
</main><script>
const deck=document.getElementById('deck'),slides=[...document.querySelectorAll('.slide')],bar=document.getElementById('progress');let index=0;
function show(i){index=Math.max(0,Math.min(slides.length-1,i));slides.forEach((s,n)=>s.classList.toggle('active',n===index));bar.style.width=((index+1)/slides.length*100)+'%';}
document.getElementById('next').onclick=()=>show(index+1);document.getElementById('prev').onclick=()=>show(index-1);document.getElementById('notes').onclick=()=>deck.classList.toggle('notes-on');document.getElementById('overview').onclick=()=>deck.classList.toggle('overview');document.getElementById('full').onclick=()=>document.fullscreenElement?document.exitFullscreen():document.documentElement.requestFullscreen();
addEventListener('keydown',e=>{if(['ArrowRight','PageDown',' '].includes(e.key))show(index+1);if(['ArrowLeft','PageUp'].includes(e.key))show(index-1);if(e.key.toLowerCase()==='n')deck.classList.toggle('notes-on');if(e.key.toLowerCase()==='o')deck.classList.toggle('overview');if(e.key.toLowerCase()==='f')document.getElementById('full').click();});show(0);
</script></body></html>'''
    for old in range(14, 8, -1):
        template = template.replace(
            f'data-number="{old} / 14"',
            f'data-number="{old + 1} / 15"',
        )
    for old in range(1, 9):
        template = template.replace(
            f'data-number="{old} / 14"',
            f'data-number="{old} / 15"',
        )
    path.write_text(_render(template, _values(summary)), encoding="utf-8")


def write_presentation_tex(path: Path, summary: dict[str, Any]) -> None:
    template = r'''\documentclass[aspectratio=169,10pt]{beamer}
\usepackage[utf8]{inputenc}\usepackage[T1]{fontenc}\usepackage[spanish]{babel}
\usepackage{lmodern,graphicx,booktabs,tabularx,tikz,hyperref}
\usetikzlibrary{positioning}
\definecolor{Ink}{HTML}{10272F}\definecolor{Teal}{HTML}{18A999}\definecolor{Blue}{HTML}{2F6BFF}\definecolor{Gold}{HTML}{F2B134}\definecolor{Red}{HTML}{D85C5C}\definecolor{Paper}{HTML}{F5F7F5}\definecolor{Muted}{HTML}{61747B}
\setbeamercolor{normal text}{fg=Ink,bg=Paper}\setbeamercolor{frametitle}{fg=Ink}\setbeamercolor{structure}{fg=Teal}\setbeamertemplate{navigation symbols}{}\setbeamertemplate{footline}{\hfill\color{Muted}\insertframenumber/\inserttotalframenumber\hspace{5mm}\vspace{3mm}}
\setbeamertemplate{itemize item}{\color{Teal}$\blacktriangleright$}\graphicspath{{assets/}}
\newcommand{\claim}{\colorbox{Gold!22}{\textcolor{Ink}{\scriptsize\bfseries SMOKE\_ONLY -- PUBLIC / SYNTHETIC}}}
\newcommand{\bigmetric}[2]{\begin{center}{\color{Blue}\fontsize{25}{27}\selectfont\bfseries #1}\\[-1mm]{\scriptsize\color{Muted}#2}\end{center}}
\begin{document}
\begin{frame}[plain]\color{white}\begin{tikzpicture}[remember picture,overlay]\fill[Ink](current page.south west)rectangle(current page.north east);\fill[Teal!35](12.5,7)circle(2.2);\end{tikzpicture}\vspace{5mm}{\color{Teal!55}\bfseries MVP TÉCNICO EJECUTADO -- [[DATE]]}\vspace{5mm}\par{\fontsize{31}{32}\selectfont\bfseries FlowTwin\\Predictive Operations}\vspace{4mm}\par{\large ETA probabilística para Shipping Board/Freight Intelligence e inteligencia de proceso para Trace Port/TWINPORTS.}\vfill\claim\hfill EVOCON Solutions\end{frame}
\begin{frame}{Encaje en los productos Kaleido}\begin{columns}[T]\column{.49\textwidth}\begin{block}{Shipping Board + Freight Intelligence}Escalas, posiciones, carga y excepciones. FlowTwin añade ETA probabilística, ventana y confianza.\end{block}\column{.49\textwidth}\begin{block}{Trace Port + TWINPORTS}Eventos, turnos, activos y espacio. FlowTwin añade proceso, tiempo restante y escenarios auditables.\end{block}\end{columns}\vfill\begin{alertblock}{Encaje}Capa \textbf{Predictive Operations} read-only; no otra plataforma logística.\end{alertblock}\end{frame}
\begin{frame}{Dos demostradores y una línea de I+D}\begin{columns}\column{.24\textwidth}\bigmetric{AIS ETA}{geofence + ventanas}\column{.24\textwidth}\bigmetric{OCEL}{objetos + proceso}\column{.24\textwidth}\bigmetric{JEPA}{Event/T/Var}\column{.24\textwidth}\bigmetric{M7}{API + dashboard}\end{columns}\vspace{3mm}\begin{itemize}\item viajes y operaciones agrupados;\item modelos y gates elegidos antes del test futuro;\item [[TEST_COUNT]] tests, Ruff y mypy verificados;\item todo permanece en \texttt{smoke\_only}: no hay export Kaleido.\end{itemize}\end{frame}
\begin{frame}{Datos y protocolo alineados}\begin{columns}\column{.25\textwidth}\bigmetric{[[AIS_SOURCE_FILES]]}{días AIS NOAA}\column{.25\textwidth}\bigmetric{[[AIS_SOURCE_GB]] GB}{comprimidos}\column{.25\textwidth}\bigmetric{[[AIS_TEST_TRIPS]]}{viajes test}\column{.25\textwidth}\bigmetric{[[OCEL_ALIGNED_OBJECTS]]}{contenedores OCEL}\end{columns}\vspace{3mm}\begin{block}{Holdout AIS congelado}[[AIS_TRAIN_TRIPS]] train / [[AIS_VALIDATION_TRIPS]] validación / [[AIS_TEST_TRIPS]] test. Test futuro: 1--7 febrero 2025. MMSI excluido de features.\end{block}\begin{alertblock}{Límite}AIS de EE. UU. y OCEL simulado demuestran capacidad, no precisión ni valor Kaleido.\end{alertblock}\end{frame}
\begin{frame}{Process intelligence sin entrenamiento}\begin{columns}\column{.25\textwidth}\bigmetric{[[PROCESS_EVENTS]]}{eventos OCEL}\column{.25\textwidth}\bigmetric{[[PROCESS_OBJECTS]]}{objetos}\column{.25\textwidth}\bigmetric{[[PROCESS_RELATIONS]]}{relaciones}\column{.25\textwidth}\bigmetric{[[PROCESS_VARIANTS]]}{variantes}\end{columns}\vfill\begin{block}{Valor inmediato}Variantes, esperas, rework, bottlenecks, conformance y calidad temporal funcionan incluso cuando el histórico no permite un predictor.\end{block}\end{frame}
\begin{frame}{ETA AIS: 1,88 horas de error medio}\begin{columns}\column{.58\textwidth}\includegraphics[width=\linewidth]{model_comparison.pdf}\column{.40\textwidth}\begin{block}{Boosting ETA}\bigmetric{[[AIS_MAE]] h}{IC95\% [[AIS_CI_LOW]]--[[AIS_CI_HIGH]] h}\end{block}\begin{block}{Ventana demostrable}\bigmetric{[[AIS_WITHIN_2]]\%}{dentro de $\pm$2 h}\end{block}\end{columns}\vfill\begin{alertblock}{6/6 gates aprobados}[[AIS_GAIN_KINEMATIC]]\% mejor que ETA cinemática; [[AIS_GAIN_HISTORICAL]]\% mejor que mediana puerto-distancia; [[AIS_TEST_TRIPS]] viajes futuros.\end{alertblock}\end{frame}
\begin{frame}{Horizonte, incertidumbre y límites}\begin{columns}\column{.33\textwidth}\bigmetric{[[AIS_MAE_2_6]] h}{MAE a 2--6 h}\column{.33\textwidth}\bigmetric{[[AIS_MAE_6_12]] h}{MAE a 6--12 h}\column{.33\textwidth}\bigmetric{[[AIS_P90_COVERAGE]]\%}{cobertura P90}\end{columns}\vfill\begin{alertblock}{Lo que falta}P90 ancho [[AIS_P90_WIDTH]] h y [[AIS_NOLA_SHARE]]\% de prefijos test en Nueva Orleans. Hace falta calibración por puerto y datos Kaleido/Vigo.\end{alertblock}\end{frame}
\begin{frame}{Ablations: qué aporta JEPA}\begin{columns}\column{.60\textwidth}\includegraphics[width=\linewidth]{jepa_ablations.pdf}\column{.38\textwidth}\begin{itemize}\item mejora al encoder aleatorio;\item SIGReg evita colapso;\item completion-only gana marginalmente;\item pares temporales correctos no ganan de forma robusta.\end{itemize}\end{columns}\vfill\claim\end{frame}
\begin{frame}{T-JEPA mejora; Var-JEPA no añade valor incremental}\begin{columns}[T]\column{.49\textwidth}\begin{block}{Temporal T-JEPA}\bigmetric{[[TJEPA_MAE]] $\pm$ [[TJEPA_SD]]}{futuro disjunto + EMA + VISReg}\small Mejora al Event-JEPA y gana al futuro barajado en 3/3 seeds, pero no a raw.\end{block}\column{.49\textwidth}\begin{block}{Var-Event-JEPA}\bigmetric{[[VAR_JEPA_MAE]] $\pm$ [[VAR_JEPA_SD]]}{Spearman incertidumbre/error [[VAR_SPEARMAN]]}\small La incertidumbre falla en dos seeds y no está calibrada en minutos.\end{block}\end{columns}\vfill\begin{alertblock}{Gate híbrido cerrado}Raw + Var-JEPA: [[HYBRID_MAE]] $\pm$ [[HYBRID_SD]] frente a [[RAW3_MAE]] $\pm$ [[RAW3_SD]] de raw.\end{alertblock}\end{frame}
\begin{frame}{World model sintético: señal recuperada, modelo no promovido}\begin{columns}\column{.60\textwidth}\includegraphics[width=\linewidth]{action_recovery.pdf}\column{.38\textwidth}\bigmetric{--[[ACTION_GAIN]] min}{correcta frente a barajada; gana 3/3 seeds}\small Rango efectivo [[ACTION_RANK_LOW]]--[[ACTION_RANK_HIGH]], pero escala [[ACTION_SCALE_LOW]]--[[ACTION_SCALE_HIGH]] bajo el umbral. Acciones y efectos generados.\end{columns}\vfill\begin{alertblock}{Límite del resultado}Recuperación de señal inyectada; no causalidad, ahorro ni acción Kaleido.\end{alertblock}\end{frame}
\begin{frame}{Decisión de producto}\begin{columns}[T]\column{.49\textwidth}\begin{block}{Demostrar ahora}\begin{itemize}\item ETA AIS + ventana $\pm$2 h;\item auditoría y process mining OCEL;\item explicación, intervalo y abstención;\item replay/API read-only.\end{itemize}\end{block}\column{.49\textwidth}\begin{block}{I+D con gates}\begin{itemize}\item Temporal T-JEPA y Var-JEPA;\item calibración por puerto;\item acciones reales correct-vs-shuffled;\item object graph sobre datos Kaleido.\end{itemize}\end{block}\end{columns}\vfill El predictor de 734 minutos queda como negativo histórico, no como demostrador.\end{frame}
\begin{frame}{Arquitectura complementaria}\centering\begin{tikzpicture}[node distance=2.2mm,box/.style={draw,rounded corners,fill=white,minimum width=1.75cm,minimum height=1.05cm,align=center,font=\tiny},arr/.style={->,thick,Teal}]\node[box](ais){AIS/Shipping\\posición/escala};\node[box,right=of ais](tp){Trace Port\\eventos/turnos};\node[box,right=of tp](tw){TWINPORTS\\activos/espacio};\node[box,right=of tw](ct){contrato temporal\\planes versionados};\node[box,right=of ct](ml){ETA + proceso\\JEPA shadow};\node[box,right=of ml](ui){P50/P90\\API read-only};\draw[arr](ais)--(tp);\draw[arr](tp)--(tw);\draw[arr](tw)--(ct);\draw[arr](ct)--(ml);\draw[arr](ml)--(ui);\end{tikzpicture}\vfill Cutoff visible · revisión de plan visible · humano decide.\end{frame}
\begin{frame}{Dashboard read-only: explorar, auditar y exportar}\begin{columns}[T]\column{.32\textwidth}\begin{block}{Operación viva}Filtros, timeline, P50/P90, objetos, cutoff y plan.\end{block}\column{.32\textwidth}\begin{block}{Evidencia}MAE interpretado, model card, audit trail y export JSON/CSV.\end{block}\column{.32\textwidth}\begin{block}{Escenarios}Acciones aprobadas y etiqueta de simulación, no ahorro realizado.\end{block}\end{columns}\vfill\begin{alertblock}{Formato}Web local; API versionada; ninguna escritura a Trace Port, TOS, ERP o equipos.\end{alertblock}\end{frame}
\begin{frame}{Piloto con Kaleido}\begin{columns}[T]\column{.24\textwidth}\begin{block}{1. Contrato}Esquema + 3--5 casos.\end{block}\column{.24\textwidth}\begin{block}{2. Evidencia}Planes, revisiones, outcomes.\end{block}\column{.24\textwidth}\begin{block}{3. Shadow}4--8 semanas + operador.\end{block}\column{.24\textwidth}\begin{block}{4. Producto}Buyer, coste, integración.\end{block}\end{columns}\vfill\begin{alertblock}{Siguiente paso}Sesión de 90 minutos sobre una operación repetible + export read-only pseudonimizado.\end{alertblock}\end{frame}
\begin{frame}[plain]\color{white}\begin{tikzpicture}[remember picture,overlay]\fill[Ink](current page.south west)rectangle(current page.north east);\end{tikzpicture}\vspace{12mm}{\color{Teal!60}\bfseries LA PROPUESTA}\vspace{5mm}\par{\fontsize{27}{30}\selectfont\bfseries Shipping Board anticipa.\\Trace Port explica.\\FlowTwin conecta.}\vfill ETA que pasa gates · procesos auditables · JEPA debe ganar.\hfill\claim\end{frame}
\end{document}'''
    path.write_text(_render(template, _values(summary)), encoding="utf-8")


def write_technical_tex(path: Path, summary: dict[str, Any]) -> None:
    template = r'''\documentclass[11pt,a4paper]{article}
\usepackage[utf8]{inputenc}\usepackage[T1]{fontenc}\usepackage[spanish]{babel}\usepackage{lmodern}
\usepackage[a4paper,margin=2.2cm]{geometry}\usepackage{graphicx,booktabs,longtable,tabularx,array,xcolor,hyperref,fancyhdr,microtype}
\definecolor{Ink}{HTML}{10272F}\definecolor{Teal}{HTML}{18A999}\definecolor{Blue}{HTML}{2F6BFF}\definecolor{Gold}{HTML}{F2B134}\definecolor{Red}{HTML}{D85C5C}\definecolor{Paper}{HTML}{F5F7F5}\definecolor{Muted}{HTML}{61747B}
\hypersetup{colorlinks=true,linkcolor=Blue,urlcolor=Blue}\graphicspath{{../../presentacion/assets/}}
\pagestyle{fancy}\fancyhf{}\setlength{\headheight}{14pt}\lhead{Kaleido FlowTwin}\rhead{MVP técnico -- smoke\_only}\cfoot{\thepage}
\newcommand{\claimbox}[1]{\begin{center}\fcolorbox{Gold}{Gold!18}{\parbox{.92\linewidth}{\textbf{#1}}}\end{center}}
\title{\textbf{Kaleido FlowTwin}\\Informe técnico y ejecutivo del MVP}\author{EVOCON Solutions -- Álvaro Schwiedop Souto}\date{[[DATE]]}
\begin{document}\maketitle
\claimbox{Estado de toda la evidencia: \texttt{smoke\_only}. Los datos públicos y sintéticos demuestran competencia técnica del pipeline; no demuestran precisión, valor, causalidad, ahorro ni despliegue para Kaleido.}
\section*{Resumen ejecutivo}
Se implementó y ejecutó una capa read-only de inteligencia operativa con ETA, incertidumbre, OCEL/process mining, baselines, modelos secuenciales, Event-JEPA, Temporal T-JEPA, Var-Event-JEPA, API y dashboard. El demostrador principal usa [[AIS_SOURCE_FILES]] días de NOAA MarineCadastre AIS y predice entrada en una geofence portuaria. \textbf{Boosting ETA}, elegido solo en validación, obtiene [[AIS_MAE]] horas de MAE en [[AIS_TEST_TRIPS]] viajes futuros del 1 al 7 de febrero de 2025, con bootstrap por viaje IC95\% [[AIS_CI_LOW]]--[[AIS_CI_HIGH]].

La mediana del error es [[AIS_MEDIAN_AE]] horas; [[AIS_WITHIN_1]]\% de los puntos queda dentro de $\pm$1 hora, [[AIS_WITHIN_2]]\% dentro de $\pm$2 y [[AIS_WITHIN_4]]\% dentro de $\pm$4. Mejora [[AIS_GAIN_KINEMATIC]]\% a una ETA distancia/velocidad y [[AIS_GAIN_HISTORICAL]]\% a la mediana puerto-distancia. Pasa los seis gates predeclarados del nuevo holdout. La cobertura P90 es [[AIS_P90_COVERAGE]]\%, pero su anchura media de [[AIS_P90_WIDTH]] horas obliga a mostrar incertidumbre y abstención.

El segundo dataset, OCEL 2.0 Container Logistics, prueba contratos objeto-céntricos y revisiones de plan. El grafo correcto reduce el MAE de test de [[OCEL_ALIGNED_FLAT]] a [[OCEL_ALIGNED_GRAPH]] horas, pero validación seleccionó la traza plana; el gate del grafo permanece cerrado. Este resultado se usa para process intelligence y como diagnóstico de I+D, no como claim predictivo.

Temporal T-JEPA incorpora futuros disjuntos, teacher EMA y selección de regularización solo en validación. Mejora al Event-JEPA anterior con [[TJEPA_MAE]] $\pm$ [[TJEPA_SD]], pero no a raw boosting. Var-Event-JEPA obtiene [[VAR_JEPA_MAE]] $\pm$ [[VAR_JEPA_SD]] y su incertidumbre latente no sigue el error en todas las seeds. El mejor híbrido elegible, raw + Var-JEPA, obtiene [[HYBRID_MAE]] $\pm$ [[HYBRID_SD]], [[HYBRID_DELTA]] minutos peor que raw; el gate de promoción queda cerrado.

La recomendación es demostrar ETA y ventanas dentro de Shipping Board/Freight Intelligence, llevar process intelligence a Trace Port/TWINPORTS y mantener JEPA como módulo shadow sujeto a gates. Ningún resultado público prueba precisión o valor para Kaleido.

\section{Hipótesis y encaje con Kaleido}
\textbf{Hipótesis principal.} A partir de posiciones AIS y prefijos causales de eventos, el sistema puede estimar ETA/tiempo restante con incertidumbre y, con datos Kaleido, riesgo de desviación material con lead time accionable.

Shipping Board y Freight Intelligence aportan la superficie natural para ETA y excepciones. Trace Port declara eventos, turnos, equipos, packing lists, incidencias, histórico y API; TWINPORTS desarrolla estado físico y espacial. FlowTwin no debe competir con ellos: debe consumir posiciones/eventos versionados y devolver salidas read-only dentro de los productos existentes.

\section{Datos, provenance y protocolo}
\noindent\begin{tabularx}{\linewidth}{@{}>{\bfseries}p{2.6cm}X@{}}\toprule
Dataset principal & NOAA MarineCadastre AIS 2025, [[AIS_SOURCE_FILES]] días, [[AIS_SOURCE_GB]] GB comprimidos \\
Dominio & Cargueros/petroleros en Nueva York, Houston, Los Ángeles y Nueva Orleans \\
Task & Horas hasta entrada en geofence circular, con prefijos entre 12 h y 15 min \\
Split & Futuro fijo por viaje: [[AIS_TRAIN_TRIPS]] / [[AIS_VALIDATION_TRIPS]] / [[AIS_TEST_TRIPS]] viajes \\
Test & 1--7 febrero 2025; [[AIS_TEST_PREFIXES]] puntos; MMSI excluido del modelo \\
Dataset secundario & OCEL 2.0 Container Logistics, [[OCEL_ALIGNED_OBJECTS]] contenedores y [[OCEL_ALIGNED_PREFIXES]] prefijos \\
Selección & Variantes y gates con validación; el test futuro no eligió modelo ni umbral \\
Seeds & 3 (41, 42, 43) para los benchmarks alineados \\
Claim state & \texttt{smoke\_only} \\
\bottomrule\end{tabularx}

El primer AIS test se invalidó como evidencia de presentación por contener solo 25 viajes. El segundo test de 73 viajes falló el gate subhorario predeclarado. Ambos se preservaron; el protocolo final mantuvo puertos/features/modelos, convirtió enero en desarrollo/validación y reservó febrero como nuevo futuro intacto. También se conservan tres runs históricos invalidados por fuga o selección con test.

\section{Resultado principal: ETA AIS}
\begin{center}\includegraphics[width=.93\linewidth]{model_comparison.pdf}\end{center}
\begin{tabularx}{\linewidth}{Xrrrrl}\toprule
Modelo & MAE h & Mediana AE & $\pm$1 h & $\pm$2 h & Decisión \\
\midrule
Boosting ETA & [[AIS_MAE]] & [[AIS_MEDIAN_AE]] & [[AIS_WITHIN_1]]\% & [[AIS_WITHIN_2]]\% & seleccionado \\
Híbrido físico-residual & [[AIS_RESIDUAL]] & -- & -- & -- & segundo \\
Mediana puerto-distancia & [[AIS_HISTORICAL]] & -- & -- & -- & baseline histórico \\
ETA distancia/velocidad & [[AIS_KINEMATIC]] & -- & -- & -- & baseline físico \\
\bottomrule\end{tabularx}

Los seis gates predeclarados pasan: al menos 50 viajes, MAE máximo 2,5 h, extremo superior del IC95\% menor de 3 h, al menos 50\% dentro de $\pm$2 h y mejoras mínimas frente a ambos baselines. Por horizonte, el MAE es [[AIS_MAE_0_2]] h a 0--2 h, [[AIS_MAE_2_6]] h a 2--6 h y [[AIS_MAE_6_12]] h a 6--12 h.

\textbf{Límites.} [[AIS_NOLA_SHARE]]\% de los prefijos de test pertenecen a Nueva Orleans. La cobertura P90 de [[AIS_P90_COVERAGE]]\% usa intervalos de [[AIS_P90_WIDTH]] h de anchura media. Las geofences son circulares inferidas y el dominio es EE. UU. El resultado habilita un demostrador de capacidad, no una promesa operativa para Kaleido.

\section{Process intelligence}
El loader OCEL público procesó [[PROCESS_EVENTS]] eventos, [[PROCESS_OBJECTS]] objetos y [[PROCESS_RELATIONS]] relaciones evento-objeto, con [[PROCESS_VARIANTS]] variantes e integridad de grafo aprobada. Este bloque funciona sin entrenamiento y es el primer valor del piloto: variantes, tiempos de espera, rework, cuellos de botella, conformance y calidad temporal.

\section{Experimento histórico rechazado: Warehouse remaining time}
Este benchmark se conserva por auditabilidad y para estudiar JEPA, pero se retira de la demostración predictiva. Su escala absoluta e intervalo no resultan adecuados frente al nuevo ejemplo AIS.
\begin{tabularx}{\linewidth}{Xrrrrl}\toprule
Modelo & Seeds & MAE min & SD/IC & P90 & Decisión \\
\midrule
Boosting histórico & 1 & [[BOOST_MAE]] & IC [[BOOST_CI_LOW]]--[[BOOST_CI_HIGH]] & [[BOOST_P90]]\% & referencia \\
Raw boosting rerun & 3 & [[RAW3_MAE]] & SD [[RAW3_SD]] & -- & no servir \\
Raw + Var-JEPA & 3 & [[HYBRID_MAE]] & SD [[HYBRID_SD]] & -- & rechazado \\
Temporal T-JEPA & 3 & [[TJEPA_MAE]] & SD [[TJEPA_SD]] & -- & shadow I+D \\
Var-Event-JEPA & 3 & [[VAR_JEPA_MAE]] & SD [[VAR_JEPA_SD]] & -- & shadow I+D \\
Event-JEPA frozen & 3 & [[JEPA_MAE]] & SD [[JEPA_SD]] & [[JEPA_P90]]\% & shadow I+D \\
ProcessTransformer & 3 & [[TRANS_MAE]] & SD [[TRANS_SD]] & -- & no supera floor \\
\bottomrule\end{tabularx}

\subsection{Por qué los aproximadamente 700 minutos no son el demostrador}
El MAE histórico de [[BOOST_MAE]] minutos equivale a [[BOOST_MAE_HOURS]] horas de error absoluto medio. La comparación relativa dentro del protocolo sigue siendo válida: mejora [[GLOBAL_GAIN_PCT]]\% a la mediana global y [[ACTIVITY_GAIN_PCT]]\% a la mediana por actividad. Sin embargo, ganar por poco a baselines débiles no convierte la escala absoluta en una predicción presentable.

La lectura tampoco debe quedarse en la media. El error absoluto mediano de boosting es [[BOOST_MEDIAN_AE]] minutos, frente a [[GLOBAL_MEDIAN_AE]] de la mediana global; por tanto, la mejora de MAE procede sobre todo de reducir errores grandes, no de ganar en cada caso típico. El peor grupo, \texttt{[[WORST_GROUP]]}, alcanza [[WORST_GROUP_MAE]] minutos. La cobertura P90 es [[BOOST_P90]]\% con anchura media [[BOOST_WIDTH]] minutos, [[BOOST_WIDTH_HOURS]] horas. Ese intervalo amplio evidencia incertidumbre elevada y debe mostrarse.

La conclusión es de rechazo: este predictor no se sirve ni se usa como ejemplo comercial. Se mantiene como resultado negativo que motivó buscar una tarea más cercana a Shipping Board/Freight Intelligence. El benchmark AIS expresa error en horas, supera gates absolutos y compara contra una ETA física; aun así continúa en \texttt{smoke\_only}.

\subsection{Proxy de riesgo}
No existe plan versionado en el dataset, por lo que el target es duración larga por encima del P75 de training, no desviación material. AUPRC [[RISK_AUPRC]], ECE [[RISK_ECE]] y [[RISK_FALSE]] falsas alertas por 100 operaciones son diagnósticos de desarrollo. No habilitan alerta portuaria ni claim de early warning.

\section{Event-JEPA y ablations}
\begin{center}\includegraphics[width=.93\linewidth]{jepa_ablations.pdf}\end{center}
\begin{tabularx}{\linewidth}{lrrX}\toprule
Variante & MAE & SD & Lectura \\
\midrule
Main multi-horizon + SIGReg & [[JEPA_MAE]] & [[JEPA_SD]] & referencia seleccionada en validación \\
Completion-only + SIGReg & [[ONE_HORIZON_MAE]] & 1.77 & multi-horizonte no aporta mejora \\
Shuffled temporal pairs & [[SHUFFLED_PAIR_MAE]] & 2.15 & diferencia media 0.40; no consistente por seed \\
Random encoder, no JEPA & [[RANDOM_MAE]] & 2.67 & el objetivo JEPA aporta señal \\
Multi-horizon sin SIGReg & [[NO_SIGREG_MAE]] & 13.90 & colapso; regularizador necesario \\
\bottomrule\end{tabularx}

La interpretación correcta no es ``JEPA gana''. Es: (1) hay valor de representación frente a encoder aleatorio; (2) SIGReg es necesario en el log; (3) no se ha validado ventaja multi-horizonte; (4) el emparejamiento temporal apenas cambia el downstream, por lo que aún no hay evidencia robusta de dinámica futura rica.

\section{Temporal T-JEPA, Var-JEPA y boosting híbrido}
Temporal T-JEPA corrige dos debilidades de la primera implementación: el target contiene solo el sufijo futuro y un teacher EMA con stop-gradient produce la representación objetivo. La regularización se eligió entre SiGReg, VISReg y token de registro usando validación. \texttt{multi\_visreg} obtiene [[TJEPA_MAE]] $\pm$ [[TJEPA_SD]]; completion-only obtiene [[TJEPA_COMPLETION]] y el futuro barajado [[TJEPA_SHUFFLED]]. El futuro correcto gana al barajado en 3/3 seeds, pero multi-horizonte no gana a completion-only en todas.

Var-Event-JEPA sustituye el punto latente determinista por distribuciones gaussianas de contexto, variable auxiliar y futuro. El ELBO combina reconstrucción, generación y términos KL. Obtiene [[VAR_JEPA_MAE]] $\pm$ [[VAR_JEPA_SD]]; la correlación Spearman media entre incertidumbre latente y error es [[VAR_SPEARMAN]], con dos seeds negativas. Esa incertidumbre es diagnóstica y no está calibrada en minutos.

El experimento híbrido entrena raw, raw+T-JEPA, raw+Var-JEPA y raw+ambos con el mismo split y tres seeds. Validación selecciona raw. Entre híbridos selecciona \texttt{[[HYBRID_NAME]]}, que obtiene [[HYBRID_MAE]] $\pm$ [[HYBRID_SD]], frente a [[RAW3_MAE]] $\pm$ [[RAW3_SD]] de raw. La diferencia de [[HYBRID_DELTA]] minutos cierra el gate de promoción.

\textbf{Lectura ELI5.} Boosting ve directamente reloj, progreso, actividad y espera. JEPA comprime una traza corta en un vector; en este log regular el resumen descarta detalle temporal y añade ruido. Esto no prueba que JEPA sea inferior en general: indica que todavía no gana su complejidad sin planes, objetos, contexto o acciones reales más ricos.

\section{Acciones sintéticas y world model}
Las actividades públicas no se relabelaron como acciones. Se generó un overlay separado con acciones timestamped, propensión, elegibilidad, coste y efecto estructural conocido. El boosting de recuperación v1 falló correct-vs-shuffled. El Action-Event-JEPA con SIGReg mejoró en media, pero no en cada seed y colapsó. La ablation condicional VISReg se activó porque había colapso en validación.

\begin{center}\includegraphics[width=.93\linewidth]{action_recovery.pdf}\end{center}
VISReg obtiene [[ACTION_CORRECT]] $\pm$ [[ACTION_CORRECT_SD]] minutos con acción correcta frente a [[ACTION_SHUFFLED]] con acción barajada y [[ACTION_PREFIX]] con prefijo solo. Correct gana en 3/3 seeds y mejora [[ACTION_GAIN]] minutos de media frente a shuffled. Sin embargo, aunque el rango efectivo sube a [[ACTION_RANK_LOW]]--[[ACTION_RANK_HIGH]], la desviación media por dimensión permanece [[ACTION_SCALE_LOW]]--[[ACTION_SCALE_HIGH]], bajo el umbral. El resultado prueba recuperación de señal inyectada, no un world model operativo estable.

\section{Serving y producto recomendado}
La entrada demostrable es un panel ETA/ventanas embebible en Shipping Board o Freight Intelligence. Trace Port/TWINPORTS reciben process intelligence y, cuando exista export Kaleido, tiempo restante de operación. Cada salida expone cutoff, geofence/plan visible, P50/P90, confianza/abstención, razones y fuente. El sistema continúa read-only y los escenarios permanecen separados y etiquetados como simulación.

\section{Seguridad, límites y gates de piloto}
\begin{itemize}
\item conectores y API read-only; sin endpoints de escritura ni control de equipos;
\item seudonimización bajo control de Kaleido; fotos/notas fuera del primer modelo;
\item planes inmutables por \texttt{valid\_from}; censura explícita; split por operación;
\item umbrales en validación; reference model congelado; adaptación solo shadow con rollback;
\item GDPR, retención, roles y revisión de operador pendientes antes del piloto.
\end{itemize}

\claimbox{Lo que esto no demuestra: precisión o valor para Kaleido; generalización portuaria; desviación material; causalidad o contrafactuales; ROI, ahorro realizado, producción o despliegue exitoso.}

\section{Próximo paso falsable}
Sesión de 90 minutos para elegir una operación repetible, usuario/buyer, desviación material, acción disponible y coste de intervenir. Solicitar esquema y 3--5 operaciones pseudonimizadas para validar semántica; después un histórico suficiente con plan original y revisiones, outcomes y acciones timestamped. Ejecutar replay shadow 4--8 semanas con gates preacordados de cobertura, falsas alertas, lead time, peor grupo y utilidad del operador.

\section*{Cierre de tarea}
\textbf{Hipótesis:} una capa predictiva sobre posiciones y eventos puede ampliar Shipping Board, Freight Intelligence, Trace Port y TWINPORTS.\\
\textbf{Cambios:} benchmark ETA AIS con holdout futuro, benchmark OCEL objeto-céntrico, pipeline causal, JEPA/ablations, API/dashboard e informes generados.\\
\textbf{Tests:} suites unitarias/integración/adversariales, lint y typing; hashes de los runs citados verificados al generar este documento.\\
\textbf{Evidencia:} ETA boosting pasa 6/6 gates en [[AIS_TEST_TRIPS]] viajes futuros; el grafo OCEL y JEPA no superan sus gates; VISReg recupera señal sintética sin validar causalidad.\\
\textbf{Limitaciones:} dominio AIS de EE. UU. concentrado en Nueva Orleans; OCEL simulado; no hay datos Kaleido, outcomes materiales ni acciones reales.\\
\textbf{Siguiente paso falsable:} replay sobre export Kaleido congelado y revisión operativa.

\section*{Fuentes primarias y corporativas}
\begin{itemize}\small
\item \href{https://www.kaleidologistics.com/en/productos-ktech/trace-port/}{Kaleido Trace Port -- producto oficial}
\item \href{https://www.kaleidologistics.com/es/proyectos-financiados/el-proyecto-twinports-despega-en-el-puerto-de-vigo-comienzan-los-trabajos-para-crear-el-gemelo-digital-de-la-terminal-portuaria-de-kaleido/}{Kaleido TWINPORTS -- proyecto oficial}
\item \href{https://coast.noaa.gov/digitalcoast/tools/ais.html}{NOAA MarineCadastre AccessAIS -- fuente oficial}
\item \href{https://coast.noaa.gov/data/marinecadastre/ais/faq.pdf}{NOAA AIS FAQ -- cálculo de ETA y cobertura}
\item \href{https://doi.org/10.5281/zenodo.18373888}{OCEL 2.0 Container Logistics -- registro oficial}
\item \href{https://doi.org/10.6084/m9.figshare.29500898}{Warehouse Outbound Event Log -- DOI}
\item \href{https://arxiv.org/abs/2511.08544}{LeJEPA}; \href{https://arxiv.org/abs/2603.19312}{LeWorldModel}
\item \href{https://arxiv.org/abs/2606.02572}{VISReg}; \href{https://haiyuwu.github.io/visreg/}{proyecto oficial}
\item \href{https://arxiv.org/abs/2410.05016}{T-JEPA}; \href{https://arxiv.org/abs/2603.20111}{Var-JEPA}
\end{itemize}
\end{document}'''
    path.write_text(_render(template, _values(summary)), encoding="utf-8")


def write_speaker_script_tex(path: Path, summary: dict[str, Any]) -> None:
    template = r'''\documentclass[11pt,a4paper]{article}
\usepackage[utf8]{inputenc}\usepackage[T1]{fontenc}\usepackage[spanish]{babel}\usepackage{lmodern}
\usepackage[a4paper,margin=2cm,headheight=15pt]{geometry}
\usepackage{xcolor,hyperref,fancyhdr,enumitem,tabularx,array,booktabs,microtype}
\definecolor{Ink}{HTML}{10272F}\definecolor{Teal}{HTML}{18A999}\definecolor{Blue}{HTML}{2F6BFF}
\definecolor{Gold}{HTML}{F2B134}\definecolor{Red}{HTML}{D85C5C}\definecolor{Paper}{HTML}{F5F7F5}\definecolor{Muted}{HTML}{61747B}
\hypersetup{colorlinks=true,linkcolor=Blue,urlcolor=Blue}
\pagestyle{fancy}\fancyhf{}\lhead{Kaleido FlowTwin}\rhead{Guion de presentación}\cfoot{\thepage}
\setlength{\parindent}{0pt}\setlength{\parskip}{5pt}
\setlist[itemize]{leftmargin=5mm,itemsep=2pt,topsep=2pt}
\newcommand{\claim}{\texttt{smoke\_only}}
\newcommand{\slidehead}[3]{%
  \par\vspace{7pt}\noindent
  \colorbox{Ink}{\parbox{\dimexpr\linewidth-2\fboxsep\relax}{%
    \color{white}\textbf{Diapositiva #1 -- #2}\hfill\color{Teal!45}\textbf{#3}}}%
  \par\vspace{5pt}}
\newcommand{\cue}[1]{\textcolor{Teal}{\textbf{#1}}}
\newcommand{\avoid}[1]{\textcolor{Red}{\textbf{No decir:}} #1}
\newcommand{\transition}[1]{\textcolor{Blue}{\textbf{Transición:}} \emph{``#1''}}
\title{\textbf{Kaleido FlowTwin}\\Guion de la presentación del MVP}
\author{EVOCON Solutions -- Álvaro Schwiedop Souto}
\date{[[DATE]]}
\begin{document}
\maketitle

\begin{center}
\fcolorbox{Gold}{Gold!16}{\parbox{.92\linewidth}{
\textbf{Estado de evidencia: \claim.} El objetivo de la reunión no es afirmar que el
modelo ya funciona para Kaleido, sino demostrar un MVP ejecutado, explicar qué se ha
aprendido y acordar el acceso a una operación real para un replay shadow.
}}
\end{center}

\section*{Mapa rápido}
\begin{tabularx}{\linewidth}{@{}>{\raggedright\arraybackslash\bfseries}p{3.6cm}X@{}}\toprule
Duración objetivo & 20--22 minutos de presentación + 10 minutos de preguntas. \\
Petición final & Sesión de 90 minutos sobre una operación repetible y 3--5 casos pseudonimizados. \\
Idea central & Shipping Board informa; Freight Intelligence analiza; FlowTwin anticipa ETA y excepciones. \\
Resultado principal & ETA AIS: [[AIS_MAE]] h de MAE, [[AIS_WITHIN_2]]\% dentro de $\pm$2 h y 6/6 gates aprobados. \\
Límite & Es evidencia pública de capacidad, no precisión Kaleido, ROI ni despliegue. \\
\bottomrule\end{tabularx}

\subsection*{Antes de empezar}
\begin{itemize}
\item Abrir la presentación HTML a pantalla completa y dejar el dashboard preparado en otra pestaña.
\item Comprobar que el watermark \claim{} es visible y que el dashboard muestra la evidencia ETA AIS.
\item No empezar por JEPA. Empezar por el problema operativo y llegar a JEPA como resultado de I+D falsable.
\item Tener preparada la pregunta final: operación, responsables, muestra y fecha.
\end{itemize}

\clearpage
\section{Guion diapositiva a diapositiva}

\slidehead{1}{FlowTwin Predictive Operations}{0:45}
\cue{Objetivo.} Situar la propuesta y marcar desde el primer minuto la frontera de evidencia.

\cue{Qué decir.} ``Gracias por el tiempo. Hemos construido un MVP técnico de una capa
predictiva read-only. El demostrador principal estima la llegada de buques a una zona
portuaria a partir de AIS, muestra ventanas de incertidumbre y conserva toda la trazabilidad.
No venimos a presentar resultados de Kaleido: venimos a enseñar una idea ejecutada sobre
datos públicos y un protocolo para validarla de forma segura con vosotros.''

\avoid{``Ya predice vuestras operaciones'', ``gemelo autónomo'' o ``ahorro demostrado''.}

\transition{Primero quiero encajar la idea dentro de lo que Kaleido ya tiene.}

\slidehead{2}{Encaje en productos Kaleido}{1:15}
\cue{Objetivo.} Dejar claro que FlowTwin complementa productos existentes.

\cue{Qué decir.} ``Shipping Board y Freight Intelligence son el encaje inmediato para una
ETA explicable: anticipar llegada, detectar excepciones y priorizar qué revisar. Trace Port
aporta después la memoria de operaciones, turnos e incidencias; TWINPORTS, el estado físico
y espacial. FlowTwin no compite con esos productos: convierte sus eventos en una predicción
con intervalo, cutoff y razones.''

\cue{Pregunta opcional.} ``¿En qué pantalla toma hoy una decisión el responsable de turno
cuando una operación empieza a desviarse?''

\transition{Con ese encaje como restricción, construimos el MVP completo de extremo a extremo.}

\slidehead{3}{MVP y límites explícitos}{1:10}
\cue{Objetivo.} Resumir el alcance y convertir las invalidaciones en prueba de disciplina.

\cue{Qué decir.} ``Construimos dos demostradores alineados: ETA con AIS para Shipping Board
y Freight Intelligence, y proceso objeto-céntrico OCEL para Trace Port/TWINPORTS. JEPA queda
como línea de I+D sometida a ablations. Mantuvimos viajes y operaciones enteros en cada
partición, elegimos modelos en validación y reservamos un test futuro del 1 al 7 de febrero.
Los runs que no pasaron el protocolo se conservaron como auditoría y no entraron en la cifra.''

\avoid{Presentar M0--M7 como un despliegue productivo. Son hitos técnicos de smoke.}

\transition{Veamos exactamente qué datos y protocolo sostienen las cifras.}

\slidehead{4}{Dataset y protocolo}{1:20}
\cue{Objetivo.} Dar credibilidad sin confundir dominio público con dominio portuario.

\cue{Qué decir.} ``Procesamos [[AIS_SOURCE_FILES]] días de NOAA MarineCadastre AIS,
[[AIS_SOURCE_GB]] GB comprimidos. Una geofence define la llegada y cada prefijo usa solo lo
visible en ese instante. El split contiene [[AIS_TRAIN_TRIPS]] viajes de entrenamiento,
[[AIS_VALIDATION_TRIPS]] de validación y [[AIS_TEST_TRIPS]] de test futuro, con
[[AIS_TEST_PREFIXES]] predicciones en ese test. MMSI se excluyó como feature y ningún viaje
cruza particiones. El dataset OCEL secundario aporta [[OCEL_ALIGNED_OBJECTS]] contenedores.''

\cue{Frase clave.} ``Es una prueba técnica real de ETA portuaria en EE. UU., no una prueba de precisión en Vigo.''

\transition{Antes de entrenar nada, los eventos ya permiten inteligencia de proceso.}

\clearpage
\slidehead{5}{Process intelligence}{1:00}
\cue{Objetivo.} Mostrar valor temprano aunque ML no supere los gates.

\cue{Qué decir.} ``El ejemplo OCEL conserva contenedores, vehículos y revisiones de plan
como objetos distintos. El grafo correcto bajó el MAE de test de [[OCEL_ALIGNED_FLAT]] a
[[OCEL_ALIGNED_GRAPH]] horas, pero validación eligió la traza plana, así que cerramos el gate
predictivo. Aun así, el bloque ya permite mapas de proceso, variantes, esperas, rework y
conformance sin depender de que gane un modelo.''

\transition{En el demostrador de ETA sí obtuvimos un resultado concreto sobre futuro no visto.}

\slidehead{6}{Comparación de modelos}{1:40}
\cue{Objetivo.} Explicar el número principal, sus comparadores y el gate fijado antes del test.

\cue{Qué decir.} ``El modelo elegido solo en validación obtiene [[AIS_MAE]] horas de MAE
en [[AIS_TEST_TRIPS]] viajes futuros. El error mediano es [[AIS_MEDIAN_AE]] horas y el
bootstrap por viaje da un IC95\% de [[AIS_CI_LOW]] a [[AIS_CI_HIGH]]. No lo juzgamos solo:
la ETA distancia/velocidad tiene [[AIS_KINEMATIC]] horas y la mediana por puerto y distancia,
[[AIS_HISTORICAL]]. El modelo mejora un [[AIS_GAIN_KINEMATIC]]\% y un
[[AIS_GAIN_HISTORICAL]]\%, respectivamente.''

\cue{Traducción operativa.} ``El [[AIS_WITHIN_1]]\% queda dentro de $\pm$1 hora, el
[[AIS_WITHIN_2]]\% dentro de $\pm$2 y el [[AIS_WITHIN_4]]\% dentro de $\pm$4. Pasa los seis
gates predeclarados: volumen, MAE, extremo del intervalo, tolerancia y mejora contra dos
baselines.''

\avoid{Llamarlo precisión Kaleido, SOTA o prometer que $\pm$2 horas sea su tolerancia de negocio.}

\transition{Una predicción operativa no puede ser solo una cifra puntual.}

\slidehead{7}{Incertidumbre y riesgo}{1:15}
\cue{Objetivo.} Defender intervalos, abstención y semántica del target.

\cue{Qué decir.} ``Por horizonte, el MAE es [[AIS_MAE_0_2]] horas cuando faltan 0--2,
[[AIS_MAE_2_6]] entre 2 y 6, y [[AIS_MAE_6_12]] entre 6 y 12. El intervalo P90 cubre
[[AIS_P90_COVERAGE]]\%, pero mide [[AIS_P90_WIDTH]] horas de ancho medio: la cobertura se
consigue con una banda todavía demasiado amplia. Además, Nueva Orleans concentra
[[AIS_NOLA_SHARE]]\% de los prefijos de test.''

\cue{Conclusión.} ``La predicción puntual funciona como demostrador; el siguiente trabajo
falsable es calibrar por puerto, medir peor grupo y hacer que el sistema se abstenga.''

\transition{Con el baseline fijado, podemos preguntar qué está aprendiendo realmente JEPA.}

\slidehead{8}{Ablations de Event-JEPA}{2:00}
\cue{Objetivo.} Explicar por qué JEPA sigue siendo interesante aunque no gane el benchmark.

\cue{Qué decir.} ``La referencia multi-horizonte obtiene [[JEPA_MAE]]. El encoder aleatorio
queda en [[RANDOM_MAE]], por lo que el objetivo JEPA sí aporta representación downstream.
Sin SIGReg el MAE sube a [[NO_SIGREG_MAE]] y aparece colapso: la regularización es esencial.
Pero completion-only obtiene [[ONE_HORIZON_MAE]], ligeramente mejor, y los pares temporales
barajados quedan en [[SHUFFLED_PAIR_MAE]]. La diferencia temporal es pequeña e inconsistente
por seed. Todavía no hemos demostrado una dinámica futura multihorizonte rica.''

\cue{Mensaje de investigación.} ``JEPA aporta señal, pero cada parte debe ganarse su lugar
mediante una ablation que pueda falsarla.''

\transition{Por eso repetimos el experimento con mecanismos más cercanos a T-JEPA y Var-JEPA.}

\clearpage
\slidehead{9}{T-JEPA, Var-JEPA y el gate híbrido}{2:15}
\cue{Objetivo.} Explicar en lenguaje simple qué se cambió y por qué boosting sigue ganando.

\cue{Qué decir.} ``En la primera versión el target aún contenía el prefijo y compartía
encoder. En Temporal T-JEPA hicimos tres cambios con sentido temporal: el target contiene
solo eventos futuros, un teacher suavizado por EMA produce el target sin gradiente y elegimos
SiGReg, VISReg o token de registro únicamente en validación. Gana multi-VISReg con
[[TJEPA_MAE]] $\pm$ [[TJEPA_SD]]: mejora al Event-JEPA anterior, y el futuro correcto gana
al barajado en tres de tres semillas, pero no supera boosting ni demuestra ventaja
multihorizonte estable.''

\cue{Continuación.} ``Var-Event-JEPA sustituye el único punto latente por distribuciones:
contexto, una variable auxiliar y futuro tienen media y varianza; el ELBO combina
reconstrucción, generación y KL. Obtiene [[VAR_JEPA_MAE]] $\pm$ [[VAR_JEPA_SD]]. Su
incertidumbre latente tiene Spearman medio [[VAR_SPEARMAN]] con el error y falla en dos
semillas. Finalmente añadimos los embeddings al boosting: validación conserva raw. El mejor
híbrido elegido, raw+Var, obtiene [[HYBRID_MAE]] $\pm$ [[HYBRID_SD]], [[HYBRID_DELTA]]
minutos peor que raw.''

\cue{ELI5.} ``Boosting recibe directamente reloj, progreso, actividad y espera. JEPA intenta
resumir una película de pocos eventos; en este log el resumen comprime más información de la
que descubre. Con objetos, planes, contexto y acciones reales la película sería más rica,
pero eso hay que medir, no asumir.''

\avoid{Decir que se reprodujeron exactamente T-JEPA o Var-T-JEPA. Son adaptaciones temporales
CPU-budgeted y el resultado es \claim.}

\transition{La pregunta siguiente es si un canal de acción permite aprender transiciones condicionadas.}

\clearpage
\slidehead{10}{World model sintético con VISReg}{2:00}
\cue{Objetivo.} Mostrar la investigación de acciones sin convertirla en un claim causal.

\cue{Qué decir.} ``No renombramos actividades del log como acciones. Creamos un overlay
separado con acciones, elegibilidad, propensión, coste y efecto conocido. El primer
benchmark tabular y el Action-JEPA con SIGReg fallaron sus gates. Con VISReg, la acción
correcta obtiene [[ACTION_CORRECT]] $\pm$ [[ACTION_CORRECT_SD]] frente a
[[ACTION_SHUFFLED]] barajada: gana en tres de tres seeds, con [[ACTION_GAIN]] minutos de
mejora media. El rango efectivo sube a [[ACTION_RANK_LOW]]--[[ACTION_RANK_HIGH]], pero la
escala sigue en [[ACTION_SCALE_LOW]]--[[ACTION_SCALE_HIGH]], bajo el umbral. Recuperamos una
señal inyectada; no promocionamos el world model.''

\avoid{``La acción reduce 14,67 minutos en la realidad''. La magnitud pertenece al target generado.}

\transition{Esto conduce a una separación clara entre producto e investigación.}

\slidehead{11}{Decisión de producto}{1:00}
\cue{Objetivo.} Convertir los resultados en una arquitectura de producto prudente.

\cue{Qué decir.} ``La demo sirve ETA AIS, comparadores, tolerancias, intervalo y auditoría
read-only. OCEL demuestra process intelligence con identidad de objetos. Temporal T-JEPA,
Var-JEPA, acciones reales y object graphs permanecen como I+D con gates. El predictor
histórico de almacén con 734 minutos queda conservado como resultado negativo: no se sirve
ni se usa para vender la idea.''

\transition{La integración propuesta mantiene esa separación y no escribe en sistemas fuente.}

\slidehead{12}{Arquitectura complementaria}{1:10}
\cue{Objetivo.} Explicar el flujo read-only y la trazabilidad de cada predicción.

\cue{Qué decir.} ``AIS o Shipping Board aportan posición y contexto de viaje; Trace Port,
eventos y turnos; TWINPORTS, activos y espacio. El contrato temporal conserva planes
versionados. El baseline ETA, process mining y JEPA shadow producen punto, intervalo y
razones; la salida vuelve a una pestaña o API read-only. Cada resultado muestra cutoff,
versión y procedencia. El humano sigue decidiendo.''

\avoid{Hablar de control automático, escritura en TOS/ERP o optimización autónoma.}

\transition{La forma más clara de entenderlo es verlo en una interfaz de operación.}

\clearpage
\slidehead{13}{Demo del dashboard}{2:30}
\cue{Objetivo.} Enseñar cómo se consumiría el software, sin vender el fixture como Kaleido.

\cue{Secuencia de demo.}
\begin{enumerate}[leftmargin=6mm,itemsep=2pt]
\item Señalar el watermark \claim{} y el modo read-only.
\item Abrir la evidencia ETA: [[AIS_MAE]] h, IC95\% y [[AIS_TEST_TRIPS]] viajes futuros.
\item Comparar boosting con la ETA cinemática y la mediana puerto-distancia.
\item Mostrar los porcentajes dentro de $\pm$1, $\pm$2 y $\pm$4 horas y los 6/6 gates.
\item Abrir incertidumbre y límites: intervalo P90, reparto por puerto y abstención.
\item Recorrer procedencia: fechas de test, exclusión de MMSI, hashes y model card.
\item Terminar en la separación: ETA demostrable; OCEL diagnóstico; JEPA en I+D.
\end{enumerate}

\cue{Frase de cierre de demo.} ``La interfaz es el ejemplo de consumo; los valores son
sintéticos/públicos. En piloto, esta misma superficie se alimentaría de un export congelado.''

\transition{Para convertir la demo en evidencia Kaleido proponemos cuatro gates pequeños.}

\slidehead{14}{Piloto con Kaleido}{1:15}
\cue{Objetivo.} Pedir una acción concreta, no una aprobación abstracta.

\cue{Qué decir.} ``Primero validamos esquema y semántica sobre 3--5 casos. Después
congelamos planes, revisiones y outcomes. Ejecutamos un replay shadow durante 4--8 semanas,
con falsas alertas, lead time, cobertura, peor grupo y utilidad de operador. Solo entonces
decidimos integración, buyer, coste y packaging.''

\cue{Petición literal.} ``¿Podemos agendar una sesión de 90 minutos con una persona de
Trace Port y una persona de operaciones para elegir una operación y revisar 3--5 casos
pseudonimizados?''

\transition{La propuesta se resume en una sola frase.}

\slidehead{15}{Cierre}{0:35}
\cue{Qué decir.} ``Shipping Board informa. Freight Intelligence analiza. FlowTwin anticipa.
Ya existe una prueba de ETA sobre futuro no visto y un marco objeto-céntrico para procesos.
El siguiente paso no es prometer generalización: es acordar operación, tolerancia, muestra y
fecha para medirlo con datos Kaleido.''

\cue{Acción.} Callar y esperar respuesta. No rellenar el silencio con detalles técnicos.

\clearpage
\section{Preguntas previsibles}
\begin{tabularx}{\linewidth}{@{}>{\raggedright\arraybackslash\bfseries}p{4.2cm}X@{}}\toprule
¿Por qué JEPA si boosting gana? & Porque Temporal T-JEPA mejora al JEPA anterior y distingue
futuros correctos de barajados, pero los embeddings no añaden valor incremental al boosting.
Se mantiene shadow como hipótesis para datos con acciones, objetos y contexto más ricos. \\
¿Esto es un world model? & Es un experimento de transición latente condicionado por acciones
generadas. Recupera señal inyectada, pero no supera el gate de estabilidad y no tiene acciones Kaleido. \\
¿Qué hace falta de Kaleido? & IDs pseudonimizados, eventos con timestamps, plan original y
revisiones con \texttt{valid\_from}, outcomes, objetos y acciones realmente controlables. \\
¿Cuántos datos? & Primero 3--5 casos para semántica. El volumen de entrenamiento se decide
después de medir variantes, eventos por operación, censura y prevalencia del outcome. \\
¿Puede escribir o controlar equipos? & No. El MVP y la API son read-only y advisory. \\
¿Cuál es el ROI? & Todavía no se ha medido. En shadow se separan tiempo de reporting,
evitabilidad simulada, aceptación del operador y valor realizado. \\
¿Por qué el intervalo es tan ancho? & Porque refleja heterogeneidad e incertidumbre del dataset.
Se muestra para permitir abstención; no se estrecha de forma cosmética. \\
¿1,88 horas es bueno o malo? & Para este demostrador pasa los seis criterios fijados antes
del test y mejora dos comparadores. Además, [[AIS_WITHIN_2]]\% cae dentro de $\pm$2 h. No
sabemos si esa tolerancia sirve a Kaleido: debe acordarse por decisión y horizonte. \\
¿Qué ocurrió con los 734 minutos? & Era el MAE de otro dataset de almacén, unas 12,2 h.
Ganaba por poco a baselines débiles, pero la escala absoluta y los intervalos no servían
como demostrador. Se rechazó para producto y se conserva solo como evidencia auditable. \\
¿El test representa todos los puertos? & No. [[AIS_NOLA_SHARE]]\% de sus prefijos son de
Nueva Orleans; Houston y Los Ángeles tienen menos viajes y Nueva York no aparece en el test.
Por eso el siguiente gate es por puerto y, después, Vigo/Kaleido. \\
\bottomrule\end{tabularx}

\section{Versiones por tiempo}
\textbf{Si solo hay 10 minutos:} diapositivas 1, 2, 4, 6, 7, 11, 14 y 15. Omitir JEPA
o resumirlo como línea de I+D que todavía no gana el gate.

\textbf{Si hay 30 minutos:} recorrido completo, demo de 4--5 minutos y preguntas tras la
diapositiva 10 antes de cerrar con el piloto.

\section{Fuentes y recordatorio de claims}
\begin{itemize}
\item \href{https://coast.noaa.gov/htdata/CMSP/AISDataHandler/2025/}{NOAA MarineCadastre AIS 2025}.
\item \href{https://doi.org/10.5281/zenodo.18373888}{OCEL 2.0 Container Logistics}.
\item \href{https://www.kaleidologistics.com/en/productos-ktech/shipping-board/}{Kaleido Shipping Board} y
\href{https://www.kaleidologistics.com/en/productos-ktech/freight-intelligence/}{Freight Intelligence}.
\item \href{https://www.kaleidologistics.com/en/productos-ktech/trace-port/}{Kaleido Trace Port}.
\item \href{https://www.kaleidologistics.com/es/proyectos-financiados/el-proyecto-twinports-despega-en-el-puerto-de-vigo-comienzan-los-trabajos-para-crear-el-gemelo-digital-de-la-terminal-portuaria-de-kaleido/}{Kaleido TWINPORTS}.
\item \href{https://arxiv.org/abs/2511.08544}{LeJEPA};
\href{https://arxiv.org/abs/2603.19312}{LeWorldModel};
\href{https://arxiv.org/abs/2606.02572}{VISReg};
\href{https://arxiv.org/abs/2410.05016}{T-JEPA};
\href{https://arxiv.org/abs/2603.20111}{Var-JEPA}.
\end{itemize}

\begin{center}
\fcolorbox{Red}{Red!8}{\parbox{.92\linewidth}{
\textbf{Recordatorio final.} Dataset/export y hash, split, modelos/baselines, métrica e
incertidumbre, seeds, selección de umbral, influencia de test y claim state deben acompañar
cualquier resultado. Los datos públicos o sintéticos no prueban valor Kaleido.
}}
\end{center}
\end{document}'''
    path.write_text(_render(template, _values(summary)), encoding="utf-8")


def write_result_tables(output_dir: Path, summary: dict[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    ais = summary["aligned_public_benchmarks"]["ais_eta"]
    ais_rows = [
        (
            model,
            values["mae"],
            values["median_absolute_error"],
            values["within_tolerance"]["within_2"],
            "selected_validation_only" if model == ais["selected_model_validation_only"] else "comparator",
        )
        for model, values in ais["test_metrics"].items()
    ]
    with (output_dir / "model_comparison.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["model", "test_mae_hours", "test_median_ae_hours", "within_2h", "decision"]
        )
        writer.writerows(ais_rows)

    remaining = summary["remaining_time"]
    legacy_rows = [
        ("quantile_boosting", 1, remaining["boosting"]["mae_minutes"], "validation_mae", "serve"),
        ("raw_boosting_rerun", 3, remaining["jepa_hybrid"]["aggregates"]["raw"]["mae_mean_minutes"], "validation_mae", "serve"),
        ("raw_plus_var_jepa", 3, remaining["jepa_hybrid"]["aggregates"]["raw_var_jepa"]["mae_mean_minutes"], "validation_mae", "reject_vs_raw"),
        ("temporal_t_jepa", 3, remaining["temporal_t_jepa"]["mae_mean_minutes"], "validation_pinball", "research_shadow"),
        ("var_event_jepa", 3, remaining["var_event_jepa"]["mae_mean_minutes"], "validation_pinball", "research_shadow"),
        ("gru", 3, remaining["gru"]["mae_mean_minutes"], "validation_pinball", "reject_vs_floor"),
        ("process_transformer", 3, remaining["transformer"]["mae_mean_minutes"], "validation_pinball", "reject_vs_floor"),
        ("event_jepa_frozen", 3, remaining["event_jepa"]["mae_mean_minutes"], "validation_pinball", "research_shadow"),
    ]
    with (output_dir / "legacy_remaining_time_comparison.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(["model", "seeds", "test_mae_minutes", "selection", "decision"])
        writer.writerows(legacy_rows)
    action = summary["synthetic_actions"]["action_jepa_visreg"]["aggregates"]
    with (output_dir / "synthetic_action_comparison.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(["mode", "seeds", "test_mae_minutes", "scope"])
        for mode, values in action.items():
            writer.writerow(
                [mode, 3, values["mae_mean_minutes"], "synthetic_injected_transition_only"]
            )


def _write_package_manifest(repository_root: Path, summary: dict[str, Any]) -> None:
    presentation_path = repository_root / "presentacion"
    pdf_path = repository_root / "output/pdf"
    evidence_path = repository_root / "outputs/final_evidence"
    assets = presentation_path / "assets"
    candidates = (
        presentation_path / "Kaleido_FlowTwin_Presentacion.html",
        presentation_path / "Kaleido_FlowTwin_Presentacion.tex",
        presentation_path / "Kaleido_FlowTwin_Presentacion.pdf",
        pdf_path / "Kaleido_FlowTwin_MVP_Informe_Tecnico.tex",
        pdf_path / "Kaleido_FlowTwin_MVP_Informe_Tecnico.pdf",
        pdf_path / "Kaleido_FlowTwin_Guion_Presentacion.tex",
        pdf_path / "Kaleido_FlowTwin_Guion_Presentacion.pdf",
        assets / "model_comparison.pdf",
        assets / "model_comparison.svg",
        assets / "jepa_ablations.pdf",
        assets / "jepa_ablations.svg",
        assets / "action_recovery.pdf",
        assets / "action_recovery.svg",
        evidence_path / "summary.json",
        evidence_path / "model_comparison.csv",
        evidence_path / "legacy_remaining_time_comparison.csv",
        evidence_path / "synthetic_action_comparison.csv",
    )
    atomic_json(
        evidence_path / "package_manifest.json",
        {
            "claim_state": "smoke_only",
            "generated_on": summary["generated_on"],
            "sources": summary["provenance"],
            "generated_files": {
                str(path.relative_to(repository_root)): sha256_file(path)
                for path in candidates
                if path.is_file()
            },
        },
    )


def finalize_final_package(repository_root: Path = Path(".")) -> dict[str, Any]:
    summary = build_summary(repository_root)
    presentation = repository_root / "presentacion/Kaleido_FlowTwin_Presentacion.pdf"
    presentation_source = repository_root / "presentacion/Kaleido_FlowTwin_Presentacion.tex"
    report = repository_root / "output/pdf/Kaleido_FlowTwin_MVP_Informe_Tecnico.pdf"
    report_source = repository_root / "output/pdf/Kaleido_FlowTwin_MVP_Informe_Tecnico.tex"
    script = repository_root / "output/pdf/Kaleido_FlowTwin_Guion_Presentacion.pdf"
    script_source = repository_root / "output/pdf/Kaleido_FlowTwin_Guion_Presentacion.tex"
    for pdf, source in (
        (presentation, presentation_source),
        (report, report_source),
        (script, script_source),
    ):
        if not pdf.is_file() or not source.is_file():
            raise RuntimeError(f"missing compiled deliverable: {pdf}")
        if pdf.stat().st_mtime_ns < source.stat().st_mtime_ns:
            raise RuntimeError(f"compiled PDF is older than its source: {pdf}")
    _write_package_manifest(repository_root, summary)
    return summary


def build_final_package(
    repository_root: Path = Path("."),
    *,
    presentation_dir: Path = Path("presentacion"),
    pdf_dir: Path = Path("output/pdf"),
    evidence_dir: Path = Path("outputs/final_evidence"),
) -> dict[str, Any]:
    summary = build_summary(repository_root)
    presentation_path = repository_root / presentation_dir
    pdf_path = repository_root / pdf_dir
    evidence_path = repository_root / evidence_dir
    assets = presentation_path / "assets"
    presentation_path.mkdir(parents=True, exist_ok=True)
    pdf_path.mkdir(parents=True, exist_ok=True)
    evidence_path.mkdir(parents=True, exist_ok=True)
    write_charts(summary, assets)
    write_presentation_html(
        presentation_path / "Kaleido_FlowTwin_Presentacion.html", summary
    )
    write_presentation_tex(
        presentation_path / "Kaleido_FlowTwin_Presentacion.tex", summary
    )
    write_technical_tex(pdf_path / "Kaleido_FlowTwin_MVP_Informe_Tecnico.tex", summary)
    write_speaker_script_tex(
        pdf_path / "Kaleido_FlowTwin_Guion_Presentacion.tex", summary
    )
    write_result_tables(evidence_path, summary)
    atomic_json(evidence_path / "summary.json", summary)
    _write_package_manifest(repository_root, summary)
    return summary
