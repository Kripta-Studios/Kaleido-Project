from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any, cast

import yaml

from flowtwin.benchmarks.ais_world_model import (
    _train_jepa,
    build_ais_trajectory_data,
)
from flowtwin.models.ais_phys_jepa import AISWorldModelConfig
from flowtwin.provenance import atomic_json


def main() -> None:
    config_path = Path("configs/experiment/noaa_ais_phys_jepa_development.yaml")
    resolved = cast(
        dict[str, Any], yaml.safe_load(config_path.read_text(encoding="utf-8"))
    )
    data = build_ais_trajectory_data(
        Path("outputs/noaa_ais_eta_v3/prefixes.parquet"), resolved
    )
    compute = copy.deepcopy(cast(dict[str, Any], resolved["compute"]))
    compute["pretrain_epochs"] = 40
    candidates: list[dict[str, Any]] = [
        {
            "name": "fw2_rw05",
            "forecast_weight": 2.0,
            "regularizer_weight": 0.05,
            "hidden_size": 64,
            "latent_size": 32,
        },
        {
            "name": "fw5_rw05",
            "forecast_weight": 5.0,
            "regularizer_weight": 0.05,
            "hidden_size": 64,
            "latent_size": 32,
        },
        {
            "name": "fw10_rw05",
            "forecast_weight": 10.0,
            "regularizer_weight": 0.05,
            "hidden_size": 64,
            "latent_size": 32,
        },
        {
            "name": "fw5_rw01",
            "forecast_weight": 5.0,
            "regularizer_weight": 0.01,
            "hidden_size": 64,
            "latent_size": 32,
        },
        {
            "name": "fw10_rw01",
            "forecast_weight": 10.0,
            "regularizer_weight": 0.01,
            "hidden_size": 64,
            "latent_size": 32,
        },
        {
            "name": "fw10_rw001",
            "forecast_weight": 10.0,
            "regularizer_weight": 0.001,
            "hidden_size": 64,
            "latent_size": 32,
        },
        {
            "name": "wide_fw10_rw01",
            "forecast_weight": 10.0,
            "regularizer_weight": 0.01,
            "hidden_size": 128,
            "latent_size": 64,
        },
    ]
    output_dir = Path("outputs/noaa_ais_phys_jepa_development_candidates_v1")
    output_dir.mkdir(parents=True, exist_ok=True)
    results: list[dict[str, Any]] = []
    for specification in candidates:
        model_config = AISWorldModelConfig(
            max_length=8,
            horizon_count=3,
            hidden_size=int(specification["hidden_size"]),
            latent_size=int(specification["latent_size"]),
            layers=2,
            attention_heads=4,
            dropout=0.0,
            regularizer_slices=32,
            regularizer_weight=float(specification["regularizer_weight"]),
            forecast_weight=float(specification["forecast_weight"]),
        )
        result, _, _ = _train_jepa(
            "phys_visreg",
            data,
            model_config,
            compute,
            seed=42,
            checkpoint_dir=output_dir / str(specification["name"]),
        )
        row = {
            **specification,
            "validation_distance_mae_km": result["validation_distance_mae_km"],
            "transition_alignment_validation": result[
                "transition_alignment_validation"
            ],
            "embedding": result["embedding_diagnostics_validation"],
        }
        results.append(row)
        print(json.dumps(row), flush=True)
    selected = min(results, key=lambda item: item["validation_distance_mae_km"])
    atomic_json(
        output_dir / "validation_candidates.json",
        {
            "selection": "validation_distance_mae_km",
            "test_used_for_selection": False,
            "results": results,
            "selected": selected,
        },
    )


if __name__ == "__main__":
    main()
