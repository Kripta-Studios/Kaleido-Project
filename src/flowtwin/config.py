from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator


class ComputeConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    device: Literal["cpu", "cuda"] = "cpu"
    max_cases: int | None = Field(default=None, gt=0)


class ExperimentConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    experiment_name: str
    hypothesis: str
    dataset_manifest: str
    schema_version: str
    split_manifest: str
    seed: int
    prediction_points: list[float]
    horizons: list[str]
    input_roles: list[str]
    action_fields: list[str]
    context_fields: list[str]
    observation_fields: list[str]
    outcome_fields: list[str]
    forbidden_fields: list[str]
    model: str
    baselines: list[str]
    metrics: list[str]
    threshold_selection: str
    calibration: str
    compute: ComputeConfig
    claim_state: Literal[
        "planned",
        "smoke_only",
        "diagnostic",
        "claim_eligible",
        "pilot_shadow",
        "validated_pilot",
    ]

    @model_validator(mode="after")
    def roles_are_disjoint(self) -> ExperimentConfig:
        role_sets = [
            set(self.action_fields),
            set(self.context_fields),
            set(self.observation_fields),
            set(self.outcome_fields),
            set(self.forbidden_fields),
        ]
        for index, left in enumerate(role_sets):
            for right in role_sets[index + 1 :]:
                overlap = left & right
                if overlap:
                    raise ValueError(f"fields assigned to multiple roles: {sorted(overlap)}")
        return self


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = yaml.safe_load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a YAML mapping")
    return value


def load_experiment(path: Path) -> ExperimentConfig:
    return ExperimentConfig.model_validate(load_yaml(path))
