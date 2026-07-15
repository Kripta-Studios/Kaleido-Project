from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import pandas as pd


@dataclass
class RegisteredModel:
    version: str
    run_dir: Path
    metrics: dict[str, Any]
    remaining_model: Any
    risk_model: Any
    conformal: Any


class ModelRegistry:
    def __init__(self, artifact_root: Path) -> None:
        self.artifact_root = artifact_root

    def latest(self) -> RegisteredModel | None:
        candidates = sorted(
            (
                path
                for path in self.artifact_root.glob("*")
                if path.is_dir() and (path / "metrics.json").is_file()
            ),
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )
        for run_dir in candidates:
            metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
            selected = metrics.get("model_selection", {}).get("selected_model")
            model_path = run_dir / "models" / f"{selected}.joblib"
            risk_path = run_dir / "models" / "long_duration_risk_logistic.joblib"
            conformal_path = run_dir / "models" / "conformal.joblib"
            if not all(path.is_file() for path in (model_path, risk_path, conformal_path)):
                continue
            return RegisteredModel(
                version=run_dir.name,
                run_dir=run_dir,
                metrics=metrics,
                remaining_model=joblib.load(model_path),
                risk_model=joblib.load(risk_path),
                conformal=joblib.load(conformal_path),
            )
        return None

    @staticmethod
    def feature_row(
        *,
        activity: str,
        elapsed_minutes: float,
        since_previous_minutes: float,
        prefix_events: int,
        hour_utc: int,
        weekday_utc: int,
    ) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    "elapsed_minutes": elapsed_minutes,
                    "since_previous_minutes": since_previous_minutes,
                    "prefix_events": float(prefix_events),
                    "hour_utc": float(hour_utc),
                    "weekday_utc": float(weekday_utc),
                    "last_activity": activity,
                }
            ]
        )
