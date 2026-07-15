from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor, HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def preprocessing(numeric: list[str], categorical: list[str]) -> ColumnTransformer:
    numeric_pipeline = Pipeline(
        [("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]
    )
    categorical_pipeline = Pipeline(
        [
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", min_frequency=5)),
        ]
    )
    return ColumnTransformer(
        [("numeric", numeric_pipeline, numeric), ("categorical", categorical_pipeline, categorical)]
    )


def ridge_remaining_pipeline(numeric: list[str], categorical: list[str]) -> Pipeline:
    return Pipeline(
        [
            ("features", preprocessing(numeric, categorical)),
            ("model", Ridge(alpha=10.0)),
        ]
    )


def quantile_boosting_pipeline(
    numeric: list[str],
    categorical: list[str],
    *,
    quantile: float,
    seed: int,
    estimators: int = 120,
) -> Pipeline:
    return Pipeline(
        [
            ("features", preprocessing(numeric, categorical)),
            (
                "model",
                GradientBoostingRegressor(
                    loss="quantile",
                    alpha=quantile,
                    n_estimators=estimators,
                    max_depth=3,
                    min_samples_leaf=20,
                    learning_rate=0.05,
                    random_state=seed,
                ),
            ),
        ]
    )


def logistic_risk_pipeline(numeric: list[str], categorical: list[str]) -> Pipeline:
    return Pipeline(
        [
            ("features", preprocessing(numeric, categorical)),
            (
                "model",
                LogisticRegression(
                    penalty="l2",
                    C=0.5,
                    class_weight="balanced",
                    max_iter=1000,
                ),
            ),
        ]
    )


@dataclass
class ConformalIntervals:
    residual_q50: float = 0.0
    residual_q90: float = 0.0

    def fit(self, target: np.ndarray, prediction: np.ndarray) -> ConformalIntervals:
        residuals = np.abs(np.asarray(target, dtype=float) - np.asarray(prediction, dtype=float))
        if residuals.size == 0:
            raise ValueError("validation residuals are empty")
        self.residual_q50 = float(np.quantile(residuals, 0.5, method="higher"))
        self.residual_q90 = float(np.quantile(residuals, 0.9, method="higher"))
        return self

    def interval(self, prediction: np.ndarray, coverage: float) -> tuple[np.ndarray, np.ndarray]:
        radius = self.residual_q50 if coverage == 0.5 else self.residual_q90
        values = np.asarray(prediction, dtype=float)
        return np.maximum(0.0, values - radius), values + radius


class DenseRiskBoosting:
    """Small numeric risk baseline used after explicit encoding."""

    def __init__(self, seed: int = 42) -> None:
        self.model = HistGradientBoostingClassifier(
            learning_rate=0.05,
            max_iter=150,
            max_leaf_nodes=15,
            min_samples_leaf=20,
            random_state=seed,
        )

    def fit(self, features: np.ndarray, target: np.ndarray) -> DenseRiskBoosting:
        self.model.fit(features, target)
        return self

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        result: np.ndarray = self.model.predict_proba(features)[:, 1]
        return result
