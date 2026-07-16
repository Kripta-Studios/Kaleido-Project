from __future__ import annotations

from typing import Any

import numpy as np


def embedding_diagnostics(values: Any) -> dict[str, float | bool]:
    embeddings = np.asarray(values, dtype=float)
    if embeddings.ndim != 2 or embeddings.shape[0] < 2:
        raise ValueError("embedding diagnostics require a two-dimensional sample")
    centered = embeddings - embeddings.mean(axis=0, keepdims=True)
    standard_deviation = centered.std(axis=0, ddof=1)
    covariance = np.cov(centered, rowvar=False)
    eigenvalues = np.linalg.eigvalsh(covariance).clip(min=0)
    total = float(eigenvalues.sum())
    probabilities = eigenvalues / max(total, 1e-12)
    nonzero = probabilities[probabilities > 1e-12]
    effective_rank = float(np.exp(-(nonzero * np.log(nonzero)).sum()))
    participation_ratio = float(total**2 / max(float(np.square(eigenvalues).sum()), 1e-12))
    identity = np.eye(covariance.shape[0])
    isotropy_error = float(np.linalg.norm(covariance - identity, ord="fro"))
    collapsed = bool(standard_deviation.mean() < 0.05 or effective_rank < 2.0)
    return {
        "mean_dimension_std": float(standard_deviation.mean()),
        "min_dimension_std": float(standard_deviation.min()),
        "effective_rank": effective_rank,
        "participation_ratio": participation_ratio,
        "covariance_isotropy_frobenius": isotropy_error,
        "collapsed": collapsed,
    }
