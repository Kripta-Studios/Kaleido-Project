from __future__ import annotations

import polars as pl


def join_published_context(
    prefixes: pl.DataFrame,
    context: pl.DataFrame,
    *,
    by: str,
    prefix_time: str = "prediction_cutoff",
    published_time: str = "published_at",
) -> pl.DataFrame:
    """As-of join that never exposes context published after the prediction cutoff."""
    left = prefixes.sort([by, prefix_time])
    right = context.sort([by, published_time])
    result = left.join_asof(
        right,
        left_on=prefix_time,
        right_on=published_time,
        by=by,
        strategy="backward",
    )
    violation = result.filter(pl.col(published_time) > pl.col(prefix_time)).height
    if violation:
        raise RuntimeError("external context join exposed future publication")
    return result
