from __future__ import annotations

from typing import Any


def update_ema_target(target_encoder: Any, context_encoder: Any, momentum: float) -> None:
    """EMA utility reserved for the teacher-target ablation."""

    for target_parameter, context_parameter in zip(
        target_encoder.parameters(),
        context_encoder.parameters(),
        strict=True,
    ):
        target_parameter.data.mul_(momentum).add_(
            context_parameter.data,
            alpha=1.0 - momentum,
        )
