from __future__ import annotations

from typing import Any

from flowtwin.baselines.process_transformer import require_torch


def build_latent_predictor(latent_size: int, horizon_count: int) -> Any:
    torch = require_torch()
    nn = torch.nn

    class LatentPredictor(nn.Module):  # type: ignore[name-defined,misc]
        def __init__(self) -> None:
            super().__init__()
            self.horizon = nn.Embedding(horizon_count, latent_size)
            self.network = nn.Sequential(
                nn.Linear(latent_size * 2, latent_size * 2),
                nn.GELU(),
                nn.LayerNorm(latent_size * 2),
                nn.Linear(latent_size * 2, latent_size),
            )

        def forward(self, context: Any, horizon_ids: Any) -> Any:
            horizon = self.horizon(horizon_ids)
            return self.network(torch.cat([context, horizon], dim=-1))

    return LatentPredictor()
