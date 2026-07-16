from __future__ import annotations

from typing import Any

from flowtwin.baselines.process_transformer import require_torch


def build_quantile_head(latent_size: int) -> Any:
    torch = require_torch()
    nn = torch.nn

    class QuantileHead(nn.Module):  # type: ignore[name-defined,misc]
        def __init__(self) -> None:
            super().__init__()
            self.network = nn.Sequential(
                nn.Linear(latent_size, latent_size),
                nn.GELU(),
                nn.Linear(latent_size, 2),
            )

        def forward(self, latent: Any) -> Any:
            raw = self.network(latent)
            p50 = torch.nn.functional.softplus(raw[:, 0])
            p90 = p50 + torch.nn.functional.softplus(raw[:, 1])
            return torch.stack([p50, p90], dim=1)

    return QuantileHead()
