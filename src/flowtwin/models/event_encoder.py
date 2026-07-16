from __future__ import annotations

from typing import Any

from flowtwin.baselines.process_transformer import require_torch
from flowtwin.models.contracts import EventJEPAConfig


def build_event_encoder(config: EventJEPAConfig) -> Any:
    """Build a causal-prefix encoder without importing torch in base installs."""

    torch = require_torch()
    nn = torch.nn

    class EventEncoder(nn.Module):  # type: ignore[name-defined,misc]
        def __init__(self) -> None:
            super().__init__()
            self.activity = nn.Embedding(
                config.vocabulary_size,
                config.hidden_size,
                padding_idx=0,
            )
            self.position = nn.Embedding(config.max_length, config.hidden_size)
            layer = nn.TransformerEncoderLayer(
                d_model=config.hidden_size,
                nhead=config.attention_heads,
                dim_feedforward=config.hidden_size * 4,
                dropout=config.dropout,
                batch_first=True,
                norm_first=True,
            )
            self.backbone = nn.TransformerEncoder(layer, config.layers)
            self.norm = nn.LayerNorm(config.hidden_size)
            self.projector = nn.Sequential(
                nn.Linear(config.hidden_size + 3, config.hidden_size),
                nn.GELU(),
                nn.Linear(config.hidden_size, config.latent_size),
            )

        def forward(self, tokens: Any, lengths: Any, numeric: Any) -> Any:
            batch, sequence = tokens.shape
            positions = torch.arange(sequence, device=tokens.device).expand(batch, sequence)
            values = self.activity(tokens) + self.position(positions)
            padding_mask = tokens.eq(0)
            values = self.backbone(values, src_key_padding_mask=padding_mask)
            indices = (lengths - 1).clamp(min=0)
            last = values[torch.arange(batch, device=tokens.device), indices]
            return self.projector(torch.cat([self.norm(last), numeric], dim=1))

    return EventEncoder()
