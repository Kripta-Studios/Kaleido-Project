from __future__ import annotations

import copy
from typing import Any

from flowtwin.baselines.process_transformer import require_torch
from flowtwin.models.contracts import EventJEPAConfig
from flowtwin.models.event_jepa import sigreg_loss, visreg_loss
from flowtwin.models.predictor import build_latent_predictor


def build_temporal_encoder(config: EventJEPAConfig, *, register_token: bool) -> Any:
    """Build an event encoder whose optional register participates in attention only."""

    torch = require_torch()
    nn = torch.nn

    class TemporalEncoder(nn.Module):  # type: ignore[name-defined,misc]
        def __init__(self) -> None:
            super().__init__()
            self.activity = nn.Embedding(
                config.vocabulary_size,
                config.hidden_size,
                padding_idx=0,
            )
            self.position = nn.Embedding(config.max_length + 1, config.hidden_size)
            self.register = (
                nn.Parameter(torch.zeros(1, 1, config.hidden_size))
                if register_token
                else None
            )
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
            if self.register is not None:
                register = self.register.expand(batch, -1, -1) + self.position(
                    torch.full((batch, 1), sequence, device=tokens.device)
                )
                values = torch.cat([values, register], dim=1)
                padding_mask = torch.cat(
                    [
                        padding_mask,
                        torch.zeros((batch, 1), dtype=torch.bool, device=tokens.device),
                    ],
                    dim=1,
                )
            values = self.backbone(values, src_key_padding_mask=padding_mask)
            indices = (lengths - 1).clamp(min=0, max=sequence - 1)
            last = values[torch.arange(batch, device=tokens.device), indices]
            return self.projector(torch.cat([self.norm(last), numeric], dim=1))

    return TemporalEncoder()


def build_temporal_t_jepa(
    config: EventJEPAConfig,
    *,
    ema_momentum: float,
    register_token: bool,
) -> Any:
    """Build a temporal JEPA with an EMA target encoder and stopped target gradients."""

    torch = require_torch()
    nn = torch.nn

    class TemporalTJEPA(nn.Module):  # type: ignore[name-defined,misc]
        def __init__(self) -> None:
            super().__init__()
            self.context_encoder = build_temporal_encoder(
                config,
                register_token=register_token,
            )
            self.target_encoder = copy.deepcopy(self.context_encoder)
            for parameter in self.target_encoder.parameters():
                parameter.requires_grad_(False)
            self.predictor = build_latent_predictor(
                config.latent_size,
                config.horizon_count,
            )

        def train(self, mode: bool = True) -> Any:
            super().train(mode)
            self.target_encoder.eval()
            return self

        def encode(self, tokens: Any, lengths: Any, numeric: Any) -> Any:
            return self.context_encoder(tokens, lengths, numeric)

        def update_target(self) -> None:
            with torch.no_grad():
                for target, context in zip(
                    self.target_encoder.parameters(),
                    self.context_encoder.parameters(),
                    strict=True,
                ):
                    target.data.mul_(ema_momentum).add_(
                        context.data,
                        alpha=1.0 - ema_momentum,
                    )

        def forward(
            self,
            context_tokens: Any,
            context_lengths: Any,
            context_numeric: Any,
            target_tokens: Any,
            target_lengths: Any,
            target_numeric: Any,
        ) -> tuple[Any, Any, Any]:
            batch, horizons, sequence = target_tokens.shape
            context = self.context_encoder(
                context_tokens,
                context_lengths,
                context_numeric,
            )
            with torch.no_grad():
                targets = self.target_encoder(
                    target_tokens.reshape(batch * horizons, sequence),
                    target_lengths.reshape(batch * horizons),
                    target_numeric.reshape(batch * horizons, 3),
                ).reshape(batch, horizons, config.latent_size)
            horizon_ids = torch.arange(horizons, device=context.device).expand(batch, horizons)
            predictions = self.predictor(
                context.unsqueeze(1).expand(batch, horizons, config.latent_size),
                horizon_ids,
            )
            return context, targets, predictions

    return TemporalTJEPA()


def temporal_t_jepa_loss(
    context: Any,
    targets: Any,
    predictions: Any,
    *,
    config: EventJEPAConfig,
    step: int,
    regularizer: str,
) -> tuple[Any, dict[str, float]]:
    torch = require_torch()
    alignment = torch.nn.functional.mse_loss(predictions, targets)
    penalty = alignment.new_zeros(())
    if regularizer == "sigreg":
        penalty = sigreg_loss(context, num_slices=config.sigreg_slices, seed=step)
    elif regularizer == "visreg":
        penalty = visreg_loss(context, num_slices=config.sigreg_slices, seed=step)
    elif regularizer != "none":
        raise ValueError(f"unsupported Temporal T-JEPA regularizer: {regularizer}")
    total = alignment + config.sigreg_weight * penalty
    return total, {
        "alignment": float(alignment.detach()),
        regularizer: float(penalty.detach()),
        "total": float(total.detach()),
    }
