from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any

from flowtwin.baselines.process_transformer import require_torch
from flowtwin.models.event_jepa import anticollapse_loss


@dataclass(frozen=True)
class DispatchWorldJEPAConfig:
    vocabulary_size: int
    type_vocabulary_size: int
    max_length: int = 32
    max_action_length: int = 4
    event_numeric_size: int = 5
    action_numeric_size: int = 4
    hidden_size: int = 48
    latent_size: int = 32
    layers: int = 1
    attention_heads: int = 4
    dropout: float = 0.0
    horizon_count: int = 3
    regularizer_slices: int = 32
    regularizer_weight: float = 0.05


def _masked_mean(values: Any, mask: Any) -> Any:
    weights = mask.unsqueeze(-1).to(values.dtype)
    return (values * weights).sum(dim=-2) / weights.sum(dim=-2).clamp_min(1.0)


def build_dispatch_state_encoder(config: DispatchWorldJEPAConfig) -> Any:
    """Encode a delivered-task prefix without importing torch in base installs."""

    torch = require_torch()
    nn = torch.nn

    class DispatchStateEncoder(nn.Module):  # type: ignore[name-defined,misc]
        def __init__(self) -> None:
            super().__init__()
            self.aoi = nn.Embedding(
                config.vocabulary_size,
                config.hidden_size,
                padding_idx=0,
            )
            self.aoi_type = nn.Embedding(
                config.type_vocabulary_size,
                config.hidden_size,
                padding_idx=0,
            )
            self.numeric = nn.Linear(config.event_numeric_size, config.hidden_size)
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
                nn.Linear(config.hidden_size * 2, config.hidden_size),
                nn.GELU(),
                nn.Linear(config.hidden_size, config.latent_size),
            )

        def forward(
            self,
            tokens: Any,
            type_tokens: Any,
            numeric: Any,
            lengths: Any,
        ) -> Any:
            batch, sequence = tokens.shape
            positions = torch.arange(sequence, device=tokens.device).expand(batch, sequence)
            values = (
                self.aoi(tokens)
                + self.aoi_type(type_tokens)
                + self.numeric(numeric)
                + self.position(positions)
            )
            padding_mask = tokens.eq(0)
            values = self.backbone(values, src_key_padding_mask=padding_mask)
            indices = (lengths - 1).clamp(min=0, max=sequence - 1)
            last = values[torch.arange(batch, device=tokens.device), indices]
            pooled = _masked_mean(values, ~padding_mask)
            return self.projector(
                torch.cat([self.norm(last), self.norm(pooled)], dim=-1)
            )

    return DispatchStateEncoder()


def build_action_plan_encoder(config: DispatchWorldJEPAConfig) -> Any:
    """Encode the timestamp-valid next-task action prefix for each horizon."""

    torch = require_torch()
    nn = torch.nn

    class ActionPlanEncoder(nn.Module):  # type: ignore[name-defined,misc]
        def __init__(self) -> None:
            super().__init__()
            self.aoi = nn.Embedding(
                config.vocabulary_size,
                config.hidden_size,
                padding_idx=0,
            )
            self.aoi_type = nn.Embedding(
                config.type_vocabulary_size,
                config.hidden_size,
                padding_idx=0,
            )
            self.numeric = nn.Linear(config.action_numeric_size, config.hidden_size)
            self.position = nn.Embedding(config.max_action_length, config.hidden_size)
            self.network = nn.Sequential(
                nn.LayerNorm(config.hidden_size),
                nn.Linear(config.hidden_size, config.latent_size),
                nn.GELU(),
                nn.Linear(config.latent_size, config.latent_size),
            )

        def forward(
            self,
            tokens: Any,
            type_tokens: Any,
            numeric: Any,
            lengths: Any,
        ) -> Any:
            batch, horizons, sequence = tokens.shape
            flat_tokens = tokens.reshape(batch * horizons, sequence)
            flat_types = type_tokens.reshape(batch * horizons, sequence)
            flat_numeric = numeric.reshape(
                batch * horizons,
                sequence,
                config.action_numeric_size,
            )
            positions = torch.arange(sequence, device=tokens.device).expand(
                batch * horizons,
                sequence,
            )
            values = (
                self.aoi(flat_tokens)
                + self.aoi_type(flat_types)
                + self.numeric(flat_numeric)
                + self.position(positions)
            )
            pooled = _masked_mean(values, flat_tokens.ne(0))
            encoded = self.network(pooled)
            active = lengths.reshape(batch * horizons).gt(0).unsqueeze(-1)
            encoded = torch.where(active, encoded, torch.zeros_like(encoded))
            return encoded.reshape(batch, horizons, config.latent_size)

    return ActionPlanEncoder()


def build_dispatch_world_jepa(
    config: DispatchWorldJEPAConfig,
    *,
    ema_momentum: float,
) -> Any:
    """Build an action-conditioned temporal JEPA for dispatch-state transitions."""

    torch = require_torch()
    nn = torch.nn

    class DispatchWorldJEPA(nn.Module):  # type: ignore[name-defined,misc]
        def __init__(self) -> None:
            super().__init__()
            self.context_encoder = build_dispatch_state_encoder(config)
            self.target_encoder = copy.deepcopy(self.context_encoder)
            for parameter in self.target_encoder.parameters():
                parameter.requires_grad_(False)
            self.action_encoder = build_action_plan_encoder(config)
            self.horizon = nn.Embedding(config.horizon_count, config.latent_size)
            self.predictor = nn.Sequential(
                nn.Linear(config.latent_size * 3, config.latent_size * 2),
                nn.GELU(),
                nn.LayerNorm(config.latent_size * 2),
                nn.Linear(config.latent_size * 2, config.latent_size),
            )

        def train(self, mode: bool = True) -> Any:
            super().train(mode)
            self.target_encoder.eval()
            return self

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

        def encode_state(
            self,
            tokens: Any,
            type_tokens: Any,
            numeric: Any,
            lengths: Any,
        ) -> Any:
            return self.context_encoder(tokens, type_tokens, numeric, lengths)

        def predict_future(
            self,
            state: Any,
            action_tokens: Any,
            action_type_tokens: Any,
            action_numeric: Any,
            action_lengths: Any,
        ) -> Any:
            batch, horizons, _ = action_tokens.shape
            actions = self.action_encoder(
                action_tokens,
                action_type_tokens,
                action_numeric,
                action_lengths,
            )
            horizon_ids = torch.arange(horizons, device=state.device).expand(
                batch,
                horizons,
            )
            horizon = self.horizon(horizon_ids)
            repeated = state.unsqueeze(1).expand(batch, horizons, config.latent_size)
            return self.predictor(torch.cat([repeated, actions, horizon], dim=-1))

        def forward(
            self,
            context_tokens: Any,
            context_type_tokens: Any,
            context_numeric: Any,
            context_lengths: Any,
            action_tokens: Any,
            action_type_tokens: Any,
            action_numeric: Any,
            action_lengths: Any,
            target_tokens: Any,
            target_type_tokens: Any,
            target_numeric: Any,
            target_lengths: Any,
        ) -> tuple[Any, Any, Any]:
            batch, horizons, sequence = target_tokens.shape
            state = self.encode_state(
                context_tokens,
                context_type_tokens,
                context_numeric,
                context_lengths,
            )
            predictions = self.predict_future(
                state,
                action_tokens,
                action_type_tokens,
                action_numeric,
                action_lengths,
            )
            with torch.no_grad():
                targets = self.target_encoder(
                    target_tokens.reshape(batch * horizons, sequence),
                    target_type_tokens.reshape(batch * horizons, sequence),
                    target_numeric.reshape(
                        batch * horizons,
                        sequence,
                        config.event_numeric_size,
                    ),
                    target_lengths.reshape(batch * horizons),
                ).reshape(batch, horizons, config.latent_size)
            return state, targets, predictions

    return DispatchWorldJEPA()


def dispatch_world_jepa_loss(
    state: Any,
    targets: Any,
    predictions: Any,
    *,
    config: DispatchWorldJEPAConfig,
    step: int,
    regularizer: str,
) -> tuple[Any, dict[str, float]]:
    torch = require_torch()
    alignment = torch.nn.functional.smooth_l1_loss(predictions, targets)
    regularization = alignment.new_zeros(())
    if regularizer != "none":
        regularization = anticollapse_loss(
            state,
            regularizer=regularizer,
            num_slices=config.regularizer_slices,
            seed=step,
        )
    total = alignment + config.regularizer_weight * regularization
    return total, {
        "alignment": float(alignment.detach()),
        regularizer: float(regularization.detach()),
        "total": float(total.detach()),
    }


def build_dispatch_supervised_transformer(config: DispatchWorldJEPAConfig) -> Any:
    """Strong supervised sequence floor using the same state and action inputs."""

    torch = require_torch()
    nn = torch.nn

    class DispatchSupervisedTransformer(nn.Module):  # type: ignore[name-defined,misc]
        def __init__(self) -> None:
            super().__init__()
            self.state_encoder = build_dispatch_state_encoder(config)
            self.action_encoder = build_action_plan_encoder(config)
            self.head = nn.Sequential(
                nn.Linear(config.latent_size * 2, config.hidden_size),
                nn.GELU(),
                nn.Linear(config.hidden_size, 2),
            )

        def forward(
            self,
            context_tokens: Any,
            context_type_tokens: Any,
            context_numeric: Any,
            context_lengths: Any,
            action_tokens: Any,
            action_type_tokens: Any,
            action_numeric: Any,
            action_lengths: Any,
        ) -> Any:
            state = self.state_encoder(
                context_tokens,
                context_type_tokens,
                context_numeric,
                context_lengths,
            )
            actions = self.action_encoder(
                action_tokens,
                action_type_tokens,
                action_numeric,
                action_lengths,
            )[:, -1]
            return self.head(torch.cat([state, actions], dim=-1))

    return DispatchSupervisedTransformer()
