from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any

from flowtwin.baselines.process_transformer import require_torch
from flowtwin.models.event_jepa import anticollapse_loss


@dataclass(frozen=True)
class AISWorldModelConfig:
    input_size: int = 7
    forecast_size: int = 5
    max_length: int = 8
    horizon_count: int = 3
    port_vocabulary_size: int = 5
    vessel_vocabulary_size: int = 3
    hidden_size: int = 64
    latent_size: int = 32
    layers: int = 2
    attention_heads: int = 4
    dropout: float = 0.0
    regularizer_slices: int = 32
    regularizer_weight: float = 0.05
    forecast_weight: float = 1.0


def build_ais_context_encoder(config: AISWorldModelConfig, *, kind: str = "transformer") -> Any:
    torch = require_torch()
    nn = torch.nn

    class AISContextEncoder(nn.Module):  # type: ignore[name-defined,misc]
        def __init__(self) -> None:
            super().__init__()
            self.kind = kind
            self.numeric = nn.Linear(config.input_size, config.hidden_size)
            self.port = nn.Embedding(
                config.port_vocabulary_size, config.hidden_size, padding_idx=0
            )
            self.vessel = nn.Embedding(
                config.vessel_vocabulary_size, config.hidden_size, padding_idx=0
            )
            self.position = nn.Embedding(config.max_length, config.hidden_size)
            if kind == "transformer":
                layer = nn.TransformerEncoderLayer(
                    d_model=config.hidden_size,
                    nhead=config.attention_heads,
                    dim_feedforward=config.hidden_size * 4,
                    dropout=config.dropout,
                    batch_first=True,
                    norm_first=True,
                    activation="gelu",
                )
                self.backbone = nn.TransformerEncoder(layer, config.layers)
            elif kind == "gru":
                self.backbone = nn.GRU(
                    config.hidden_size,
                    config.hidden_size,
                    num_layers=config.layers,
                    batch_first=True,
                    dropout=config.dropout if config.layers > 1 else 0.0,
                )
            else:
                raise ValueError(f"unknown AIS encoder kind: {kind}")
            self.output = nn.Sequential(
                nn.LayerNorm(config.hidden_size),
                nn.Linear(config.hidden_size, config.latent_size),
            )

        def forward(
            self,
            numeric: Any,
            lengths: Any,
            port_tokens: Any,
            vessel_tokens: Any,
        ) -> Any:
            positions = torch.arange(numeric.shape[1], device=numeric.device)
            values = (
                self.numeric(numeric)
                + self.position(positions).unsqueeze(0)
                + self.port(port_tokens).unsqueeze(1)
                + self.vessel(vessel_tokens).unsqueeze(1)
            )
            valid = positions.unsqueeze(0) < lengths.unsqueeze(1)
            if self.kind == "transformer":
                encoded = self.backbone(values, src_key_padding_mask=~valid)
            else:
                encoded, _ = self.backbone(values)
            last = encoded[
                torch.arange(encoded.shape[0], device=encoded.device),
                (lengths - 1).clamp_min(0),
            ]
            return self.output(last)

    return AISContextEncoder()


def build_ais_phys_jepa(
    config: AISWorldModelConfig,
    *,
    use_physics: bool,
    ema_momentum: float,
) -> Any:
    torch = require_torch()
    nn = torch.nn

    class AISPhysJEPA(nn.Module):  # type: ignore[name-defined,misc]
        def __init__(self) -> None:
            super().__init__()
            self.use_physics = use_physics
            self.ema_momentum = ema_momentum
            self.context_encoder = build_ais_context_encoder(config)
            self.target_encoder = copy.deepcopy(self.context_encoder)
            for parameter in self.target_encoder.parameters():
                parameter.requires_grad_(False)
            self.horizon = nn.Embedding(config.horizon_count, config.latent_size)
            self.physics = nn.Sequential(
                nn.Linear(config.forecast_size, config.latent_size),
                nn.GELU(),
                nn.Linear(config.latent_size, config.latent_size),
            )
            predictor_inputs = config.latent_size * (3 if use_physics else 2)
            self.predictor = nn.Sequential(
                nn.Linear(predictor_inputs, config.hidden_size * 2),
                nn.GELU(),
                nn.Linear(config.hidden_size * 2, config.latent_size),
            )
            self.decoder = nn.Sequential(
                nn.Linear(config.latent_size, config.hidden_size),
                nn.GELU(),
                nn.Linear(config.hidden_size, config.forecast_size),
            )

        def forward(
            self,
            context_numeric: Any,
            context_lengths: Any,
            port_tokens: Any,
            vessel_tokens: Any,
            physics_forecast: Any,
            target_context_numeric: Any,
            target_context_lengths: Any,
        ) -> tuple[Any, Any, Any, Any]:
            state = self.context_encoder(
                context_numeric, context_lengths, port_tokens, vessel_tokens
            )
            batch, horizons, length, features = target_context_numeric.shape
            with torch.no_grad():
                target = self.target_encoder(
                    target_context_numeric.reshape(batch * horizons, length, features),
                    target_context_lengths.reshape(batch * horizons),
                    port_tokens.unsqueeze(1).expand(-1, horizons).reshape(-1),
                    vessel_tokens.unsqueeze(1).expand(-1, horizons).reshape(-1),
                ).reshape(batch, horizons, -1)
            horizon = self.horizon(
                torch.arange(horizons, device=context_numeric.device)
            ).unsqueeze(0).expand(batch, -1, -1)
            repeated_state = state.unsqueeze(1).expand(-1, horizons, -1)
            inputs = [repeated_state, horizon]
            if self.use_physics:
                inputs.append(self.physics(physics_forecast))
            predicted = self.predictor(torch.cat(inputs, dim=-1))
            residual = self.decoder(predicted)
            forecast = physics_forecast + residual if self.use_physics else residual
            return state, target, predicted, forecast

        def train(self, mode: bool = True) -> Any:
            super().train(mode)
            self.target_encoder.eval()
            return self

        def update_target(self) -> None:
            with torch.no_grad():
                for online, target in zip(
                    self.context_encoder.parameters(),
                    self.target_encoder.parameters(),
                    strict=True,
                ):
                    target.data.mul_(self.ema_momentum).add_(
                        online.data, alpha=1.0 - self.ema_momentum
                    )

    return AISPhysJEPA()


def build_ais_supervised_forecaster(
    config: AISWorldModelConfig,
    *,
    kind: str,
    use_physics: bool,
) -> Any:
    torch = require_torch()
    nn = torch.nn

    class AISForecaster(nn.Module):  # type: ignore[name-defined,misc]
        def __init__(self) -> None:
            super().__init__()
            self.use_physics = use_physics
            self.encoder = build_ais_context_encoder(config, kind=kind)
            self.horizon = nn.Embedding(config.horizon_count, config.latent_size)
            self.physics = nn.Linear(config.forecast_size, config.latent_size)
            inputs = config.latent_size * (3 if use_physics else 2)
            self.head = nn.Sequential(
                nn.Linear(inputs, config.hidden_size * 2),
                nn.GELU(),
                nn.Linear(config.hidden_size * 2, config.forecast_size),
            )

        def forward(
            self,
            context_numeric: Any,
            context_lengths: Any,
            port_tokens: Any,
            vessel_tokens: Any,
            physics_forecast: Any,
        ) -> Any:
            state = self.encoder(
                context_numeric, context_lengths, port_tokens, vessel_tokens
            )
            batch, horizons, _ = physics_forecast.shape
            horizon = self.horizon(
                torch.arange(horizons, device=context_numeric.device)
            ).unsqueeze(0).expand(batch, -1, -1)
            values = [state.unsqueeze(1).expand(-1, horizons, -1), horizon]
            if self.use_physics:
                values.append(self.physics(physics_forecast))
            residual = self.head(torch.cat(values, dim=-1))
            return physics_forecast + residual if self.use_physics else residual

    return AISForecaster()


def ais_jepa_loss(
    state: Any,
    target: Any,
    predicted: Any,
    forecast: Any,
    target_state: Any,
    *,
    config: AISWorldModelConfig,
    regularizer: str,
    step: int,
) -> tuple[Any, dict[str, float]]:
    torch = require_torch()
    latent = torch.nn.functional.smooth_l1_loss(predicted, target.detach())
    forecast_loss = torch.nn.functional.smooth_l1_loss(forecast, target_state)
    regularization = anticollapse_loss(
        torch.cat([state, predicted.reshape(-1, predicted.shape[-1])], dim=0),
        regularizer=regularizer,
        num_slices=config.regularizer_slices,
        seed=step,
    )
    total = (
        latent
        + config.forecast_weight * forecast_loss
        + config.regularizer_weight * regularization
    )
    return total, {
        "total": float(total.detach()),
        "latent": float(latent.detach()),
        "forecast": float(forecast_loss.detach()),
        "regularizer": float(regularization.detach()),
    }
