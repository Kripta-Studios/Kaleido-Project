from __future__ import annotations

from typing import Any

from flowtwin.baselines.process_transformer import require_torch
from flowtwin.models.contracts import EventJEPAConfig
from flowtwin.models.event_encoder import build_event_encoder
from flowtwin.models.predictor import build_latent_predictor


def sigreg_loss(embeddings: Any, *, num_slices: int, seed: int) -> Any:
    """Epps-Pulley SIGReg over resampled random one-dimensional projections."""

    torch = require_torch()
    generator = torch.Generator(device=embeddings.device).manual_seed(seed)
    directions = torch.randn(
        (embeddings.shape[1], num_slices),
        generator=generator,
        device=embeddings.device,
        dtype=embeddings.dtype,
    )
    directions = directions / directions.norm(dim=0, keepdim=True).clamp_min(1e-8)
    integration = torch.linspace(
        -5,
        5,
        17,
        device=embeddings.device,
        dtype=embeddings.dtype,
    )
    theoretical = torch.exp(-0.5 * integration.square())
    projected = (embeddings @ directions).unsqueeze(2) * integration
    empirical_real = projected.cos().mean(dim=0)
    empirical_imag = projected.sin().mean(dim=0)
    error = (
        (empirical_real - theoretical).square() + empirical_imag.square()
    ) * theoretical
    statistic = torch.trapezoid(error, integration, dim=1) * embeddings.shape[0]
    return statistic.mean()


def visreg_loss(embeddings: Any, *, num_slices: int, seed: int) -> Any:
    """VISReg center, scale and sliced-Wasserstein shape regularization."""

    torch = require_torch()
    center = embeddings.mean(dim=0)
    center_loss = center.square().mean()
    centered = embeddings - center
    standard_deviation = centered.std(dim=0, unbiased=False)
    scale_loss = (1.0 - standard_deviation).square().mean()
    normalized = centered / standard_deviation.detach().clamp_min(1e-6)
    generator = torch.Generator(device=embeddings.device).manual_seed(seed)
    directions = torch.randn(
        (embeddings.shape[1], num_slices),
        generator=generator,
        device=embeddings.device,
        dtype=embeddings.dtype,
    )
    directions = directions / directions.norm(dim=0, keepdim=True).clamp_min(1e-8)
    projected = torch.sort(normalized @ directions, dim=0).values
    quantiles = (
        torch.arange(
            1,
            embeddings.shape[0] + 1,
            device=embeddings.device,
            dtype=embeddings.dtype,
        )
        / (embeddings.shape[0] + 1)
    )
    normal = torch.distributions.Normal(
        embeddings.new_zeros(()),
        embeddings.new_ones(()),
    )
    target = normal.icdf(quantiles).unsqueeze(1)
    shape_loss = (projected - target).square().mean()
    return center_loss + scale_loss + shape_loss


def build_event_jepa(config: EventJEPAConfig) -> Any:
    torch = require_torch()
    nn = torch.nn

    class EventJEPA(nn.Module):  # type: ignore[name-defined,misc]
        def __init__(self) -> None:
            super().__init__()
            self.encoder = build_event_encoder(config)
            self.predictor = build_latent_predictor(
                config.latent_size,
                config.horizon_count,
            )

        def encode(self, tokens: Any, lengths: Any, numeric: Any) -> Any:
            return self.encoder(tokens, lengths, numeric)

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
            context = self.encode(context_tokens, context_lengths, context_numeric)
            targets = self.encode(
                target_tokens.reshape(batch * horizons, sequence),
                target_lengths.reshape(batch * horizons),
                target_numeric.reshape(batch * horizons, 3),
            ).reshape(batch, horizons, config.latent_size)
            horizon_ids = torch.arange(horizons, device=context.device).expand(batch, horizons)
            repeated_context = context.unsqueeze(1).expand(batch, horizons, config.latent_size)
            predictions = self.predictor(repeated_context, horizon_ids)
            return context, targets, predictions

    return EventJEPA()


def event_jepa_loss(
    context: Any,
    targets: Any,
    predictions: Any,
    *,
    config: EventJEPAConfig,
    step: int,
    use_sigreg: bool = True,
    use_visreg: bool = False,
) -> tuple[Any, dict[str, float]]:
    torch = require_torch()
    if use_sigreg and use_visreg:
        raise ValueError("SIGReg and VISReg are mutually exclusive")
    alignment = torch.nn.functional.mse_loss(predictions, targets)
    regularization = alignment.new_zeros(())
    regularizer = "none"
    if use_visreg:
        regularizer = "visreg"
        flat_targets = targets.reshape(-1, config.latent_size)
        regularization = 0.5 * (
            visreg_loss(context, num_slices=config.sigreg_slices, seed=step * 2)
            + visreg_loss(
                flat_targets,
                num_slices=config.sigreg_slices,
                seed=step * 2 + 1,
            )
        )
    elif use_sigreg:
        regularizer = "sigreg"
        flat_targets = targets.reshape(-1, config.latent_size)
        regularization = 0.5 * (
            sigreg_loss(context, num_slices=config.sigreg_slices, seed=step * 2)
            + sigreg_loss(
                flat_targets,
                num_slices=config.sigreg_slices,
                seed=step * 2 + 1,
            )
        )
    total = alignment + config.sigreg_weight * regularization
    return total, {
        "alignment": float(alignment.detach()),
        regularizer: float(regularization.detach()),
        "total": float(total.detach()),
    }
