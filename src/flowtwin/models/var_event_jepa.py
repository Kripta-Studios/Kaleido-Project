from __future__ import annotations

from typing import Any

from flowtwin.baselines.process_transformer import require_torch
from flowtwin.models.contracts import EventJEPAConfig, VarEventJEPAConfig
from flowtwin.models.temporal_t_jepa import build_temporal_encoder


def _event_config(config: VarEventJEPAConfig) -> EventJEPAConfig:
    return EventJEPAConfig(
        vocabulary_size=config.vocabulary_size,
        max_length=config.max_length,
        hidden_size=config.hidden_size,
        latent_size=config.latent_size,
        layers=config.layers,
        attention_heads=config.attention_heads,
        dropout=config.dropout,
        horizon_count=config.horizon_count,
    )


def build_var_event_jepa(config: VarEventJEPAConfig) -> Any:
    """Build a temporal variational JEPA with a learned conditional latent prior."""

    torch = require_torch()
    nn = torch.nn
    event_config = _event_config(config)

    class VarEventJEPA(nn.Module):  # type: ignore[name-defined,misc]
        def __init__(self) -> None:
            super().__init__()
            self.context_encoder = build_temporal_encoder(
                event_config,
                register_token=False,
            )
            self.target_encoder = build_temporal_encoder(
                event_config,
                register_token=False,
            )
            self.context_distribution = nn.Linear(config.latent_size, config.latent_size * 2)
            self.auxiliary_distribution = nn.Sequential(
                nn.Linear(config.latent_size, config.latent_size),
                nn.GELU(),
                nn.Linear(config.latent_size, config.auxiliary_size * 2),
            )
            self.horizon = nn.Embedding(config.horizon_count, config.latent_size)
            posterior_input = config.latent_size * 3 + config.auxiliary_size
            prior_input = config.latent_size * 2 + config.auxiliary_size
            self.target_posterior = nn.Sequential(
                nn.Linear(posterior_input, config.latent_size * 2),
                nn.GELU(),
                nn.Linear(config.latent_size * 2, config.latent_size * 2),
            )
            self.target_prior = nn.Sequential(
                nn.Linear(prior_input, config.latent_size * 2),
                nn.GELU(),
                nn.Linear(config.latent_size * 2, config.latent_size * 2),
            )
            self.context_token_decoder = nn.Linear(
                config.latent_size,
                config.max_length * config.vocabulary_size,
            )
            self.context_numeric_decoder = nn.Linear(config.latent_size, 3)
            self.target_token_decoder = nn.Linear(
                config.latent_size,
                config.max_length * config.vocabulary_size,
            )
            self.target_numeric_decoder = nn.Linear(config.latent_size, 3)

        @staticmethod
        def _split(parameters: Any) -> tuple[Any, Any]:
            mean, log_variance = parameters.chunk(2, dim=-1)
            return mean, log_variance.clamp(min=-8.0, max=4.0)

        @staticmethod
        def _sample(mean: Any, log_variance: Any) -> Any:
            return mean + torch.exp(0.5 * log_variance) * torch.randn_like(mean)

        def encode_context(self, tokens: Any, lengths: Any, numeric: Any) -> tuple[Any, Any]:
            encoded = self.context_encoder(tokens, lengths, numeric)
            return self._split(self.context_distribution(encoded))

        def forward(
            self,
            context_tokens: Any,
            context_lengths: Any,
            context_numeric: Any,
            target_tokens: Any,
            target_lengths: Any,
            target_numeric: Any,
        ) -> dict[str, Any]:
            batch, horizons, sequence = target_tokens.shape
            context_mean, context_log_variance = self.encode_context(
                context_tokens,
                context_lengths,
                context_numeric,
            )
            context_state = self._sample(context_mean, context_log_variance)
            auxiliary_mean, auxiliary_log_variance = self._split(
                self.auxiliary_distribution(context_state)
            )
            auxiliary = self._sample(auxiliary_mean, auxiliary_log_variance)
            target_base = self.target_encoder(
                target_tokens.reshape(batch * horizons, sequence),
                target_lengths.reshape(batch * horizons),
                target_numeric.reshape(batch * horizons, 3),
            ).reshape(batch, horizons, config.latent_size)
            horizon_ids = torch.arange(horizons, device=context_tokens.device).expand(
                batch,
                horizons,
            )
            horizon = self.horizon(horizon_ids)
            repeated_context = context_state.unsqueeze(1).expand(
                batch,
                horizons,
                config.latent_size,
            )
            repeated_auxiliary = auxiliary.unsqueeze(1).expand(
                batch,
                horizons,
                config.auxiliary_size,
            )
            posterior_mean, posterior_log_variance = self._split(
                self.target_posterior(
                    torch.cat(
                        [repeated_context, repeated_auxiliary, target_base, horizon],
                        dim=-1,
                    )
                )
            )
            prior_mean, prior_log_variance = self._split(
                self.target_prior(
                    torch.cat([repeated_context, repeated_auxiliary, horizon], dim=-1)
                )
            )
            target_state = self._sample(posterior_mean, posterior_log_variance)
            context_token_logits = self.context_token_decoder(context_state).reshape(
                batch,
                config.max_length,
                config.vocabulary_size,
            )
            target_token_logits = self.target_token_decoder(target_state).reshape(
                batch,
                horizons,
                config.max_length,
                config.vocabulary_size,
            )
            return {
                "context_mean": context_mean,
                "context_log_variance": context_log_variance,
                "context_state": context_state,
                "auxiliary_mean": auxiliary_mean,
                "auxiliary_log_variance": auxiliary_log_variance,
                "posterior_mean": posterior_mean,
                "posterior_log_variance": posterior_log_variance,
                "prior_mean": prior_mean,
                "prior_log_variance": prior_log_variance,
                "target_state": target_state,
                "context_token_logits": context_token_logits,
                "context_numeric_prediction": self.context_numeric_decoder(context_state),
                "target_token_logits": target_token_logits,
                "target_numeric_prediction": self.target_numeric_decoder(target_state),
            }

        def inference_embedding(
            self,
            tokens: Any,
            lengths: Any,
            numeric: Any,
        ) -> tuple[Any, Any, Any]:
            with torch.no_grad():
                context_mean, context_log_variance = self.encode_context(
                    tokens,
                    lengths,
                    numeric,
                )
                auxiliary_mean, auxiliary_log_variance = self._split(
                    self.auxiliary_distribution(context_mean)
                )
                horizon_ids = torch.arange(
                    config.horizon_count,
                    device=tokens.device,
                ).expand(len(tokens), config.horizon_count)
                horizon = self.horizon(horizon_ids)
                _prior_mean, prior_log_variance = self._split(
                    self.target_prior(
                        torch.cat(
                            [
                                context_mean.unsqueeze(1).expand(
                                    len(tokens),
                                    config.horizon_count,
                                    config.latent_size,
                                ),
                                auxiliary_mean.unsqueeze(1).expand(
                                    len(tokens),
                                    config.horizon_count,
                                    config.auxiliary_size,
                                ),
                                horizon,
                            ],
                            dim=-1,
                        )
                    )
                )
                context_uncertainty = torch.exp(0.5 * context_log_variance).mean(dim=1)
                predictive_uncertainty = torch.exp(0.5 * prior_log_variance).mean(
                    dim=(1, 2)
                )
                predictive_uncertainty = predictive_uncertainty + torch.exp(
                    0.5 * auxiliary_log_variance
                ).mean(dim=1)
                return context_mean, context_uncertainty, predictive_uncertainty

    return VarEventJEPA()


def _standard_normal_kl(mean: Any, log_variance: Any) -> Any:
    return 0.5 * (mean.square() + log_variance.exp() - 1.0 - log_variance).mean()


def _normal_kl(
    posterior_mean: Any,
    posterior_log_variance: Any,
    prior_mean: Any,
    prior_log_variance: Any,
) -> Any:
    variance_ratio = (posterior_log_variance - prior_log_variance).exp()
    mean_term = (posterior_mean - prior_mean).square() * (-prior_log_variance).exp()
    return 0.5 * (
        prior_log_variance
        - posterior_log_variance
        + variance_ratio
        + mean_term
        - 1.0
    ).mean()


def var_event_jepa_loss(
    outputs: dict[str, Any],
    context_tokens: Any,
    context_numeric: Any,
    target_tokens: Any,
    target_numeric: Any,
    *,
    config: VarEventJEPAConfig,
    kl_scale: float,
) -> tuple[Any, dict[str, float]]:
    torch = require_torch()
    context_mask = context_tokens.ne(0)
    target_mask = target_tokens.ne(0)
    context_logits = outputs["context_token_logits"][context_mask]
    context_labels = context_tokens[context_mask]
    target_logits = outputs["target_token_logits"][target_mask]
    target_labels = target_tokens[target_mask]
    context_categorical = torch.nn.functional.cross_entropy(context_logits, context_labels)
    target_categorical = torch.nn.functional.cross_entropy(target_logits, target_labels)
    context_numeric_loss = torch.nn.functional.mse_loss(
        outputs["context_numeric_prediction"],
        context_numeric,
    )
    target_numeric_loss = torch.nn.functional.mse_loss(
        outputs["target_numeric_prediction"],
        target_numeric,
    )
    reconstruction = context_categorical + context_numeric_loss
    generation = target_categorical + target_numeric_loss
    context_kl = _standard_normal_kl(
        outputs["context_mean"],
        outputs["context_log_variance"],
    )
    auxiliary_kl = _standard_normal_kl(
        outputs["auxiliary_mean"],
        outputs["auxiliary_log_variance"],
    )
    target_kl = _normal_kl(
        outputs["posterior_mean"],
        outputs["posterior_log_variance"],
        outputs["prior_mean"],
        outputs["prior_log_variance"],
    )
    total = (
        config.reconstruction_weight * reconstruction
        + config.generation_weight * generation
        + kl_scale * config.context_kl_weight * context_kl
        + kl_scale * config.auxiliary_kl_weight * auxiliary_kl
        + kl_scale * config.target_kl_weight * target_kl
    )
    return total, {
        "reconstruction": float(reconstruction.detach()),
        "generation": float(generation.detach()),
        "context_kl": float(context_kl.detach()),
        "auxiliary_kl": float(auxiliary_kl.detach()),
        "target_kl": float(target_kl.detach()),
        "kl_scale": kl_scale,
        "total": float(total.detach()),
    }
