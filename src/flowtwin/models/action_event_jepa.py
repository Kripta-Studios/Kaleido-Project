from __future__ import annotations

from typing import Any

from flowtwin.baselines.process_transformer import require_torch
from flowtwin.models.contracts import EventJEPAConfig
from flowtwin.models.event_encoder import build_event_encoder


def build_action_event_jepa(config: EventJEPAConfig, action_count: int) -> Any:
    """Build a synthetic action-conditioned latent transition model."""

    torch = require_torch()
    nn = torch.nn

    class ActionEventJEPA(nn.Module):  # type: ignore[name-defined,misc]
        def __init__(self) -> None:
            super().__init__()
            self.encoder = build_event_encoder(config)
            self.action = nn.Embedding(action_count, config.latent_size)
            self.context = nn.Sequential(
                nn.Linear(2, config.latent_size),
                nn.GELU(),
            )
            self.predictor = nn.Sequential(
                nn.Linear(config.latent_size * 3, config.latent_size * 2),
                nn.GELU(),
                nn.LayerNorm(config.latent_size * 2),
                nn.Linear(config.latent_size * 2, config.latent_size),
            )

        def encode(self, tokens: Any, lengths: Any, numeric: Any) -> Any:
            return self.encoder(tokens, lengths, numeric)

        def predict_state(
            self,
            tokens: Any,
            lengths: Any,
            numeric: Any,
            action_codes: Any,
            action_context: Any,
            mode: str,
        ) -> Any:
            state = self.encode(tokens, lengths, numeric)
            action = self.action(action_codes)
            context = self.context(action_context)
            if mode in {"action_only", "context_only"}:
                state = torch.zeros_like(state)
            if mode in {"current_prefix_only", "context_only"}:
                action = torch.zeros_like(action)
            if mode in {"current_prefix_only", "action_only"}:
                context = torch.zeros_like(context)
            return self.predictor(torch.cat([state, action, context], dim=1))

        def forward(
            self,
            tokens: Any,
            lengths: Any,
            numeric: Any,
            action_codes: Any,
            action_context: Any,
            target_tokens: Any,
            target_lengths: Any,
            target_numeric: Any,
            mode: str,
        ) -> tuple[Any, Any, Any]:
            context = self.encode(tokens, lengths, numeric)
            target = self.encode(target_tokens, target_lengths, target_numeric)
            prediction = self.predict_state(
                tokens,
                lengths,
                numeric,
                action_codes,
                action_context,
                mode,
            )
            return context, target, prediction

    return ActionEventJEPA()
