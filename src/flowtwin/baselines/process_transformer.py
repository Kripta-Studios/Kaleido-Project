from __future__ import annotations

import importlib
from dataclasses import dataclass
from typing import Any, Literal


def require_torch() -> Any:
    try:
        torch = importlib.import_module("torch")
    except ImportError as exc:
        raise RuntimeError(
            "sequential models require the optional dependency: uv sync --extra sequence"
        ) from exc
    return torch


@dataclass(frozen=True)
class SequenceModelConfig:
    vocabulary_size: int
    hidden_size: int = 64
    layers: int = 2
    dropout: float = 0.1
    architecture: Literal["gru", "transformer"] = "transformer"
    max_length: int = 256


def build_sequence_model(config: SequenceModelConfig) -> Any:
    torch = require_torch()
    nn = torch.nn

    class RemainingTimeSequenceModel(nn.Module):  # type: ignore[name-defined,misc]
        def __init__(self) -> None:
            super().__init__()
            self.embedding = nn.Embedding(config.vocabulary_size, config.hidden_size, padding_idx=0)
            self.position = nn.Embedding(config.max_length, config.hidden_size)
            if config.architecture == "gru":
                self.encoder = nn.GRU(
                    config.hidden_size,
                    config.hidden_size,
                    config.layers,
                    batch_first=True,
                    dropout=config.dropout if config.layers > 1 else 0.0,
                )
            else:
                layer = nn.TransformerEncoderLayer(
                    d_model=config.hidden_size,
                    nhead=4,
                    dim_feedforward=config.hidden_size * 4,
                    dropout=config.dropout,
                    batch_first=True,
                    norm_first=True,
                )
                self.encoder = nn.TransformerEncoder(layer, config.layers)
            self.norm = nn.LayerNorm(config.hidden_size)
            self.head = nn.Sequential(
                nn.Linear(config.hidden_size + 3, config.hidden_size),
                nn.GELU(),
                nn.Linear(config.hidden_size, 2),
            )

        def forward(self, tokens: Any, lengths: Any, numeric: Any) -> Any:
            batch, sequence = tokens.shape
            positions = torch.arange(sequence, device=tokens.device).expand(batch, sequence)
            encoded = self.embedding(tokens) + self.position(positions)
            if config.architecture == "gru":
                output, _ = self.encoder(encoded)
            else:
                padding_mask = tokens.eq(0)
                output = self.encoder(encoded, src_key_padding_mask=padding_mask)
            indices = (lengths - 1).clamp(min=0)
            state = output[torch.arange(batch, device=tokens.device), indices]
            raw = self.head(torch.cat([self.norm(state), numeric], dim=1))
            p50 = torch.nn.functional.softplus(raw[:, 0])
            p90 = p50 + torch.nn.functional.softplus(raw[:, 1])
            return torch.stack([p50, p90], dim=1)

    return RemainingTimeSequenceModel()


def pinball_loss(prediction: Any, target: Any, quantiles: tuple[float, ...] = (0.5, 0.9)) -> Any:
    torch = require_torch()
    losses = []
    for index, quantile in enumerate(quantiles):
        error = target - prediction[:, index]
        losses.append(torch.maximum(quantile * error, (quantile - 1) * error))
    return torch.stack(losses, dim=1).mean()
