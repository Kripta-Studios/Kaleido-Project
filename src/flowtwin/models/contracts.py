from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class EventJEPAConfig:
    vocabulary_size: int
    max_length: int = 16
    hidden_size: int = 64
    latent_size: int = 64
    layers: int = 2
    attention_heads: int = 4
    dropout: float = 0.1
    horizon_count: int = 3
    sigreg_slices: int = 64
    sigreg_weight: float = 0.1


@dataclass(frozen=True)
class VarEventJEPAConfig:
    vocabulary_size: int
    max_length: int = 16
    hidden_size: int = 64
    latent_size: int = 32
    auxiliary_size: int = 16
    layers: int = 2
    attention_heads: int = 4
    dropout: float = 0.0
    horizon_count: int = 3
    reconstruction_weight: float = 0.1
    generation_weight: float = 0.5
    context_kl_weight: float = 1e-3
    auxiliary_kl_weight: float = 1e-3
    target_kl_weight: float = 1e-3
