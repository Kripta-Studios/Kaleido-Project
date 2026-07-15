from __future__ import annotations

from collections import Counter
from collections.abc import Iterable


class Vocabulary:
    def __init__(self, values: Iterable[str], min_frequency: int = 1) -> None:
        counts = Counter(values)
        tokens = sorted(token for token, count in counts.items() if count >= min_frequency)
        self.token_to_index = {
            "<PAD>": 0,
            "<UNK>": 1,
            **{token: i + 2 for i, token in enumerate(tokens)},
        }

    def encode(self, value: str) -> int:
        return self.token_to_index.get(value, 1)

    def __len__(self) -> int:
        return len(self.token_to_index)
