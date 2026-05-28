"""
Vocabulary management for event-type tokenisation.

Maps event-type strings to integer IDs in a deterministic, persistable way.
Reserved tokens (<PAD>, <UNK>, <GAP>, <START>, <END>) always occupy IDs 0–4.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Iterable

from sensqml.data.schema import (
    END_TOKEN,
    GAP_TOKEN,
    PAD_TOKEN,
    RESERVED_TOKENS,
    START_TOKEN,
    UNK_TOKEN,
)


class Vocabulary:
    def __init__(self, min_frequency: int = 1, max_size: int | None = None):
        self.min_frequency = min_frequency
        self.max_size = max_size
        self._token_to_id: dict[str, int] = {}
        self._id_to_token: list[str] = []
        self._frozen = False
        for tok in RESERVED_TOKENS:
            self._add(tok)

    # ------------------------------------------------------------------
    # Build
    # ------------------------------------------------------------------

    def fit(self, event_types: Iterable[str]) -> "Vocabulary":
        counts = Counter(event_types)
        # Sort by count desc, then alphabetically for determinism
        sorted_items = sorted(counts.items(), key=lambda x: (-x[1], x[0]))

        for token, count in sorted_items:
            if count < self.min_frequency:
                continue
            if self.max_size and len(self._id_to_token) >= self.max_size:
                break
            if token not in self._token_to_id:
                self._add(token)

        self._frozen = True
        return self

    def _add(self, token: str) -> int:
        if token not in self._token_to_id:
            tid = len(self._id_to_token)
            self._token_to_id[token] = tid
            self._id_to_token.append(token)
        return self._token_to_id[token]

    # ------------------------------------------------------------------
    # Lookup
    # ------------------------------------------------------------------

    def encode(self, token: str) -> int:
        return self._token_to_id.get(token, self._token_to_id[UNK_TOKEN])

    def decode(self, token_id: int) -> str:
        if 0 <= token_id < len(self._id_to_token):
            return self._id_to_token[token_id]
        return UNK_TOKEN

    def __len__(self) -> int:
        return len(self._id_to_token)

    # ------------------------------------------------------------------
    # Reserved token convenience
    # ------------------------------------------------------------------

    @property
    def pad_id(self) -> int:
        return self._token_to_id[PAD_TOKEN]

    @property
    def unk_id(self) -> int:
        return self._token_to_id[UNK_TOKEN]

    @property
    def gap_id(self) -> int:
        return self._token_to_id[GAP_TOKEN]

    @property
    def start_id(self) -> int:
        return self._token_to_id[START_TOKEN]

    @property
    def end_id(self) -> int:
        return self._token_to_id[END_TOKEN]

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump({
                "tokens": self._id_to_token,
                "min_frequency": self.min_frequency,
                "max_size": self.max_size,
            }, f, indent=2)

    @classmethod
    def load(cls, path: str | Path) -> "Vocabulary":
        with open(path) as f:
            data = json.load(f)
        vocab = cls(min_frequency=data.get("min_frequency", 1), max_size=data.get("max_size"))
        # Clear default reserved-only state and rebuild from saved tokens
        vocab._token_to_id = {}
        vocab._id_to_token = []
        for tok in data["tokens"]:
            vocab._add(tok)
        vocab._frozen = True
        return vocab
