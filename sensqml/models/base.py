"""
Base interface for SensQML models.

Every model — whether scikit-learn, LSTM, or Transformer — implements the same
four methods: `fit`, `predict_proba`, `save`, `load`. This lets the training
and inference pipelines stay agnostic to the underlying architecture.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np

from sensqml.features.sequence_builder import Sequence


class BaseSequenceModel(ABC):
    """Abstract base for all sequence models."""

    @abstractmethod
    def fit(self, sequences: list[Sequence], labels: np.ndarray, **kwargs) -> "BaseSequenceModel":
        ...

    @abstractmethod
    def predict_proba(self, sequences: list[Sequence]) -> np.ndarray:
        """Return shape (N, 2) with class probabilities."""

    def predict(self, sequences: list[Sequence], threshold: float = 0.5) -> np.ndarray:
        return (self.predict_proba(sequences)[:, 1] >= threshold).astype(np.int64)

    @abstractmethod
    def save(self, path: str | Path) -> None:
        ...

    @classmethod
    @abstractmethod
    def load(cls, path: str | Path) -> "BaseSequenceModel":
        ...
