"""
Sequence → fixed-width feature aggregation.

Models like logistic regression cannot consume variable-length token sequences
directly. This module converts a `Sequence` into a dense feature vector using
configurable aggregation strategies:

  * **count**       — bag-of-events: per-token frequency counts (vocab_size dim)
  * **tf_idf**      — sklearn TF-IDF over the token sequence (vocab_size dim)
  * **recency_weighted** — frequencies weighted by inverse recency
  * **statistical** — sequence length, unique-token count, gap count, recency mean, etc.

These outputs feed directly into `LogisticRegressionModel`; LSTM and Transformer
models consume the raw `Sequence` objects.
"""

from __future__ import annotations

from typing import Literal

import numpy as np

from sensqml.features.sequence_builder import Sequence
from sensqml.features.vocab import Vocabulary

AggregationStrategy = Literal["count", "tf_idf", "recency_weighted", "statistical"]


class SequenceAggregator:
    def __init__(
        self,
        vocab: Vocabulary,
        strategy: AggregationStrategy = "count",
        idf_weights: np.ndarray | None = None,
    ):
        self.vocab = vocab
        self.strategy = strategy
        self.idf_weights = idf_weights

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def transform(self, sequence: Sequence) -> np.ndarray:
        if self.strategy == "count":
            return self._count_vector(sequence)
        if self.strategy == "tf_idf":
            return self._tfidf_vector(sequence)
        if self.strategy == "recency_weighted":
            return self._recency_weighted(sequence)
        if self.strategy == "statistical":
            return self._statistical(sequence)
        raise ValueError(f"Unknown aggregation strategy: {self.strategy}")

    def transform_batch(self, sequences: list[Sequence]) -> np.ndarray:
        return np.vstack([self.transform(s) for s in sequences])

    @property
    def output_dim(self) -> int:
        if self.strategy in ("count", "tf_idf", "recency_weighted"):
            return len(self.vocab)
        if self.strategy == "statistical":
            return 7  # see _statistical()
        return 0

    # ------------------------------------------------------------------
    # Strategies
    # ------------------------------------------------------------------

    def _count_vector(self, sequence: Sequence) -> np.ndarray:
        v = np.zeros(len(self.vocab), dtype=np.float32)
        valid = sequence.token_ids[: sequence.length]
        for tid in valid:
            v[tid] += 1.0
        return v

    def _tfidf_vector(self, sequence: Sequence) -> np.ndarray:
        counts = self._count_vector(sequence)
        if self.idf_weights is None:
            return counts
        return counts * self.idf_weights

    def _recency_weighted(self, sequence: Sequence) -> np.ndarray:
        """
        Weight each occurrence by 1 / (1 + position-from-end).
        More recent events count more.
        """
        v = np.zeros(len(self.vocab), dtype=np.float32)
        valid = sequence.token_ids[: sequence.length]
        n = len(valid)
        for i, tid in enumerate(valid):
            recency = n - i
            v[tid] += 1.0 / recency
        return v

    def _statistical(self, sequence: Sequence) -> np.ndarray:
        valid = sequence.token_ids[: sequence.length]
        if len(valid) == 0:
            return np.zeros(7, dtype=np.float32)
        unique = len(set(valid.tolist()))
        gap_count = int(np.sum(valid == self.vocab.gap_id))
        unk_count = int(np.sum(valid == self.vocab.unk_id))
        return np.array([
            sequence.length,                                # 0: total length
            unique,                                          # 1: unique events
            gap_count,                                       # 2: # of <GAP> tokens
            unk_count,                                       # 3: # of <UNK> tokens
            unique / max(sequence.length, 1),                # 4: vocab diversity
            gap_count / max(sequence.length, 1),             # 5: gap fraction
            float(np.std(valid)) if len(valid) > 1 else 0.0, # 6: token-ID dispersion
        ], dtype=np.float32)


# ---------------------------------------------------------------------------
# IDF fitting helper
# ---------------------------------------------------------------------------
def fit_idf(sequences: list[Sequence], vocab: Vocabulary, smoothing: float = 1.0) -> np.ndarray:
    """
    Compute IDF weights per token from a corpus of sequences.
    Returns an array of length vocab_size.
    """
    n_docs = len(sequences)
    df = np.zeros(len(vocab), dtype=np.float32)
    for s in sequences:
        seen = set(s.token_ids[: s.length].tolist())
        for tid in seen:
            df[tid] += 1.0
    return np.log((n_docs + smoothing) / (df + smoothing)) + 1.0
