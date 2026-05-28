"""
SequenceBuilder
---------------
The heart of SensQML.

Converts a per-user list of `Event` objects into an ordered token sequence,
applying configurable sequence semantics:

  * **lookback_window**   — how far back (in time) to include events
  * **max_sequence_len**  — truncation from the *left* (keep most recent)
  * **gap_threshold**     — time gap (seconds) that triggers a <GAP> token
  * **insert_start_end**  — wrap sequences with <START> / <END> markers
  * **positional_encoding** — "absolute" | "relative" | "time_delta" | "none"

The output is a `Sequence` object containing token IDs, position vectors, and
attention/padding masks — ready to drop into PyTorch DataLoaders or scikit-learn
pipelines (after a flattening transform).
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Literal, Optional

import numpy as np

from sensqml.data.schema import Event
from sensqml.features.vocab import Vocabulary

logger = logging.getLogger(__name__)

PositionalEncoding = Literal["absolute", "relative", "time_delta", "none"]


# ---------------------------------------------------------------------------
# Output container
# ---------------------------------------------------------------------------
@dataclass
class Sequence:
    """A single user's encoded sequence ready for model consumption."""
    user_id: str
    token_ids: np.ndarray       # shape (max_sequence_len,), int64
    positions: np.ndarray       # shape (max_sequence_len, d_pos), float32
    attention_mask: np.ndarray  # shape (max_sequence_len,), int8  (1 = real token)
    length: int                 # actual non-pad length

    def to_dict(self) -> dict:
        return {
            "user_id": self.user_id,
            "token_ids": self.token_ids.tolist(),
            "positions": self.positions.tolist(),
            "attention_mask": self.attention_mask.tolist(),
            "length": self.length,
        }


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
@dataclass
class SequenceConfig:
    lookback_window: Optional[timedelta] = None
    max_sequence_len: int = 128
    gap_threshold_seconds: float = 3600.0            # 1 hour
    insert_start_end: bool = True
    positional_encoding: PositionalEncoding = "time_delta"
    pos_dim: int = 16                                # used by absolute / time_delta


# ---------------------------------------------------------------------------
# Builder
# ---------------------------------------------------------------------------
class SequenceBuilder:
    """
    Configurable sequence-construction engine.

    A single builder instance is stateless w.r.t. user data; configure once,
    call `build(user_id, events)` per user.
    """

    def __init__(self, vocab: Vocabulary, config: SequenceConfig):
        self.vocab = vocab
        self.cfg = config

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def build(self, user_id: str, events: list[Event], reference_time: Optional[datetime] = None) -> Sequence:
        """
        Build a single Sequence for one user.

        Parameters
        ----------
        user_id        : the user this sequence belongs to
        events         : pre-sorted (ascending timestamp) list of events
        reference_time : the "now" anchor; defaults to the last event's time.
                         Used for lookback windows and time-delta positions.

        Returns
        -------
        Sequence
        """
        if not events:
            return self._empty_sequence(user_id)

        events = self._apply_lookback(events, reference_time)
        if not events:
            return self._empty_sequence(user_id)

        # Inject GAP tokens where time deltas are large
        events_with_gaps = self._inject_gaps(events)

        # Optionally wrap with START/END
        tokens = [e.event_type for e in events_with_gaps]
        if self.cfg.insert_start_end:
            from sensqml.data.schema import END_TOKEN, START_TOKEN
            tokens = [START_TOKEN] + tokens + [END_TOKEN]

        # Encode → IDs
        token_ids = [self.vocab.encode(t) for t in tokens]

        # Truncate from the left (keep most recent context)
        if len(token_ids) > self.cfg.max_sequence_len:
            token_ids = token_ids[-self.cfg.max_sequence_len :]
            events_with_gaps = events_with_gaps[-(self.cfg.max_sequence_len - 2) :]

        # Build positional encodings (over the *un-padded* portion)
        positions = self._positional_encoding(token_ids, events_with_gaps, reference_time)

        # Right-pad to max_sequence_len
        pad_len = self.cfg.max_sequence_len - len(token_ids)
        attention_mask = np.array([1] * len(token_ids) + [0] * pad_len, dtype=np.int8)
        token_ids_padded = token_ids + [self.vocab.pad_id] * pad_len

        # Pad positions with zeros
        if positions.shape[0] < self.cfg.max_sequence_len:
            pad_block = np.zeros((pad_len, positions.shape[1]), dtype=np.float32)
            positions = np.concatenate([positions, pad_block], axis=0)

        return Sequence(
            user_id=user_id,
            token_ids=np.array(token_ids_padded, dtype=np.int64),
            positions=positions,
            attention_mask=attention_mask,
            length=len(token_ids),
        )

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _apply_lookback(self, events: list[Event], reference_time: Optional[datetime]) -> list[Event]:
        if self.cfg.lookback_window is None:
            return events
        ref = reference_time or events[-1].timestamp
        cutoff = ref - self.cfg.lookback_window
        return [e for e in events if e.timestamp >= cutoff]

    def _inject_gaps(self, events: list[Event]) -> list[Event]:
        """
        Walk through events; whenever the time delta exceeds gap_threshold,
        insert a synthetic GAP event before the current one.
        """
        from sensqml.data.schema import GAP_TOKEN

        if len(events) < 2:
            return list(events)

        result: list[Event] = [events[0]]
        for prev, curr in zip(events[:-1], events[1:]):
            delta = (curr.timestamp - prev.timestamp).total_seconds()
            if delta > self.cfg.gap_threshold_seconds:
                # Synthesise a gap event positioned midway in time
                mid_ts = prev.timestamp + (curr.timestamp - prev.timestamp) / 2
                result.append(Event(
                    user_id=curr.user_id,
                    timestamp=mid_ts,
                    event_type=GAP_TOKEN,
                    metadata={"_synthetic": True},
                ))
            result.append(curr)
        return result

    def _positional_encoding(
        self,
        token_ids: list[int],
        events: list[Event],
        reference_time: Optional[datetime],
    ) -> np.ndarray:
        """
        Compute per-position vectors of length `pos_dim`.

        Modes
        -----
        "none"        — zero vectors
        "absolute"    — sinusoidal encoding of position index (à la Transformer)
        "relative"    — index / max_len, broadcast to pos_dim
        "time_delta"  — sinusoidal encoding of seconds-since-event w.r.t. reference_time
        """
        n = len(token_ids)
        d = self.cfg.pos_dim

        if self.cfg.positional_encoding == "none":
            return np.zeros((n, d), dtype=np.float32)

        if self.cfg.positional_encoding == "relative":
            rel = np.linspace(0.0, 1.0, n, dtype=np.float32).reshape(-1, 1)
            return np.tile(rel, (1, d))

        if self.cfg.positional_encoding == "absolute":
            return _sinusoidal(np.arange(n, dtype=np.float32), d)

        # time_delta
        if events:
            ref = reference_time or events[-1].timestamp
            # Reconstruct timestamps for non-special tokens; assign 0 for START/END
            ts_seconds = []
            # token_ids may include START / END that aren't in events; align by walking
            event_iter = iter(events)
            for tid in token_ids:
                if tid in (self.vocab.start_id, self.vocab.end_id):
                    ts_seconds.append(0.0)
                else:
                    try:
                        ev = next(event_iter)
                        ts_seconds.append((ref - ev.timestamp).total_seconds())
                    except StopIteration:
                        ts_seconds.append(0.0)
            # Log-scale to handle wide ranges
            arr = np.log1p(np.array(ts_seconds, dtype=np.float32).clip(min=0))
            return _sinusoidal(arr, d)

        return np.zeros((n, d), dtype=np.float32)

    def _empty_sequence(self, user_id: str) -> Sequence:
        L = self.cfg.max_sequence_len
        return Sequence(
            user_id=user_id,
            token_ids=np.full(L, self.vocab.pad_id, dtype=np.int64),
            positions=np.zeros((L, self.cfg.pos_dim), dtype=np.float32),
            attention_mask=np.zeros(L, dtype=np.int8),
            length=0,
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _sinusoidal(values: np.ndarray, d_model: int) -> np.ndarray:
    """
    Standard sinusoidal positional encoding applied to an arbitrary 1-D
    sequence of scalar positions.
    """
    pos = values.reshape(-1, 1)
    i = np.arange(d_model // 2, dtype=np.float32)
    denom = np.power(10000.0, 2 * i / d_model)
    angle = pos / denom
    pe = np.zeros((len(values), d_model), dtype=np.float32)
    pe[:, 0::2] = np.sin(angle)
    pe[:, 1::2] = np.cos(angle)
    return pe
