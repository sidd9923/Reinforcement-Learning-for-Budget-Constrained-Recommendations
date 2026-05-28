"""
Event schema definitions.

Every raw row consumed by SensQML must conform to the `Event` schema below.
This is the *only* contract between upstream data producers and the framework.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Optional


# ---------------------------------------------------------------------------
# Reserved token IDs
# ---------------------------------------------------------------------------
PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"
GAP_TOKEN = "<GAP>"        # inserted when the time delta exceeds gap_threshold
START_TOKEN = "<START>"
END_TOKEN = "<END>"

RESERVED_TOKENS = [PAD_TOKEN, UNK_TOKEN, GAP_TOKEN, START_TOKEN, END_TOKEN]


# ---------------------------------------------------------------------------
# Event dataclass
# ---------------------------------------------------------------------------
@dataclass
class Event:
    """
    A single user event.

    Attributes
    ----------
    user_id    : stable identifier for the user / session owner
    timestamp  : event time (UTC); used for ordering and lookback windows
    event_type : categorical action token (e.g. "login", "purchase", "view")
    value      : optional numeric value (e.g. amount, duration_ms)
    metadata   : optional free-form dict for downstream feature extractors
    """

    user_id: str
    timestamp: datetime
    event_type: str
    value: Optional[float] = None
    metadata: Optional[dict] = None

    def __post_init__(self):
        if not self.user_id:
            raise ValueError("user_id is required")
        # Reserved tokens are forbidden in user-supplied data but allowed when
        # the framework itself constructs synthetic events (e.g. <GAP> markers
        # inserted by SequenceBuilder). Internal construction sets a flag in
        # metadata to bypass this check.
        is_synthetic = isinstance(self.metadata, dict) and self.metadata.get("_synthetic", False)
        if self.event_type in RESERVED_TOKENS and not is_synthetic:
            raise ValueError(f"event_type cannot be a reserved token: {self.event_type}")


REQUIRED_COLUMNS = ["user_id", "timestamp", "event_type"]
OPTIONAL_COLUMNS = ["value", "metadata"]
