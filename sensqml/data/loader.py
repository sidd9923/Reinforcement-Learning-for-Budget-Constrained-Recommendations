"""
Event stream ingestion.

Loads raw event data from CSV / Parquet / in-memory DataFrames, validates the
schema, and groups events by user with stable temporal ordering.

Returns an `EventStream` — an ordered iterator of (user_id, list[Event]) pairs.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterator, Optional, Union

import pandas as pd

from sensqml.data.schema import Event, REQUIRED_COLUMNS

logger = logging.getLogger(__name__)


class EventStream:
    """
    Wraps a pandas DataFrame of validated events grouped by user.

    Construction validates schema; iteration yields (user_id, sorted events).
    """

    def __init__(self, df: pd.DataFrame):
        self._validate(df)
        self._df = df.copy()
        self._df["timestamp"] = pd.to_datetime(self._df["timestamp"], utc=True)
        self._df = self._df.sort_values(["user_id", "timestamp"]).reset_index(drop=True)

    @staticmethod
    def _validate(df: pd.DataFrame) -> None:
        missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
        if missing:
            raise ValueError(f"DataFrame missing required columns: {missing}")
        if df["user_id"].isna().any():
            raise ValueError("user_id column contains nulls")
        if df["event_type"].isna().any():
            raise ValueError("event_type column contains nulls")

    @property
    def num_users(self) -> int:
        return self._df["user_id"].nunique()

    @property
    def num_events(self) -> int:
        return len(self._df)

    @property
    def vocabulary(self) -> list[str]:
        """Distinct event-type strings observed in the stream."""
        return sorted(self._df["event_type"].unique().tolist())

    def __iter__(self) -> Iterator[tuple[str, list[Event]]]:
        for user_id, group in self._df.groupby("user_id", sort=False):
            events = [
                Event(
                    user_id=row.user_id,
                    timestamp=row.timestamp.to_pydatetime(),
                    event_type=row.event_type,
                    value=row.value if "value" in group.columns and pd.notna(row.value) else None,
                    metadata=row.metadata if "metadata" in group.columns else None,
                )
                for row in group.itertuples(index=False)
            ]
            yield user_id, events

    def to_dataframe(self) -> pd.DataFrame:
        return self._df.copy()


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def load_csv(path: Union[str, Path]) -> EventStream:
    df = pd.read_csv(path)
    logger.info("Loaded %d events from %s", len(df), path)
    return EventStream(df)


def load_parquet(path: Union[str, Path]) -> EventStream:
    df = pd.read_parquet(path)
    logger.info("Loaded %d events from %s", len(df), path)
    return EventStream(df)


def load_dataframe(df: pd.DataFrame) -> EventStream:
    return EventStream(df)
