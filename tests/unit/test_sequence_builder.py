"""Tests for sensqml.features.sequence_builder"""

from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

from sensqml.data.schema import GAP_TOKEN, Event
from sensqml.features.sequence_builder import SequenceBuilder, SequenceConfig
from sensqml.features.vocab import Vocabulary


@pytest.fixture
def vocab():
    return Vocabulary().fit(["login", "view", "purchase", "logout"])


@pytest.fixture
def base_time():
    return datetime(2024, 1, 1, tzinfo=timezone.utc)


def _make_events(types: list[str], base: datetime, deltas_sec: list[int]) -> list[Event]:
    events = []
    t = base
    for et, dt in zip(types, deltas_sec):
        t = t + timedelta(seconds=dt)
        events.append(Event(user_id="u1", timestamp=t, event_type=et))
    return events


class TestSequenceBuilder:
    def test_empty_events_returns_pad_sequence(self, vocab):
        cfg = SequenceConfig(max_sequence_len=10)
        builder = SequenceBuilder(vocab=vocab, config=cfg)
        seq = builder.build("user_x", [])
        assert seq.length == 0
        assert np.all(seq.attention_mask == 0)
        assert np.all(seq.token_ids == vocab.pad_id)

    def test_token_ids_match_vocab(self, vocab, base_time):
        events = _make_events(["login", "view", "purchase"], base_time, [0, 60, 120])
        cfg = SequenceConfig(max_sequence_len=10, insert_start_end=False, gap_threshold_seconds=10000)
        seq = SequenceBuilder(vocab=vocab, config=cfg).build("u1", events)
        # First 3 tokens should decode to the input events
        decoded = [vocab.decode(int(t)) for t in seq.token_ids[: seq.length]]
        assert decoded == ["login", "view", "purchase"]

    def test_start_end_tokens_wrap_sequence(self, vocab, base_time):
        events = _make_events(["login"], base_time, [0])
        cfg = SequenceConfig(max_sequence_len=10, insert_start_end=True)
        seq = SequenceBuilder(vocab=vocab, config=cfg).build("u1", events)
        assert seq.token_ids[0] == vocab.start_id
        assert seq.token_ids[seq.length - 1] == vocab.end_id

    def test_gap_token_inserted_on_large_delta(self, vocab, base_time):
        # Two events 2 hours apart with gap_threshold of 1 hour → GAP injected
        events = _make_events(["login", "purchase"], base_time, [0, 7200])
        cfg = SequenceConfig(
            max_sequence_len=10,
            insert_start_end=False,
            gap_threshold_seconds=3600,
        )
        seq = SequenceBuilder(vocab=vocab, config=cfg).build("u1", events)
        decoded = [vocab.decode(int(t)) for t in seq.token_ids[: seq.length]]
        assert GAP_TOKEN in decoded
        assert decoded.index("login") < decoded.index(GAP_TOKEN) < decoded.index("purchase")

    def test_lookback_window_filters_old_events(self, vocab, base_time):
        # 3 events, last one at +10 days; lookback of 5 days should drop the first
        events = _make_events(
            ["login", "view", "purchase"],
            base_time,
            [0, 60 * 60 * 24, 60 * 60 * 24 * 10],
        )
        ref = events[-1].timestamp
        cfg = SequenceConfig(
            max_sequence_len=10,
            insert_start_end=False,
            lookback_window=timedelta(days=5),
            gap_threshold_seconds=10**9,
        )
        seq = SequenceBuilder(vocab=vocab, config=cfg).build("u1", events, reference_time=ref)
        decoded = [vocab.decode(int(t)) for t in seq.token_ids[: seq.length]]
        assert "login" not in decoded
        assert "purchase" in decoded

    def test_truncation_keeps_most_recent(self, vocab, base_time):
        events = _make_events(["login", "view", "purchase", "logout"], base_time, [0, 60, 120, 180])
        cfg = SequenceConfig(max_sequence_len=3, insert_start_end=False, gap_threshold_seconds=10000)
        seq = SequenceBuilder(vocab=vocab, config=cfg).build("u1", events)
        decoded = [vocab.decode(int(t)) for t in seq.token_ids[: seq.length]]
        assert decoded[-1] == "logout"
        assert "login" not in decoded

    def test_attention_mask_matches_length(self, vocab, base_time):
        events = _make_events(["login", "view"], base_time, [0, 60])
        cfg = SequenceConfig(max_sequence_len=10, insert_start_end=False, gap_threshold_seconds=10000)
        seq = SequenceBuilder(vocab=vocab, config=cfg).build("u1", events)
        assert seq.attention_mask.sum() == seq.length

    def test_positional_encoding_shape(self, vocab, base_time):
        events = _make_events(["login", "view"], base_time, [0, 60])
        for mode in ("absolute", "relative", "time_delta", "none"):
            cfg = SequenceConfig(
                max_sequence_len=10,
                pos_dim=8,
                positional_encoding=mode,
                gap_threshold_seconds=10000,
            )
            seq = SequenceBuilder(vocab=vocab, config=cfg).build("u1", events)
            assert seq.positions.shape == (10, 8)
