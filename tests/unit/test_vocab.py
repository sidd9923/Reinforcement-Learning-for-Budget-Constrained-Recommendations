"""Tests for sensqml.features.vocab"""

import json

import pytest

from sensqml.data.schema import PAD_TOKEN, RESERVED_TOKENS, UNK_TOKEN
from sensqml.features.vocab import Vocabulary


class TestVocabulary:
    def test_reserved_tokens_take_first_ids(self):
        vocab = Vocabulary()
        for i, tok in enumerate(RESERVED_TOKENS):
            assert vocab.encode(tok) == i

    def test_fit_assigns_ids_by_frequency(self):
        vocab = Vocabulary()
        events = ["login"] * 10 + ["click"] * 5 + ["purchase"] * 1
        vocab.fit(events)
        # login is more frequent, so its ID should be lower (after reserved)
        assert vocab.encode("login") < vocab.encode("click")
        assert vocab.encode("click") < vocab.encode("purchase")

    def test_min_frequency_filters_rare_tokens(self):
        vocab = Vocabulary(min_frequency=3)
        events = ["a"] * 5 + ["b"] * 2 + ["c"] * 1
        vocab.fit(events)
        assert vocab.encode("a") != vocab.unk_id
        assert vocab.encode("b") == vocab.unk_id
        assert vocab.encode("c") == vocab.unk_id

    def test_max_size_caps_vocabulary(self):
        vocab = Vocabulary(max_size=8)
        vocab.fit([f"tok_{i}" for i in range(100)])
        assert len(vocab) <= 8

    def test_unknown_token_returns_unk_id(self):
        vocab = Vocabulary().fit(["a", "b"])
        assert vocab.encode("never_seen") == vocab.unk_id

    def test_save_load_roundtrip(self, tmp_path):
        vocab = Vocabulary().fit(["login", "click", "purchase"] * 3)
        path = tmp_path / "vocab.json"
        vocab.save(path)

        loaded = Vocabulary.load(path)
        assert len(loaded) == len(vocab)
        assert loaded.encode("login") == vocab.encode("login")
        assert loaded.encode("purchase") == vocab.encode("purchase")
        assert loaded.pad_id == vocab.pad_id
