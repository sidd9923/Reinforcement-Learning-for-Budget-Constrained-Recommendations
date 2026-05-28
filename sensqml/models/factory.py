"""Model factory — instantiate by name."""

from __future__ import annotations

from typing import Literal

from sensqml.features.aggregator import SequenceAggregator
from sensqml.features.vocab import Vocabulary
from sensqml.models.base import BaseSequenceModel
from sensqml.models.logistic import LogisticRegressionModel
from sensqml.models.lstm import LSTMModel
from sensqml.models.transformer import TransformerModel

ModelName = Literal["logistic", "lstm", "transformer"]


def build_model(name: ModelName, vocab: Vocabulary, **hparams) -> BaseSequenceModel:
    """
    Build a fresh, untrained model by name.

    Examples
    --------
    >>> model = build_model("lstm", vocab, hidden_dim=256, num_layers=3)
    """
    if name == "logistic":
        strategy = hparams.pop("aggregation", "count")
        aggregator = SequenceAggregator(vocab=vocab, strategy=strategy)
        return LogisticRegressionModel(aggregator=aggregator, **hparams)

    if name == "lstm":
        return LSTMModel(vocab_size=len(vocab), **hparams)

    if name == "transformer":
        return TransformerModel(vocab_size=len(vocab), **hparams)

    raise ValueError(f"Unknown model name: {name!r}. Expected one of: logistic, lstm, transformer")
