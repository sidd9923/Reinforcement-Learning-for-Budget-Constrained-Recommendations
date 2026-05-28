"""
Low-latency inference engine.

Loads a trained model + vocab + SequenceConfig once at startup, then handles
single-request and batch prediction with minimal per-call overhead:

  * vocabulary lookup is O(1) hash-map
  * sequence construction is O(L)
  * model inference runs in eval mode with `torch.inference_mode`

Designed to be plugged behind a Flask route or any RPC layer.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np

from sensqml.data.schema import Event
from sensqml.features.sequence_builder import SequenceBuilder, SequenceConfig
from sensqml.features.vocab import Vocabulary
from sensqml.models.base import BaseSequenceModel
from sensqml.models.logistic import LogisticRegressionModel
from sensqml.models.lstm import LSTMModel
from sensqml.models.transformer import TransformerModel

logger = logging.getLogger(__name__)


@dataclass
class Prediction:
    user_id: str
    score: float           # probability of positive class
    label: int             # thresholded label
    latency_ms: float
    model_name: str


class InferenceEngine:
    """
    Production-grade inference engine. Construct once per process.

    Parameters
    ----------
    run_dir : path to a directory containing vocab.json, model.bin, config.json
    threshold : decision threshold for the positive class
    """

    def __init__(self, run_dir: str | Path, threshold: float = 0.5):
        self.run_dir = Path(run_dir)
        self.threshold = threshold

        # Load artefacts
        self.vocab = Vocabulary.load(self.run_dir / "vocab.json")
        with open(self.run_dir / "config.json") as f:
            config = json.load(f)

        self.model_name: str = config["model_name"]
        seq_cfg_dict = config["sequence_config"]
        if seq_cfg_dict.get("lookback_window") is not None:
            from datetime import timedelta
            seq_cfg_dict["lookback_window"] = timedelta(seconds=seq_cfg_dict["lookback_window"])
        self.sequence_config = SequenceConfig(**seq_cfg_dict)
        self.builder = SequenceBuilder(vocab=self.vocab, config=self.sequence_config)

        self.model: BaseSequenceModel = self._load_model(self.model_name, self.run_dir / "model.bin")
        logger.info(
            "InferenceEngine ready: model=%s, vocab=%d, max_seq=%d",
            self.model_name,
            len(self.vocab),
            self.sequence_config.max_sequence_len,
        )

    @staticmethod
    def _load_model(model_name: str, path: Path) -> BaseSequenceModel:
        if model_name == "logistic":
            return LogisticRegressionModel.load(path)
        if model_name == "lstm":
            return LSTMModel.load(path)
        if model_name == "transformer":
            return TransformerModel.load(path)
        raise ValueError(f"Unknown model_name: {model_name}")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def predict(self, user_id: str, events: list[Event], reference_time: Optional[datetime] = None) -> Prediction:
        t0 = time.perf_counter()
        seq = self.builder.build(user_id, events, reference_time=reference_time)
        proba = self.model.predict_proba([seq])[0, 1]
        latency_ms = (time.perf_counter() - t0) * 1000
        return Prediction(
            user_id=user_id,
            score=float(proba),
            label=int(proba >= self.threshold),
            latency_ms=latency_ms,
            model_name=self.model_name,
        )

    def predict_batch(self, requests: list[tuple[str, list[Event]]]) -> list[Prediction]:
        """
        Batch prediction — much higher throughput than looping over `predict`.
        """
        t0 = time.perf_counter()
        sequences = [self.builder.build(uid, evs) for uid, evs in requests]
        probas = self.model.predict_proba(sequences)[:, 1]
        latency_ms = (time.perf_counter() - t0) * 1000
        per_request_ms = latency_ms / max(len(requests), 1)

        return [
            Prediction(
                user_id=uid,
                score=float(p),
                label=int(p >= self.threshold),
                latency_ms=per_request_ms,
                model_name=self.model_name,
            )
            for (uid, _), p in zip(requests, probas)
        ]
