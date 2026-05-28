"""
Logistic regression baseline.

Consumes aggregated fixed-width feature vectors from `SequenceAggregator` and
trains a scikit-learn LogisticRegression with L2 regularisation. Strong
baseline that often surprises with how well it tracks LSTM/Transformer on
churn-style targets when good features are engineered.
"""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression as SkLogReg
from sklearn.preprocessing import StandardScaler

from sensqml.features.aggregator import SequenceAggregator
from sensqml.features.sequence_builder import Sequence
from sensqml.models.base import BaseSequenceModel


class LogisticRegressionModel(BaseSequenceModel):
    def __init__(
        self,
        aggregator: SequenceAggregator,
        C: float = 1.0,
        max_iter: int = 1000,
        class_weight: str | None = "balanced",
        random_state: int = 42,
    ):
        self.aggregator = aggregator
        self._scaler = StandardScaler()
        self._clf = SkLogReg(
            C=C,
            max_iter=max_iter,
            class_weight=class_weight,
            random_state=random_state,
            solver="liblinear",
        )

    def fit(self, sequences: list[Sequence], labels: np.ndarray, **kwargs) -> "LogisticRegressionModel":
        X = self.aggregator.transform_batch(sequences)
        X = self._scaler.fit_transform(X)
        self._clf.fit(X, labels)
        return self

    def predict_proba(self, sequences: list[Sequence]) -> np.ndarray:
        X = self.aggregator.transform_batch(sequences)
        X = self._scaler.transform(X)
        return self._clf.predict_proba(X)

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({"scaler": self._scaler, "clf": self._clf, "aggregator": self.aggregator}, f)

    @classmethod
    def load(cls, path: str | Path) -> "LogisticRegressionModel":
        with open(path, "rb") as f:
            state = pickle.load(f)
        model = cls(aggregator=state["aggregator"])
        model._scaler = state["scaler"]
        model._clf = state["clf"]
        return model
