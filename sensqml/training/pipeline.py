"""
End-to-end training pipeline.

Steps:
  1. Load events from CSV/Parquet → EventStream
  2. Build vocabulary
  3. Build sequences per user via SequenceBuilder
  4. Align with labels by user_id
  5. Train/val split, fit model, log metrics
  6. Persist vocab + model + config under run_dir/
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

from sensqml.data.loader import EventStream
from sensqml.features.sequence_builder import SequenceBuilder, SequenceConfig
from sensqml.features.vocab import Vocabulary
from sensqml.models.base import BaseSequenceModel
from sensqml.models.factory import ModelName, build_model

logger = logging.getLogger(__name__)


def train_pipeline(
    stream: EventStream,
    labels_df: pd.DataFrame,
    *,
    model_name: ModelName = "lstm",
    sequence_config: Optional[SequenceConfig] = None,
    model_hparams: Optional[dict] = None,
    train_kwargs: Optional[dict] = None,
    test_size: float = 0.2,
    random_state: int = 42,
    run_dir: str | Path = "runs/latest",
) -> dict:
    """
    Run the full training pipeline.

    Parameters
    ----------
    stream         : the input EventStream
    labels_df      : DataFrame with columns ["user_id", "label"]; one row per user
    model_name     : "logistic" | "lstm" | "transformer"
    sequence_config: configuration for SequenceBuilder (defaults applied if None)
    model_hparams  : kwargs forwarded to build_model
    train_kwargs   : kwargs forwarded to model.fit (epochs, batch_size, lr, …)
    test_size      : holdout fraction for evaluation
    run_dir        : where to persist artefacts (vocab, model, metrics)

    Returns
    -------
    dict of evaluation metrics
    """
    sequence_config = sequence_config or SequenceConfig()
    model_hparams = model_hparams or {}
    train_kwargs = train_kwargs or {}
    run_path = Path(run_dir)
    run_path.mkdir(parents=True, exist_ok=True)

    # 1. Vocabulary
    logger.info("Fitting vocabulary over %d events", stream.num_events)
    vocab = Vocabulary(min_frequency=1).fit(stream.to_dataframe()["event_type"])
    vocab.save(run_path / "vocab.json")
    logger.info("Vocabulary size: %d", len(vocab))

    # 2. Sequences
    builder = SequenceBuilder(vocab=vocab, config=sequence_config)
    sequences = []
    user_ids_in_order = []
    for user_id, events in stream:
        sequences.append(builder.build(user_id, events))
        user_ids_in_order.append(user_id)
    logger.info("Built %d sequences", len(sequences))

    # 3. Align labels
    label_map = dict(zip(labels_df["user_id"], labels_df["label"]))
    aligned_labels = np.array([label_map[uid] for uid in user_ids_in_order], dtype=np.int64)

    # 4. Train/val split (by sequence; users are already unique per sequence)
    indices = np.arange(len(sequences))
    train_idx, test_idx = train_test_split(
        indices,
        test_size=test_size,
        stratify=aligned_labels,
        random_state=random_state,
    )
    seq_train = [sequences[i] for i in train_idx]
    seq_test = [sequences[i] for i in test_idx]
    y_train = aligned_labels[train_idx]
    y_test = aligned_labels[test_idx]

    # 5. Model
    model = build_model(model_name, vocab=vocab, **model_hparams)
    logger.info("Training model: %s", model_name)
    model.fit(seq_train, y_train, **train_kwargs)

    # 6. Evaluation
    proba = model.predict_proba(seq_test)[:, 1]
    preds = (proba >= 0.5).astype(np.int64)
    metrics = {
        "model": model_name,
        "n_train": int(len(y_train)),
        "n_test": int(len(y_test)),
        "auc": float(roc_auc_score(y_test, proba)),
        "auprc": float(average_precision_score(y_test, proba)),
        "accuracy": float(accuracy_score(y_test, preds)),
        "f1": float(f1_score(y_test, preds)),
        "vocab_size": len(vocab),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    logger.info("Metrics: %s", metrics)

    # 7. Persist
    model.save(run_path / "model.bin")
    with open(run_path / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    with open(run_path / "config.json", "w") as f:
        json.dump({
            "model_name": model_name,
            "model_hparams": model_hparams,
            "train_kwargs": train_kwargs,
            "sequence_config": _serialise_seq_config(sequence_config),
        }, f, indent=2, default=str)

    logger.info("Artefacts persisted to %s", run_path)
    return metrics


def _serialise_seq_config(cfg: SequenceConfig) -> dict:
    d = asdict(cfg)
    if d.get("lookback_window") is not None:
        d["lookback_window"] = d["lookback_window"].total_seconds()
    return d
