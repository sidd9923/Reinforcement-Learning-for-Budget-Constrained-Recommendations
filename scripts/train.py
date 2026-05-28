#!/usr/bin/env python3
"""
scripts/train.py
----------------
CLI driver for end-to-end training.

Usage:
    python scripts/train.py \\
        --events data/events.csv \\
        --labels data/labels.csv \\
        --model lstm \\
        --epochs 10 \\
        --run-dir runs/lstm_v1
"""

import argparse
import logging
import os
import sys
from datetime import timedelta

import pandas as pd

# Allow running without installing
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from sensqml.data.loader import load_csv, load_parquet
from sensqml.features.sequence_builder import SequenceConfig
from sensqml.training.pipeline import train_pipeline
from sensqml.utils.logging import setup_logging


def main():
    parser = argparse.ArgumentParser(description="SensQML training CLI")
    parser.add_argument("--events", required=True, help="Path to events CSV or Parquet")
    parser.add_argument("--labels", required=True, help="Path to labels CSV (user_id, label)")
    parser.add_argument(
        "--model", choices=["logistic", "lstm", "transformer"], default="lstm"
    )

    # Sequence config
    parser.add_argument("--max-seq-len", type=int, default=128)
    parser.add_argument("--lookback-days", type=int, default=None)
    parser.add_argument("--gap-threshold-sec", type=float, default=3600.0)
    parser.add_argument(
        "--pos-encoding",
        choices=["absolute", "relative", "time_delta", "none"],
        default="time_delta",
    )
    parser.add_argument("--pos-dim", type=int, default=16)

    # Training
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--run-dir", default="runs/latest")

    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    setup_logging(args.log_level)
    log = logging.getLogger("sensqml.train")

    # Load events
    if args.events.endswith(".parquet"):
        stream = load_parquet(args.events)
    else:
        stream = load_csv(args.events)
    labels_df = pd.read_csv(args.labels)
    if {"user_id", "label"} - set(labels_df.columns):
        raise ValueError("labels CSV must have columns: user_id, label")

    # Sequence config
    seq_cfg = SequenceConfig(
        max_sequence_len=args.max_seq_len,
        lookback_window=timedelta(days=args.lookback_days) if args.lookback_days else None,
        gap_threshold_seconds=args.gap_threshold_sec,
        positional_encoding=args.pos_encoding,
        pos_dim=args.pos_dim,
    )

    # Train kwargs differ per model
    train_kwargs = {"epochs": args.epochs, "batch_size": args.batch_size, "lr": args.lr}
    if args.model == "logistic":
        train_kwargs = {}  # LR doesn't take these

    metrics = train_pipeline(
        stream=stream,
        labels_df=labels_df,
        model_name=args.model,
        sequence_config=seq_cfg,
        model_hparams={},
        train_kwargs=train_kwargs,
        test_size=args.test_size,
        run_dir=args.run_dir,
    )

    log.info("Final metrics: %s", metrics)
    print()
    print(f"AUC:       {metrics['auc']:.4f}")
    print(f"AUPRC:     {metrics['auprc']:.4f}")
    print(f"Accuracy:  {metrics['accuracy']:.4f}")
    print(f"F1:        {metrics['f1']:.4f}")
    print(f"\nArtefacts written to: {args.run_dir}/")


if __name__ == "__main__":
    main()
