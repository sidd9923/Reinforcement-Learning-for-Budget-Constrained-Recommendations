"""
Flask application for SensQML inference.

Endpoints
---------
GET  /health
POST /api/v1/predict        — single user
POST /api/v1/predict/batch  — many users
GET  /api/v1/model/info     — model metadata
"""

from __future__ import annotations

import logging
import os
from datetime import datetime

from flask import Flask, jsonify, request

from sensqml.data.schema import Event
from sensqml.inference.engine import InferenceEngine

logger = logging.getLogger(__name__)


def create_app(run_dir: str | None = None, threshold: float | None = None) -> Flask:
    app = Flask(__name__)
    run_dir = run_dir or os.getenv("SENSQML_RUN_DIR", "runs/latest")
    threshold = threshold if threshold is not None else float(os.getenv("SENSQML_THRESHOLD", "0.5"))

    engine = InferenceEngine(run_dir=run_dir, threshold=threshold)
    app.config["ENGINE"] = engine

    @app.get("/health")
    def health():
        return jsonify({"status": "ok", "model": engine.model_name})

    @app.get("/api/v1/model/info")
    def model_info():
        return jsonify({
            "model_name": engine.model_name,
            "vocab_size": len(engine.vocab),
            "max_sequence_len": engine.sequence_config.max_sequence_len,
            "positional_encoding": engine.sequence_config.positional_encoding,
            "gap_threshold_seconds": engine.sequence_config.gap_threshold_seconds,
            "threshold": engine.threshold,
        })

    @app.post("/api/v1/predict")
    def predict():
        body = request.get_json(silent=True) or {}
        user_id = body.get("user_id")
        raw_events = body.get("events", [])

        if not user_id or not isinstance(raw_events, list):
            return jsonify({"error": "user_id and events (list) are required"}), 400

        try:
            events = _parse_events(user_id, raw_events)
        except Exception as exc:
            return jsonify({"error": f"Invalid event payload: {exc}"}), 400

        result = engine.predict(user_id=user_id, events=events)
        return jsonify({
            "user_id": result.user_id,
            "score": round(result.score, 6),
            "label": result.label,
            "latency_ms": round(result.latency_ms, 2),
            "model": result.model_name,
        })

    @app.post("/api/v1/predict/batch")
    def predict_batch():
        body = request.get_json(silent=True) or {}
        requests_list = body.get("requests", [])
        if not isinstance(requests_list, list) or not requests_list:
            return jsonify({"error": "`requests` must be a non-empty list"}), 400

        try:
            parsed = [
                (r["user_id"], _parse_events(r["user_id"], r.get("events", [])))
                for r in requests_list
            ]
        except Exception as exc:
            return jsonify({"error": f"Invalid payload: {exc}"}), 400

        predictions = engine.predict_batch(parsed)
        return jsonify({
            "predictions": [
                {
                    "user_id": p.user_id,
                    "score": round(p.score, 6),
                    "label": p.label,
                }
                for p in predictions
            ],
            "model": engine.model_name,
            "count": len(predictions),
        })

    return app


def _parse_events(user_id: str, raw_events: list[dict]) -> list[Event]:
    events: list[Event] = []
    for r in raw_events:
        events.append(Event(
            user_id=user_id,
            timestamp=datetime.fromisoformat(r["timestamp"].replace("Z", "+00:00")),
            event_type=r["event_type"],
            value=r.get("value"),
            metadata=r.get("metadata"),
        ))
    return events
