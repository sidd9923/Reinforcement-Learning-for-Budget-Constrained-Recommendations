"""
Integration test: train a tiny logistic model end-to-end, save it, spin up the
Flask app pointing at that run_dir, and hit /predict.
"""

from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from sensqml.api.app import create_app
from sensqml.data.loader import load_dataframe
from sensqml.features.sequence_builder import SequenceConfig
from sensqml.training.pipeline import train_pipeline


@pytest.fixture(scope="module")
def trained_run_dir(tmp_path_factory):
    # Build a tiny synthetic dataset
    rows = []
    labels = []
    base = datetime(2024, 1, 1, tzinfo=timezone.utc)
    rng = np.random.default_rng(42)

    for i in range(60):
        uid = f"u_{i}"
        churn = i % 2 == 0
        n = rng.integers(5, 15)
        for k in range(n):
            etype = "purchase" if (not churn and k % 2 == 0) else "view"
            rows.append({
                "user_id": uid,
                "timestamp": (base + timedelta(hours=int(k))).isoformat(),
                "event_type": etype,
            })
        labels.append({"user_id": uid, "label": 1 if churn else 0})

    df = pd.DataFrame(rows)
    labels_df = pd.DataFrame(labels)
    stream = load_dataframe(df)

    run_dir = tmp_path_factory.mktemp("runs") / "test_run"
    train_pipeline(
        stream=stream,
        labels_df=labels_df,
        model_name="logistic",
        sequence_config=SequenceConfig(max_sequence_len=32),
        run_dir=run_dir,
        test_size=0.3,
    )
    return str(run_dir)


@pytest.fixture
def client(trained_run_dir):
    app = create_app(run_dir=trained_run_dir)
    app.config["TESTING"] = True
    with app.test_client() as c:
        yield c


class TestInferenceAPI:
    def test_health(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        assert resp.get_json()["status"] == "ok"

    def test_model_info_returns_metadata(self, client):
        resp = client.get("/api/v1/model/info")
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["model_name"] == "logistic"
        assert data["vocab_size"] > 0

    def test_predict_single_user(self, client):
        resp = client.post(
            "/api/v1/predict",
            json={
                "user_id": "test_user",
                "events": [
                    {"timestamp": "2024-01-01T00:00:00Z", "event_type": "view"},
                    {"timestamp": "2024-01-01T01:00:00Z", "event_type": "view"},
                    {"timestamp": "2024-01-01T02:00:00Z", "event_type": "view"},
                ],
            },
        )
        assert resp.status_code == 200
        data = resp.get_json()
        assert "score" in data
        assert "label" in data
        assert "latency_ms" in data
        assert 0.0 <= data["score"] <= 1.0

    def test_predict_missing_user_id_returns_400(self, client):
        resp = client.post("/api/v1/predict", json={"events": []})
        assert resp.status_code == 400

    def test_predict_batch(self, client):
        resp = client.post(
            "/api/v1/predict/batch",
            json={
                "requests": [
                    {
                        "user_id": "u1",
                        "events": [{"timestamp": "2024-01-01T00:00:00Z", "event_type": "view"}],
                    },
                    {
                        "user_id": "u2",
                        "events": [{"timestamp": "2024-01-01T00:00:00Z", "event_type": "purchase"}],
                    },
                ]
            },
        )
        assert resp.status_code == 200
        data = resp.get_json()
        assert len(data["predictions"]) == 2
