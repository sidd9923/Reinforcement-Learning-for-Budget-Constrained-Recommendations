# SensQML

**A Unified Sequence Feature Framework for Predictive Modeling.**

SensQML converts raw user event streams into ordered behavioural representations that can be consumed by three model families — logistic regression, LSTM, and Transformer — from a **single reusable feature pipeline**. One representation, three targets: churn, risk, and engagement.

On the reference churn benchmark used during development, the LSTM configuration reaches **AUC 0.86** with sub-millisecond per-user inference and a vocab of ~50 event types.

---

## Why SensQML

Most behavioural ML projects re-implement the same scaffolding three times — once for the regression baseline, again for the recurrent model, and a third time for the attention model — usually with subtly different definitions of "sequence", "lookback", and "gap". SensQML standardises that scaffolding behind one configurable builder, so the only thing that changes between baselines is the model on top.

| What's reusable | What's swappable |
|---|---|
| Event schema | Model family (LR / LSTM / Transformer) |
| Vocabulary | Aggregation strategy (count, TF-IDF, recency-weighted, statistical) |
| `SequenceBuilder` (lookback, gap tokens, positional encodings) | Positional encoding (`absolute` / `relative` / `time_delta` / `none`) |
| Training pipeline | Hyperparameters, run directories |

---

## Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                       Raw Event Stream                            │
│            user_id, timestamp, event_type, value, …               │
└───────────────────────────────┬──────────────────────────────────┘
                                │
              ┌─────────────────▼─────────────────┐
              │   EventStream (validated, sorted)  │
              └─────────────────┬─────────────────┘
                                │
              ┌─────────────────▼─────────────────┐
              │       Vocabulary (token → ID)      │
              │   <PAD>, <UNK>, <GAP>, <START>,    │
              │   <END> reserved at IDs 0–4        │
              └─────────────────┬─────────────────┘
                                │
              ┌─────────────────▼─────────────────┐
              │         SequenceBuilder            │
              │   • lookback_window                │
              │   • max_sequence_len               │
              │   • gap_threshold → <GAP> tokens   │
              │   • positional encoding            │
              │     (absolute / relative /         │
              │      time_delta / none)            │
              └─────────────────┬─────────────────┘
                                │
            ┌───────────────────┼───────────────────┐
            │                   │                   │
   ┌────────▼────────┐ ┌────────▼────────┐ ┌────────▼────────┐
   │  SequenceAggr.  │ │   LSTM (BiLSTM  │ │  Transformer    │
   │  → fixed vec    │ │   + masked mean │ │  encoder + CLS  │
   │  ↓              │ │   pool)         │ │  mean pool      │
   │ Logistic Regr.  │ │                 │ │                 │
   └────────┬────────┘ └────────┬────────┘ └────────┬────────┘
            │                   │                   │
            └───────────────────┼───────────────────┘
                                │
              ┌─────────────────▼─────────────────┐
              │    InferenceEngine + Flask API     │
              │  POST /api/v1/predict              │
              │  POST /api/v1/predict/batch        │
              │  (Dockerised, K8s-deployable)      │
              └────────────────────────────────────┘
```

---

## Sequence Semantics

The `SequenceBuilder` is the heart of the project. Its configuration determines what "a user's sequence" actually means:

| Option | Purpose |
|---|---|
| `lookback_window` | Only consider events within the last *N* days/hours. Older context is dropped before tokenisation. |
| `max_sequence_len` | Truncate from the **left**, keeping the most recent context — the half of the sequence the model usually needs. |
| `gap_threshold_seconds` | When the inter-event delta exceeds this, a synthetic `<GAP>` token is inserted. Lets the model see *that there was a pause* without smearing real events together. |
| `insert_start_end` | Wrap with `<START>` and `<END>` so models can learn boundary effects. |
| `positional_encoding` | `absolute` (sinusoidal of position index), `relative` (linear 0→1), `time_delta` (sinusoidal of log-seconds since the event), or `none`. |

This single layer is what makes the same input usable by a bag-of-events logistic regression *and* a time-aware Transformer with no rewiring.

---

## Project Structure

```
sensqml/
├── sensqml/
│   ├── data/
│   │   ├── schema.py            # Event dataclass + reserved tokens
│   │   └── loader.py            # CSV / Parquet / DataFrame → EventStream
│   ├── features/
│   │   ├── vocab.py             # Persistable Vocabulary
│   │   ├── sequence_builder.py  # ★ The core builder (lookback, gaps, pos-enc)
│   │   └── aggregator.py        # Sequence → fixed-width vector (for LR)
│   ├── models/
│   │   ├── base.py              # BaseSequenceModel ABC
│   │   ├── logistic.py          # Aggregated features + sklearn LR
│   │   ├── lstm.py              # BiLSTM with masked mean pool
│   │   ├── transformer.py       # Encoder + CLS-style pool
│   │   └── factory.py           # build_model("lstm", vocab, …)
│   ├── training/
│   │   ├── pipeline.py          # End-to-end train_pipeline()
│   │   └── metrics.py           # AUC / AUPRC / threshold tuning
│   ├── inference/
│   │   └── engine.py            # InferenceEngine — load-once, fast-predict
│   ├── api/
│   │   └── app.py               # Flask app + endpoints
│   └── utils/
│       └── logging.py
├── config/
│   └── settings.py
├── scripts/
│   ├── train.py                 # CLI training driver
│   └── generate_dummy_data.py   # Synthetic data for end-to-end testing
├── tests/
│   ├── unit/
│   │   ├── test_vocab.py
│   │   └── test_sequence_builder.py
│   └── integration/
│       └── test_inference_api.py
├── deploy/
│   ├── docker/
│   │   ├── Dockerfile           # Multi-stage build, non-root user
│   │   └── docker-compose.yml
│   └── k8s/
│       └── deployment.yaml      # Deployment + Service + HPA + PVC
├── pyproject.toml
├── requirements.txt
└── wsgi.py
```

> **Data policy:** real event logs are IP-protected and excluded from the repo. Use `scripts/generate_dummy_data.py` to produce a synthetic dataset that exercises every code path locally.

---

## Quick Start

### Install

```bash
git clone https://github.com/sidd9923/sensqml.git
cd sensqml

python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Generate synthetic data

```bash
python scripts/generate_dummy_data.py --n-users 1000 --out-dir data/
```

This writes `data/events.csv` and `data/labels.csv`.

### Train

```bash
# Strong baseline
python scripts/train.py \
    --events data/events.csv \
    --labels data/labels.csv \
    --model logistic \
    --run-dir runs/lr_v1

# LSTM
python scripts/train.py \
    --events data/events.csv \
    --labels data/labels.csv \
    --model lstm \
    --epochs 10 \
    --batch-size 64 \
    --pos-encoding time_delta \
    --run-dir runs/lstm_v1

# Transformer
python scripts/train.py \
    --events data/events.csv \
    --labels data/labels.csv \
    --model transformer \
    --epochs 15 \
    --run-dir runs/transformer_v1
```

Each run writes `vocab.json`, `model.bin`, `metrics.json`, and `config.json` into the run directory.

### Serve

```bash
SENSQML_RUN_DIR=runs/lstm_v1 python wsgi.py
```

Or with Docker:

```bash
docker compose -f deploy/docker/docker-compose.yml up --build
```

### Predict

```bash
curl -s -X POST http://localhost:8000/api/v1/predict \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "u_demo",
    "events": [
      {"timestamp": "2024-12-01T10:15:00Z", "event_type": "login"},
      {"timestamp": "2024-12-01T10:16:30Z", "event_type": "view_product"},
      {"timestamp": "2024-12-03T08:42:00Z", "event_type": "purchase", "value": 49.99}
    ]
  }' | python -m json.tool
```

```json
{
  "user_id": "u_demo",
  "score": 0.123456,
  "label": 0,
  "latency_ms": 3.21,
  "model": "lstm"
}
```

---

## API Reference

### `GET /health`
Liveness probe. Returns `{"status": "ok", "model": "<name>"}`.

### `GET /api/v1/model/info`
Model metadata: name, vocab size, max sequence length, positional encoding type, decision threshold.

### `POST /api/v1/predict`
Score a single user.

```json
{
  "user_id": "string",
  "events": [
    { "timestamp": "ISO-8601", "event_type": "string", "value": 1.0 }
  ]
}
```

### `POST /api/v1/predict/batch`
Score many users in one request — meaningfully faster than looping.

```json
{ "requests": [ { "user_id": "...", "events": [...] }, ... ] }
```

---

## Results

Reference benchmark on the development churn dataset (50K users, ~3.5M events):

| Model | AUC | AUPRC | F1 | p99 latency |
|---|---|---|---|---|
| Logistic Regression (statistical features) | 0.78 | 0.62 | 0.69 | ~0.4 ms |
| Logistic Regression (TF-IDF over event tokens) | 0.81 | 0.66 | 0.73 | ~0.5 ms |
| BiLSTM (`time_delta` positions) | **0.86** | **0.74** | **0.78** | ~3.0 ms |
| Transformer (2 layers, 4 heads) | 0.85 | 0.73 | 0.77 | ~5.5 ms |

The LSTM beats the Transformer slightly at this dataset size; the Transformer scales better as the corpus grows past ~500K users.

---

## Testing

```bash
pytest tests/ -v --cov=sensqml --cov-report=term-missing
```

The integration suite trains a tiny model end-to-end and exercises the full Flask API.

---

## Tech Stack

`Python` · `PyTorch` · `scikit-learn` · `Flask` · `pandas` · `Docker` · `Kubernetes`

---

## License

MIT — see [LICENSE](LICENSE).
