# F&O AI Trading Alert Agent (Upstox)

A Python-based, **alert-only** trading agent. It downloads historical 1-minute
OHLCV data from the Upstox V3 REST API, engineers `ta-lib` features, trains a
Random Forest classifier with `TimeSeriesSplit` cross-validation, streams live
market data over the Upstox WebSocket, runs inference on every closed 1-minute
candle, and sends buy/sell signals to Telegram.

> **This system never places, modifies, or cancels orders.** It only sends
> Telegram alerts. There is no order-placement code anywhere in this repo.

---

## How it fits together

```
Offline (run manually):      train_model.py  ->  model.pkl
Live agent (Render Web Svc): app.py  ->  gunicorn(Flask /health)  +  WS thread
                                   WS ticks -> 1-min candles -> features -> model -> Telegram
```

- **Training** (`train_model.py`): download (or synthetic) → `ta-lib` features
  → forward-return labels → `RandomForest` + `TimeSeriesSplit` → `model.pkl`.
- **Live agent** (`app.py`): gunicorn serves `/health` on the main thread; a
  daemon thread runs the Upstox WebSocket, aggregates ticks into 1-minute
  candles, recomputes the same features, predicts, and alerts on `1`/`-1`.
- Feature logic lives in **one** module (`src/features.py`) imported by both the
  training and live pipelines (NFR-3 consistency). The model bundle stores the
  exact ordered `FEATURE_COLUMNS`; inference validates the order before
  predicting.

## Project layout

```
train_model.py        CLI: download / synthetic -> train -> model.pkl
app.py                gunicorn entry: Flask /health + WS stream daemon thread
src/
  config.py           typed env config + feature/label constants
  auth.py             optional OAuth code->token helper (not a runtime dep)
  data_fetcher.py     Upstox REST historical (monthly pagination, CSV cache)
  features.py         SHARED ta-lib features + FEATURE_COLUMNS (single source)
  label.py            forward-return 1/-1/0 labels (training)
  model_trainer.py    RandomForest + TimeSeriesSplit + bundle export
  protobuf/           Upstox V3 feed decode (decode_feed)
  candle_aggregator.py tick -> 1-min OHLCV, rolling 1000
  websocket_client.py  Upstox WS V3 + auto-reconnect
  inference.py        load model.pkl, predict + confidence
  alerts.py           Telegram HTTP alerts + cooldown
  server.py           Flask /health
tests/                pure-logic synthetic unit tests (pytest)
requirements.txt      pinned runtime deps
render.yaml / Procfile / build.sh   Render deployment
```

## Prerequisites

- Python 3.10+
- The native **TA-Lib** C library installed (required for `ta-lib` features).
  - **Ubuntu/Debian** (Render): see `build.sh`.
  - **macOS**: `brew install ta-lib`
  - **Windows**: download the TA-Lib C library (MSVC build), extract to
    `C:\TA-Lib`, then install the Python wheel:
    ```
    pip install TA-Lib
    ```
    (If the wheel build can't find the C lib, set `TA_LIBRARY_PATH=C:\TA-Lib\lib`
    and `TA_INCLUDE_PATH=C:\TA-Lib\include`.)
  - **Note**: `ta-lib` import is lazily performed inside `compute_features`, so
    every other module imports cleanly even if the C library is absent; only
    feature computation and the training pipeline require it.

## Setup

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate    | macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
pip install -r requirements-dev.txt   # only for running tests

cp .env.example .env                 # fill in real secrets (never commit .env)
```

## Training

```bash
# Synthetic data smoke test (no Upstox token needed):
python train_model.py --synthetic 5000 --output model.pkl

# From a cached OHLCV CSV (columns: open,high,low,close,volume, datetime index):
python train_model.py --from-csv data/historical_1min.csv --output model.pkl

# Real Upstox historical data (needs UPSTOX_ACCESS_TOKEN):
python train_model.py --instrument "NSE_INDEX|Nifty 50" \
    --from 2025-11-01 --to 2026-10-31 --output model.pkl
```
The CLI prints per-fold `TimeSeriesSplit` accuracy, a classification report,
and the top-10 feature importances, then writes `model.pkl`.

## Running the live agent

```bash
# Local dev (Flask's built-in server; gunicorn is Unix-only):
python app.py            # serves http://0.0.0.0:$PORT/health

# Render / production (Linux):
gunicorn --workers 1 app:app
```
The stream thread starts automatically on import. `GET /health` returns
`{"status":"ok","instrument":...,"last_candle":...}` and HTTP 200. If required
secrets are missing the Flask app still boots and `/health` returns 200 while
the stream thread logs the missing-secret error and stays alive.

## Environment variables

See `.env.example` for the full list. Required at runtime for live signals:
`UPSTOX_ACCESS_TOKEN`, `TELEGRAM_BOT_TOKEN`, `TELEGRAM_CHAT_ID`.
`UPSTOX_API_KEY`/`UPSTOX_API_SECRET` are only for the optional `auth.py` helper.

## Regenerating the protobuf module

`src/protobuf/market_data_v3.proto` is a best-effort reconstruction of the
Upstox schema. `src/protobuf/market_data_v3_pb2.py` is a runtime-built shim
(uses only the `protobuf` runtime, no `protoc`) so the live feed decoder works
without a build step. To regenerate from the official proto once obtained:

```bash
python -m grpc_tools.protoc -I src/protobuf --python_out=src/protobuf \
    src/protobuf/market_data_v3.proto
# Then replace the runtime-built module with the generated one (same filename).
```

`decode_feed(raw_bytes) -> dict` in `market_data_pb2.py` is the only decode
surface used by the WebSocket client; swapping the underlying module is a
one-file change.

## Alert-only disclaimer

No order-placement, modification, or cancellation code exists in this repo.
Signals are delivered to Telegram for the trader to act on manually.
