# F&O AI Trading Alert Agent — Tasks

Implementation order. Each task lists its deliverable and acceptance check. Tasks 1–3 are foundational and unblock everything else; 4–8 build the live pipeline; 9–11 wire deployment; 12 is verification.

Legend: `[ ]` pending · `[x]` done. Status tracked here as work proceeds.

---

## Phase 0 — Scaffolding

### Task 1: Project scaffolding & dependencies
- [ ] Create the file structure from design §7 (`src/`, `src/protobuf/`, `data/`).
- [ ] Write `requirements.txt` pinning: `pandas`, `numpy`, `TA-Lib`, `scikit-learn`, `joblib`, `requests`, `websocket-client`, `protobuf`, `flask`, `gunicorn`.
- [ ] Write `.env.example` with every variable from requirements §6 (placeholder values).
- [ ] Write `.gitignore` excluding `.env`, `model.pkl`, `data/`, `__pycache__/`, `*.pyc`.
- [ ] Write `README.md` with setup, training, and run instructions.
- **Acceptance:** `pip install -r requirements.txt` succeeds locally; `python -c "import pandas, numpy, talib, sklearn, joblib, requests, websocket, google.protobuf, flask"` imports cleanly.

### Task 2: Config module (`src/config.py`)
- [ ] Load all env vars into a typed `Config` object with defaults (instrument key, label horizon, thresholds, log level, paths).
- [ ] Validate required secrets at live startup; raise a clear error listing missing vars.
- [ ] Centralize feature config constants (indicator periods, `FEATURE_COLUMNS` reference lives in `features.py`).
- **Acceptance:** `from src.config import Config; Config.from_env()` works with a full `.env`; raises a descriptive error when `UPSTOX_ACCESS_TOKEN` / Telegram vars are absent.

### Task 3: Shared feature engine (`src/features.py`)
- [ ] Implement `compute_features(df)` using `ta-lib`: RSI(14), MACD(12,26,9), BBANDS(20,2), EMA(9/21/50), ATR(14), plus `return_1`, `return_5`.
- [ ] Define and export `FEATURE_COLUMNS` (exact ordered list).
- [ ] Return a DataFrame indexed identically to input; do NOT drop NaNs here (caller does).
- **Acceptance:** Unit check — feed a 60-row synthetic OHLCV DataFrame, get all `FEATURE_COLUMNS` back; first ~50 rows contain expected NaN warm-up.

---

## Phase 1 — Training Pipeline

### Task 4: Historical data fetcher (`src/data_fetcher.py`)
- [ ] Implement `fetch_historical_candles(instrument_key, from_date, to_date)` calling Upstox V3 `/v3/historical-candle/{key}/minutes/1/{to_date}/{from_date}` with Bearer auth.
- [ ] Paginate monthly (1-min cap = 1 month/request) over the full range; sleep between calls.
- [ ] Normalize candle arrays `[ts, o, h, l, c, vol, oi]` → typed DataFrame with datetime index.
- [ ] Cache to / read from `data/historical_1min.csv`, appending only uncached date ranges.
- [ ] Handle 429 (backoff), 401 (raise flagged error), and generic HTTP errors distinctly.
- **Acceptance:** Fetch 2–3 months for a test instrument; row count ≈ trading days × ~375 min/day; CSV grows on a second run only for the new range.

### Task 5: Label construction (`src/label.py`)
- [ ] Implement `make_labels(df, horizon=5, threshold=None)` producing `1/-1/0` from forward return, with ATR-scaled default threshold.
- [ ] Caller drops the last `horizon` rows (no forward target) and the NaN warm-up rows.
- **Acceptance:** Synthetic trending-then-falling series yields Buys then Sells; flat series yields Holds.

### Task 6: Model trainer (`src/model_trainer.py` + `train_model.py`)
- [ ] Implement `train(df)`: features → labels → `TimeSeriesSplit` CV → `RandomForestClassifier(class_weight='balanced')` → print per-fold accuracy + classification report → bundle.
- [ ] Bundle = `{"model", "feature_columns": FEATURE_COLUMNS, "config": {...}}`; `joblib.dump` to `model.pkl`.
- [ ] `train_model.py` CLI: `--instrument`, `--from`, `--to`, `--output`; prints summary (rows, CV accuracy, top-10 importances).
- [ ] No `train_test_split` / shuffled CV anywhere.
- **Acceptance:** Run `python train_model.py --instrument "NSE_INDEX|Nifty 50" --from 2025-11-01 --to 2026-10-31`; `model.pkl` written; console shows 5 CV folds + report; `joblib.load` returns a bundle whose `feature_columns` equals `FEATURE_COLUMNS`.

---

## Phase 2 — Live Pipeline

### Task 7: Protobuf decoding (`src/protobuf/`)
- [ ] Add Upstox `market_data_v3.proto`; compile to `market_data_pb2.py` (committed).
- [ ] Implement `decode_feed(raw_bytes) -> dict` returning normalized `{instrument: {ltp, ltt, ltq, cp, ...}}`.
- **Acceptance:** A sample captured WS binary frame decodes to a dict with numeric `ltp` and an ISO-parseable `ltt`. (If a real sample is hard to capture, decode a constructed Protobuf message as a smoke test.)

### Task 8: Candle aggregator (`src/candle_aggregator.py`)
- [ ] Implement `CandleAggregator` with in-progress minute candle + rolling buffer of last 1,000 closed candles.
- [ ] `on_tick(tick)`: bucket by IST minute; update in-progress OHLCV; on minute rollover, close candle, append to buffer, emit a closed-candle event.
- [ ] Thread-safe with a lock; returns the closed candle + updated DataFrame on close, else None.
- **Acceptance:** Feed synthetic ticks spanning 3 minutes → exactly 2 closed candles emitted with correct OHLCV; buffer length grows then caps at 1,000.

### Task 9: WebSocket client (`src/websocket_client.py`)
- [ ] Obtain authorized WS URL (via Upstox authorized-redirect endpoint) using the Bearer token.
- [ ] Open `websocket.WebSocketApp`; on open, send binary subscription JSON (`mode: ltpc`, instrument key).
- [ ] `on_message`: `decode_feed` → `aggregator.on_tick`; on a closed candle, hand off to the inference engine (wired in Task 10).
- [ ] Auto-reconnect in `run_forever()` with exponential backoff (1s→60s cap), resubscribing on reconnect; flag 401 for the Telegram re-auth alert.
- **Acceptance:** During market hours, logs tick receipt and a candle close every minute; killing the socket triggers a logged reconnect with backoff.

### Task 10: Inference engine (`src/inference.py`)
- [ ] `InferenceEngine` loads `model.pkl`, validates `feature_columns`, stores model + columns.
- [ ] `predict(latest_row_df) -> (label, confidence)` selects columns in order, runs `predict` + `predict_proba`.
- [ ] Called on each closed candle by the WS client; passes actionable (1/-1) results to the alerter.
- **Acceptance:** Given the latest feature row from `compute_features`, returns a label in `{-1,0,1}` and a confidence in [0,1]; a `1` or `-1` triggers the alerter path.

### Task 11: Telegram alerter (`src/alerts.py`)
- [ ] `send_alert(signal, instrument, price, timestamp, confidence)` POSTs to Telegram Bot API `sendMessage` per FR-4.4.
- [ ] try/except wraps the call; logs failures, never raises into the stream loop.
- [ ] Cooldown guard: ignore duplicate consecutive same-signal alerts within 1 minute.
- **Acceptance:** With a real bot token + chat id, a test alert arrives in Telegram formatted per FR-4.4; a second identical call within the cooldown is suppressed.

---

## Phase 3 — Server & Deployment

### Task 12: Flask health server (`src/server.py` + `app.py`)
- [ ] Flask `app` with `GET /health` → `{"status":"ok","instrument":<key>,"last_candle":<ts>}`, HTTP 200.
- [ ] `app.py`: on import, build Config + InferenceEngine + CandleAggregator + Alerter + LiveStreamEngine, start `run_forever()` in a daemon thread (once-guarded), expose `app`.
- [ ] `if __name__=="__main__"` runs Flask on `$PORT`/5000 for local dev.
- **Acceptance:** `gunicorn --workers 1 app:app` starts; `curl localhost:<PORT>/health` returns 200; stream thread logs live activity.

### Task 13: Render deployment config
- [ ] Write `Procfile` (`web: gunicorn --workers 1 app:app`) or `render.yaml`.
- [ ] Write `build.sh` to install the native TA-Lib C library + Python wheel on the Render build image.
- [ ] Document env vars to set in the Render dashboard (requirements §6).
- **Acceptance:** Render build completes; service goes live; `/health` returns 200 from the deployed URL; (live signal verification requires market hours + valid token).

---

## Phase 4 — Verification

### Task 14: End-to-end verification
- [ ] Train a model on ≥1 year of data; confirm `model.pkl` + CV report.
- [ ] Start the agent during market hours; confirm tick → candle → feature → predict → alert flow for at least one candle close.
- [ ] Force a WebSocket disconnect; confirm auto-reconnect.
- [ ] Confirm `/health` stays 200 throughout.
- [ ] Confirm no secret is committed (grep repo for token-like strings).
- **Acceptance:** A real (or simulated-market) buy/sell Telegram alert is received; reconnect and health checks pass; `git log`/working tree contains no secrets.

---

## Notes
- Resolve the open questions in design §8 before/during implementation (instrument type, Telegram lib, TA-Lib fallback, token refresh model).
- Feature parity between train and live is enforced by `features.py` — never duplicate feature logic in either pipeline.
- The system is alert-only: no order-placement code should ever be introduced.
