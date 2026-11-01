# Implementation Plan — F&O AI Trading Alert Agent (Upstox)

Alert-only system: NO order placement, modification, or cancellation anywhere. The plan below maps the 14 tasks from `tasks.md` (5 phases) to concrete, ordered file-creation steps so the project stays importable/runnable incrementally. Each item leaves the tree in a working state.

All paths are absolute under `c:/Users/win 10/Desktop/capstone project/`. Run all commands from that root so the `src` package and root scripts (`train_model.py`, `app.py`) resolve. Python 3.10+.

---

## Environment & Verification Constraints (READ FIRST)

- **No real credentials are available** for Upstox or Telegram. Do NOT make real network calls to Upstox/Telegram, and do NOT block on them. Verification is synthetic/local only.
- The implementation loop's environment must have **Python 3.10+** and `pip` available. (The planning machine had neither; assume the build/verify environment does.)
- **TA-Lib native build is the known deployment risk.** `TA-Lib` (the Python wheel) requires the native C library installed first. Code MUST target the `talib` API (`talib.RSI`, `talib.MACD`, `talib.BBANDS`, `talib.EMA`, `talib.ATR`) — never rewrite features to another library. If the native TA-Lib cannot be installed in the verify environment, that is the documented gap: the features unit test (item 4) will not run, but all other tests (label, aggregator, inference, protobuf, alerts, health) still pass because they are decoupled from feature computation. `build.sh` documents the native install for Render.
- Pure-logic modules are decoupled from I/O: `features`, `label`, `candle_aggregator`, `inference`, `protobuf` are unit-tested with **synthetic** data; `data_fetcher`, `websocket_client`, `alerts` are thin I/O wrappers verified by import only.
- **Alert-only invariant:** no order-placement code anywhere. A reviewer should grep for `place_order`, `modify_order`, `cancel_order`, `/v3/order` and confirm none exist.

## Feature-Parity Contract (CRITICAL — NFR-3)

A single shared module `src/features.py` is the ONLY source of feature logic. Training and live MUST import it; feature logic must NEVER be duplicated in either pipeline.

- `src/features.py` defines `compute_features(df: pd.DataFrame) -> pd.DataFrame` and exports `FEATURE_COLUMNS` (the exact ordered list of feature column names).
- `train_model.py` and `src/model_trainer.py` import `compute_features` and `FEATURE_COLUMNS` from `src.features`.
- The live pipeline (`src/websocket_client.py` / `src/candle_aggregator.py` callers) imports `compute_features` from `src.features` and recomputes the identical features on each candle close.
- The model bundle `model.pkl` is `joblib.dump({"model": rf, "feature_columns": FEATURE_COLUMNS, "config": {...}})`.
- `src/inference.py` `InferenceEngine` loads `model.pkl`, validates that `bundle["feature_columns"]` equals `FEATURE_COLUMNS` (order + names) BEFORE predicting, and selects columns in that exact order for `model.predict` / `predict_proba`.
- `compute_features` returns a DataFrame indexed identically to input and does NOT drop NaN rows — the caller drops warm-up NaN rows (and label horizon tail rows) to control the boundary.

## Verification Strategy (synthetic, no network)

1. **Imports:** every module imports cleanly: `python -c "import ..."`.
2. **Pure-logic unit tests** under `tests/` (pytest), synthetic data:
   - `test_features.py`: 200-row synthetic OHLCV → `compute_features` returns all `FEATURE_COLUMNS`; first ~50 rows contain NaN warm-up.
   - `test_label.py`: synthetic trending/falling/flat series with an explicit threshold → Buys/Sells/Holds (uses explicit threshold so it does NOT require talib).
   - `test_candle_aggregator.py`: synthetic ticks spanning 3+ minutes → correct closed candles OHLCV; buffer caps at 1000.
   - `test_inference.py`: tiny mock `RandomForestClassifier` bundle (train on synthetic, `joblib.dump`, load, predict) → label in `{-1,0,1}`, confidence in `[0,1]`.
   - `test_protobuf.py`: construct a `FeedResponse` with an `LTPC` feed, serialize, `decode_feed` → dict with numeric `ltp`/`ltt`/`ltq`/`cp`.
   - `test_alerts.py`: cooldown suppresses duplicate consecutive same-signal within 1 min; message formatted per FR-4.4 (mock `requests.post`).
   - `test_health.py`: Flask app boots with NO secrets set, `GET /health` returns 200.
3. **Training smoke test:** `python train_model.py --synthetic 5000 --output model_test.pkl` runs end-to-end (feature compute → label → TimeSeriesSplit CV → RF fit → bundle export → console CV report), no Upstox token needed. `joblib.load("model_test.pkl")["feature_columns"] == FEATURE_COLUMNS`.
4. **Health resilience:** start `app.py` with no secrets in the environment; `/health` returns 200 while the stream thread logs the missing-secret error and stays alive (no crash).

Run the full suite: `python -m pip install -r requirements.txt -r requirements-dev.txt` then `python -m pytest tests/ -q`.

---

## Implementation Plan

- [ ] 1. **Scaffolding: project structure, dependencies, deployment config, env/gitignore.** Create directories `src/`, `src/protobuf/`, `data/`, `tests/` with `__init__.py` files (`src/__init__.py`, `src/protobuf/__init__.py`; `tests/__init__.py`). Write `requirements.txt` pinning: `pandas`, `numpy`, `TA-Lib`, `scikit-learn`, `joblib`, `requests`, `websocket-client`, `protobuf`, `flask`, `gunicorn`. Write `requirements-dev.txt` pinning `pytest` and `grpcio-tools` (dev/protoc only — not deployed). Write `.env.example` documenting every variable from requirements §6 with placeholder values (UPSTOX_ACCESS_TOKEN, UPSTOX_API_KEY, UPSTOX_API_SECRET, UPSTOX_INSTRUMENT_KEY, TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID, PORT, MODEL_PATH, SIGNAL_THRESHOLD, LOG_LEVEL). Write `.gitignore` excluding `.env`, `model.pkl`, `model_test.pkl`, `data/`, `__pycache__/`, `*.pyc`, `.venv/`. Write `Procfile` (`web: gunicorn --workers 1 app:app`). Write `render.yaml` (Render Web Service definition, startCommand `gunicorn --workers 1 app:app`, healthCheckPath `/health`, env vars). Write `build.sh` installing the native TA-Lib C library + Python wheel on the Render build image (apt-get install build deps, fetch+build ta-lib C lib, then `pip install -r requirements.txt`) — document that this is the main deployment risk. Write `README.md` with setup, native TA-Lib install, training (`python train_model.py --synthetic N` and the real `--instrument/--from/--to` flow), run (`gunicorn --workers 1 app:app`), env vars, and the alert-only disclaimer.
      Files: `requirements.txt`, `requirements-dev.txt`, `.env.example`, `.gitignore`, `Procfile`, `render.yaml`, `build.sh`, `README.md`, `src/__init__.py`, `src/protobuf/__init__.py`, `tests/__init__.py`
      Verify: `python -m pip install -r requirements.txt -r requirements-dev.txt` succeeds (note: native TA-Lib may fail to install here — that is the documented risk; the rest install cleanly). `python -c "import pandas, numpy, sklearn, joblib, requests, websocket, google.protobuf, flask, gunicorn"` passes (talib import guarded separately).
      Acceptance: tasks.md Task 1, Task 13 (deploy config), NFR-6.

- [ ] 2. **`src/config.py` — central configuration (typed env loader).** Define a `Config` dataclass with typed fields for all env vars (requirements §6) plus feature/label config constants (indicator periods: rsi=14, macd=(12,26,9), bbands=(20,2), ema=[9,21,50], atr=14; label horizon=5; default instrument `NSE_INDEX|Nifty 50`; model path default `model.pkl`; alert cooldown seconds=60). Class method `Config.from_env() -> Config` reads `os.environ` with typed defaults. `Config.require_secrets() -> list[str]` returns missing required live secrets (`UPSTOX_ACCESS_TOKEN`, `TELEGRAM_BOT_TOKEN`, `TELEGRAM_CHAT_ID`); `Config.validate_for_live()` raises a clear `RuntimeError` listing missing vars. Access tokens logged redacted (first 6 chars + `***`) — add a `redact(token)` helper. Centralize feature config constants here so train/live stay synchronized; `FEATURE_COLUMNS` itself lives in `features.py` (imported, not redefined).
      Files: `src/config.py`, `tests/test_config.py`
      Verify: `python -m pytest tests/test_config.py -q` — `from_env()` works with a full env dict; `validate_for_live()` raises listing the missing vars when secrets absent.
      Acceptance: tasks.md Task 2, FR-1.2.

- [ ] 3. **`src/auth.py` — optional OAuth helper (standalone, not a runtime dependency).** Implement `exchange_auth_code(auth_code: str, api_key: str | None, api_secret: str | None) -> str` that POSTs to the Upstox token endpoint with `UPSTOX_API_KEY`/`UPSTOX_API_SECRET` and the auth code (from the manual login URL) to obtain an access token, then prints/returns it for the operator to set as `UPSTOX_ACCESS_TOKEN`. Use `requests`; wrap errors clearly. The live agent does NOT import this module at runtime (it reads `UPSTOX_ACCESS_TOKEN` directly). Guard with a clear message if `api_key`/`api_secret` are missing.
      Files: `src/auth.py`
      Verify: `python -c "from src.auth import exchange_auth_code; print('ok')"` imports cleanly (no network call).
      Acceptance: tasks.md (FR-1.3), design §2.2.

- [ ] 4. **`src/features.py` — shared feature engine (CRITICAL, depends on talib).** Implement `compute_features(df: pd.DataFrame) -> pd.DataFrame`. Input columns `['open','high','low','close','volume']` with a datetime index. Use `ta-lib`: `talib.RSI(close, timeperiod=14)` → `rsi_14`; `talib.MACD(close, 12, 26, 9)` → `macd`, `macd_signal`, `macd_hist`; `talib.BBANDS(close, timeperiod=20, nbdevup=2, nbdevdn=2)` → `bb_upper`, `bb_middle`, `bb_lower` (+ derived `bb_width = bb_upper - bb_lower`); `talib.EMA(close, 9/21/50)` → `ema_9`, `ema_21`, `ema_50`; `talib.ATR(high, low, close, 14)` → `atr_14`; plus `return_1 = np.log(close/close.shift(1))` and `return_5 = np.log(close/close.shift(5))`. Return a DataFrame indexed identically to input; do NOT drop NaNs. Export `FEATURE_COLUMNS` as the exact ordered list: `["rsi_14","macd","macd_signal","macd_hist","bb_upper","bb_middle","bb_lower","bb_width","ema_9","ema_21","ema_50","atr_14","return_1","return_5"]`. Keep all params reading from the `Config` constants where reasonable so train/live stay aligned.
      Files: `src/features.py`, `tests/test_features.py`
      Verify: `python -m pytest tests/test_features.py -q` — feed a 200-row synthetic OHLCV DataFrame, get all `FEATURE_COLUMNS` back, first ~50 rows contain NaN warm-up. (Requires talib; if native TA-Lib unavailable this test is the documented gap — keep code targeting talib API regardless.)
      Acceptance: tasks.md Task 3, FR-2.5, NFR-3, NFR-4.

- [ ] 5. **`src/label.py` — forward-return labels (training only).** Implement `make_labels(df: pd.DataFrame, horizon: int = 5, threshold: float | None = None) -> pd.Series`. Compute forward return `(close.shift(-horizon) - close) / close`. If `threshold is None`, use an ATR-scaled default (`0.5 * atr_14 / close`) so the threshold adapts to volatility (ATR from `compute_features` or recomputed). Label `1` if forward return > threshold, `-1` if < -threshold, else `0`. The caller drops the last `horizon` rows (no forward target) and NaN warm-up rows. For testability without talib, support an explicit numeric `threshold` so the ATR path is skipped.
      Files: `src/label.py`, `tests/test_label.py`
      Verify: `python -m pytest tests/test_label.py -q` — synthetic trending series → Buys, falling series → Sells, flat series → Holds (pass explicit threshold to avoid the talib ATR path).
      Acceptance: tasks.md Task 5, FR-2.6.

- [ ] 6. **`src/data_fetcher.py` — Upstox REST historical (monthly pagination, CSV cache).** Implement `fetch_historical_candles(instrument_key: str, from_date, to_date, token: str | None = None) -> pd.DataFrame`. Call `GET https://api.upstox.com/v3/historical-candle/{instrument_key}/minutes/1/{to_date}/{from_date}` with `Authorization: Bearer <token>` and `Accept: application/json`. Paginate MONTHLY (1-min data capped at 1 month/request) — iterate month-by-month from `from_date` to `to_date`, sleeping between calls to respect rate limits. Normalize the Upstox candle array `[timestamp, open, high, low, close, volume, open_interest]` into a typed DataFrame with a datetime index (UTC/IST-aware). Cache to / read from `data/historical_1min.csv`, appending only date ranges not already cached (compute cached min/max, fetch only the gaps). Handle 429 (backoff+retry), 401 (raise a flagged `TokenExpiredError` for the re-auth alert path), and generic HTTP errors distinctly. This is an I/O wrapper — keep it thin and importable.
      Files: `src/data_fetcher.py`
      Verify: `python -c "from src.data_fetcher import fetch_historical_candles; print('ok')"` imports cleanly (no network call).
      Acceptance: tasks.md Task 4, FR-2.1–FR-2.4, FR-1.4.

- [ ] 7. **`src/model_trainer.py` + `train_model.py` — RF + TimeSeriesSplit + bundle + CLI (with synthetic mode).** In `model_trainer.py`: implement `train(df: pd.DataFrame, config: Config | None = None) -> dict` orchestrating `compute_features` → `make_labels` → drop NaN/horizon rows → `sklearn.model_selection.TimeSeriesSplit(n_splits=5)` CV → `sklearn.ensemble.RandomForestClassifier(class_weight='balanced')` fit → print per-fold accuracy + a final `classification_report` on the last fold → build bundle `{"model": rf, "feature_columns": FEATURE_COLUMNS, "config": {...}}` → `joblib.dump` to the output path. NO `train_test_split` / shuffled CV anywhere. In `train_model.py`: argparse CLI with `--instrument` (default `NSE_INDEX|Nifty 50`), `--from`, `--to`, `--output` (default `model.pkl`), `--from-csv PATH` (read an OHLCV CSV, bypass Upstox fetch), and `--synthetic N` (generate N synthetic OHLCV rows in-memory and train directly — no file/network). Flow: load CSV/synthetic or `data_fetcher.fetch_historical_candles` → `model_trainer.train` → write bundle → print summary (rows, feature count, per-fold CV accuracy, top-10 feature importances). `--synthetic` is the smoke-test path (no Upstox token needed).
      Files: `src/model_trainer.py`, `train_model.py`
      Verify: `python train_model.py --synthetic 5000 --output model_test.pkl` runs end-to-end, prints 5 CV folds + classification report, writes `model_test.pkl`. `python -c "import joblib; from src.features import FEATURE_COLUMNS; b=joblib.load('model_test.pkl'); assert b['feature_columns']==FEATURE_COLUMNS; print('ok')"`. Delete `model_test.pkl` after.
      Acceptance: tasks.md Task 6, FR-2.7–FR-2.9, NFR-4. Feature-parity contract satisfied (bundle stores `FEATURE_COLUMNS`).

- [ ] 8. **`src/protobuf/` — official Upstox proto + compiled module + `decode_feed` shim.** Save the official Upstox `MarketDataFeedV3.proto` schema (verified source: `https://raw.githubusercontent.com/upstox/upstox-python/master/examples/websocket/market_data/v3/MarketDataFeedV3.proto`) to `src/protobuf/market_data_v3.proto`. Top-level message `FeedResponse { Type type; map<string,Feed> feeds; int64 currentTs; MarketInfo marketInfo; }`; `Feed` oneof includes `LTPC ltpc`; `LTPC { double ltp; int64 ltt; int64 ltq; double cp; DoubleValue iep; }`; `RequestMode.ltpc = 0`. Compile to `src/protobuf/market_data_v3_pb2.py` via `python -m grpc_tools.protoc -I src/protobuf --python_out=src/protobuf src/protobuf/market_data_v3.proto` (grpcio-tools from `requirements-dev.txt`); commit the generated file so runtime needs only `protobuf`. Write `src/protobuf/market_data_pb2.py` as the decode shim per design §2.8: it imports the generated module and exposes `decode_feed(raw_bytes) -> dict` returning `{instrument_key: {"ltp": float, "ltt": int, "ltq": int, "cp": float}}` (parse `FeedResponse`, iterate `feeds`, read the `ltpc` field since we subscribe in ltpc mode). Add `src/protobuf/__init__.py` re-exporting `decode_feed`. If protoc cannot run in the env, fall back to a `market_data_pb2.py` shim that builds the message classes at runtime from the proto via `google.protobuf.descriptor_pool` + `message_factory` (one-file swap-in later) — clearly comment which parts are placeholders to verify against the live feed.
      Files: `src/protobuf/market_data_v3.proto`, `src/protobuf/market_data_v3_pb2.py`, `src/protobuf/market_data_pb2.py`, `src/protobuf/__init__.py`, `tests/test_protobuf.py`
      Verify: `python -m pytest tests/test_protobuf.py -q` — construct a `FeedResponse` with one feed keyed by an instrument containing an `LTPC` (ltp/ltt/ltq/cp), `SerializeToString`, then `decode_feed(bytes)` returns a dict with numeric `ltp` and an int `ltt`. No network needed.
      Acceptance: tasks.md Task 7, FR-3.3.

- [ ] 9. **`src/candle_aggregator.py` — tick → 1-min OHLCV, rolling 1000.** Implement `CandleAggregator(max_candles: int = 1000)` with an in-progress minute candle and a rolling buffer of the last 1000 closed candles (pandas DataFrame). `on_tick(tick: dict) -> dict | None`: determine the minute bucket (truncate tick time to minute, IST-aware); if no current candle, start one (open=high=low=ltp, volume=ltq); if tick minute == current minute, update high/low/close=ltp, accumulate volume; if tick minute > current minute, CLOSE the current candle, append to the rolling DataFrame (cap at `max_candles`), start a new candle from this tick, and return a closed-candle event dict `{"candle": <closed row>, "df": <updated DataFrame>}`; else return `None`. Thread-safe with a `threading.Lock` (called from the WS thread). `tick` dict carries `ltp`, `ltt` (epoch), `ltq`.
      Files: `src/candle_aggregator.py`, `tests/test_candle_aggregator.py`
      Verify: `python -m pytest tests/test_candle_aggregator.py -q` — feed synthetic ticks spanning 3 minutes → exactly 2 closed candles emitted with correct OHLCV; buffer length grows then caps at 1000.
      Acceptance: tasks.md Task 8, FR-3.4–FR-3.5.

- [ ] 10. **`src/inference.py` — load model.pkl, predict + confidence.** Implement `InferenceEngine(model_path: str)` whose `__init__` does `joblib.load`, stores `self.model` and `self.feature_columns`, and validates `self.feature_columns == FEATURE_COLUMNS` (names + order) BEFORE any predict — raise a clear error on mismatch. `predict(latest_row_df: pd.DataFrame) -> tuple[int, float]`: select `self.feature_columns` from the latest row in the stored order, run `self.model.predict` for the label (map to `{-1,0,1}`) and `self.model.predict_proba` (max probability) for confidence. Return `(label, confidence)`. The caller ignores `0`; only `1`/`-1` go to the alerter. Make the engine constructible from an in-memory bundle too (for tests).
      Files: `src/inference.py`, `tests/test_inference.py`
      Verify: `python -m pytest tests/test_inference.py -q` — train a tiny `RandomForestClassifier` on synthetic feature rows, `joblib.dump` a bundle with `feature_columns=FEATURE_COLUMNS`, load via `InferenceEngine`, predict on a feature row → label in `{-1,0,1}`, confidence in `[0,1]`; a mismatched-feature-columns bundle raises.
      Acceptance: tasks.md Task 10, FR-4.1–FR-4.3, feature-parity contract (validates order before predict).

- [ ] 11. **`src/alerts.py` — Telegram HTTP alerts + cooldown guard.** Implement `TelegramAlerter(bot_token: str, chat_id: str, cooldown_seconds: int = 60)` with `send_alert(signal: int, instrument: str, price: float, timestamp, confidence: float) -> bool`. Build the message per FR-4.4 (🚨 BUY/SELL SIGNAL, Instrument, Timeframe 1-Minute, Time IST, Price, Confidence) and POST to `https://api.telegram.org/bot<token>/sendMessage` with `chat_id` and `parse_mode` using `requests`. Wrap the HTTP call in try/except: log failures, never raise into the stream loop. Cooldown guard: ignore duplicate consecutive same-signal alerts within `cooldown_seconds` (track last signal + last-sent time); different signal or expired cooldown resets. Keep bot token/chat id out of logs (redact).
      Files: `src/alerts.py`, `tests/test_alerts.py`
      Verify: `python -m pytest tests/test_alerts.py -q` — message matches FR-4.4 format; a second identical same-signal call within cooldown is suppressed (mock `requests.post`); a different signal is sent. `python -c "from src.alerts import TelegramAlerter; print('ok')"`.
      Acceptance: tasks.md Task 11, FR-4.4–FR-4.6.

- [ ] 12. **`src/websocket_client.py` — Upstox WS V3 + auto-reconnect (I/O wrapper).** Implement `UpstoxMarketDataClient(config: Config, aggregator: CandleAggregator, on_candle_close: Callable)`. Obtain the authorized WS URL via `GET https://api.upstox.com/v3/feed/market-data-feed/authorize` with `Authorization: Bearer <token>` → `data.authorized_redirect_uri`. Open a `websocket.WebSocketApp` (from `websocket-client`, sync) against that URL; `on_open` sends the binary subscription JSON `{"guid": <id>, "method": "sub", "data": {"mode": "ltpc", "instrumentKeys": [instrument_key]}}` (UTF-8 encoded). `on_message(ws, raw_bytes)`: `decode_feed(raw_bytes)` → for each instrument tick call `aggregator.on_tick`; on a closed candle, call `on_candle_close(closed_df)` (wired to inference+alerts in item 13). `on_error`/`on_close`: log, trigger reconnect. `run_forever()` wraps everything in `while True: try...except` with capped exponential backoff (1s,2s,4s,…,60s) and resubscribes on reconnect. A 401 during connect raises a flagged error the caller maps to a Telegram re-auth alert. Use `ltpc` mode (only price+time needed). This is an I/O wrapper — keep it importable and thin; the live connection cannot be tested without real creds (expected).
      Files: `src/websocket_client.py`
      Verify: `python -c "from src.websocket_client import UpstoxMarketDataClient; print('ok')"` imports cleanly (no network call).
      Acceptance: tasks.md Task 9, FR-3.1–FR-3.2, FR-3.7, NFR-2.

- [ ] 13. **`src/server.py` + `app.py` — Flask /health (resilient to missing secrets) + gunicorn entry point.** In `src/server.py`: `app = Flask(__name__)` with `GET /health` returning `{"status":"ok","instrument":<key>,"last_candle":<ts>}` and HTTP 200; `last_candle` read from a shared thread-safe status object updated by the stream engine (default `None`). No other routes. In `app.py`: on import, build `Config.from_env()`, load the model into `InferenceEngine` (guard if model file missing — log, set engine=None, keep booting), build `CandleAggregator`, `TelegramAlerter`, construct a `LiveStreamEngine` that wires the WS client + aggregator + inference + alerter + status object, and start its `run_forever()` in a daemon background thread guarded by a module-level `started` flag (starts once even if imported multiple times). CRITICAL resilience: if required secrets are absent, Flask STILL boots and `/health` returns 200 (optionally degraded status), while the stream thread logs the missing-secret error and stays alive (does not crash) — wrap stream startup in try/except so a missing secret/config error never kills the process. Expose `app` for gunicorn (`gunicorn --workers 1 app:app`). `if __name__ == "__main__":` runs Flask directly on `int(os.environ.get("PORT", 5000))` for local dev.
      Files: `src/server.py`, `app.py`, `tests/test_health.py`
      Verify: with NO secrets in the environment, `python -m pytest tests/test_health.py -q` boots the app and asserts `GET /health` returns 200; OR start `python app.py` and `curl http://localhost:<PORT>/health` returns 200. The stream thread must not crash the process when secrets are missing.
      Acceptance: tasks.md Task 12, FR-5.1–FR-5.5, US-4.

- [ ] 14. **End-to-end verification: full suite, training smoke, health resilience, secret scan.** Run the complete synthetic verification: (a) `python -m pip install -r requirements.txt -r requirements-dev.txt`; (b) `python -m pytest tests/ -q` — all pure-logic tests pass (features test is the only talib-dependent one; if native TA-Lib is unavailable, document it and confirm the rest pass); (c) `python train_model.py --synthetic 5000 --output model_test.pkl` succeeds, prints 5 CV folds + classification report, and `joblib.load("model_test.pkl")["feature_columns"] == FEATURE_COLUMNS` (then remove `model_test.pkl`); (d) `python app.py` (no secrets) → `GET /health` returns 200, process stays alive; (e) grep the repo for secret-like strings and order-placement code: confirm no hardcoded secrets and no `place_order`/`modify_order`/`cancel_order`/`/v3/order` references (alert-only invariant). Confirm `.gitignore` excludes `.env`, `model.pkl`, `data/`.
      Files: `tests/` (whole suite), `model_test.pkl` (temporary, removed)
      Verify: full `python -m pytest tests/ -q` green (modulo documented talib gap), training smoke produces a valid bundle, `/health` returns 200 with no secrets, no secrets/order code in the tree.
      Acceptance: tasks.md Task 14 (synthetic subset — real live-signal verification needs market hours + valid token, out of scope here), §8 acceptance criteria 1,3,4,5.

---

## Notes for the Implementer

- **Feature parity is non-negotiable:** import `compute_features` and `FEATURE_COLUMNS` from `src.features` in both `train_model.py`/`model_trainer.py` and the live pipeline. Never recompute indicators inline. The bundle stores `FEATURE_COLUMNS`; `InferenceEngine` validates equality before predict.
- **No shuffled splits:** `TimeSeriesSplit` only, anywhere. A reviewer should grep for `train_test_split` and `ShuffleSplit` and confirm absence.
- **Alert-only:** no order-placement code anywhere. Reviewer greps `place_order|modify_order|cancel_order|/v3/order`.
- **ta-lib:** code targets the `talib` API. `build.sh` documents the native install for Render; native-build failure in a sandbox is the documented risk, not a code defect.
- **Protobuf:** the official proto is at the verified GitHub raw URL above; generate and commit `market_data_v3_pb2.py` so runtime needs only `protobuf`. The `decode_feed` shim lives in `market_data_pb2.py`.
- **Resilience:** `app.py` must boot and serve `/health` 200 even with no secrets; the stream thread logs and stays alive.
- **Logs:** structured, timestamped, to stdout (NFR-5); redact access tokens (first 6 chars + `***`); never log full Telegram chat id.
- **model.pkl:** the design says it may be committed (small) so the deployed instance loads it at startup. A generated `model_test.pkl` from the smoke test must NOT be committed (it is gitignored).

## Verification

Recorded during implementation iteration 1. This is the evidence the reviewer reads without re-running the suites.

### Runtime verification blocker (environment)

Command execution was unavailable in this implementation session: `execute_pwsh` calls returned only an echoed command and `Exit Code: -1` with no stdout, and file-creation side effects did not occur (canary `Set-Content`/`Out-File` files were not created on disk). A hook attempted to install Python 3.12 mid-session (a 25.7 MB download was observed) and appears to have disrupted the shell; the install did not land in a usable state in this session. As a result, the following runtime checks could NOT be executed/recorded here and must be run in an environment with Python 3.10+, the pinned dependencies, and the native TA-Lib C library:

- `python -m pip install -r requirements.txt -r requirements-dev.txt`
- `python -m pytest tests/ -q`
- `python train_model.py --synthetic 5000 --output model_test.pkl`
- `python app.py` -> `GET /health` returns 200 with no secrets

The code is written to pass these; the evidence below was produced by static review (read/grep tools, which functioned).

### Static verification (performed with file/grep tools)

1. **File inventory** — all design §7 files present and complete (each read end-to-end; no truncation from the interrupted write batch):
   - Root: `requirements.txt`, `requirements-dev.txt`, `Procfile`, `render.yaml`, `build.sh`, `.env.example`, `.gitignore`, `README.md`, `train_model.py`, `app.py`
   - `src/`: `__init__.py`, `config.py`, `auth.py`, `data_fetcher.py`, `features.py`, `label.py`, `model_trainer.py`, `candle_aggregator.py`, `websocket_client.py`, `inference.py`, `alerts.py`, `server.py`
   - `src/protobuf/`: `__init__.py`, `market_data_v3.proto`, `market_data_v3_pb2.py`, `market_data_pb2.py`
   - `tests/`: `__init__.py`, `test_config.py`, `test_features.py`, `test_label.py`, `test_candle_aggregator.py`, `test_inference.py`, `test_protobuf.py`, `test_alerts.py`, `test_health.py`

2. **Alert-only invariant** — grep `place_order|modify_order|cancel_order|/v3/order|placeOrder|modifyOrder|cancelOrder` matches ONLY `plan.md` documentation; NO matches in any source file.

3. **TimeSeriesSplit-only (NFR-4)** — grep `train_test_split|ShuffleSplit|shuffle=True` matches ONLY docs and a `model_trainer.py` docstring stating they are NOT used; NO usage in code. `model_trainer.train()` uses `sklearn.model_selection.TimeSeriesSplit(n_splits=5)` exclusively.

4. **Feature parity (NFR-3)** — grep `def compute_features` is defined exactly ONCE (in `src/features.py`). grep `import talib|talib\.RSI|talib\.MACD|talib\.BBANDS|talib\.EMA|talib\.ATR` shows talib used ONLY in `src/features.py` (no duplicated feature logic). `train_model.py`/`model_trainer.py` import `compute_features`/`FEATURE_COLUMNS` from `src.features`; the live engine (`app.py::LiveStreamEngine.on_candle_close`) calls `compute_features`; `InferenceEngine.__init__` validates `bundle["feature_columns"] == FEATURE_COLUMNS` (order+names) before any `predict`.

5. **Resilience (FR-5.1)** — `app.py` boots with no secrets: `Config.from_env()` → `require_secrets()` non-empty → `status.degraded=True`; `load_or_none(model_path)` returns `None` if `model.pkl` absent (guarded); the stream thread logs the missing-secret error and `time.sleep`-loops (never crashes); `_start_stream()` is wrapped in `try/except` so a startup failure never blocks app boot. `create_app` `/health` returns HTTP 200 with `{"status":"ok",...,"degraded":true}`. Verified by reading `app.py` + `src/server.py`.

6. **Native TA-Lib is the documented gap** — `import talib` is lazy (inside `compute_features`), so every module imports cleanly without the C library; only feature computation (`test_features`) and the training smoke test require it. `test_label` uses an explicit numeric `threshold` to skip the talib ATR path; `test_inference`/`test_candle_aggregator`/`test_protobuf`/`test_alerts`/`test_health`/`test_config` need no talib.

7. **Decoupling** — pure-logic modules (`features`, `label`, `candle_aggregator`, `inference`, `protobuf`) are independent of I/O; `data_fetcher`/`websocket_client`/`alerts` are thin wrappers importable without network. `websocket-client` is imported lazily inside `LiveStreamEngine.run_forever` so `app.py` imports without that package.

### Bug fixes applied during static review

- `tests/test_candle_aggregator.py`: fixed an off-by-one in `test_buffer_grows_then_caps_at_max` — 6 ticks emit 5 closed candles (09:15–09:19); the capped(3) buffer keeps `[09:17, 09:18, 09:19]` → `index[0].minute==17`, `index[-1].minute==19` (previously asserted 18/20).
- `tests/test_protobuf.py`: hardened message builders to use explicit `CopyFrom` (robust across protobuf backends; the pinned `protobuf==4.25.5` uses the upb C++ backend).
- `src/alerts.py`: added `send_message` (FR-1.4 re-auth alert, used by `app.py::_on_auth_error`); removed unused imports (`Union`, `quote`).
- `src/inference.py` / `src/data_fetcher.py`: removed unused imports (`Union` / `Sequence`).

### Expected runtime results (to confirm in a working env)

- `python -m pytest tests/ -q`: all tests pass; `test_features` is **skipped** if native TA-Lib is absent (`pytest.importorskip("talib")`) and passes when it is present.
- `python train_model.py --synthetic 5000 --output model_test.pkl`: prints 5 TimeSeriesSplit CV folds + a classification report + top-10 importances; writes a bundle whose `feature_columns == FEATURE_COLUMNS`. (Requires native TA-Lib.)
- `python app.py` with no secrets: process stays alive; `curl http://localhost:<PORT>/health` returns 200 with `degraded=true`.
- Secret scan: only placeholder values in `.env.example`; no real tokens committed (`.gitignore` excludes `.env`, `model.pkl`, `model_test.pkl`, `data/`, and root-level `_*` scratch files).

### Cleanup note

A few root-level scratch files from environment probing (`_probe*`, `_pycheck.txt`, `_*.ps1`, etc.) remain on disk (command execution could not delete them in this session). They are gitignored via root-anchored `/_*.txt`, `/_*.ps1`, `/_*.py`, `/_probe*` patterns and contain no project code or secrets.
