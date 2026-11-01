# F&O AI Trading Alert Agent — Requirements

## 1. Overview

A Python-based alert-only trading agent that connects to the Upstox V3 API. It downloads historical 1-minute OHLCV data, engineers technical-analysis features with `ta-lib`, trains a classification model to predict short-term price direction, streams live market data over WebSocket, runs the model on every closed 1-minute candle, and pushes buy/sell signals to Telegram.

**This is an alert-only system. It will NOT place, modify, or cancel any live trades.**

## 2. User Stories

- **US-1 (Trader):** As a trader, I want to train a model on a year of NIFTY 1-minute data so that the agent learns price-direction patterns.
- **US-2 (Trader):** As a trader, I want the agent to stream live 1-minute candles so that signals reflect current market conditions.
- **US-3 (Trader):** As a trader, I want to receive a Telegram message the moment the model predicts a buy or sell, so I can act on it manually.
- **US-4 (Operator):** As an operator, I want a `/health` endpoint so the Render Web Service stays alive and I can monitor uptime.
- **US-5 (Operator):** As an operator, I want all secrets in environment variables so nothing sensitive is committed to source control.

## 3. Functional Requirements

### 3.1 Authentication
- **FR-1.1** The system shall authenticate with Upstox using an OAuth2 Bearer access token.
- **FR-1.2** The access token shall be supplied via the `UPSTOX_ACCESS_TOKEN` environment variable.
- **FR-1.3** An optional helper (`auth.py`) shall implement the Upstox authorization-code flow to exchange an auth code (obtained manually) for an access token, using `UPSTOX_API_KEY` and `UPSTOX_API_SECRET`.
- **FR-1.4** When the API returns 401 (token expired), the system shall alert via Telegram that re-authentication is required and continue retrying with backoff.

### 3.2 Training Pipeline (Historical Data)
- **FR-2.1** A script `train_model.py` shall download at least 1 year of 1-minute OHLCV data for a configurable instrument key (default: NIFTY 50 index).
- **FR-2.2** The script shall use the Upstox V3 endpoint `GET /v3/historical-candle/{instrument_key}/{unit}/{interval}/{to_date}/{from_date}` with `unit=minutes`, `interval=1`.
- **FR-2.3** Because 1-minute data is capped at 1 month per request, the script shall paginate by month, issuing ~12 sequential requests to cover the requested date range, and concatenate the results.
- **FR-2.4** The script shall persist raw downloaded candles to a CSV cache (`data/historical_1min.csv`) to avoid re-downloading on re-runs, and append only missing date ranges.
- **FR-2.5** The script shall compute features using `ta-lib`: RSI(14), MACD(12,26,9), Bollinger Bands(20,2), EMA(9), EMA(21), EMA(50), and ATR(14).
- **FR-2.6** The script shall create a target label: `1` (Buy) if the 5-candle forward return exceeds a positive threshold, `-1` (Sell) if it falls below a negative threshold, `0` (Hold) otherwise. The threshold shall default to an ATR-scaled value and be configurable.
- **FR-2.7** The script shall train a Random Forest classifier (`scikit-learn`).
- **FR-2.8** The script shall evaluate the model using `TimeSeriesSplit` cross-validation and print per-fold accuracy and a final classification report. Random `train_test_split` is explicitly disallowed.
- **FR-2.9** The script shall export a model bundle to `model.pkl` using `joblib`, containing the fitted model, the ordered list of feature column names, and the feature/threshold configuration.

### 3.3 Live Data Pipeline (WebSocket)
- **FR-3.1** The system shall connect to the Upstox Market Data Feed V3 WebSocket to stream live tick data for the configured instrument.
- **FR-3.2** The system shall NOT use REST polling for live prices.
- **FR-3.3** Incoming ticks are Protobuf-encoded; the system shall decode them using the Upstox-provided `market_data_v3.proto` compiled for Python and the `protobuf` library.
- **FR-3.4** The system shall aggregate incoming ticks into 1-minute OHLCV candles in real time, keyed by the minute timestamp (IST). The in-progress candle is updated on every tick; a candle is finalized when the clock rolls into a new minute.
- **FR-3.5** The system shall maintain a rolling window of the latest 1,000 closed candles in a Pandas DataFrame.
- **FR-3.6** On each candle close, the system shall recompute the exact same `ta-lib` features used in training (FR-2.5).
- **FR-3.7** The system shall include auto-reconnect logic in a `try...except` block: on disconnect or error, it shall reconnect with exponential backoff (capped) and resubscribe.

### 3.4 Inference & Alerting
- **FR-4.1** The system shall load `model.pkl` at startup.
- **FR-4.2** On every closed 1-minute candle, the system shall pass the latest feature row to the model.
- **FR-4.3** If the model predicts `1` (Buy) or `-1` (Sell), the system shall trigger the Telegram alert function. Prediction `0` (Hold) produces no alert.
- **FR-4.4** The Telegram message shall be clearly formatted, e.g.:
  ```
  🚨 BUY SIGNAL
  Instrument: NIFTY 50
  Timeframe: 1-Minute
  Time: 2026-10-31 10:15:00 IST
  Price: 24150.35
  Confidence: 0.82
  ```
- **FR-4.5** The system shall send alerts using a Telegram Bot token and chat ID from environment variables, via the `requests` library (HTTP call to the Telegram Bot API) or `python-telegram-bot`.
- **FR-4.6** The system shall log alert send failures but must not crash the streaming loop if Telegram is unreachable.

### 3.5 Server & Deployment
- **FR-5.1** A lightweight Flask server shall bind to `0.0.0.0` on the port from `$PORT` and expose a `GET /health` endpoint returning HTTP 200 with a simple JSON body.
- **FR-5.2** The WebSocket streaming + inference loop shall run in a background thread, started at application startup, so the Flask worker and the stream coexist in one process.
- **FR-5.3** A `requirements.txt` shall pin all dependencies.
- **FR-5.4** The service shall be deployable to Render as a Web Service with a gunicorn start command (`gunicorn --workers 1 app:app`).
- **FR-5.5** No API keys, tokens, or secrets shall be hardcoded. All shall come from environment variables.

## 4. Non-Functional Requirements

- **NFR-1 Performance:** Feature recomputation on candle close shall complete in under 200 ms for 1,000 candles.
- **NFR-2 Reliability:** The WebSocket client shall auto-reconnect indefinitely (with capped backoff) without operator intervention.
- **NFR-3 Consistency:** Training and live feature engineering MUST use the identical function so feature columns and order never drift. A single shared `features.py` module enforces this.
- **NFR-4 Data Integrity:** `TimeSeriesSplit` must be used for cross-validation to prevent look-ahead leakage. Feature rows containing NaN from indicator warm-up shall be dropped before training.
- **NFR-5 Observability:** The system shall log to stdout (structured, timestamped) for Render log capture: connections, candle closes, predictions, alerts, and errors.
- **NFR-6 Security:** Secrets are environment variables only; `.env.example` documents them without real values; `.gitignore` excludes `.env`, `model.pkl`, and `data/`.
- **NFR-7 Portability:** Code shall run on Python 3.10+ and be OS-agnostic for local development, while targeting the Render runtime for deployment.

## 5. Constraints & Assumptions

- Upstox 1-minute historical data is available from January 2022 and capped at 1 month per API request — pagination is mandatory for multi-month ranges.
- The Upstox access token expires daily (~03:30 IST); the operator is responsible for refreshing it (manual re-auth), and the system alerts on 401.
- Market hours are IST, roughly 09:15–15:30; outside hours the feed may be idle and no candles close — the system shall handle this gracefully.
- `ta-lib` requires a native TA-Lib C library installed on the system; this is a deployment prerequisite.
- Instrument is configurable; default NIFTY 50 index uses instrument key `NSE_INDEX|Nifty 50`. F&O contract instrument keys can be substituted.

## 6. Environment Variables

| Variable | Purpose | Required |
|---|---|---|
| `UPSTOX_ACCESS_TOKEN` | Bearer token for Upstox API calls | Yes (at runtime) |
| `UPSTOX_API_KEY` | OAuth client id (for the optional auth helper) | Optional |
| `UPSTOX_API_SECRET` | OAuth client secret (for the optional auth helper) | Optional |
| `UPSTOX_INSTRUMENT_KEY` | Instrument to track (default `NSE_INDEX\|Nifty 50`) | No |
| `TELEGRAM_BOT_TOKEN` | Telegram bot token for alerts | Yes |
| `TELEGRAM_CHAT_ID` | Target Telegram chat/channel id | Yes |
| `PORT` | Port the Flask server binds to (set by Render) | Yes (Render) |
| `MODEL_PATH` | Path to model bundle (default `model.pkl`) | No |
| `SIGNAL_THRESHOLD` | Forward-return threshold for labels (default ATR-scaled) | No |
| `LOG_LEVEL` | Logging verbosity (default `INFO`) | No |

## 7. Out of Scope

- Order placement, modification, or cancellation of any kind — this is alert-only.
- Portfolio management, position sizing, or backtesting UI.
- A user-facing web dashboard beyond the `/health` endpoint.
- Real-time charting or a frontend.
- Multi-instrument simultaneous streaming (single instrument per instance by design; multiple instances can be run for more instruments).
- Automated OAuth refresh without operator involvement (token refresh is manual).

## 8. High-Level Acceptance Criteria

1. `train_model.py` runs end-to-end, fetches ≥1 year of 1-minute candles, prints `TimeSeriesSplit` CV metrics, and writes `model.pkl`.
2. The live agent connects to the Upstox WebSocket, aggregates ticks into 1-minute candles, and logs each candle close.
3. On a candle close, the model is queried; a `1` or `-1` prediction sends a Telegram message matching FR-4.4.
4. `GET /health` returns 200; the service starts with gunicorn on `$PORT`.
5. No secret appears anywhere in source; all come from environment variables / `.env`.
6. Killing the WebSocket connection triggers an automatic reconnect within the configured backoff window.
