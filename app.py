"""Gunicorn entry point with automatic intraday historical backfilling (FR-5.2 / FR-5.4 / FR-5.5).

On import: build ``Config``, load the model (degraded-safe), build the
aggregator + alerter + status, pre-load today's 5-minute candles via Upstox REST API,
wire the ``LiveStreamEngine``, and start its ``run_forever()`` daemon thread.
"""

from __future__ import annotations

import logging
import os
import threading
import time
import requests
import pandas as pd

from src.alerts import TelegramAlerter
from src.candle_aggregator import CandleAggregator
from src.config import (
    Config,
    MAX_RECONNECT_BACKOFF_SECONDS,
    MIN_RECONNECT_BACKOFF_SECONDS,
    configure_logging,
)
from src.features import compute_features
from src.inference import load_or_none
from src.server import StreamStatus, create_app

configure_logging(os.environ.get("LOG_LEVEL", "INFO"))
logger = logging.getLogger(__name__)


def preload_historical_candles(config: Config, aggregator: CandleAggregator) -> None:
    """Pre-load today's 5-minute intraday candles from Upstox V3 REST API to bypass warm-up delay."""
    token = config.access_token
    if not token:
        logger.warning("No access token available for historical pre-load; starting with empty aggregator.")
        api_key = config.instrument_key
        # URL encode pipe if needed
        encoded_key = api_key.replace("|", "%7C")
        url = f"https://api.upstox.com/v3/historical-candle/intraday/{encoded_key}/minutes/5"
        headers = {"Accept": "application/json", "Authorization": f"Bearer {token}"}
        try:
            logger.info("Fetching historical 5-minute intraday candles for %s...", api_key)
            resp = requests.get(url, headers=headers, timeout=15)
            if resp.status_code == 200:
                data = resp.json().get("data", {})
                candles_list = data.get("candles", [])
                if candles_list:
                    # Upstox intraday format: [timestamp, open, high, low, close, volume, oi]
                    rows = []
                    for c in candles_list:
                        ts = pd.Timestamp(c[0])
                        if ts.tz is None:
                            ts = ts.tz_localize("Asia/Kolkata")
                        else:
                            ts = ts.tz_convert("Asia/Kolkata")
                        rows.append({
                            "timestamp": ts,
                            "open": float(c[1]),
                            "high": float(c[2]),
                            "low": float(c[3]),
                            "close": float(c[4]),
                            "volume": float(c[5])
                        })
                    if rows:
                        df_hist = pd.DataFrame(rows).set_index("timestamp").sort_index()
                        with aggregator._lock:
                            aggregator.candles = pd.concat([aggregator.candles, df_hist]).drop_duplicates().tail(aggregator.max_candles)
                        logger.info("Successfully pre-loaded %d historical candles into aggregator buffer.", len(df_hist))
                else:
                    logger.warning("Historical pre-load returned zero candles.")
            else:
                logger.warning("Historical pre-load API returned status %d: %s", resp.status_code, resp.text[:150])
        except Exception as exc:
            logger.exception("Failed to pre-load historical candles: %s", exc)


class LiveStreamEngine:
    def __init__(self, config, aggregator, inference_engine, alerter, status) -> None:
        self.config = config
        self.aggregator = aggregator
        self.inference_engine = inference_engine
        self.alerter = alerter
        self.status = status

    def on_candle_close(self, closed_event: dict) -> None:
        try:
            df = self.aggregator.snapshot()
            if df is None or df.empty:
                return
            
            feats = compute_features(df)
            last = feats.iloc[[-1]]
            
            if last.isna().any().any():
                logger.info("Warming up: %d candles collected so far (NaN present in features); skipping prediction.", len(df))
                return
                
            if self.inference_engine is None:
                logger.warning("Inference engine unavailable (degraded); skipping prediction.")
                return
                
            label, confidence = self.inference_engine.predict(last)
            price = float(df.iloc[-1]["close"])
            ts = df.index[-1]
            self.status.update(last_candle=ts, last_signal=label, last_confidence=confidence)
            logger.info("Candle close %s | pred=%s conf=%.2f", ts, label, confidence)
            
            if label in (1, -1):
                self.alerter.send_alert(label, self.config.instrument_key, price, ts, confidence)
                
        except ImportError as exc:
            logger.error("Feature computation requires native TA-Lib (unavailable): %s", exc)
        except Exception as exc:
            logger.exception("on_candle_close error: %s", exc)

    def _on_auth_error(self, message: str) -> None:
        logger.error("Re-authentication required: %s", message)
        try:
            self.alerter.send_message(f"⚠️ Re-authentication required: {message}")
        except Exception as exc:
            logger.warning("Failed to send re-auth alert: %s", exc)

    def run_forever(self) -> None:
        missing = self.config.require_secrets()
        if missing:
            logger.error("Missing required live secrets: %s.", ", ".join(missing))
            while True:
                time.sleep(3600)
            return

        # Pre-load historical candles before opening the websocket loop
        preload_historical_candles(self.config, self.aggregator)

        backoff = MIN_RECONNECT_BACKOFF_SECONDS
        while True:
            try:
                from src.websocket_client import UpstoxMarketDataClient
                client = UpstoxMarketDataClient(
                    self.config,
                    self.aggregator,
                    self.on_candle_close,
                    on_auth_error=self._on_auth_error,
                )
                client.run_forever()
            except KeyboardInterrupt:
                raise
            except Exception as exc:
                logger.exception("Stream engine error: %s. Backoff %.1fs.", exc, backoff)
                time.sleep(backoff)
                backoff = min(backoff * 2, MAX_RECONNECT_BACKOFF_SECONDS)


config: Config = Config.from_env()
status: StreamStatus = StreamStatus()
status.instrument = config.instrument_key
if config.require_secrets():
    status.degraded = True

inference_engine = load_or_none(config.model_path)
if inference_engine is None:
    status.degraded = True

aggregator = CandleAggregator(max_candles=1000)
alerter = TelegramAlerter(
    config.telegram_bot_token, config.telegram_chat_id, config.alert_cooldown_seconds
)

app = create_app(config.instrument_key, status)

@app.route('/')
def home():
    return "Nifty 50 ML Trading Bot is Live and Running!", 200

_started = False

def _start_stream() -> None:
    global _started
    if _started:
        return
    _started = True
    engine = LiveStreamEngine(config, aggregator, inference_engine, alerter, status)
    t = threading.Thread(target=engine.run_forever, name="live-stream", daemon=True)
    t.start()
    logger.info("Started live-stream daemon thread.")

try:
    _start_stream()
except Exception as exc:
    logger.exception("Failed to start stream thread: %s.", exc)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)