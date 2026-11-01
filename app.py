"""Gunicorn entry point (FR-5.2 / FR-5.4 / FR-5.5).

On import: build ``Config``, load the model (degraded-safe), build the
aggregator + alerter + status, wire the ``LiveStreamEngine``, and start its
``run_forever()`` in a daemon background thread (once-guarded). Expose ``app``
for gunicorn (``gunicorn --workers 1 app:app``).

Resilience: if required secrets are missing (or the model is absent) the Flask
app STILL boots and ``/health`` returns 200; the stream thread logs the
missing-secret/model error and stays alive (does not crash the process).

Alert-only: no order placement, modification, or cancellation anywhere.
"""

from __future__ import annotations

import logging
import os
import threading
import time

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


class LiveStreamEngine:
    """Glue: WS client + aggregator + inference + alerter + status.

    ``run_forever()`` is the daemon-thread entry point. If required secrets are
    missing it logs the error and stays alive (sleep-loop) so ``/health`` keeps
    serving — it never crashes the process.
    """

    def __init__(self, config, aggregator, inference_engine, alerter, status) -> None:
        self.config = config
        self.aggregator = aggregator
        self.inference_engine = inference_engine
        self.alerter = alerter
        self.status = status

    def on_candle_close(self, closed_event: dict) -> None:
        """Recompute features on the rolling buffer, predict, alert on 1 / -1."""
        try:
            df = self.aggregator.snapshot()
            if df is None or df.empty:
                return
            # `compute_features` raises ImportError if the native TA-Lib lib is
            # missing — caught below so the stream stays alive.
            feats = compute_features(df)
            last = feats.iloc[[-1]]
            if last.isna().any().any():
                logger.debug("Latest candle features still warming up (NaN); skipping prediction.")
                return
            if self.inference_engine is None:
                logger.debug("Inference engine unavailable (degraded); skipping prediction.")
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
        except Exception as exc:  # noqa: BLE001 - never crash the stream loop
            logger.exception("on_candle_close error: %s", exc)

    def _on_auth_error(self, message: str) -> None:
        """Called by the WS client on 401 — alert the operator to re-auth (FR-1.4)."""
        logger.error("Re-authentication required: %s", message)
        try:
            self.alerter.send_message(f"⚠️ Re-authentication required: {message}")
        except Exception as exc:  # noqa: BLE001 - never raise into the WS loop
            logger.warning("Failed to send re-auth alert: %s", exc)

    def run_forever(self) -> None:
        missing = self.config.require_secrets()
        if missing:
            logger.error(
                "Missing required live secrets: %s. Live streaming disabled; "
                "/health remains up. Set the vars and restart.",
                ", ".join(missing),
            )
            # Stay alive so /health keeps serving; do not crash the process.
            while True:
                time.sleep(3600)
            return

        backoff = MIN_RECONNECT_BACKOFF_SECONDS
        while True:
            try:
                # Lazy import so app.py imports without websocket-client installed.
                from src.websocket_client import UpstoxMarketDataClient

                client = UpstoxMarketDataClient(
                    self.config,
                    self.aggregator,
                    self.on_candle_close,
                    on_auth_error=self._on_auth_error,
                )
                client.run_forever()  # internal backoff loop; returns on drop
            except KeyboardInterrupt:
                raise
            except Exception as exc:  # noqa: BLE001 - top-level guard
                logger.exception("Stream engine error: %s. Backoff %.1fs.", exc, backoff)
                time.sleep(backoff)
                backoff = min(backoff * 2, MAX_RECONNECT_BACKOFF_SECONDS)


# --- module-level wiring (runs once on import by gunicorn) ---
config: Config = Config.from_env()

status: StreamStatus = StreamStatus()
status.instrument = config.instrument_key
if config.require_secrets():
    status.degraded = True

# Model is optional at boot (degraded if missing) — never crash on load.
inference_engine = load_or_none(config.model_path)
if inference_engine is None:
    status.degraded = True

aggregator = CandleAggregator(max_candles=1000)
alerter = TelegramAlerter(
    config.telegram_bot_token, config.telegram_chat_id, config.alert_cooldown_seconds
)

app = create_app(config.instrument_key, status)

_started = False


def _start_stream() -> None:
    """Start the live stream daemon thread once, guarded against double-start."""
    global _started
    if _started:
        return
    _started = True
    engine = LiveStreamEngine(config, aggregator, inference_engine, alerter, status)
    t = threading.Thread(target=engine.run_forever, name="live-stream", daemon=True)
    t.start()
    logger.info("Started live-stream daemon thread.")


# Start the stream thread at import time (gunicorn loads this module once per worker).
try:
    _start_stream()
except Exception as exc:  # noqa: BLE001 - never block app boot
    logger.exception("Failed to start stream thread: %s. /health still served.", exc)


if __name__ == "__main__":  # pragma: no cover
    port = int(os.environ.get("PORT", 5000))
    logger.info("Running Flask directly on 0.0.0.0:%d", port)
    app.run(host="0.0.0.0", port=port)
