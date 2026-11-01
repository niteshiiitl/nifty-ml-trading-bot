"""Flask health endpoint (FR-5.1).

A minimal Flask app exposing ``GET /health`` returning HTTP 200 with a JSON
body. ``last_candle`` is read from a shared, thread-safe ``StreamStatus``
object updated by the live stream engine. No other routes — intentionally
minimal (US-4).

Resilience: the app boots and ``/health`` returns 200 even when required
secrets are missing (the stream thread handles missing secrets and stays alive
separately in app.py). The ``degraded`` field reports degraded state for
observability but the status code stays 200.
"""

from __future__ import annotations

import threading
from typing import Any, Optional

from flask import Flask, jsonify


class StreamStatus:
    """Thread-safe holder for stream liveness state shared with ``/health``."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.instrument: Optional[str] = None
        self.last_candle: Optional[Any] = None  # pd.Timestamp or None
        self.last_signal: Optional[int] = None
        self.last_confidence: Optional[float] = None
        self.degraded: bool = False

    def update(
        self,
        last_candle: Optional[Any] = None,
        last_signal: Optional[int] = None,
        last_confidence: Optional[float] = None,
        degraded: Optional[bool] = None,
    ) -> None:
        with self._lock:
            if last_candle is not None:
                self.last_candle = last_candle
            if last_signal is not None:
                self.last_signal = int(last_signal)
            if last_confidence is not None:
                self.last_confidence = float(last_confidence)
            if degraded is not None:
                self.degraded = bool(degraded)

    def snapshot(self) -> dict:
        with self._lock:
            return {
                "instrument": self.instrument,
                "last_candle": None if self.last_candle is None else str(self.last_candle),
                "last_signal": self.last_signal,
                "last_confidence": self.last_confidence,
                "degraded": self.degraded,
            }


def create_app(instrument_key: Optional[str] = None, status: Optional[StreamStatus] = None) -> Flask:
    """Build a Flask app with ``GET /health`` returning 200 (even when degraded).

    Args:
        instrument_key: Instrument to report in the health payload.
        status: Shared ``StreamStatus`` updated by the stream thread. If None, a
            fresh local one is created.
    """
    app = Flask(__name__)
    if status is None:
        status = StreamStatus()
        status.instrument = instrument_key

    @app.get("/health")
    def health():
        snap = status.snapshot()
        return (
            jsonify(
                {
                    "status": "ok",
                    "instrument": snap["instrument"] or instrument_key,
                    "last_candle": snap["last_candle"],
                    "last_signal": snap["last_signal"],
                    "last_confidence": snap["last_confidence"],
                    "degraded": snap["degraded"],
                }
            ),
            200,
        )

    return app


# Module-level default app so ``from src.server import app`` works standalone.
app: Flask = create_app()
