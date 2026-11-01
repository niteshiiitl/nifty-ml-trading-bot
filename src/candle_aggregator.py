"""Tick -> 1-minute OHLCV candle aggregator (live pipeline, FR-3.4 / FR-3.5).

``CandleAggregator`` maintains the in-progress candle for the current minute
and a rolling buffer of the last ``max_candles`` closed candles (a pandas
DataFrame). ``on_tick`` is called from the WebSocket thread; a ``threading.Lock``
guards state defensively.

A candle is finalized when a tick arrives whose minute is later than the
in-progress candle's minute; on that close, ``on_tick`` returns a closed-candle
event ``{"candle": {...}, "df": <updated DataFrame>}`` so the caller can run
inference + alerts. Otherwise it returns ``None``.

This module performs no trading — it only aggregates market data.
"""

from __future__ import annotations

import threading
from typing import Optional

import pandas as pd

COLUMNS: list[str] = ["open", "high", "low", "close", "volume"]
TIMEZONE = "Asia/Kolkata"


class CandleAggregator:
    """Aggregate live ticks into 1-minute OHLCV candles with a rolling buffer.

    Args:
        max_candles: Keep at most this many closed candles in the buffer (default 1000).
    """

    def __init__(self, max_candles: int = 1000) -> None:
        self.max_candles = max_candles
        self._lock = threading.Lock()
        # In-progress candle: {ts: pd.Timestamp(minute), open, high, low, close, volume}.
        self._current: Optional[dict] = None
        # Rolling buffer of closed candles (tz-aware DatetimeIndex).
        self.candles: pd.DataFrame = pd.DataFrame(columns=COLUMNS)

    def on_tick(self, tick: dict) -> Optional[dict]:
        """Process one tick; return a closed-candle event on minute rollover.

        Args:
            tick: ``{"ltp": float, "ltt": <epoch seconds>, "ltq": <int/float>}``.

        Returns:
            ``{"candle": <closed-candle dict>, "df": <updated DataFrame>}`` when a
            candle closed, else ``None``. Stale ticks (older than the current
            minute) are ignored.
        """
        ltp = float(tick["ltp"])
        ltq = float(tick.get("ltq", 0.0) or 0.0)
        ltt = tick["ltt"]
        minute = pd.Timestamp(float(ltt), unit="s", tz=TIMEZONE).floor("5min")

        with self._lock:
            cur = self._current
            if cur is None:
                self._current = self._start_candle(minute, ltp, ltq)
                return None

            if minute == cur["ts"]:
                # Same minute: update OHLCV in place.
                if ltp > cur["high"]:
                    cur["high"] = ltp
                if ltp < cur["low"]:
                    cur["low"] = ltp
                cur["close"] = ltp
                cur["volume"] += ltq
                return None

            if minute > cur["ts"]:
                # New minute: close the current candle, append, start a new one.
                closed = self._close_candle(cur)
                self._append(closed)
                self._current = self._start_candle(minute, ltp, ltq)
                return {"candle": closed, "df": self.candles.copy()}

            # minute < cur["ts"]: a stale/backdated tick — ignore it.
            return None

    def latest_closed(self) -> Optional[pd.Series]:
        """Return the most recent closed candle as a Series, or None if empty."""
        with self._lock:
            if self.candles.empty:
                return None
            return self.candles.iloc[-1]

    def snapshot(self) -> pd.DataFrame:
        """Return a defensive copy of the closed-candle buffer."""
        with self._lock:
            return self.candles.copy()

    # --- internals ---
    @staticmethod
    def _start_candle(minute: pd.Timestamp, ltp: float, ltq: float) -> dict:
        return {
            "ts": minute,
            "open": ltp,
            "high": ltp,
            "low": ltp,
            "close": ltp,
            "volume": ltq,
        }

    def _close_candle(self, cur: dict) -> dict:
        """Convert the in-progress dict to a closed-candle record (with timestamp)."""
        return {
            "timestamp": cur["ts"],
            "open": float(cur["open"]),
            "high": float(cur["high"]),
            "low": float(cur["low"]),
            "close": float(cur["close"]),
            "volume": float(cur["volume"]),
        }

    def _append(self, closed: dict) -> None:
        """Append a closed candle to the rolling buffer, capping at max_candles."""
        ts = closed["timestamp"]
        row = pd.DataFrame(
            [[closed["open"], closed["high"], closed["low"], closed["close"], closed["volume"]]],
            index=pd.DatetimeIndex([ts]),
            columns=COLUMNS,
        )
        self.candles = pd.concat([self.candles, row])
        if len(self.candles) > self.max_candles:
            self.candles = self.candles.iloc[-self.max_candles:]
