"""Tests for src/candle_aggregator.py — tick -> 1-min OHLCV, rolling buffer."""
from __future__ import annotations

import pandas as pd

from src.candle_aggregator import CandleAggregator


def _tick(epoch, ltp, ltq=10.0):
    return {"ltp": ltp, "ltt": epoch, "ltq": ltq}


def test_three_minutes_emit_two_closed_candles():
    # Minute 1: 09:15:00..09:15:59 (epoch base), Minute 2: 09:16, Minute 3: 09:17.
    base = pd.Timestamp("2026-01-05 09:15:00", tz="Asia/Kolkata")
    m1 = int(base.timestamp())
    m2 = m1 + 60
    m3 = m2 + 60

    agg = CandleAggregator(max_candles=1000)
    # Minute 1 ticks (prices 100, 101, 99, 102)
    events = []
    for p in (100.0, 101.0, 99.0, 102.0):
        events.append(agg.on_tick(_tick(m1 + 5, p)))
    # Minute 2 first tick closes minute 1
    events.append(agg.on_tick(_tick(m2 + 5, 103.0)))
    # Minute 2 ticks
    for p in (103.0, 105.0, 104.0):
        events.append(agg.on_tick(_tick(m2 + 20, p)))
    # Minute 3 first tick closes minute 2
    events.append(agg.on_tick(_tick(m3 + 5, 106.0)))

    closed = [e for e in events if e is not None]
    assert len(closed) == 2, f"expected 2 closed candles, got {len(closed)}"

    c1 = closed[0]["candle"]
    assert c1["open"] == 100.0
    assert c1["high"] == 102.0
    assert c1["low"] == 99.0
    assert c1["close"] == 102.0  # last price in minute 1
    assert c1["volume"] == 40.0  # 4 ticks * 10

    c2 = closed[1]["candle"]
    assert c2["open"] == 103.0
    assert c2["high"] == 105.0
    assert c2["low"] == 103.0
    assert c2["close"] == 104.0
    assert c2["volume"] == 40.0  # 4 ticks in minute 2 (incl. the first that opened)


def test_buffer_grows_then_caps_at_max():
    agg = CandleAggregator(max_candles=3)
    base = int(pd.Timestamp("2026-01-05 09:15:00", tz="Asia/Kolkata").timestamp())
    # Emit 5 closed candles (minutes 1..5) by sending the first tick of each next minute.
    for i in range(6):
        agg.on_tick(_tick(base + i * 60, 100.0 + i))
    # 5 minute-transitions => 5 closed candles, but buffer caps at 3.
    assert len(agg.candles) == 3
    # 6 ticks => 5 closed candles (09:15..09:19); buffer keeps the newest 3
    # (09:17, 09:18, 09:19). Minute 09:20 is in-progress and never closed (no later tick).
    assert agg.candles.index[0].minute == 17  # 3rd closed candle -> 09:17
    assert agg.candles.index[-1].minute == 19  # 5th closed candle -> 09:19


def test_no_event_within_same_minute():
    agg = CandleAggregator()
    m1 = int(pd.Timestamp("2026-01-05 09:15:00", tz="Asia/Kolkata").timestamp())
    assert agg.on_tick(_tick(m1 + 10, 100.0)) is None
    assert agg.on_tick(_tick(m1 + 20, 101.0)) is None
    assert agg.on_tick(_tick(m1 + 50, 102.0)) is None


def test_stale_tick_ignored():
    agg = CandleAggregator()
    m1 = int(pd.Timestamp("2026-01-05 09:15:00", tz="Asia/Kolkata").timestamp())
    m2 = m1 + 60
    agg.on_tick(_tick(m2 + 5, 100.0))  # starts minute 2
    stale = agg.on_tick(_tick(m1 + 10, 50.0))  # backdated to minute 1
    assert stale is None
    # In-progress candle unaffected by the stale tick.
    assert agg._current["low"] == 100.0


def test_snapshot_returns_dataframe_copy():
    agg = CandleAggregator()
    snap = agg.snapshot()
    assert isinstance(snap, pd.DataFrame)
    assert list(snap.columns) == ["open", "high", "low", "close", "volume"]


def test_latest_closed_returns_last():
    agg = CandleAggregator()
    base = int(pd.Timestamp("2026-01-05 09:15:00", tz="Asia/Kolkata").timestamp())
    for i in range(3):
        agg.on_tick(_tick(base + i * 60, 100.0 + i))
    latest = agg.latest_closed()
    assert latest is not None
    assert latest["close"] == 101.0  # minute 2 closed with price 101
