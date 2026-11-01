"""Tests for src/label.py — forward-return labels (synthetic, no talib needed).

Uses an explicit numeric ``threshold`` so the ATR path (which requires talib)
is never exercised. This keeps the test independent of the native TA-Lib build.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.label import make_labels


def _ohlcv_from_closes(closes, start="2026-01-05 09:15"):
    idx = pd.date_range(start, periods=len(closes), freq="1min", tz="Asia/Kolkata")
    c = pd.Series(closes, dtype=float)
    return pd.DataFrame(
        {
            "open": c.values,
            "high": (c + 0.5).values,
            "low": (c - 0.5).values,
            "close": c.values,
            "volume": np.full(len(c), 1000.0),
        },
        index=idx,
    )


def test_trending_series_produces_buys():
    # Monotonically rising closes: every forward 5-candle return is positive
    # and large, so all labels (except the tail) should be 1 (Buy).
    closes = np.linspace(100.0, 200.0, 60)
    df = _ohlcv_from_closes(closes)
    labels = make_labels(df, horizon=5, threshold=0.001)
    # Drop the last `horizon` rows (no forward target).
    body = labels.iloc[:-5]
    assert (body == 1).all(), f"expected all Buys, got {body.value_counts().to_dict()}"


def test_falling_series_produces_sells():
    closes = np.linspace(200.0, 100.0, 60)  # falling
    df = _ohlcv_from_closes(closes)
    labels = make_labels(df, horizon=5, threshold=0.001)
    body = labels.iloc[:-5]
    assert (body == -1).all(), f"expected all Sells, got {body.value_counts().to_dict()}"


def test_flat_series_produces_holds():
    # Perfectly flat: forward return == 0, within threshold => Hold (0).
    closes = np.full(60, 150.0)
    df = _ohlcv_from_closes(closes)
    labels = make_labels(df, horizon=5, threshold=0.001)
    body = labels.iloc[:-5]
    assert (body == 0).all(), f"expected all Holds, got {body.value_counts().to_dict()}"


def test_mixed_series_has_all_three_classes():
    # Rise then fall: the early part should have Buys, the late part Sells,
    # with at least some Holds around the turn (depending on threshold).
    closes = np.concatenate(
        [np.linspace(100.0, 150.0, 30), np.linspace(150.0, 100.0, 30)]
    )
    df = _ohlcv_from_closes(closes)
    labels = make_labels(df, horizon=5, threshold=0.005)
    body = labels.iloc[:-5]
    assert (body == 1).any() and (body == -1).any(), "expected both Buys and Sells"


def test_labels_in_valid_set_and_indexed():
    closes = np.linspace(100.0, 160.0, 50)
    df = _ohlcv_from_closes(closes)
    labels = make_labels(df, horizon=5, threshold=0.001)
    assert set(labels.unique()).issubset({-1, 0, 1})
    assert labels.index.equals(df.index)
    assert labels.name == "label"


def test_requires_close_column():
    bad = pd.DataFrame({"x": [1, 2, 3]})
    try:
        make_labels(bad, threshold=0.001)
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_horizon_must_be_positive():
    df = _ohlcv_from_closes(np.linspace(100, 110, 20))
    try:
        make_labels(df, horizon=0, threshold=0.001)
        assert False, "expected ValueError"
    except ValueError:
        pass
