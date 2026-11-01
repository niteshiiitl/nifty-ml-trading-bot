"""Tests for src/features.py — shared TA feature computation.

Requires the native TA-Lib C library + the `talib` Python package. If talib is
not installed (the documented deployment prerequisite), the whole module is
skipped so the rest of the suite still runs.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

# Skip the entire module if the native TA-Lib is unavailable (documented gap).
pytest.importorskip("talib")

from src.features import FEATURE_COLUMNS, compute_features  # noqa: E402


def _synthetic_ohlcv(rows: int = 200, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    # A gentle upward random walk so indicators are well-defined.
    steps = rng.normal(loc=0.0, scale=1.0, size=rows)
    price = 100.0 + np.cumsum(steps)
    idx = pd.date_range("2026-01-05 09:15", periods=rows, freq="1min", tz="Asia/Kolkata")
    df = pd.DataFrame(
        {
            "open": price,
            "high": price + rng.uniform(0.0, 1.5, size=rows),
            "low": price - rng.uniform(0.0, 1.5, size=rows),
            "close": price + rng.normal(0.0, 0.5, size=rows),
            "volume": rng.integers(100, 10_000, size=rows).astype(float),
        },
        index=idx,
    )
    return df


def test_compute_features_returns_all_feature_columns():
    df = _synthetic_ohlcv(200)
    feats = compute_features(df)
    assert list(feats.columns) == FEATURE_COLUMNS
    # All FEATURE_COLUMNS present exactly.
    assert set(feats.columns) == set(FEATURE_COLUMNS)
    assert len(feats) == len(df)


def test_compute_features_index_identical_to_input():
    df = _synthetic_ohlcv(200)
    feats = compute_features(df)
    assert feats.index.equals(df.index)


def test_compute_features_has_warmup_nans():
    """The first ~50 rows should contain NaN warm-up from the indicators."""
    df = _synthetic_ohlcv(200)
    feats = compute_features(df)
    # EMA(50) needs 50 samples; RSI(14) ~14; ATR(14) ~14. The 50-period EMA
    # dominates the warm-up, so early rows must contain at least one NaN.
    first_block = feats.iloc[:50]
    assert first_block.isna().any().any(), "expected NaN warm-up in the first ~50 rows"
    # The later block (after warm-up) should be fully populated.
    later_block = feats.iloc[60:]
    assert later_block.notna().all().all(), "expected no NaNs after indicator warm-up"


def test_compute_features_bb_width_is_upper_minus_lower():
    df = _synthetic_ohlcv(200)
    feats = compute_features(df)
    later = feats.dropna()
    diff = (later["bb_upper"] - later["bb_lower"])
    assert np.allclose(diff.values, later["bb_width"].values)


def test_compute_features_requires_ohlcv_columns():
    bad = pd.DataFrame({"a": [1, 2, 3]})
    with pytest.raises(ValueError):
        compute_features(bad)


def test_feature_columns_order_is_stable_contract():
    # The exact ordered contract used by the model bundle + inference.
    assert FEATURE_COLUMNS == [
        "rsi_14", "macd", "macd_signal", "macd_hist",
        "bb_upper", "bb_middle", "bb_lower", "bb_width",
        "ema_9", "ema_21", "ema_50", "atr_14",
        "return_1", "return_5",
    ]
