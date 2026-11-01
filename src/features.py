"""SHARED feature engineering (single source of truth — NFR-3).

Both ``train_model.py``/``model_trainer.py`` and the live WebSocket pipeline
import ``compute_features`` and ``FEATURE_COLUMNS`` from this module. Feature
logic is NEVER duplicated in either pipeline.

``compute_features(df)`` takes a DataFrame with columns
``['open','high','low','close','volume']`` (datetime index) and returns a
DataFrame indexed identically to the input, with the engineered columns. It
uses `ta-lib` (``talib.RSI``, ``talib.MACD``, ``talib.BBANDS``, ``talib.EMA``,
``talib.ATR``) per the spec. NaN warm-up rows are NOT dropped here — the caller
controls that boundary (and drops the label-horizon tail rows).

Note on talib availability: ``import talib`` is performed lazily inside
``compute_features`` so that importing this module (and therefore
``FEATURE_COLUMNS``) works even when the native TA-Lib C library is not
installed. Only feature *computation* (and thus the training pipeline) require
the C library. This keeps inference, config, and the live skeleton importable
without it, while the code still targets the ``talib`` API exactly as the spec
mandates (it does NOT fall back to another library).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import (
    ATR_PERIOD,
    BBANDS_NBDEV,
    BBANDS_PERIOD,
    EMA_PERIODS,
    MACD_FAST,
    MACD_SIGNAL,
    MACD_SLOW,
    RSI_PERIOD,
)

# Exact ordered list of feature column names the model expects. This is saved
# into model.pkl and validated by InferenceEngine before predicting. Changing
# the order or names here changes the model contract.
FEATURE_COLUMNS: list[str] = [
    "rsi_14",
    "macd",
    "macd_signal",
    "macd_hist",
    "bb_upper",
    "bb_middle",
    "bb_lower",
    "bb_width",
    "ema_9",
    "ema_21",
    "ema_50",
    "atr_14",
    "return_1",
    "return_5",
    # --- richer features (temporal context + volatility regime + time-of-day) ---
    "return_1_lag1",
    "return_1_lag2",
    "return_1_lag3",
    "rsi_14_lag1",
    "vol_20",
    "sin_time",
    "cos_time",
]

_REQUIRED_COLUMNS: list[str] = ["open", "high", "low", "close", "volume"]


def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    """Engineer the TA features for an OHLCV DataFrame.

    Args:
        df: DataFrame with columns ``['open','high','low','close','volume']``
            and a (ideally datetime) index.

    Returns:
        A DataFrame indexed identically to ``df`` whose columns are exactly
        ``FEATURE_COLUMNS`` (in that order). Warm-up rows contain NaN; they are
        intentionally kept so the caller controls the drop boundary.

    Raises:
        ValueError: if required OHLCV columns are missing.
        ImportError: if the native TA-Lib C library / ``talib`` package is not
            installed (the documented deployment prerequisite).
    """
    missing = [c for c in _REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"compute_features requires columns {_REQUIRED_COLUMNS}; missing: {missing}")

    # Lazily import talib so FEATURE_COLUMNS remains importable without the C lib.
    try:
        import talib
    except ImportError as exc:  # pragma: no cover - environment-dependent
        raise ImportError(
            "talib could not be imported. Install the native TA-Lib C library "
            "first (see README.md and build.sh), then `pip install TA-Lib`."
        ) from exc

    high = np.asarray(df["high"], dtype=float)
    low = np.asarray(df["low"], dtype=float)
    close = np.asarray(df["close"], dtype=float)

    # RSI (14)
    rsi_14 = talib.RSI(close, timeperiod=RSI_PERIOD)

    # MACD (12, 26, 9) -> (macd, macdsignal, macdhist)
    macd, macd_signal, macd_hist = talib.MACD(
        close, fastperiod=MACD_FAST, slowperiod=MACD_SLOW, signalperiod=MACD_SIGNAL
    )

    # Bollinger Bands (20, 2) -> (upper, middle, lower)
    bb_upper, bb_middle, bb_lower = talib.BBANDS(
        close, timeperiod=BBANDS_PERIOD, nbdevup=BBANDS_NBDEV, nbdevdn=BBANDS_NBDEV
    )
    bb_width = bb_upper - bb_lower

    # EMA (9 / 21 / 50)
    ema = {}
    for p in EMA_PERIODS:
        ema[p] = talib.EMA(close, timeperiod=p)

    # ATR (14)
    atr_14 = talib.ATR(high, low, close, timeperiod=ATR_PERIOD)

    # Log returns (1- and 5-candle). shift preserves the index alignment.
    close_s = df["close"].astype(float)
    return_1 = np.log(close_s / close_s.shift(1))
    return_5 = np.log(close_s / close_s.shift(5))

    # --- richer features: temporal context + volatility regime + time-of-day ---
    # Lagged 1-min returns give the model recent momentum context.
    return_1_lag1 = return_1.shift(1)
    return_1_lag2 = return_1.shift(2)
    return_1_lag3 = return_1.shift(3)
    # Lagged RSI (momentum one candle ago); rsi_14 is a numpy array -> Series to shift.
    rsi_14_lag1 = pd.Series(rsi_14, index=df.index).shift(1)
    # Rolling 20-candle volatility of 1-min returns (volatility regime clustering).
    vol_20 = return_1.rolling(20).std()
    # Cyclical time-of-day encoding (NSE open/close sessions have real structure).
    if isinstance(df.index, pd.DatetimeIndex):
        minute_of_day = pd.Series(df.index.hour * 60 + df.index.minute, index=df.index).astype(float)
    else:
        minute_of_day = pd.Series(0.0, index=df.index)
    sin_time = np.sin(2.0 * np.pi * minute_of_day / 1440.0)
    cos_time = np.cos(2.0 * np.pi * minute_of_day / 1440.0)

    out = pd.DataFrame(index=df.index)
    out["rsi_14"] = rsi_14
    out["macd"] = macd
    out["macd_signal"] = macd_signal
    out["macd_hist"] = macd_hist
    out["bb_upper"] = bb_upper
    out["bb_middle"] = bb_middle
    out["bb_lower"] = bb_lower
    out["bb_width"] = bb_width
    out["ema_9"] = ema[9]
    out["ema_21"] = ema[21]
    out["ema_50"] = ema[50]
    out["atr_14"] = atr_14
    out["return_1"] = return_1.values
    out["return_5"] = return_5.values
    out["return_1_lag1"] = return_1_lag1.values
    out["return_1_lag2"] = return_1_lag2.values
    out["return_1_lag3"] = return_1_lag3.values
    out["rsi_14_lag1"] = rsi_14_lag1.values
    out["vol_20"] = vol_20.values
    out["sin_time"] = sin_time.values
    out["cos_time"] = cos_time.values

    # Return columns in the exact FEATURE_COLUMNS order.
    return out[FEATURE_COLUMNS]
