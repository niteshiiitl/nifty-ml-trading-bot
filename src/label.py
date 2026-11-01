"""Forward-return target labels for training (FR-2.6).

``make_labels(df, horizon=5, threshold=None)`` computes the forward return
over ``horizon`` candles and labels each row:

    1  (Buy)  if forward return >  threshold
   -1  (Sell) if forward return < -threshold
    0  (Hold) otherwise

If ``threshold`` is None, an ATR-scaled default is used (``factor * atr / close``)
so the threshold adapts to volatility. An explicit numeric ``threshold`` may be
passed to bypass the ATR path — this keeps the unit test independent of the
native TA-Lib library.

The caller is responsible for dropping the last ``horizon`` rows (no forward
target) and the NaN warm-up rows. ``make_labels`` does not drop them.
"""

from __future__ import annotations

from typing import Optional, Union

import numpy as np
import pandas as pd

from src.config import DEFAULT_THRESHOLD_ATR_FACTOR, LABEL_HORIZON


def make_labels(
    df: pd.DataFrame,
    horizon: int = LABEL_HORIZON,
    threshold: Optional[Union[float, pd.Series]] = None,
    atr_factor: float = DEFAULT_THRESHOLD_ATR_FACTOR,
) -> pd.Series:
    """Produce 1/-1/0 labels from forward price direction.

    Args:
        df: OHLCV DataFrame with a ``close`` column.
        horizon: Number of candles ahead for the forward return (default 5).
        threshold: Forward-return threshold. If a number, used directly. If
            None, an ATR-scaled Series threshold is computed as
            ``atr_factor * atr_14 / close`` (requires talib via features.py).
        atr_factor: Multiplier on ATR/close when threshold is None.

    Returns:
        A pandas Series of int labels (1, -1, 0) indexed like ``df``. The last
        ``horizon`` rows have NaN forward return and are labelled 0 by the
        element-wise rule (the caller should drop them).

    Raises:
        ValueError: if ``close`` is missing or horizon < 1.
        ImportError: if threshold is None and the native TA-Lib library is not
            installed (the ATR path needs features.compute_features).
    """
    if "close" not in df.columns:
        raise ValueError("make_labels requires a 'close' column.")
    if horizon < 1:
        raise ValueError("horizon must be >= 1.")

    close = df["close"].astype(float)
    forward_return = (close.shift(-horizon) - close) / close

    if threshold is None:
        # ATR-scaled, volatility-adaptive threshold. Lazy import so this module
        # imports cleanly without the native TA-Lib library; only the default
        # (no explicit threshold) path needs it.
        from src.features import compute_features  # local import: talib needed here

        atr = compute_features(df)["atr_14"]
        thr = atr_factor * atr / close
    else:
        thr = threshold

    # Element-wise rule, supporting both scalar and Series thresholds.
    labels = np.where(
        forward_return > thr,
        1,
        np.where(forward_return < -thr, -1, 0),
    )
    # Rows without a forward target (last `horizon`) become NaN forward_return;
    # NaN comparisons are False, so they fall to the 0 branch. The caller drops
    # them; we keep the index intact here.
    result = pd.Series(labels, index=df.index, name="label", dtype=int)
    return result
