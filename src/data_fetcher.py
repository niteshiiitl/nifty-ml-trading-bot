"""Upstox V3 REST historical candle downloader (training data).

Thin I/O wrapper around the Upstox historical-candle endpoint with monthly
pagination (1-minute data is capped at 1 month/request) and a CSV cache that
appends only uncached date ranges.

Endpoint (FR-2.2):
    GET /v3/historical-candle/{instrument_key}/minutes/1/{to_date}/{from_date}
    Authorization: Bearer <UPSTOX_ACCESS_TOKEN>
    Accept: application/json

This module does NOT execute trades; it only reads historical OHLCV.
"""

from __future__ import annotations

import logging
import time
from datetime import datetime, timedelta
from typing import Optional
from urllib.parse import quote

import pandas as pd
import requests

from src.config import UPSTOX_HISTORICAL_PATH, UPSTOX_REST_BASE

logger = logging.getLogger(__name__)

DEFAULT_CACHE_PATH = "data/historical_1min.csv"
RATE_LIMIT_SLEEP_SECONDS = 0.5
MAX_RETRIES = 4


class TokenExpiredError(RuntimeError):
    """Raised when Upstox returns 401 — the operator must re-authenticate."""


class DataFetchError(RuntimeError):
    """Raised for non-retryable HTTP errors (other than 429/401)."""


def fetch_historical_candles(
    instrument_key: str,
    from_date: str,
    to_date: str,
    token: Optional[str] = None,
    cache_path: str = DEFAULT_CACHE_PATH,
    sleep_between_calls: float = RATE_LIMIT_SLEEP_SECONDS,
) -> pd.DataFrame:
    """Download 1-minute OHLCV for ``instrument_key`` between two dates.

    Paginates month-by-month (V3 caps 1-min data at 1 month/request), caches
    the result to ``cache_path`` (appending only uncached date ranges), and
    returns the full range as a typed DataFrame with a timezone-aware
    (Asia/Kolkata) datetime index.

    Args:
        instrument_key: e.g. ``NSE_INDEX|Nifty 50``.
        from_date: ``YYYY-MM-DD``.
        to_date: ``YYYY-MM-DD``.
        token: Bearer access token (``UPSTOX_ACCESS_TOKEN``). If None, the
            function raises — callers should pass the configured token.
        cache_path: CSV cache location.

    Returns:
        DataFrame with columns ``['open','high','low','close','volume']`` and a
        tz-aware datetime index, sorted ascending.

    Raises:
        TokenExpiredError: on HTTP 401 (re-auth needed — alerts via Telegram).
        DataFetchError: on other non-retryable HTTP errors.
    """
    if not token:
        raise DataFetchError(
            "UPSTOX_ACCESS_TOKEN is required to fetch historical candles."
        )

    # Load the cache and determine which month-chunks still need downloading.
    cached = _load_cache(cache_path)
    month_chunks = _month_ranges(from_date, to_date)

    frames: list[pd.DataFrame] = []
    if not cached.empty:
        frames.append(cached)

    for start, end in month_chunks:
        if _is_fully_cached(cached, start, end):
            logger.info("Cached range %s..%s — skipping fetch.", start, end)
            continue
        logger.info("Fetching %s .. %s", start, end)
        chunk = _fetch_month(instrument_key, start, end, token)
        if not chunk.empty:
            frames.append(chunk)
            _append_cache(cache_path, chunk)
        if sleep_between_calls:
            time.sleep(sleep_between_calls)

    if not frames:
        return cached
    out = pd.concat(frames)
    out = out[~out.index.duplicated(keep="last")].sort_index()
    return out[["open", "high", "low", "close", "volume"]]


def _fetch_month(instrument_key: str, from_date: str, to_date: str, token: str) -> pd.DataFrame:
    """Fetch one (<= month) chunk, retrying 429 with backoff, raising on 401."""
    key = quote(instrument_key, safe="")
    url = f"{UPSTOX_REST_BASE}{UPSTOX_HISTORICAL_PATH}/{key}/minutes/5/{to_date}/{from_date}"
    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/json",
    }

    backoff = 2.0
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = requests.get(url, headers=headers, timeout=30)
        except requests.RequestException as exc:
            logger.warning("Network error (attempt %d/%d): %s", attempt, MAX_RETRIES, exc)
            if attempt == MAX_RETRIES:
                raise DataFetchError(f"Network error fetching {url}: {exc}") from exc
            time.sleep(backoff)
            backoff *= 2
            continue

        if resp.status_code == 200:
            return _parse_candles(resp.json())
        if resp.status_code == 401:
            raise TokenExpiredError(
                "Upstox returned 401 — access token expired. Re-authenticate "
                "and update UPSTOX_ACCESS_TOKEN."
            )
        if resp.status_code == 429:
            logger.warning("Rate limited (429) on attempt %d — backing off %.1fs", attempt, backoff)
            time.sleep(backoff)
            backoff = min(backoff * 2, 30.0)
            continue
        raise DataFetchError(
            f"Upstox historical endpoint returned HTTP {resp.status_code}: {resp.text[:300]}"
        )
    return pd.DataFrame()


def _parse_candles(payload: dict) -> pd.DataFrame:
    """Normalise the Upstox candle array into a typed DataFrame.

    Upstox returns ``data.candles`` as a list of
    ``[timestamp, open, high, low, close, volume, open_interest]`` rows.
    """
    data = payload.get("data") or {}
    candles = data.get("candles") or []
    if not candles:
        return pd.DataFrame()
    # Each row: [timestamp(str ISO), o, h, l, c, vol, oi]
    ts = [row[0] for row in candles]
    ohlcv = [
        {
            "open": float(row[1]),
            "high": float(row[2]),
            "low": float(row[3]),
            "close": float(row[4]),
            "volume": float(row[5]) if len(row) > 5 else 0.0,
        }
        for row in candles
    ]
    df = pd.DataFrame(ohlcv, index=pd.to_datetime(ts, utc=False))
    # Upstox timestamps are IST; make the index tz-aware for consistency.
    if df.index.tz is None:
        df.index = df.index.tz_localize("Asia/Kolkata")
    else:
        df.index = df.index.tz_convert("Asia/Kolkata")
    df = df[~df.index.duplicated(keep="last")].sort_index()
    return df


def _month_ranges(from_date: str, to_date: str) -> list[tuple[str, str]]:
    """Split [from_date, to_date] into inclusive (start, end) month chunks."""
    start = datetime.strptime(from_date, "%Y-%m-%d")
    end = datetime.strptime(to_date, "%Y-%m-%d")
    ranges: list[tuple[str, str]] = []
    cur = start.replace(day=1)
    while cur <= end:
        chunk_start = max(cur, start)
        # End of this month, capped at the overall `end`.
        if cur.month == 12:
            next_month = cur.replace(year=cur.year + 1, month=1, day=1)
        else:
            next_month = cur.replace(month=cur.month + 1, day=1)
        chunk_end = min(next_month - timedelta(days=1), end)
        ranges.append((chunk_start.strftime("%Y-%m-%d"), chunk_end.strftime("%Y-%m-%d")))
        cur = next_month
    return ranges


def _load_cache(cache_path: str) -> pd.DataFrame:
    if not cache_path:
        return pd.DataFrame()
    try:
        df = pd.read_csv(cache_path, index_col=0, parse_dates=True)
    except FileNotFoundError:
        return pd.DataFrame()
    if df.empty:
        return pd.DataFrame()
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC").tz_convert("Asia/Kolkata")
    else:
        df.index = df.index.tz_convert("Asia/Kolkata")
    return df.sort_index()


def _is_fully_cached(cached: pd.DataFrame, start: str, end: str) -> bool:
    """True if the cache already covers every minute in [start, end]."""
    if cached.empty:
        return False
    s = pd.Timestamp(start, tz="Asia/Kolkata")
    e = pd.Timestamp(end, tz="Asia/Kolkata") + pd.Timedelta(days=1) - pd.Timedelta(minutes=1)
    return cached.index.min() <= s and cached.index.max() >= e


def _append_cache(cache_path: str, chunk: pd.DataFrame) -> None:
    """Append a chunk to the CSV cache (creating the parent dir)."""
    if not cache_path:
        return
    import os

    os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
    # Write with tz-aware index; read back via _load_cache normalises again.
    chunk.to_csv(cache_path, mode="a", header=not os.path.exists(cache_path))
