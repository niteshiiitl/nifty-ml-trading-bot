"""CLI entry point for model training.

Modes:
    --synthetic N          generate N synthetic OHLCV rows in-memory and train
                           (no Upstox token / network needed — the smoke test).
    --from-csv PATH        train on a cached OHLCV CSV (bypass the Upstox fetch).
    --instrument/--from/--to   download real historical data via Upstox REST
                           (needs UPSTOX_ACCESS_TOKEN) then train.

Outputs a ``model.pkl`` bundle (joblib) containing the fitted Random Forest,
the exact ``FEATURE_COLUMNS`` order, and the config snapshot. Prints per-fold
``TimeSeriesSplit`` accuracy, a classification report, and top-10 importances.

No order placement — training only.
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Optional

import numpy as np
import pandas as pd

from src.config import DEFAULT_INSTRUMENT_KEY, configure_logging
from src.data_fetcher import fetch_historical_candles
from src.features import FEATURE_COLUMNS
from src.model_trainer import print_summary, train


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Train the F&O alert model (alert-only).")
    p.add_argument("--instrument", default=DEFAULT_INSTRUMENT_KEY,
                   help=f"Instrument key (default: {DEFAULT_INSTRUMENT_KEY}).")
    p.add_argument("--from", dest="from_date", help="Start date YYYY-MM-DD (real fetch).")
    p.add_argument("--to", dest="to_date", help="End date YYYY-MM-DD (real fetch).")
    p.add_argument("--output", default="model.pkl", help="Output bundle path (default: model.pkl).")
    p.add_argument("--from-csv", dest="from_csv", default=None,
                   help="Train on a cached OHLCV CSV instead of downloading.")
    p.add_argument("--synthetic", type=int, default=None,
                   help="Generate N synthetic OHLCV rows and train (no token needed).")
    return p


def main(argv: Optional[list[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)
    configure_logging(os.environ.get("LOG_LEVEL", "INFO"))

    # Decide the data source (mutually exclusive precedence: synthetic > csv > fetch).
    if args.synthetic is not None:
        df = generate_synthetic_ohlcv(args.synthetic)
        print(f"[train_model] Using SYNTHETIC data: {len(df)} rows.")
    elif args.from_csv:
        df = load_ohlcv_csv(args.from_csv)
        print(f"[train_model] Loaded CSV '{args.from_csv}': {len(df)} rows.")
    else:
        if not args.from_date or not args.to_date:
            print("[train_model] Real-fetch mode requires --from and --to (YYYY-MM-DD). "
                  "Use --synthetic N or --from-csv PATH to train without the API.",
                  file=sys.stderr)
            return 2
        token = os.environ.get("UPSTOX_ACCESS_TOKEN")
        if not token:
            print("[train_model] UPSTOX_ACCESS_TOKEN is required for real fetch. "
                  "Use --synthetic N or --from-csv PATH instead.", file=sys.stderr)
            return 2
        df = fetch_historical_candles(args.instrument, args.from_date, args.to_date, token=token)
        print(f"[train_model] Fetched {len(df)} rows for {args.instrument} "
              f"({args.from_date}..{args.to_date}).")

    if df.empty:
        print("[train_model] No data to train on.", file=sys.stderr)
        return 1

    summary = train(df, output_path=args.output)
    print_summary(summary)

    # Assert the contract the live pipeline depends on (sanity, in-process).
    assert summary["bundle"]["feature_columns"] == FEATURE_COLUMNS, \
        "bundle feature_columns must equal FEATURE_COLUMNS"
    print(f"\n[train_model] Bundle written to {args.output} "
          f"(feature_columns == FEATURE_COLUMNS, {len(FEATURE_COLUMNS)} features).")
    return 0


def generate_synthetic_ohlcv(rows: int, seed: int = 42) -> pd.DataFrame:
    """Generate N synthetic 1-min OHLCV rows for the smoke-test path.

    A sinusoidal trend plus noise produces both up and down forward returns so
    the label set contains Buys/Sells/Holds (a more meaningful CV report than a
    pure monotonic walk). This path never touches the network.
    """
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2025-01-06 09:15", periods=rows, freq="1min", tz="Asia/Kolkata")
    t = np.arange(rows)
    # Slow sine regime + noise gives variation in forward direction.
    base = 100.0 + 8.0 * np.sin(2.0 * np.pi * t / 200.0) + 0.02 * t
    noise = rng.normal(0.0, 0.6, size=rows)
    close = base + noise
    open_ = np.concatenate([[close[0]], close[:-1]]) + rng.normal(0.0, 0.1, size=rows)
    high = np.maximum(open_, close) + rng.uniform(0.0, 0.8, size=rows)
    low = np.minimum(open_, close) - rng.uniform(0.0, 0.8, size=rows)
    volume = rng.integers(500, 10_000, size=rows).astype(float)
    df = pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
        index=idx,
    )
    return df


def load_ohlcv_csv(path: str) -> pd.DataFrame:
    """Load an OHLCV CSV with a datetime index (or a timestamp/date column)."""
    df = pd.read_csv(path)
    # Accept either an index column or a named timestamp column.
    ts_col = None
    for cand in ("timestamp", "date", "datetime", "time"):
        if cand in df.columns:
            ts_col = cand
            break
    if ts_col:
        df.index = pd.to_datetime(df[ts_col])
        df = df.drop(columns=[ts_col])
    else:
        # Assume the first column is the datetime index.
        df = pd.read_csv(path, index_col=0, parse_dates=True)
    needed = ["open", "high", "low", "close", "volume"]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(f"CSV missing required OHLCV columns: {missing}")
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC").tz_convert("Asia/Kolkata")
    return df[needed].sort_index()


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
