"""Model training logic (FR-2.7 — FR-2.9).

Orchestrates: feature compute → label → drop warm-up/horizon rows →
``TimeSeriesSplit`` cross-validation → ``RandomForestClassifier`` fit → bundle
export. ``train_test_split`` and shuffled CV are deliberately NOT used anywhere
(NFR-4): only ``TimeSeriesSplit`` prevents look-ahead leakage.

The bundle is ``{"model": rf, "feature_columns": FEATURE_COLUMNS, "config": ...}``
saved via ``joblib.dump``. ``FEATURE_COLUMNS`` is imported from the single shared
``src.features`` module so train and live cannot drift.
"""

from __future__ import annotations

import logging
from typing import Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import TimeSeriesSplit

from src.config import Config, configure_logging
from src.features import FEATURE_COLUMNS, compute_features
from src.label import make_labels

logger = logging.getLogger(__name__)

DEFAULT_N_SPLITS = 5
DEFAULT_N_ESTIMATORS = 300
DEFAULT_MAX_DEPTH = 10
DEFAULT_MIN_SAMPLES_LEAF = 4
RANDOM_STATE = 42


def train(
    df: pd.DataFrame,
    config: Optional[Config] = None,
    output_path: Optional[str] = None,
    n_splits: int = DEFAULT_N_SPLITS,
) -> dict:
    """Train a Random Forest on OHLCV ``df`` and (optionally) save the bundle.

    Steps:
        1. compute_features(df) -> feature matrix X (exact FEATURE_COLUMNS order).
        2. make_labels(df, horizon, threshold) -> int labels y in {-1,0,1}.
        3. Drop warm-up NaN feature rows and the trailing ``horizon`` rows that
           have no forward target (controls the boundary, NFR-4).
        4. TimeSeriesSplit(n_splits) CV: per-fold accuracy + final report.
        5. Fit the final RandomForest on all valid rows (class_weight='balanced').
        6. Build & (optionally) save the bundle.

    Returns:
        A summary dict with row counts, per-fold CV accuracy, the final
        classification report (str), and top-10 feature importances.
    """
    if config is None:
        config = Config.from_env()

    horizon = config.label_horizon
    threshold = config.signal_threshold  # None => ATR-scaled (requires talib)

    # 1. Features (exact contract order).
    X = compute_features(df)
    # 2. Labels.
    y = make_labels(df, horizon=horizon, threshold=threshold)

    # 3. Boundary control: drop warm-up NaNs and the label-horizon tail.
    feature_notna = X.notna().all(axis=1)
    has_target = np.arange(len(df)) < (len(df) - horizon)
    has_target = pd.Series(has_target, index=df.index)
    valid = feature_notna & has_target
    Xv, yv = X.loc[valid], y.loc[valid]

    if len(Xv) < n_splits + 1:
        raise ValueError(
            f"Not enough valid rows after warm-up/horizon drop ({len(Xv)}) "
            f"for {n_splits} TimeSeriesSplit folds."
        )
    if yv.nunique() < 2:
        logger.warning("Only one class present in labels (%r). CV metrics will be trivial.", yv.unique())

    logger.info(
        "Training set: %d valid rows, %d features, label counts=%s",
        len(Xv), Xv.shape[1], yv.value_counts().to_dict(),
    )

    # 4. TimeSeriesSplit CV (sequential folds — no shuffling, no leakage).
    tscv = TimeSeriesSplit(n_splits=n_splits)
    cv_accuracies: list[float] = []
    last_fold_report = ""
    for fold, (train_idx, test_idx) in enumerate(tscv.split(Xv), start=1):
        rf = _new_rf()
        rf.fit(Xv.iloc[train_idx], yv.iloc[train_idx])
        pred = rf.predict(Xv.iloc[test_idx])
        acc = accuracy_score(yv.iloc[test_idx], pred)
        cv_accuracies.append(float(acc))
        logger.info("CV fold %d/%d accuracy=%.4f (train=%d test=%d)", fold, n_splits, acc, len(train_idx), len(test_idx))
        if fold == n_splits:
            last_fold_report = classification_report(
                yv.iloc[test_idx], pred, zero_division=0, digits=4
            )

    # 5. Final model trained on ALL valid rows for deployment.
    final_rf = _new_rf()
    final_rf.fit(Xv, yv)

    # Bundle: model + the exact feature contract + config snapshot.
    bundle = {
        "model": final_rf,
        "feature_columns": list(FEATURE_COLUMNS),
        "config": {
            "label_horizon": horizon,
            "signal_threshold": threshold,
            **config.feature_config_dict(),
        },
    }
    if output_path:
        joblib.dump(bundle, output_path)
        logger.info("Saved model bundle to %s", output_path)

    # Top-10 feature importances for the summary.
    importances = getattr(final_rf, "feature_importances_", None)
    top10: list[tuple[str, float]] = []
    if importances is not None:
        order = np.argsort(importances)[::-1][:10]
        top10 = [(FEATURE_COLUMNS[i], float(importances[i])) for i in order]

    return {
        "bundle": bundle,
        "n_rows": len(df),
        "n_valid_rows": len(Xv),
        "n_features": Xv.shape[1],
        "label_counts": yv.value_counts().to_dict(),
        "cv_accuracies": cv_accuracies,
        "last_fold_report": last_fold_report,
        "top10_importances": top10,
    }


def _new_rf() -> RandomForestClassifier:
    """Construct the RF with balanced class weights (Hold-class imbalance)."""
    return RandomForestClassifier(
        n_estimators=DEFAULT_N_ESTIMATORS,
        max_depth=DEFAULT_MAX_DEPTH,
        min_samples_leaf=DEFAULT_MIN_SAMPLES_LEAF,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )


def print_summary(summary: dict) -> None:
    """Print the training summary to stdout (used by train_model.py)."""
    print("\n=== Training summary ===")
    print(f"Rows fetched:        {summary['n_rows']}")
    print(f"Valid rows (post warm-up/horizon): {summary['n_valid_rows']}")
    print(f"Feature count:       {summary['n_features']}")
    print(f"Label counts:        {summary['label_counts']}")
    accs = summary["cv_accuracies"]
    print(f"TimeSeriesSplit CV accuracy per fold: " +
          ", ".join(f"{a:.4f}" for a in accs))
    if accs:
        print(f"Mean CV accuracy:    {float(np.mean(accs)):.4f}")
    if summary["last_fold_report"]:
        print("\n--- Classification report (last fold) ---")
        print(summary["last_fold_report"])
    if summary["top10_importances"]:
        print("--- Top-10 feature importances ---")
        for name, imp in summary["top10_importances"]:
            print(f"  {name:12s} {imp:.4f}")


def main_cli() -> None:  # pragma: no cover - thin CLI dispatcher
    """Entry hook; the real CLI lives in train_model.py (kept here for parity)."""
    configure_logging("INFO")
    print("Use train_model.py (the CLI) instead of calling model_trainer directly.")
