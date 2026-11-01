"""Inference engine — load model.pkl and predict on a closed candle (FR-4.1-4.3).

``InferenceEngine`` loads the joblib bundle ``{"model", "feature_columns", ...}``
and validates that ``bundle["feature_columns"]`` exactly equals ``FEATURE_COLUMNS``
(names AND order) BEFORE any prediction. This enforces the feature-parity
contract (NFR-3): a model trained with one feature order cannot be served with
another.

``predict(latest_row_df) -> (label, confidence)`` selects the feature columns in
the stored order from the latest row, runs ``model.predict`` (label in {-1,0,1})
and ``model.predict_proba`` (max probability = confidence in [0,1]). The caller
ignores ``0`` (Hold); only ``1``/``-1`` trigger the alerter.

Only ``FEATURE_COLUMNS`` is imported from ``src.features`` (no talib needed), so
this module imports cleanly without the native TA-Lib library.
"""

from __future__ import annotations

import logging
from typing import Optional

import joblib
import numpy as np
import pandas as pd

from src.features import FEATURE_COLUMNS

logger = logging.getLogger(__name__)


class InferenceEngine:
    """Wrap a trained classifier for per-candle prediction.

    Args:
        model_path: Path to a joblib bundle. Mutually exclusive with ``bundle``.
        bundle: An already-loaded bundle dict (used by tests / in-memory wiring).

    Raises:
        ValueError: if the bundle's feature_columns differ from FEATURE_COLUMNS.
        FileNotFoundError: if model_path is given but does not exist.
    """

    def __init__(self, model_path: Optional[str] = None, bundle: Optional[dict] = None) -> None:
        if bundle is not None:
            b = bundle
        elif model_path:
            b = joblib.load(model_path)
        else:
            raise ValueError("InferenceEngine requires either model_path or bundle.")

        if "model" not in b or "feature_columns" not in b:
            raise ValueError("Bundle must contain 'model' and 'feature_columns'.")

        self.model = b["model"]
        self.feature_columns = list(b["feature_columns"])

        # Validate the contract before any predict (NFR-3 feature parity).
        if list(self.feature_columns) != list(FEATURE_COLUMNS):
            raise ValueError(
                "Bundle feature_columns do not match FEATURE_COLUMNS (names/order).\n"
                f"  bundle: {self.feature_columns}\n"
                f"  expected: {list(FEATURE_COLUMNS)}"
            )
        logger.info("InferenceEngine loaded (features=%d).", len(self.feature_columns))

    def predict(self, latest_row_df: pd.DataFrame) -> tuple[int, float]:
        """Predict the label and confidence for the latest closed candle.

        Args:
            latest_row_df: A DataFrame whose last row carries the latest feature
                values (at least the FEATURE_COLUMNS).

        Returns:
            ``(label, confidence)`` where label is in ``{-1, 0, 1}`` and
            confidence (the max class probability) is in ``[0.0, 1.0]``.
        """
        if latest_row_df is None or len(latest_row_df) == 0:
            raise ValueError("latest_row_df is empty; cannot predict.")

        # Select features in the exact stored order from the latest row.
        row = latest_row_df.iloc[[-1]][self.feature_columns]
        X = row.to_numpy(dtype=float)

        label = int(self.model.predict(X)[0])

        confidence = 1.0
        proba = getattr(self.model, "predict_proba", None)
        if callable(proba):
            try:
                p = proba(X)[0]
                confidence = float(np.max(p))
            except Exception as exc:  # noqa: BLE001 - some classifiers lack proba
                logger.debug("predict_proba unavailable/failed (%s); confidence=1.0", exc)
                confidence = 1.0
        return label, confidence

    @property
    def classes_(self) -> np.ndarray:
        """The class labels the underlying model was trained on (e.g. [-1,0,1])."""
        return np.asarray(getattr(self.model, "classes_", np.array([-1, 0, 1])))


def load_or_none(model_path: str) -> Optional["InferenceEngine"]:
    """Load an InferenceEngine, returning None (and logging) if unavailable.

    Used by app.py so a missing/corrupt model does not prevent the Flask app
    from booting — /health still serves 200 (degraded).
    """
    try:
        return InferenceEngine(model_path=model_path)
    except FileNotFoundError:
        logger.warning("Model not found at %s — running degraded (no inference).", model_path)
        return None
    except Exception as exc:  # noqa: BLE001 - keep the process alive
        logger.warning("Failed to load model from %s: %s — running degraded.", model_path, exc)
        return None
