"""Tests for src/inference.py — load bundle, validate feature order, predict."""
from __future__ import annotations

import os
import tempfile

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier

from src.features import FEATURE_COLUMNS
from src.inference import InferenceEngine


def _train_tiny_rf(n=300, seed=0):
    """Train a tiny RF on synthetic feature rows with 3 classes (-1,0,1)."""
    rng = np.random.default_rng(seed)
    X = rng.normal(0.0, 1.0, size=(n, len(FEATURE_COLUMNS)))
    # Class depends on the sign of the first feature (so the model learns something).
    y = np.where(X[:, 0] > 0.5, 1, np.where(X[:, 0] < -0.5, -1, 0))
    rf = RandomForestClassifier(n_estimators=30, random_state=seed, class_weight="balanced")
    rf.fit(X, y)
    return rf, X, y


def _save_bundle(path, feature_columns):
    rf, _, _ = _train_tiny_rf()
    bundle = {"model": rf, "feature_columns": list(feature_columns),
              "config": {"label_horizon": 5}}
    joblib.dump(bundle, path)
    return bundle


def test_predict_returns_label_in_set_and_confidence_in_range():
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "model_test.pkl")
        _save_bundle(path, FEATURE_COLUMNS)
        eng = InferenceEngine(model_path=path)
        row = pd.DataFrame([np.zeros(len(FEATURE_COLUMNS))], columns=FEATURE_COLUMNS)
        label, conf = eng.predict(row)
        assert label in (-1, 0, 1)
        assert 0.0 <= conf <= 1.0


def test_predict_uses_last_row():
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "model_test.pkl")
        _save_bundle(path, FEATURE_COLUMNS)
        eng = InferenceEngine(model_path=path)
        # Two rows; predict should use the LAST row.
        rows = pd.DataFrame(
            np.zeros((2, len(FEATURE_COLUMNS))),
            columns=FEATURE_COLUMNS,
        )
        rows.iloc[1, 0] = 10.0  # strongly positive first feature -> likely Buy
        label, _ = eng.predict(rows)
        assert label == 1


def test_from_in_memory_bundle():
    rf, _, _ = _train_tiny_rf()
    bundle = {"model": rf, "feature_columns": list(FEATURE_COLUMNS), "config": {}}
    eng = InferenceEngine(bundle=bundle)
    row = pd.DataFrame([np.zeros(len(FEATURE_COLUMNS))], columns=FEATURE_COLUMNS)
    label, conf = eng.predict(row)
    assert label in (-1, 0, 1)
    assert 0.0 <= conf <= 1.0


def test_mismatched_feature_columns_raises():
    rf, _, _ = _train_tiny_rf()
    bad_cols = list(FEATURE_COLUMNS)[::-1]  # reversed order
    bundle = {"model": rf, "feature_columns": bad_cols, "config": {}}
    with pytest.raises(ValueError, match="feature_columns"):
        InferenceEngine(bundle=bundle)


def test_bundle_missing_keys_raises():
    with pytest.raises(ValueError):
        InferenceEngine(bundle={"model": None})  # no feature_columns


def test_requires_path_or_bundle():
    with pytest.raises(ValueError):
        InferenceEngine()


def test_empty_latest_row_raises():
    rf, _, _ = _train_tiny_rf()
    bundle = {"model": rf, "feature_columns": list(FEATURE_COLUMNS), "config": {}}
    eng = InferenceEngine(bundle=bundle)
    with pytest.raises(ValueError):
        eng.predict(pd.DataFrame(columns=FEATURE_COLUMNS))


def test_classes_property_present():
    rf, _, _ = _train_tiny_rf()
    bundle = {"model": rf, "feature_columns": list(FEATURE_COLUMNS), "config": {}}
    eng = InferenceEngine(bundle=bundle)
    assert set(int(c) for c in eng.classes_) <= {-1, 0, 1}
