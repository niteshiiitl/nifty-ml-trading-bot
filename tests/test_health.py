"""Tests for /health resilience — the app boots and /health returns 200 with no secrets.

Covers FR-5.1 and US-4: a lightweight Flask app exposes GET /health returning
HTTP 200 even when required live secrets are missing (degraded mode). The
stream thread is started by app.py at import; with no secrets it logs the
missing-secret error and stays alive (does not crash the process).
"""
from __future__ import annotations

import os

import pandas as pd
import pytest

# Ensure no live secrets are set BEFORE importing the app module, so the test
# exercises the degraded / no-secrets boot path. Done at module collection time
# (before any test imports `app`), and none of the other test modules import app.
for _k in (
    "UPSTOX_ACCESS_TOKEN",
    "UPSTOX_API_KEY",
    "UPSTOX_API_SECRET",
    "TELEGRAM_BOT_TOKEN",
    "TELEGRAM_CHAT_ID",
):
    os.environ.pop(_k, None)


def test_health_via_create_app_returns_200_degraded():
    """Direct: create_app with a degraded status still returns 200 + 'ok'."""
    from src.server import StreamStatus, create_app

    status = StreamStatus()
    status.instrument = "NSE_INDEX|Nifty 50"
    status.degraded = True  # simulate missing secrets/model
    flask_app = create_app("NSE_INDEX|Nifty 50", status)
    client = flask_app.test_client()
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.get_json()
    assert body["status"] == "ok"
    assert body["instrument"] == "NSE_INDEX|Nifty 50"
    assert "last_candle" in body
    assert body["degraded"] is True


def test_app_module_boots_with_no_secrets():
    """Importing app.py runs module-level wiring with no secrets; must not raise
    and must expose a Flask app whose /health returns 200."""
    import app as app_module  # noqa: WPS433 - intentional runtime import

    assert hasattr(app_module, "app")
    client = app_module.app.test_client()
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.get_json()
    assert body["status"] == "ok"
    assert body["instrument"] == "NSE_INDEX|Nifty 50"
    # No secrets => degraded should be reported (but status code stays 200).
    assert body["degraded"] is True


def test_stream_status_threadsafe_update_and_snapshot():
    from src.server import StreamStatus

    s = StreamStatus()
    s.instrument = "X"
    s.update(
        last_candle=pd.Timestamp("2026-01-05 09:15", tz="Asia/Kolkata"),
        last_signal=1,
        last_confidence=0.8,
        degraded=False,
    )
    snap = s.snapshot()
    assert snap["instrument"] == "X"
    assert "2026-01-05 09:15" in snap["last_candle"]
    assert snap["last_signal"] == 1
    assert snap["last_confidence"] == pytest.approx(0.8)
    assert snap["degraded"] is False
