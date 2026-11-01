"""Tests for src/alerts.py — Telegram formatting + cooldown guard (mocked HTTP)."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pandas as pd
import requests

from src.alerts import TelegramAlerter


def _alerter(cooldown: int = 60) -> TelegramAlerter:
    return TelegramAlerter(bot_token="bot123", chat_id="chat456", cooldown_seconds=cooldown)


def _mock_response(status: int = 200):
    r = MagicMock()
    r.status_code = status
    r.text = "ok"
    return r


def test_format_message_matches_fr44():
    al = _alerter()
    ts = pd.Timestamp("2026-10-31 10:15:00", tz="Asia/Kolkata")
    msg = al.format_message(1, "NIFTY 50", 24150.35, ts, 0.82)
    assert "🚨 BUY SIGNAL" in msg
    assert "Instrument: NIFTY 50" in msg
    assert "Timeframe: 1-Minute" in msg
    assert "Time: 2026-10-31 10:15:00 IST" in msg
    assert "Price: 24150.35" in msg
    assert "Confidence: 0.82" in msg


def test_format_message_sell():
    al = _alerter()
    ts = pd.Timestamp("2026-10-31 10:16:00", tz="Asia/Kolkata")
    msg = al.format_message(-1, "NIFTY 50", 24100.0, ts, 0.70)
    assert "🔻 SELL SIGNAL" in msg


def test_send_alert_posts_to_telegram():
    al = _alerter()
    ts = pd.Timestamp("2026-10-31 10:15:00", tz="Asia/Kolkata")
    with patch("src.alerts.requests.post", return_value=_mock_response(200)) as mock_post:
        sent = al.send_alert(1, "NIFTY 50", 24150.35, ts, 0.82)
    assert sent is True
    mock_post.assert_called_once()
    args, kwargs = mock_post.call_args
    assert args[0].startswith("https://api.telegram.org/bot")
    assert "bot123" in args[0]  # token embedded in URL
    assert kwargs["json"]["chat_id"] == "chat456"
    assert "🚨 BUY SIGNAL" in kwargs["json"]["text"]


def test_duplicate_same_signal_within_cooldown_suppressed():
    al = _alerter(cooldown=60)
    ts = pd.Timestamp("2026-10-31 10:15:00", tz="Asia/Kolkata")
    with patch("src.alerts.requests.post", return_value=_mock_response(200)) as mock_post:
        first = al.send_alert(1, "NIFTY 50", 100.0, ts, 0.8)
        second = al.send_alert(1, "NIFTY 50", 101.0, ts, 0.8)  # same signal, within cooldown
    assert first is True
    assert second is False
    assert mock_post.call_count == 1  # only the first is actually sent


def test_different_signal_is_sent():
    al = _alerter(cooldown=60)
    ts = pd.Timestamp("2026-10-31 10:15:00", tz="Asia/Kolkata")
    with patch("src.alerts.requests.post", return_value=_mock_response(200)) as mock_post:
        al.send_alert(1, "NIFTY 50", 100.0, ts, 0.8)
        al.send_alert(-1, "NIFTY 50", 99.0, ts, 0.7)  # different signal -> not suppressed
    assert mock_post.call_count == 2


def test_hold_signal_not_sent():
    al = _alerter()
    ts = pd.Timestamp("2026-10-31 10:15:00", tz="Asia/Kolkata")
    with patch("src.alerts.requests.post", return_value=_mock_response(200)) as mock_post:
        sent = al.send_alert(0, "NIFTY 50", 100.0, ts, 0.5)
    assert sent is False
    mock_post.assert_not_called()


def test_send_alert_handles_http_error_without_raising():
    al = _alerter()
    ts = pd.Timestamp("2026-10-31 10:15:00", tz="Asia/Kolkata")
    with patch("src.alerts.requests.post", return_value=_mock_response(500)):
        sent = al.send_alert(1, "NIFTY 50", 100.0, ts, 0.8)  # must not raise
    assert sent is False


def test_send_alert_handles_network_error_without_raising():
    al = _alerter()
    ts = pd.Timestamp("2026-10-31 10:15:00", tz="Asia/Kolkata")
    with patch("src.alerts.requests.post", side_effect=requests.ConnectionError("boom")):
        sent = al.send_alert(1, "NIFTY 50", 100.0, ts, 0.8)
    assert sent is False


def test_missing_credentials_not_sent():
    al = TelegramAlerter(bot_token=None, chat_id=None)
    ts = pd.Timestamp("2026-10-31 10:15:00", tz="Asia/Kolkata")
    with patch("src.alerts.requests.post", return_value=_mock_response(200)) as mock_post:
        sent = al.send_alert(1, "NIFTY 50", 100.0, ts, 0.8)
    assert sent is False
    mock_post.assert_not_called()


def test_send_message_posts_arbitrary_text():
    """The re-auth alert path (FR-1.4) uses send_message for a plain notice."""
    al = _alerter()
    with patch("src.alerts.requests.post", return_value=_mock_response(200)) as mock_post:
        sent = al.send_message("⚠️ Re-authentication required: Upstox token expired.")
    assert sent is True
    args, kwargs = mock_post.call_args
    assert kwargs["json"]["chat_id"] == "chat456"
    assert "Re-authentication required" in kwargs["json"]["text"]
