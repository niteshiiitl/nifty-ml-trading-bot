"""Telegram alerting via the Bot HTTP API (FR-4.4 — FR-4.6).

Uses plain ``requests`` HTTP calls to ``https://api.telegram.org/bot<token>/sendMessage``
(no python-telegram-bot dependency). The HTTP call is wrapped so a Telegram
failure logs but never raises into the streaming loop (FR-4.6).

A lightweight cooldown guard suppresses duplicate consecutive same-signal alerts
within ``cooldown_seconds`` (default 60s) to avoid spamming on a trending candle.
A different signal, or an expired cooldown, resets the guard.
"""

from __future__ import annotations

import logging
import time
from typing import Optional

import pandas as pd
import requests

from src.config import DEFAULT_ALERT_COOLDOWN_SECONDS, redact

logger = logging.getLogger(__name__)

TELEGRAM_API_BASE = "https://api.telegram.org"

_SIGNAL_EMOJI = {1: "🚨 BUY SIGNAL", -1: "🔻 SELL SIGNAL"}
_SIGNAL_NAME = {1: "BUY", -1: "SELL"}


class TelegramAlerter:
    """Send formatted buy/sell alerts to a Telegram chat with a cooldown guard.

    Args:
        bot_token: Telegram bot token (``TELEGRAM_BOT_TOKEN``).
        chat_id: Target chat/channel id (``TELEGRAM_CHAT_ID``).
        cooldown_seconds: Suppress duplicate consecutive same-signal alerts within
            this window (default 60s).
    """

    def __init__(
        self,
        bot_token: Optional[str],
        chat_id: Optional[str],
        cooldown_seconds: int = DEFAULT_ALERT_COOLDOWN_SECONDS,
    ) -> None:
        self.bot_token = bot_token
        self.chat_id = chat_id
        self.cooldown_seconds = cooldown_seconds
        self._last_signal: Optional[int] = None
        self._last_sent_ts: Optional[float] = None

    def format_message(
        self, signal: int, instrument: str, price: float, timestamp, confidence: float
    ) -> str:
        """Build the FR-4.4 alert message."""
        title = _SIGNAL_EMOJI.get(signal, f"🔔 SIGNAL ({signal})")
        ts_str = _format_timestamp(timestamp)
        return (
            f"{title}\n"
            f"Instrument: {instrument}\n"
            f"Timeframe: 1-Minute\n"
            f"Time: {ts_str}\n"
            f"Price: {price:.2f}\n"
            f"Confidence: {confidence:.2f}"
        )

    def send_alert(
        self, signal: int, instrument: str, price: float, timestamp, confidence: float
    ) -> bool:
        """Send an alert unless suppressed by the cooldown. Returns True if sent.

        Never raises: Telegram/network errors are logged and False is returned.
        """
        if signal not in _SIGNAL_EMOJI:
            logger.debug("Ignoring non-actionable signal %r (Hold/unknown).", signal)
            return False

        # Cooldown guard: suppress duplicate consecutive same-signal within window.
        now = time.monotonic()
        if (
            self._last_signal == signal
            and self._last_sent_ts is not None
            and (now - self._last_sent_ts) < self.cooldown_seconds
        ):
            logger.info(
                "Suppressing duplicate %s alert within %.0fs cooldown.",
                _SIGNAL_NAME[signal], self.cooldown_seconds,
            )
            return False

        if not self.bot_token or not self.chat_id:
            logger.error(
                "Cannot send alert: Telegram bot_token/chat_id not configured "
                "(bot_token=%s, chat_id set=%s).",
                redact(self.bot_token), bool(self.chat_id),
            )
            return False

        message = self.format_message(signal, instrument, price, timestamp, confidence)
        url = f"{TELEGRAM_API_BASE}/bot{self.bot_token}/sendMessage"
        payload = {
            "chat_id": self.chat_id,
            "text": message,
            "parse_mode": "HTML",
        }

        try:
            resp = requests.post(url, json=payload, timeout=10)
        except requests.RequestException as exc:
            logger.error("Telegram send failed (network): %s", exc)
            return False

        if resp.status_code != 200:
            logger.error(
                "Telegram API returned HTTP %d: %s",
                resp.status_code, resp.text[:300],
            )
            return False

        # Record this signal + time so the cooldown applies to the next call.
        self._last_signal = signal
        self._last_sent_ts = now
        logger.info("Sent %s alert for %s @ %.2f (conf %.2f).",
                    _SIGNAL_NAME[signal], instrument, price, confidence)
        return True

    def send_message(self, text: str) -> bool:
        """Send an arbitrary plain-text notice (no signal, no cooldown).

        Used for operator notices such as the re-authentication-required alert
        (FR-1.4) when Upstox returns 401. Never raises into the stream loop;
        returns True only on a successful Telegram API response.
        """
        if not self.bot_token or not self.chat_id:
            logger.error(
                "Cannot send message: Telegram bot_token/chat_id not configured "
                "(bot_token=%s, chat_id set=%s).",
                redact(self.bot_token), bool(self.chat_id),
            )
            return False
        url = f"{TELEGRAM_API_BASE}/bot{self.bot_token}/sendMessage"
        payload = {"chat_id": self.chat_id, "text": text, "parse_mode": "HTML"}
        try:
            resp = requests.post(url, json=payload, timeout=10)
        except requests.RequestException as exc:
            logger.error("Telegram send_message failed (network): %s", exc)
            return False
        if resp.status_code != 200:
            logger.error("Telegram API returned HTTP %d: %s", resp.status_code, resp.text[:300])
            return False
        logger.info("Sent Telegram notice (%d chars).", len(text))
        return True


def _format_timestamp(timestamp) -> str:
    """Format a timestamp (epoch / pd.Timestamp / datetime / str) as IST string."""
    if timestamp is None:
        return "n/a"
    try:
        if isinstance(timestamp, (int, float)):
            ts = pd.Timestamp(float(timestamp), unit="s", tz="Asia/Kolkata")
        elif isinstance(timestamp, pd.Timestamp):
            ts = timestamp.tz_convert("Asia/Kolkata") if timestamp.tzinfo else timestamp.tz_localize("Asia/Kolkata")
        else:
            ts = pd.Timestamp(timestamp)
            if ts.tzinfo is None:
                ts = ts.tz_localize("Asia/Kolkata")
            else:
                ts = ts.tz_convert("Asia/Kolkata")
    except Exception:
        return str(timestamp)
    return ts.strftime("%Y-%m-%d %H:%M:%S IST")
