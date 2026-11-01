"""Tests for src/config.py — typed env loading and secret validation."""
from __future__ import annotations

import pytest

from src.config import Config, redact


def test_from_env_with_full_env():
    env = {
        "UPSTOX_ACCESS_TOKEN": "tok_abc",
        "UPSTOX_API_KEY": "key",
        "UPSTOX_API_SECRET": "secret",
        "UPSTOX_INSTRUMENT_KEY": "NSE_INDEX|Nifty 50",
        "TELEGRAM_BOT_TOKEN": "bot_123",
        "TELEGRAM_CHAT_ID": "987654",
        "PORT": "8080",
        "MODEL_PATH": "model.pkl",
        "SIGNAL_THRESHOLD": "0.002",
        "LOG_LEVEL": "debug",
    }
    cfg = Config.from_env(env)
    assert cfg.access_token == "tok_abc"
    assert cfg.api_key == "key"
    assert cfg.api_secret == "secret"
    assert cfg.instrument_key == "NSE_INDEX|Nifty 50"
    assert cfg.telegram_bot_token == "bot_123"
    assert cfg.telegram_chat_id == "987654"
    assert cfg.port == 8080
    assert cfg.signal_threshold == pytest.approx(0.002)
    assert cfg.log_level == "DEBUG"  # upper-cased


def test_from_env_defaults_when_empty():
    cfg = Config.from_env({})  # nothing set
    assert cfg.access_token is None
    assert cfg.telegram_bot_token is None
    assert cfg.telegram_chat_id is None
    assert cfg.instrument_key == "NSE_INDEX|Nifty 50"  # default
    assert cfg.port == 5000
    assert cfg.model_path == "model.pkl"
    assert cfg.signal_threshold is None
    assert cfg.log_level == "INFO"
    assert cfg.label_horizon == 5


def test_require_secrets_lists_missing():
    cfg = Config.from_env({})
    missing = cfg.require_secrets()
    # All three required live secrets should be reported missing.
    assert "UPSTOX_ACCESS_TOKEN" in missing
    assert "TELEGRAM_BOT_TOKEN" in missing
    assert "TELEGRAM_CHAT_ID" in missing
    assert len(missing) == 3


def test_validate_for_live_raises_when_secrets_absent():
    cfg = Config.from_env({})
    with pytest.raises(RuntimeError) as excinfo:
        cfg.validate_for_live()
    msg = str(excinfo.value)
    assert "UPSTOX_ACCESS_TOKEN" in msg
    assert "TELEGRAM_BOT_TOKEN" in msg
    assert "TELEGRAM_CHAT_ID" in msg


def test_validate_for_live_passes_when_secrets_present():
    cfg = Config.from_env(
        {
            "UPSTOX_ACCESS_TOKEN": "tok",
            "TELEGRAM_BOT_TOKEN": "bot",
            "TELEGRAM_CHAT_ID": "123",
        }
    )
    # Should not raise.
    cfg.validate_for_live()
    assert cfg.require_secrets() == []


def test_redact_does_not_leak_full_secret():
    assert redact("abcdefghij", show=6) == "abcdef***"
    assert redact("short") == "***"  # too short to show any prefix safely
    assert redact(None) == "<unset>"
    assert redact("") == "<unset>"


def test_feature_config_dict_round_trips_periods():
    cfg = Config.from_env({})
    fc = cfg.feature_config_dict()
    assert fc["rsi_period"] == 14
    assert fc["macd_fast"] == 12 and fc["macd_slow"] == 26 and fc["macd_signal"] == 9
    assert fc["bbands_period"] == 20 and fc["bbands_nbdev"] == 2
    assert fc["ema_periods"] == [9, 21, 50]
    assert fc["atr_period"] == 14
    assert fc["label_horizon"] == 5
