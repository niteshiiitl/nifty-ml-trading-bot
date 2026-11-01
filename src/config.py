"""Central configuration for the F&O AI Trading Alert Agent.

Loads all environment variables (requirements §6) into a typed ``Config``
object with safe defaults, validates the required live secrets, and keeps the
feature/label constants in ONE place so the training and live pipelines stay
synchronized.

The ordered list of feature column names itself lives in ``src/features.py``
(``FEATURE_COLUMNS``) — that is the single source of truth for the model. This
module only holds the *parameters* (indicator periods, label horizon, etc.)
that ``features.py`` reads to compute them, so train/live cannot drift.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Optional

# Auto-load a local .env file when python-dotenv is available (local-dev
# convenience). On Render the env vars come from the dashboard and no .env
# exists, so this is a safe no-op there. Existing env vars are NOT overridden,
# so shell exports still take precedence.
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:  # pragma: no cover - python-dotenv not installed
    pass


# --- Feature/label configuration constants (single source for parameters) ---
# These mirror the periods used in src/features.py::compute_features. Keep
# them aligned: changing one without the other will drift train vs. live.

RSI_PERIOD: int = 14
MACD_FAST: int = 12
MACD_SLOW: int = 26
MACD_SIGNAL: int = 9
BBANDS_PERIOD: int = 20
BBANDS_NBDEV: int = 2
EMA_PERIODS: tuple[int, ...] = (9, 21, 50)
ATR_PERIOD: int = 14

# Forward-looking label horizon (number of candles ahead for the target).
# 15 candles = ~15 min ahead. The original 5 (~5 min) was mostly noise at 1-min
# granularity; a longer horizon gives the trend more room to emerge.
LABEL_HORIZON: int = 5
# ATR scaling applied to the default label threshold: threshold = factor * atr/close.
# 0.75 (up from 0.5) labels only meaningful moves as Buy/Sell (clearer targets).
DEFAULT_THRESHOLD_ATR_FACTOR: float = 0.75

# Default instrument (NIFTY 50 index). F&O contract keys may be substituted.
DEFAULT_INSTRUMENT_KEY: str = "NSE_INDEX|Nifty 50"

# Default paths / runtime knobs.
DEFAULT_MODEL_PATH: str = "model.pkl"
DEFAULT_ALERT_COOLDOWN_SECONDS: int = 60

# Backoff for WebSocket reconnect and HTTP 429 handling (seconds), capped.
MIN_RECONNECT_BACKOFF_SECONDS: float = 1.0
MAX_RECONNECT_BACKOFF_SECONDS: float = 60.0

# Upstox endpoints.
UPSTOX_REST_BASE: str = "https://api.upstox.com"
UPSTOX_HISTORICAL_PATH: str = "/v3/historical-candle"
UPSTOX_WS_AUTHORIZE_PATH: str = "/v3/feed/market-data-feed/authorize"


@dataclass
class Config:
    """Typed application configuration loaded from environment variables."""

    # Upstox
    access_token: Optional[str] = None
    api_key: Optional[str] = None
    api_secret: Optional[str] = None
    instrument_key: str = DEFAULT_INSTRUMENT_KEY

    # Telegram
    telegram_bot_token: Optional[str] = None
    telegram_chat_id: Optional[str] = None

    # Server / paths
    port: int = 5000
    model_path: str = DEFAULT_MODEL_PATH

    # Model / signals
    signal_threshold: Optional[float] = None  # None => ATR-scaled default

    # Observability
    log_level: str = "INFO"

    # Runtime knobs
    alert_cooldown_seconds: int = DEFAULT_ALERT_COOLDOWN_SECONDS

    # Feature/label constants (mirrored here for the bundle config + live use).
    rsi_period: int = RSI_PERIOD
    macd_fast: int = MACD_FAST
    macd_slow: int = MACD_SLOW
    macd_signal: int = MACD_SIGNAL
    bbands_period: int = BBANDS_PERIOD
    bbands_nbdev: int = BBANDS_NBDEV
    ema_periods: tuple[int, ...] = field(default_factory=lambda: tuple(EMA_PERIODS))
    atr_period: int = ATR_PERIOD
    label_horizon: int = LABEL_HORIZON

    # --- Construction ---
    @classmethod
    def from_env(cls, env: Optional[dict[str, str]] = None) -> "Config":
        """Build a Config from an environment mapping (defaults to os.environ)."""
        e = env if env is not None else os.environ
        return cls(
            access_token=_opt(e, "UPSTOX_ACCESS_TOKEN"),
            api_key=_opt(e, "UPSTOX_API_KEY"),
            api_secret=_opt(e, "UPSTOX_API_SECRET"),
            instrument_key=e.get("UPSTOX_INSTRUMENT_KEY", DEFAULT_INSTRUMENT_KEY) or DEFAULT_INSTRUMENT_KEY,
            telegram_bot_token=_opt(e, "TELEGRAM_BOT_TOKEN"),
            telegram_chat_id=_opt(e, "TELEGRAM_CHAT_ID"),
            port=_int(e, "PORT", 5000),
            model_path=e.get("MODEL_PATH", DEFAULT_MODEL_PATH) or DEFAULT_MODEL_PATH,
            signal_threshold=_float_or_none(e, "SIGNAL_THRESHOLD"),
            log_level=(e.get("LOG_LEVEL", "INFO") or "INFO").upper(),
            alert_cooldown_seconds=_int(e, "ALERT_COOLDOWN_SECONDS", DEFAULT_ALERT_COOLDOWN_SECONDS),
        )

    # --- Validation ---
    def require_secrets(self) -> list[str]:
        """Return the names of required live secrets that are missing/empty."""
        missing: list[str] = []
        if not self.access_token:
            missing.append("UPSTOX_ACCESS_TOKEN")
        if not self.telegram_bot_token:
            missing.append("TELEGRAM_BOT_TOKEN")
        if not self.telegram_chat_id:
            missing.append("TELEGRAM_CHAT_ID")
        return missing

    def validate_for_live(self) -> None:
        """Raise a clear RuntimeError listing any missing required live secrets."""
        missing = self.require_secrets()
        if missing:
            raise RuntimeError(
                "Missing required live secrets: "
                + ", ".join(missing)
                + ". Set them in the environment (see .env.example). The Flask "
                "app still boots and /health returns 200, but live streaming "
                "and alerts are disabled until they are provided."
            )

    # --- Feature config dict (stored in the model bundle) ---
    def feature_config_dict(self) -> dict:
        """A serialisable snapshot of the feature/label params for the bundle."""
        return {
            "rsi_period": self.rsi_period,
            "macd_fast": self.macd_fast,
            "macd_slow": self.macd_slow,
            "macd_signal": self.macd_signal,
            "bbands_period": self.bbands_period,
            "bbands_nbdev": self.bbands_nbdev,
            "ema_periods": list(self.ema_periods),
            "atr_period": self.atr_period,
            "label_horizon": self.label_horizon,
            "default_threshold_atr_factor": DEFAULT_THRESHOLD_ATR_FACTOR,
        }


def redact(secret: Optional[str], show: int = 6) -> str:
    """Return a safe-to-log representation of a secret (first `show` chars + ***)."""
    if not secret:
        return "<unset>"
    if len(secret) <= show:
        return "***"
    return f"{secret[:show]}***"


def configure_logging(level: str = "INFO") -> logging.Logger:
    """Configure root logging to stdout (structured, timestamped) per NFR-5."""
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )
    return logging.getLogger("fo-alert-agent")


# --- env helpers ---
def _opt(env: dict[str, str], key: str) -> Optional[str]:
    val = env.get(key)
    if val is None or val.strip() == "":
        return None
    return val.strip()


def _int(env: dict[str, str], key: str, default: int) -> int:
    val = env.get(key)
    if val is None or val.strip() == "":
        return default
    try:
        return int(float(val.strip()))
    except (TypeError, ValueError):
        return default


def _float_or_none(env: dict[str, str], key: str) -> Optional[float]:
    val = env.get(key)
    if val is None or val.strip() == "":
        return None
    try:
        return float(val.strip())
    except (TypeError, ValueError):
        return None
