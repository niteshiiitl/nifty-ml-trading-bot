"""Upstox Market Data Feed V3 WebSocket client (live pipeline, FR-3.1 — FR-3.7).

Thin I/O wrapper. Connects to the authorized Upstox WS URL, subscribes in
``ltpc`` mode (only last-traded price + time are needed for 1-minute candles),
decodes Protobuf ticks, feeds the ``CandleAggregator``, and on a closed candle
calls the caller's ``on_candle_close`` callback (wired to inference + alerts in
app.py). Auto-reconnects with capped exponential backoff and resubscribes.

This module never places orders; it only consumes market data. The live
connection cannot be tested without real credentials — that is expected; the
module is import-safe and the decode/aggregation paths are unit-tested
elsewhere with synthetic data.
"""

from __future__ import annotations

import json
import logging
import time
from typing import Callable, Optional

import requests

from src.config import (
    Config,
    MAX_RECONNECT_BACKOFF_SECONDS,
    MIN_RECONNECT_BACKOFF_SECONDS,
    UPSTOX_REST_BASE,
    UPSTOX_WS_AUTHORIZE_PATH,
    redact,
)
from src.protobuf import decode_feed

logger = logging.getLogger(__name__)

AUTHORIZE_TIMEOUT = 15
# WebSocket binary frame opcode (RFC 6455). Upstox expects the subscription
# JSON sent as a binary frame.
OPCODE_BINARY = 0x2


class AuthError(RuntimeError):
    """Raised when the Upstox WS authorize endpoint returns 401 (re-auth needed)."""


class UpstoxMarketDataClient:
    """Upstox V3 market data WebSocket client with auto-reconnect.

    Args:
        config: Config carrying the access token + instrument key.
        aggregator: ``CandleAggregator`` that receives decoded ticks.
        on_candle_close: ``Callable[[closed_event: dict], None]`` invoked when a
            1-minute candle closes (the caller runs inference + alerts).
        on_auth_error: Optional ``Callable[[str], None]`` invoked on a 401 so the
            caller can alert the operator that re-authentication is required.
    """

    def __init__(
        self,
        config: Config,
        aggregator,
        on_candle_close: Callable[[dict], None],
        on_auth_error: Optional[Callable[[str], None]] = None,
    ) -> None:
        self.config = config
        self.aggregator = aggregator
        self.on_candle_close = on_candle_close
        self.on_auth_error = on_auth_error
        self._guid = "fo-alert-agent"
        self._ws = None

    # --- authorization ---
    def _authorized_url(self) -> str:
        """Obtain the authorized WS redirect URL via the Upstox REST endpoint."""
        token = self.config.access_token
        if not token:
            raise AuthError("UPSTOX_ACCESS_TOKEN is not set; cannot authorize WS feed.")
        url = f"{UPSTOX_REST_BASE}{UPSTOX_WS_AUTHORIZE_PATH}"
        headers = {"Authorization": f"Bearer {token}", "Accept": "application/json"}
        try:
            resp = requests.get(url, headers=headers, timeout=AUTHORIZE_TIMEOUT)
        except requests.RequestException as exc:
            raise AuthError(f"Network error authorizing WS feed: {exc}") from exc
        if resp.status_code == 401:
            raise AuthError("Upstox returned 401 authorizing WS feed — token expired; re-auth required.")
        if resp.status_code != 200:
            raise AuthError(f"WS authorize endpoint returned HTTP {resp.status_code}: {resp.text[:200]}")
        try:
            data = resp.json().get("data") or {}
        except ValueError as exc:
            raise AuthError(f"WS authorize returned non-JSON: {exc}") from exc
        ws_url = data.get("authorized_redirect_uri")
        if not ws_url:
            raise AuthError("WS authorize response missing 'authorized_redirect_uri'.")
        logger.info("Obtained authorized WS URL (token %s).", redact(token))
        return ws_url

    # --- subscription ---
    def _subscription_payload(self) -> dict:
        return {
            "guid": self._guid,
            "method": "sub",
            "data": {
                "mode": "ltpc",
                "instrumentKeys": [self.config.instrument_key],
            },
        }

    def _on_open(self, ws) -> None:
        payload = json.dumps(self._subscription_payload()).encode("utf-8")
        try:
            ws.send(payload, opcode=OPCODE_BINARY)
            logger.info("Subscribed to %s in ltpc mode.", self.config.instrument_key)
        except Exception as exc:  # noqa: BLE001 - never crash on send
            logger.error("Failed to send subscription: %s", exc)

    def _on_message(self, ws, raw_bytes) -> None:
        try:
            ticks = decode_feed(raw_bytes)
        except Exception as exc:  # noqa: BLE001 - decode must not crash the loop
            logger.warning("decode_feed error: %s", exc)
            return
        for _instrument, tick in ticks.items():
            event = self.aggregator.on_tick(tick)
            if event is not None:  # a candle closed
                try:
                    self.on_candle_close(event)
                except Exception as exc:  # noqa: BLE001 - isolate callback errors
                    logger.exception("on_candle_close raised: %s", exc)

    def _on_error(self, ws, error) -> None:
        logger.warning("WebSocket error: %s", error)

    def _on_close(self, ws, close_status, close_msg) -> None:
        logger.warning("WebSocket closed (%s %s). Will reconnect.", close_status, close_msg)

    # --- run loop ---
    def run_forever(self) -> None:
        """Connect + subscribe, reconnecting with capped exponential backoff.

        Blocks indefinitely (until KeyboardInterrupt). On a 401 it calls
        ``on_auth_error`` and keeps retrying so the live agent stays up awaiting
        operator re-authentication. Resubscription happens in ``_on_open`` on
        every reconnect, so no manual resubscribe bookkeeping is needed.
        """
        backoff = MIN_RECONNECT_BACKOFF_SECONDS
        while True:
            # Obtain the authorized URL (may raise AuthError on 401).
            try:
                url = self._authorized_url()
            except KeyboardInterrupt:
                raise
            except AuthError as exc:
                logger.error("Auth error: %s", exc)
                if self.on_auth_error:
                    try:
                        self.on_auth_error(str(exc))
                    except Exception as e2:  # noqa: BLE001
                        logger.warning("on_auth_error raised: %s", e2)
                time.sleep(min(backoff, MAX_RECONNECT_BACKOFF_SECONDS))
                backoff = min(backoff * 2, MAX_RECONNECT_BACKOFF_SECONDS)
                continue
            except Exception as exc:  # noqa: BLE001
                logger.exception("Unexpected error obtaining WS URL: %s", exc)
                time.sleep(min(backoff, MAX_RECONNECT_BACKOFF_SECONDS))
                backoff = min(backoff * 2, MAX_RECONNECT_BACKOFF_SECONDS)
                continue

            try:
                # Lazy import so this module imports without websocket-client installed.
                import websocket

                self._ws = websocket.WebSocketApp(
                    url,
                    on_open=self._on_open,
                    on_message=self._on_message,
                    on_error=self._on_error,
                    on_close=self._on_close,
                )
                logger.info("Connecting to Upstox WS feed...")
                self._ws.run_forever()  # blocks until the socket drops
            except KeyboardInterrupt:
                raise
            except Exception as exc:  # noqa: BLE001
                logger.exception("WebSocket run_forever crashed: %s", exc)

            # Disconnected — back off and reconnect (resubscription is in on_open).
            logger.info("Reconnecting in %.1fs...", backoff)
            time.sleep(backoff)
            backoff = min(backoff * 2, MAX_RECONNECT_BACKOFF_SECONDS)
