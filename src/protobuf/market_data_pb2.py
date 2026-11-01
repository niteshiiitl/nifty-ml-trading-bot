"""``decode_feed`` shim for the Upstox V3 market data feed.

Imported by the WebSocket client as the single decode surface:

    from src.protobuf import decode_feed
    ticks = decode_feed(raw_bytes)   # -> {instrument_key: {"ltp","ltt","ltq","cp"}}

It uses the runtime-built message classes in ``market_data_v3_pb2`` (which only
depend on the ``protobuf`` runtime). Swap the underlying module for a protoc-
generated one later without touching callers.
"""

from __future__ import annotations

import logging
from typing import Any

from src.protobuf.market_data_v3_pb2 import FeedResponse

logger = logging.getLogger(__name__)


def decode_feed(raw_bytes: bytes | bytearray | memoryview) -> dict[str, dict[str, Any]]:
    """Decode a Protobuf-encoded Upstox V3 FeedResponse into a normalized dict.

    Args:
        raw_bytes: The raw binary payload received from the WebSocket.

    Returns:
        ``{instrument_key: {"ltp": float, "ltt": int, "ltq": int, "cp": float}}``
        for every feed entry present. Returns ``{}`` for control messages or
        malformed payloads (parse errors are logged, not raised, so the WS loop
        is never crashed by a bad frame).
    """
    if not raw_bytes:
        return {}
    resp = FeedResponse()
    try:
        resp.ParseFromString(bytes(raw_bytes))
    except Exception as exc:  # noqa: BLE001 - malformed binary frame
        logger.warning("Failed to decode feed frame (%s); skipping.", exc)
        return {}

    out: dict[str, dict[str, Any]] = {}
    for entry in resp.feeds:
        key = entry.key
        ltpc = entry.value.ltpc
        out[key] = {
            "ltp": float(ltpc.ltp),
            "ltt": int(ltpc.ltt),
            "ltq": int(ltpc.ltq),
            "cp": float(ltpc.cp),
        }
    return out
