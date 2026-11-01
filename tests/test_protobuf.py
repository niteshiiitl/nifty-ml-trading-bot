"""Tests for src/protobuf decode_feed shim (no network; uses runtime-built classes)."""
from __future__ import annotations

import pytest

from src.protobuf import decode_feed
from src.protobuf.market_data_v3_pb2 import FeedResponse, LTPC, Feed, FeedEntry


def _build_response() -> bytes:
    # Build via explicit construction + CopyFrom so the test is robust across
    # protobuf backends (avoids any lazy-sub-message subtlety on read).
    ltpc = LTPC()
    ltpc.ltp = 24150.35
    ltpc.ltt = 1730352300
    ltpc.ltq = 250
    ltpc.cp = 24145.10
    feed = Feed()
    feed.ltpc.CopyFrom(ltpc)
    resp = FeedResponse()
    resp.type = 5  # RESPONSE
    entry = resp.feeds.add()
    entry.key = "NSE_INDEX|Nifty 50"
    entry.value.CopyFrom(feed)
    resp.currentTs = 1730352300
    return resp.SerializeToString()


def test_decode_feed_returns_numeric_ltp_and_int_ltt():
    raw = _build_response()
    decoded = decode_feed(raw)
    assert "NSE_INDEX|Nifty 50" in decoded
    d = decoded["NSE_INDEX|Nifty 50"]
    assert d["ltp"] == pytest.approx(24150.35)
    assert isinstance(d["ltp"], float)
    assert d["ltt"] == 1730352300
    assert isinstance(d["ltt"], int)
    assert d["ltq"] == 250
    assert isinstance(d["ltq"], int)
    assert d["cp"] == pytest.approx(24145.10)


def test_decode_feed_multiple_instruments():
    resp = FeedResponse()
    for k, price in (("A|X", 100.0), ("B|Y", 200.0)):
        ltpc = LTPC()
        ltpc.ltp = price
        ltpc.ltt = 1000
        ltpc.ltq = 10
        ltpc.cp = price
        feed = Feed()
        feed.ltpc.CopyFrom(ltpc)
        e = resp.feeds.add()
        e.key = k
        e.value.CopyFrom(feed)
    decoded = decode_feed(resp.SerializeToString())
    assert set(decoded.keys()) == {"A|X", "B|Y"}
    assert decoded["A|X"]["ltp"] == pytest.approx(100.0)
    assert decoded["B|Y"]["ltp"] == pytest.approx(200.0)


def test_decode_feed_empty_message_returns_empty():
    resp = FeedResponse()
    assert decode_feed(resp.SerializeToString()) == {}


def test_decode_feed_empty_bytes_returns_empty():
    assert decode_feed(b"") == {}


def test_decode_feed_garbage_does_not_raise():
    # A random byte string is unlikely to be a valid FeedResponse but must not
    # crash the caller; the shim logs and returns {}.
    result = decode_feed(b"\x00\x01\x02\x03not-a-proto")
    assert isinstance(result, dict)


def test_classes_construct_and_serialize():
    # Sanity: the runtime-built classes behave like normal protobuf messages.
    ltpc = LTPC()
    ltpc.ltp = 1.5
    ltpc.ltt = 7
    feed = Feed()
    feed.ltpc.CopyFrom(ltpc)
    fe = FeedEntry()
    fe.key = "k"
    fe.value.CopyFrom(feed)
    assert fe.key == "k"
    assert fe.value.ltpc.ltp == pytest.approx(1.5)
