"""Upstox Market Data Feed V3 protobuf decoding package.

Re-exports ``decode_feed`` so callers import a single stable name:

    from src.protobuf import decode_feed

The underlying module (``market_data_pb2.py``) is a runtime-built shim that
depends only on the ``protobuf`` runtime library. See README.md for how to
regenerate it from the official ``market_data_v3.proto``.
"""

from src.protobuf.market_data_pb2 import decode_feed

__all__ = ["decode_feed"]
