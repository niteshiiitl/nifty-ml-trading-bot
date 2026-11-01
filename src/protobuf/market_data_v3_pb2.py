"""Runtime-built protobuf message classes for the Upstox V3 market data feed.

This is a PLACEHOLDER compiled-equivalent. It builds the message classes at
runtime from a ``FileDescriptorProto`` using only the ``protobuf`` runtime
library — no ``protoc`` / ``grpcio-tools`` needed at runtime. This makes the
live feed decoder work without a build step.

Schema notes (wire compatibility):
- ``feeds`` is modelled as ``repeated FeedEntry{string key=1; Feed value=2}``
  with field number 2, which is WIRE-IDENTICAL to the official
  ``map<string, Feed> feeds = 2`` (a proto map encodes exactly as repeated
  entries carrying field numbers 1/2). So decoding a real Upstox frame with
  ``FeedResponse.feeds`` repeated entries works correctly.
- ``Feed`` is modelled with a plain ``LTPC ltpc = 2`` field rather than a
  ``oneof``. Reading a plain field is wire-compatible with reading the oneof
  member that was set; since we subscribe in `ltpc` mode, only the ``ltpc``
  branch is ever read.

To regenerate from the official proto (recommended for production): run
``python -m grpc_tools.protoc -I src/protobuf --python_out=src/protobuf
src/protobuf/market_data_v3.proto`` and replace this file with the generated
``market_data_v3_pb2.py``. The ``decode_feed`` shim in ``market_data_pb2.py``
is unaffected — it only imports these message classes.

Field numbers are a best-effort reconstruction; VERIFY against the live feed
before relying on real decoding (live decoding cannot be tested without real
credentials — that is expected).
"""

from __future__ import annotations

from google.protobuf import descriptor_pb2, descriptor_pool, message_factory

# Protobuf field label/type constants (from descriptor.proto).
_LABEL_OPTIONAL = descriptor_pb2.FieldDescriptorProto.LABEL_OPTIONAL  # 1
_LABEL_REPEATED = descriptor_pb2.FieldDescriptorProto.LABEL_REPEATED  # 3
_TYPE_DOUBLE = descriptor_pb2.FieldDescriptorProto.TYPE_DOUBLE  # 1
_TYPE_INT64 = descriptor_pb2.FieldDescriptorProto.TYPE_INT64  # 3
_TYPE_INT32 = descriptor_pb2.FieldDescriptorProto.TYPE_INT32  # 5
_TYPE_STRING = descriptor_pb2.FieldDescriptorProto.TYPE_STRING  # 9
_TYPE_MESSAGE = descriptor_pb2.FieldDescriptorProto.TYPE_MESSAGE  # 11

_PACKAGE = "upstox.marketdatafeederv3"


def _add_field(msg: descriptor_pb2.DescriptorProto, name: str, number: int,
               label: int, type_: int, type_name: str | None = None) -> None:
    f = msg.field.add()
    f.name = name
    f.number = number
    f.label = label
    f.type = type_
    if type_name is not None:
        f.type_name = type_name


def _build_file_descriptor() -> descriptor_pb2.FileDescriptorProto:
    fd = descriptor_pb2.FileDescriptorProto()
    fd.name = "market_data_v3.proto"
    fd.package = _PACKAGE
    fd.syntax = "proto3"

    # message DoubleValue { double value = 1; }
    dv = fd.message_type.add()
    dv.name = "DoubleValue"
    _add_field(dv, "value", 1, _LABEL_OPTIONAL, _TYPE_DOUBLE)

    # message LTPC { double ltp=1; int64 ltt=2; int64 ltq=3; double cp=4; DoubleValue iep=5; }
    ltpc = fd.message_type.add()
    ltpc.name = "LTPC"
    _add_field(ltpc, "ltp", 1, _LABEL_OPTIONAL, _TYPE_DOUBLE)
    _add_field(ltpc, "ltt", 2, _LABEL_OPTIONAL, _TYPE_INT64)
    _add_field(ltpc, "ltq", 3, _LABEL_OPTIONAL, _TYPE_INT64)
    _add_field(ltpc, "cp", 4, _LABEL_OPTIONAL, _TYPE_DOUBLE)
    _add_field(ltpc, "iep", 5, _LABEL_OPTIONAL, _TYPE_MESSAGE, f".{_PACKAGE}.DoubleValue")

    # message Feed { LTPC ltpc = 2; }   (plain field; wire-compatible with oneof)
    feed = fd.message_type.add()
    feed.name = "Feed"
    _add_field(feed, "ltpc", 2, _LABEL_OPTIONAL, _TYPE_MESSAGE, f".{_PACKAGE}.LTPC")

    # message FeedEntry { string key=1; Feed value=2; }   (== map<string,Feed> feeds=2)
    fe = fd.message_type.add()
    fe.name = "FeedEntry"
    _add_field(fe, "key", 1, _LABEL_OPTIONAL, _TYPE_STRING)
    _add_field(fe, "value", 2, _LABEL_OPTIONAL, _TYPE_MESSAGE, f".{_PACKAGE}.Feed")

    # message FeedResponse { int32 type=1; repeated FeedEntry feeds=2; int64 currentTs=3; }
    fr = fd.message_type.add()
    fr.name = "FeedResponse"
    _add_field(fr, "type", 1, _LABEL_OPTIONAL, _TYPE_INT32)
    _add_field(fr, "feeds", 2, _LABEL_REPEATED, _TYPE_MESSAGE, f".{_PACKAGE}.FeedEntry")
    _add_field(fr, "currentTs", 3, _LABEL_OPTIONAL, _TYPE_INT64)

    return fd


# Build a private pool + classes once at import.
_POOL = descriptor_pool.DescriptorPool()
_POOL.Add(_build_file_descriptor())


def _message_class(name: str):
    """Return the runtime-built message class for a top-level message name."""
    return message_factory.GetMessageClass(_POOL.FindMessageTypeByName(f"{_PACKAGE}.{name}"))


# Public message classes (mirroring a protoc-generated module).
DoubleValue = _message_class("DoubleValue")
LTPC = _message_class("LTPC")
Feed = _message_class("Feed")
FeedEntry = _message_class("FeedEntry")
FeedResponse = _message_class("FeedResponse")

__all__ = ["DoubleValue", "LTPC", "Feed", "FeedEntry", "FeedResponse"]
