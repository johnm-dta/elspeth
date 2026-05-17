"""Tests that the Phase-7A reference-content fields are declared on all
four plugin protocol classes in contracts/plugin_protocols.py.

Uses protocol.__annotations__ rather than typing.get_type_hints() or
hasattr(). plugin_protocols.py defers context types (PluginSchema,
SourceContext, TransformContext, SinkContext, etc.) into TYPE_CHECKING
blocks; typing.get_type_hints() attempts to resolve every annotation as
a runtime expression and raises NameError on those forward references —
the test would error rather than assert, and would never pass in green
state. hasattr() is unconditionally banned (CLAUDE.md). SourceProtocol
and SinkProtocol are not @runtime_checkable, ruling out isinstance().
protocol.__annotations__ is runtime-safe: it returns the directly-
declared annotation dict on the class without resolving forward
references, which is all the test needs to assert field presence.
"""

from __future__ import annotations

from elspeth.contracts.plugin_protocols import (
    BatchTransformProtocol,
    SinkProtocol,
    SourceProtocol,
    TransformProtocol,
)

_PHASE_7A_FIELDS = {
    "usage_when_to_use",
    "usage_when_not_to_use",
    "example_use",
    "capability_tags",
    "audit_characteristics",
    "data_trust_tier",
}


def _assert_protocol_has_fields(protocol: type, protocol_name: str) -> None:
    hints = protocol.__annotations__
    missing = _PHASE_7A_FIELDS - hints.keys()
    assert not missing, (
        f"{protocol_name} is missing Phase-7A fields: {sorted(missing)}. Add them to src/elspeth/contracts/plugin_protocols.py."
    )


def test_source_protocol_has_phase_7a_fields() -> None:
    _assert_protocol_has_fields(SourceProtocol, "SourceProtocol")


def test_transform_protocol_has_phase_7a_fields() -> None:
    _assert_protocol_has_fields(TransformProtocol, "TransformProtocol")


def test_batch_transform_protocol_has_phase_7a_fields() -> None:
    _assert_protocol_has_fields(BatchTransformProtocol, "BatchTransformProtocol")


def test_sink_protocol_has_phase_7a_fields() -> None:
    _assert_protocol_has_fields(SinkProtocol, "SinkProtocol")
