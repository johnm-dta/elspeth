"""Tests that the Phase-7A reference-content fields are declared on all
four plugin protocol classes in contracts/plugin_protocols.py.

Walks ``cls.__mro__`` and unions each base's ``__annotations__`` rather
than calling ``typing.get_type_hints()`` or ``hasattr()``.
``plugin_protocols.py`` defers context types (``PluginSchema``,
``SourceContext``, ``TransformContext``, ``SinkContext``, etc.) into
TYPE_CHECKING blocks; ``typing.get_type_hints()`` would attempt to
resolve every annotation at runtime and raise ``NameError`` on those
forward references — the test would error rather than assert, and
would never pass in green state. ``hasattr()`` is unconditionally
banned (CLAUDE.md). ``SourceProtocol`` and ``SinkProtocol`` are not
``@runtime_checkable``, ruling out ``isinstance()``.

The MRO walk is the runtime-safe equivalent of structural inheritance:
the Phase-7A reference-content fields now live on the
``_PluginReferenceContent`` mixin and are inherited by each protocol.
The test verifies inheritance by union'ing annotations across every
base in the MRO, which mirrors how a static checker (mypy) resolves
inherited attributes for a Protocol's structural contract.
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
}


def _collect_inherited_annotations(cls: type) -> set[str]:
    """Union ``__annotations__`` across the MRO via per-class dicts.

    ``__annotations__`` is per-class, not inherited, but accessing it
    via attribute lookup falls back through the MRO and would conflate
    contributions across bases. ``vars(base).get("__annotations__", {})``
    reads the class's own ``__dict__`` directly — strictly per-class.
    The empty-dict default handles the one MRO entry that lacks the
    attribute outright (``object``); every Protocol subclass on the
    walk path has its own ``__annotations__`` populated by the
    interpreter at class-creation time.

    Walking the MRO and union'ing reproduces what a structural
    type-checker sees as the protocol's full attribute set, including
    names contributed by mixin bases like ``_PluginReferenceContent``.
    """
    names: set[str] = set()
    for base in cls.__mro__:
        base_annotations = vars(base).get("__annotations__", {})
        names |= set(base_annotations.keys())
    return names


def _assert_protocol_has_fields(protocol: type, protocol_name: str) -> None:
    hints = _collect_inherited_annotations(protocol)
    missing = _PHASE_7A_FIELDS - hints
    assert not missing, (
        f"{protocol_name} is missing Phase-7A fields: {sorted(missing)}. "
        f"They should be inherited from _PluginReferenceContent in "
        f"src/elspeth/contracts/plugin_protocols.py."
    )


def test_source_protocol_has_phase_7a_fields() -> None:
    _assert_protocol_has_fields(SourceProtocol, "SourceProtocol")


def test_transform_protocol_has_phase_7a_fields() -> None:
    _assert_protocol_has_fields(TransformProtocol, "TransformProtocol")


def test_batch_transform_protocol_has_phase_7a_fields() -> None:
    _assert_protocol_has_fields(BatchTransformProtocol, "BatchTransformProtocol")


def test_sink_protocol_has_phase_7a_fields() -> None:
    _assert_protocol_has_fields(SinkProtocol, "SinkProtocol")
