"""Protocol alignment tests for PluginContext decomposition.

Modeled on tests/unit/core/test_config_alignment.py — these verify that
PluginContext satisfies all 4 phase-based protocols structurally.
"""

import inspect
from collections.abc import Callable
from dataclasses import fields
from typing import Any, cast

from elspeth.contracts.contexts import (
    LifecycleContext,
    SinkContext,
    SourceContext,
    TransformContext,
)

PROTOCOLS = (SourceContext, TransformContext, SinkContext, LifecycleContext)


def _get_protocol_members(protocol_cls: type) -> set[str]:
    """Extract members declared directly on a Protocol class."""
    return {name for name in vars(protocol_cls) if not name.startswith("_")}


def _get_protocol_methods(protocol_cls: type) -> dict[str, Callable[..., Any]]:
    """Extract callable protocol methods declared directly on a Protocol class."""
    return {
        name: cast(Callable[..., Any], value)
        for name, value in vars(protocol_cls).items()
        if not name.startswith("_") and inspect.isfunction(value)
    }


def _signature_shape(func: Callable[..., Any]) -> tuple[tuple[tuple[str, Any, object], ...], object]:
    """Compare call contracts without evaluating TYPE_CHECKING-only annotations."""
    signature = inspect.signature(func)
    parameters = tuple((name, parameter.kind, parameter.default) for name, parameter in signature.parameters.items())
    return parameters, signature.return_annotation


class TestPluginContextProtocolContracts:
    """Verify PluginContext satisfies protocol contracts mechanically.

    Runtime ``isinstance(..., Protocol)`` checks only prove member names exist;
    these checks also pin method call shapes so loose ``*args``/``**kwargs``
    fakes cannot masquerade as valid audit-capable contexts.
    """

    def test_protocol_members_are_real_plugin_context_fields_or_methods(self) -> None:
        """Every protocol member must map to a concrete PluginContext field or method."""
        from elspeth.contracts.plugin_context import PluginContext

        plugin_context_fields = {field.name for field in fields(PluginContext)} | {
            name for name, value in vars(PluginContext).items() if isinstance(value, property)
        }
        plugin_context_methods = {
            name for name, value in inspect.getmembers(PluginContext, predicate=inspect.isfunction) if not name.startswith("_")
        }

        for protocol in PROTOCOLS:
            missing = _get_protocol_members(protocol) - plugin_context_fields - plugin_context_methods
            assert not missing, f"PluginContext missing {protocol.__name__} members: {missing}"

    def test_protocol_method_signatures_match_plugin_context(self) -> None:
        """Protocol methods must keep the same callable shape as PluginContext."""
        from elspeth.contracts.plugin_context import PluginContext

        for protocol in PROTOCOLS:
            for method_name, protocol_method in _get_protocol_methods(protocol).items():
                plugin_context_method = cast(Callable[..., Any], getattr(PluginContext, method_name))
                assert _signature_shape(plugin_context_method) == _signature_shape(protocol_method), (
                    f"{protocol.__name__}.{method_name} signature drifted from PluginContext.{method_name}"
                )

    def test_loose_record_call_fake_would_not_satisfy_signature_contract(self) -> None:
        """Guard against accepting ``record_call(*args, **kwargs) -> None`` fakes."""

        class LooseRecordCallFake:
            def record_call(self, *args: object, **kwargs: object) -> None: ...

        assert _signature_shape(LooseRecordCallFake.record_call) != _signature_shape(SourceContext.record_call)


# [R3] Executor-only fields: on PluginContext but intentionally NOT in any protocol.
# These are fields the engine mutates directly and plugins never access via ctx.
# _pending_quarantine_validation_errors is orchestrator bookkeeping used to
# link a validation error audit record to the later persisted quarantine row.
EXECUTOR_ONLY_FIELDS = {
    "config",
    "_config",
    "_pending_quarantine_validation_errors",
}

# [R4] Engine-internal methods: on PluginContext but called by engine, not plugins.
ENGINE_INTERNAL_METHODS = {
    "record_transform_error",
    "pop_pending_quarantine_validation_error_id",
    # Executors clone an operation-scoped context (e.g. sink primary/failsink
    # row contracts) so per-operation attribution cannot leak through the
    # shared context. Plugins receive the scoped copy; they never call this.
    "for_contract",
}


class TestProtocolFieldCoverage:
    """Verify protocol fields map to real PluginContext attributes.

    [R3] Uses mechanical introspection (not hardcoded lists) to catch drift.
    Modeled on test_config_alignment.py bidirectional verification pattern.
    """

    def _get_plugin_context_fields(self) -> set[str]:
        """Get all concrete field/property names from PluginContext."""
        from elspeth.contracts.plugin_context import PluginContext

        dataclass_fields = {field.name for field in fields(PluginContext)}
        properties = {name for name, value in vars(PluginContext).items() if isinstance(value, property)}
        return dataclass_fields | properties

    def _get_plugin_context_methods(self) -> set[str]:
        """Get all public method names from PluginContext."""
        from elspeth.contracts.plugin_context import PluginContext

        return {name for name, val in inspect.getmembers(PluginContext, predicate=inspect.isfunction) if not name.startswith("_")}

    def test_all_protocol_fields_exist_on_plugin_context(self) -> None:
        """Every field/method declared in any protocol must exist on PluginContext."""
        plugin_context_members = self._get_plugin_context_fields() | self._get_plugin_context_methods()
        for protocol in PROTOCOLS:
            missing = _get_protocol_members(protocol) - plugin_context_members
            assert not missing, f"PluginContext missing {protocol.__name__} members: {missing}"

    def test_all_plugin_context_fields_accounted_for(self) -> None:
        """[R3] Bidirectional: every PluginContext field must be in at least one protocol
        OR in the explicit EXECUTOR_ONLY_FIELDS allowlist."""
        all_protocol_members: set[str] = set()
        for protocol in PROTOCOLS:
            all_protocol_members |= _get_protocol_members(protocol)

        plugin_context_fields = self._get_plugin_context_fields()
        unaccounted = plugin_context_fields - all_protocol_members - EXECUTOR_ONLY_FIELDS
        assert not unaccounted, (
            f"PluginContext fields not in any protocol or EXECUTOR_ONLY_FIELDS: {unaccounted}. "
            f"Either add to a protocol or to EXECUTOR_ONLY_FIELDS with justification."
        )

    def test_all_plugin_context_methods_accounted_for(self) -> None:
        """[R3] Bidirectional: every PluginContext public method must be in at least one
        protocol OR in the explicit ENGINE_INTERNAL_METHODS allowlist."""
        all_protocol_members: set[str] = set()
        for protocol in PROTOCOLS:
            all_protocol_members |= _get_protocol_members(protocol)

        plugin_context_methods = self._get_plugin_context_methods()
        unaccounted = plugin_context_methods - all_protocol_members - ENGINE_INTERNAL_METHODS
        assert not unaccounted, (
            f"PluginContext methods not in any protocol or ENGINE_INTERNAL_METHODS: {unaccounted}. "
            f"Either add to a protocol or to ENGINE_INTERNAL_METHODS with justification."
        )

    def test_executor_only_fields_are_real(self) -> None:
        """Every entry in EXECUTOR_ONLY_FIELDS must exist on PluginContext."""
        plugin_context_fields = self._get_plugin_context_fields()
        phantom = EXECUTOR_ONLY_FIELDS - plugin_context_fields
        assert not phantom, f"EXECUTOR_ONLY_FIELDS entries not on PluginContext: {phantom}"

    def test_engine_internal_methods_are_real(self) -> None:
        """Every entry in ENGINE_INTERNAL_METHODS must exist on PluginContext."""
        plugin_context_methods = self._get_plugin_context_methods()
        phantom = ENGINE_INTERNAL_METHODS - plugin_context_methods
        assert not phantom, f"ENGINE_INTERNAL_METHODS entries not on PluginContext: {phantom}"


class TestProtocolOverlapDocumentation:
    """Document field overlap between protocols.

    run_id is intentionally in all 4 protocols. Other fields should
    have minimal overlap. This test serves as documentation — it
    fails if overlap changes unexpectedly.
    """

    EXPECTED_UNIVERSAL: frozenset[str] = frozenset({"run_id"})  # In all protocols by design

    @staticmethod
    def _protocol_properties(cls: type) -> set[str]:
        """Extract @property names defined directly on a Protocol class."""
        return {name for name, val in vars(cls).items() if not name.startswith("_") and isinstance(val, property)}

    def test_universal_fields_are_only_run_id(self) -> None:
        """Only run_id should appear in all 4 protocols."""
        source_fields = self._protocol_properties(SourceContext)
        transform_fields = self._protocol_properties(TransformContext)
        sink_fields = self._protocol_properties(SinkContext)
        lifecycle_fields = self._protocol_properties(LifecycleContext)

        universal = source_fields & transform_fields & sink_fields & lifecycle_fields
        assert universal == self.EXPECTED_UNIVERSAL, f"Expected only {self.EXPECTED_UNIVERSAL} in all protocols, got {universal}"
