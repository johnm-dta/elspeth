"""Protocol alignment tests for PluginContext decomposition.

Modeled on tests/unit/core/test_config_alignment.py — these verify that
PluginContext satisfies all 4 phase-based protocols structurally.
"""

import dataclasses
import inspect

from elspeth.contracts.contexts import (
    LifecycleContext,
    SinkContext,
    SourceContext,
    TransformContext,
)
from elspeth.core.landscape.database import LandscapeDB
from elspeth.core.landscape.factory import RecorderFactory


class TestPluginContextSatisfiesProtocols:
    """Verify PluginContext structurally satisfies all 4 protocols.

    These are the critical alignment tests. If PluginContext is missing
    a field or method that a protocol declares, isinstance() fails.
    """

    def _make_ctx(self) -> object:
        from elspeth.contracts.plugin_context import PluginContext

        db = LandscapeDB.in_memory()
        recorder = RecorderFactory(db).plugin_audit_writer()
        return PluginContext(run_id="test", config={}, landscape=recorder)

    def test_satisfies_source_context(self) -> None:
        ctx = self._make_ctx()
        assert isinstance(ctx, SourceContext)

    def test_satisfies_transform_context(self) -> None:
        ctx = self._make_ctx()
        assert isinstance(ctx, TransformContext)

    def test_satisfies_sink_context(self) -> None:
        ctx = self._make_ctx()
        assert isinstance(ctx, SinkContext)

    def test_satisfies_lifecycle_context(self) -> None:
        ctx = self._make_ctx()
        assert isinstance(ctx, LifecycleContext)


class TestProtocolDiscrimination:
    """Verify protocols are not trivially satisfied — each has unique requirements.

    [R2] Uses real minimal objects, not just set arithmetic on field names.
    A minimal SourceContext-only object must NOT satisfy TransformContext, etc.
    """

    def test_minimal_source_does_not_satisfy_transform(self) -> None:
        """A minimal SourceContext-only object should not satisfy TransformContext."""

        @dataclasses.dataclass
        class MinimalSource:
            run_id: str = "test"
            node_id: str | None = None
            operation_id: str | None = None
            landscape: object = None
            telemetry_emit: object = lambda _: None

            def record_validation_error(self, *a: object, **kw: object) -> None: ...

            def record_call(self, *a: object, **kw: object) -> None: ...

        obj = MinimalSource()
        # Intentional runtime protocol structural subtyping checks — mypy considers
        # these unreachable because the concrete class doesn't nominally inherit
        # from the protocol, but that's exactly what we're verifying at runtime.
        assert isinstance(obj, SourceContext), "MinimalSource should satisfy SourceContext"  # type: ignore[unreachable]
        assert not isinstance(obj, TransformContext), "MinimalSource should NOT satisfy TransformContext"  # type: ignore[unreachable]

    def test_minimal_transform_does_not_satisfy_source(self) -> None:
        """A minimal TransformContext-only object should not satisfy SourceContext."""

        @dataclasses.dataclass
        class MinimalTransform:
            run_id: str = "test"
            state_id: str | None = None
            node_id: str | None = None
            token: object = None
            batch_token_ids: object = None
            contract: object = None

            def record_call(self, *a: object, **kw: object) -> None: ...

        obj = MinimalTransform()
        # Intentional runtime protocol structural subtyping checks (see above).
        assert isinstance(obj, TransformContext), "MinimalTransform should satisfy TransformContext"  # type: ignore[unreachable]
        assert not isinstance(obj, SourceContext), "MinimalTransform should NOT satisfy SourceContext"  # type: ignore[unreachable]

    def test_minimal_sink_does_not_satisfy_lifecycle(self) -> None:
        """A minimal SinkContext-only object should not satisfy LifecycleContext."""

        @dataclasses.dataclass
        class MinimalSink:
            run_id: str = "test"
            contract: object = None
            landscape: object = None
            operation_id: str | None = None

            def record_call(self, *a: object, **kw: object) -> None: ...

        obj = MinimalSink()
        # Intentional runtime protocol structural subtyping checks (see above).
        assert isinstance(obj, SinkContext), "MinimalSink should satisfy SinkContext"  # type: ignore[unreachable]
        assert not isinstance(obj, LifecycleContext), "MinimalSink should NOT satisfy LifecycleContext"  # type: ignore[unreachable]


# [R3] Executor-only fields: on PluginContext but intentionally NOT in any protocol.
# These are fields the engine mutates directly and plugins never access via ctx.
# _pending_quarantine_validation_errors is orchestrator bookkeeping used to
# link a validation error audit record to the later persisted quarantine row.
EXECUTOR_ONLY_FIELDS = {
    "config",
    "_pending_quarantine_validation_errors",
}

# [R4] Engine-internal methods: on PluginContext but called by engine, not plugins.
ENGINE_INTERNAL_METHODS = {
    "record_transform_error",
    "pop_pending_quarantine_validation_error_id",
}


class TestProtocolFieldCoverage:
    """Verify protocol fields map to real PluginContext attributes.

    [R3] Uses mechanical introspection (not hardcoded lists) to catch drift.
    Modeled on test_config_alignment.py bidirectional verification pattern.
    """

    def _get_protocol_members(self, protocol_cls: type) -> set[str]:
        """Extract all declared members from a Protocol class.

        Uses __dict__ inspection rather than get_type_hints() because
        TYPE_CHECKING imports are unavailable at runtime.
        """
        return {name for name in protocol_cls.__dict__ if not name.startswith("_")}

    def _get_plugin_context_fields(self) -> set[str]:
        """Get all field names from PluginContext dataclass."""
        from elspeth.contracts.plugin_context import PluginContext

        return {f.name for f in dataclasses.fields(PluginContext)}

    def _get_plugin_context_methods(self) -> set[str]:
        """Get all public method names from PluginContext."""
        from elspeth.contracts.plugin_context import PluginContext

        return {name for name, val in inspect.getmembers(PluginContext, predicate=inspect.isfunction) if not name.startswith("_")}

    def test_all_protocol_fields_exist_on_plugin_context(self) -> None:
        """Every field/method declared in any protocol must exist on PluginContext."""
        from elspeth.contracts.plugin_context import PluginContext

        db = LandscapeDB.in_memory()
        recorder = RecorderFactory(db).plugin_audit_writer()
        ctx = PluginContext(run_id="test", config={}, landscape=recorder)
        all_protocols = [SourceContext, TransformContext, SinkContext, LifecycleContext]
        for protocol in all_protocols:
            for member in self._get_protocol_members(protocol):
                assert hasattr(ctx, member), f"PluginContext missing {protocol.__name__} member: {member}"

    def test_all_plugin_context_fields_accounted_for(self) -> None:
        """[R3] Bidirectional: every PluginContext field must be in at least one protocol
        OR in the explicit EXECUTOR_ONLY_FIELDS allowlist."""
        all_protocol_members: set[str] = set()
        for protocol in [SourceContext, TransformContext, SinkContext, LifecycleContext]:
            all_protocol_members |= self._get_protocol_members(protocol)

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
        for protocol in [SourceContext, TransformContext, SinkContext, LifecycleContext]:
            all_protocol_members |= self._get_protocol_members(protocol)

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
