"""Tests for DAG builder enforcement of _output_schema_config.

Verifies that transforms declaring output fields without providing
an _output_schema_config raise FrameworkBugError at graph-build time.
"""

from typing import Any, ClassVar

import pytest

from elspeth.contracts.errors import FrameworkBugError
from elspeth.contracts.schema import SchemaConfig


class _StubTransform:
    """Minimal stub implementing enough of TransformProtocol for the builder check."""

    name = "stub_transform"
    config: ClassVar[dict[str, Any]] = {"schema": {"mode": "observed"}}
    input_schema = None
    output_schema = None
    declared_output_fields: frozenset[str] = frozenset()
    _output_schema_config: SchemaConfig | None = None


class TestOutputSchemaEnforcement:
    def test_nonempty_declared_fields_without_config_raises(self):
        """Transform declares output fields but no _output_schema_config -> FrameworkBugError."""
        stub = _StubTransform()
        stub.declared_output_fields = frozenset({"field_a", "field_b"})
        stub._output_schema_config = None

        from elspeth.core.dag.builder import _validate_output_schema_contract

        with pytest.raises(FrameworkBugError, match="declares output fields"):
            _validate_output_schema_contract(stub)

    def test_nonempty_declared_fields_with_valid_config_passes(self):
        stub = _StubTransform()
        stub.declared_output_fields = frozenset({"field_a"})
        stub._output_schema_config = SchemaConfig(mode="observed", fields=None, guaranteed_fields=("field_a",))

        from elspeth.core.dag.builder import _validate_output_schema_contract

        _validate_output_schema_contract(stub)  # Should not raise

    def test_empty_declared_fields_without_config_passes(self):
        """Shape-preserving transforms (no declared fields) don't need _output_schema_config."""
        stub = _StubTransform()
        stub.declared_output_fields = frozenset()
        stub._output_schema_config = None

        from elspeth.core.dag.builder import _validate_output_schema_contract

        _validate_output_schema_contract(stub)  # Should not raise

    def test_declared_fields_subset_of_guaranteed_passes(self):
        """Declared output fields that are a subset of guaranteed_fields pass."""
        stub = _StubTransform()
        stub.declared_output_fields = frozenset({"field_a"})
        stub._output_schema_config = SchemaConfig(mode="observed", fields=None, guaranteed_fields=("field_a", "field_b"))

        from elspeth.core.dag.builder import _validate_output_schema_contract

        _validate_output_schema_contract(stub)  # Should not raise

    def test_declared_fields_not_in_guaranteed_raises(self):
        """Declared output fields missing from guaranteed_fields → FrameworkBugError."""
        stub = _StubTransform()
        stub.declared_output_fields = frozenset({"field_a", "field_b"})
        stub._output_schema_config = SchemaConfig(mode="observed", fields=None, guaranteed_fields=("field_a",))

        from elspeth.core.dag.builder import _validate_output_schema_contract

        with pytest.raises(FrameworkBugError, match="field_b"):
            _validate_output_schema_contract(stub)

    def test_declared_fields_with_none_guaranteed_skips_containment(self):
        """When declares_guaranteed_fields is False, containment check is skipped."""
        stub = _StubTransform()
        stub.declared_output_fields = frozenset({"field_a"})
        stub._output_schema_config = SchemaConfig(mode="observed", fields=None, guaranteed_fields=None)

        from elspeth.core.dag.builder import _validate_output_schema_contract

        # Forward check passes (config exists), containment skipped (no declaration)
        _validate_output_schema_contract(stub)  # Should not raise

    def test_empty_declared_with_nonempty_config_passes(self):
        """Vacuous case: empty declared_output_fields with config present → no checks needed."""
        stub = _StubTransform()
        stub.declared_output_fields = frozenset()
        stub._output_schema_config = SchemaConfig(mode="observed", fields=None, guaranteed_fields=("field_a",))

        from elspeth.core.dag.builder import _validate_output_schema_contract

        _validate_output_schema_contract(stub)  # Should not raise
