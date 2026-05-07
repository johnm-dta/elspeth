# tests/unit/mcp/test_contract_tools.py
"""Tests for MCP server contract analysis type contracts.

Migrated from tests/mcp/test_contract_tools.py.
Tests that require LandscapeDB (actual contract queries, SQL blocklist tests)
are deferred to integration tier.
"""


class TestContractFieldTypedDictIncludesNullable:
    """Bug fix for elspeth-7fec8cecec: ContractField and FieldExplanation must include nullable."""

    def test_contract_field_has_nullable_key(self) -> None:
        """ContractField TypedDict must include 'nullable' key."""
        from elspeth.mcp.types import ContractField

        assert "nullable" in ContractField.__annotations__
        assert ContractField.__annotations__["nullable"] is bool

    def test_field_explanation_has_nullable_key(self) -> None:
        """FieldExplanation TypedDict must include 'nullable' key."""
        from elspeth.mcp.types import FieldExplanation

        assert "nullable" in FieldExplanation.__annotations__
        assert FieldExplanation.__annotations__["nullable"] is bool


class TestFailureValidationErrorPluginNullable:
    """Bug fix for elspeth-296609d523: FailureValidationError.plugin can be None."""

    def test_plugin_field_allows_none(self) -> None:
        """FailureValidationError.plugin must be typed as str | None."""
        import types

        from elspeth.mcp.types import FailureValidationError

        annotation = FailureValidationError.__annotations__["plugin"]
        # str | None becomes types.UnionType in Python 3.10+
        assert isinstance(annotation, types.UnionType)
        assert type(None) in annotation.__args__
        assert str in annotation.__args__
