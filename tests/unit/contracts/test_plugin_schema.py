"""Tests for the public PluginSchema contract export surface."""

from __future__ import annotations

import pytest

import elspeth.contracts as contracts
import elspeth.contracts.data as data_contracts
from elspeth.contracts import CompatibilityResult

DATA_CONTRACT_EXPORTS = frozenset(
    {
        "CompatibilityResult",
        "PluginSchema",
        "SchemaValidationError",
        "check_compatibility",
        "validate_row",
    }
)


def test_contracts_re_exports_plugin_schema_data_surface() -> None:
    """The package-level contracts API points at the L0 data contract module."""
    assert set(contracts.__all__) >= DATA_CONTRACT_EXPORTS
    for name in DATA_CONTRACT_EXPORTS:
        assert getattr(contracts, name) is getattr(data_contracts, name)


class TestCompatibilityResultErrorMessage:
    """Tests for CompatibilityResult.error_message formatting logic."""

    def test_compatible_result_returns_none(self) -> None:
        result = CompatibilityResult(compatible=True)
        assert result.error_message is None

    def test_combined_errors_are_ordered_and_joined_with_semicolon(self) -> None:
        result = CompatibilityResult(
            compatible=False,
            missing_fields=("name",),
            type_mismatches=(("age", "int", "str"),),
            constraint_mismatches=(("score", "out of range"),),
            extra_fields=("debug",),
        )

        assert result.error_message == (
            "Missing fields: name; "
            "Type mismatches: age (expected int, got str); "
            "Constraint mismatches: score: out of range; "
            "Extra fields forbidden by consumer: debug"
        )

    def test_incompatible_with_no_details_returns_empty_string(self) -> None:
        """Edge case: compatible=False but no error details produces empty string."""
        result = CompatibilityResult(compatible=False)
        assert result.error_message == ""


class TestPluginSchemaNotInOldLocation:
    """Verify plugins/schemas.py has been deleted."""

    def test_old_import_path_removed(self) -> None:
        """Importing from plugins.schemas should fail - module deleted."""
        with pytest.raises(ModuleNotFoundError):
            from elspeth.plugins.schemas import (  # type: ignore[import-not-found]
                PluginSchema,  # noqa: F401
            )
