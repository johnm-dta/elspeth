# tests/contracts/transform_contracts/test_truncate_contract.py
"""Contract tests for Truncate transform.

Verifies Truncate honors the TransformProtocol contract.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest

from elspeth.plugins.transforms.truncate import Truncate

from .test_transform_protocol import (
    TransformContractPropertyTestBase,
    TransformErrorContractTestBase,
)

if TYPE_CHECKING:
    from elspeth.plugins.protocols import TransformProtocol


class TestTruncateContract(TransformContractPropertyTestBase):
    """Contract tests for Truncate plugin."""

    @pytest.fixture
    def transform(self) -> TransformProtocol:
        """Return a configured transform instance."""
        return Truncate(
            {
                "fields": {"title": 20, "description": 50},
                "schema": {"fields": "dynamic"},
            }
        )

    @pytest.fixture
    def valid_input(self) -> dict[str, Any]:
        """Return input that should process successfully."""
        return {"title": "Short title", "description": "Short description", "id": 1}


class TestTruncateWithSuffixContract(TransformContractPropertyTestBase):
    """Contract tests for Truncate plugin with suffix."""

    @pytest.fixture
    def transform(self) -> TransformProtocol:
        """Return a configured transform instance with suffix."""
        return Truncate(
            {
                "fields": {"title": 20},
                "suffix": "...",
                "schema": {"fields": "dynamic"},
            }
        )

    @pytest.fixture
    def valid_input(self) -> dict[str, Any]:
        """Return input that should process successfully."""
        return {"title": "Short", "id": 1}


class TestTruncateStrictContract(TransformErrorContractTestBase):
    """Contract tests for Truncate error handling in strict mode."""

    @pytest.fixture
    def transform(self) -> TransformProtocol:
        """Return a configured transform instance in strict mode."""
        return Truncate(
            {
                "fields": {"required_field": 100},
                "strict": True,
                "schema": {"fields": "dynamic"},
            }
        )

    @pytest.fixture
    def valid_input(self) -> dict[str, Any]:
        """Return input that should process successfully (has required field)."""
        return {"required_field": "value", "id": 1}

    @pytest.fixture
    def error_input(self) -> dict[str, Any]:
        """Return input that triggers error (missing required field in strict mode)."""
        return {"other_field": "value", "id": 2}

    def test_strict_missing_field_returns_error(
        self,
        transform: TransformProtocol,
        error_input: dict[str, Any],
        ctx: Any,
    ) -> None:
        """Contract: Strict mode MUST return error with missing_field reason."""
        from elspeth.plugins.context import PluginContext

        ctx = PluginContext(run_id="test", config={})
        result = transform.process(error_input, ctx)
        assert result.status == "error"
        assert result.reason is not None
        assert result.reason["reason"] == "missing_field"
