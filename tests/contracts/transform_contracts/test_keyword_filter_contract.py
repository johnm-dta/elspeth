# tests/contracts/transform_contracts/test_keyword_filter_contract.py
"""Contract tests for KeywordFilter transform.

Verifies KeywordFilter honors the TransformProtocol contract.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest

from elspeth.plugins.transforms.keyword_filter import KeywordFilter

from .test_transform_protocol import (
    TransformContractPropertyTestBase,
    TransformErrorContractTestBase,
)

if TYPE_CHECKING:
    from elspeth.plugins.protocols import TransformProtocol


class TestKeywordFilterContract(TransformContractPropertyTestBase):
    """Contract tests for KeywordFilter plugin."""

    @pytest.fixture
    def transform(self) -> TransformProtocol:
        """Return a configured transform instance."""
        return KeywordFilter(
            {
                "fields": ["content"],
                "blocked_patterns": [r"\btest\b"],
                "schema": {"fields": "dynamic"},
            }
        )

    @pytest.fixture
    def valid_input(self) -> dict[str, Any]:
        """Return input that should process successfully (no blocked words)."""
        return {"content": "safe message without blocked words", "id": 1}


class TestKeywordFilterErrorContract(TransformErrorContractTestBase):
    """Contract tests for KeywordFilter error handling."""

    @pytest.fixture
    def transform(self) -> TransformProtocol:
        """Return a configured transform instance."""
        return KeywordFilter(
            {
                "fields": ["content"],
                "blocked_patterns": [r"\bblocked\b"],
                "schema": {"fields": "dynamic"},
            }
        )

    @pytest.fixture
    def valid_input(self) -> dict[str, Any]:
        """Return input that should process successfully."""
        return {"content": "safe message", "id": 1}

    @pytest.fixture
    def error_input(self) -> dict[str, Any]:
        """Return input that triggers blocking (contains blocked pattern)."""
        return {"content": "this is blocked content", "id": 2}
