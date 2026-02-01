# tests/contracts/transform_contracts/test_keyword_filter_contract.py
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
    @pytest.fixture
    def transform(self) -> TransformProtocol:
        t = KeywordFilter(
            {
                "fields": ["content"],
                "blocked_patterns": [r"\btest\b"],
                "schema": {"fields": "dynamic"},
                "on_error": "quarantine_sink",
            }
        )
        assert t._on_error is not None
        return t

    @pytest.fixture
    def valid_input(self) -> dict[str, Any]:
        return {"content": "safe message without blocked words", "id": 1}


class TestKeywordFilterErrorContract(TransformErrorContractTestBase):
    @pytest.fixture
    def transform(self) -> TransformProtocol:
        t = KeywordFilter(
            {
                "fields": ["content"],
                "blocked_patterns": [r"\bblocked\b"],
                "schema": {"fields": "dynamic"},
                "on_error": "quarantine_sink",
            }
        )
        assert t._on_error is not None
        return t

    @pytest.fixture
    def valid_input(self) -> dict[str, Any]:
        return {"content": "safe message", "id": 1}

    @pytest.fixture
    def error_input(self) -> dict[str, Any]:
        return {"content": "this is blocked content", "id": 2}
