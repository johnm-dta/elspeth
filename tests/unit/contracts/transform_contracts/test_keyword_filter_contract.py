# tests/unit/contracts/transform_contracts/test_keyword_filter_contract.py
from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest

from elspeth.plugins.infrastructure.config_base import PluginConfigError
from elspeth.plugins.transforms.keyword_filter import KeywordFilter
from elspeth.testing import make_pipeline_row
from tests.fixtures.factories import make_context

from .test_transform_protocol import (
    TransformContractPropertyTestBase,
    TransformErrorContractTestBase,
)

if TYPE_CHECKING:
    from elspeth.contracts import TransformProtocol


def _keyword_filter(*, blocked_patterns: list[str]) -> KeywordFilter:
    transform = KeywordFilter(
        {
            "fields": ["content"],
            "blocked_patterns": blocked_patterns,
            "schema": {"mode": "observed"},
        }
    )
    transform.on_error = "quarantine_sink"
    return transform


class TestKeywordFilterContract(TransformContractPropertyTestBase):
    @pytest.fixture
    def transform(self) -> TransformProtocol:
        t = KeywordFilter(
            {
                "fields": ["content"],
                "blocked_patterns": [r"\btest\b"],
                "schema": {"mode": "observed"},
            }
        )
        t.on_error = "quarantine_sink"
        return t

    @pytest.fixture
    def valid_input(self) -> dict[str, Any]:
        return {"content": "safe message without blocked words", "id": 1}

    def test_blocks_literal_regex_metacharacters_when_pattern_escapes_them(self) -> None:
        transform = _keyword_filter(blocked_patterns=[r"C\+\+"])
        row = make_pipeline_row({"content": "candidate mentions C++", "id": 1})

        result = transform.process(row, make_context(run_id="test"))

        assert result.status == "error"
        assert result.reason is not None
        assert result.reason["reason"] == "blocked_content"
        assert result.reason["matched_pattern"] == r"C\+\+"
        assert result.reason["match_position"] == len("candidate mentions ")
        assert result.reason["match_length"] == len("C++")

    def test_blocks_unicode_pattern_matches(self) -> None:
        transform = _keyword_filter(blocked_patterns=["naive|naïve"])
        row = make_pipeline_row({"content": "review says naïve cafe", "id": 1})

        result = transform.process(row, make_context(run_id="test"))

        assert result.status == "error"
        assert result.reason is not None
        assert result.reason["reason"] == "blocked_content"
        assert result.reason["matched_pattern"] == "naive|naïve"

    def test_patterns_are_case_sensitive_unless_operator_opts_in(self) -> None:
        case_sensitive = _keyword_filter(blocked_patterns=["secret"])
        case_insensitive = _keyword_filter(blocked_patterns=["(?i)secret"])
        row = make_pipeline_row({"content": "SECRET", "id": 1})

        sensitive_result = case_sensitive.process(row, make_context(run_id="test"))
        insensitive_result = case_insensitive.process(row, make_context(run_id="test"))

        assert sensitive_result.status == "success"
        assert insensitive_result.status == "error"
        assert insensitive_result.reason is not None
        assert insensitive_result.reason["matched_pattern"] == "(?i)secret"

    @pytest.mark.parametrize(
        ("fields", "message"),
        [
            ("", "fields cannot be empty"),
            ([], "fields list cannot be empty"),
            ([""], r"fields\[0\] cannot be empty"),
        ],
    )
    def test_empty_scan_fields_fail_at_construction(self, fields: str | list[str], message: str) -> None:
        with pytest.raises(PluginConfigError, match=message):
            KeywordFilter(
                {
                    "fields": fields,
                    "blocked_patterns": ["blocked"],
                    "schema": {"mode": "observed"},
                }
            )


class TestKeywordFilterErrorContract(TransformErrorContractTestBase):
    @pytest.fixture
    def transform(self) -> TransformProtocol:
        t = KeywordFilter(
            {
                "fields": ["content"],
                "blocked_patterns": [r"\bblocked\b"],
                "schema": {"mode": "observed"},
            }
        )
        t.on_error = "quarantine_sink"
        return t

    @pytest.fixture
    def valid_input(self) -> dict[str, Any]:
        return {"content": "safe message", "id": 1}

    @pytest.fixture
    def error_input(self) -> dict[str, Any]:
        return {"content": "this is blocked content", "id": 2}
