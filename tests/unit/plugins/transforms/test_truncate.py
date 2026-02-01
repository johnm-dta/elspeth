# tests/unit/plugins/transforms/test_truncate.py
"""Unit tests for Truncate transform plugin."""

from __future__ import annotations

import pytest

from elspeth.plugins.context import PluginContext
from elspeth.plugins.transforms.truncate import Truncate


@pytest.fixture
def ctx() -> PluginContext:
    """Create a minimal plugin context for testing."""
    return PluginContext(
        run_id="test-run",
        config={},
        node_id="test-node",
    )


class TestTruncate:
    """Unit tests for Truncate transform."""

    def test_truncates_long_string(self, ctx: PluginContext) -> None:
        """Strings longer than max length are truncated."""
        transform = Truncate(
            {
                "fields": {"title": 10},
                "schema": {"fields": "dynamic"},
            }
        )

        result = transform.process({"title": "This is a very long title"}, ctx)

        assert result.status == "success"
        assert result.row is not None
        assert result.row["title"] == "This is a "
        assert len(result.row["title"]) == 10

    def test_preserves_short_string(self, ctx: PluginContext) -> None:
        """Strings shorter than max length are unchanged."""
        transform = Truncate(
            {
                "fields": {"title": 100},
                "schema": {"fields": "dynamic"},
            }
        )

        result = transform.process({"title": "Short"}, ctx)

        assert result.status == "success"
        assert result.row is not None
        assert result.row["title"] == "Short"

    def test_truncates_with_suffix(self, ctx: PluginContext) -> None:
        """Truncation includes suffix within max length."""
        transform = Truncate(
            {
                "fields": {"title": 13},
                "suffix": "...",
                "schema": {"fields": "dynamic"},
            }
        )

        result = transform.process({"title": "This is a very long title"}, ctx)

        assert result.status == "success"
        assert result.row is not None
        assert result.row["title"] == "This is a ..."
        assert len(result.row["title"]) == 13

    def test_no_suffix_when_not_truncated(self, ctx: PluginContext) -> None:
        """Suffix is not added if string doesn't need truncation."""
        transform = Truncate(
            {
                "fields": {"title": 100},
                "suffix": "...",
                "schema": {"fields": "dynamic"},
            }
        )

        result = transform.process({"title": "Short"}, ctx)

        assert result.status == "success"
        assert result.row is not None
        assert result.row["title"] == "Short"

    def test_multiple_fields(self, ctx: PluginContext) -> None:
        """Multiple fields can be truncated with different lengths."""
        transform = Truncate(
            {
                "fields": {"title": 5, "description": 10},
                "schema": {"fields": "dynamic"},
            }
        )

        result = transform.process(
            {
                "title": "Very long title",
                "description": "Very long description text",
            },
            ctx,
        )

        assert result.status == "success"
        assert result.row is not None
        assert result.row["title"] == "Very "
        assert result.row["description"] == "Very long "

    def test_missing_field_non_strict(self, ctx: PluginContext) -> None:
        """Missing fields are skipped in non-strict mode."""
        transform = Truncate(
            {
                "fields": {"title": 10, "description": 20},
                "strict": False,
                "schema": {"fields": "dynamic"},
            }
        )

        result = transform.process({"title": "Test"}, ctx)

        assert result.status == "success"
        assert result.row is not None
        assert result.row["title"] == "Test"
        assert "description" not in result.row

    def test_missing_field_strict(self, ctx: PluginContext) -> None:
        """Missing fields error in strict mode."""
        transform = Truncate(
            {
                "fields": {"title": 10, "description": 20},
                "strict": True,
                "schema": {"fields": "dynamic"},
            }
        )

        result = transform.process({"title": "Test"}, ctx)

        assert result.status == "error"
        assert result.reason is not None
        assert result.reason["reason"] == "missing_field"
        assert result.reason["field"] == "description"

    def test_non_string_field_unchanged(self, ctx: PluginContext) -> None:
        """Non-string fields are passed through unchanged."""
        transform = Truncate(
            {
                "fields": {"count": 5},
                "schema": {"fields": "dynamic"},
            }
        )

        result = transform.process({"count": 12345678}, ctx)

        assert result.status == "success"
        assert result.row is not None
        assert result.row["count"] == 12345678

    def test_preserves_unspecified_fields(self, ctx: PluginContext) -> None:
        """Fields not in the truncate list are preserved."""
        transform = Truncate(
            {
                "fields": {"title": 5},
                "schema": {"fields": "dynamic"},
            }
        )

        result = transform.process(
            {
                "title": "Long title here",
                "id": 123,
                "other": "unchanged",
            },
            ctx,
        )

        assert result.status == "success"
        assert result.row is not None
        assert result.row["title"] == "Long "
        assert result.row["id"] == 123
        assert result.row["other"] == "unchanged"

    def test_suffix_length_validation(self) -> None:
        """Suffix longer than max length raises error."""
        with pytest.raises(ValueError, match="Suffix length"):
            Truncate(
                {
                    "fields": {"title": 3},
                    "suffix": "...",  # 3 chars, same as max - invalid
                    "schema": {"fields": "dynamic"},
                }
            )

    def test_exact_length_not_truncated(self, ctx: PluginContext) -> None:
        """String exactly at max length is not truncated."""
        transform = Truncate(
            {
                "fields": {"title": 5},
                "schema": {"fields": "dynamic"},
            }
        )

        result = transform.process({"title": "12345"}, ctx)

        assert result.status == "success"
        assert result.row is not None
        assert result.row["title"] == "12345"

    def test_empty_string(self, ctx: PluginContext) -> None:
        """Empty strings are handled correctly."""
        transform = Truncate(
            {
                "fields": {"title": 10},
                "schema": {"fields": "dynamic"},
            }
        )

        result = transform.process({"title": ""}, ctx)

        assert result.status == "success"
        assert result.row is not None
        assert result.row["title"] == ""
