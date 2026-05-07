# tests/unit/contracts/test_error_edge_label.py
"""Tests for error_edge_label() canonical label generation.

error_edge_label() is the single source of truth for error-divert edge labels,
shared between DAG construction (builder.py) and error-routing audit recording
(executors.py, processor.py). Label drift would cause silent routing failures.
"""

from __future__ import annotations

from elspeth.contracts.enums import error_edge_label


class TestErrorEdgeLabel:
    """Verify error_edge_label() produces the canonical format."""

    def test_basic_format(self) -> None:
        """Label wraps transform_id in __error_...__."""
        assert error_edge_label("my_transform") == "__error_my_transform__"

    def test_preserves_transform_id_exactly(self) -> None:
        """No normalization — the transform_id appears verbatim."""
        assert error_edge_label("CamelCase-123") == "__error_CamelCase-123__"

    def test_empty_transform_id(self) -> None:
        """Edge case: empty string still produces valid label structure."""
        assert error_edge_label("") == "__error___"

    def test_transform_id_with_special_characters(self) -> None:
        """Labels must work with plugin names containing dots, slashes, etc."""
        assert error_edge_label("llm.batch/azure") == "__error_llm.batch/azure__"

    def test_return_type_is_str(self) -> None:
        result = error_edge_label("x")
        assert type(result) is str

    def test_importable_from_contracts_top_level(self) -> None:
        """error_edge_label must be importable from the contracts package."""
        from elspeth.contracts import error_edge_label as imported

        assert imported("test") == "__error_test__"
