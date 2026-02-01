"""Tests for identifier validation utilities."""

from __future__ import annotations

import pytest


class TestValidateFieldNames:
    """Tests for validate_field_names function."""

    def test_valid_identifiers_pass(self) -> None:
        """Valid identifiers pass validation."""
        from elspeth.core.identifiers import validate_field_names

        # Should not raise
        validate_field_names(["user_id", "amount", "date"], "test_context")

    def test_invalid_identifier_raises(self) -> None:
        """Invalid identifier raises with context."""
        from elspeth.core.identifiers import validate_field_names

        with pytest.raises(ValueError, match=r"valid.*identifier") as exc_info:
            validate_field_names(["valid", "123_invalid"], "columns")

        assert "columns[1]" in str(exc_info.value)
        assert "123_invalid" in str(exc_info.value)

    def test_python_keyword_raises(self) -> None:
        """Python keyword raises with context."""
        from elspeth.core.identifiers import validate_field_names

        with pytest.raises(ValueError, match="Python keyword") as exc_info:
            validate_field_names(["id", "class", "name"], "field_mapping values")

        assert "field_mapping values[1]" in str(exc_info.value)
        assert "class" in str(exc_info.value)

    def test_duplicate_raises(self) -> None:
        """Duplicate field name raises."""
        from elspeth.core.identifiers import validate_field_names

        with pytest.raises(ValueError, match=r"[Dd]uplicate") as exc_info:
            validate_field_names(["id", "name", "id"], "columns")

        assert "id" in str(exc_info.value)

    def test_empty_list_passes(self) -> None:
        """Empty list passes validation."""
        from elspeth.core.identifiers import validate_field_names

        # Should not raise
        validate_field_names([], "test")
