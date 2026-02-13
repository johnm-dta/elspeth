"""Tests for KeywordFilter transform."""

from typing import TYPE_CHECKING
from unittest.mock import Mock

import pytest

from elspeth.plugins.config_base import PluginConfigError
from elspeth.testing import make_pipeline_row

if TYPE_CHECKING:
    from elspeth.contracts.plugin_context import PluginContext


def make_mock_context() -> "PluginContext":
    """Create a mock PluginContext for testing."""
    from elspeth.contracts.plugin_context import PluginContext

    return Mock(spec=PluginContext, run_id="test-run")


class TestKeywordFilterConfig:
    """Tests for KeywordFilterConfig validation."""

    def test_config_requires_fields(self) -> None:
        """Config must specify fields - no defaults allowed."""
        from elspeth.plugins.transforms.keyword_filter import KeywordFilterConfig

        with pytest.raises(PluginConfigError) as exc_info:
            KeywordFilterConfig.from_dict(
                {
                    "blocked_patterns": ["test"],
                    "schema": {"mode": "observed"},
                }
            )
        assert "fields" in str(exc_info.value).lower()

    def test_config_requires_blocked_patterns(self) -> None:
        """Config must specify blocked_patterns - no defaults allowed."""
        from elspeth.plugins.transforms.keyword_filter import KeywordFilterConfig

        with pytest.raises(PluginConfigError) as exc_info:
            KeywordFilterConfig.from_dict(
                {
                    "fields": ["content"],
                    "schema": {"mode": "observed"},
                }
            )
        assert "blocked_patterns" in str(exc_info.value).lower()

    def test_config_accepts_single_field(self) -> None:
        """Config accepts single field as string."""
        from elspeth.plugins.transforms.keyword_filter import KeywordFilterConfig

        cfg = KeywordFilterConfig.from_dict(
            {
                "fields": "content",
                "blocked_patterns": ["test"],
                "schema": {"mode": "observed"},
            }
        )
        assert cfg.fields == "content"

    def test_config_accepts_field_list(self) -> None:
        """Config accepts list of fields."""
        from elspeth.plugins.transforms.keyword_filter import KeywordFilterConfig

        cfg = KeywordFilterConfig.from_dict(
            {
                "fields": ["content", "subject"],
                "blocked_patterns": ["test"],
                "schema": {"mode": "observed"},
            }
        )
        assert cfg.fields == ["content", "subject"]

    def test_config_accepts_all_keyword(self) -> None:
        """Config accepts 'all' to scan all string fields."""
        from elspeth.plugins.transforms.keyword_filter import KeywordFilterConfig

        cfg = KeywordFilterConfig.from_dict(
            {
                "fields": "all",
                "blocked_patterns": ["test"],
                "schema": {"mode": "observed"},
            }
        )
        assert cfg.fields == "all"

    def test_config_validates_patterns_not_empty(self) -> None:
        """Config rejects empty blocked_patterns list."""
        from elspeth.plugins.transforms.keyword_filter import KeywordFilterConfig

        with pytest.raises(PluginConfigError) as exc_info:
            KeywordFilterConfig.from_dict(
                {
                    "fields": ["content"],
                    "blocked_patterns": [],
                    "schema": {"mode": "observed"},
                }
            )
        assert "blocked_patterns" in str(exc_info.value).lower()


class TestValidateRegexSafety:
    """Tests for _validate_regex_safety ReDoS detection."""

    def test_rejects_nested_quantifier_star_star(self) -> None:
        """(a*)* is a classic ReDoS pattern."""
        from elspeth.plugins.transforms.keyword_filter import _validate_regex_safety

        with pytest.raises(ValueError, match="nested quantifiers"):
            _validate_regex_safety("(a*)*")

    def test_rejects_nested_quantifier_plus_plus(self) -> None:
        """(a+)+ is the canonical ReDoS example."""
        from elspeth.plugins.transforms.keyword_filter import _validate_regex_safety

        with pytest.raises(ValueError, match="nested quantifiers"):
            _validate_regex_safety("(a+)+")

    def test_rejects_nested_quantifier_plus_brace(self) -> None:
        """(a+){2,} nests a quantifier inside a brace-quantified group."""
        from elspeth.plugins.transforms.keyword_filter import _validate_regex_safety

        with pytest.raises(ValueError, match="nested quantifiers"):
            _validate_regex_safety("(a+){2,}")

    def test_rejects_non_capturing_group_nested(self) -> None:
        """(?:a+)+ uses non-capturing group — still ReDoS."""
        from elspeth.plugins.transforms.keyword_filter import _validate_regex_safety

        with pytest.raises(ValueError, match="nested quantifiers"):
            _validate_regex_safety("(?:a+)+")

    def test_accepts_simple_quantifier(self) -> None:
        """a+ is a simple quantifier, not nested."""
        from elspeth.plugins.transforms.keyword_filter import _validate_regex_safety

        _validate_regex_safety("a+")  # Should not raise

    def test_accepts_alternation(self) -> None:
        """(foo|bar) is safe — no nested quantifiers."""
        from elspeth.plugins.transforms.keyword_filter import _validate_regex_safety

        _validate_regex_safety("(foo|bar)")  # Should not raise

    def test_accepts_character_class_quantifier(self) -> None:
        """[a-z]+ is a single quantifier, not nested."""
        from elspeth.plugins.transforms.keyword_filter import _validate_regex_safety

        _validate_regex_safety("[a-z]+")  # Should not raise

    def test_accepts_group_without_inner_quantifier(self) -> None:
        """(abc)+ has a quantified group but no inner quantifier."""
        from elspeth.plugins.transforms.keyword_filter import _validate_regex_safety

        _validate_regex_safety("(abc)+")  # Should not raise

    def test_rejects_pattern_exceeding_max_length(self) -> None:
        """Patterns exceeding _MAX_PATTERN_LENGTH are rejected."""
        from elspeth.plugins.transforms.keyword_filter import _validate_regex_safety

        long_pattern = "a" * 1001
        with pytest.raises(ValueError, match="exceeds maximum length"):
            _validate_regex_safety(long_pattern)

    def test_accepts_pattern_at_max_length(self) -> None:
        """Pattern at exactly _MAX_PATTERN_LENGTH is accepted."""
        from elspeth.plugins.transforms.keyword_filter import _validate_regex_safety

        pattern = "a" * 1000
        _validate_regex_safety(pattern)  # Should not raise

    def test_instantiation_rejects_redos_pattern(self) -> None:
        """KeywordFilter __init__ rejects ReDoS-prone patterns."""
        from elspeth.plugins.transforms.keyword_filter import KeywordFilter

        with pytest.raises(ValueError, match="nested quantifiers"):
            KeywordFilter(
                {
                    "fields": ["content"],
                    "blocked_patterns": ["(a+)+"],
                    "schema": {"mode": "observed"},
                }
            )


class TestKeywordFilterInstantiation:
    """Tests for KeywordFilter transform instantiation."""

    def test_transform_has_required_attributes(self) -> None:
        """Transform has all protocol-required attributes."""
        from elspeth.plugins.transforms.keyword_filter import KeywordFilter

        transform = KeywordFilter(
            {
                "fields": ["content"],
                "blocked_patterns": ["test"],
                "schema": {"mode": "observed"},
            }
        )

        assert transform.name == "keyword_filter"
        assert transform.determinism.value == "deterministic"
        assert transform.plugin_version == "1.0.0"
        assert transform.is_batch_aware is False
        assert transform.creates_tokens is False
        assert transform.input_schema is not None
        assert transform.output_schema is not None

    def test_transform_compiles_patterns_at_init(self) -> None:
        """Transform compiles regex patterns at initialization."""
        from elspeth.plugins.transforms.keyword_filter import KeywordFilter

        transform = KeywordFilter(
            {
                "fields": ["content"],
                "blocked_patterns": [r"\bpassword\b", r"(?i)secret"],
                "schema": {"mode": "observed"},
            }
        )

        # Patterns should be compiled (implementation detail, but important for perf)
        assert len(transform._compiled_patterns) == 2

    def test_transform_rejects_invalid_regex(self) -> None:
        """Transform fails at init if regex pattern is invalid."""
        import re

        from elspeth.plugins.transforms.keyword_filter import KeywordFilter

        with pytest.raises(re.error):
            KeywordFilter(
                {
                    "fields": ["content"],
                    "blocked_patterns": ["[invalid(regex"],
                    "schema": {"mode": "observed"},
                }
            )


class TestKeywordFilterProcessing:
    """Tests for KeywordFilter.process() method."""

    def test_row_without_matches_passes_through(self) -> None:
        """Rows without pattern matches pass through unchanged."""
        from elspeth.plugins.transforms.keyword_filter import KeywordFilter

        transform = KeywordFilter(
            {
                "fields": ["content"],
                "blocked_patterns": [r"\bpassword\b"],
                "schema": {"mode": "observed"},
            }
        )

        row = {"content": "Hello world", "id": 1}
        result = transform.process(make_pipeline_row(row), make_mock_context())

        assert result.status == "success"
        assert result.row is not None
        assert result.row.to_dict() == row

    def test_row_with_match_returns_error(self) -> None:
        """Rows with pattern matches return error result."""
        from elspeth.plugins.transforms.keyword_filter import KeywordFilter

        transform = KeywordFilter(
            {
                "fields": ["content"],
                "blocked_patterns": [r"\bpassword\b"],
                "schema": {"mode": "observed"},
            }
        )

        row = {"content": "My password is secret", "id": 1}
        result = transform.process(make_pipeline_row(row), make_mock_context())

        assert result.status == "error"
        assert result.reason is not None
        assert result.reason["reason"] == "blocked_content"
        assert result.reason["field"] == "content"
        assert result.reason["matched_pattern"] == r"\bpassword\b"

    def test_error_includes_positional_metadata(self) -> None:
        """Error result includes positional metadata instead of content."""
        from elspeth.plugins.transforms.keyword_filter import KeywordFilter

        transform = KeywordFilter(
            {
                "fields": ["content"],
                "blocked_patterns": [r"\bssn\b"],
                "schema": {"mode": "observed"},
            }
        )

        row = {"content": "Please provide your ssn for verification purposes"}
        result = transform.process(make_pipeline_row(row), make_mock_context())

        assert result.status == "error"
        assert result.reason is not None
        assert result.reason["match_position"] == 20
        assert result.reason["match_length"] == 3
        assert result.reason["field_length"] == len(row["content"])

    def test_scans_multiple_fields(self) -> None:
        """Transform scans all configured fields."""
        from elspeth.plugins.transforms.keyword_filter import KeywordFilter

        transform = KeywordFilter(
            {
                "fields": ["subject", "body"],
                "blocked_patterns": [r"(?i)confidential"],
                "schema": {"mode": "observed"},
            }
        )

        # Match in second field
        row = {"subject": "Hello", "body": "This is CONFIDENTIAL"}
        result = transform.process(make_pipeline_row(row), make_mock_context())

        assert result.status == "error"
        assert result.reason is not None
        assert result.reason["field"] == "body"

    def test_all_keyword_scans_string_fields(self) -> None:
        """'all' keyword scans all string-valued fields."""
        from elspeth.plugins.transforms.keyword_filter import KeywordFilter

        transform = KeywordFilter(
            {
                "fields": "all",
                "blocked_patterns": [r"secret"],
                "schema": {"mode": "observed"},
            }
        )

        row = {"name": "test", "data": "contains secret", "count": 42}
        result = transform.process(make_pipeline_row(row), make_mock_context())

        assert result.status == "error"
        assert result.reason is not None
        assert result.reason["field"] == "data"

    def test_skips_non_string_fields_when_all(self) -> None:
        """'all' mode skips non-string fields without error."""
        from elspeth.plugins.transforms.keyword_filter import KeywordFilter

        transform = KeywordFilter(
            {
                "fields": "all",
                "blocked_patterns": [r"secret"],
                "schema": {"mode": "observed"},
            }
        )

        row = {"name": "safe", "count": 42, "active": True}
        result = transform.process(make_pipeline_row(row), make_mock_context())

        assert result.status == "success"

    def test_case_sensitive_by_default(self) -> None:
        """Pattern matching is case-sensitive by default."""
        from elspeth.plugins.transforms.keyword_filter import KeywordFilter

        transform = KeywordFilter(
            {
                "fields": ["content"],
                "blocked_patterns": [r"Password"],  # Capital P
                "schema": {"mode": "observed"},
            }
        )

        row = {"content": "my password is..."}  # lowercase
        result = transform.process(make_pipeline_row(row), make_mock_context())

        assert result.status == "success"  # No match - case matters

    def test_case_insensitive_with_flag(self) -> None:
        """Regex (?i) flag enables case-insensitive matching."""
        from elspeth.plugins.transforms.keyword_filter import KeywordFilter

        transform = KeywordFilter(
            {
                "fields": ["content"],
                "blocked_patterns": [r"(?i)password"],
                "schema": {"mode": "observed"},
            }
        )

        row = {"content": "my PASSWORD is..."}
        result = transform.process(make_pipeline_row(row), make_mock_context())

        assert result.status == "error"

    def test_skips_missing_configured_field(self) -> None:
        """Transform skips fields not present in the row."""
        from elspeth.plugins.transforms.keyword_filter import KeywordFilter

        transform = KeywordFilter(
            {
                "fields": ["content", "optional_field"],
                "blocked_patterns": [r"secret"],
                "schema": {"mode": "observed"},
            }
        )

        # Row is missing "optional_field" but has "content"
        row = {"content": "safe data", "id": 1}
        result = transform.process(make_pipeline_row(row), make_mock_context())

        assert result.status == "success"

    def test_detects_pattern_in_present_field_when_other_missing(self) -> None:
        """Transform still detects patterns in fields that ARE present."""
        from elspeth.plugins.transforms.keyword_filter import KeywordFilter

        transform = KeywordFilter(
            {
                "fields": ["content", "optional_field"],
                "blocked_patterns": [r"secret"],
                "schema": {"mode": "observed"},
            }
        )

        # Row is missing "optional_field" but "content" has blocked pattern
        row = {"content": "contains secret data", "id": 1}
        result = transform.process(make_pipeline_row(row), make_mock_context())

        assert result.status == "error"
        assert result.reason is not None
        assert result.reason["field"] == "content"

    def test_blocks_when_config_uses_original_field_name(self) -> None:
        """Configured original field names resolve through PipelineRow contract."""
        from elspeth.contracts.schema_contract import FieldContract, PipelineRow, SchemaContract
        from elspeth.plugins.transforms.keyword_filter import KeywordFilter

        transform = KeywordFilter(
            {
                "fields": ["Amount USD"],
                "blocked_patterns": [r"secret"],
                "schema": {"mode": "observed"},
            }
        )

        contract = SchemaContract(
            mode="OBSERVED",
            fields=(
                FieldContract(
                    normalized_name="amount_usd",
                    original_name="Amount USD",
                    python_type=str,
                    required=False,
                    source="inferred",
                ),
            ),
            locked=True,
        )
        row = PipelineRow({"amount_usd": "contains secret value"}, contract)

        result = transform.process(row, make_mock_context())

        assert result.status == "error"
        assert result.reason is not None
        assert result.reason["reason"] == "blocked_content"
        assert result.reason["field"] == "Amount USD"
