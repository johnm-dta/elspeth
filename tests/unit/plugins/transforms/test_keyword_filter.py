"""Tests for KeywordFilter transform."""

import pytest

from elspeth.plugins.infrastructure.config_base import PluginConfigError
from elspeth.testing import make_pipeline_row
from tests.fixtures.factories import make_context


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

    def test_rejects_nested_quantifier_brace_plus(self) -> None:
        """(a{2,})+ must be rejected — brace quantifiers are still nested repetition."""
        from elspeth.plugins.transforms.keyword_filter import _validate_regex_safety

        with pytest.raises(ValueError, match="nested quantifiers"):
            _validate_regex_safety("(a{2,})+")

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
        """KeywordFilter config validation rejects ReDoS-prone patterns."""
        from elspeth.plugins.infrastructure.config_base import PluginConfigError
        from elspeth.plugins.transforms.keyword_filter import KeywordFilter

        with pytest.raises(PluginConfigError, match="nested quantifiers"):
            KeywordFilter(
                {
                    "fields": ["content"],
                    "blocked_patterns": ["(a+)+"],
                    "schema": {"mode": "observed"},
                }
            )

    @pytest.mark.parametrize("pattern", [r"(\+)+", r"(\*)+", r"(a\+)+"])
    def test_instantiation_accepts_escaped_literal_quantifier_patterns(self, pattern: str) -> None:
        """Escaped literal metacharacters are safe and must pass config validation."""
        from elspeth.plugins.transforms.keyword_filter import KeywordFilter

        transform = KeywordFilter(
            {
                "fields": ["content"],
                "blocked_patterns": [pattern],
                "schema": {"mode": "observed"},
            }
        )

        assert [compiled.pattern for _, compiled in transform._compiled_patterns] == [pattern]


class TestKeywordFilterInstantiation:
    """Tests for KeywordFilter transform instantiation."""

    def test_transform_has_required_schema_attributes(self) -> None:
        """Transform exposes input and output schemas."""
        from elspeth.plugins.transforms.keyword_filter import KeywordFilter

        transform = KeywordFilter(
            {
                "fields": ["content"],
                "blocked_patterns": ["test"],
                "schema": {"mode": "observed"},
            }
        )

        assert transform.input_schema is not None
        assert transform.output_schema is not None

    def test_transform_is_deterministic(self) -> None:
        """KeywordFilter declares deterministic since it's pure regex matching."""
        from elspeth.contracts.enums import Determinism
        from elspeth.plugins.transforms.keyword_filter import KeywordFilter

        transform = KeywordFilter(
            {
                "fields": ["content"],
                "blocked_patterns": ["test"],
                "schema": {"mode": "observed"},
            }
        )

        assert transform.determinism is Determinism.DETERMINISTIC

    def test_transform_compiles_patterns_at_init(self) -> None:
        """Patterns are compiled at init — matching works immediately without lazy init."""
        from elspeth.contracts.schema_contract import SchemaContract
        from elspeth.plugins.transforms.keyword_filter import KeywordFilter
        from elspeth.testing import make_field, make_row

        transform = KeywordFilter(
            {
                "fields": ["content"],
                "blocked_patterns": [r"\bpassword\b", r"(?i)secret"],
                "schema": {"mode": "observed"},
            }
        )

        # Prove compilation happened: a matching row gets blocked
        fields = tuple(make_field(k, object, original_name=k, required=False, source="inferred") for k in ["content"])
        contract = SchemaContract(mode="OBSERVED", fields=fields, locked=True)
        row = make_row({"content": "my password is 123"}, contract=contract)
        ctx = make_context()
        result = transform.process(row, ctx)
        assert result.status == "error"

    def test_transform_rejects_invalid_regex(self) -> None:
        """Transform fails at config validation if regex pattern is invalid."""
        from elspeth.plugins.infrastructure.config_base import PluginConfigError
        from elspeth.plugins.transforms.keyword_filter import KeywordFilter

        with pytest.raises(PluginConfigError, match="not a valid regex"):
            KeywordFilter(
                {
                    "fields": ["content"],
                    "blocked_patterns": ["[invalid(regex"],
                    "schema": {"mode": "observed"},
                }
            )

    def test_transform_initializes_schema_contract_state_for_pass_through_checks(self) -> None:
        """Pass-through KeywordFilter instances must expose a typed output schema contract."""
        from elspeth.engine.executors.schema_config_mode import SchemaConfigModeContract
        from elspeth.plugins.transforms.keyword_filter import KeywordFilter

        transform = KeywordFilter(
            {
                "fields": ["content"],
                "blocked_patterns": [r"\bblocked\b"],
                "schema": {"mode": "fixed", "fields": ["id: int", "content: str"]},
            }
        )

        assert transform._schema_config is not None
        assert transform._output_schema_config is not None
        assert sorted(transform._output_schema_config.get_effective_guaranteed_fields()) == ["content", "id"]
        assert SchemaConfigModeContract().applies_to(transform) is True


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
        result = transform.process(make_pipeline_row(row), make_context())

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
        result = transform.process(make_pipeline_row(row), make_context())

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
        result = transform.process(make_pipeline_row(row), make_context())

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
        result = transform.process(make_pipeline_row(row), make_context())

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
        result = transform.process(make_pipeline_row(row), make_context())

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
        result = transform.process(make_pipeline_row(row), make_context())

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
        result = transform.process(make_pipeline_row(row), make_context())

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
        result = transform.process(make_pipeline_row(row), make_context())

        assert result.status == "error"

    def test_missing_named_field_fails_closed(self) -> None:
        """Named field missing from row must fail closed — security check incomplete."""
        from elspeth.plugins.transforms.keyword_filter import KeywordFilter

        transform = KeywordFilter(
            {
                "fields": ["content", "optional_field"],
                "blocked_patterns": [r"secret"],
                "schema": {"mode": "observed"},
            }
        )

        row = {"content": "safe data", "id": 1}
        result = transform.process(make_pipeline_row(row), make_context())

        assert result.status == "error"
        assert result.reason is not None
        assert result.reason["reason"] == "missing_scan_field"
        assert result.reason["field"] == "optional_field"

    def test_missing_named_field_not_retryable(self) -> None:
        """Missing named field error is not retryable — data issue, not transient."""
        from elspeth.plugins.transforms.keyword_filter import KeywordFilter

        transform = KeywordFilter(
            {
                "fields": ["missing_field"],
                "blocked_patterns": [r"secret"],
                "schema": {"mode": "observed"},
            }
        )

        row = {"content": "safe data"}
        result = transform.process(make_pipeline_row(row), make_context())

        assert result.status == "error"
        assert result.retryable is False

    def test_missing_field_in_all_mode_passes(self) -> None:
        """'all' mode skips missing fields — only scans present string fields."""
        from elspeth.plugins.transforms.keyword_filter import KeywordFilter

        transform = KeywordFilter(
            {
                "fields": "all",
                "blocked_patterns": [r"secret"],
                "schema": {"mode": "observed"},
            }
        )

        row = {"content": "safe data", "id": 1}
        result = transform.process(make_pipeline_row(row), make_context())

        assert result.status == "success"

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

        result = transform.process(row, make_context())

        assert result.status == "error"
        assert result.reason is not None
        assert result.reason["reason"] == "blocked_content"
        assert result.reason["field"] == "Amount USD"


class TestKeywordFilterFieldsValidation:
    """Regression: Bug 4 — Empty fields config makes keyword filter a no-op.

    A keyword filter with fields=[] or fields="" scans nothing, silently
    passing all content through. For a security transform, this is a
    fail-OPEN vulnerability. The fix adds a field_validator that rejects
    empty strings, empty lists, and lists containing empty strings.
    """

    def test_empty_list_rejected(self) -> None:
        """fields=[] must raise ValidationError."""
        from elspeth.plugins.transforms.keyword_filter import KeywordFilterConfig

        with pytest.raises(PluginConfigError) as exc_info:
            KeywordFilterConfig.from_dict(
                {
                    "fields": [],
                    "blocked_patterns": ["test"],
                    "schema": {"mode": "observed"},
                }
            )
        assert "fields" in str(exc_info.value).lower()

    def test_empty_string_rejected(self) -> None:
        """fields="" must raise ValidationError."""
        from elspeth.plugins.transforms.keyword_filter import KeywordFilterConfig

        with pytest.raises(PluginConfigError) as exc_info:
            KeywordFilterConfig.from_dict(
                {
                    "fields": "",
                    "blocked_patterns": ["test"],
                    "schema": {"mode": "observed"},
                }
            )
        assert "fields" in str(exc_info.value).lower()

    def test_whitespace_only_string_rejected(self) -> None:
        """fields="  " must raise ValidationError (whitespace-only)."""
        from elspeth.plugins.transforms.keyword_filter import KeywordFilterConfig

        with pytest.raises(PluginConfigError) as exc_info:
            KeywordFilterConfig.from_dict(
                {
                    "fields": "   ",
                    "blocked_patterns": ["test"],
                    "schema": {"mode": "observed"},
                }
            )
        assert "fields" in str(exc_info.value).lower()

    def test_list_containing_empty_string_rejected(self) -> None:
        """fields=[""] must raise ValidationError."""
        from elspeth.plugins.transforms.keyword_filter import KeywordFilterConfig

        with pytest.raises(PluginConfigError) as exc_info:
            KeywordFilterConfig.from_dict(
                {
                    "fields": [""],
                    "blocked_patterns": ["test"],
                    "schema": {"mode": "observed"},
                }
            )
        assert "fields" in str(exc_info.value).lower()

    def test_list_with_empty_string_among_valid_rejected(self) -> None:
        """fields=["valid", ""] must raise ValidationError."""
        from elspeth.plugins.transforms.keyword_filter import KeywordFilterConfig

        with pytest.raises(PluginConfigError) as exc_info:
            KeywordFilterConfig.from_dict(
                {
                    "fields": ["valid", ""],
                    "blocked_patterns": ["test"],
                    "schema": {"mode": "observed"},
                }
            )
        assert "fields" in str(exc_info.value).lower()

    def test_valid_single_field_accepted(self) -> None:
        """fields=["valid"] must be accepted."""
        from elspeth.plugins.transforms.keyword_filter import KeywordFilterConfig

        cfg = KeywordFilterConfig.from_dict(
            {
                "fields": ["valid"],
                "blocked_patterns": ["test"],
                "schema": {"mode": "observed"},
            }
        )
        assert cfg.fields == ["valid"]


class TestKeywordFilterBlockedPatternsValidation:
    """Regression: Bug 4 (supplement) — Empty blocked_patterns entries accepted.

    The blocked_patterns validator was strengthened to reject individual empty
    strings within the list. An empty regex pattern matches everything, which
    would block all content indiscriminately — a misconfiguration that should
    fail loud at config time.
    """

    def test_empty_list_rejected(self) -> None:
        """blocked_patterns=[] must raise ValidationError."""
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

    def test_empty_string_pattern_rejected(self) -> None:
        """blocked_patterns=[""] must raise ValidationError.

        An empty regex matches everything — this is almost certainly a
        misconfiguration. It would block every row, regardless of content.
        """
        from elspeth.plugins.transforms.keyword_filter import KeywordFilterConfig

        with pytest.raises(PluginConfigError) as exc_info:
            KeywordFilterConfig.from_dict(
                {
                    "fields": ["content"],
                    "blocked_patterns": [""],
                    "schema": {"mode": "observed"},
                }
            )
        err_str = str(exc_info.value).lower()
        assert "blocked_patterns" in err_str or "empty" in err_str

    def test_empty_string_among_valid_patterns_rejected(self) -> None:
        """blocked_patterns=["valid", ""] must reject the empty entry."""
        from elspeth.plugins.transforms.keyword_filter import KeywordFilterConfig

        with pytest.raises(PluginConfigError) as exc_info:
            KeywordFilterConfig.from_dict(
                {
                    "fields": ["content"],
                    "blocked_patterns": [r"\bpassword\b", ""],
                    "schema": {"mode": "observed"},
                }
            )
        err_str = str(exc_info.value).lower()
        assert "blocked_patterns" in err_str or "empty" in err_str

    def test_valid_patterns_accepted(self) -> None:
        """Non-empty blocked_patterns must be accepted."""
        from elspeth.plugins.transforms.keyword_filter import KeywordFilterConfig

        cfg = KeywordFilterConfig.from_dict(
            {
                "fields": ["content"],
                "blocked_patterns": [r"\bpassword\b", r"(?i)secret"],
                "schema": {"mode": "observed"},
            }
        )
        assert len(cfg.blocked_patterns) == 2


class TestKeywordFilterNonStringFailClosed:
    """T16: KeywordFilter must fail-closed when configured fields have non-string values.

    A security filter that silently passes non-string values is fail-OPEN.
    When the operator explicitly names a field to scan and it contains a
    non-string value, the row must be quarantined (error result), not
    silently passed through unchecked.

    The 'all' mode is exempt — it dynamically selects string-valued fields,
    so non-string fields are correctly excluded by design.
    """

    def test_non_string_value_in_named_field_returns_error(self) -> None:
        """Integer in a configured field must return error, not pass through."""
        from elspeth.plugins.transforms.keyword_filter import KeywordFilter

        transform = KeywordFilter(
            {
                "fields": ["amount"],
                "blocked_patterns": [r"secret"],
                "schema": {"mode": "observed"},
            }
        )

        row = {"amount": 42, "content": "safe text"}
        result = transform.process(make_pipeline_row(row), make_context())

        assert result.status == "error", "Non-string value in named field must fail-closed"
        assert result.reason is not None
        assert result.reason["reason"] == "non_string_field"
        assert result.reason["field"] == "amount"

    def test_non_string_value_reports_actual_type(self) -> None:
        """Error reason must include the actual type for diagnostics."""
        from elspeth.plugins.transforms.keyword_filter import KeywordFilter

        transform = KeywordFilter(
            {
                "fields": ["score"],
                "blocked_patterns": [r"test"],
                "schema": {"mode": "observed"},
            }
        )

        row = {"score": 3.14}
        result = transform.process(make_pipeline_row(row), make_context())

        assert result.status == "error"
        assert result.reason is not None
        assert result.reason["actual_type"] == "float"

    def test_bool_value_in_named_field_returns_error(self) -> None:
        """Boolean in a configured field must fail-closed."""
        from elspeth.plugins.transforms.keyword_filter import KeywordFilter

        transform = KeywordFilter(
            {
                "fields": ["active"],
                "blocked_patterns": [r"secret"],
                "schema": {"mode": "observed"},
            }
        )

        row = {"active": True}
        result = transform.process(make_pipeline_row(row), make_context())

        assert result.status == "error"
        assert result.reason is not None
        assert result.reason["reason"] == "non_string_field"

    def test_none_value_in_named_field_returns_error(self) -> None:
        """None in a configured field must fail-closed."""
        from elspeth.plugins.transforms.keyword_filter import KeywordFilter

        transform = KeywordFilter(
            {
                "fields": ["content"],
                "blocked_patterns": [r"secret"],
                "schema": {"mode": "observed"},
            }
        )

        row = {"content": None}
        result = transform.process(make_pipeline_row(row), make_context())

        assert result.status == "error"
        assert result.reason is not None
        assert result.reason["reason"] == "non_string_field"

    def test_list_value_in_named_field_returns_error(self) -> None:
        """List in a configured field must fail-closed."""
        from elspeth.plugins.transforms.keyword_filter import KeywordFilter

        transform = KeywordFilter(
            {
                "fields": ["tags"],
                "blocked_patterns": [r"secret"],
                "schema": {"mode": "observed"},
            }
        )

        row = {"tags": ["a", "b"]}
        result = transform.process(make_pipeline_row(row), make_context())

        assert result.status == "error"
        assert result.reason is not None
        assert result.reason["reason"] == "non_string_field"

    def test_non_string_not_retryable(self) -> None:
        """Non-string field error must not be retryable (data issue, not transient)."""
        from elspeth.plugins.transforms.keyword_filter import KeywordFilter

        transform = KeywordFilter(
            {
                "fields": ["count"],
                "blocked_patterns": [r"test"],
                "schema": {"mode": "observed"},
            }
        )

        row = {"count": 99}
        result = transform.process(make_pipeline_row(row), make_context())

        assert result.status == "error"
        assert result.retryable is False

    def test_mixed_fields_string_checked_non_string_errors(self) -> None:
        """When one named field is non-string, error before checking string fields.

        The filter should detect non-string values in named fields and error
        immediately, even if other fields contain scannable strings.
        """
        from elspeth.plugins.transforms.keyword_filter import KeywordFilter

        transform = KeywordFilter(
            {
                "fields": ["amount", "content"],
                "blocked_patterns": [r"safe_pattern_no_match"],
                "schema": {"mode": "observed"},
            }
        )

        row = {"amount": 42, "content": "safe text"}
        result = transform.process(make_pipeline_row(row), make_context())

        assert result.status == "error", "Must error on non-string 'amount' even though 'content' is scannable"
        assert result.reason is not None
        assert result.reason["field"] == "amount"

    def test_all_mode_still_skips_non_string_fields(self) -> None:
        """'all' mode must continue to skip non-string fields (no regression).

        The 'all' keyword dynamically selects string-valued fields, so
        non-string fields are correctly excluded by design. This must
        not change with the fail-closed fix for named fields.
        """
        from elspeth.plugins.transforms.keyword_filter import KeywordFilter

        transform = KeywordFilter(
            {
                "fields": "all",
                "blocked_patterns": [r"secret"],
                "schema": {"mode": "observed"},
            }
        )

        # Only non-string fields — should pass through, not error
        row = {"count": 42, "active": True, "score": 3.14}
        result = transform.process(make_pipeline_row(row), make_context())

        assert result.status == "success", "'all' mode must still skip non-string fields"

    def test_string_field_with_match_still_blocks(self) -> None:
        """String fields with pattern matches must still be blocked (no regression)."""
        from elspeth.plugins.transforms.keyword_filter import KeywordFilter

        transform = KeywordFilter(
            {
                "fields": ["content"],
                "blocked_patterns": [r"secret"],
                "schema": {"mode": "observed"},
            }
        )

        row = {"content": "contains secret data"}
        result = transform.process(make_pipeline_row(row), make_context())

        assert result.status == "error"
        assert result.reason is not None
        assert result.reason["reason"] == "blocked_content"
