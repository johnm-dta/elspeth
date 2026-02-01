# tests/unit/plugins/llm/test_metadata_fields.py
"""Unit tests for LLM metadata field helper functions."""

import pytest

from elspeth.plugins.llm import (
    LLM_AUDIT_SUFFIXES,
    LLM_GUARANTEED_SUFFIXES,
    get_llm_audit_fields,
    get_llm_guaranteed_fields,
)


class TestLLMMetadataFieldHelpers:
    """Tests for LLM metadata field helper functions."""

    def test_guaranteed_suffixes_count(self):
        """Verify expected number of guaranteed suffixes."""
        assert len(LLM_GUARANTEED_SUFFIXES) == 3
        assert "" in LLM_GUARANTEED_SUFFIXES
        assert "_usage" in LLM_GUARANTEED_SUFFIXES
        assert "_model" in LLM_GUARANTEED_SUFFIXES

    def test_audit_suffixes_count(self):
        """Verify expected number of audit suffixes."""
        assert len(LLM_AUDIT_SUFFIXES) == 6
        assert "_template_hash" in LLM_AUDIT_SUFFIXES
        assert "_variables_hash" in LLM_AUDIT_SUFFIXES

    def test_get_llm_guaranteed_fields(self):
        """Verify guaranteed field name generation."""
        fields = get_llm_guaranteed_fields("llm_response")
        assert fields == ("llm_response", "llm_response_usage", "llm_response_model")

    def test_get_llm_audit_fields(self):
        """Verify audit field name generation."""
        fields = get_llm_audit_fields("result")
        assert "result_template_hash" in fields
        assert "result_variables_hash" in fields
        assert "result_template_source" in fields
        assert "result_lookup_hash" in fields
        assert "result_lookup_source" in fields
        assert "result_system_prompt_source" in fields
        assert len(fields) == 6

    def test_empty_response_field_raises(self):
        """Empty response_field should raise ValueError."""
        with pytest.raises(ValueError, match="response_field cannot be empty or whitespace-only"):
            get_llm_guaranteed_fields("")

        with pytest.raises(ValueError, match="response_field cannot be empty or whitespace-only"):
            get_llm_audit_fields("")

    def test_whitespace_response_field_raises(self):
        """Whitespace-only response_field should raise ValueError."""
        with pytest.raises(ValueError, match="response_field cannot be empty or whitespace-only"):
            get_llm_guaranteed_fields("   ")

        with pytest.raises(ValueError, match="response_field cannot be empty or whitespace-only"):
            get_llm_audit_fields("   ")

    def test_custom_response_field(self):
        """Custom response field names work correctly."""
        guaranteed = get_llm_guaranteed_fields("custom_output")
        assert "custom_output" in guaranteed
        assert "custom_output_usage" in guaranteed

        audit = get_llm_audit_fields("custom_output")
        assert "custom_output_template_hash" in audit
