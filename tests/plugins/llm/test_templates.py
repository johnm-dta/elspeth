# tests/plugins/llm/test_templates.py
"""Tests for Jinja2 prompt template engine."""

import pytest

from elspeth.plugins.llm.templates import PromptTemplate, TemplateError


class TestPromptTemplate:
    """Tests for PromptTemplate wrapper."""

    def test_simple_variable_substitution(self) -> None:
        """Basic variable substitution works."""
        template = PromptTemplate("Hello, {{ row.name }}!")
        result = template.render({"name": "World"})
        assert result == "Hello, World!"

    def test_template_with_loop(self) -> None:
        """Jinja2 loops work."""
        template = PromptTemplate(
            """
Analyze these entries:
{% for entry in row.entries %}
- {{ entry.name }}: {{ entry.value }}
{% endfor %}
""".strip()
        )
        result = template.render(
            {
                "entries": [
                    {"name": "A", "value": 1},
                    {"name": "B", "value": 2},
                ]
            }
        )
        assert "- A: 1" in result
        assert "- B: 2" in result

    def test_template_with_default_filter(self) -> None:
        """Jinja2 default filter works."""
        template = PromptTemplate("Focus: {{ row.focus | default('general') }}")
        assert template.render({}) == "Focus: general"
        assert template.render({"focus": "quality"}) == "Focus: quality"

    def test_template_hash_is_stable(self) -> None:
        """Same template string produces same hash."""
        t1 = PromptTemplate("Hello, {{ name }}!")
        t2 = PromptTemplate("Hello, {{ name }}!")
        assert t1.template_hash == t2.template_hash

    def test_different_templates_have_different_hashes(self) -> None:
        """Different templates have different hashes."""
        t1 = PromptTemplate("Hello, {{ name }}!")
        t2 = PromptTemplate("Goodbye, {{ name }}!")
        assert t1.template_hash != t2.template_hash

    def test_render_returns_metadata(self) -> None:
        """render() returns prompt and audit metadata."""
        template = PromptTemplate("Analyze: {{ row.text }}")
        result = template.render_with_metadata({"text": "sample"})

        assert result.prompt == "Analyze: sample"
        assert result.template_hash is not None
        assert result.variables_hash is not None
        assert result.rendered_hash is not None

    def test_undefined_variable_raises_error(self) -> None:
        """Missing required variable raises TemplateError."""
        template = PromptTemplate("Hello, {{ row.name }}!")
        with pytest.raises(TemplateError, match="name"):
            template.render({})  # No 'name' provided in row

    def test_sandboxed_prevents_dangerous_operations(self) -> None:
        """Sandboxed environment blocks dangerous operations."""
        # Attempt to access dunder attributes (blocked by SandboxedEnvironment)
        dangerous = PromptTemplate("{{ ''.__class__.__mro__ }}")
        # SecurityError is wrapped in TemplateError with "Sandbox violation" message
        with pytest.raises(TemplateError, match="Sandbox violation"):
            dangerous.render({})

    def test_rendered_prompt_includes_source_metadata(self) -> None:
        """RenderedPrompt includes template and lookup source paths."""
        template = PromptTemplate(
            "Hello, {{ row.name }}!",
            template_source="prompts/greeting.j2",
            lookup_data={"greetings": ["Hi", "Hello"]},
            lookup_source="prompts/lookups.yaml",
        )
        result = template.render_with_metadata({"name": "World"})

        assert result.template_source == "prompts/greeting.j2"
        assert result.lookup_hash is not None
        assert result.lookup_source == "prompts/lookups.yaml"

    def test_lookup_simple_access(self) -> None:
        """Templates can access lookup data."""
        template = PromptTemplate(
            "Category: {{ lookup.categories[0] }}",
            lookup_data={"categories": ["Electronics", "Clothing", "Food"]},
        )
        result = template.render({})
        assert result == "Category: Electronics"

    def test_lookup_two_dimensional(self) -> None:
        """Templates can do two-dimensional lookups: lookup.X[row.Y]."""
        template = PromptTemplate(
            "Tone: {{ lookup.tones[row.tone_id] }}",
            lookup_data={"tones": {"0": "formal", "1": "casual", "2": "technical"}},
        )
        result = template.render({"tone_id": "1"})
        assert result == "Tone: casual"

    def test_lookup_missing_key_raises_error(self) -> None:
        """Missing lookup key raises TemplateError (strict mode)."""
        template = PromptTemplate(
            "Category: {{ lookup.categories[row.cat_id] }}",
            lookup_data={"categories": {"0": "A", "1": "B"}},
        )
        with pytest.raises(TemplateError):
            template.render({"cat_id": "99"})  # No key "99"

    def test_lookup_iteration(self) -> None:
        """Templates can iterate over lookup data."""
        template = PromptTemplate(
            """Categories:
{% for cat in lookup.categories %}
- {{ cat.name }}
{% endfor %}""",
            lookup_data={
                "categories": [
                    {"name": "Electronics"},
                    {"name": "Clothing"},
                ]
            },
        )
        result = template.render({})
        assert "- Electronics" in result
        assert "- Clothing" in result

    def test_lookup_hash_is_stable(self) -> None:
        """Same lookup data produces same hash."""
        data = {"categories": ["A", "B", "C"]}
        t1 = PromptTemplate("{{ lookup.categories }}", lookup_data=data)
        t2 = PromptTemplate("{{ lookup.categories }}", lookup_data=data)
        assert t1.lookup_hash == t2.lookup_hash

    def test_no_lookup_has_none_hash(self) -> None:
        """Template without lookup data has None lookup_hash."""
        template = PromptTemplate("Hello, {{ row.name }}!")
        assert template.lookup_hash is None
        assert template.lookup_source is None

    def test_empty_lookup_has_hash(self) -> None:
        """Template with empty lookup_data={} still gets a hash.

        We distinguish None (no lookup configured) from {} (empty lookup).
        An empty lookup is still a valid configuration that should be auditable.
        Per CLAUDE.md: "No inference - if it's not recorded, it didn't happen."
        """
        template = PromptTemplate("Hello, {{ row.name }}!", lookup_data={})
        assert template.lookup_hash is not None  # Empty dict still gets hashed
        assert template.lookup_source is None  # No source file specified


class TestPromptTemplateCanonicalSafety:
    """Tests for canonicalization error handling in render_with_metadata().

    Regression tests for P2-2026-01-31-azure-canonicalization-crash:
    Row data containing NaN or Infinity should raise TemplateError
    (row-scoped failure) instead of crashing the entire pipeline.
    """

    def test_nan_in_row_raises_template_error(self) -> None:
        """NaN in row data raises TemplateError, not ValueError.

        Per Three-Tier Trust Model: row data failures should be quarantined,
        not crash the entire run. TemplateError is caught by transform plugins
        and converted to TransformResult.error().
        """
        template = PromptTemplate("Value: {{ row.value }}")
        row = {"value": float("nan")}

        with pytest.raises(TemplateError, match="Cannot compute variables hash"):
            template.render_with_metadata(row)

    def test_infinity_in_row_raises_template_error(self) -> None:
        """Infinity in row data raises TemplateError, not ValueError."""
        template = PromptTemplate("Value: {{ row.value }}")
        row = {"value": float("inf")}

        with pytest.raises(TemplateError, match="Cannot compute variables hash"):
            template.render_with_metadata(row)

    def test_negative_infinity_in_row_raises_template_error(self) -> None:
        """Negative infinity in row data raises TemplateError, not ValueError."""
        template = PromptTemplate("Value: {{ row.value }}")
        row = {"value": float("-inf")}

        with pytest.raises(TemplateError, match="Cannot compute variables hash"):
            template.render_with_metadata(row)

    def test_nan_in_nested_structure_raises_template_error(self) -> None:
        """NaN in nested dict/list raises TemplateError."""
        template = PromptTemplate("Data: {{ row.data }}")
        row = {"data": {"nested": [1, 2, float("nan"), 4]}}

        with pytest.raises(TemplateError, match="Cannot compute variables hash"):
            template.render_with_metadata(row)

    def test_normal_render_still_works(self) -> None:
        """Normal row data still renders correctly (sanity check)."""
        template = PromptTemplate("Name: {{ row.name }}, Age: {{ row.age }}")
        row = {"name": "Alice", "age": 30}

        result = template.render_with_metadata(row)

        assert result.prompt == "Name: Alice, Age: 30"
        assert result.variables_hash is not None
        assert result.template_hash is not None

    def test_render_without_metadata_unaffected(self) -> None:
        """render() (without metadata) is unaffected by NaN since it doesn't canonicalize.

        The rendering itself works - it's only the hash computation that fails.
        This confirms the bug is specifically in render_with_metadata().
        """
        template = PromptTemplate("Value: {{ row.value }}")
        row = {"value": float("nan")}

        # render() works (doesn't compute hash)
        result = template.render(row)
        assert result == "Value: nan"

        # render_with_metadata() fails (computes hash)
        with pytest.raises(TemplateError, match="Cannot compute variables hash"):
            template.render_with_metadata(row)
