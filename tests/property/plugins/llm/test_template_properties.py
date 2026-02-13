# tests/property/plugins/llm/test_template_properties.py
"""Property-based tests for PromptTemplate hash determinism and rendering invariants.

PromptTemplate is the audit-critical path for LLM prompts. Key invariants:
- template_hash is deterministic: same string → same hash
- variables_hash is deterministic: same row data → same hash (via canonical_json)
- rendered_hash changes when row data changes (content sensitivity)
- render is idempotent: same row → same output
- undefined variables raise TemplateError (StrictUndefined)
- sandbox violations raise TemplateError (SandboxedEnvironment)
- lookup_hash is None when no lookup data provided, present when provided
"""

from __future__ import annotations

import pytest
from hypothesis import assume, given, settings
from hypothesis import strategies as st

from elspeth.plugins.llm.templates import PromptTemplate, TemplateError

# =============================================================================
# Strategies
# =============================================================================

# Simple field names
field_names = st.text(min_size=1, max_size=15, alphabet="abcdefghijklmnopqrstuvwxyz_")

# Simple string values
string_values = st.text(min_size=0, max_size=50)

# Row data dicts
simple_rows = st.dictionaries(
    keys=field_names,
    values=st.one_of(st.integers(min_value=-1000, max_value=1000), string_values),
    min_size=1,
    max_size=5,
)

# Lookup data dicts
lookup_data = st.dictionaries(
    keys=field_names,
    values=string_values,
    min_size=0,
    max_size=3,
)


# =============================================================================
# Template Hash Determinism Properties
# =============================================================================


class TestTemplateHashProperties:
    """Template hash must be deterministic and content-sensitive."""

    @given(template_str=string_values)
    @settings(max_examples=100)
    def test_template_hash_deterministic(self, template_str: str) -> None:
        """Property: Same template string always produces same hash."""
        try:
            t1 = PromptTemplate(template_str)
        except TemplateError:
            assume(False)  # Skip invalid Jinja2 syntax
        t2 = PromptTemplate(template_str)
        assert t1.template_hash == t2.template_hash

    @given(template_str=string_values)
    @settings(max_examples=100)
    def test_template_hash_is_64_hex_chars(self, template_str: str) -> None:
        """Property: template_hash is always 64 hex characters (SHA-256)."""
        try:
            t = PromptTemplate(template_str)
        except TemplateError:
            assume(False)  # Skip invalid Jinja2 syntax
        assert len(t.template_hash) == 64
        assert all(c in "0123456789abcdef" for c in t.template_hash)

    def test_different_templates_different_hashes(self) -> None:
        """Property: Different template strings produce different hashes."""
        t1 = PromptTemplate("Hello {{ row.name }}")
        t2 = PromptTemplate("Goodbye {{ row.name }}")
        assert t1.template_hash != t2.template_hash

    def test_empty_template_has_valid_hash(self) -> None:
        """Property: Empty template produces a valid hash."""
        t = PromptTemplate("")
        assert len(t.template_hash) == 64


# =============================================================================
# Render Determinism Properties
# =============================================================================


class TestRenderDeterminismProperties:
    """Rendering must be idempotent for the same inputs."""

    @given(value=string_values)
    @settings(max_examples=100)
    def test_render_idempotent(self, value: str) -> None:
        """Property: Same row data always produces same rendered output."""
        t = PromptTemplate("Value: {{ row.x }}")
        row = {"x": value}
        r1 = t.render(row)
        r2 = t.render(row)
        assert r1 == r2

    @given(value=string_values)
    @settings(max_examples=100)
    def test_render_with_metadata_idempotent(self, value: str) -> None:
        """Property: render_with_metadata produces identical results for same input."""
        t = PromptTemplate("Value: {{ row.x }}")
        row = {"x": value}
        m1 = t.render_with_metadata(row)
        m2 = t.render_with_metadata(row)
        assert m1.prompt == m2.prompt
        assert m1.template_hash == m2.template_hash
        assert m1.variables_hash == m2.variables_hash
        assert m1.rendered_hash == m2.rendered_hash

    @given(value=st.integers(min_value=-1000, max_value=1000))
    @settings(max_examples=100)
    def test_variables_hash_deterministic(self, value: int) -> None:
        """Property: Same row data produces same variables_hash (canonical JSON)."""
        t = PromptTemplate("N: {{ row.n }}")
        row = {"n": value}
        m1 = t.render_with_metadata(row)
        m2 = t.render_with_metadata(row)
        assert m1.variables_hash == m2.variables_hash

    def test_different_rows_different_variables_hash(self) -> None:
        """Property: Different row data produces different variables_hash."""
        t = PromptTemplate("N: {{ row.n }}")
        m1 = t.render_with_metadata({"n": 1})
        m2 = t.render_with_metadata({"n": 2})
        assert m1.variables_hash != m2.variables_hash

    def test_different_rows_different_rendered_hash(self) -> None:
        """Property: Different row data produces different rendered_hash."""
        t = PromptTemplate("N: {{ row.n }}")
        m1 = t.render_with_metadata({"n": 1})
        m2 = t.render_with_metadata({"n": 2})
        assert m1.rendered_hash != m2.rendered_hash


# =============================================================================
# Metadata Completeness Properties
# =============================================================================


class TestMetadataCompletenessProperties:
    """render_with_metadata must produce complete audit metadata."""

    @given(value=string_values)
    @settings(max_examples=50)
    def test_all_hash_fields_present(self, value: str) -> None:
        """Property: All required hash fields are non-None strings."""
        t = PromptTemplate("V: {{ row.v }}")
        m = t.render_with_metadata({"v": value})
        assert isinstance(m.template_hash, str) and len(m.template_hash) == 64
        assert isinstance(m.variables_hash, str) and len(m.variables_hash) == 64
        assert isinstance(m.rendered_hash, str) and len(m.rendered_hash) == 64

    def test_lookup_hash_none_when_no_lookup(self) -> None:
        """Property: lookup_hash is None when no lookup_data provided."""
        t = PromptTemplate("Hello")
        m = t.render_with_metadata({"v": 1})
        assert m.lookup_hash is None

    @given(data=lookup_data)
    @settings(max_examples=50)
    def test_lookup_hash_present_when_lookup_provided(self, data: dict[str, str]) -> None:
        """Property: lookup_hash is a valid hash when lookup_data provided."""
        t = PromptTemplate("Hello", lookup_data=data)
        m = t.render_with_metadata({"v": 1})
        assert m.lookup_hash is not None
        assert len(m.lookup_hash) == 64

    @given(data=lookup_data)
    @settings(max_examples=50)
    def test_lookup_hash_deterministic(self, data: dict[str, str]) -> None:
        """Property: Same lookup_data always produces same lookup_hash."""
        t1 = PromptTemplate("Hello", lookup_data=data)
        t2 = PromptTemplate("Hello", lookup_data=data)
        assert t1.lookup_hash == t2.lookup_hash

    def test_lookup_external_mutation_after_init_does_not_change_render(self) -> None:
        """Property: external lookup mutations cannot change rendered output post-init."""
        lookup = {"k": "v1"}
        t = PromptTemplate("Value: {{ lookup.k }}", lookup_data=lookup)
        first = t.render_with_metadata({"row_id": "r1"})

        lookup["k"] = "v2"
        second = t.render_with_metadata({"row_id": "r1"})

        assert first.prompt == second.prompt == "Value: v1"
        assert first.lookup_hash == second.lookup_hash

    def test_template_source_preserved(self) -> None:
        """Property: template_source is passed through to metadata."""
        t = PromptTemplate("Hello", template_source="/path/to/template.j2")
        m = t.render_with_metadata({"v": 1})
        assert m.template_source == "/path/to/template.j2"

    def test_template_source_none_when_inline(self) -> None:
        """Property: template_source is None for inline templates."""
        t = PromptTemplate("Hello")
        m = t.render_with_metadata({"v": 1})
        assert m.template_source is None


# =============================================================================
# Error Handling Properties
# =============================================================================


class TestTemplateErrorProperties:
    """Template errors must be raised, not silently swallowed."""

    def test_undefined_variable_raises(self) -> None:
        """Property: Accessing undefined variable raises TemplateError."""
        t = PromptTemplate("{{ row.nonexistent }}")
        with pytest.raises(TemplateError, match="Undefined variable"):
            t.render({"other": "value"})

    def test_invalid_syntax_raises_at_init(self) -> None:
        """Property: Invalid Jinja2 syntax raises TemplateError at construction."""
        with pytest.raises(TemplateError, match="Invalid template syntax"):
            PromptTemplate("{% if %}")

    def test_unclosed_block_raises(self) -> None:
        """Property: Unclosed block raises TemplateError."""
        with pytest.raises(TemplateError, match="Invalid template syntax"):
            PromptTemplate("{% if True %}no end")

    def test_template_with_no_variables_renders(self) -> None:
        """Property: Template with no variables renders to literal string."""
        t = PromptTemplate("Just a plain string")
        assert t.render({}) == "Just a plain string"

    def test_nan_in_row_raises_template_error(self) -> None:
        """Property: NaN in row data raises TemplateError (canonical_json rejects NaN).

        This is defense-in-depth per CLAUDE.md: NaN/Infinity are strictly rejected
        to protect audit integrity. The hash computation via canonical_json catches
        this even though the render itself would succeed.
        """
        t = PromptTemplate("{{ row.x }}")
        with pytest.raises(TemplateError, match="Cannot compute variables hash"):
            t.render_with_metadata({"x": float("nan")})

    def test_infinity_in_row_raises_template_error(self) -> None:
        """Property: Infinity in row data raises TemplateError."""
        t = PromptTemplate("{{ row.x }}")
        with pytest.raises(TemplateError, match="Cannot compute variables hash"):
            t.render_with_metadata({"x": float("inf")})
