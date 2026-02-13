# tests/core/test_templates.py
"""Tests for Jinja2 template field extraction utility."""

import pytest


class TestExtractJinja2Fields:
    """Tests for extract_jinja2_fields function."""

    def test_simple_field_access(self) -> None:
        """Parse single field access via dot notation."""
        from elspeth.core.templates import extract_jinja2_fields

        result = extract_jinja2_fields("{{ row.name }}")
        assert result == frozenset({"name"})

    def test_multiple_fields(self) -> None:
        """Parse multiple field accesses."""
        from elspeth.core.templates import extract_jinja2_fields

        result = extract_jinja2_fields("{{ row.a }} and {{ row.b }}")
        assert result == frozenset({"a", "b"})

    def test_bracket_syntax(self) -> None:
        """Parse field access via bracket notation."""
        from elspeth.core.templates import extract_jinja2_fields

        result = extract_jinja2_fields('{{ row["field_name"] }}')
        assert result == frozenset({"field_name"})

    def test_bracket_syntax_with_special_chars(self) -> None:
        """Bracket syntax allows non-identifier field names."""
        from elspeth.core.templates import extract_jinja2_fields

        result = extract_jinja2_fields('{{ row["field-with-dashes"] }}')
        assert result == frozenset({"field-with-dashes"})

    def test_conditional_extracts_all_branches(self) -> None:
        """Conditional fields are all extracted (documents limitation)."""
        from elspeth.core.templates import extract_jinja2_fields

        template = "{% if row.active %}{{ row.value }}{% endif %}"
        result = extract_jinja2_fields(template)
        # Both fields extracted, even though 'value' is conditional
        assert result == frozenset({"active", "value"})

    def test_else_branch_extracted(self) -> None:
        """Fields from else branches are also extracted."""
        from elspeth.core.templates import extract_jinja2_fields

        template = "{% if row.a %}{{ row.b }}{% else %}{{ row.c }}{% endif %}"
        result = extract_jinja2_fields(template)
        assert result == frozenset({"a", "b", "c"})

    def test_different_namespace_ignored(self) -> None:
        """Fields from non-row namespace are ignored."""
        from elspeth.core.templates import extract_jinja2_fields

        result = extract_jinja2_fields("{{ lookup.data }}")
        assert result == frozenset()

    def test_custom_namespace(self) -> None:
        """Custom namespace can be specified."""
        from elspeth.core.templates import extract_jinja2_fields

        result = extract_jinja2_fields("{{ ctx.field }}", namespace="ctx")
        assert result == frozenset({"field"})

    def test_mixed_namespaces(self) -> None:
        """Only specified namespace is extracted."""
        from elspeth.core.templates import extract_jinja2_fields

        template = "{{ row.a }} {{ lookup.b }} {{ row.c }}"
        result = extract_jinja2_fields(template)
        assert result == frozenset({"a", "c"})

    def test_mixed_syntax(self) -> None:
        """Both dot and bracket syntax work together."""
        from elspeth.core.templates import extract_jinja2_fields

        result = extract_jinja2_fields('{{ row.a }} {{ row["b"] }}')
        assert result == frozenset({"a", "b"})

    def test_with_filter(self) -> None:
        """Filters don't affect field extraction."""
        from elspeth.core.templates import extract_jinja2_fields

        result = extract_jinja2_fields("{{ row.price | round(2) }}")
        assert result == frozenset({"price"})

    def test_with_default_filter(self) -> None:
        """Default filter doesn't affect extraction of primary field."""
        from elspeth.core.templates import extract_jinja2_fields

        result = extract_jinja2_fields("{{ row.value | default('N/A') }}")
        assert result == frozenset({"value"})

    def test_row_get_extracts_static_key(self) -> None:
        """row.get('field') extracts the string-literal key."""
        from elspeth.core.templates import extract_jinja2_fields

        result = extract_jinja2_fields("{{ row.get('status') }}")
        assert result == frozenset({"status"})

    def test_row_get_with_default_extracts_static_key(self) -> None:
        """row.get('field', default) extracts the string-literal key."""
        from elspeth.core.templates import extract_jinja2_fields

        result = extract_jinja2_fields("{{ row.get('status', 'N/A') }}")
        assert result == frozenset({"status"})

    def test_row_get_with_dynamic_key_ignored(self) -> None:
        """row.get(dynamic_key, default) is ignored (dynamic dependency)."""
        from elspeth.core.templates import extract_jinja2_fields

        result = extract_jinja2_fields("{{ row.get(key, 'N/A') }}")
        assert result == frozenset()

    def test_for_loop(self) -> None:
        """Field used in for loop is extracted."""
        from elspeth.core.templates import extract_jinja2_fields

        template = "{% for item in row.items %}{{ item.name }}{% endfor %}"
        result = extract_jinja2_fields(template)
        # Only row.items is extracted; item.name is different namespace
        assert result == frozenset({"items"})

    def test_empty_template(self) -> None:
        """Empty template returns empty set."""
        from elspeth.core.templates import extract_jinja2_fields

        result = extract_jinja2_fields("")
        assert result == frozenset()

    def test_no_row_references(self) -> None:
        """Template without row references returns empty set."""
        from elspeth.core.templates import extract_jinja2_fields

        result = extract_jinja2_fields("Hello, world!")
        assert result == frozenset()

    def test_nested_field_access_only_gets_first_level(self) -> None:
        """Only first-level field is extracted for nested access."""
        from elspeth.core.templates import extract_jinja2_fields

        # row.customer is extracted, but not the nested .address
        result = extract_jinja2_fields("{{ row.customer.address }}")
        assert result == frozenset({"customer"})

    def test_duplicate_fields_deduplicated(self) -> None:
        """Same field used multiple times appears once."""
        from elspeth.core.templates import extract_jinja2_fields

        result = extract_jinja2_fields("{{ row.id }} - {{ row.id }}")
        assert result == frozenset({"id"})

    def test_invalid_template_raises(self) -> None:
        """Invalid Jinja2 syntax raises error."""
        from jinja2 import TemplateSyntaxError

        from elspeth.core.templates import extract_jinja2_fields

        with pytest.raises(TemplateSyntaxError):
            extract_jinja2_fields("{{ row.field }")  # Missing closing brace

    def test_complex_template(self) -> None:
        """Complex template with multiple patterns."""
        from elspeth.core.templates import extract_jinja2_fields

        template = """
        Customer: {{ row.customer_name }}
        Order: {{ row.order_id }}
        {% if row.is_priority %}
        Priority: HIGH
        Amount: {{ row.amount | round(2) }}
        {% endif %}
        Notes: {{ row["special_notes"] | default("none") }}
        """
        result = extract_jinja2_fields(template)
        assert result == frozenset(
            {
                "customer_name",
                "order_id",
                "is_priority",
                "amount",
                "special_notes",
            }
        )


class TestExtractJinja2FieldsWithDetails:
    """Tests for extract_jinja2_fields_with_details function."""

    def test_basic_details(self) -> None:
        """Returns access type information."""
        from elspeth.core.templates import extract_jinja2_fields_with_details

        result = extract_jinja2_fields_with_details("{{ row.name }}")
        assert result == {"name": ["attr"]}

    def test_bracket_access_details(self) -> None:
        """Bracket access is recorded as 'item'."""
        from elspeth.core.templates import extract_jinja2_fields_with_details

        result = extract_jinja2_fields_with_details('{{ row["name"] }}')
        assert result == {"name": ["item"]}

    def test_mixed_access_details(self) -> None:
        """Same field accessed both ways shows both types."""
        from elspeth.core.templates import extract_jinja2_fields_with_details

        result = extract_jinja2_fields_with_details('{{ row.a }} {{ row["a"] }}')
        assert result == {"a": ["attr", "item"]}

    def test_multiple_same_access(self) -> None:
        """Multiple accesses of same type recorded multiple times."""
        from elspeth.core.templates import extract_jinja2_fields_with_details

        result = extract_jinja2_fields_with_details("{{ row.x }} {{ row.x }}")
        assert result == {"x": ["attr", "attr"]}

    def test_row_get_details_recorded_as_item(self) -> None:
        """row.get('field') is treated like item access for extraction details."""
        from elspeth.core.templates import extract_jinja2_fields_with_details

        result = extract_jinja2_fields_with_details("{{ row.get('status') }}")
        assert result == {"status": ["item"]}

    def test_row_get_dynamic_key_details_ignored(self) -> None:
        """row.get(dynamic_key) does not create a synthetic 'get' field detail."""
        from elspeth.core.templates import extract_jinja2_fields_with_details

        result = extract_jinja2_fields_with_details("{{ row.get(key, 'N/A') }}")
        assert result == {}
