# tests/property/core/test_templates_properties.py
"""Property-based tests for Jinja2 template field extraction.

These tests verify the properties of ELSPETH's template field discovery:

Extraction Properties:
- Attribute access (row.field) is extracted
- Item access (row["field"]) is extracted
- Namespace filtering works correctly
- Only static keys are extracted (dynamic ignored)

Return Type Properties:
- Always returns frozenset (immutable)
- Empty frozenset for no matches

Consistency Properties:
- Idempotent: same template yields same fields
- extract_jinja2_fields_with_details keys match extract_jinja2_fields

Edge Cases:
- Empty template returns empty frozenset
- Nested control structures are traversed
"""

from __future__ import annotations

import string

import pytest
from hypothesis import assume, given, settings
from hypothesis import strategies as st

from elspeth.core.templates import (
    extract_jinja2_fields,
    extract_jinja2_fields_with_details,
)

# =============================================================================
# Strategies for generating template components
# =============================================================================

# Valid Python-style field names (for row.field)
valid_field_names = st.text(
    min_size=1,
    max_size=20,
    alphabet=string.ascii_letters + "_",
).filter(lambda s: s.isidentifier())

# Field names that can include dashes/special chars (for row["field"])
bracket_field_names = st.text(
    min_size=1,
    max_size=20,
    alphabet=string.ascii_letters + string.digits + "_-",
)

# Namespace names
namespaces = st.sampled_from(["row", "data", "record", "item", "ctx"])


# =============================================================================
# Attribute Access (row.field) Property Tests
# =============================================================================


class TestAttributeAccessProperties:
    """Property tests for row.field style extraction."""

    @given(field=valid_field_names)
    @settings(max_examples=100)
    def test_single_attribute_extracted(self, field: str) -> None:
        """Property: Single attribute access is extracted."""
        template = f"{{{{ row.{field} }}}}"
        fields = extract_jinja2_fields(template)

        assert field in fields
        assert len(fields) == 1

    @given(field1=valid_field_names, field2=valid_field_names)
    @settings(max_examples=100)
    def test_multiple_attributes_extracted(self, field1: str, field2: str) -> None:
        """Property: Multiple attribute accesses are all extracted."""
        assume(field1 != field2)

        template = f"{{{{ row.{field1} }}}} and {{{{ row.{field2} }}}}"
        fields = extract_jinja2_fields(template)

        assert field1 in fields
        assert field2 in fields
        assert len(fields) == 2

    @given(field=valid_field_names)
    @settings(max_examples=50)
    def test_duplicate_attributes_deduplicated(self, field: str) -> None:
        """Property: Same field used multiple times appears once in result."""
        template = f"{{{{ row.{field} }}}} {{{{ row.{field} }}}} {{{{ row.{field} }}}}"
        fields = extract_jinja2_fields(template)

        assert field in fields
        assert len(fields) == 1


# =============================================================================
# Item Access (row["field"]) Property Tests
# =============================================================================


class TestItemAccessProperties:
    """Property tests for row["field"] style extraction."""

    @given(field=bracket_field_names)
    @settings(max_examples=100)
    def test_single_item_extracted(self, field: str) -> None:
        """Property: Single item access is extracted."""
        template = f'{{{{ row["{field}"] }}}}'
        fields = extract_jinja2_fields(template)

        assert field in fields
        assert len(fields) == 1

    @given(field=bracket_field_names)
    @settings(max_examples=50)
    def test_item_with_dashes_extracted(self, field: str) -> None:
        """Property: Field names with dashes work (can't use dot notation)."""
        # Add a dash to ensure it has one
        field_with_dash = f"prefix-{field}"
        template = f'{{{{ row["{field_with_dash}"] }}}}'
        fields = extract_jinja2_fields(template)

        assert field_with_dash in fields

    @given(field1=bracket_field_names, field2=bracket_field_names)
    @settings(max_examples=50)
    def test_mixed_access_styles_both_extracted(self, field1: str, field2: str) -> None:
        """Property: Both dot and bracket notation are extracted together."""
        assume(field1 != field2)
        # field1 uses dot notation (must be valid identifier)
        assume(field1.isidentifier())

        template = f'{{{{ row.{field1} }}}} {{{{ row["{field2}"] }}}}'
        fields = extract_jinja2_fields(template)

        assert field1 in fields
        assert field2 in fields


# =============================================================================
# Namespace Filtering Property Tests
# =============================================================================


class TestNamespaceFilteringProperties:
    """Property tests for namespace parameter behavior."""

    @given(field=valid_field_names, namespace=namespaces)
    @settings(max_examples=100)
    def test_custom_namespace_extracted(self, field: str, namespace: str) -> None:
        """Property: Fields from specified namespace are extracted."""
        template = f"{{{{ {namespace}.{field} }}}}"
        fields = extract_jinja2_fields(template, namespace=namespace)

        assert field in fields

    @given(field=valid_field_names)
    @settings(max_examples=50)
    def test_different_namespace_ignored(self, field: str) -> None:
        """Property: Fields from other namespaces are NOT extracted."""
        template = f"{{{{ other.{field} }}}}"
        fields = extract_jinja2_fields(template, namespace="row")

        assert len(fields) == 0
        assert field not in fields

    @given(field1=valid_field_names, field2=valid_field_names)
    @settings(max_examples=50)
    def test_only_target_namespace_extracted(self, field1: str, field2: str) -> None:
        """Property: Only fields from target namespace appear in result."""
        assume(field1 != field2)

        template = f"{{{{ row.{field1} }}}} {{{{ other.{field2} }}}}"
        fields = extract_jinja2_fields(template, namespace="row")

        assert field1 in fields
        assert field2 not in fields


# =============================================================================
# Return Type Property Tests
# =============================================================================


class TestReturnTypeProperties:
    """Property tests for return type invariants."""

    @given(field=valid_field_names)
    @settings(max_examples=50)
    def test_returns_frozenset(self, field: str) -> None:
        """Property: Return type is always frozenset (immutable)."""
        template = f"{{{{ row.{field} }}}}"
        fields = extract_jinja2_fields(template)

        assert isinstance(fields, frozenset)

    def test_empty_template_returns_empty_frozenset(self) -> None:
        """Property: Empty template returns empty frozenset."""
        fields = extract_jinja2_fields("")

        assert isinstance(fields, frozenset)
        assert len(fields) == 0

    def test_no_namespace_access_returns_empty_frozenset(self) -> None:
        """Property: Template without namespace access returns empty frozenset."""
        fields = extract_jinja2_fields("Hello {{ name }}!")

        assert len(fields) == 0

    @given(field=valid_field_names)
    @settings(max_examples=50)
    def test_frozenset_is_truly_immutable(self, field: str) -> None:
        """Property: Returned frozenset cannot be modified."""
        template = f"{{{{ row.{field} }}}}"
        fields = extract_jinja2_fields(template)

        with pytest.raises(AttributeError):
            fields.add("new_field")  # type: ignore[attr-defined]


# =============================================================================
# Control Structure Traversal Property Tests
# =============================================================================


class TestControlStructureProperties:
    """Property tests for nested control structure extraction."""

    @given(field=valid_field_names)
    @settings(max_examples=50)
    def test_if_block_fields_extracted(self, field: str) -> None:
        """Property: Fields in if blocks are extracted."""
        template = f"{{% if row.{field} %}}yes{{% endif %}}"
        fields = extract_jinja2_fields(template)

        assert field in fields

    @given(field1=valid_field_names, field2=valid_field_names)
    @settings(max_examples=50)
    def test_if_else_all_branches_extracted(self, field1: str, field2: str) -> None:
        """Property: Fields from all branches are extracted (conservative)."""
        assume(field1 != field2)

        template = f"{{% if row.{field1} %}}yes{{% else %}}{{{{ row.{field2} }}}}{{% endif %}}"
        fields = extract_jinja2_fields(template)

        assert field1 in fields
        assert field2 in fields

    @given(field=valid_field_names)
    @settings(max_examples=50)
    def test_for_loop_fields_extracted(self, field: str) -> None:
        """Property: Fields used in for loops are extracted."""
        template = f"{{% for x in row.{field} %}}{{{{ x }}}}{{% endfor %}}"
        fields = extract_jinja2_fields(template)

        assert field in fields


# =============================================================================
# Dynamic Key Exclusion Property Tests
# =============================================================================


class TestDynamicKeyExclusionProperties:
    """Property tests for dynamic key handling (row[variable])."""

    def test_dynamic_key_ignored(self) -> None:
        """Property: Dynamic keys (row[variable]) are NOT extracted.

        We can't know what 'key' contains at parse time, so we correctly
        ignore it. Developers must manually declare such dependencies.
        """
        template = "{{ row[key] }}"  # key is a variable, not a string literal
        fields = extract_jinja2_fields(template)

        assert len(fields) == 0

    @given(static_field=valid_field_names)
    @settings(max_examples=50)
    def test_static_keys_extracted_dynamic_ignored(self, static_field: str) -> None:
        """Property: Static keys extracted, dynamic keys ignored."""
        template = f'{{{{ row["{static_field}"] }}}} {{{{ row[dynamic_var] }}}}'
        fields = extract_jinja2_fields(template)

        assert static_field in fields
        assert len(fields) == 1  # Only the static one


# =============================================================================
# Idempotency Property Tests
# =============================================================================


class TestIdempotencyProperties:
    """Property tests for extraction determinism."""

    @given(field=valid_field_names)
    @settings(max_examples=100)
    def test_extraction_is_idempotent(self, field: str) -> None:
        """Property: Same template always yields same fields."""
        template = f"{{{{ row.{field} }}}}"

        result1 = extract_jinja2_fields(template)
        result2 = extract_jinja2_fields(template)
        result3 = extract_jinja2_fields(template)

        assert result1 == result2 == result3

    @given(fields=st.lists(valid_field_names, min_size=1, max_size=5, unique=True))
    @settings(max_examples=50)
    def test_order_independent_extraction(self, fields: list[str]) -> None:
        """Property: Field order in template doesn't affect result set."""
        # Create template with fields in original order
        template1 = " ".join(f"{{{{ row.{f} }}}}" for f in fields)

        # Create template with fields in reversed order
        template2 = " ".join(f"{{{{ row.{f} }}}}" for f in reversed(fields))

        # Create template with fields in rotated order
        rotated = fields[1:] + fields[:1] if len(fields) > 1 else fields
        template3 = " ".join(f"{{{{ row.{f} }}}}" for f in rotated)

        result1 = extract_jinja2_fields(template1)
        result2 = extract_jinja2_fields(template2)
        result3 = extract_jinja2_fields(template3)

        assert result1 == result2 == result3


# =============================================================================
# extract_jinja2_fields_with_details Consistency Property Tests
# =============================================================================


class TestWithDetailsConsistencyProperties:
    """Property tests for extract_jinja2_fields_with_details consistency."""

    @given(field=valid_field_names)
    @settings(max_examples=100)
    def test_details_keys_match_simple_extraction(self, field: str) -> None:
        """Property: with_details keys exactly match simple extraction result."""
        template = f"{{{{ row.{field} }}}}"

        simple = extract_jinja2_fields(template)
        details = extract_jinja2_fields_with_details(template)

        assert set(details.keys()) == simple

    @given(field=bracket_field_names)
    @settings(max_examples=50)
    def test_attr_access_labeled_as_attr(self, field: str) -> None:
        """Property: Attribute access (row.field) labeled as 'attr'."""
        assume(field.isidentifier())

        template = f"{{{{ row.{field} }}}}"
        details = extract_jinja2_fields_with_details(template)

        assert field in details
        assert "attr" in details[field]

    @given(field=bracket_field_names)
    @settings(max_examples=50)
    def test_item_access_labeled_as_item(self, field: str) -> None:
        """Property: Item access (row["field"]) labeled as 'item'."""
        template = f'{{{{ row["{field}"] }}}}'
        details = extract_jinja2_fields_with_details(template)

        assert field in details
        assert "item" in details[field]

    @given(field=valid_field_names)
    @settings(max_examples=50)
    def test_mixed_access_records_both(self, field: str) -> None:
        """Property: Same field accessed both ways records both."""
        template = f'{{{{ row.{field} }}}} {{{{ row["{field}"] }}}}'
        details = extract_jinja2_fields_with_details(template)

        assert field in details
        assert "attr" in details[field]
        assert "item" in details[field]


# =============================================================================
# Error Handling Property Tests
# =============================================================================


class TestErrorHandlingProperties:
    """Property tests for error cases."""

    def test_malformed_template_raises(self) -> None:
        """Property: Malformed Jinja2 raises TemplateSyntaxError."""
        from jinja2 import TemplateSyntaxError

        with pytest.raises(TemplateSyntaxError):
            extract_jinja2_fields("{{ row.field")  # Unclosed

    def test_unclosed_block_raises(self) -> None:
        """Property: Unclosed blocks raise TemplateSyntaxError."""
        from jinja2 import TemplateSyntaxError

        with pytest.raises(TemplateSyntaxError):
            extract_jinja2_fields("{% if row.x %}yes")  # Missing endif
