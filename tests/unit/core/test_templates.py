# tests/core/test_templates.py
"""Tests for Jinja2 template field extraction utility."""

import ast
import inspect
import textwrap

import pytest


class TestExtractJinja2Fields:
    """Tests for extract_jinja2_fields function."""

    def test_field_extraction_context_checks_api_alias_presence_explicitly(self) -> None:
        """api_aliases is a local accumulator, so missing entries are internal state."""
        from elspeth.core.templates import _field_extraction_context

        tree = ast.parse(textwrap.dedent(inspect.getsource(_field_extraction_context)))
        hidden_absence_checks = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "get"
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "api_aliases"
        ]

        assert hidden_absence_checks == []

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
        """Field-only extraction excludes row.get(dynamic_key, default)."""
        from elspeth.core.templates import extract_jinja2_fields

        result = extract_jinja2_fields("{{ row.get(key, 'N/A') }}")
        assert result == frozenset()

    def test_dynamic_item_access_reported_by_usage(self) -> None:
        """Structured usage reports dynamic row[expr] access separately."""
        from elspeth.core.templates import extract_jinja2_field_usage

        result = extract_jinja2_field_usage('{% set k = "ssn" %}{{ row[k] }}')

        assert result.fields == frozenset()
        assert result.dynamic_accesses == ("item",)
        assert result.has_dynamic_access is True

    def test_dynamic_row_get_reported_by_usage(self) -> None:
        """Structured usage reports dynamic row.get(expr) access separately."""
        from elspeth.core.templates import extract_jinja2_field_usage

        result = extract_jinja2_field_usage("{{ row.get(key, 'N/A') }}")

        assert result.fields == frozenset()
        assert result.dynamic_accesses == ("get",)
        assert result.has_dynamic_access is True

    def test_dynamic_attr_filter_reported_by_usage(self) -> None:
        """row|attr(expr) is a dynamic row-field read."""
        from elspeth.core.templates import extract_jinja2_field_usage

        result = extract_jinja2_field_usage("{{ row | attr(row.selector) }}")

        assert result.fields == frozenset({"selector"})
        assert result.dynamic_accesses == ("attr",)
        assert result.has_dynamic_access is True

    def test_dynamic_map_attribute_filter_reported_by_usage(self) -> None:
        """row|map(attribute=expr) is an attribute-resolving dynamic access."""
        from elspeth.core.templates import extract_jinja2_field_usage

        result = extract_jinja2_field_usage("{{ row | map(attribute=field_name) | list }}")

        assert result.fields == frozenset()
        assert result.dynamic_accesses == ("map(attribute)",)
        assert result.has_dynamic_access is True

    @pytest.mark.parametrize(
        "template",
        (
            "{{ rows | map(attribute=row.selector) | list }}",
            "{{ rows | map(attribute=row['selector']) | list }}",
            "{{ rows | map(attribute=row.get('selector')) | list }}",
        ),
    )
    def test_row_derived_map_attribute_filter_reported_by_usage(self, template: str) -> None:
        """map(attribute=row.*) is dynamic even when the mapped sequence is not row."""
        from elspeth.core.templates import extract_jinja2_field_usage

        result = extract_jinja2_field_usage(template)

        assert result.fields == frozenset({"selector"})
        assert result.dynamic_accesses == ("map(attribute)",)
        assert result.has_dynamic_access is True

    def test_attr_filter_pipeline_row_api_name_reported_by_usage(self) -> None:
        """row|attr('get') exposes a dynamic row-field reader and must fail closed."""
        from elspeth.core.templates import extract_jinja2_field_usage

        result = extract_jinja2_field_usage("{{ (row | attr('get'))(row.selector) }}")

        assert result.fields == frozenset({"selector"})
        assert result.dynamic_accesses == ("attr",)
        assert result.has_dynamic_access is True

    def test_row_alias_derived_map_attribute_filter_reported_by_usage(self) -> None:
        """A row alias inside map(attribute=...) is still dynamic row access."""
        from elspeth.core.templates import extract_jinja2_field_usage

        result = extract_jinja2_field_usage("{% set r = row %}{{ rows | map(attribute=r.selector) | list }}")

        assert result.fields == frozenset({"selector"})
        assert result.dynamic_accesses == ("map(attribute)",)
        assert result.has_dynamic_access is True

    def test_row_alias_attr_filter_pipeline_row_api_name_reported_by_usage(self) -> None:
        """A row alias can expose row.get(expr) through attr and must fail closed."""
        from elspeth.core.templates import extract_jinja2_field_usage

        result = extract_jinja2_field_usage("{% set r = row %}{{ (r | attr('get'))(r.selector) }}")

        assert result.fields == frozenset({"selector"})
        assert result.dynamic_accesses == ("attr",)
        assert result.has_dynamic_access is True

    def test_with_row_alias_derived_map_attribute_filter_reported_by_usage(self) -> None:
        """A with-block row alias inside map(attribute=...) is dynamic row access."""
        from elspeth.core.templates import extract_jinja2_field_usage

        result = extract_jinja2_field_usage("{% with r = row %}{{ lookup.rows | map(attribute=r.selector) | list }}{% endwith %}")

        assert result.fields == frozenset({"selector"})
        assert result.dynamic_accesses == ("map(attribute)",)
        assert result.has_dynamic_access is True

    def test_method_alias_dynamic_row_get_reported_by_usage(self) -> None:
        """Aliasing row.get must not hide a dynamic row-field read."""
        from elspeth.core.templates import extract_jinja2_field_usage

        result = extract_jinja2_field_usage("{% set g = row.get %}{{ g(row.selector) }}")

        assert result.fields == frozenset({"selector"})
        assert result.dynamic_accesses == ("get",)
        assert result.has_dynamic_access is True

    @pytest.mark.parametrize(
        "template",
        (
            "{% set d = {'g': row.get} %}{{ d['g']('secret') }}",
            "{% set ns = namespace(g=row.get) %}{{ ns.g('secret') }}",
            "{% set g = row.get %}{% set d = {'g': g} %}{{ d['g']('secret') }}",
        ),
    )
    def test_container_carried_row_get_alias_reported_by_usage(self, template: str) -> None:
        """Dict and namespace carriers cannot hide row.get aliases."""
        from elspeth.core.templates import extract_jinja2_field_usage

        result = extract_jinja2_field_usage(template)

        assert result.fields == frozenset()
        assert result.dynamic_accesses == ("get",)
        assert result.has_dynamic_access is True

    def test_container_carried_row_get_alias_dynamic_key_reported_by_usage(self) -> None:
        """Carrier-held row.get with row-derived keys remains dynamic."""
        from elspeth.core.templates import extract_jinja2_field_usage

        result = extract_jinja2_field_usage("{% set ns = namespace(g=row.get) %}{{ ns.g(row.selector) }}")

        assert result.fields == frozenset({"selector"})
        assert result.dynamic_accesses == ("get",)
        assert result.has_dynamic_access is True

    @pytest.mark.parametrize(
        "template",
        (
            "{% set ns = namespace() %}{% set ns.g = row.get %}{{ ns.g('secret') }}",
            "{% set d = {'inner': {'g': row.get}} %}{{ d['inner']['g']('secret') }}",
        ),
    )
    def test_nested_or_assigned_carried_row_get_alias_reported_by_usage(self, template: str) -> None:
        """Namespace assignment and nested carriers cannot hide row.get aliases."""
        from elspeth.core.templates import extract_jinja2_field_usage

        result = extract_jinja2_field_usage(template)

        assert result.fields == frozenset()
        assert result.dynamic_accesses == ("get",)
        assert result.has_dynamic_access is True

    def test_namespace_assigned_row_alias_dynamic_get_reported_by_usage(self) -> None:
        """Namespace attribute assignment can carry row itself."""
        from elspeth.core.templates import extract_jinja2_field_usage

        result = extract_jinja2_field_usage("{% set ns = namespace() %}{% set ns.r = row %}{{ ns.r.get(row.selector) }}")

        assert result.fields == frozenset({"selector"})
        assert result.dynamic_accesses == ("get",)
        assert result.has_dynamic_access is True

    def test_nested_row_object_carrier_dynamic_get_reported_by_usage(self) -> None:
        """Nested containers can carry row itself, not only row APIs."""
        from elspeth.core.templates import extract_jinja2_field_usage

        result = extract_jinja2_field_usage("{% set d = {'inner': {'r': row}} %}{{ d['inner']['r'].get(row.selector) }}")

        assert result.fields == frozenset({"selector"})
        assert result.dynamic_accesses == ("get",)
        assert result.has_dynamic_access is True

    def test_list_carried_row_get_alias_reported_by_usage(self) -> None:
        """List carriers cannot hide row.get aliases."""
        from elspeth.core.templates import extract_jinja2_field_usage

        result = extract_jinja2_field_usage("{% set xs = [row.get] %}{{ xs[0](row.selector) }}")

        assert result.fields == frozenset({"selector"})
        assert result.dynamic_accesses == ("get",)
        assert result.has_dynamic_access is True

    def test_loop_target_from_row_get_alias_collection_reported_by_usage(self) -> None:
        """Loop targets over row.get collections inherit API alias guards."""
        from elspeth.core.templates import extract_jinja2_field_usage

        result = extract_jinja2_field_usage("{% set xs = [row.get] %}{% for g in xs %}{{ g(row.selector) }}{% endfor %}")

        assert result.fields == frozenset({"selector"})
        assert result.dynamic_accesses == ("get",)
        assert result.has_dynamic_access is True

    @pytest.mark.parametrize(
        "template",
        (
            ("{% set d={'xs':[row.get]} %}{% macro use(g) %}{{ g(row.selector) }}{% endmacro %}{{ use(*d['xs']) }}"),
            ("{% set d={'kw': {'g': row.get}} %}{% macro use(g) %}{{ g(row.selector) }}{% endmacro %}{{ use(**d['kw']) }}"),
            (
                "{% set d={'xs':[row.get]} %}"
                "{% macro wrap() %}{{ caller(*d['xs']) }}{% endmacro %}"
                "{% call(g) wrap() %}{{ g(row.selector) }}{% endcall %}"
            ),
            (
                "{% set d={'kw': {'g': row.get}} %}"
                "{% macro wrap() %}{{ caller(**d['kw']) }}{% endmacro %}"
                "{% call(g) wrap() %}{{ g(row.selector) }}{% endcall %}"
            ),
        ),
    )
    def test_macro_or_callblock_row_get_splats_reported_by_usage(self, template: str) -> None:
        """Macro and callblock splats can carry row.get aliases."""
        from elspeth.core.templates import extract_jinja2_field_usage

        result = extract_jinja2_field_usage(template)

        assert result.fields == frozenset({"selector"})
        assert result.dynamic_accesses == ("get",)
        assert result.has_dynamic_access is True

    @pytest.mark.parametrize(
        "template",
        (
            ("{% set d={'kw': {'g': row.get}} %}{% set kw=d['kw'] %}{% macro use(g) %}{{ g(row.selector) }}{% endmacro %}{{ use(**kw) }}"),
            (
                "{% set d={'kw': {'g': row.get}} %}{% set kw=d['kw'] %}"
                "{% macro wrap() %}{{ caller(**kw) }}{% endmacro %}"
                "{% call(g) wrap() %}{{ g(row.selector) }}{% endcall %}"
            ),
        ),
    )
    def test_realiased_row_get_mapping_splats_reported_by_usage(self, template: str) -> None:
        """Assigning a carrier slice locally must preserve row.get aliases."""
        from elspeth.core.templates import extract_jinja2_field_usage

        result = extract_jinja2_field_usage(template)

        assert result.fields == frozenset({"selector"})
        assert result.dynamic_accesses == ("get",)
        assert result.has_dynamic_access is True

    @pytest.mark.parametrize(
        "template",
        (
            "{% macro use(g) %}{{ g('secret') }}{% endmacro %}{{ use(**{row.selector: row.get}) }}",
            ("{% macro wrap() %}{{ caller(**{row.selector: row.get}) }}{% endmacro %}{% call(g) wrap() %}{{ g('secret') }}{% endcall %}"),
            "{% set d={'g': row.get} %}{% set g=d[row.selector] %}{{ g('secret') }}",
            "{% set xs=[row.get] %}{% set g=xs[row.idx] %}{{ g('secret') }}",
            "{% set d={'kw': {'g': row.get}} %}{% set g=d[row.selector]['g'] %}{{ g('secret') }}",
            "{% set d={'kw': {'g': row.get}} %}{{ d[row.selector].g('secret') }}",
            "{% set d={'kw': {'g': row.get}} %}{% macro use(g) %}{{ g('secret') }}{% endmacro %}{{ use(**d[row.selector]) }}",
            "{% set d={'xs': [row.get]} %}{% for g in d[row.selector] %}{{ g('secret') }}{% endfor %}",
            "{% set d={'xs': [row.get]} %}{% macro use(g) %}{{ g('secret') }}{% endmacro %}{{ use(*d[row.selector]) }}",
        ),
    )
    def test_dynamic_key_row_get_carriers_reported_by_usage(self, template: str) -> None:
        """Dynamic carrier keys can select row.get aliases at render time."""
        from elspeth.core.templates import extract_jinja2_field_usage

        result = extract_jinja2_field_usage(template)

        assert result.fields <= frozenset({"selector", "idx"})
        assert result.dynamic_accesses == ("get",)
        assert result.has_dynamic_access is True

    @pytest.mark.parametrize(
        ("template", "expected_fields"),
        (
            ("{% set d = {'xs': [row]} %}{% for r in d['xs'] %}{{ r.to_dict() }}{% endfor %}", frozenset()),
            ("{% set ns = namespace(xs=[row]) %}{% for r in ns.xs %}{{ r.to_dict() }}{% endfor %}", frozenset()),
            ("{% set xs = [row] %}{% set d = {'xs': xs} %}{% for r in d['xs'] %}{{ r.to_dict() }}{% endfor %}", frozenset()),
            ("{% set d = {'xs': [row]} %}{% set xs = d['xs'] %}{% for r in xs %}{{ r.to_dict() }}{% endfor %}", frozenset()),
            (
                "{% set d = {'xs': [row]} %}{% macro leak(r) %}{{ r.to_dict() }}{% endmacro %}{{ leak(*d['xs']) }}",
                frozenset(),
            ),
            (
                "{% set d = {'xs': [row]} %}{% for r in d[row.selector] %}{{ r.to_dict() }}{% endfor %}",
                frozenset({"selector"}),
            ),
            (
                "{% set d = {'xs': [row]} %}{% macro leak(r) %}{{ r.to_dict() }}{% endmacro %}{{ leak(*d[row.selector]) }}",
                frozenset({"selector"}),
            ),
        ),
    )
    def test_carried_row_collection_iterable_reported_by_usage(self, template: str, expected_fields: frozenset[str]) -> None:
        """Row collections carried by dict/namespace paths still yield row aliases."""
        from elspeth.core.templates import extract_jinja2_field_usage

        result = extract_jinja2_field_usage(template)

        assert result.fields == expected_fields
        assert result.dynamic_accesses == ("row-api",)
        assert result.has_dynamic_access is True

    @pytest.mark.parametrize(
        "template",
        (
            "{% set d = {'kw': {'r': row}} %}{{ d[row.selector]['r'].to_dict() }}",
            "{% set d = {'kw': {'r': row}} %}{{ d[row.selector].r.to_dict() }}",
        ),
    )
    def test_dynamic_key_row_object_carriers_reported_by_usage(self, template: str) -> None:
        """Dynamic carrier keys can select containers that carry row objects."""
        from elspeth.core.templates import extract_jinja2_field_usage

        result = extract_jinja2_field_usage(template)

        assert result.fields == frozenset({"selector"})
        assert result.dynamic_accesses == ("row-api",)
        assert result.has_dynamic_access is True

    def test_scalar_row_value_collection_alias_not_dynamic(self) -> None:
        """A list of declared row values is not a list of row objects."""
        from elspeth.core.templates import extract_jinja2_field_usage

        result = extract_jinja2_field_usage("{% set xs = [row.text] %}{{ xs[0] }}")

        assert result.fields == frozenset({"text"})
        assert result.dynamic_accesses == ()
        assert result.has_dynamic_access is False

    def test_container_sibling_local_dict_lookup_not_dynamic(self) -> None:
        """A non-row sibling inside a namespace remains an ordinary local value."""
        from elspeth.core.templates import extract_jinja2_field_usage

        result = extract_jinja2_field_usage("{% set ns=namespace(r=row, params={'a': 'A'}) %}{{ ns.params.get(row.selector) }}")

        assert result.fields == frozenset({"selector"})
        assert result.dynamic_accesses == ()
        assert result.has_dynamic_access is False

    def test_direct_row_to_dict_reported_by_usage(self) -> None:
        """row.to_dict() exposes the full row and must fail closed."""
        from elspeth.core.templates import extract_jinja2_field_usage

        result = extract_jinja2_field_usage("{{ row.to_dict() }}")

        assert result.fields == frozenset()
        assert result.dynamic_accesses == ("row-api",)
        assert result.has_dynamic_access is True

    @pytest.mark.parametrize("template", ("{{ row._data }}", "{{ row.__class__ }}", "{{ row.contract }}"))
    def test_direct_pipeline_row_private_or_api_attr_reported_by_usage(self, template: str) -> None:
        """Private/API row attributes are not auditable data fields."""
        from elspeth.core.templates import extract_jinja2_field_usage

        result = extract_jinja2_field_usage(template)

        assert result.fields == frozenset()
        assert result.dynamic_accesses == ("row-api",)
        assert result.has_dynamic_access is True

    def test_row_scalar_alias_map_attribute_filter_reported_by_usage(self) -> None:
        """A scalar alias derived from row cannot choose map(attribute=...)."""
        from elspeth.core.templates import extract_jinja2_field_usage

        result = extract_jinja2_field_usage("{% set attr_name = row.selector %}{{ rows | map(attribute=attr_name) | list }}")

        assert result.fields == frozenset({"selector"})
        assert result.dynamic_accesses == ("map(attribute)",)
        assert result.has_dynamic_access is True

    def test_for_loop_row_alias_private_attr_reported_by_usage(self) -> None:
        """A loop target bound to row is still a row alias."""
        from elspeth.core.templates import extract_jinja2_field_usage

        result = extract_jinja2_field_usage("{% for r in [row] %}{{ r._data }}{% endfor %}")

        assert result.fields == frozenset()
        assert result.dynamic_accesses == ("row-api",)
        assert result.has_dynamic_access is True

    def test_macro_row_arg_to_dict_reported_by_usage(self) -> None:
        """A macro parameter called with row must inherit row API guards."""
        from elspeth.core.templates import extract_jinja2_field_usage

        result = extract_jinja2_field_usage("{% macro leak(r) %}{{ r.to_dict() }}{% endmacro %}{{ leak(row) }}")

        assert result.fields == frozenset()
        assert result.dynamic_accesses == ("row-api",)
        assert result.has_dynamic_access is True

    def test_macro_row_arg_map_attribute_reported_by_usage(self) -> None:
        """A macro parameter called with row cannot hide dynamic map attributes."""
        from elspeth.core.templates import extract_jinja2_field_usage

        result = extract_jinja2_field_usage("{% macro leak(r) %}{{ rows | map(attribute=r.selector) | list }}{% endmacro %}{{ leak(row) }}")

        assert result.fields == frozenset({"selector"})
        assert result.dynamic_accesses == ("map(attribute)",)
        assert result.has_dynamic_access is True

    @pytest.mark.parametrize("template", ("{{ ([row]|first)._data }}", "{{ ([row]|first).to_dict() }}"))
    def test_row_expression_private_or_api_attr_reported_by_usage(self, template: str) -> None:
        """Row-valued expressions are not auditable data fields."""
        from elspeth.core.templates import extract_jinja2_field_usage

        result = extract_jinja2_field_usage(template)

        assert result.fields == frozenset()
        assert result.dynamic_accesses == ("row-api",)
        assert result.has_dynamic_access is True

    def test_row_expression_dynamic_get_reported_by_usage(self) -> None:
        """A row-valued expression can still expose row.get(expr)."""
        from elspeth.core.templates import extract_jinja2_field_usage

        result = extract_jinja2_field_usage("{{ ([row]|first).get(row.selector) }}")

        assert result.fields == frozenset({"selector"})
        assert result.dynamic_accesses == ("get",)
        assert result.has_dynamic_access is True

    @pytest.mark.parametrize(
        "template",
        (
            "{{ ([row] | map(attribute='to_dict') | first)() }}",
            "{{ ([row] | map(attribute='get') | first)(row.selector) }}",
            "{{ [row] | map(attribute='_data') | first }}",
        ),
    )
    def test_static_map_attribute_api_or_private_name_reported_by_usage(self, template: str) -> None:
        """Static map(attribute=...) cannot expose private/API row attributes."""
        from elspeth.core.templates import extract_jinja2_field_usage

        result = extract_jinja2_field_usage(template)

        assert result.fields <= frozenset({"selector"})
        assert result.dynamic_accesses == ("row-api",)
        assert result.has_dynamic_access is True

    def test_for_loop_row_collection_alias_reported_by_usage(self) -> None:
        """A collection alias containing row still makes loop targets row aliases."""
        from elspeth.core.templates import extract_jinja2_field_usage

        result = extract_jinja2_field_usage("{% set xs = [row] %}{% for r in xs %}{{ r.to_dict() }}{% endfor %}")

        assert result.fields == frozenset()
        assert result.dynamic_accesses == ("row-api",)
        assert result.has_dynamic_access is True

    def test_macro_default_row_arg_reported_by_usage(self) -> None:
        """A macro default bound to row must inherit row API guards."""
        from elspeth.core.templates import extract_jinja2_field_usage

        result = extract_jinja2_field_usage("{% macro leak(r=row) %}{{ r.to_dict() }}{% endmacro %}{{ leak() }}")

        assert result.fields == frozenset()
        assert result.dynamic_accesses == ("row-api",)
        assert result.has_dynamic_access is True

    def test_macro_alias_row_arg_reported_by_usage(self) -> None:
        """Calling a macro through a local alias must still bind row parameters."""
        from elspeth.core.templates import extract_jinja2_field_usage

        result = extract_jinja2_field_usage("{% macro leak(r) %}{{ r.to_dict() }}{% endmacro %}{% set fn = leak %}{{ fn(row) }}")

        assert result.fields == frozenset()
        assert result.dynamic_accesses == ("row-api",)
        assert result.has_dynamic_access is True

    def test_callblock_caller_row_arg_reported_by_usage(self) -> None:
        """A caller parameter passed row by a macro must inherit row API guards."""
        from elspeth.core.templates import extract_jinja2_field_usage

        result = extract_jinja2_field_usage(
            "{% macro wrap() %}{{ caller(row) }}{% endmacro %}{% call(r) wrap() %}{{ r.to_dict() }}{% endcall %}"
        )

        assert result.fields == frozenset()
        assert result.dynamic_accesses == ("row-api",)
        assert result.has_dynamic_access is True

    @pytest.mark.parametrize(
        ("template", "expected_fields"),
        (
            (
                "{{ row.visible }}{% macro leak(r) %}{{ r.to_dict() }}{% endmacro %}{% set ns=namespace(fn=leak) %}{{ ns.fn(row) }}",
                frozenset({"visible"}),
            ),
            (
                "{% macro leak(r) %}{{ r.to_dict() }}{% endmacro %}{% set d={'fn': leak} %}{{ d[row.selector](row) }}",
                frozenset({"selector"}),
            ),
            (
                "{% macro leak(r) %}{{ r.to_dict() }}{% endmacro %}{% macro run(fn) %}{{ fn(row) }}{% endmacro %}{{ run(leak) }}",
                frozenset(),
            ),
        ),
    )
    def test_carried_or_parameter_macro_alias_reported_by_usage(self, template: str, expected_fields: frozenset[str]) -> None:
        """Macro aliases carried through containers or parameters still bind row arguments."""
        from elspeth.core.templates import extract_jinja2_field_usage

        result = extract_jinja2_field_usage(template)

        assert result.fields == expected_fields
        assert result.dynamic_accesses == ("row-api",)
        assert result.has_dynamic_access is True

    def test_carried_callblock_macro_alias_reported_by_usage(self) -> None:
        """Callblock macros invoked through carriers still bind caller row arguments."""
        from elspeth.core.templates import extract_jinja2_field_usage

        result = extract_jinja2_field_usage(
            "{% macro wrap() %}{{ caller(row) }}{% endmacro %}"
            "{% set ns=namespace(fn=wrap) %}{% call(r) ns.fn() %}{{ r.to_dict() }}{% endcall %}"
        )

        assert result.fields == frozenset()
        assert result.dynamic_accesses == ("row-api",)
        assert result.has_dynamic_access is True

    def test_direct_row_to_checkpoint_format_reported_by_usage(self) -> None:
        """row.to_checkpoint_format() exposes serialized row data and must fail closed."""
        from elspeth.core.templates import extract_jinja2_field_usage

        result = extract_jinja2_field_usage("{{ row.to_checkpoint_format() }}")

        assert result.fields == frozenset()
        assert result.dynamic_accesses == ("row-api",)
        assert result.has_dynamic_access is True

    @pytest.mark.parametrize(
        "template",
        (
            "{% set xs = [row] %}{{ xs[0].get(row.selector) }}",
            "{{ [row][0].to_dict() }}",
            "{{ [row][0]._data }}",
        ),
    )
    def test_indexed_row_collection_receiver_reported_by_usage(self, template: str) -> None:
        """Indexing a known row collection can still produce a row receiver."""
        from elspeth.core.templates import extract_jinja2_field_usage

        result = extract_jinja2_field_usage(template)

        assert result.fields <= frozenset({"selector"})
        assert result.dynamic_accesses
        assert result.has_dynamic_access is True

    @pytest.mark.parametrize(
        "template",
        (
            "{{ [row] | join(',', attribute=row.selector) }}",
            "{{ rows | selectattr(row.selector) | list }}",
            "{{ rows | rejectattr(row.selector) | list }}",
            "{{ rows | sort(attribute=row.selector) | list }}",
            "{{ rows | groupby(row.selector) | list }}",
            "{{ rows | unique(attribute=row.selector) | list }}",
            "{{ rows | sum(attribute=row.selector) }}",
            "{{ rows | min(attribute=row.selector) }}",
            "{{ rows | max(attribute=row.selector) }}",
        ),
    )
    def test_attribute_resolving_filter_dynamic_argument_reported_by_usage(self, template: str) -> None:
        """Attribute-resolving filters must fail closed on row-derived attributes."""
        from elspeth.core.templates import extract_jinja2_field_usage

        result = extract_jinja2_field_usage(template)

        assert result.fields == frozenset({"selector"})
        assert result.dynamic_accesses == ("map(attribute)",)
        assert result.has_dynamic_access is True

    @pytest.mark.parametrize(
        "template",
        (
            "{{ row | attr(name=row.selector) }}",
            "{{ [row] | join(',', row.selector) }}",
            "{{ [row] | map('attr', row.selector) | first }}",
        ),
    )
    def test_dynamic_filter_argument_forms_reported_by_usage(self, template: str) -> None:
        """Keyword/positional filter variants can also resolve dynamic attributes."""
        from elspeth.core.templates import extract_jinja2_field_usage

        result = extract_jinja2_field_usage(template)

        assert result.fields == frozenset({"selector"})
        assert result.dynamic_accesses
        assert result.has_dynamic_access is True

    @pytest.mark.parametrize(
        "template",
        (
            "{% macro leak(r) %}{{ r.to_dict() }}{% endmacro %}{{ leak(*[row]) }}",
            "{% macro leak(r) %}{{ r.to_dict() }}{% endmacro %}{{ leak(**{'r': row}) }}",
            "{% macro wrap() %}{{ caller(*[row]) }}{% endmacro %}{% call(r) wrap() %}{{ r.to_dict() }}{% endcall %}",
        ),
    )
    def test_macro_or_callblock_splat_row_arg_reported_by_usage(self, template: str) -> None:
        """Literal splats that pass row into macro parameters must be traced."""
        from elspeth.core.templates import extract_jinja2_field_usage

        result = extract_jinja2_field_usage(template)

        assert result.fields == frozenset()
        assert result.dynamic_accesses == ("row-api",)
        assert result.has_dynamic_access is True

    @pytest.mark.parametrize(
        "template",
        (
            "{% set r, x = row, 1 %}{{ r.to_dict() }}",
            "{% set d = {'r': row} %}{{ d['r'].to_dict() }}",
            "{% set ns = namespace(r=row) %}{{ ns.r.to_dict() }}",
        ),
    )
    def test_destructured_or_carried_row_alias_reported_by_usage(self, template: str) -> None:
        """Tuple destructuring and row carrier containers must preserve row API guards."""
        from elspeth.core.templates import extract_jinja2_field_usage

        result = extract_jinja2_field_usage(template)

        assert result.fields == frozenset()
        assert result.dynamic_accesses == ("row-api",)
        assert result.has_dynamic_access is True

    @pytest.mark.parametrize(
        "template",
        (
            "{{ (row if true else row).to_dict() }}",
            "{{ (row or {}).get(row.selector) }}",
            "{{ (row|default({})).to_dict() }}",
        ),
    )
    def test_generic_row_expression_receiver_reported_by_usage(self, template: str) -> None:
        """Generic expressions that may yield row must keep row receiver guards."""
        from elspeth.core.templates import extract_jinja2_field_usage

        result = extract_jinja2_field_usage(template)

        assert result.fields <= frozenset({"selector"})
        assert result.dynamic_accesses
        assert result.has_dynamic_access is True

    @pytest.mark.parametrize(
        "template",
        (
            "{{ row.get(key=row.selector) }}",
            "{{ row.get(*[row.selector]) }}",
            "{{ row.get(**{'key': row.selector}) }}",
        ),
    )
    def test_row_get_keyword_or_splat_key_reported_by_usage(self, template: str) -> None:
        """row.get keyword/splat key forms are dynamic when row-derived."""
        from elspeth.core.templates import extract_jinja2_field_usage

        result = extract_jinja2_field_usage(template)

        assert result.fields == frozenset({"selector"})
        assert result.dynamic_accesses == ("get",)
        assert result.has_dynamic_access is True

    @pytest.mark.parametrize(
        "template",
        (
            "{% set args = lookup.args %}{{ row.get(*args) }}",
            "{% set kwargs = lookup.kwargs %}{{ row.get(**kwargs) }}",
        ),
    )
    def test_row_get_unknown_splats_reported_by_usage(self, template: str) -> None:
        """Unknown row.get splats can supply a dynamic key at render time."""
        from elspeth.core.templates import extract_jinja2_field_usage

        result = extract_jinja2_field_usage(template)

        assert result.fields == frozenset()
        assert result.dynamic_accesses == ("get",)
        assert result.has_dynamic_access is True

    @pytest.mark.parametrize(
        "template",
        (
            "{% set args = [row] %}{% macro leak(r) %}{{ r.to_dict() }}{% endmacro %}{{ leak(*args) }}",
            "{% set kwargs = {'r': row} %}{% macro leak(r) %}{{ r.to_dict() }}{% endmacro %}{{ leak(**kwargs) }}",
        ),
    )
    def test_macro_aliased_splat_row_arg_reported_by_usage(self, template: str) -> None:
        """Aliased macro splats that contain row must be traced."""
        from elspeth.core.templates import extract_jinja2_field_usage

        result = extract_jinja2_field_usage(template)

        assert result.fields == frozenset()
        assert result.dynamic_accesses == ("row-api",)
        assert result.has_dynamic_access is True

    @pytest.mark.parametrize(
        "template",
        (
            "{{ (row | attr(**{'name': 'get'}))(row.selector) }}",
            "{{ row | attr(*['get']) }}",
            "{{ ([row] | map(**{'attribute': 'to_dict'}) | first)() }}",
        ),
    )
    def test_attribute_filter_literal_splats_reported_by_usage(self, template: str) -> None:
        """Literal filter splats must fold into attr/map guard analysis."""
        from elspeth.core.templates import extract_jinja2_field_usage

        result = extract_jinja2_field_usage(template)

        assert result.fields <= frozenset({"name", "selector"})
        assert result.dynamic_accesses
        assert result.has_dynamic_access is True

    @pytest.mark.parametrize(
        "template",
        (
            "{% set opts = lookup.opts %}{{ row | attr(**opts) }}",
            "{% set opts = lookup.opts %}{{ [row] | map(**opts) | list }}",
            "{% set opts = lookup.opts %}{{ [row] | join(',', **opts) }}",
            "{% set args = lookup.args %}{{ [row] | selectattr(*args) | list }}",
        ),
    )
    def test_attribute_filter_unknown_splats_reported_by_usage(self, template: str) -> None:
        """Unknown splats on attribute filters can carry attribute names."""
        from elspeth.core.templates import extract_jinja2_field_usage

        result = extract_jinja2_field_usage(template)

        assert result.fields == frozenset()
        assert result.dynamic_accesses
        assert result.has_dynamic_access is True

    @pytest.mark.parametrize(
        ("template", "expected_fields"),
        (
            ("{{ [row] | groupby(attribute='name') | list }}", frozenset({"name"})),
            ("{{ [row] | groupby(attribute=row.selector) | list }}", frozenset({"selector"})),
        ),
    )
    def test_groupby_attribute_keyword_reported_by_usage(self, template: str, expected_fields: frozenset[str]) -> None:
        """groupby supports keyword attribute arguments like other filters."""
        from elspeth.core.templates import extract_jinja2_field_usage

        result = extract_jinja2_field_usage(template)

        assert result.fields == expected_fields
        if "selector" in expected_fields:
            assert result.dynamic_accesses == ("map(attribute)",)
            assert result.has_dynamic_access is True
        else:
            assert result.dynamic_accesses == ()
            assert result.has_dynamic_access is False

    @pytest.mark.parametrize(
        "template",
        (
            (
                "{% set args = [row.selector] %}"
                "{% macro leak(attr) %}{{ [row] | map(attribute=attr) | list }}{% endmacro %}"
                "{{ leak(*args) }}"
            ),
            (
                "{% set kwargs = {'attr': row.selector} %}"
                "{% macro leak(attr) %}{{ [row] | map(attribute=attr) | list }}{% endmacro %}"
                "{{ leak(**kwargs) }}"
            ),
            (
                "{% set args = [row.selector] %}"
                "{% macro leak() %}{{ caller(*args) }}{% endmacro %}"
                "{% call(attr) leak() %}{{ [row] | map(attribute=attr) | list }}{% endcall %}"
            ),
            (
                "{% set kwargs = {'attr': row.selector} %}"
                "{% macro leak() %}{{ caller(**kwargs) }}{% endmacro %}"
                "{% call(attr) leak() %}{{ [row] | map(attribute=attr) | list }}{% endcall %}"
            ),
        ),
    )
    def test_macro_or_callblock_row_value_splats_reported_by_usage(self, template: str) -> None:
        """Row-derived scalar aliases passed through macro splats stay dynamic."""
        from elspeth.core.templates import extract_jinja2_field_usage

        result = extract_jinja2_field_usage(template)

        assert result.fields == frozenset({"selector"})
        assert result.dynamic_accesses == ("map(attribute)",)
        assert result.has_dynamic_access is True

    def test_non_attribute_filter_splat_row_values_not_dynamic(self) -> None:
        """Ordinary filters may use declared row values as non-attribute args."""
        from elspeth.core.templates import extract_jinja2_field_usage

        result = extract_jinja2_field_usage("{{ row.text | replace(*[row.selector, 'x']) }}")

        assert result.fields == frozenset({"text", "selector"})
        assert result.dynamic_accesses == ()
        assert result.has_dynamic_access is False

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
        """row.get(dynamic_key) is visible as a dynamic access detail."""
        from elspeth.core.templates import DYNAMIC_ROW_FIELD, extract_jinja2_fields_with_details

        result = extract_jinja2_fields_with_details("{{ row.get(key, 'N/A') }}")
        assert result == {DYNAMIC_ROW_FIELD: ["get_dynamic"]}

    def test_dynamic_attr_filter_details_reported(self) -> None:
        """row|attr(expr) is visible in detailed dynamic access output."""
        from elspeth.core.templates import DYNAMIC_ROW_FIELD, extract_jinja2_fields_with_details

        result = extract_jinja2_fields_with_details("{{ row | attr(row.selector) }}")

        assert result == {"selector": ["attr"], DYNAMIC_ROW_FIELD: ["attr_dynamic"]}

    def test_dynamic_map_attribute_filter_details_reported(self) -> None:
        """row|map(attribute=expr) is visible in detailed dynamic access output."""
        from elspeth.core.templates import DYNAMIC_ROW_FIELD, extract_jinja2_fields_with_details

        result = extract_jinja2_fields_with_details("{{ row | map(attribute=field_name) | list }}")

        assert result == {DYNAMIC_ROW_FIELD: ["map_attribute_dynamic"]}

    def test_dynamic_item_access_details_reported(self) -> None:
        """row[dynamic_key] is visible as a dynamic item access detail."""
        from elspeth.core.templates import DYNAMIC_ROW_FIELD, extract_jinja2_fields_with_details

        result = extract_jinja2_fields_with_details('{% set k = "ssn" %}{{ row[k] }}')
        assert result == {DYNAMIC_ROW_FIELD: ["item_dynamic"]}
