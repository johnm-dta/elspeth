"""Template field extraction utilities for development assistance.

This module helps developers discover which fields their templates reference.
The extracted fields should be EXPLICITLY declared in plugin config - this
utility does NOT automatically populate config at runtime.

Usage:
    from elspeth.core.templates import extract_jinja2_fields

    template = "Hello {{ row.name }}, your balance is {{ row.balance }}"
    fields = extract_jinja2_fields(template)
    # Returns: frozenset({"name", "balance"})
    # Developer then adds to config: required_input_fields: [name, balance]

This is a DEVELOPMENT HELPER for discovering template dependencies:
- Run this when writing/modifying templates to see what fields are used
- Use the output to populate required_input_fields in your plugin config
- Do NOT rely on runtime auto-extraction (auditability requires explicitness)

Limitations (documented so developers know when to override):
- Cannot analyze conditional access (extracts all branches)
- Cannot resolve dynamic keys (row[variable]) to concrete field names; use
  extract_jinja2_field_usage() when callers must fail closed on dynamic access
- Cannot analyze macro internals from imports
- May include fields only used in optional branches
- Bracket syntax (row["Original Name"]) returns names verbatim, which may be
  non-identifier original headers. Use extract_jinja2_fields_with_names() for
  contract-aware resolution of original → normalized names.

For templates with conditional logic, developers should review extracted
fields and declare only the truly required subset in required_input_fields.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from jinja2 import Environment
from jinja2.nodes import Call, Const, Filter, Getattr, Getitem, Name, Node

if TYPE_CHECKING:
    from elspeth.contracts.schema_contract import SchemaContract

__all__ = [
    "DYNAMIC_ROW_FIELD",
    "Jinja2FieldExtraction",
    "extract_jinja2_field_usage",
    "extract_jinja2_fields",
    "extract_jinja2_fields_with_details",
    "extract_jinja2_fields_with_names",
]

DYNAMIC_ROW_FIELD = "<dynamic-row-field>"
ATTR_FILTER_DYNAMIC_ACCESS = "attr"
MAP_ATTRIBUTE_FILTER_DYNAMIC_ACCESS = "map(attribute)"


@dataclass(frozen=True, slots=True)
class Jinja2FieldExtraction:
    """Concrete and dynamic row-field references found in a Jinja2 template."""

    fields: frozenset[str]
    dynamic_accesses: tuple[str, ...] = ()

    @property
    def has_dynamic_access(self) -> bool:
        """Return whether the template has row[expr] or row.get(expr)."""
        return bool(self.dynamic_accesses)


def _create_field_extraction_environment() -> Environment:
    return Environment(autoescape=True)


def extract_jinja2_field_usage(
    template_string: str,
    namespace: str = "row",
) -> Jinja2FieldExtraction:
    """Extract concrete fields and flag dynamic row-field access.

    Dynamic accesses such as ``row[key]`` and ``row.get(key)`` cannot be
    resolved to a concrete field name at parse time. They are therefore
    reported separately so security-sensitive callers can fail closed instead
    of treating an empty concrete-field set as "no row fields referenced."
    """
    env = _create_field_extraction_environment()
    ast = env.parse(template_string)
    fields: set[str] = set()
    dynamic_accesses: list[str] = []
    _walk_ast(ast, namespace, fields, dynamic_accesses)
    return Jinja2FieldExtraction(fields=frozenset(fields), dynamic_accesses=tuple(dynamic_accesses))


def extract_jinja2_fields(
    template_string: str,
    namespace: str = "row",
) -> frozenset[str]:
    """Extract field names accessed via namespace.field or namespace["field"].

    NOTE: This is a development helper for discovering template dependencies.
    Results should be reviewed and explicitly declared in config as
    `required_input_fields` - do NOT use this for automatic runtime population.

    Args:
        template_string: Jinja2 template to parse
        namespace: Variable name to search for (default: "row")

    Returns:
        Frozenset of field names found (may include conditionally-used fields)

    Raises:
        jinja2.TemplateSyntaxError: If template is malformed

    Examples:
        >>> extract_jinja2_fields("{{ row.name }}")
        frozenset({'name'})

        >>> extract_jinja2_fields("{{ row.a }} and {{ row.b }}")
        frozenset({'a', 'b'})

        >>> extract_jinja2_fields('{{ row["field-with-dashes"] }}')
        frozenset({'field-with-dashes'})

        >>> extract_jinja2_fields("{% if row.active %}{{ row.value }}{% endif %}")
        frozenset({'active', 'value'})  # Extracts all, even conditional

        >>> extract_jinja2_fields("{{ lookup.data }}")  # Different namespace
        frozenset()
    """
    env = _create_field_extraction_environment()
    ast = env.parse(template_string)
    fields: set[str] = set()
    dynamic_accesses: list[str] = []
    _walk_ast(ast, namespace, fields, dynamic_accesses)
    return frozenset(fields)


# PipelineRow API names that can never be valid data field names — excluded
# from field extraction. Only includes names that are unambiguously API:
# - "get" is already handled as a Call pattern (row.get("field"))
# - "contract" is a @property exposing the SchemaContract
# - "to_dict" is a serialization method
# Note: "keys", "items", "values" are NOT excluded because they can be
# legitimate column names in user data (e.g., row.items in a for loop).
_PIPELINE_ROW_API_NAMES: frozenset[str] = frozenset(
    {
        "get",
        "contract",
        "to_dict",
    }
)


def _walk_ast(node: Node, namespace: str, fields: set[str], dynamic_accesses: list[str]) -> None:
    """Recursively walk AST to find namespace attribute/item accesses.

    Args:
        node: Current AST node
        namespace: Variable name to search for
        fields: Set to accumulate found field names (mutated)
        dynamic_accesses: List to accumulate dynamic access kinds (mutated)
    """
    if (
        isinstance(node, Call)
        and isinstance(node.node, Getattr)
        and isinstance(node.node.node, Name)
        and node.node.node.name == namespace
        and node.node.attr == "get"
    ):
        # Handle row.get("field") syntax and fail-visible dynamic keys.
        if len(node.args) >= 1 and isinstance(node.args[0], Const) and isinstance(node.args[0].value, str):
            fields.add(node.args[0].value)
        elif len(node.args) >= 1:
            dynamic_accesses.append("get")

    # Handle row.field_name syntax (Getattr node)
    # Exclude PipelineRow API names (get, keys, contract, etc.) — these are
    # object methods/properties, not row data fields.
    if (
        isinstance(node, Getattr)
        and isinstance(node.node, Name)
        and node.node.name == namespace
        and node.attr not in _PIPELINE_ROW_API_NAMES
    ):
        fields.add(node.attr)

    # Handle row["field_name"] syntax and fail-visible dynamic keys.
    if isinstance(node, Getitem) and isinstance(node.node, Name) and node.node.name == namespace:
        if isinstance(node.arg, Const) and isinstance(node.arg.value, str):
            fields.add(node.arg.value)
        else:
            dynamic_accesses.append("item")

    if isinstance(node, Filter):
        _record_dynamic_attribute_filter_access(node, namespace, fields, dynamic_accesses)

    # Recurse into child nodes
    for child in node.iter_child_nodes():
        _walk_ast(child, namespace, fields, dynamic_accesses)


def _record_dynamic_attribute_filter_access(
    node: Filter,
    namespace: str,
    fields: set[str],
    dynamic_accesses: list[str],
) -> None:
    """Record attribute-resolving filters that can hide dynamic row-field reads."""
    if not _node_references_namespace(node.node, namespace):
        return

    if node.name == "attr":
        if node.args and isinstance(node.args[0], Const) and isinstance(node.args[0].value, str):
            if isinstance(node.node, Name) and node.node.name == namespace:
                fields.add(node.args[0].value)
            return
        if node.args:
            dynamic_accesses.append(ATTR_FILTER_DYNAMIC_ACCESS)
        return

    if node.name == "map":
        attribute_arg = _filter_keyword_value(node, "attribute")
        if attribute_arg is not None and not (isinstance(attribute_arg, Const) and isinstance(attribute_arg.value, str)):
            dynamic_accesses.append(MAP_ATTRIBUTE_FILTER_DYNAMIC_ACCESS)


def _filter_keyword_value(node: Filter, key: str) -> Node | None:
    for keyword in node.kwargs:
        if keyword.key == key:
            return keyword.value
    return None


def _node_references_namespace(node: Node | None, namespace: str) -> bool:
    if node is None:
        return False
    if isinstance(node, Name):
        return node.name == namespace
    return any(_node_references_namespace(child, namespace) for child in node.iter_child_nodes())


def extract_jinja2_fields_with_details(
    template_string: str,
    namespace: str = "row",
) -> dict[str, list[str]]:
    """Extract field names with access type information.

    Like extract_jinja2_fields but returns a dict showing how each field
    is accessed, useful for debugging complex templates.

    Args:
        template_string: Jinja2 template to parse
        namespace: Variable name to search for (default: "row")

    Returns:
        Dict mapping field names to list of access types ("attr" or "item")

    Examples:
        >>> extract_jinja2_fields_with_details('{{ row.a }} {{ row["a"] }}')
        {'a': ['attr', 'item']}
    """
    env = _create_field_extraction_environment()
    ast = env.parse(template_string)
    fields: dict[str, list[str]] = {}

    def append_access(field_name: str, access_type: str) -> None:
        if field_name in fields:
            fields[field_name].append(access_type)
            return
        fields[field_name] = [access_type]

    def walk(node: Node) -> None:
        if (
            isinstance(node, Call)
            and isinstance(node.node, Getattr)
            and isinstance(node.node.node, Name)
            and node.node.node.name == namespace
            and node.node.attr == "get"
        ):
            if len(node.args) >= 1 and isinstance(node.args[0], Const) and isinstance(node.args[0].value, str):
                append_access(node.args[0].value, "item")
            elif len(node.args) >= 1:
                append_access(DYNAMIC_ROW_FIELD, "get_dynamic")

        if (
            isinstance(node, Getattr)
            and isinstance(node.node, Name)
            and node.node.name == namespace
            and node.attr not in _PIPELINE_ROW_API_NAMES
        ):
            append_access(node.attr, "attr")

        if isinstance(node, Getitem) and isinstance(node.node, Name) and node.node.name == namespace:
            if isinstance(node.arg, Const) and isinstance(node.arg.value, str):
                append_access(node.arg.value, "item")
            else:
                append_access(DYNAMIC_ROW_FIELD, "item_dynamic")

        if isinstance(node, Filter) and _node_references_namespace(node.node, namespace):
            if node.name == "attr":
                if node.args and isinstance(node.args[0], Const) and isinstance(node.args[0].value, str):
                    if isinstance(node.node, Name) and node.node.name == namespace:
                        append_access(node.args[0].value, "attr")
                elif node.args:
                    append_access(DYNAMIC_ROW_FIELD, "attr_dynamic")
            elif node.name == "map":
                attribute_arg = _filter_keyword_value(node, "attribute")
                if attribute_arg is not None and not (isinstance(attribute_arg, Const) and isinstance(attribute_arg.value, str)):
                    append_access(DYNAMIC_ROW_FIELD, "map_attribute_dynamic")

        for child in node.iter_child_nodes():
            walk(child)

    walk(ast)
    return fields


def extract_jinja2_fields_with_names(
    template_string: str,
    contract: SchemaContract | None = None,
    namespace: str = "row",
) -> dict[str, dict[str, str | bool]]:
    """Extract field names with original/normalized name resolution.

    Enhanced version of extract_jinja2_fields that:
    - Reports both original and normalized names when contract provided
    - Resolves original names to their normalized form
    - Indicates whether resolution was successful

    This helps developers understand which fields their templates need
    and see both name forms for documentation/debugging.

    Args:
        template_string: Jinja2 template to parse
        contract: Optional SchemaContract for name resolution
        namespace: Variable name to search for (default: "row")

    Returns:
        Dict mapping normalized_name -> {
            "normalized": str,  # Normalized name (key)
            "original": str,    # Original name (or same as normalized if unknown)
            "resolved": bool,   # True if found in contract
        }

    Examples:
        >>> # Without contract
        >>> extract_jinja2_fields_with_names("{{ row.field }}")
        {'field': {'normalized': 'field', 'original': 'field', 'resolved': False}}

        >>> # With contract (has "'Amount USD'" -> "amount_usd")
        >>> extract_jinja2_fields_with_names(
        ...     "{{ row[\"'Amount USD'\"] }}",
        ...     contract=contract,
        ... )
        {'amount_usd': {'normalized': 'amount_usd', 'original': "'Amount USD'", 'resolved': True}}
    """
    # First, extract all field references as-written
    raw_fields = extract_jinja2_fields(template_string, namespace)

    result: dict[str, dict[str, str | bool]] = {}

    for field_as_written in raw_fields:
        if contract is not None:
            normalized = contract.find_name(field_as_written)
            if normalized is None:
                # Not in contract - report as-is
                result[field_as_written] = {
                    "normalized": field_as_written,
                    "original": field_as_written,
                    "resolved": False,
                }
                continue

            fc = contract.get_field(normalized)
            result[normalized] = {
                "normalized": normalized,
                "original": fc.original_name,
                "resolved": True,
            }
        else:
            # No contract - report as-is
            result[field_as_written] = {
                "normalized": field_as_written,
                "original": field_as_written,
                "resolved": False,
            }

    return result
