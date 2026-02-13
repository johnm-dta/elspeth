# src/elspeth/core/templates.py
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
- Cannot analyze dynamic keys (row[variable] is ignored)
- Cannot analyze macro internals from imports
- May include fields only used in optional branches

For templates with conditional logic, developers should review extracted
fields and declare only the truly required subset in required_input_fields.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from jinja2 import Environment
from jinja2.nodes import Call, Const, Getattr, Getitem, Name, Node

if TYPE_CHECKING:
    from elspeth.contracts.schema_contract import SchemaContract

__all__ = [
    "extract_jinja2_fields",
    "extract_jinja2_fields_with_details",
    "extract_jinja2_fields_with_names",
]


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
    env = Environment()
    ast = env.parse(template_string)
    fields: set[str] = set()
    _walk_ast(ast, namespace, fields)
    return frozenset(fields)


def _walk_ast(node: Node, namespace: str, fields: set[str]) -> None:
    """Recursively walk AST to find namespace attribute/item accesses.

    Args:
        node: Current AST node
        namespace: Variable name to search for
        fields: Set to accumulate found field names (mutated)
    """
    # Handle row.get("field") syntax (Call node with string literal key)
    if (
        isinstance(node, Call)
        and isinstance(node.node, Getattr)
        and isinstance(node.node.node, Name)
        and node.node.node.name == namespace
        and node.node.attr == "get"
        and len(node.args) >= 1
        and isinstance(node.args[0], Const)
        and isinstance(node.args[0].value, str)
    ):
        fields.add(node.args[0].value)

    # Handle row.field_name syntax (Getattr node)
    # Exclude the mapping method row.get itself (handled via Call above).
    if isinstance(node, Getattr) and isinstance(node.node, Name) and node.node.name == namespace and node.attr != "get":
        fields.add(node.attr)

    # Handle row["field_name"] syntax (Getitem node with string constant)
    if (
        isinstance(node, Getitem)
        and isinstance(node.node, Name)
        and node.node.name == namespace
        and isinstance(node.arg, Const)
        and isinstance(node.arg.value, str)
    ):
        fields.add(node.arg.value)

    # Recurse into child nodes
    for child in node.iter_child_nodes():
        _walk_ast(child, namespace, fields)


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
    env = Environment()
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
            and len(node.args) >= 1
            and isinstance(node.args[0], Const)
            and isinstance(node.args[0].value, str)
        ):
            append_access(node.args[0].value, "item")

        if isinstance(node, Getattr) and isinstance(node.node, Name) and node.node.name == namespace and node.attr != "get":
            append_access(node.attr, "attr")

        if (
            isinstance(node, Getitem)
            and isinstance(node.node, Name)
            and node.node.name == namespace
            and isinstance(node.arg, Const)
            and isinstance(node.arg.value, str)
        ):
            append_access(node.arg.value, "item")

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
