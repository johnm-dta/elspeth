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

from jinja2 import Environment
from jinja2.nodes import Const, Getattr, Getitem, Name, Node


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
    # Handle row.field_name syntax (Getattr node)
    if isinstance(node, Getattr) and isinstance(node.node, Name) and node.node.name == namespace:
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

    def walk(node: Node) -> None:
        if isinstance(node, Getattr) and isinstance(node.node, Name) and node.node.name == namespace:
            fields.setdefault(node.attr, []).append("attr")

        if (
            isinstance(node, Getitem)
            and isinstance(node.node, Name)
            and node.node.name == namespace
            and isinstance(node.arg, Const)
            and isinstance(node.arg.value, str)
        ):
            fields.setdefault(node.arg.value, []).append("item")

        for child in node.iter_child_nodes():
            walk(child)

    walk(ast)
    return fields
