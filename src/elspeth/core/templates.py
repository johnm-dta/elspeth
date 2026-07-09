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

from collections.abc import Iterable
from dataclasses import dataclass
from typing import TYPE_CHECKING

from jinja2 import Environment
from jinja2.nodes import (
    Assign,
    Call,
    CallBlock,
    Const,
    Filter,
    For,
    Getattr,
    Getitem,
    List,
    Macro,
    Name,
    Node,
    NSRef,
    Tuple,
    With,
)
from jinja2.nodes import (
    Dict as DictNode,
)

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
ROW_API_DYNAMIC_ACCESS = "row-api"

_ATTRIBUTE_KEYWORD_FILTERS: frozenset[str] = frozenset({"map", "join", "sort", "unique", "sum", "min", "max"})
_ATTRIBUTE_POSITIONAL_FILTERS: frozenset[str] = frozenset({"selectattr", "rejectattr", "groupby"})
_CarrierPath = tuple[str | int, ...]
_CarrierPathPattern = tuple[str | int | None, ...]
_MacroAliases = dict[str, frozenset[str]]
_MacroContainerAliases = dict[str, dict[_CarrierPath, frozenset[str]]]


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
    namespaces, api_aliases, row_api_container_aliases, row_value_aliases, row_collection_aliases, row_container_aliases = (
        _field_extraction_context(ast, namespace)
    )
    fields: set[str] = set()
    dynamic_accesses: list[str] = []
    _walk_ast(
        ast,
        namespaces,
        api_aliases,
        row_api_container_aliases,
        row_value_aliases,
        row_collection_aliases,
        row_container_aliases,
        fields,
        dynamic_accesses,
    )
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
    namespaces, api_aliases, row_api_container_aliases, row_value_aliases, row_collection_aliases, row_container_aliases = (
        _field_extraction_context(ast, namespace)
    )
    fields: set[str] = set()
    dynamic_accesses: list[str] = []
    _walk_ast(
        ast,
        namespaces,
        api_aliases,
        row_api_container_aliases,
        row_value_aliases,
        row_collection_aliases,
        row_container_aliases,
        fields,
        dynamic_accesses,
    )
    return frozenset(fields)


# PipelineRow API names that can never be valid data field names — excluded
# from field extraction. Only includes names that are unambiguously API:
# - "get" is already handled as a Call pattern (row.get("field"))
# - "contract" is a @property exposing the SchemaContract
# - "to_dict" / "to_checkpoint_format" serialize the row payload
# Note: "keys", "items", "values" are NOT excluded because they can be
# legitimate column names in user data (e.g., row.items in a for loop).
_PIPELINE_ROW_API_NAMES: frozenset[str] = frozenset(
    {
        "get",
        "contract",
        "to_dict",
        "to_checkpoint_format",
    }
)


def _walk_ast(
    node: Node,
    namespaces: frozenset[str],
    api_aliases: dict[str, str],
    row_api_container_aliases: dict[str, dict[_CarrierPath, str]],
    row_value_aliases: frozenset[str],
    row_collection_aliases: frozenset[str],
    row_container_aliases: dict[str, frozenset[_CarrierPath]],
    fields: set[str],
    dynamic_accesses: list[str],
) -> None:
    """Recursively walk AST to find namespace attribute/item accesses.

    Args:
        node: Current AST node
        namespaces: Variable names to search for, including direct aliases
        fields: Set to accumulate found field names (mutated)
        dynamic_accesses: List to accumulate dynamic access kinds (mutated)
    """
    if isinstance(node, Call):
        alias_kind = _row_api_alias_expression_kind(node.node, api_aliases, row_api_container_aliases)
        if alias_kind is not None:
            _append_dynamic_access(dynamic_accesses, alias_kind)

    if (
        isinstance(node, Call)
        and isinstance(node.node, Getattr)
        and _node_may_be_row_receiver(node.node.node, namespaces, row_collection_aliases, row_container_aliases)
        and node.node.attr == "get"
    ):
        # Handle row.get("field") syntax and fail-visible dynamic keys.
        key_arg = _call_positional_or_keyword_value(node, 0, "key")
        if isinstance(key_arg, Const) and isinstance(key_arg.value, str):
            fields.add(key_arg.value)
        elif (
            key_arg is not None
            or _has_unknown_star_values(node.dyn_args)
            or _has_unknown_kwarg_values(node.dyn_kwargs)
            or _node_references_tracked_row(node.dyn_args, namespaces, row_value_aliases, row_collection_aliases, row_container_aliases)
            or _node_references_tracked_row(node.dyn_kwargs, namespaces, row_value_aliases, row_collection_aliases, row_container_aliases)
        ):
            _append_dynamic_access(dynamic_accesses, "get")

    if (
        isinstance(node, Call)
        and isinstance(node.node, Getattr)
        and _node_may_be_row_receiver(node.node.node, namespaces, row_collection_aliases, row_container_aliases)
        and node.node.attr in _PIPELINE_ROW_API_NAMES
        and node.node.attr != "get"
    ):
        _append_dynamic_access(dynamic_accesses, ROW_API_DYNAMIC_ACCESS)

    # Handle row.field_name syntax (Getattr node)
    # Exclude PipelineRow API names (get, keys, contract, etc.) — these are
    # object methods/properties, not row data fields.
    if (
        isinstance(node, Getattr)
        and _node_may_be_row_receiver(node.node, namespaces, row_collection_aliases, row_container_aliases)
        and _is_blocked_row_attribute_name(node.attr)
    ):
        _append_dynamic_access(dynamic_accesses, ROW_API_DYNAMIC_ACCESS)

    if (
        isinstance(node, Getattr)
        and _node_may_be_row_receiver(node.node, namespaces, row_collection_aliases, row_container_aliases)
        and not _is_blocked_attr_filter_name(node.attr)
    ):
        fields.add(node.attr)

    # Handle row["field_name"] syntax and fail-visible dynamic keys.
    if isinstance(node, Getitem) and _node_may_be_row_receiver(node.node, namespaces, row_collection_aliases, row_container_aliases):
        if isinstance(node.arg, Const) and isinstance(node.arg.value, str):
            fields.add(node.arg.value)
        else:
            _append_dynamic_access(dynamic_accesses, "item")

    if isinstance(node, Filter):
        _record_dynamic_attribute_filter_access(
            node, namespaces, row_value_aliases, row_collection_aliases, row_container_aliases, fields, dynamic_accesses
        )

    # Recurse into child nodes
    for child in node.iter_child_nodes():
        _walk_ast(
            child,
            namespaces,
            api_aliases,
            row_api_container_aliases,
            row_value_aliases,
            row_collection_aliases,
            row_container_aliases,
            fields,
            dynamic_accesses,
        )


def _record_dynamic_attribute_filter_access(
    node: Filter,
    namespaces: frozenset[str],
    row_value_aliases: frozenset[str],
    row_collection_aliases: frozenset[str],
    row_container_aliases: dict[str, frozenset[_CarrierPath]],
    fields: set[str],
    dynamic_accesses: list[str],
) -> None:
    """Record attribute-resolving filters that can hide dynamic row-field reads."""
    if node.name == "attr":
        if not _node_may_be_row_receiver(node.node, namespaces, row_collection_aliases, row_container_aliases):
            return
        attr_arg = _filter_positional_or_keyword_value(node, 0, "name")
        if attr_arg is not None and isinstance(attr_arg, Const) and isinstance(attr_arg.value, str):
            if _is_blocked_attr_filter_name(attr_arg.value):
                _append_dynamic_access(dynamic_accesses, ATTR_FILTER_DYNAMIC_ACCESS)
                return
            if _node_may_be_row_receiver(node.node, namespaces, row_collection_aliases, row_container_aliases):
                fields.add(attr_arg.value)
            return
        if (
            attr_arg is not None
            or _has_unknown_star_values(node.dyn_args)
            or _has_unknown_kwarg_values(node.dyn_kwargs)
            or _node_references_tracked_row(node.dyn_args, namespaces, row_value_aliases, row_collection_aliases, row_container_aliases)
            or _node_references_tracked_row(node.dyn_kwargs, namespaces, row_value_aliases, row_collection_aliases, row_container_aliases)
        ):
            _append_dynamic_access(dynamic_accesses, ATTR_FILTER_DYNAMIC_ACCESS)
        return

    attribute_arg = _attribute_resolving_filter_argument(node)
    dynamic_splat = (
        _has_unknown_star_values(node.dyn_args)
        or _has_unknown_kwarg_values(node.dyn_kwargs)
        or _node_references_tracked_row(node.dyn_args, namespaces, row_value_aliases, row_collection_aliases, row_container_aliases)
        or _node_references_tracked_row(node.dyn_kwargs, namespaces, row_value_aliases, row_collection_aliases, row_container_aliases)
    )
    maps_row_objects = _iter_may_yield_row_object(node.node, namespaces, row_collection_aliases, row_container_aliases)
    if attribute_arg is None:
        if dynamic_splat and (
            maps_row_objects or _node_may_be_row_receiver(node.node, namespaces, row_collection_aliases, row_container_aliases)
        ):
            _append_dynamic_access(dynamic_accesses, MAP_ATTRIBUTE_FILTER_DYNAMIC_ACCESS)
        return

    if maps_row_objects and isinstance(attribute_arg, Const) and isinstance(attribute_arg.value, str):
        if _is_blocked_attr_filter_name(attribute_arg.value):
            _append_dynamic_access(dynamic_accesses, ROW_API_DYNAMIC_ACCESS)
        else:
            fields.add(attribute_arg.value)
        return
    if dynamic_splat or (
        attribute_arg is not None
        and not (isinstance(attribute_arg, Const) and isinstance(attribute_arg.value, str))
        and (
            maps_row_objects
            or _node_may_be_row_receiver(node.node, namespaces, row_collection_aliases, row_container_aliases)
            or _node_references_namespace(attribute_arg, namespaces)
            or _node_references_row_value_alias(attribute_arg, row_value_aliases)
        )
    ):
        _append_dynamic_access(dynamic_accesses, MAP_ATTRIBUTE_FILTER_DYNAMIC_ACCESS)


def _filter_keyword_value(node: Filter, key: str) -> Node | None:
    for keyword in node.kwargs:
        if keyword.key == key:
            return keyword.value
    if isinstance(node.dyn_kwargs, DictNode):
        return _literal_kwarg_values(node.dyn_kwargs).get(key)
    return None


def _filter_positional_or_keyword_value(node: Filter, position: int, key: str) -> Node | None:
    if len(node.args) > position:
        return node.args[position]
    star_values = _literal_star_values(node.dyn_args)
    if len(star_values) > position:
        return star_values[position]
    return _filter_keyword_value(node, key)


def _call_keyword_value(node: Call, key: str) -> Node | None:
    for keyword in node.kwargs:
        if keyword.key == key:
            return keyword.value
    if isinstance(node.dyn_kwargs, DictNode):
        return _literal_kwarg_values(node.dyn_kwargs).get(key)
    return None


def _call_positional_or_keyword_value(node: Call, position: int, key: str) -> Node | None:
    if len(node.args) > position:
        return node.args[position]
    star_values = _literal_star_values(node.dyn_args)
    if len(star_values) > position:
        return star_values[position]
    return _call_keyword_value(node, key)


def _attribute_resolving_filter_argument(node: Filter) -> Node | None:
    if node.name == "map":
        if len(node.args) > 1 and isinstance(node.args[0], Const) and node.args[0].value == "attr":
            return node.args[1]
        return _filter_keyword_value(node, "attribute")
    if node.name == "join":
        return _filter_positional_or_keyword_value(node, 1, "attribute")
    if node.name in _ATTRIBUTE_KEYWORD_FILTERS:
        return _filter_keyword_value(node, "attribute")
    if node.name in _ATTRIBUTE_POSITIONAL_FILTERS:
        return _filter_positional_or_keyword_value(node, 0, "attribute")
    return None


def _append_dynamic_access(dynamic_accesses: list[str], kind: str) -> None:
    if kind not in dynamic_accesses:
        dynamic_accesses.append(kind)


def _field_extraction_context(
    ast: Node,
    namespace: str,
) -> tuple[
    frozenset[str],
    dict[str, str],
    dict[str, dict[_CarrierPath, str]],
    frozenset[str],
    frozenset[str],
    dict[str, frozenset[_CarrierPath]],
]:
    namespaces = {namespace}
    api_aliases: dict[str, str] = {}
    row_api_container_aliases: dict[str, dict[_CarrierPath, str]] = {}
    row_value_aliases: set[str] = set()
    row_collection_aliases: set[str] = set()
    row_container_aliases: dict[str, set[_CarrierPath]] = {}
    macros = {node.name: node for node in ast.find_all(Macro)}
    macro_aliases, macro_container_aliases = _macro_alias_context(ast, macros)
    changed = True
    while changed:
        changed = False
        current_api_aliases = dict(api_aliases)
        current_row_api_container_aliases = {name: dict(entries) for name, entries in row_api_container_aliases.items()}
        current_namespaces = frozenset(namespaces)
        current_row_value_aliases = frozenset(row_value_aliases)
        current_row_collection_aliases = frozenset(row_collection_aliases)
        current_row_container_aliases = {name: frozenset(paths) for name, paths in row_container_aliases.items()}
        for target, value in _assignment_pairs(ast):
            if _record_context_binding(
                target,
                value,
                namespaces,
                api_aliases,
                row_api_container_aliases,
                row_value_aliases,
                row_collection_aliases,
                row_container_aliases,
                current_api_aliases,
                current_row_api_container_aliases,
                current_namespaces,
                current_row_value_aliases,
                current_row_collection_aliases,
                current_row_container_aliases,
            ):
                changed = True
        for target_name in _for_row_alias_targets(ast, current_namespaces, current_row_collection_aliases, current_row_container_aliases):
            if target_name not in namespaces:
                namespaces.add(target_name)
                changed = True
        for target_name, alias_kind in _for_api_alias_targets(ast, current_api_aliases, current_row_api_container_aliases):
            if target_name not in api_aliases or api_aliases[target_name] != alias_kind:
                api_aliases[target_name] = alias_kind
                changed = True
        for target, value in _macro_argument_bindings(ast, macros, macro_aliases, macro_container_aliases):
            if _record_context_binding(
                target,
                value,
                namespaces,
                api_aliases,
                row_api_container_aliases,
                row_value_aliases,
                row_collection_aliases,
                row_container_aliases,
                current_api_aliases,
                current_row_api_container_aliases,
                current_namespaces,
                current_row_value_aliases,
                current_row_collection_aliases,
                current_row_container_aliases,
            ):
                changed = True
        for target_name in _macro_row_splat_targets(
            ast, macros, macro_aliases, macro_container_aliases, current_row_collection_aliases, current_row_container_aliases
        ):
            if target_name not in namespaces:
                namespaces.add(target_name)
                changed = True
        for target_name, alias_kind in _macro_api_splat_targets(
            ast,
            macros,
            macro_aliases,
            macro_container_aliases,
            current_api_aliases,
            current_row_api_container_aliases,
            current_namespaces,
            current_row_collection_aliases,
            current_row_container_aliases,
        ):
            if target_name not in api_aliases or api_aliases[target_name] != alias_kind:
                api_aliases[target_name] = alias_kind
                changed = True
        for target, value in _callblock_argument_bindings(ast, macros, macro_aliases, macro_container_aliases):
            if _record_context_binding(
                target,
                value,
                namespaces,
                api_aliases,
                row_api_container_aliases,
                row_value_aliases,
                row_collection_aliases,
                row_container_aliases,
                current_api_aliases,
                current_row_api_container_aliases,
                current_namespaces,
                current_row_value_aliases,
                current_row_collection_aliases,
                current_row_container_aliases,
            ):
                changed = True
        for target_name in _callblock_row_splat_targets(
            ast, macros, macro_aliases, macro_container_aliases, current_row_collection_aliases, current_row_container_aliases
        ):
            if target_name not in namespaces:
                namespaces.add(target_name)
                changed = True
        for target_name, alias_kind in _callblock_api_splat_targets(
            ast,
            macros,
            macro_aliases,
            macro_container_aliases,
            current_api_aliases,
            current_row_api_container_aliases,
            current_namespaces,
            current_row_collection_aliases,
            current_row_container_aliases,
        ):
            if target_name not in api_aliases or api_aliases[target_name] != alias_kind:
                api_aliases[target_name] = alias_kind
                changed = True
    return (
        frozenset(namespaces),
        api_aliases,
        row_api_container_aliases,
        frozenset(row_value_aliases),
        frozenset(row_collection_aliases),
        {name: frozenset(paths) for name, paths in row_container_aliases.items()},
    )


def _record_context_binding(
    target: Node,
    value: Node,
    namespaces: set[str],
    api_aliases: dict[str, str],
    row_api_container_aliases: dict[str, dict[_CarrierPath, str]],
    row_value_aliases: set[str],
    row_collection_aliases: set[str],
    row_container_aliases: dict[str, set[_CarrierPath]],
    current_api_aliases: dict[str, str],
    current_row_api_container_aliases: dict[str, dict[_CarrierPath, str]],
    current_namespaces: frozenset[str],
    current_row_value_aliases: frozenset[str],
    current_row_collection_aliases: frozenset[str],
    current_row_container_aliases: dict[str, frozenset[_CarrierPath]],
) -> bool:
    if isinstance(target, NSRef):
        return _record_namespace_ref_context_binding(
            target,
            value,
            namespaces,
            api_aliases,
            row_api_container_aliases,
            row_value_aliases,
            row_collection_aliases,
            row_container_aliases,
            current_api_aliases,
            current_row_api_container_aliases,
            current_namespaces,
            current_row_value_aliases,
            current_row_collection_aliases,
            current_row_container_aliases,
        )
    if not isinstance(target, Name):
        return False
    api_container_entries = _row_api_container_entries(
        value,
        current_api_aliases,
        current_row_api_container_aliases,
        current_namespaces,
        current_row_collection_aliases,
        current_row_container_aliases,
    )
    if api_container_entries and row_api_container_aliases.get(target.name) != api_container_entries:
        row_api_container_aliases[target.name] = api_container_entries
        return True
    if _node_is_row_collection_expression(value, current_namespaces, current_row_collection_aliases, current_row_container_aliases):
        if target.name not in row_collection_aliases:
            row_collection_aliases.add(target.name)
            return True
        return False
    row_container_entries = _row_object_container_paths(
        value, current_namespaces, current_row_collection_aliases, current_row_container_aliases
    )
    if row_container_entries:
        existing_paths = row_container_aliases.setdefault(target.name, set())
        if not row_container_entries <= existing_paths:
            existing_paths.update(row_container_entries)
            return True
        return False
    if _node_is_row_object_expression(value, current_namespaces, current_row_collection_aliases, current_row_container_aliases):
        if target.name not in namespaces:
            namespaces.add(target.name)
            return True
        return False
    alias_kind: str | None = None
    if isinstance(value, Name):
        alias_kind = current_api_aliases.get(value.name)
    if alias_kind is None:
        alias_kind = _row_api_alias_expression_kind(value, current_api_aliases, current_row_api_container_aliases)
    if alias_kind is None:
        alias_kind = _row_api_dynamic_access_kind(value, current_namespaces, current_row_collection_aliases, current_row_container_aliases)
    if alias_kind is not None and api_aliases.get(target.name) != alias_kind:
        api_aliases[target.name] = alias_kind
        return True
    if (
        target.name not in namespaces
        and target.name not in api_aliases
        and target.name not in row_value_aliases
        and (
            (isinstance(value, Name) and value.name in current_row_value_aliases)
            or _node_references_namespace(value, current_namespaces)
            or _node_references_row_value_alias(value, current_row_value_aliases)
        )
    ):
        row_value_aliases.add(target.name)
        return True
    return False


def _record_namespace_ref_context_binding(
    target: NSRef,
    value: Node,
    namespaces: set[str],
    api_aliases: dict[str, str],
    row_api_container_aliases: dict[str, dict[_CarrierPath, str]],
    row_value_aliases: set[str],
    row_collection_aliases: set[str],
    row_container_aliases: dict[str, set[_CarrierPath]],
    current_api_aliases: dict[str, str],
    current_row_api_container_aliases: dict[str, dict[_CarrierPath, str]],
    current_namespaces: frozenset[str],
    current_row_value_aliases: frozenset[str],
    current_row_collection_aliases: frozenset[str],
    current_row_container_aliases: dict[str, frozenset[_CarrierPath]],
) -> bool:
    changed = False
    container_entries = _row_api_container_entries(
        value,
        current_api_aliases,
        current_row_api_container_aliases,
        current_namespaces,
        current_row_collection_aliases,
        current_row_container_aliases,
    )
    alias_kind = _row_api_alias_expression_kind(value, current_api_aliases, current_row_api_container_aliases)
    if alias_kind is None:
        alias_kind = _row_api_dynamic_access_kind(value, current_namespaces, current_row_collection_aliases, current_row_container_aliases)
    if alias_kind is not None:
        container_entries[()] = alias_kind
    prefixed_entries = {((target.attr, *path) if path else (target.attr,)): kind for path, kind in container_entries.items()}
    if prefixed_entries:
        existing = row_api_container_aliases.setdefault(target.name, {})
        for path, kind in prefixed_entries.items():
            if existing.get(path) != kind:
                existing[path] = kind
                changed = True
    row_container_entries = _row_object_container_paths(
        value, current_namespaces, current_row_collection_aliases, current_row_container_aliases
    )
    if _node_is_row_object_expression(value, current_namespaces, current_row_collection_aliases, current_row_container_aliases):
        row_container_entries.add(())
    if row_container_entries:
        existing_paths = row_container_aliases.setdefault(target.name, set())
        prefixed_paths = {((target.attr, *path) if path else (target.attr,)) for path in row_container_entries}
        if not prefixed_paths <= existing_paths:
            existing_paths.update(prefixed_paths)
            changed = True
    return changed


def _assignment_pairs(ast: Node) -> list[tuple[Node, Node]]:
    pairs: list[tuple[Node, Node]] = []
    for assign_node in ast.find_all(Assign):
        pairs.extend(_binding_pairs(assign_node.target, assign_node.node))
    for with_node in ast.find_all(With):
        for target, value in zip(with_node.targets, with_node.values, strict=False):
            pairs.extend(_binding_pairs(target, value))
    return pairs


def _binding_pairs(target: Node, value: Node) -> list[tuple[Node, Node]]:
    if isinstance(target, Tuple) and isinstance(value, (Tuple, List)):
        pairs: list[tuple[Node, Node]] = []
        for target_item, value_item in zip(target.items, value.items, strict=False):
            pairs.extend(_binding_pairs(target_item, value_item))
        return pairs
    return [(target, value)]


def _for_row_alias_targets(
    ast: Node,
    namespaces: frozenset[str],
    row_collection_aliases: frozenset[str],
    row_container_aliases: dict[str, frozenset[_CarrierPath]],
) -> list[str]:
    targets: list[str] = []
    for node in ast.find_all(For):
        if _iter_may_yield_row_object(node.iter, namespaces, row_collection_aliases, row_container_aliases):
            targets.extend(_target_names(node.target))
    return targets


def _for_api_alias_targets(
    ast: Node,
    api_aliases: dict[str, str],
    row_api_container_aliases: dict[str, dict[_CarrierPath, str]],
) -> list[tuple[str, str]]:
    targets: list[tuple[str, str]] = []
    for node in ast.find_all(For):
        alias_kind = _iter_may_yield_row_api_kind(node.iter, api_aliases, row_api_container_aliases)
        if alias_kind is not None:
            targets.extend((target_name, alias_kind) for target_name in _target_names(node.target))
    return targets


def _iter_may_yield_row_api_kind(
    node: Node | None,
    api_aliases: dict[str, str],
    row_api_container_aliases: dict[str, dict[_CarrierPath, str]],
) -> str | None:
    if node is None:
        return None
    access_path = _row_api_container_access_path(node)
    if access_path is not None:
        base_name, path = access_path
        entries = row_api_container_aliases.get(base_name)
        if entries:
            return _merge_row_api_kinds(_carrier_child_kinds(entries, path))
    dynamic_access_pattern = _row_api_dynamic_container_access_pattern(node)
    if dynamic_access_pattern is not None:
        base_name, path_pattern = dynamic_access_pattern
        entries = row_api_container_aliases.get(base_name)
        if entries:
            return _merge_row_api_kinds(_carrier_pattern_child_kinds(entries, path_pattern))
    if isinstance(node, (List, Tuple)):
        kinds: list[str] = []
        for item in node.items:
            kind = _row_api_alias_expression_kind(item, api_aliases, row_api_container_aliases)
            if kind is not None:
                kinds.append(kind)
        return _merge_row_api_kinds(kinds)
    if isinstance(node, Filter) and node.name not in {"map", "first", "last", "random"}:
        return _iter_may_yield_row_api_kind(node.node, api_aliases, row_api_container_aliases)
    return None


def _iter_may_yield_row_object(
    node: Node | None,
    namespaces: frozenset[str],
    row_collection_aliases: frozenset[str],
    row_container_aliases: dict[str, frozenset[_CarrierPath]],
) -> bool:
    if node is None:
        return False
    if isinstance(node, Name):
        if node.name in row_collection_aliases:
            return True
        paths = row_container_aliases.get(node.name)
        return bool(paths and _carrier_has_child_path(paths, ()))
    access_path = _row_api_container_access_path(node)
    if access_path is not None:
        base_name, path = access_path
        paths = row_container_aliases.get(base_name)
        if paths and _carrier_has_child_path(paths, path):
            return True
    dynamic_access_pattern = _row_api_dynamic_container_access_pattern(node)
    if dynamic_access_pattern is not None:
        base_name, path_pattern = dynamic_access_pattern
        paths = row_container_aliases.get(base_name)
        if paths and _carrier_pattern_has_child_path(paths, path_pattern):
            return True
    if isinstance(node, (List, Tuple)):
        return any(_node_is_row_object_expression(item, namespaces, row_collection_aliases, row_container_aliases) for item in node.items)
    if isinstance(node, Filter) and node.name not in {"map", "first", "last", "random"}:
        return _iter_may_yield_row_object(node.node, namespaces, row_collection_aliases, row_container_aliases)
    return False


def _target_names(node: Node) -> list[str]:
    if isinstance(node, Name):
        return [node.name]
    return [child.name for child in node.find_all(Name)]


def _macro_alias_context(ast: Node, macros: dict[str, Macro]) -> tuple[_MacroAliases, _MacroContainerAliases]:
    aliases: dict[str, set[str]] = {}
    containers: dict[str, dict[_CarrierPath, frozenset[str]]] = {}
    changed = True
    while changed:
        changed = False
        current_aliases = {name: frozenset(macro_names) for name, macro_names in aliases.items()}
        current_containers = {name: dict(entries) for name, entries in containers.items()}
        for target, value in _assignment_pairs(ast):
            if _record_macro_binding(target, value, aliases, containers, macros, current_aliases, current_containers):
                changed = True
        for target, value in _macro_argument_bindings(ast, macros, current_aliases, current_containers):
            if _record_macro_binding(target, value, aliases, containers, macros, current_aliases, current_containers):
                changed = True
        for target, value in _callblock_argument_bindings(ast, macros, current_aliases, current_containers):
            if _record_macro_binding(target, value, aliases, containers, macros, current_aliases, current_containers):
                changed = True
    return {name: frozenset(macro_names) for name, macro_names in aliases.items()}, containers


def _record_macro_binding(
    target: Node,
    value: Node,
    aliases: dict[str, set[str]],
    containers: _MacroContainerAliases,
    macros: dict[str, Macro],
    current_aliases: _MacroAliases,
    current_containers: _MacroContainerAliases,
) -> bool:
    changed = False
    value_names = _macro_expression_names(value, macros, current_aliases, current_containers)
    container_entries = _macro_container_entries(value, macros, current_aliases, current_containers)
    if isinstance(target, Name):
        if value_names:
            existing = aliases.setdefault(target.name, set())
            if not value_names <= existing:
                existing.update(value_names)
                changed = True
        if container_entries and containers.get(target.name) != container_entries:
            containers[target.name] = container_entries
            changed = True
        return changed
    if isinstance(target, NSRef):
        if value_names:
            container_entries = dict(container_entries)
            container_entries[()] = value_names
        prefixed_entries = {((target.attr, *path) if path else (target.attr,)): names for path, names in container_entries.items()}
        if prefixed_entries:
            existing_entries = containers.setdefault(target.name, {})
            for path, names in prefixed_entries.items():
                if existing_entries.get(path) != names:
                    existing_entries[path] = names
                    changed = True
    return changed


def _macro_expression_names(
    node: Node,
    macros: dict[str, Macro],
    macro_aliases: _MacroAliases,
    macro_container_aliases: _MacroContainerAliases,
) -> frozenset[str]:
    if isinstance(node, Name):
        if node.name in macros:
            return frozenset({node.name})
        return macro_aliases.get(node.name, frozenset())
    access_path = _row_api_container_access_path(node)
    if access_path is not None:
        base_name, path = access_path
        entries = macro_container_aliases.get(base_name)
        if not entries:
            return frozenset()
        if path:
            return entries.get(path, frozenset())
        return _merge_macro_names(entries.values())
    dynamic_access_pattern = _row_api_dynamic_container_access_pattern(node)
    if dynamic_access_pattern is not None:
        base_name, path_pattern = dynamic_access_pattern
        entries = macro_container_aliases.get(base_name)
        if entries:
            return _merge_macro_names(_carrier_pattern_value_macro_names(entries, path_pattern))
    return frozenset()


def _macro_container_entries(
    node: Node,
    macros: dict[str, Macro],
    macro_aliases: _MacroAliases,
    macro_container_aliases: _MacroContainerAliases,
) -> dict[_CarrierPath, frozenset[str]]:
    if isinstance(node, Name):
        return dict(macro_container_aliases.get(node.name, {}))
    access_path = _row_api_container_access_path(node)
    if access_path is not None:
        base_name, path = access_path
        base_entries = macro_container_aliases.get(base_name)
        if base_entries:
            return _carrier_relative_macro_entries(base_entries, path)
    dynamic_access_pattern = _row_api_dynamic_container_access_pattern(node)
    if dynamic_access_pattern is not None:
        base_name, path_pattern = dynamic_access_pattern
        base_entries = macro_container_aliases.get(base_name)
        if base_entries:
            return _carrier_pattern_relative_macro_entries(base_entries, path_pattern)
    if isinstance(node, DictNode):
        entries: dict[_CarrierPath, frozenset[str]] = {}
        for pair in node.items:
            if not (isinstance(pair.key, Const) and isinstance(pair.key.value, str)):
                continue
            dict_key_path: _CarrierPath = (pair.key.value,)
            for child_path, child_names in _macro_container_entries(pair.value, macros, macro_aliases, macro_container_aliases).items():
                entries[dict_key_path + child_path] = child_names
            names = _macro_expression_names(pair.value, macros, macro_aliases, macro_container_aliases)
            if names:
                entries[dict_key_path] = names
        return entries
    if isinstance(node, (List, Tuple)):
        entries = {}
        for index, item in enumerate(node.items):
            list_key_path: _CarrierPath = (index,)
            for child_path, child_names in _macro_container_entries(item, macros, macro_aliases, macro_container_aliases).items():
                entries[list_key_path + child_path] = child_names
            names = _macro_expression_names(item, macros, macro_aliases, macro_container_aliases)
            if names:
                entries[list_key_path] = names
        return entries
    if isinstance(node, Call) and isinstance(node.node, Name) and node.node.name == "namespace":
        entries = {}
        for keyword in node.kwargs:
            namespace_key_path: _CarrierPath = (keyword.key,)
            for child_path, child_names in _macro_container_entries(keyword.value, macros, macro_aliases, macro_container_aliases).items():
                entries[namespace_key_path + child_path] = child_names
            names = _macro_expression_names(keyword.value, macros, macro_aliases, macro_container_aliases)
            if names:
                entries[namespace_key_path] = names
        return entries
    return {}


def _macro_names_for_callee(
    node: Node,
    macros: dict[str, Macro],
    macro_aliases: _MacroAliases,
    macro_container_aliases: _MacroContainerAliases,
) -> frozenset[str]:
    return _macro_expression_names(node, macros, macro_aliases, macro_container_aliases)


def _macro_argument_bindings(
    ast: Node,
    macros: dict[str, Macro],
    macro_aliases: _MacroAliases,
    macro_container_aliases: _MacroContainerAliases,
) -> list[tuple[Name, Node]]:
    bindings: list[tuple[Name, Node]] = []
    for node in ast.find_all(Call):
        for macro_name in _macro_names_for_callee(node.node, macros, macro_aliases, macro_container_aliases):
            macro = macros[macro_name]
            explicit_kwargs: dict[str, Node] = {keyword.key: keyword.value for keyword in node.kwargs}
            if isinstance(node.dyn_kwargs, DictNode):
                explicit_kwargs.update(_literal_kwarg_values(node.dyn_kwargs))
            star_values = _literal_star_values(node.dyn_args)
            default_offset = len(macro.args) - len(macro.defaults)
            for index, target in enumerate(macro.args):
                if index < len(node.args):
                    bindings.append((target, node.args[index]))
                    continue
                star_index = index - len(node.args)
                if star_index < len(star_values):
                    bindings.append((target, star_values[star_index]))
                    continue
                if target.name in explicit_kwargs:
                    bindings.append((target, explicit_kwargs[target.name]))
                    continue
                default_index = index - default_offset
                if default_index >= 0:
                    bindings.append((target, macro.defaults[default_index]))
    return bindings


def _macro_row_splat_targets(
    ast: Node,
    macros: dict[str, Macro],
    macro_aliases: _MacroAliases,
    macro_container_aliases: _MacroContainerAliases,
    row_collection_aliases: frozenset[str],
    row_container_aliases: dict[str, frozenset[_CarrierPath]],
) -> list[str]:
    targets: list[str] = []
    for node in ast.find_all(Call):
        for macro_name in _macro_names_for_callee(node.node, macros, macro_aliases, macro_container_aliases):
            macro = macros[macro_name]
            if _iter_may_yield_row_object(node.dyn_args, frozenset(), row_collection_aliases, row_container_aliases):
                targets.extend(target.name for target in macro.args[len(node.args) :] if isinstance(target, Name))
            if _node_references_name(node.dyn_kwargs, frozenset(row_container_aliases)):
                targets.extend(target.name for target in macro.args if isinstance(target, Name))
    return targets


def _macro_api_splat_targets(
    ast: Node,
    macros: dict[str, Macro],
    macro_aliases: _MacroAliases,
    macro_container_aliases: _MacroContainerAliases,
    api_aliases: dict[str, str],
    row_api_container_aliases: dict[str, dict[_CarrierPath, str]],
    namespaces: frozenset[str],
    row_collection_aliases: frozenset[str],
    row_container_aliases: dict[str, frozenset[_CarrierPath]],
) -> list[tuple[str, str]]:
    targets: list[tuple[str, str]] = []
    for node in ast.find_all(Call):
        for macro_name in _macro_names_for_callee(node.node, macros, macro_aliases, macro_container_aliases):
            macro = macros[macro_name]
            star_kind = _iter_may_yield_row_api_kind(node.dyn_args, api_aliases, row_api_container_aliases)
            if star_kind is not None:
                targets.extend((target.name, star_kind) for target in macro.args[len(node.args) :] if isinstance(target, Name))
            for target in macro.args:
                keyword_kind = _row_api_mapping_key_kind(
                    node.dyn_kwargs,
                    target.name,
                    api_aliases,
                    row_api_container_aliases,
                    namespaces,
                    row_collection_aliases,
                    row_container_aliases,
                )
                if keyword_kind is not None:
                    targets.append((target.name, keyword_kind))
    return targets


def _callblock_argument_bindings(
    ast: Node,
    macros: dict[str, Macro],
    macro_aliases: _MacroAliases,
    macro_container_aliases: _MacroContainerAliases,
) -> list[tuple[Name, Node]]:
    bindings: list[tuple[Name, Node]] = []
    for node in ast.find_all(CallBlock):
        for macro_name in _macro_names_for_callee(node.call.node, macros, macro_aliases, macro_container_aliases):
            macro = macros[macro_name]
            for caller_call in macro.find_all(Call):
                if not isinstance(caller_call.node, Name) or caller_call.node.name != "caller":
                    continue
                bindings.extend(
                    (target, value) for target, value in zip(node.args, caller_call.args, strict=False) if isinstance(target, Name)
                )
                star_values = _literal_star_values(caller_call.dyn_args)
                explicit_count = len(caller_call.args)
                for target, value in zip(node.args[explicit_count:], star_values, strict=False):
                    if isinstance(target, Name):
                        bindings.append((target, value))
                if isinstance(caller_call.dyn_kwargs, DictNode):
                    dyn_kwargs = _literal_kwarg_values(caller_call.dyn_kwargs)
                    bindings.extend(
                        (target, dyn_kwargs[target.name]) for target in node.args if isinstance(target, Name) and target.name in dyn_kwargs
                    )
    return bindings


def _row_api_mapping_key_kind(
    node: Node | None,
    key: str,
    api_aliases: dict[str, str],
    row_api_container_aliases: dict[str, dict[_CarrierPath, str]],
    namespaces: frozenset[str],
    row_collection_aliases: frozenset[str],
    row_container_aliases: dict[str, frozenset[_CarrierPath]],
) -> str | None:
    if node is None:
        return None
    if isinstance(node, DictNode):
        value = _literal_kwarg_values(node).get(key)
        if value is not None:
            kind = _row_api_alias_expression_kind(value, api_aliases, row_api_container_aliases)
            if kind is None:
                kind = _row_api_dynamic_access_kind(value, namespaces, row_collection_aliases, row_container_aliases)
            return kind
        dynamic_key_kinds: list[str] = []
        for pair in node.items:
            if isinstance(pair.key, Const) and isinstance(pair.key.value, str):
                continue
            kind = _row_api_alias_expression_kind(pair.value, api_aliases, row_api_container_aliases)
            if kind is None:
                kind = _row_api_dynamic_access_kind(pair.value, namespaces, row_collection_aliases, row_container_aliases)
            if kind is not None:
                dynamic_key_kinds.append(kind)
        return _merge_row_api_kinds(dynamic_key_kinds)
    access_path = _row_api_container_access_path(node)
    if access_path is None:
        dynamic_access_pattern = _row_api_dynamic_container_access_pattern(node)
        if dynamic_access_pattern is None:
            return None
        base_name, path_pattern = dynamic_access_pattern
        entries = row_api_container_aliases.get(base_name)
        if not entries:
            return None
        return _merge_row_api_kinds(_carrier_pattern_value_kinds(entries, (*path_pattern, key)))
    base_name, path = access_path
    entries = row_api_container_aliases.get(base_name)
    if not entries:
        return None
    return entries.get((*path, key))


def _callblock_row_splat_targets(
    ast: Node,
    macros: dict[str, Macro],
    macro_aliases: _MacroAliases,
    macro_container_aliases: _MacroContainerAliases,
    row_collection_aliases: frozenset[str],
    row_container_aliases: dict[str, frozenset[_CarrierPath]],
) -> list[str]:
    targets: list[str] = []
    for node in ast.find_all(CallBlock):
        for macro_name in _macro_names_for_callee(node.call.node, macros, macro_aliases, macro_container_aliases):
            macro = macros[macro_name]
            for caller_call in macro.find_all(Call):
                if not isinstance(caller_call.node, Name) or caller_call.node.name != "caller":
                    continue
                if _iter_may_yield_row_object(caller_call.dyn_args, frozenset(), row_collection_aliases, row_container_aliases):
                    explicit_count = len(caller_call.args)
                    targets.extend(target.name for target in node.args[explicit_count:] if isinstance(target, Name))
                if _node_references_name(caller_call.dyn_kwargs, frozenset(row_container_aliases)):
                    targets.extend(target.name for target in node.args if isinstance(target, Name))
    return targets


def _callblock_api_splat_targets(
    ast: Node,
    macros: dict[str, Macro],
    macro_aliases: _MacroAliases,
    macro_container_aliases: _MacroContainerAliases,
    api_aliases: dict[str, str],
    row_api_container_aliases: dict[str, dict[_CarrierPath, str]],
    namespaces: frozenset[str],
    row_collection_aliases: frozenset[str],
    row_container_aliases: dict[str, frozenset[_CarrierPath]],
) -> list[tuple[str, str]]:
    targets: list[tuple[str, str]] = []
    for node in ast.find_all(CallBlock):
        for macro_name in _macro_names_for_callee(node.call.node, macros, macro_aliases, macro_container_aliases):
            macro = macros[macro_name]
            for caller_call in macro.find_all(Call):
                if not isinstance(caller_call.node, Name) or caller_call.node.name != "caller":
                    continue
                star_kind = _iter_may_yield_row_api_kind(caller_call.dyn_args, api_aliases, row_api_container_aliases)
                if star_kind is not None:
                    explicit_count = len(caller_call.args)
                    targets.extend((target.name, star_kind) for target in node.args[explicit_count:] if isinstance(target, Name))
                for target in node.args:
                    keyword_kind = _row_api_mapping_key_kind(
                        caller_call.dyn_kwargs,
                        target.name,
                        api_aliases,
                        row_api_container_aliases,
                        namespaces,
                        row_collection_aliases,
                        row_container_aliases,
                    )
                    if keyword_kind is not None:
                        targets.append((target.name, keyword_kind))
    return targets


def _literal_star_values(node: Node | None) -> list[Node]:
    if isinstance(node, (List, Tuple)):
        return list(node.items)
    return []


def _has_unknown_star_values(node: Node | None) -> bool:
    return node is not None and not isinstance(node, (List, Tuple))


def _has_unknown_kwarg_values(node: Node | None) -> bool:
    if node is None:
        return False
    if not isinstance(node, DictNode):
        return True
    return any(not (isinstance(pair.key, Const) and isinstance(pair.key.value, str)) for pair in node.items)


def _literal_kwarg_values(node: DictNode) -> dict[str, Node]:
    values: dict[str, Node] = {}
    for pair in node.items:
        if isinstance(pair.key, Const) and isinstance(pair.key.value, str):
            values[pair.key.value] = pair.value
    return values


def _row_api_dynamic_access_kind(
    node: Node,
    namespaces: frozenset[str],
    row_collection_aliases: frozenset[str],
    row_container_aliases: dict[str, frozenset[_CarrierPath]],
) -> str | None:
    if (
        isinstance(node, Getattr)
        and _node_may_be_row_receiver(node.node, namespaces, row_collection_aliases, row_container_aliases)
        and node.attr in _PIPELINE_ROW_API_NAMES
    ):
        return "get" if node.attr == "get" else ROW_API_DYNAMIC_ACCESS
    return None


def _row_api_container_entries(
    node: Node,
    api_aliases: dict[str, str],
    row_api_container_aliases: dict[str, dict[_CarrierPath, str]],
    namespaces: frozenset[str],
    row_collection_aliases: frozenset[str],
    row_container_aliases: dict[str, frozenset[_CarrierPath]],
) -> dict[_CarrierPath, str]:
    if isinstance(node, Name):
        return dict(row_api_container_aliases.get(node.name, {}))
    access_path = _row_api_container_access_path(node)
    if access_path is not None:
        base_name, path = access_path
        base_entries = row_api_container_aliases.get(base_name)
        if base_entries:
            return _carrier_relative_entries(base_entries, path)
    dynamic_access_pattern = _row_api_dynamic_container_access_pattern(node)
    if dynamic_access_pattern is not None:
        base_name, path_pattern = dynamic_access_pattern
        base_entries = row_api_container_aliases.get(base_name)
        if base_entries:
            return _carrier_pattern_relative_entries(base_entries, path_pattern)
    if isinstance(node, DictNode):
        entries: dict[_CarrierPath, str] = {}
        for pair in node.items:
            if not (isinstance(pair.key, Const) and isinstance(pair.key.value, str)):
                continue
            dict_key_path: _CarrierPath = (pair.key.value,)
            for child_path, child_kind in _row_api_container_entries(
                pair.value, api_aliases, row_api_container_aliases, namespaces, row_collection_aliases, row_container_aliases
            ).items():
                entries[dict_key_path + child_path] = child_kind
            kind = _row_api_alias_expression_kind(pair.value, api_aliases, row_api_container_aliases)
            if kind is None:
                kind = _row_api_dynamic_access_kind(pair.value, namespaces, row_collection_aliases, row_container_aliases)
            if kind is not None:
                entries[dict_key_path] = kind
        return entries
    if isinstance(node, (List, Tuple)):
        list_entries: dict[_CarrierPath, str] = {}
        for index, item in enumerate(node.items):
            list_key_path: _CarrierPath = (index,)
            for child_path, child_kind in _row_api_container_entries(
                item, api_aliases, row_api_container_aliases, namespaces, row_collection_aliases, row_container_aliases
            ).items():
                list_entries[list_key_path + child_path] = child_kind
            kind = _row_api_alias_expression_kind(item, api_aliases, row_api_container_aliases)
            if kind is None:
                kind = _row_api_dynamic_access_kind(item, namespaces, row_collection_aliases, row_container_aliases)
            if kind is not None:
                list_entries[list_key_path] = kind
        return list_entries
    if isinstance(node, Call) and isinstance(node.node, Name) and node.node.name == "namespace":
        namespace_entries: dict[_CarrierPath, str] = {}
        for keyword in node.kwargs:
            namespace_key_path: _CarrierPath = (keyword.key,)
            for child_path, child_kind in _row_api_container_entries(
                keyword.value, api_aliases, row_api_container_aliases, namespaces, row_collection_aliases, row_container_aliases
            ).items():
                namespace_entries[namespace_key_path + child_path] = child_kind
            kind = _row_api_alias_expression_kind(keyword.value, api_aliases, row_api_container_aliases)
            if kind is None:
                kind = _row_api_dynamic_access_kind(keyword.value, namespaces, row_collection_aliases, row_container_aliases)
            if kind is not None:
                namespace_entries[namespace_key_path] = kind
        return namespace_entries
    return {}


def _row_api_alias_expression_kind(
    node: Node,
    api_aliases: dict[str, str],
    row_api_container_aliases: dict[str, dict[_CarrierPath, str]],
) -> str | None:
    if isinstance(node, Name):
        return api_aliases.get(node.name)
    access_path = _row_api_container_access_path(node)
    if access_path is not None:
        base_name, path = access_path
        entries = row_api_container_aliases.get(base_name)
        if not entries:
            return None
        if path:
            return entries.get(path)
        return _merge_row_api_kinds(entries.values())
    dynamic_access_pattern = _row_api_dynamic_container_access_pattern(node)
    if dynamic_access_pattern is not None:
        base_name, path_pattern = dynamic_access_pattern
        entries = row_api_container_aliases.get(base_name)
        if entries:
            return _merge_row_api_kinds(_carrier_pattern_value_kinds(entries, path_pattern))
    return None


def _row_api_container_access_path(node: Node) -> tuple[str, _CarrierPath] | None:
    if isinstance(node, Name):
        return node.name, ()
    if isinstance(node, Getattr):
        parent_path = _row_api_container_access_path(node.node)
        if parent_path is None:
            return None
        base_name, path = parent_path
        return base_name, (*path, node.attr)
    if isinstance(node, Getitem) and isinstance(node.arg, Const) and isinstance(node.arg.value, (str, int)):
        parent_path = _row_api_container_access_path(node.node)
        if parent_path is None:
            return None
        base_name, path = parent_path
        return base_name, (*path, node.arg.value)
    return None


def _row_api_container_access_pattern(node: Node) -> tuple[str, _CarrierPathPattern] | None:
    if isinstance(node, Name):
        return node.name, ()
    if isinstance(node, Getattr):
        parent_path = _row_api_container_access_pattern(node.node)
        if parent_path is None:
            return None
        base_name, path = parent_path
        return base_name, (*path, node.attr)
    if isinstance(node, Getitem):
        parent_path = _row_api_container_access_pattern(node.node)
        if parent_path is None:
            return None
        base_name, path = parent_path
        if isinstance(node.arg, Const) and isinstance(node.arg.value, (str, int)):
            return base_name, (*path, node.arg.value)
        return base_name, (*path, None)
    return None


def _row_api_dynamic_container_access_pattern(node: Node) -> tuple[str, _CarrierPathPattern] | None:
    access_pattern = _row_api_container_access_pattern(node)
    if access_pattern is None:
        return None
    base_name, path_pattern = access_pattern
    if None not in path_pattern:
        return None
    return base_name, path_pattern


def _carrier_pattern_value_kinds(entries: dict[_CarrierPath, str], path_pattern: _CarrierPathPattern) -> list[str]:
    return [kind for entry_path, kind in entries.items() if _carrier_path_matches_pattern(entry_path, path_pattern)]


def _carrier_pattern_value_macro_names(
    entries: dict[_CarrierPath, frozenset[str]], path_pattern: _CarrierPathPattern
) -> list[frozenset[str]]:
    return [names for entry_path, names in entries.items() if _carrier_path_matches_pattern(entry_path, path_pattern)]


def _carrier_pattern_child_kinds(entries: dict[_CarrierPath, str], path_pattern: _CarrierPathPattern) -> list[str]:
    return [kind for entry_path, kind in entries.items() if _carrier_path_has_pattern_child(entry_path, path_pattern)]


def _carrier_pattern_relative_entries(entries: dict[_CarrierPath, str], path_pattern: _CarrierPathPattern) -> dict[_CarrierPath, str]:
    return {
        entry_path[len(path_pattern) :]: kind
        for entry_path, kind in entries.items()
        if _carrier_path_matches_pattern_prefix(entry_path, path_pattern)
    }


def _carrier_path_matches_pattern(entry_path: _CarrierPath, path_pattern: _CarrierPathPattern) -> bool:
    return len(entry_path) == len(path_pattern) and _carrier_path_matches_pattern_prefix(entry_path, path_pattern)


def _carrier_path_matches_pattern_prefix(entry_path: _CarrierPath, path_pattern: _CarrierPathPattern) -> bool:
    if len(entry_path) < len(path_pattern):
        return False
    return all(
        pattern_segment is None or pattern_segment == entry_segment
        for entry_segment, pattern_segment in zip(entry_path, path_pattern, strict=False)
    )


def _carrier_child_kinds(entries: dict[_CarrierPath, str], path: _CarrierPath) -> list[str]:
    return [kind for entry_path, kind in entries.items() if _carrier_path_has_child(entry_path, path)]


def _carrier_relative_entries(entries: dict[_CarrierPath, str], path: _CarrierPath) -> dict[_CarrierPath, str]:
    return {entry_path[len(path) :]: kind for entry_path, kind in entries.items() if entry_path[: len(path)] == path}


def _carrier_relative_macro_entries(entries: dict[_CarrierPath, frozenset[str]], path: _CarrierPath) -> dict[_CarrierPath, frozenset[str]]:
    return {entry_path[len(path) :]: names for entry_path, names in entries.items() if entry_path[: len(path)] == path}


def _carrier_pattern_relative_macro_entries(
    entries: dict[_CarrierPath, frozenset[str]], path_pattern: _CarrierPathPattern
) -> dict[_CarrierPath, frozenset[str]]:
    return {
        entry_path[len(path_pattern) :]: names
        for entry_path, names in entries.items()
        if _carrier_path_matches_pattern_prefix(entry_path, path_pattern)
    }


def _carrier_has_child_path(paths: frozenset[_CarrierPath], path: _CarrierPath) -> bool:
    return any(_carrier_path_has_child(entry_path, path) for entry_path in paths)


def _carrier_pattern_has_child_path(paths: frozenset[_CarrierPath], path_pattern: _CarrierPathPattern) -> bool:
    return any(_carrier_path_has_pattern_child(entry_path, path_pattern) for entry_path in paths)


def _carrier_path_has_child(entry_path: _CarrierPath, path: _CarrierPath) -> bool:
    return len(entry_path) > len(path) and entry_path[: len(path)] == path and isinstance(entry_path[len(path)], int)


def _carrier_path_has_pattern_child(entry_path: _CarrierPath, path_pattern: _CarrierPathPattern) -> bool:
    return (
        len(entry_path) > len(path_pattern)
        and _carrier_path_matches_pattern_prefix(entry_path, path_pattern)
        and isinstance(entry_path[len(path_pattern)], int)
    )


def _merge_row_api_kinds(kinds: Iterable[str]) -> str | None:
    unique_kinds = set(kinds)
    if not unique_kinds:
        return None
    if len(unique_kinds) == 1:
        return next(iter(unique_kinds))
    return ROW_API_DYNAMIC_ACCESS


def _merge_macro_names(name_sets: Iterable[frozenset[str]]) -> frozenset[str]:
    names: set[str] = set()
    for name_set in name_sets:
        names.update(name_set)
    return frozenset(names)


def _row_object_container_paths(
    node: Node,
    namespaces: frozenset[str],
    row_collection_aliases: frozenset[str],
    row_container_aliases: dict[str, frozenset[_CarrierPath]],
) -> set[_CarrierPath]:
    if isinstance(node, Name):
        if node.name in row_collection_aliases:
            return {(0,)}
        return set(row_container_aliases.get(node.name, ()))
    access_path = _row_api_container_access_path(node)
    if access_path is not None:
        base_name, path = access_path
        entries = row_container_aliases.get(base_name)
        if entries:
            return {entry_path[len(path) :] for entry_path in entries if entry_path[: len(path)] == path}
    dynamic_access_pattern = _row_api_dynamic_container_access_pattern(node)
    if dynamic_access_pattern is not None:
        base_name, path_pattern = dynamic_access_pattern
        entries = row_container_aliases.get(base_name)
        if entries:
            return {
                entry_path[len(path_pattern) :] for entry_path in entries if _carrier_path_matches_pattern_prefix(entry_path, path_pattern)
            }
    if isinstance(node, DictNode):
        paths: set[_CarrierPath] = set()
        for pair in node.items:
            if not (isinstance(pair.key, Const) and isinstance(pair.key.value, str)):
                continue
            dict_key_path: _CarrierPath = (pair.key.value,)
            if _node_is_row_object_expression(pair.value, namespaces, row_collection_aliases, row_container_aliases):
                paths.add(dict_key_path)
            paths.update(
                dict_key_path + child_path
                for child_path in _row_object_container_paths(pair.value, namespaces, row_collection_aliases, row_container_aliases)
            )
        return paths
    if isinstance(node, (List, Tuple)):
        list_paths: set[_CarrierPath] = set()
        for index, item in enumerate(node.items):
            list_key_path: _CarrierPath = (index,)
            if _node_is_row_object_expression(item, namespaces, row_collection_aliases, row_container_aliases):
                list_paths.add(list_key_path)
            list_paths.update(
                list_key_path + child_path
                for child_path in _row_object_container_paths(item, namespaces, row_collection_aliases, row_container_aliases)
            )
        return list_paths
    if isinstance(node, Call) and isinstance(node.node, Name) and node.node.name == "namespace":
        namespace_paths: set[_CarrierPath] = set()
        for keyword in node.kwargs:
            namespace_key_path: _CarrierPath = (keyword.key,)
            if _node_is_row_object_expression(keyword.value, namespaces, row_collection_aliases, row_container_aliases):
                namespace_paths.add(namespace_key_path)
            namespace_paths.update(
                namespace_key_path + child_path
                for child_path in _row_object_container_paths(keyword.value, namespaces, row_collection_aliases, row_container_aliases)
            )
        return namespace_paths
    return set()


def _row_object_container_access_matches(
    node: Node,
    row_container_aliases: dict[str, frozenset[_CarrierPath]],
) -> bool:
    access_path = _row_api_container_access_path(node)
    if access_path is not None:
        base_name, path = access_path
        if not path:
            return False
        return path in row_container_aliases.get(base_name, frozenset())
    dynamic_access_pattern = _row_api_dynamic_container_access_pattern(node)
    if dynamic_access_pattern is None:
        return False
    base_name, path_pattern = dynamic_access_pattern
    if not path_pattern:
        return False
    return any(_carrier_path_matches_pattern(entry_path, path_pattern) for entry_path in row_container_aliases.get(base_name, frozenset()))


def _node_may_be_row_receiver(
    node: Node | None,
    namespaces: frozenset[str],
    row_collection_aliases: frozenset[str],
    row_container_aliases: dict[str, frozenset[_CarrierPath]],
) -> bool:
    if node is None:
        return False
    if isinstance(node, Name):
        return node.name in namespaces
    if isinstance(node, Getitem) and _iter_may_yield_row_object(node.node, namespaces, row_collection_aliases, row_container_aliases):
        return True
    if isinstance(node, (Getattr, Getitem)):
        return _row_object_container_access_matches(node, row_container_aliases)
    if (
        isinstance(node, Call)
        and isinstance(node.node, Getattr)
        and _node_may_be_row_receiver(node.node.node, namespaces, row_collection_aliases, row_container_aliases)
    ):
        return False
    if isinstance(node, Filter):
        if node.name == "attr" and _node_may_be_row_receiver(node.node, namespaces, row_collection_aliases, row_container_aliases):
            return False
        if node.name in {"first", "last", "random"} and _iter_may_yield_row_object(
            node.node, namespaces, row_collection_aliases, row_container_aliases
        ):
            return True
    return (
        _node_references_namespace(node, namespaces)
        or _node_references_name(node, row_collection_aliases)
        or _node_references_name(node, frozenset(row_container_aliases))
    )


def _node_is_row_object_expression(
    node: Node,
    namespaces: frozenset[str],
    row_collection_aliases: frozenset[str],
    row_container_aliases: dict[str, frozenset[_CarrierPath]],
) -> bool:
    if isinstance(node, DictNode):
        return False
    if isinstance(node, (List, Tuple)):
        return False
    if isinstance(node, Call) and isinstance(node.node, Name) and node.node.name == "namespace":
        return False
    if isinstance(node, Name):
        return node.name in namespaces
    if isinstance(node, Getitem) and _iter_may_yield_row_object(node.node, namespaces, row_collection_aliases, row_container_aliases):
        return True
    if isinstance(node, (Getattr, Getitem)):
        return _row_object_container_access_matches(node, row_container_aliases)
    if (
        isinstance(node, Call)
        and isinstance(node.node, Getattr)
        and _node_may_be_row_receiver(node.node.node, namespaces, row_collection_aliases, row_container_aliases)
    ):
        return False
    if isinstance(node, Filter):
        if node.name == "attr" and _node_may_be_row_receiver(node.node, namespaces, row_collection_aliases, row_container_aliases):
            return False
        if node.name in {"first", "last", "random"} and _iter_may_yield_row_object(
            node.node, namespaces, row_collection_aliases, row_container_aliases
        ):
            return True
    return any(
        _node_is_row_object_expression(child, namespaces, row_collection_aliases, row_container_aliases)
        for child in node.iter_child_nodes()
    )


def _node_is_row_collection_expression(
    node: Node,
    namespaces: frozenset[str],
    row_collection_aliases: frozenset[str],
    row_container_aliases: dict[str, frozenset[_CarrierPath]],
) -> bool:
    if isinstance(node, Name):
        return node.name in row_collection_aliases
    if isinstance(node, (List, Tuple)):
        return any(_node_is_row_object_expression(item, namespaces, row_collection_aliases, row_container_aliases) for item in node.items)
    if isinstance(node, Filter) and node.name not in {"map", "first", "last", "random"}:
        return _iter_may_yield_row_object(node.node, namespaces, row_collection_aliases, row_container_aliases)
    return False


def _node_is_row_container_expression(
    node: Node,
    namespaces: frozenset[str],
    row_collection_aliases: frozenset[str],
    row_container_aliases: dict[str, frozenset[_CarrierPath]],
) -> bool:
    if isinstance(node, DictNode):
        return bool(_row_object_container_paths(node, namespaces, row_collection_aliases, row_container_aliases))
    if isinstance(node, Call) and isinstance(node.node, Name) and node.node.name == "namespace":
        return bool(_row_object_container_paths(node, namespaces, row_collection_aliases, row_container_aliases))
    if isinstance(node, Name):
        return node.name in row_container_aliases
    return False


def _node_references_namespace(node: Node | None, namespaces: frozenset[str]) -> bool:
    if node is None:
        return False
    if isinstance(node, Name):
        return node.name in namespaces
    return any(_node_references_namespace(child, namespaces) for child in node.iter_child_nodes())


def _node_references_row_value_alias(node: Node | None, row_value_aliases: frozenset[str]) -> bool:
    if node is None:
        return False
    if isinstance(node, Name):
        return node.name in row_value_aliases
    return any(_node_references_row_value_alias(child, row_value_aliases) for child in node.iter_child_nodes())


def _node_references_tracked_row(
    node: Node | None,
    namespaces: frozenset[str],
    row_value_aliases: frozenset[str],
    row_collection_aliases: frozenset[str],
    row_container_aliases: dict[str, frozenset[_CarrierPath]],
) -> bool:
    if node is None:
        return False
    return (
        _node_references_namespace(node, namespaces)
        or _node_references_row_value_alias(node, row_value_aliases)
        or _node_references_name(node, row_collection_aliases)
        or _node_references_name(node, frozenset(row_container_aliases))
        or _node_is_row_object_expression(node, namespaces, row_collection_aliases, row_container_aliases)
        or _node_is_row_collection_expression(node, namespaces, row_collection_aliases, row_container_aliases)
        or _node_is_row_container_expression(node, namespaces, row_collection_aliases, row_container_aliases)
    )


def _node_references_name(node: Node | None, names: frozenset[str]) -> bool:
    if node is None:
        return False
    if isinstance(node, Name):
        return node.name in names
    return any(_node_references_name(child, names) for child in node.iter_child_nodes())


def _is_blocked_attr_filter_name(value: str) -> bool:
    return value.startswith("_") or value in _PIPELINE_ROW_API_NAMES


def _is_blocked_row_attribute_name(value: str) -> bool:
    return value.startswith("_") or (value in _PIPELINE_ROW_API_NAMES and value != "get")


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
    namespaces, api_aliases, row_api_container_aliases, row_value_aliases, row_collection_aliases, row_container_aliases = (
        _field_extraction_context(ast, namespace)
    )
    fields: dict[str, list[str]] = {}

    def append_access(field_name: str, access_type: str) -> None:
        if field_name in fields:
            fields[field_name].append(access_type)
            return
        fields[field_name] = [access_type]

    def append_dynamic_access(access_type: str) -> None:
        if access_type not in fields.get(DYNAMIC_ROW_FIELD, []):
            append_access(DYNAMIC_ROW_FIELD, access_type)

    def walk(node: Node) -> None:
        if isinstance(node, Call):
            alias_kind = _row_api_alias_expression_kind(node.node, api_aliases, row_api_container_aliases)
            if alias_kind is not None:
                append_dynamic_access(f"{alias_kind}_dynamic")

        if (
            isinstance(node, Call)
            and isinstance(node.node, Getattr)
            and _node_may_be_row_receiver(node.node.node, namespaces, row_collection_aliases, row_container_aliases)
            and node.node.attr == "get"
        ):
            key_arg = _call_positional_or_keyword_value(node, 0, "key")
            if isinstance(key_arg, Const) and isinstance(key_arg.value, str):
                append_access(key_arg.value, "item")
            elif (
                key_arg is not None
                or _has_unknown_star_values(node.dyn_args)
                or _has_unknown_kwarg_values(node.dyn_kwargs)
                or _node_references_tracked_row(node.dyn_args, namespaces, row_value_aliases, row_collection_aliases, row_container_aliases)
                or _node_references_tracked_row(
                    node.dyn_kwargs, namespaces, row_value_aliases, row_collection_aliases, row_container_aliases
                )
            ):
                append_dynamic_access("get_dynamic")

        if (
            isinstance(node, Call)
            and isinstance(node.node, Getattr)
            and _node_may_be_row_receiver(node.node.node, namespaces, row_collection_aliases, row_container_aliases)
            and node.node.attr in _PIPELINE_ROW_API_NAMES
            and node.node.attr != "get"
        ):
            append_dynamic_access("row_api_dynamic")

        if (
            isinstance(node, Getattr)
            and _node_may_be_row_receiver(node.node, namespaces, row_collection_aliases, row_container_aliases)
            and _is_blocked_row_attribute_name(node.attr)
        ):
            append_dynamic_access("row_api_dynamic")

        if (
            isinstance(node, Getattr)
            and _node_may_be_row_receiver(node.node, namespaces, row_collection_aliases, row_container_aliases)
            and not _is_blocked_attr_filter_name(node.attr)
        ):
            append_access(node.attr, "attr")

        if isinstance(node, Getitem) and _node_may_be_row_receiver(node.node, namespaces, row_collection_aliases, row_container_aliases):
            if isinstance(node.arg, Const) and isinstance(node.arg.value, str):
                append_access(node.arg.value, "item")
            else:
                append_dynamic_access("item_dynamic")

        if (
            isinstance(node, Filter)
            and node.name == "attr"
            and _node_may_be_row_receiver(node.node, namespaces, row_collection_aliases, row_container_aliases)
        ):
            attr_arg = _filter_positional_or_keyword_value(node, 0, "name")
            if attr_arg is not None and isinstance(attr_arg, Const) and isinstance(attr_arg.value, str):
                if _is_blocked_attr_filter_name(attr_arg.value):
                    append_dynamic_access("attr_dynamic")
                elif _node_may_be_row_receiver(node.node, namespaces, row_collection_aliases, row_container_aliases):
                    append_access(attr_arg.value, "attr")
            elif (
                attr_arg is not None
                or _has_unknown_star_values(node.dyn_args)
                or _has_unknown_kwarg_values(node.dyn_kwargs)
                or _node_references_tracked_row(node.dyn_args, namespaces, row_value_aliases, row_collection_aliases, row_container_aliases)
                or _node_references_tracked_row(
                    node.dyn_kwargs, namespaces, row_value_aliases, row_collection_aliases, row_container_aliases
                )
            ):
                append_dynamic_access("attr_dynamic")
        if isinstance(node, Filter):
            attribute_arg = _attribute_resolving_filter_argument(node)
            dynamic_splat = (
                _has_unknown_star_values(node.dyn_args)
                or _has_unknown_kwarg_values(node.dyn_kwargs)
                or _node_references_tracked_row(node.dyn_args, namespaces, row_value_aliases, row_collection_aliases, row_container_aliases)
                or _node_references_tracked_row(
                    node.dyn_kwargs, namespaces, row_value_aliases, row_collection_aliases, row_container_aliases
                )
            )
            maps_row_objects = _iter_may_yield_row_object(node.node, namespaces, row_collection_aliases, row_container_aliases)
            if attribute_arg is None:
                if dynamic_splat and (
                    maps_row_objects or _node_may_be_row_receiver(node.node, namespaces, row_collection_aliases, row_container_aliases)
                ):
                    append_dynamic_access("map_attribute_dynamic")
            else:
                if maps_row_objects and isinstance(attribute_arg, Const) and isinstance(attribute_arg.value, str):
                    if _is_blocked_attr_filter_name(attribute_arg.value):
                        append_dynamic_access("row_api_dynamic")
                    else:
                        append_access(attribute_arg.value, "attr")
                    return
                if dynamic_splat or (
                    not (isinstance(attribute_arg, Const) and isinstance(attribute_arg.value, str))
                    and (
                        maps_row_objects
                        or _node_may_be_row_receiver(node.node, namespaces, row_collection_aliases, row_container_aliases)
                        or _node_references_namespace(attribute_arg, namespaces)
                        or _node_references_row_value_alias(attribute_arg, row_value_aliases)
                    )
                ):
                    append_dynamic_access("map_attribute_dynamic")

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
