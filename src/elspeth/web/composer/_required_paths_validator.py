"""Schema-required argument-path walker for composer tool invocations.

Compiles the JSON-Schema ``required`` declarations of each composer tool's
``parameters`` block into a tuple of ``_CompiledRequiredPath`` records, then
walks LLM-supplied tool arguments against that compiled set to surface
missing-required-field diagnostics before the tool is dispatched.

The compilation happens once at import time and is cached on
``_TOOL_REQUIRED_PATHS``. The runtime walker (``_find_missing_required_paths``)
is invoked per tool call by the compose loop.

Tier discipline: the schemas come from ``get_tool_definitions()`` in
``tools.py`` — system-owned metadata, so missing/malformed entries crash
at import time (intentional Tier-1 read-side discipline). The values being
validated come from the LLM (Tier 3 boundary), so the walker only inspects
shape (presence/absence of keys); it does not coerce or validate types.

Module-internal name (underscore prefix on the filename) marks this module
as a package-internal validator — external callers should not import from it
unless they own the contract.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import cast

from elspeth.web.composer.tools import get_tool_definitions

__all__ = [
    "_ARRAY_ITEM_SEGMENT",
    "_TOOL_REQUIRED_PATHS",
    "_CompiledRequiredPath",
    "_collect_required_paths",
    "_find_missing_required_paths",
    "_optional_ancestor_present",
]

_ARRAY_ITEM_SEGMENT = "[]"

type RequiredPath = tuple[str, ...]


@dataclass(frozen=True, slots=True)
class _CompiledRequiredPath:
    """One schema-required path with conditional-on-presence semantics.

    JSON-Schema required semantics: a nested object's ``required`` list
    only applies when that object itself is present. The ``optional_ancestor``
    field captures the deepest path segment whose containing object is
    optional at its parent level — when any ancestor on that path is absent
    in the value, the inner ``path`` check short-circuits.

    For paths that are required at every level of the tree,
    ``optional_ancestor`` is the empty tuple and the path is enforced
    unconditionally.
    """

    path: RequiredPath
    optional_ancestor: RequiredPath = ()


def _collect_required_paths(
    schema: Mapping[str, object],
    prefix: RequiredPath = (),
    optional_ancestor: RequiredPath = (),
) -> tuple[_CompiledRequiredPath, ...]:
    """Compile schema-declared required fields into compiled-path records.

    The schema tree is system-owned tool metadata, so direct key access is
    intentional: a malformed tool definition should crash at import time.

    Each emitted :class:`_CompiledRequiredPath` carries the deepest optional
    ancestor seen on the way down. When the walker descends into a property
    that is NOT in the parent's ``required`` list, that property becomes the
    new ``optional_ancestor`` for everything emitted below it; the validator
    then short-circuits the inner check when that ancestor is absent in the
    value (correct JSON-Schema semantics: nested ``required`` applies only
    when the parent is present). Required-at-every-level paths keep an empty
    ``optional_ancestor`` and are enforced unconditionally.
    """
    schema_type = cast(str, schema["type"])

    if schema_type == "object":
        compiled: list[_CompiledRequiredPath] = []
        required_fields: set[str] = set()
        if "required" in schema:
            raw_required = cast(list[str], schema["required"])
            required_fields = set(raw_required)
            for field in raw_required:
                compiled.append(_CompiledRequiredPath(path=(*prefix, field), optional_ancestor=optional_ancestor))
        if "properties" in schema:
            properties = cast(Mapping[str, Mapping[str, object]], schema["properties"])
            for key, child_schema in properties.items():
                child_prefix = (*prefix, key)
                # If this property is NOT required at the current level, it
                # becomes the deepest optional ancestor for any nested-required
                # paths emitted below. If it IS required, propagate whatever
                # ancestor we already had (which is itself a required-at-this
                # -level path or empty).
                child_ancestor = optional_ancestor if key in required_fields else child_prefix
                compiled.extend(_collect_required_paths(child_schema, child_prefix, child_ancestor))
        return tuple(compiled)

    if schema_type == "array" and "items" in schema:
        item_schema = cast(Mapping[str, object], schema["items"])
        # Array items inherit the array's optional_ancestor: required fields
        # inside an item only matter if the array itself is present (and per
        # _find_missing_path_instances semantics, an empty array produces no
        # missing-path entries).
        return _collect_required_paths(item_schema, (*prefix, _ARRAY_ITEM_SEGMENT), optional_ancestor)

    return ()


def _build_tool_required_paths_index() -> dict[str, tuple[_CompiledRequiredPath, ...]]:
    """Build a lookup of required argument paths per tool definition."""
    index: dict[str, tuple[_CompiledRequiredPath, ...]] = {}
    for defn in get_tool_definitions():
        parameters = cast(Mapping[str, object], defn["parameters"])
        index[defn["name"]] = _collect_required_paths(parameters)
    return index


def _optional_ancestor_present(value: object, ancestor: RequiredPath) -> bool:
    """Walk down ``value`` along ``ancestor``; return False as soon as a segment is absent.

    Empty ``ancestor`` is the always-required case — treated as present.

    Today ``_collect_required_paths`` only sets ``optional_ancestor`` to a new
    path when descending into an OBJECT property that's not in ``required``;
    descending into array items propagates the existing ancestor unchanged.
    So array segments never appear in ``optional_ancestor`` under the current
    schema set. A future schema with an optional sub-object inside array items
    (e.g., ``tags: array<{ details?: { name: required } }>``) WOULD produce
    such an ancestor, and the all-or-nothing semantics here can't express
    "present in some items, absent in others." Per CLAUDE.md offensive
    programming: crash loudly with a diagnostic that points the maintainer at
    the extension site, rather than silently producing wrong validation.
    """
    if not ancestor:
        return True
    cursor: object = value
    for segment in ancestor:
        if segment == _ARRAY_ITEM_SEGMENT:
            raise NotImplementedError(
                "Array-segment in optional_ancestor is not yet supported by this walker. "
                f"Saw ancestor={ancestor!r}. To handle optional sub-objects inside array "
                "items, extend _find_missing_required_paths to evaluate ancestor presence "
                "per-array-item (rather than once globally) and update this walker to "
                "descend through array segments accordingly."
            )
        if not isinstance(cursor, Mapping) or segment not in cursor:
            return False
        cursor = cursor[segment]
    return True


def _find_missing_path_instances(
    value: object,
    required_path: RequiredPath,
    *,
    current_path: str = "",
) -> list[str]:
    """Return concrete missing-path instances for one required path."""
    if not required_path:
        return []

    head = required_path[0]
    tail = required_path[1:]

    if head == _ARRAY_ITEM_SEGMENT:
        match value:
            case list() as items:
                missing_paths: list[str] = []
                for index, item in enumerate(items):
                    item_path = f"{current_path}[{index}]" if current_path else f"[{index}]"
                    missing_paths.extend(_find_missing_path_instances(item, tail, current_path=item_path))
                return missing_paths
            case _:
                return []

    match value:
        case dict() as mapping:
            next_path = f"{current_path}.{head}" if current_path else head
            if head not in mapping:
                return [next_path]
            return _find_missing_path_instances(mapping[head], tail, current_path=next_path)
        case _:
            return []


def _find_missing_required_paths(
    value: object,
    required_paths: tuple[_CompiledRequiredPath, ...],
) -> list[str]:
    """Return dotted/indexed paths for missing schema-required fields.

    Skips any compiled path whose ``optional_ancestor`` is absent in the
    value: that mirrors JSON-Schema semantics where nested ``required``
    only applies when the containing optional object is itself present.
    """
    missing_paths: list[str] = []
    for compiled in required_paths:
        if not _optional_ancestor_present(value, compiled.optional_ancestor):
            continue
        missing_paths.extend(_find_missing_path_instances(value, compiled.path))
    return missing_paths


# Computed once at import time. Imports of this module trigger
# ``_build_tool_required_paths_index()`` which iterates the tool registry —
# a malformed tool definition will crash at import (intentional).
_TOOL_REQUIRED_PATHS: dict[str, tuple[_CompiledRequiredPath, ...]] = _build_tool_required_paths_index()
