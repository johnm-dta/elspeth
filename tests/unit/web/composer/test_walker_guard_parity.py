"""Behavioural parity: walk_model_schema(M, with_values=False) == walk_model_schema(M, with_values=True)
path-sets for each manifest entry's argument model.

Pins the structural coupling claim: walker and guard cannot diverge in path
enumeration or marker detection because they share one iterator. Closes
rev-2 M_walker_guard_parity (quality MAJOR-3 + systems MINOR-1).
"""

from __future__ import annotations

import pytest

from elspeth.web.composer.redaction import (
    MANIFEST,
    TraversalNode,
    _SensitiveMarker,
    walk_model_schema,
)


@pytest.mark.parametrize(
    "tool_name",
    [name for name, entry in MANIFEST.items() if entry.argument_model is not None],
)
def test_walker_guard_path_sets_are_identical(tool_name: str) -> None:
    entry = MANIFEST[tool_name]
    model = entry.argument_model

    guard_nodes = list(walk_model_schema(model, with_values=False))
    walker_nodes = list(walk_model_schema(model, with_values=True))

    def has_sensitive(node: TraversalNode) -> bool:
        return any(isinstance(m, _SensitiveMarker) for m in node.metadata)

    guard_paths = {(n.path, has_sensitive(n)) for n in guard_nodes}
    walker_paths = {(n.path, has_sensitive(n)) for n in walker_nodes}

    assert guard_paths == walker_paths, (
        f"Walker and guard path-sets diverged for tool {tool_name!r}.\n"
        f"Only in guard: {guard_paths - walker_paths}\n"
        f"Only in walker: {walker_paths - guard_paths}"
    )
