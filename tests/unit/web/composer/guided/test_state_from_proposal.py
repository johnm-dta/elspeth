"""Guided proposal private-pipeline canonicalisation.

The planner's terminal tool schema requires only ``id``/``node_type``/``input``
per node (``plugin``/``on_success``/``on_error``/``options`` optional — a
coalesce is TOLD to omit ``on_success``), and only ``plugin``/``on_success``
per source. The canonical Spec ``from_dict`` constructors are strict, so the
projection adapter must apply the same defaults the freeform candidate
builder applies — otherwise a schema-legal plan dies as
"guided proposal private pipeline is not canonical".
"""

from __future__ import annotations

from elspeth.web.composer.guided.planning import _canonical_state_from_private_pipeline


def _ab_private_pipeline() -> dict[str, object]:
    """A schema-legal fork/coalesce A/B plan exercising every optional key."""
    return {
        "source": {
            "plugin": "csv",
            "on_success": "rows",
            "options": {"path": "blob:00000000-0000-0000-0000-000000000001"},
        },
        "nodes": [
            {
                "id": "fork_gate",
                "node_type": "gate",
                "input": "rows",
                "condition": "True",
                "routes": {"true": "fork", "false": "fork"},
                "fork_to": ["branch_a", "branch_b"],
            },
            {
                "id": "llm_tone",
                "node_type": "transform",
                "plugin": "llm",
                "input": "branch_a",
                "on_success": "toned",
                "on_error": "discard",
                "options": {"provider": "openrouter"},
            },
            {
                "id": "llm_usage",
                "node_type": "transform",
                "plugin": "llm",
                "input": "branch_b",
                "on_success": "used",
                "on_error": "discard",
                "options": {"provider": "openrouter"},
            },
            {
                # The coalesce follows the corrected contract: no on_success
                # (consumed downstream by node id), no plugin, no options.
                "id": "reconcile",
                "node_type": "coalesce",
                "input": "toned",
                "branches": {"branch_a": "toned", "branch_b": "used"},
                "policy": "require_all",
                "merge": "union",
            },
            {
                "id": "cleanup",
                "node_type": "transform",
                "plugin": "field_mapper",
                "input": "reconcile",
                "on_success": "colour_ab_out",
                "on_error": "discard",
                "options": {},
            },
        ],
        "edges": [],
        "outputs": [
            {
                "sink_name": "colour_ab_out",
                "plugin": "json",
                "options": {"path": "out.json"},
                "on_write_failure": "discard",
            },
        ],
    }


def test_schema_optional_node_keys_default_instead_of_failing_canonicalisation() -> None:
    state = _canonical_state_from_private_pipeline(_ab_private_pipeline())
    reconcile = next(node for node in state.nodes if node.id == "reconcile")
    assert reconcile.on_success is None
    assert reconcile.plugin is None
    assert dict(reconcile.options) == {}
    gate = next(node for node in state.nodes if node.id == "fork_gate")
    assert gate.on_success is None
    assert gate.fork_to == ("branch_a", "branch_b")


def test_source_optional_keys_default_instead_of_failing_canonicalisation() -> None:
    raw = _ab_private_pipeline()
    state = _canonical_state_from_private_pipeline(raw)
    assert state.sources["source"].on_validation_failure == "discard"
