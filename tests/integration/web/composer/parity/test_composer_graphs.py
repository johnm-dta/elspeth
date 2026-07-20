"""Discrimination contract for the parity isomorphism helper.

A parity comparator that never fails is worthless: because the matrix drives
every surface with the same scripted arguments, the committed graphs are
near-identical and a vacuous comparator would pass. These tests prove the
helper (1) treats a consistent wire-renaming as isomorphic (canonicalization is
relabeling, not sensitivity), (2) treats same-names/different-topology as NOT
isomorphic, and (3) detects every preserved-attribute mutation the design lists
(gate route, merge mode, failure policy, plugin identity, topology).
"""

from __future__ import annotations

import copy
from typing import Any

import pytest

from tests.helpers.composer_graphs import IsomorphismError, assert_isomorphic, canonical_graph


def _fork_state() -> dict[str, Any]:
    """A rich committed-state dict (gate fork, require-all coalesce, edges)."""
    return {
        "version": 2,
        "metadata": {"name": "fork", "description": "d"},
        "sources": {
            "source": {
                "plugin": "csv",
                "on_success": "gate_in",
                "options": {"path": "/tmp/a/blobs/rows.csv", "schema": {"mode": "observed"}},
                "on_validation_failure": "discard",
            }
        },
        "nodes": [
            {
                "id": "fork_gate",
                "node_type": "gate",
                "plugin": None,
                "input": "gate_in",
                "on_success": None,
                "on_error": None,
                "options": {},
                "condition": "True",
                "routes": {"true": "fork", "false": "fork"},
                "fork_to": ["path_a", "path_b"],
            },
            {
                "id": "merge_results",
                "node_type": "coalesce",
                "plugin": None,
                "input": "path_a",
                "on_success": None,
                "on_error": None,
                "options": {},
                "branches": {"path_a": "path_a", "path_b": "path_b"},
                "policy": "require_all",
                "merge": "union",
            },
            {
                "id": "finalize",
                "node_type": "transform",
                "plugin": "passthrough",
                "input": "merge_results",
                "on_success": "merged",
                "on_error": "discard",
                "options": {"schema": {"mode": "observed"}},
            },
        ],
        "edges": [
            {"id": "e1", "from_node": "source", "to_node": "fork_gate", "edge_type": "on_success", "label": None},
            {"id": "e2", "from_node": "fork_gate", "to_node": "merge_results", "edge_type": "fork", "label": "path_a"},
            {"id": "e3", "from_node": "fork_gate", "to_node": "merge_results", "edge_type": "fork", "label": "path_b"},
            {"id": "e4", "from_node": "merge_results", "to_node": "finalize", "edge_type": "on_success", "label": None},
        ],
        "outputs": [
            {
                "name": "merged",
                "plugin": "json",
                "options": {"path": "outputs/merged.json", "format": "json"},
                "on_write_failure": "discard",
            }
        ],
    }


def _error_routing_state() -> dict[str, Any]:
    """A state whose failure policies route to real sinks (not discard)."""
    return {
        "version": 2,
        "metadata": {"name": "err", "description": ""},
        "sources": {
            "source": {
                "plugin": "csv",
                "on_success": "rows",
                "options": {"path": "records.csv", "schema": {"mode": "observed"}},
                "on_validation_failure": "rejected",
            }
        },
        "nodes": [
            {
                "id": "coerce",
                "node_type": "transform",
                "plugin": "type_coerce",
                "input": "rows",
                "on_success": "clean",
                "on_error": "errors",
                "options": {"schema": {"mode": "observed"}},
            }
        ],
        "edges": [],
        "outputs": [
            {"name": "clean", "plugin": "json", "options": {"path": "clean.json"}, "on_write_failure": "discard"},
            {"name": "errors", "plugin": "json", "options": {"path": "errors.json"}, "on_write_failure": "discard"},
            {"name": "rejected", "plugin": "json", "options": {"path": "rejected.json"}, "on_write_failure": "discard"},
        ],
    }


def _rename_connections(state: dict[str, Any], mapping: dict[str, str]) -> dict[str, Any]:
    """Consistently rename every connection wire (a graph automorphism)."""
    renamed = copy.deepcopy(state)

    def r(value: Any) -> Any:
        return mapping.get(value, value) if isinstance(value, str) else value

    for source in renamed["sources"].values():
        source["on_success"] = r(source["on_success"])
        source["on_validation_failure"] = r(source["on_validation_failure"])
    for node in renamed["nodes"]:
        node["input"] = r(node["input"])
        if node.get("on_success") is not None:
            node["on_success"] = r(node["on_success"])
        if node.get("on_error") is not None:
            node["on_error"] = r(node["on_error"])
        if isinstance(node.get("routes"), dict):
            node["routes"] = {k: r(v) for k, v in node["routes"].items()}
        if node.get("fork_to"):
            node["fork_to"] = [r(v) for v in node["fork_to"]]
        if isinstance(node.get("branches"), dict):
            node["branches"] = {k: r(v) for k, v in node["branches"].items()}
    for output in renamed["outputs"]:
        output["name"] = r(output["name"])
        output["on_write_failure"] = r(output["on_write_failure"])
    for edge in renamed["edges"]:
        if edge.get("label") is not None:
            edge["label"] = r(edge["label"])
    return renamed


def test_metadata_and_version_are_canonicalized_away() -> None:
    left = _fork_state()
    right = copy.deepcopy(left)
    right["version"] = 99
    right["metadata"] = {"name": "totally-different", "description": "noise"}
    assert_isomorphic(left, right)


def test_source_and_output_paths_reduce_to_basename() -> None:
    left = _error_routing_state()
    right = copy.deepcopy(left)
    right["sources"]["source"]["options"]["path"] = "/some/other/prefix/blobs/records.csv"
    right["outputs"][0]["options"]["path"] = "/elsewhere/clean.json"
    assert_isomorphic(left, right)


def test_effective_options_drops_explicit_plugin_defaults() -> None:
    """A plugin default made explicit on one side is effective-equal to its absence.

    The guided stage protocol persists the full ``model_dump`` (every default
    explicit); ``set_pipeline`` persists authored-minimal options. Locking this
    direction protects the comparator contract Task 4 inherits: csv
    ``delimiter``/``encoding`` and json ``encoding`` at their model defaults must
    not break isomorphism against a side that omits them.
    """
    left = _error_routing_state()
    right = copy.deepcopy(left)
    right["sources"]["source"]["options"]["delimiter"] = ","  # csv default
    right["sources"]["source"]["options"]["encoding"] = "utf-8"  # csv default
    right["outputs"][0]["options"]["encoding"] = "utf-8"  # json default
    assert_isomorphic(left, right)


@pytest.mark.parametrize(
    "mutate",
    [
        pytest.param(lambda s: s["sources"]["source"]["options"].__setitem__("delimiter", ";"), id="source-non-default"),
        pytest.param(lambda s: s["outputs"][0]["options"].__setitem__("encoding", "latin-1"), id="sink-non-default"),
    ],
)
def test_effective_options_keeps_non_default_plugin_option(mutate: Any) -> None:
    """A genuine non-default source/sink option is preserved and detected.

    The default-dropping normalization must never hide a real config change: a
    non-default value has no matching default to collapse, so it stays in the
    compared form and a divergence raises.
    """
    left = _error_routing_state()
    right = copy.deepcopy(left)
    mutate(right)
    with pytest.raises(IsomorphismError):
        assert_isomorphic(left, right)


def test_consistent_wire_renaming_is_isomorphic() -> None:
    left = _fork_state()
    right = _rename_connections(
        left,
        {
            "gate_in": "z_in",
            "fork": "z_fork",
            "path_a": "branch_alpha",
            "path_b": "branch_beta",
            "merge_results": "z_merge",
            "merged": "z_out",
        },
    )
    # Node ids differ from connection names; also rename a node id + edge refs to
    # prove node-id relabeling too.
    for node in right["nodes"]:
        if node["id"] == "finalize":
            node["id"] = "renamed_finalize"
    for edge in right["edges"]:
        if edge["to_node"] == "finalize":
            edge["to_node"] = "renamed_finalize"
        if edge["from_node"] == "finalize":
            edge["from_node"] = "renamed_finalize"
    assert_isomorphic(left, right)


def test_same_names_different_topology_is_not_isomorphic() -> None:
    left = _fork_state()
    right = copy.deepcopy(left)
    # Rewire finalize to read the raw fork_in connection instead of the coalesce
    # output — same names everywhere, genuinely different dataflow.
    right["nodes"][2]["input"] = "gate_in"
    right["edges"][3]["from_node"] = "fork_gate"
    with pytest.raises(IsomorphismError):
        assert_isomorphic(left, right)


@pytest.mark.parametrize(
    "mutate",
    [
        pytest.param(lambda s: s["nodes"][0]["routes"].__setitem__("true", "path_b"), id="gate-route-target"),
        pytest.param(lambda s: s["nodes"][1].__setitem__("merge", "select"), id="coalesce-merge-mode"),
        pytest.param(lambda s: s["nodes"][1].__setitem__("policy", "require_any"), id="coalesce-policy"),
        pytest.param(lambda s: s["nodes"][2].__setitem__("plugin", "type_coerce"), id="node-plugin-identity"),
        pytest.param(lambda s: s["nodes"][2].__setitem__("node_type", "aggregation"), id="node-kind"),
        pytest.param(lambda s: s["nodes"][2]["options"].__setitem__("schema", {"mode": "fixed"}), id="normalized-options"),
        pytest.param(lambda s: s["nodes"][0].__setitem__("condition", "False"), id="gate-condition"),
    ],
)
def test_preserved_attribute_mutation_is_detected(mutate: Any) -> None:
    left = _fork_state()
    right = copy.deepcopy(left)
    mutate(right)
    with pytest.raises(IsomorphismError):
        assert_isomorphic(left, right)


def test_failure_policy_route_vs_discard_is_detected() -> None:
    # A transform whose on_error routes to a real sink is not the same graph as
    # one that discards — the terminal-vs-routed distinction is preserved.
    left = _error_routing_state()
    right = copy.deepcopy(left)
    right["nodes"][0]["on_error"] = "discard"
    right["outputs"] = [o for o in right["outputs"] if o["name"] != "errors"]
    with pytest.raises(IsomorphismError):
        assert_isomorphic(left, right)


def test_source_validation_failure_route_vs_discard_is_detected() -> None:
    left = _error_routing_state()
    right = copy.deepcopy(left)
    right["sources"]["source"]["on_validation_failure"] = "discard"
    right["outputs"] = [o for o in right["outputs"] if o["name"] != "rejected"]
    with pytest.raises(IsomorphismError):
        assert_isomorphic(left, right)


def test_identical_graph_is_isomorphic_to_itself() -> None:
    left = _fork_state()
    graph = canonical_graph(left)
    assert graph.fingerprint == canonical_graph(copy.deepcopy(left)).fingerprint
    assert_isomorphic(graph, graph)
