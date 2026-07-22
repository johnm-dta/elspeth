"""Closed, redacted wire contract for guided pipeline proposals."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import pytest

from elspeth.web.composer.guided.protocol import (
    PROPOSAL_RATIONALE_TEMPLATE,
    PROPOSAL_SUMMARY_TEMPLATE,
    GuidedStep,
    TurnType,
    legal_turn_types_for,
    proposal_blocker_summary,
    proposal_component_label,
    proposal_structural_label,
    validate_payload,
    validate_proposal_catalog_refs,
)

PROPOSAL_ID = "00000000-0000-4000-8000-000000000401"
SOURCE_ID = "00000000-0000-4000-8000-000000000402"
EDGE_ID = "00000000-0000-4000-8000-000000000403"
NODE_ID = "00000000-0000-4000-8000-000000000404"
OUTPUT_ID = "00000000-0000-4000-8000-000000000405"
OUTPUT_2_ID = "00000000-0000-4000-8000-000000000413"
EDGE_2_ID = "00000000-0000-4000-8000-000000000406"
NODE_2_ID = "00000000-0000-4000-8000-000000000407"
EDGE_3_ID = "00000000-0000-4000-8000-000000000408"
EDGE_4_ID = "00000000-0000-4000-8000-000000000409"
EDGE_5_ID = "00000000-0000-4000-8000-000000000410"
EDGE_6_ID = "00000000-0000-4000-8000-000000000411"
EDGE_7_ID = "00000000-0000-4000-8000-000000000412"
DRAFT_HASH = "d" * 64


def _payload() -> dict[str, Any]:
    return {
        "proposal_id": PROPOSAL_ID,
        "draft_hash": DRAFT_HASH,
        "supersedes_draft_hash": None,
        "summary": PROPOSAL_SUMMARY_TEMPLATE,
        "rationale": PROPOSAL_RATIONALE_TEMPLATE,
        "component_counts": {"sources": 1, "nodes": 1, "edges": 5, "outputs": 1},
        "blockers": [
            {
                "code": "policy_review_required",
                "category": "policy",
                "summary": proposal_blocker_summary("policy_review_required"),
                "edit_target": {"kind": "node", "stable_id": NODE_ID},
            }
        ],
        "graph": {
            "sources": [
                {
                    "stable_id": SOURCE_ID,
                    "label": proposal_component_label("source", 0),
                    "plugin": {"kind": "source", "id": "csv"},
                }
            ],
            "edges": [
                {
                    "stable_id": EDGE_ID,
                    "from_endpoint": {"kind": "source", "stable_id": SOURCE_ID},
                    "to_endpoint": {"kind": "node", "stable_id": NODE_ID},
                    "flow": {"kind": "source_success", "branch": None},
                },
                {
                    "stable_id": EDGE_2_ID,
                    "from_endpoint": {"kind": "source", "stable_id": SOURCE_ID},
                    "to_endpoint": {"kind": "discard"},
                    "flow": {"kind": "source_validation_failure"},
                },
                {
                    "stable_id": EDGE_3_ID,
                    "from_endpoint": {"kind": "node", "stable_id": NODE_ID},
                    "to_endpoint": {"kind": "output", "stable_id": OUTPUT_ID},
                    "flow": {"kind": "node_success", "branch": None},
                },
                {
                    "stable_id": EDGE_4_ID,
                    "from_endpoint": {"kind": "node", "stable_id": NODE_ID},
                    "to_endpoint": {"kind": "discard"},
                    "flow": {"kind": "node_error"},
                },
                {
                    "stable_id": EDGE_5_ID,
                    "from_endpoint": {"kind": "output", "stable_id": OUTPUT_ID},
                    "to_endpoint": {"kind": "discard"},
                    "flow": {"kind": "output_write_failure"},
                },
            ],
        },
        "nodes": [
            {
                "stable_id": NODE_ID,
                "label": proposal_component_label("node", 0),
                "node_type": "transform",
                "plugin": {"kind": "transform", "id": "schema_guard"},
                "behavior": {"kind": "transform"},
            }
        ],
        "outputs": [
            {
                "stable_id": OUTPUT_ID,
                "label": proposal_component_label("output", 0),
                "plugin": {"kind": "sink", "id": "json"},
            }
        ],
        "edit_targets": [
            {"kind": "source", "stable_id": SOURCE_ID},
            {"kind": "node", "stable_id": NODE_ID},
            {"kind": "edge", "stable_id": EDGE_ID},
            {"kind": "edge", "stable_id": EDGE_2_ID},
            {"kind": "edge", "stable_id": EDGE_3_ID},
            {"kind": "edge", "stable_id": EDGE_4_ID},
            {"kind": "edge", "stable_id": EDGE_5_ID},
            {"kind": "output", "stable_id": OUTPUT_ID},
        ],
    }


def _gate_payload() -> dict[str, Any]:
    payload = _payload()
    route_aliases = [proposal_structural_label("route", index) for index in range(3)]
    payload["nodes"][0] = {
        "stable_id": NODE_ID,
        "label": proposal_component_label("node", 0),
        "node_type": "gate",
        "plugin": None,
        "behavior": {"kind": "gate", "route_aliases": route_aliases, "fork_branches": []},
    }
    payload["graph"]["edges"] = [
        {
            "stable_id": EDGE_ID,
            "from_endpoint": {"kind": "source", "stable_id": SOURCE_ID},
            "to_endpoint": {"kind": "node", "stable_id": NODE_ID},
            "flow": {"kind": "source_success", "branch": None},
        },
        {
            "stable_id": EDGE_2_ID,
            "from_endpoint": {"kind": "source", "stable_id": SOURCE_ID},
            "to_endpoint": {"kind": "discard"},
            "flow": {"kind": "source_validation_failure"},
        },
        *(
            {
                "stable_id": stable_id,
                "from_endpoint": {"kind": "node", "stable_id": NODE_ID},
                "to_endpoint": {"kind": "output", "stable_id": OUTPUT_ID},
                "flow": {"kind": "gate_route", "route": route_alias, "branch": None},
            }
            for stable_id, route_alias in zip((EDGE_3_ID, EDGE_4_ID, EDGE_5_ID), route_aliases, strict=True)
        ),
        {
            "stable_id": EDGE_6_ID,
            "from_endpoint": {"kind": "output", "stable_id": OUTPUT_ID},
            "to_endpoint": {"kind": "discard"},
            "flow": {"kind": "output_write_failure"},
        },
    ]
    payload["component_counts"]["edges"] = 6
    return payload


def _fork_coalesce_payload() -> dict[str, Any]:
    payload = _payload()
    route = proposal_structural_label("route", 0)
    branches = [proposal_structural_label("branch", index) for index in range(2)]
    payload["nodes"] = [
        {
            "stable_id": NODE_ID,
            "label": proposal_component_label("node", 0),
            "node_type": "gate",
            "plugin": None,
            "behavior": {
                "kind": "gate",
                "route_aliases": [route],
                "fork_branches": [{"routes": [route], "branch": branch} for branch in branches],
            },
        },
        {
            "stable_id": NODE_2_ID,
            "label": proposal_component_label("node", 1),
            "node_type": "coalesce",
            "plugin": None,
            "behavior": {
                "kind": "coalesce",
                "branch_aliases": branches,
                "policy": "quorum",
                "merge": "nested",
            },
        },
    ]
    payload["graph"]["edges"] = [
        {
            "stable_id": EDGE_ID,
            "from_endpoint": {"kind": "source", "stable_id": SOURCE_ID},
            "to_endpoint": {"kind": "node", "stable_id": NODE_ID},
            "flow": {"kind": "source_success", "branch": None},
        },
        {
            "stable_id": EDGE_2_ID,
            "from_endpoint": {"kind": "source", "stable_id": SOURCE_ID},
            "to_endpoint": {"kind": "discard"},
            "flow": {"kind": "source_validation_failure"},
        },
        *(
            {
                "stable_id": stable_id,
                "from_endpoint": {"kind": "node", "stable_id": NODE_ID},
                "to_endpoint": {"kind": "node", "stable_id": NODE_2_ID},
                "flow": {"kind": "gate_fork", "routes": [route], "branch": branch},
            }
            for stable_id, branch in zip((EDGE_3_ID, EDGE_4_ID), branches, strict=True)
        ),
        {
            "stable_id": EDGE_5_ID,
            "from_endpoint": {"kind": "node", "stable_id": NODE_2_ID},
            "to_endpoint": {"kind": "output", "stable_id": OUTPUT_ID},
            "flow": {"kind": "coalesce_success", "branch": None},
        },
        {
            "stable_id": EDGE_6_ID,
            "from_endpoint": {"kind": "output", "stable_id": OUTPUT_ID},
            "to_endpoint": {"kind": "discard"},
            "flow": {"kind": "output_write_failure"},
        },
    ]
    payload["component_counts"] = {"sources": 1, "nodes": 2, "edges": 6, "outputs": 1}
    return payload


def _queue_payload() -> dict[str, Any]:
    payload = _payload()
    payload["nodes"] = [
        {
            "stable_id": NODE_ID,
            "label": proposal_component_label("node", 0),
            "node_type": "queue",
            "plugin": None,
            "behavior": {"kind": "queue"},
        },
        {
            "stable_id": NODE_2_ID,
            "label": proposal_component_label("node", 1),
            "node_type": "transform",
            "plugin": {"kind": "transform", "id": "schema_guard"},
            "behavior": {"kind": "transform"},
        },
    ]
    payload["graph"]["edges"] = [
        {
            "stable_id": EDGE_ID,
            "from_endpoint": {"kind": "source", "stable_id": SOURCE_ID},
            "to_endpoint": {"kind": "node", "stable_id": NODE_ID},
            "flow": {"kind": "source_success", "branch": None},
        },
        {
            "stable_id": EDGE_2_ID,
            "from_endpoint": {"kind": "source", "stable_id": SOURCE_ID},
            "to_endpoint": {"kind": "discard"},
            "flow": {"kind": "source_validation_failure"},
        },
        {
            "stable_id": EDGE_3_ID,
            "from_endpoint": {"kind": "node", "stable_id": NODE_ID},
            "to_endpoint": {"kind": "node", "stable_id": NODE_2_ID},
            "flow": {"kind": "queue_continue", "branch": None},
        },
        {
            "stable_id": EDGE_4_ID,
            "from_endpoint": {"kind": "node", "stable_id": NODE_2_ID},
            "to_endpoint": {"kind": "output", "stable_id": OUTPUT_ID},
            "flow": {"kind": "node_success", "branch": None},
        },
        {
            "stable_id": EDGE_5_ID,
            "from_endpoint": {"kind": "node", "stable_id": NODE_2_ID},
            "to_endpoint": {"kind": "discard"},
            "flow": {"kind": "node_error"},
        },
        {
            "stable_id": EDGE_6_ID,
            "from_endpoint": {"kind": "output", "stable_id": OUTPUT_ID},
            "to_endpoint": {"kind": "discard"},
            "flow": {"kind": "output_write_failure"},
        },
    ]
    payload["component_counts"] = {"sources": 1, "nodes": 2, "edges": 6, "outputs": 1}
    return payload


def test_propose_pipeline_is_the_only_proposal_turn() -> None:
    assert TurnType.PROPOSE_PIPELINE.value == "propose_pipeline"
    assert "propose_chain" not in {turn.value for turn in TurnType}
    assert legal_turn_types_for(GuidedStep.STEP_3_TRANSFORMS) == {
        TurnType.PROPOSE_PIPELINE,
        TurnType.SINGLE_SELECT,
        TurnType.SCHEMA_FORM,
    }


def test_propose_pipeline_accepts_the_exact_redacted_projection() -> None:
    payload = _payload()

    assert validate_payload(TurnType.PROPOSE_PIPELINE, payload) is None
    assert set(payload) == {
        "proposal_id",
        "draft_hash",
        "supersedes_draft_hash",
        "summary",
        "rationale",
        "component_counts",
        "blockers",
        "graph",
        "nodes",
        "outputs",
        "edit_targets",
    }


def test_propose_pipeline_supersedes_draft_hash_is_null_or_exact_hash() -> None:
    """The revision discriminator is closed: null (first proposal) or 64-hex.

    The tutorial's pre-Send auto-proposal carries ``supersedes_draft_hash:
    None``; a frozen-prompt (or prose) revision proposal carries the draft hash
    it supersedes. The frontend withholds the tutorial "Review wiring" primary
    on the former, so the discriminator must survive the closed wire shape.
    """
    superseding = _payload()
    superseding["supersedes_draft_hash"] = "e" * 64
    assert validate_payload(TurnType.PROPOSE_PIPELINE, superseding) is None

    for bad in ("E" * 64, "e" * 63, "", 7):
        malformed = _payload()
        malformed["supersedes_draft_hash"] = bad
        assert validate_payload(TurnType.PROPOSE_PIPELINE, malformed) is not None


@pytest.mark.parametrize(
    ("path", "key"),
    [
        ((), "model_why"),
        (("blockers", 0), "raw_error"),
        (("graph",), "metadata"),
        (("graph", "sources", 0), "options"),
        (("graph", "edges", 0), "metadata"),
        (("nodes", 0), "options"),
        (("outputs", 0), "path"),
        (("edit_targets", 0), "index"),
    ],
)
def test_propose_pipeline_rejects_extra_keys_at_every_closed_layer(path: tuple[str | int, ...], key: str) -> None:
    payload = _payload()
    cursor: Any = payload
    for part in path:
        cursor = cursor[part]
    cursor[key] = "SECRET-provider-verbatim"

    error = validate_payload(TurnType.PROPOSE_PIPELINE, payload)

    assert error is not None
    assert "unexpected keys" in error


@pytest.mark.parametrize(
    ("path", "key"),
    [
        ((), "summary"),
        ((), "supersedes_draft_hash"),
        (("component_counts",), "nodes"),
        (("blockers", 0), "code"),
        (("graph",), "edges"),
        (("graph", "sources", 0), "label"),
        (("graph", "sources", 0), "plugin"),
        (("graph", "edges", 0), "flow"),
        (("nodes", 0), "node_type"),
        (("nodes", 0), "plugin"),
        (("nodes", 0, "behavior"), "kind"),
        (("outputs", 0), "plugin"),
        (("edit_targets", 0), "stable_id"),
    ],
)
def test_propose_pipeline_rejects_missing_keys_at_every_required_layer(path: tuple[str | int, ...], key: str) -> None:
    payload = _payload()
    cursor: Any = payload
    for part in path:
        cursor = cursor[part]
    del cursor[key]

    error = validate_payload(TurnType.PROPOSE_PIPELINE, payload)

    assert error is not None
    assert "missing required keys" in error


@pytest.mark.parametrize(
    ("path", "bad_value"),
    [
        (("proposal_id",), "not-a-uuid"),
        (("draft_hash",), "A" * 64),
        (("blockers", 0, "edit_target", "stable_id"), "1"),
        (("graph", "sources", 0, "stable_id"), "not-a-uuid"),
        (("graph", "edges", 0, "flow", "kind"), "provider-route"),
        (("nodes", 0, "stable_id"), "not-a-uuid"),
        (("outputs", 0, "stable_id"), "not-a-uuid"),
        (("edit_targets", 0, "kind"), "step"),
    ],
)
def test_propose_pipeline_rejects_malformed_ids_hashes_and_closed_values(
    path: tuple[str | int, ...],
    bad_value: str,
) -> None:
    payload = _payload()
    cursor: Any = payload
    for part in path[:-1]:
        cursor = cursor[part]
    cursor[path[-1]] = bad_value

    assert validate_payload(TurnType.PROPOSE_PIPELINE, payload) is not None


@pytest.mark.parametrize(
    ("path", "canary"),
    [
        (("summary",), "sk-live-secret"),
        (("rationale",), "provider traceback"),
        (("blockers", 0, "summary"), "raw validation payload"),
        (("graph", "sources", 0, "label"), "/home/operator/private.csv"),
        (("graph", "sources", 0, "label"), "password123"),
        (("nodes", 0, "label"), "Bearer-credential"),
        (("outputs", 0, "label"), "REDACTED-AWS-CANARY"),
    ],
)
def test_propose_pipeline_server_authored_text_is_exact_not_blacklist_filtered(
    path: tuple[str | int, ...],
    canary: str,
) -> None:
    payload = deepcopy(_payload())
    cursor: Any = payload
    for part in path[:-1]:
        cursor = cursor[part]
    cursor[path[-1]] = canary

    assert validate_payload(TurnType.PROPOSE_PIPELINE, payload) is not None


@pytest.mark.parametrize(
    ("surface", "canary"),
    [
        ("behavior_route", "password123"),
        ("flow_route", "Bearer-credential"),
        ("behavior_fork_branch", "REDACTED-AWS-CANARY"),
        ("flow_fork_branch", "/home/operator/private"),
    ],
)
def test_propose_pipeline_structural_aliases_are_exact_server_ordinals(surface: str, canary: str) -> None:
    payload = _gate_payload()
    behavior = payload["nodes"][0]["behavior"]
    if "fork_branch" in surface:
        branch = proposal_structural_label("branch", 0)
        behavior["fork_branches"] = [{"routes": [behavior["route_aliases"][0]], "branch": branch}]
        payload["graph"]["edges"][2]["flow"] = {
            "kind": "gate_fork",
            "routes": [behavior["route_aliases"][0]],
            "branch": branch,
        }
    if surface == "behavior_route":
        behavior["route_aliases"][0] = canary
    elif surface == "flow_route":
        payload["graph"]["edges"][2]["flow"]["route"] = canary
    elif surface == "behavior_fork_branch":
        behavior["fork_branches"][0]["branch"] = canary
    else:
        payload["graph"]["edges"][2]["flow"]["branch"] = canary

    assert validate_payload(TurnType.PROPOSE_PIPELINE, payload) is not None


def test_propose_pipeline_contains_no_arbitrary_component_name_surface() -> None:
    payload = _payload()

    def keys(value: object) -> set[str]:
        if isinstance(value, dict):
            return set(value).union(*(keys(item) for item in value.values()))
        if isinstance(value, list):
            return set().union(*(keys(item) for item in value), set())
        return set()

    assert "name" not in keys(payload)


def test_propose_pipeline_catalog_refs_have_an_explicit_authority_check_for_task4() -> None:
    payload = _payload()
    catalog = {
        "source": frozenset({"csv"}),
        "transform": frozenset({"schema_guard"}),
        "sink": frozenset({"json"}),
    }

    assert validate_proposal_catalog_refs(payload, catalog) is None

    payload["nodes"][0]["plugin"]["id"] = "password123"
    error = validate_proposal_catalog_refs(payload, catalog)

    assert error is not None
    assert "catalog" in error


def test_propose_pipeline_counts_must_match_the_exact_structural_projection() -> None:
    payload = _payload()
    payload["component_counts"]["edges"] = 3

    error = validate_payload(TurnType.PROPOSE_PIPELINE, payload)

    assert error is not None
    assert "component_counts" in error


@pytest.mark.parametrize(
    ("code", "wrong_category"),
    [
        ("pipeline_invalid", "policy"),
        ("policy_review_required", "availability"),
        ("plugin_unavailable", "interpretation"),
        ("interpretation_required", "validation"),
    ],
)
def test_propose_pipeline_blocker_code_has_one_fixed_category(code: str, wrong_category: str) -> None:
    payload = _payload()
    payload["blockers"][0]["code"] = code
    payload["blockers"][0]["category"] = wrong_category

    error = validate_payload(TurnType.PROPOSE_PIPELINE, payload)

    assert error is not None
    assert "category" in error


def test_propose_pipeline_component_ids_are_globally_unique() -> None:
    payload = _payload()
    payload["nodes"][0]["stable_id"] = SOURCE_ID

    error = validate_payload(TurnType.PROPOSE_PIPELINE, payload)

    assert error is not None
    assert "globally unique" in error


def test_propose_pipeline_rejects_self_loop() -> None:
    payload = _payload()
    payload["graph"]["edges"][2]["to_endpoint"] = {"kind": "node", "stable_id": NODE_ID}

    error = validate_payload(TurnType.PROPOSE_PIPELINE, payload)

    assert error is not None
    assert "self-loop" in error


def test_propose_pipeline_rejects_multi_node_cycle() -> None:
    payload = _payload()
    payload["nodes"].append(
        {
            "stable_id": NODE_2_ID,
            "label": proposal_component_label("node", 1),
            "node_type": "transform",
            "plugin": {"kind": "transform", "id": "schema_guard"},
            "behavior": {"kind": "transform"},
        }
    )
    payload["graph"]["edges"][2]["to_endpoint"] = {"kind": "node", "stable_id": NODE_2_ID}
    payload["graph"]["edges"].extend(
        [
            {
                "stable_id": EDGE_6_ID,
                "from_endpoint": {"kind": "node", "stable_id": NODE_2_ID},
                "to_endpoint": {"kind": "node", "stable_id": NODE_ID},
                "flow": {"kind": "node_success", "branch": None},
            },
            {
                "stable_id": EDGE_7_ID,
                "from_endpoint": {"kind": "node", "stable_id": NODE_2_ID},
                "to_endpoint": {"kind": "output", "stable_id": OUTPUT_ID},
                "flow": {"kind": "node_error"},
            },
        ]
    )
    payload["component_counts"] = {"sources": 1, "nodes": 2, "edges": 7, "outputs": 1}

    error = validate_payload(TurnType.PROPOSE_PIPELINE, payload)

    assert error is not None
    assert "cycle" in error


@pytest.mark.parametrize("orphan_kind", ["node", "output"])
def test_propose_pipeline_rejects_unreachable_components(orphan_kind: str) -> None:
    payload = _payload()
    if orphan_kind == "node":
        payload["nodes"].append(
            {
                "stable_id": NODE_2_ID,
                "label": proposal_component_label("node", 1),
                "node_type": "transform",
                "plugin": {"kind": "transform", "id": "schema_guard"},
                "behavior": {"kind": "transform"},
            }
        )
        payload["component_counts"]["nodes"] = 2
        payload["graph"]["edges"].extend(
            [
                {
                    "stable_id": EDGE_6_ID,
                    "from_endpoint": {"kind": "node", "stable_id": NODE_2_ID},
                    "to_endpoint": {"kind": "output", "stable_id": OUTPUT_ID},
                    "flow": {"kind": "node_success", "branch": None},
                },
                {
                    "stable_id": EDGE_7_ID,
                    "from_endpoint": {"kind": "node", "stable_id": NODE_2_ID},
                    "to_endpoint": {"kind": "discard"},
                    "flow": {"kind": "node_error"},
                },
            ]
        )
        payload["component_counts"]["edges"] = 7
    else:
        payload["outputs"].append(
            {
                "stable_id": OUTPUT_2_ID,
                "label": proposal_component_label("output", 1),
                "plugin": {"kind": "sink", "id": "json"},
            }
        )
        payload["graph"]["edges"].append(
            {
                "stable_id": EDGE_6_ID,
                "from_endpoint": {"kind": "output", "stable_id": OUTPUT_2_ID},
                "to_endpoint": {"kind": "discard"},
                "flow": {"kind": "output_write_failure"},
            }
        )
        payload["component_counts"]["outputs"] = 2
        payload["component_counts"]["edges"] = 6

    error = validate_payload(TurnType.PROPOSE_PIPELINE, payload)

    assert error is not None
    assert "unreachable" in error or "does not reach an output" in error


def test_propose_pipeline_gate_represents_arbitrary_multiple_routes_structurally() -> None:
    payload = _gate_payload()

    assert validate_payload(TurnType.PROPOSE_PIPELINE, payload) is None


def test_propose_pipeline_gate_represents_multiple_routes_selecting_one_fanout() -> None:
    payload = _fork_coalesce_payload()
    routes = [proposal_structural_label("route", index) for index in range(2)]
    payload["nodes"][0]["behavior"]["route_aliases"] = routes
    for branch in payload["nodes"][0]["behavior"]["fork_branches"]:
        branch["routes"] = routes
    for edge in payload["graph"]["edges"]:
        if edge["flow"]["kind"] == "gate_fork":
            edge["flow"]["routes"] = routes

    assert validate_payload(TurnType.PROPOSE_PIPELINE, payload) is None


def test_propose_pipeline_rejects_per_route_split_fanout_that_canonical_gate_cannot_represent() -> None:
    payload = _fork_coalesce_payload()
    routes = [proposal_structural_label("route", index) for index in range(2)]
    payload["nodes"][0]["behavior"]["route_aliases"] = routes
    for index, branch in enumerate(payload["nodes"][0]["behavior"]["fork_branches"]):
        branch["routes"] = [routes[index]]
    fork_edges = [edge for edge in payload["graph"]["edges"] if edge["flow"]["kind"] == "gate_fork"]
    for index, edge in enumerate(fork_edges):
        edge["flow"]["routes"] = [routes[index]]

    error = validate_payload(TurnType.PROPOSE_PIPELINE, payload)

    assert error is not None
    assert "one exact ordered route sequence" in error


def test_propose_pipeline_rejects_duplicate_direct_gate_route_action() -> None:
    payload = _gate_payload()
    duplicate = deepcopy(payload["graph"]["edges"][2])
    duplicate["stable_id"] = EDGE_7_ID
    payload["graph"]["edges"].append(duplicate)
    payload["component_counts"]["edges"] += 1

    error = validate_payload(TurnType.PROPOSE_PIPELINE, payload)

    assert error is not None
    assert "route" in error


def test_propose_pipeline_rejects_route_alias_reused_by_two_gates() -> None:
    payload = _payload()
    routes = [proposal_structural_label("route", index) for index in range(2)]
    payload["nodes"] = [
        {
            "stable_id": NODE_ID,
            "label": proposal_component_label("node", 0),
            "node_type": "gate",
            "plugin": None,
            "behavior": {"kind": "gate", "route_aliases": routes, "fork_branches": []},
        },
        {
            "stable_id": NODE_2_ID,
            "label": proposal_component_label("node", 1),
            "node_type": "gate",
            "plugin": None,
            "behavior": {"kind": "gate", "route_aliases": [routes[0]], "fork_branches": []},
        },
    ]
    payload["graph"]["edges"] = [
        {
            "stable_id": EDGE_ID,
            "from_endpoint": {"kind": "source", "stable_id": SOURCE_ID},
            "to_endpoint": {"kind": "node", "stable_id": NODE_ID},
            "flow": {"kind": "source_success", "branch": None},
        },
        {
            "stable_id": EDGE_2_ID,
            "from_endpoint": {"kind": "source", "stable_id": SOURCE_ID},
            "to_endpoint": {"kind": "discard"},
            "flow": {"kind": "source_validation_failure"},
        },
        {
            "stable_id": EDGE_3_ID,
            "from_endpoint": {"kind": "node", "stable_id": NODE_ID},
            "to_endpoint": {"kind": "node", "stable_id": NODE_2_ID},
            "flow": {"kind": "gate_route", "route": routes[0], "branch": None},
        },
        {
            "stable_id": EDGE_4_ID,
            "from_endpoint": {"kind": "node", "stable_id": NODE_ID},
            "to_endpoint": {"kind": "output", "stable_id": OUTPUT_ID},
            "flow": {"kind": "gate_route", "route": routes[1], "branch": None},
        },
        {
            "stable_id": EDGE_5_ID,
            "from_endpoint": {"kind": "node", "stable_id": NODE_2_ID},
            "to_endpoint": {"kind": "output", "stable_id": OUTPUT_ID},
            "flow": {"kind": "gate_route", "route": routes[0], "branch": None},
        },
        {
            "stable_id": EDGE_6_ID,
            "from_endpoint": {"kind": "output", "stable_id": OUTPUT_ID},
            "to_endpoint": {"kind": "discard"},
            "flow": {"kind": "output_write_failure"},
        },
    ]
    payload["component_counts"] = {"sources": 1, "nodes": 2, "edges": 6, "outputs": 1}
    payload["blockers"] = []
    payload["edit_targets"] = []

    error = validate_payload(TurnType.PROPOSE_PIPELINE, payload)

    assert error is not None
    assert "globally unique" in error


@pytest.mark.parametrize(
    ("node_type", "behavior"),
    [
        (
            "aggregation",
            {
                "kind": "aggregation",
                "trigger_kinds": ["count", "timeout", "condition"],
                "count": "100",
                "timeout_seconds": 30.0,
                "output_mode": "transform",
                "expected_output_count": "2",
            },
        ),
    ],
)
def test_propose_pipeline_has_closed_behavior_for_non_linear_node_types(
    node_type: str,
    behavior: dict[str, Any],
) -> None:
    payload = _payload()
    payload["nodes"][0]["node_type"] = node_type
    payload["nodes"][0]["behavior"] = behavior
    payload["nodes"][0]["plugin"] = {"kind": "transform", "id": "schema_guard"} if node_type == "aggregation" else None
    assert validate_payload(TurnType.PROPOSE_PIPELINE, payload) is None


def test_propose_pipeline_queue_projects_one_ordinary_node_continuation() -> None:
    assert validate_payload(TurnType.PROPOSE_PIPELINE, _queue_payload()) is None


def test_propose_pipeline_rejects_two_queue_continuations() -> None:
    payload = _queue_payload()
    duplicate = deepcopy(payload["graph"]["edges"][2])
    duplicate["stable_id"] = EDGE_7_ID
    payload["graph"]["edges"].append(duplicate)
    payload["component_counts"]["edges"] += 1

    error = validate_payload(TurnType.PROPOSE_PIPELINE, payload)

    assert error is not None
    assert "exactly one" in error


@pytest.mark.parametrize("target_kind", ["output", "queue"])
def test_propose_pipeline_rejects_nonordinary_queue_target(target_kind: str) -> None:
    payload = _queue_payload()
    if target_kind == "output":
        payload["graph"]["edges"][2]["to_endpoint"] = {"kind": "output", "stable_id": OUTPUT_ID}
    else:
        payload["nodes"][1]["node_type"] = "queue"
        payload["nodes"][1]["plugin"] = None
        payload["nodes"][1]["behavior"] = {"kind": "queue"}

    error = validate_payload(TurnType.PROPOSE_PIPELINE, payload)

    assert error is not None


def test_propose_pipeline_rejects_branch_use_without_gate_fork_origin() -> None:
    payload = _payload()
    payload["graph"]["edges"][0]["flow"]["branch"] = proposal_structural_label("branch", 0)

    error = validate_payload(TurnType.PROPOSE_PIPELINE, payload)

    assert error is not None
    assert "origin" in error


def test_propose_pipeline_aggregation_preserves_distinct_safe_trigger_values() -> None:
    first = _payload()
    first["nodes"][0]["node_type"] = "aggregation"
    first["nodes"][0]["behavior"] = {
        "kind": "aggregation",
        "trigger_kinds": ["count", "timeout"],
        "count": "9007199254740993123456789",
        "timeout_seconds": 1.25,
        "output_mode": "transform",
        "expected_output_count": "-2",
    }
    second = deepcopy(first)
    second["nodes"][0]["behavior"]["count"] = "9007199254740993123456790"
    second["nodes"][0]["behavior"]["timeout_seconds"] = 2.5

    assert validate_payload(TurnType.PROPOSE_PIPELINE, first) is None
    assert validate_payload(TurnType.PROPOSE_PIPELINE, second) is None
    assert first["nodes"][0]["behavior"] != second["nodes"][0]["behavior"]


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("count", "01"),
        ("count", 4),
        ("timeout_seconds", float("inf")),
        ("timeout_seconds", 0),
        ("expected_output_count", "-0"),
    ],
)
def test_propose_pipeline_rejects_noncanonical_aggregation_values(field: str, value: object) -> None:
    payload = _payload()
    payload["nodes"][0]["node_type"] = "aggregation"
    payload["nodes"][0]["behavior"] = {
        "kind": "aggregation",
        "trigger_kinds": ["count", "timeout"],
        "count": "10",
        "timeout_seconds": 1.0,
        "output_mode": "transform",
        "expected_output_count": "1",
    }
    payload["nodes"][0]["behavior"][field] = value

    assert validate_payload(TurnType.PROPOSE_PIPELINE, payload) is not None


def test_propose_pipeline_rejects_oversized_integer_timeout_without_raising() -> None:
    payload = _payload()
    payload["nodes"][0]["node_type"] = "aggregation"
    payload["nodes"][0]["behavior"] = {
        "kind": "aggregation",
        "trigger_kinds": ["timeout"],
        "count": None,
        "timeout_seconds": 10**10_000,
        "output_mode": "transform",
        "expected_output_count": None,
    }

    error = validate_payload(TurnType.PROPOSE_PIPELINE, payload)

    assert error is not None
    assert "timeout_seconds" in error


def test_propose_pipeline_represents_fork_and_coalesce_with_shared_branch_aliases() -> None:
    payload = _fork_coalesce_payload()

    assert validate_payload(TurnType.PROPOSE_PIPELINE, payload) is None


def test_propose_pipeline_rejects_one_fork_branch_set_consumed_by_two_coalesces() -> None:
    payload = _payload()
    gate_a_id = NODE_ID
    gate_b_id = NODE_2_ID
    gate_c_id = "00000000-0000-4000-8000-000000000414"
    coalesce_a_id = "00000000-0000-4000-8000-000000000415"
    coalesce_b_id = "00000000-0000-4000-8000-000000000416"
    routes = [proposal_structural_label("route", index) for index in range(5)]
    branches = [proposal_structural_label("branch", index) for index in range(2)]
    payload["nodes"] = [
        {
            "stable_id": gate_a_id,
            "label": proposal_component_label("node", 0),
            "node_type": "gate",
            "plugin": None,
            "behavior": {
                "kind": "gate",
                "route_aliases": [routes[0]],
                "fork_branches": [
                    {"routes": [routes[0]], "branch": branches[0]},
                    {"routes": [routes[0]], "branch": branches[1]},
                ],
            },
        },
        {
            "stable_id": gate_b_id,
            "label": proposal_component_label("node", 1),
            "node_type": "gate",
            "plugin": None,
            "behavior": {"kind": "gate", "route_aliases": routes[1:3], "fork_branches": []},
        },
        {
            "stable_id": gate_c_id,
            "label": proposal_component_label("node", 2),
            "node_type": "gate",
            "plugin": None,
            "behavior": {"kind": "gate", "route_aliases": routes[3:5], "fork_branches": []},
        },
        *(
            {
                "stable_id": stable_id,
                "label": proposal_component_label("node", index),
                "node_type": "coalesce",
                "plugin": None,
                "behavior": {
                    "kind": "coalesce",
                    "branch_aliases": branches,
                    "policy": "require_all",
                    "merge": "union",
                },
            }
            for index, stable_id in ((3, coalesce_a_id), (4, coalesce_b_id))
        ),
    ]
    edge_ids = [f"00000000-0000-4000-8000-{index:012d}" for index in range(500, 511)]
    payload["graph"]["edges"] = [
        {
            "stable_id": edge_ids[0],
            "from_endpoint": {"kind": "source", "stable_id": SOURCE_ID},
            "to_endpoint": {"kind": "node", "stable_id": gate_a_id},
            "flow": {"kind": "source_success", "branch": None},
        },
        {
            "stable_id": edge_ids[1],
            "from_endpoint": {"kind": "source", "stable_id": SOURCE_ID},
            "to_endpoint": {"kind": "discard"},
            "flow": {"kind": "source_validation_failure"},
        },
        {
            "stable_id": edge_ids[2],
            "from_endpoint": {"kind": "node", "stable_id": gate_a_id},
            "to_endpoint": {"kind": "node", "stable_id": gate_b_id},
            "flow": {"kind": "gate_fork", "routes": [routes[0]], "branch": branches[0]},
        },
        {
            "stable_id": edge_ids[3],
            "from_endpoint": {"kind": "node", "stable_id": gate_a_id},
            "to_endpoint": {"kind": "node", "stable_id": gate_c_id},
            "flow": {"kind": "gate_fork", "routes": [routes[0]], "branch": branches[1]},
        },
        {
            "stable_id": edge_ids[4],
            "from_endpoint": {"kind": "node", "stable_id": gate_b_id},
            "to_endpoint": {"kind": "node", "stable_id": coalesce_a_id},
            "flow": {"kind": "gate_route", "route": routes[1], "branch": branches[0]},
        },
        {
            "stable_id": edge_ids[5],
            "from_endpoint": {"kind": "node", "stable_id": gate_b_id},
            "to_endpoint": {"kind": "node", "stable_id": coalesce_b_id},
            "flow": {"kind": "gate_route", "route": routes[2], "branch": branches[0]},
        },
        {
            "stable_id": edge_ids[6],
            "from_endpoint": {"kind": "node", "stable_id": gate_c_id},
            "to_endpoint": {"kind": "node", "stable_id": coalesce_a_id},
            "flow": {"kind": "gate_route", "route": routes[3], "branch": branches[1]},
        },
        {
            "stable_id": edge_ids[7],
            "from_endpoint": {"kind": "node", "stable_id": gate_c_id},
            "to_endpoint": {"kind": "node", "stable_id": coalesce_b_id},
            "flow": {"kind": "gate_route", "route": routes[4], "branch": branches[1]},
        },
        *(
            {
                "stable_id": edge_id,
                "from_endpoint": {"kind": "node", "stable_id": stable_id},
                "to_endpoint": {"kind": "output", "stable_id": OUTPUT_ID},
                "flow": {"kind": "coalesce_success", "branch": None},
            }
            for edge_id, stable_id in ((edge_ids[8], coalesce_a_id), (edge_ids[9], coalesce_b_id))
        ),
        {
            "stable_id": edge_ids[10],
            "from_endpoint": {"kind": "output", "stable_id": OUTPUT_ID},
            "to_endpoint": {"kind": "discard"},
            "flow": {"kind": "output_write_failure"},
        },
    ]
    payload["component_counts"] = {"sources": 1, "nodes": 5, "edges": 11, "outputs": 1}
    payload["blockers"] = []
    payload["edit_targets"] = []

    error = validate_payload(TurnType.PROPOSE_PIPELINE, payload)

    assert error is not None
    assert "more than one coalesce" in error


def test_propose_pipeline_rejects_coalesce_branch_without_matching_fork_origin() -> None:
    payload = _fork_coalesce_payload()
    payload["nodes"][1]["behavior"]["branch_aliases"][1] = "branch-3"

    error = validate_payload(TurnType.PROPOSE_PIPELINE, payload)

    assert error is not None
    assert "branch" in error


def test_propose_pipeline_rejects_sensitive_canary_in_coalesce_branch_alias() -> None:
    payload = _fork_coalesce_payload()
    payload["nodes"][1]["behavior"]["branch_aliases"][0] = "REDACTED-AWS-CANARY"

    assert validate_payload(TurnType.PROPOSE_PIPELINE, payload) is not None


def test_propose_pipeline_rejects_behavior_that_does_not_match_node_type() -> None:
    payload = _payload()
    payload["nodes"][0]["behavior"] = {"kind": "queue"}

    error = validate_payload(TurnType.PROPOSE_PIPELINE, payload)

    assert error is not None
    assert "behavior" in error


@pytest.mark.parametrize(
    ("endpoint", "bad_endpoint"),
    [
        ("from_endpoint", {"kind": "output", "stable_id": OUTPUT_ID}),
        ("to_endpoint", {"kind": "source", "stable_id": SOURCE_ID}),
        ("from_endpoint", {"kind": "source", "stable_id": "00000000-0000-4000-8000-000000000499"}),
        ("to_endpoint", {"kind": "node", "stable_id": "00000000-0000-4000-8000-000000000498"}),
    ],
)
def test_propose_pipeline_edges_resolve_to_legal_component_sets(endpoint: str, bad_endpoint: dict[str, str]) -> None:
    payload = _payload()
    payload["graph"]["edges"][0][endpoint] = bad_endpoint

    error = validate_payload(TurnType.PROPOSE_PIPELINE, payload)

    assert error is not None
    assert endpoint in error


@pytest.mark.parametrize("target_owner", ["edit_targets", "blockers"])
def test_propose_pipeline_targets_resolve_exact_kind_and_id(target_owner: str) -> None:
    payload = _payload()
    if target_owner == "edit_targets":
        payload["edit_targets"][0] = {"kind": "edge", "stable_id": SOURCE_ID}
    else:
        payload["blockers"][0]["edit_target"] = {"kind": "output", "stable_id": NODE_ID}

    error = validate_payload(TurnType.PROPOSE_PIPELINE, payload)

    assert error is not None
    assert "target" in error


@pytest.mark.parametrize("target_owner", ["edit_targets", "blockers"])
def test_propose_pipeline_rejects_duplicate_targets(target_owner: str) -> None:
    payload = _payload()
    if target_owner == "edit_targets":
        payload["edit_targets"].append(deepcopy(payload["edit_targets"][0]))
    else:
        payload["blockers"].append(deepcopy(payload["blockers"][0]))

    error = validate_payload(TurnType.PROPOSE_PIPELINE, payload)

    assert error is not None
    assert "duplicate" in error


def test_protocol_maps_are_exhaustive_over_turn_enum() -> None:
    from elspeth.web.composer.guided import protocol

    expected = set(TurnType)
    assert set(protocol._REQUIRED_KEYS) == expected
    assert set(protocol._ALLOWED_KEYS) == expected
    assert set(protocol._PAYLOAD_VALIDATORS) == expected
    assert set().union(*(legal_turn_types_for(step) for step in GuidedStep)) == expected


def test_active_production_exposes_no_removed_chain_or_positional_response_contract() -> None:
    root = Path(__file__).resolve().parents[5]
    production_root = root / "src/elspeth/web"
    paths = (
        path
        for path in production_root.rglob("*")
        if path.suffix in {".py", ".ts", ".tsx"}
        and "tests" not in path.parts
        and ".test." not in path.name
        and "__pycache__" not in path.parts
    )
    forbidden = (
        "PROPOSE_CHAIN",
        "ProposeChain",
        "propose_chain",
        "ChainProposal",
        "handle_step_3_chain_accept",
        "chain_solver",
        "_guided_solve_chain",
        "accepted_step_index",
        "edit_step_index",
    )

    violations = {str(path.relative_to(root)): token for path in paths for token in forbidden if token in path.read_text(encoding="utf-8")}

    assert violations == {}
