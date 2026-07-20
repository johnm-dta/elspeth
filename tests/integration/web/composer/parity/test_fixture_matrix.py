"""Three-surface real-path parity matrix (Plan 05 Task 3).

Nine canonical capability fixtures x two arbitrary authoring surfaces
(freeform, guided-full) = 18 real-path cases. Each case drives one surface's
production entrypoint from the fixture intent all the way to the immutable
committed ``CompositionState`` (real prompt/tool assembly → terminal parser →
custody → candidate validation → durable proposal → acceptance → audited
``set_pipeline`` → public YAML compiler) and asserts the committed graph is
semantically isomorphic to the ground-truth reference, that its public compiled
pipeline agrees, and that it satisfies the fixture's declared capability shape.

No provider network, no skips, no xfail. Cross-surface parity is transitive:
every surface is anchored to the same per-fixture reference committed graph.

The guided-staged surface is added in the next stage; this stage lands the
shared helper (``tests/helpers/composer_graphs.py``), the shared adapters
(``conftest.py``), and the freeform + guided-full columns.
"""

from __future__ import annotations

from typing import Any

import pytest

from tests.helpers.composer_graphs import assert_isomorphic, public_pipeline_semantics

from .conftest import PARITY_FIXTURES, ParityEnv

SURFACES = ["freeform", "guided_full"]


def _committed_nodes(state: Any) -> dict[str, dict[str, Any]]:
    return {node["id"]: node for node in state.to_dict()["nodes"]}


def _assert_semantic_expectations(state: Any, fixture: dict[str, Any]) -> None:
    """Verify the committed graph matches the fixture's declared capability shape.

    This is the "equivalent validation / runtime graph" leg: beyond isomorphism
    to the reference, the committed graph must expose the exact declared
    node/plugin kinds, wiring connections, routes, policies, and failure paths
    the fixture claims — proving the real path derived the intended capability,
    not merely a self-consistent graph.
    """
    expectations = fixture["semantic_expectations"]
    committed = state.to_dict()

    if "source" in expectations:
        sources = committed["sources"]
        assert len(sources) == 1, f"{fixture['class']}: expected a single source, got {list(sources)}"
        source = next(iter(sources.values()))
        for key, value in expectations["source"].items():
            assert source.get(key) == value, f"{fixture['class']}: source.{key} = {source.get(key)!r} != {value!r}"

    if "sources" in expectations:
        by_name = committed["sources"]
        for expected in expectations["sources"]:
            name = expected["name"]
            assert name in by_name, f"{fixture['class']}: missing source {name!r}"
            actual = by_name[name]
            for key in ("plugin", "on_success", "on_validation_failure"):
                if key in expected:
                    assert actual.get(key) == expected[key], f"{fixture['class']}: source[{name}].{key}"

    nodes = _committed_nodes(state)
    for expected in expectations["nodes"]:
        node_id = expected["id"]
        assert node_id in nodes, f"{fixture['class']}: missing node {node_id!r}"
        actual = nodes[node_id]
        for key in ("node_type", "plugin", "input", "on_success", "condition", "policy", "merge", "output_mode"):
            if key in expected:
                assert actual.get(key) == expected[key], (
                    f"{fixture['class']}: node[{node_id}].{key} = {actual.get(key)!r} != {expected[key]!r}"
                )
        if "routes" in expected:
            assert actual.get("routes") == expected["routes"], f"{fixture['class']}: node[{node_id}].routes"
        if "fork_to" in expected:
            assert actual.get("fork_to") == expected["fork_to"], f"{fixture['class']}: node[{node_id}].fork_to"
        if "branches" in expected:
            assert actual.get("branches") == expected["branches"], f"{fixture['class']}: node[{node_id}].branches"
        if "trigger" in expected:
            # The committed trigger carries defaulted keys (condition,
            # timeout_seconds); assert the declared trigger keys are a subset.
            actual_trigger = actual.get("trigger") or {}
            for trigger_key, trigger_value in expected["trigger"].items():
                assert actual_trigger.get(trigger_key) == trigger_value, f"{fixture['class']}: node[{node_id}].trigger.{trigger_key}"

    outputs = {output["name"]: output for output in committed["outputs"]}
    for expected in expectations["outputs"]:
        name = expected["sink_name"]
        assert name in outputs, f"{fixture['class']}: missing output {name!r}"
        actual = outputs[name]
        assert actual["plugin"] == expected["plugin"], f"{fixture['class']}: output[{name}].plugin"
        if "on_write_failure" in expected:
            assert actual["on_write_failure"] == expected["on_write_failure"], f"{fixture['class']}: output[{name}].on_write_failure"


@pytest.mark.asyncio
@pytest.mark.parametrize("surface", SURFACES)
@pytest.mark.parametrize("fixture", PARITY_FIXTURES, ids=lambda f: f["class"])
async def test_surface_derives_isomorphic_committed_graph(
    parity_env: ParityEnv,
    fixture: dict[str, Any],
    surface: str,
) -> None:
    reference = parity_env.reference_state(fixture)
    committed = await parity_env.drive(surface, fixture)

    # 1. Semantic graph isomorphism (primary parity surface).
    assert_isomorphic(committed, reference, left=f"{surface}:{fixture['class']}", right="reference")

    # 2. Public compiled-pipeline (runtime graph) semantics agree.
    assert public_pipeline_semantics(committed) == public_pipeline_semantics(reference), (
        f"{surface}:{fixture['class']}: public pipeline semantics diverged from reference"
    )

    # 3. The committed graph exposes the fixture's declared capability shape.
    _assert_semantic_expectations(committed, fixture)
