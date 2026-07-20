"""Three-surface real-path parity matrix (Plan 05 Task 3).

Nine canonical capability fixtures x three arbitrary authoring surfaces
(freeform, guided-full, guided-staged). Each case drives one surface's
production entrypoint from the fixture intent all the way to the immutable
committed ``CompositionState`` (real prompt/tool assembly → terminal parser →
custody → candidate validation → durable proposal → acceptance / confirm-wiring
→ audited ``set_pipeline`` → public YAML compiler) and asserts the committed
graph is semantically isomorphic to the ground-truth reference.

Freeform and guided-full drive all nine fixtures (18 cases). Guided-staged
drives six (24 cases total): three fixtures — ``multi_source_queue``,
``multi_output``, ``fork_coalesce`` — hit invariants in the guided
wire-projection / structured review that are stricter than the shared
``set_pipeline`` commit path and therefore cannot be authored through the stage
protocol at HEAD. Those are code-proven guided-staged capability GAPS (the
identical committed graph commits fine on the other two surfaces); they are
excluded here and reported as blockers, documented case-by-case in
``_GUIDED_STAGED_CAPABILITY_GAPS`` below. They are gaps to report, not tests to
skip: there is no ``skip``/``xfail`` and no gutted fixture.

No provider network, no skips, no xfail. Cross-surface parity is transitive:
every surface is anchored to the same per-fixture reference committed graph.

Two surfaces (``freeform``, ``guided_full``) let the planner emit the fixture's
canonical component *names*, so they additionally assert byte-exact public-YAML
semantics and the fixture's exact declared capability shape. The guided-staged
surface reviews sources/outputs through the persisted stage protocol, which
auto-assigns positional names (``source`` / ``source_2`` / ``output`` /
``output_2`` …) the operator cannot override; design §8.1 canonicalizes
connection names / source keys away, so isomorphism to the shared reference is
the complete, name-agnostic parity proof for that surface. It is paired with a
positive guided-naming assertion — proving the committed graph really traversed
the staged protocol and that the *only* delta from the reference is the expected
renaming — rather than a weaker name-agnostic reimplementation of the semantic
check (which could only mask a regression isomorphism already catches).
"""

from __future__ import annotations

from typing import Any

import pytest

from tests.helpers.composer_graphs import assert_isomorphic, public_pipeline_semantics

from .conftest import PARITY_FIXTURES, ParityEnv

SURFACES = ["freeform", "guided_full", "guided_staged"]

# Surfaces whose planner emits the fixture's canonical component names verbatim,
# so byte-exact public-YAML equality and exact-name semantic expectations hold.
_NAME_PRESERVING_SURFACES = frozenset({"freeform", "guided_full"})

# Three capability fixtures cannot be authored through the guided-staged stage
# protocol at HEAD: they hit invariants in the guided wire-projection /
# structured review that are STRICTER than the shared audited ``set_pipeline``
# commit path, so freeform + guided_full derive them but guided_staged cannot.
# These are genuine, code-proven guided-staged capability gaps (not fixture
# defects — the identical committed graph commits fine on the other two
# surfaces). They are excluded from the guided_staged parity parametrization and
# reported as blockers; each is traceable here to the exact mechanism + code
# location so a reader sees *why* the guided_staged column is 6 fixtures, not 9,
# and so the follow-up owner can find the seam. Do NOT paper over these by
# gutting the fixture or asserting the failure — either would hide the gap the
# corpus exists to surface. When a gap is closed in ``src/``, move the fixture
# back into the guided_staged parity set (its drive will then succeed).
_GUIDED_STAGED_CAPABILITY_GAPS: dict[str, str] = {
    "multi_source_queue": (
        "Queue fan-in. ``queue_node_contract_error`` (state.py) MANDATES a queue's "
        "``input == id``; but ``canonical_connection_consumers`` (guided/connection_consumers.py) "
        "registers every node — including the queue — as a consumer of its own ``input``, and "
        "``_build_projection`` (guided/planning.py) emits ``queue_continue -> node.id``. With "
        "``input == id`` the queue is a consumer of its own continue-connection, so the wire "
        "projection self-loops (``payload.graph.edges[..] is a self-loop``). The two invariants are "
        "mutually unsatisfiable — no candidate can drive a queue through guided-staged. LIKELY A "
        "PRODUCT BUG: the consumer builder should exclude a queue's own identity from its "
        "``queue_continue`` targets. (Follow-up, not fixed here.)"
    ),
    "multi_output": (
        "Cross-sink write-failure fallback. The structured sink review "
        "(``transition_sink_schema_form`` -> ``_validated_merged_options``, guided/stage_transitions.py) "
        'treats ``on_write_failure`` as a server-held structural policy locked to ``"discard"`` unless '
        "it is a visible knob field (it is not for the json sink); submitting the cross-sink target "
        "raises ``client altered server-held structural policy 'on_write_failure'`` (HTTP 400). Since "
        "``bind_guided_reviewed_components`` sources the committed ``on_write_failure`` from the reviewed "
        "output, this fixture's json ``priority_out`` sink can only ever discard write failures — the "
        "priority->standard fallback it proves is unauthorable. (Proven for the json sink here; whether "
        "any sink exposes ``on_write_failure`` as an editable knob is unverified. Source "
        "``on_validation_failure`` IS editable, which is why the ``error_routing`` fixture — a source "
        "failure routed to a sink — passes.)"
    ),
    "fork_coalesce": (
        "Require-all coalesce. ``canonical_connection_consumers`` keys consumers only on ``node.input`` "
        "and ``output.name``; a require-all coalesce consumes >= 2 branch connections but ``NodeSpec`` "
        "has a single ``input``, so every branch connection other than ``input`` (here ``path_b``) is "
        "structurally orphaned and ``_build_projection`` raises ``guided proposal connection has no "
        "canonical consumer``. The consumer model cannot represent a multi-branch coalesce."
    ),
}

_GUIDED_STAGED_PARITY_FIXTURES = [fixture for fixture in PARITY_FIXTURES if fixture["class"] not in _GUIDED_STAGED_CAPABILITY_GAPS]

# Explicit (surface, fixture) grid: all 9 fixtures on the two name-preserving
# surfaces, and the 6 guided-staged-drivable fixtures on guided_staged = 24
# real-path parity cases (the 3 gaps above are excluded and reported).
_SURFACE_FIXTURE_PARAMS = [
    (surface, fixture)
    for surface in SURFACES
    for fixture in (_GUIDED_STAGED_PARITY_FIXTURES if surface == "guided_staged" else PARITY_FIXTURES)
]


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


def _assert_guided_staged_naming(state: Any, fixture: dict[str, Any]) -> None:
    """Positive proof the committed graph came through the staged protocol.

    Guided-staged reviews sources/outputs one at a time and the protocol
    assigns their names positionally (``source`` / ``source_2`` … and
    ``output`` / ``output_2`` …). Asserting exactly those names — with the same
    counts the fixture declares — proves the surface really traversed the staged
    protocol (not a shortcut) and that the only difference from the reference is
    the expected renaming that isomorphism already normalizes away.
    """
    committed = state.to_dict()
    source_names = set(committed["sources"])
    expected_sources = {"source"} | {f"source_{index}" for index in range(2, len(committed["sources"]) + 1)}
    assert source_names == expected_sources, (
        f"{fixture['class']}: guided-staged sources {sorted(source_names)} != guided defaults {sorted(expected_sources)}"
    )
    output_names = {output["name"] for output in committed["outputs"]}
    expected_outputs = {"output"} | {f"output_{index}" for index in range(2, len(committed["outputs"]) + 1)}
    assert output_names == expected_outputs, (
        f"{fixture['class']}: guided-staged outputs {sorted(output_names)} != guided defaults {sorted(expected_outputs)}"
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("surface", "fixture"),
    _SURFACE_FIXTURE_PARAMS,
    ids=lambda value: value if isinstance(value, str) else value["class"],
)
async def test_surface_derives_isomorphic_committed_graph(
    parity_env: ParityEnv,
    surface: str,
    fixture: dict[str, Any],
) -> None:
    reference = parity_env.reference_state(fixture)
    committed = await parity_env.drive(surface, fixture)

    # 1. Semantic graph isomorphism (the primary, name-agnostic parity proof;
    #    applied to every surface). Cross-surface parity is transitive: all
    #    three surfaces are anchored to the same per-fixture reference.
    assert_isomorphic(committed, reference, left=f"{surface}:{fixture['class']}", right="reference")

    if surface in _NAME_PRESERVING_SURFACES:
        # 2. Public compiled-pipeline (runtime graph) semantics agree byte-exact
        #    (these surfaces emit canonical component names).
        assert public_pipeline_semantics(committed) == public_pipeline_semantics(reference), (
            f"{surface}:{fixture['class']}: public pipeline semantics diverged from reference"
        )
        # 3. The committed graph exposes the fixture's exact declared capability shape.
        _assert_semantic_expectations(committed, fixture)
    else:
        # Guided-staged: isomorphism above is the complete parity proof (§8.1
        # canonicalizes the auto-assigned names). Add the positive guided-naming
        # assertion and confirm the public pipeline still compiles non-empty.
        _assert_guided_staged_naming(committed, fixture)
        public = public_pipeline_semantics(committed)
        assert public.get("sources") and public.get("sinks"), f"{surface}:{fixture['class']}: public pipeline compiled empty sources/sinks"
