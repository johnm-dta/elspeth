"""Schema-mutation controls: narrowing the advertised terminal schema is caught
by the capability-manifest SCHEMA-IDENTITY gate, not by graph isomorphism (Plan
05 Task 4).

Why these controls exist (the false-green trap they defend against)
-------------------------------------------------------------------
The generated-DAG and fixture-matrix parity tests prove that three authoring
surfaces derive the *same committed graph* by comparing each surface's committed
``CompositionState`` to a shared reference with ``assert_isomorphic``. That proof
has a blind spot the plan calls out explicitly: the committed graph is a function
of the pipeline the LLM *emits*, not of the schema the planner *advertises* to
the LLM in the ``emit_pipeline_proposal`` terminal tool. Under this suite's
scripted completion — which emits the full canonical payload regardless of the
advertised schema — narrowing the advertised terminal schema (dropping a
capability field, capping a collection, closing an open enum) yields a
byte-identical committed graph. Graph isomorphism therefore *cannot* detect an
advertised-schema regression: a control routed through isomorphism is a control
that can never fail.

The real defense is ``build_planner_capability_manifest``
(``src/elspeth/web/composer/capability_skill.py``), which — immediately before
every planner completion call — hashes the terminal tool's advertised ``pipeline``
schema and the ground-truth ``canonical_set_pipeline_schema()`` and raises
``AuditIntegrityError`` when they diverge:

    if stable_hash(advertised_schema) != stable_hash(canonical_schema):
        raise AuditIntegrityError("planner terminal does not advertise the canonical pipeline schema")

Every control below narrows ONE named field of the freeform surface's ACTUAL
advertised terminal schema and asserts that exact gate fires — the
advertised-vs-canonical ``stable_hash`` compare (capability_skill.py ~L223-224),
NOT the tool-identity check (~L217, tool names/order are untouched) and NOT the
downstream field-contract check (~L225, unreachable because the hash compare
raises first). Each control asserts the target field is *present* before it
narrows, so a stale path can never silently no-op into a green test that
misattributes the field.

Two layers, deliberately:

* ``TestManifestSchemaIdentityGate`` calls the gate directly with the real
  production advertised tools (``planner_tool_definitions()``) and the genuine
  canonical schema. The baseline proves the direction (un-narrowed → hashes
  match, manifest builds); each narrowing proves detection (hashes diverge →
  ``AuditIntegrityError``). This is the "(or the manifest-identity assertion)"
  path the task sanctions, and it pins each control to the schema-identity
  compare with zero ambiguity about which check fired.

* ``test_freeform_drive_narrowed_advertised_schema_trips_gate_upstream_of_graph``
  drives the REAL freeform ``compose()`` path with the advertised terminal schema
  narrowed at its real production seam (patched
  ``planner_terminal_tool_definition``, resolved at call time by
  ``planner_tool_definitions``; the manifest's canonical argument uses the
  separate ``canonical_set_pipeline_schema`` binding and stays genuine). It
  confirms the gate fires in situ "immediately before completion" and — the
  concrete false-green evidence — that it fires UPSTREAM of any committed graph:
  no proposal is staged and no state is committed, so isomorphism never gets a
  graph to compare. Under the full-payload scripted completion the committed
  graph would otherwise be identical to the un-narrowed reference.
"""

from __future__ import annotations

import copy
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import pytest

import elspeth.web.composer.pipeline_planner as planner_module
from elspeth.contracts.errors import AuditIntegrityError
from elspeth.core.canonical import stable_hash
from elspeth.web.composer.capability_skill import (
    CAPABILITY_CORE_NODE_GUIDANCE,
    build_planner_capability_manifest,
    load_pipeline_capability_core,
)
from elspeth.web.composer.pipeline_planner import planner_tool_definitions
from elspeth.web.composer.pipeline_proposal import PlannerSurface
from elspeth.web.composer.tools.schema_contract import canonical_set_pipeline_schema

from .conftest import PARITY_FIXTURES, ParityEnv, _empty_state

_SCHEMA_IDENTITY_GATE_MESSAGE = "planner terminal does not advertise the canonical pipeline schema"


# --------------------------------------------------------------------------- #
# Narrowing controls — each mutates ONE named field of the advertised pipeline #
# schema in place, after asserting the field is present at the named path.     #
# --------------------------------------------------------------------------- #


def _node_props(pipeline: dict[str, Any]) -> dict[str, Any]:
    """Return the advertised ``nodes[].properties`` map (the node item schema)."""
    return pipeline["properties"]["nodes"]["items"]["properties"]


def _remove_fork_to(pipeline: dict[str, Any]) -> None:
    """Drop ``nodes[].fork_to`` — narrows away every fork-based topology."""
    props = _node_props(pipeline)
    assert "fork_to" in props, "control precondition: advertised nodes[].fork_to must exist to be narrowed away"
    del props["fork_to"]


def _remove_merge(pipeline: dict[str, Any]) -> None:
    """Drop ``nodes[].merge`` — narrows away the coalesce merge strategy."""
    props = _node_props(pipeline)
    assert "merge" in props, "control precondition: advertised nodes[].merge must exist to be narrowed away"
    del props["merge"]


def _enum_node_type_omitting_queue(pipeline: dict[str, Any]) -> None:
    """Close ``nodes[].node_type`` to an enum that omits ``queue``.

    The canonical schema advertises ``node_type`` as an open ``{"type": "string"}``
    — there is no existing enum to shrink, so introducing one is itself the
    structural narrowing that changes the schema hash. The enum lists every real
    node family (``CAPABILITY_CORE_NODE_GUIDANCE``) except ``queue``, so a queue
    node is no longer advertised as authorable.
    """
    node_type = _node_props(pipeline)["node_type"]
    assert "enum" not in node_type, (
        "control precondition: advertised nodes[].node_type must be an open string (no enum) "
        "so that introducing one is a genuine structural narrowing"
    )
    assert "queue" in CAPABILITY_CORE_NODE_GUIDANCE, "queue must be a real node family for its omission to narrow"
    node_type["enum"] = [family for family in sorted(CAPABILITY_CORE_NODE_GUIDANCE) if family != "queue"]


def _cap_named_sources_at_one(pipeline: dict[str, Any]) -> None:
    """Cap the ``sources`` object at one entry — narrows away multi-source."""
    sources = pipeline["properties"]["sources"]
    assert "maxProperties" not in sources, "control precondition: advertised sources must be uncapped to be narrowed"
    sources["maxProperties"] = 1


@dataclass(frozen=True)
class _NarrowingControl:
    """One named narrowing of the advertised terminal ``pipeline`` schema."""

    field: str
    narrow: Callable[[dict[str, Any]], None]


_CONTROLS: tuple[_NarrowingControl, ...] = (
    _NarrowingControl("nodes[].fork_to", _remove_fork_to),
    _NarrowingControl("nodes[].merge", _remove_merge),
    _NarrowingControl("nodes[].node_type[enum-omit-queue]", _enum_node_type_omitting_queue),
    _NarrowingControl("sources[maxProperties=1]", _cap_named_sources_at_one),
)


# --------------------------------------------------------------------------- #
# Layer 1 — direct manifest gate                                              #
# --------------------------------------------------------------------------- #


def _planner_messages() -> list[dict[str, Any]]:
    """Minimal messages that satisfy the manifest's pre-schema checks.

    The first system message must start with (and contain exactly once) the
    capability core, else ``build_planner_capability_manifest`` raises on the
    system-message check before it reaches the advertised-vs-canonical compare.
    """
    core = load_pipeline_capability_core()
    return [
        {"role": "system", "content": core},
        {"role": "user", "content": "schema-mutation control probe"},
    ]


def _advertised_pipeline(tools: list[dict[str, Any]]) -> dict[str, Any]:
    """The terminal tool's advertised ``pipeline`` schema (what the gate hashes)."""
    return tools[-1]["function"]["parameters"]["properties"]["pipeline"]


def _build_manifest(tools: list[dict[str, Any]]) -> Any:
    """Run the real gate for the freeform surface against the genuine canonical schema."""
    return build_planner_capability_manifest(
        surface=PlannerSurface.FREEFORM,
        profile="ordinary",
        messages=_planner_messages(),
        tools=tools,
        canonical_schema=canonical_set_pipeline_schema(),
    )


class TestManifestSchemaIdentityGate:
    """The advertised-vs-canonical ``stable_hash`` compare, exercised directly."""

    def test_baseline_un_narrowed_schema_matches(self) -> None:
        """Direction, positive half: the real advertised schema equals canonical.

        With no narrowing, the freeform surface's advertised terminal schema
        hashes identically to ``canonical_set_pipeline_schema()``, so the gate
        builds a manifest whose recorded ``canonical_schema_hash`` is that shared
        digest — no ``AuditIntegrityError``.
        """
        tools = planner_tool_definitions()
        advertised = _advertised_pipeline(tools)
        assert stable_hash(advertised) == stable_hash(canonical_set_pipeline_schema())

        manifest = _build_manifest(tools)
        assert manifest.canonical_schema_hash == stable_hash(advertised)

    @pytest.mark.parametrize("control", _CONTROLS, ids=lambda control: control.field)
    def test_narrowing_trips_schema_identity_gate(self, control: _NarrowingControl) -> None:
        """Direction, negative half: narrowing ONE field diverges the hash and fires the gate.

        The narrowing touches only the advertised ``pipeline`` schema (never a
        tool name or order), so the tool-identity check passes and control lands
        on the schema-identity compare. The genuine canonical schema is supplied
        unchanged, so advertised != canonical and the gate raises. Pinning the
        exact message proves it is the schema-identity compare that fired, not the
        tool-identity or field-contract check.
        """
        tools = planner_tool_definitions()
        advertised = _advertised_pipeline(tools)

        control.narrow(advertised)  # asserts the named field is present, then narrows it
        # The narrowing is a genuine structural change to the advertised schema.
        assert stable_hash(advertised) != stable_hash(canonical_set_pipeline_schema()), (
            f"{control.field}: narrowing did not change the advertised schema hash"
        )

        with pytest.raises(AuditIntegrityError) as excinfo:
            _build_manifest(tools)
        assert str(excinfo.value) == _SCHEMA_IDENTITY_GATE_MESSAGE


# --------------------------------------------------------------------------- #
# Layer 2 — real freeform surface drive at the production seam                #
# --------------------------------------------------------------------------- #


def _fixture(class_name: str) -> dict[str, Any]:
    return copy.deepcopy(next(fixture for fixture in PARITY_FIXTURES if fixture["class"] == class_name))


@pytest.mark.asyncio
async def test_freeform_drive_narrowed_advertised_schema_trips_gate_upstream_of_graph(
    parity_env: ParityEnv,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Narrowing the freeform surface's ACTUAL advertised terminal schema, in situ.

    ``planner_tool_definitions`` resolves ``planner_terminal_tool_definition`` at
    call time, so patching that module global narrows the advertised terminal
    schema that ``plan_pipeline`` sends to the completion. The manifest's
    canonical argument uses the separate ``canonical_set_pipeline_schema`` binding
    and stays genuine — so only the advertised side narrows, and the gate fires
    "immediately before completion".

    The ``fork_coalesce`` fixture is chosen because its full canonical payload
    exercises ``fork_to``: the scripted completion still emits it verbatim, so the
    committed graph would be byte-identical to the un-narrowed reference. That the
    drive instead fails at the gate — with NO proposal staged and NO state
    committed — is the concrete proof that the narrowing is caught upstream of any
    graph, where isomorphism can never reach it. (The inline setup mirrors
    ``ParityEnv.drive_freeform`` but retains the session handle past the expected
    failure so the upstream-of-graph assertions can inspect it.)
    """
    real_terminal = planner_module.planner_terminal_tool_definition

    def _narrowed_terminal() -> dict[str, Any]:
        definition = real_terminal()
        _remove_fork_to(definition["function"]["parameters"]["properties"]["pipeline"])
        return definition

    monkeypatch.setattr(planner_module, "planner_terminal_tool_definition", _narrowed_terminal)

    fixture = _fixture("fork_coalesce")
    session = await parity_env.sessions.create_session("alice", "Alice", "local")
    parity_env._script(fixture, str(session.id))
    await parity_env.sessions.update_composer_preferences(session.id, trust_mode="explicit_approve", density_default="high", actor="test")
    user_message = await parity_env.sessions.add_message(session.id, "user", fixture["intent"], writer_principal="route_user_message")

    with pytest.raises(AuditIntegrityError) as excinfo:
        await parity_env.composer.compose(
            fixture["intent"],
            [],
            _empty_state(),
            session_id=str(session.id),
            user_id="alice",
            user_message_id=str(user_message.id),
        )
    assert str(excinfo.value) == _SCHEMA_IDENTITY_GATE_MESSAGE

    # The gate fired upstream of any committed graph: isomorphism never ran.
    assert await parity_env.sessions.get_current_state(session.id) is None
    pending = await parity_env.sessions.list_composition_proposals(session.id, status="pending")
    assert len(pending) == 0
