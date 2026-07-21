"""Negative / behavioural parity cases for the capability-parity matrix (Plan 05 Task 3).

The positive matrix (``test_fixture_matrix.py``) proves that every ordinary
authoring surface derives the *same committed graph* for a well-behaved planner.
These cases prove the surfaces stay at parity when the planner misbehaves or the
operator drives the protocol off the happy path — the behaviours the corpus
cannot express as a committed graph:

* **one-repair success** — a first *malformed* terminal proposal (a pipeline
  missing a required top-level field) trips the planner's canonical-schema
  repair feedback; the very next terminal is valid and the run converges to the
  same committed graph the reference derives, carrying ``repair_count == 1``.
* **repair exhaustion** — every terminal proposal stays malformed; the planner
  exhausts its repair budget (2) and raises ``REPAIR_EXHAUSTED``, which the
  freeform ``/messages`` route (Task 0) translates into a deliberate ``502`` and
  one durable, redacted ``planner_failure_disposition`` audit row — NOT a guided
  lease terminalization.
* **policy rejection** — a first terminal names a policy-denied plugin
  (installed but absent from this operator's allowlist); candidate validation
  rejects it and the planner is handed the *allowlisted structured* candidate
  feedback (``error_class == "ValidationError"``, only ``component`` /
  ``severity`` / ``error_code`` / ``error_class`` — no message, no plugin name,
  no options), then a valid terminal converges. This is distinct from
  one-repair, which trips the *schema* feedback (``SchemaValidationError``).
* **unavailable plugin** — the guided-staged source ``single_select`` turn only
  offers the operator-available plugins; choosing one absent from the snapshot
  (``aws_s3``, installed-but-unauthorized under the parity policy) is rejected
  structurally, so an unavailable plugin can never enter a guided commit.
* **tutorial identity** — a ``tutorial``-profile guided-staged run reaches the
  same committed graph as the ``live`` staged run and the reference, while its
  sole planner call runs on the ``TUTORIAL_PROFILE`` surface with ``profile ==
  "tutorial"`` (its frozen lesson). Same planner, same schema, same commit — only
  the frozen lesson differs.
* **future-stage retention across restart** — a wrong-stage instruction is
  retained as a private deferred intent and survives a full stack restart rather
  than being discarded.
* **completed-stage back/edit** — editing a retained (already-reviewed) intent
  preserves its stable id and rewinds the pending proposal by that id.

**Harness split (deliberate).** The freeform repair/rejection cases and the two
guided-staged cases that need the *real* planner + real operator snapshot
(unavailable-plugin, tutorial-identity) reuse the parity production stack
(``parity_env``). The two deferred-intent cases need a scripted guided chat
provider *and* a restart-capable stack, both of which the guided integration
fixture (``composer_test_client``) already provides — so they reuse it plus that
suite's helpers rather than rebuilding restart + provider scripting on
``parity_env`` (no coverage gain, real cost). ``unavailable-plugin`` uses
``parity_env`` (not ``composer_test_client``) precisely because it wants the real
availability snapshot the parity policy produces, which is what "absent from the
trained-operator snapshot" means.

No provider network, no skips, no xfail.
"""

from __future__ import annotations

import copy
import json
from dataclasses import replace
from typing import Any
from uuid import uuid4

import pytest
from sqlalchemy import select

from elspeth.web.composer import pipeline_planner as planner_module
from elspeth.web.composer.guided.deferred_intents import DeferredIntentAction, DeferredIntentEditAction
from elspeth.web.composer.guided.intent_management import deferred_intent_management_option
from elspeth.web.composer.guided.planning import guided_private_reviewed_facts
from elspeth.web.composer.guided.stage_subjects import ComponentCountConstraint
from elspeth.web.composer.pipeline_proposal import PipelineProposal
from elspeth.web.sessions.models import chat_messages_table, composition_proposals_table
from elspeth.web.sessions.routes.composer import guided as guided_route
from tests.helpers.composer_graphs import assert_isomorphic

# Guided deferred-intent suite helpers. The restart-capable ``composer_test_client``
# fixture the two deferred-intent cases below depend on is re-registered for this
# package in ``parity/conftest.py`` (importing it here would shadow the fixture
# parameter and trip F811).
# Aliased to a non-``Test*`` name so pytest does not re-collect its methods here.
from tests.integration.web.composer.guided.test_respond import TestStep2IntraStep as _Step2Stager
from tests.integration.web.composer.guided.test_step_chat import _create_session
from tests.integration.web.composer.guided.test_wrong_stage_intent import (
    _action,
    _guided,
    _management_provider,
    _post,
    _provider,
)

from .conftest import (
    PARITY_FIXTURES,
    ParityEnv,
    _empty_state,
    _ScriptedCompletion,
    emit_proposal_response,
    rewrite_source_paths,
)

# The simplest guided-staged-drivable capability class: single csv source, one
# passthrough transform, one json sink. Every negative that needs a "valid"
# terminal reuses it so the assertions stay about the *behaviour*, not the shape.
_LINEAR = next(fixture for fixture in PARITY_FIXTURES if fixture["class"] == "linear_transform")

# A real transform plugin that is installed in the catalog but NOT admitted by the
# parity web policy (absent from ``_PARITY_ALLOWLIST`` and ``REQUIRED_WEB_PLUGIN_IDS``),
# so a candidate that names it is a genuine policy denial.
_POLICY_DENIED_TRANSFORM = "truncate"

# A real source plugin installed in the catalog but unavailable under the parity
# policy, so it never appears in the guided source ``single_select`` permitted set.
_UNAVAILABLE_SOURCE = "aws_s3"


# --------------------------------------------------------------------------- #
# Freeform helpers                                                            #
# --------------------------------------------------------------------------- #


def _valid_pipeline(env: ParityEnv, fixture: dict[str, Any]) -> dict[str, Any]:
    """The fixture's canonical pipeline with source paths bound under the S2 allowlist."""
    return rewrite_source_paths(fixture["canonical_arguments"], env.data_dir)


def _malformed_missing_edges(pipeline: dict[str, Any]) -> dict[str, Any]:
    """A schema-malformed terminal: drop the required top-level ``edges`` key.

    ``SetPipelineArgumentsModel`` requires ``edges``; omitting it fails the
    terminal schema check (``_canonical_schema_feedback``) — a genuine malformed
    terminal, not a candidate-validation rejection.
    """
    return {key: value for key, value in pipeline.items() if key != "edges"}


def _policy_denied_variant(pipeline: dict[str, Any]) -> dict[str, Any]:
    """A shape-valid terminal that names a policy-denied transform plugin.

    Passes ``SetPipelineArgumentsModel`` (structurally a valid transform node) but
    fails ``build_set_pipeline_candidate`` authorization, so the planner receives
    the allowlisted *candidate* feedback (``ValidationError``), not schema feedback.
    """
    denied = copy.deepcopy(pipeline)
    denied["nodes"][0]["plugin"] = _POLICY_DENIED_TRANSFORM
    return denied


def _last_tool_feedback(request: dict[str, Any]) -> dict[str, Any]:
    """Parse the last ``role="tool"`` message the planner fed back into a request."""
    messages = request["messages"]
    tool_messages = [message for message in messages if message.get("role") == "tool"]
    if not tool_messages:
        raise AssertionError("scripted request carried no tool-role feedback message")
    return json.loads(tool_messages[-1]["content"])


async def _drive_freeform_scripted(
    env: ParityEnv,
    fixture: dict[str, Any],
    completion: _ScriptedCompletion,
) -> tuple[Any, Any]:
    """Run the real freeform ``compose`` empty-build path under ``completion``, then accept.

    Mirrors ``ParityEnv.drive_freeform`` but injects a caller-supplied multi-response
    completion (the positive driver scripts exactly one). Returns the committed
    ``CompositionState`` and the accepted proposal record.
    """
    env.monkeypatch.setattr("elspeth.web.composer.service._litellm_acompletion", completion)
    session = await env.sessions.create_session("alice", "Alice", "local")
    await env.sessions.update_composer_preferences(
        session.id,
        trust_mode="explicit_approve",
        density_default="high",
        actor="test",
    )
    user_message = await env.sessions.add_message(
        session.id,
        "user",
        fixture["intent"],
        writer_principal="route_user_message",
    )
    await env.composer.compose(
        fixture["intent"],
        [],
        _empty_state(),
        session_id=str(session.id),
        user_id="alice",
        user_message_id=str(user_message.id),
    )
    proposals = await env.sessions.list_composition_proposals(session.id, status="pending")
    if len(proposals) != 1:
        raise AssertionError(f"freeform scripted repair staged {len(proposals)} proposals, expected exactly one")
    proposal = proposals[0]
    async with env._client() as client:
        response = await client.post(
            f"/api/sessions/{session.id}/proposals/{proposal.id}/accept",
            json={"draft_hash": proposal.pipeline_metadata.draft_hash},
        )
    if response.status_code != 200:
        raise AssertionError(f"freeform scripted repair accept failed ({response.status_code}): {response.text}")
    return await env._committed_state(session.id), proposal


def _disposition_rows(engine: Any) -> list[Any]:
    """The durable ``planner_failure_disposition`` audit rows (freeform terminalization)."""
    with engine.connect() as conn:
        rows = conn.execute(select(chat_messages_table.c.role, chat_messages_table.c.tool_calls)).all()
    return [
        row for row in rows if row.role == "audit" and row.tool_calls and row.tool_calls[0].get("_kind") == "planner_failure_disposition"
    ]


def _assert_allowlisted_feedback_shape(feedback: dict[str, Any], *, error_class: str) -> None:
    """The repair feedback is the redaction-safe structured projection only."""
    assert set(feedback) == {"success", "validation"}, feedback
    assert feedback["success"] is False
    validation = feedback["validation"]
    assert set(validation) == {"is_valid", "errors"}, validation
    assert validation["is_valid"] is False
    assert validation["errors"], "expected at least one structured error"
    for entry in validation["errors"]:
        assert set(entry) == {"component", "severity", "error_code", "error_class"}, entry
        assert entry["error_class"] == error_class
    # No provider prose, plugin name, or option value may ride the feedback.
    blob = json.dumps(feedback)
    assert _POLICY_DENIED_TRANSFORM not in blob
    assert "options" not in blob


# --------------------------------------------------------------------------- #
# Freeform: repair success / exhaustion / policy rejection                    #
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_freeform_one_repair_converges_to_reference_graph(parity_env: ParityEnv) -> None:
    """A malformed terminal, then a valid one: converges with repair_count == 1."""
    reference = parity_env.reference_state(_LINEAR)
    valid = _valid_pipeline(parity_env, _LINEAR)
    completion = _ScriptedCompletion(
        emit_proposal_response(_malformed_missing_edges(valid)),
        emit_proposal_response(valid),
    )

    committed, proposal = await _drive_freeform_scripted(parity_env, _LINEAR, completion)

    assert_isomorphic(committed, reference, left="freeform-one-repair:linear_transform", right="reference")
    assert proposal.pipeline_metadata.repair_count == 1
    assert len(completion.requests) == 2, "exactly one malformed attempt plus one valid attempt"
    # The repair feedback that provoked the second attempt is the schema projection.
    _assert_allowlisted_feedback_shape(_last_tool_feedback(completion.requests[1]), error_class="SchemaValidationError")


@pytest.mark.asyncio
async def test_freeform_repair_exhaustion_is_translated_to_a_safe_disposition(parity_env: ParityEnv) -> None:
    """Every terminal malformed: REPAIR_EXHAUSTED → 502 + one closed disposition row."""
    valid = _valid_pipeline(parity_env, _LINEAR)
    malformed = emit_proposal_response(_malformed_missing_edges(valid))
    # Budget is 2 (parity settings inherit the WebSettings default); the third
    # malformed terminal makes repair_count == 3 > 2, which now engages the
    # escape-hatch overtime turn on the advisor model. A fourth malformed
    # terminal spends the hatch, so the original REPAIR_EXHAUSTED stands.
    completion = _ScriptedCompletion(malformed, malformed, malformed, malformed)
    parity_env.monkeypatch.setattr("elspeth.web.composer.service._litellm_acompletion", completion)

    async with parity_env._client() as client:
        created = await client.post("/api/sessions", json={"title": "freeform repair exhaustion"})
        assert created.status_code == 201, created.text
        session_id = created.json()["id"]
        response = await client.post(f"/api/sessions/{session_id}/messages", json={"content": _LINEAR["intent"]})

    assert response.status_code == 502, response.text
    detail = response.json()["detail"]
    assert detail["error_type"] == "composer_planner_failure"
    assert detail["failure_code"] == "invalid_provider_response"
    assert len(completion.requests) == 4, "repair_budget + 1 primary calls, then the spent escape-hatch turn"
    assert completion.requests[3]["model"] != completion.requests[0]["model"], "overtime turn runs on the advisor model"

    rows = _disposition_rows(parity_env.app.state.session_engine)
    assert len(rows) == 1, "exactly one durable closed failure-disposition audit row"
    envelope = rows[0].tool_calls[0]
    assert envelope["failure_code"] == "invalid_provider_response"
    assert envelope["surface"] == "freeform"


@pytest.mark.asyncio
async def test_freeform_policy_denied_candidate_is_rejected_with_allowlisted_shape(parity_env: ParityEnv) -> None:
    """A policy-denied plugin is rejected with the allowlisted candidate feedback, then repaired."""
    reference = parity_env.reference_state(_LINEAR)
    valid = _valid_pipeline(parity_env, _LINEAR)
    completion = _ScriptedCompletion(
        emit_proposal_response(_policy_denied_variant(valid)),
        emit_proposal_response(valid),
    )

    committed, proposal = await _drive_freeform_scripted(parity_env, _LINEAR, completion)

    assert_isomorphic(committed, reference, left="freeform-policy-rejection:linear_transform", right="reference")
    assert proposal.pipeline_metadata.repair_count == 1
    assert len(completion.requests) == 2
    # Candidate authorization rejection → ValidationError projection (NOT the schema
    # projection the one-repair case trips), carrying no denied-plugin leak.
    _assert_allowlisted_feedback_shape(_last_tool_feedback(completion.requests[1]), error_class="ValidationError")


# --------------------------------------------------------------------------- #
# Guided-staged: unavailable plugin (real operator snapshot)                  #
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_guided_staged_rejects_source_plugin_absent_from_snapshot(parity_env: ParityEnv) -> None:
    """The source single_select only offers available plugins; a snapshot-absent one is rejected."""
    async with parity_env._client() as client:
        created = await client.post("/api/sessions", json={"title": "guided unavailable plugin"})
        assert created.status_code == 201, created.text
        session_id = created.json()["id"]

        current = await client.get(f"/api/sessions/{session_id}/guided")
        assert current.status_code == 200, current.text
        turn = current.json()["next_turn"]
        assert turn["type"] == "single_select", turn
        permitted = {option["id"] for option in turn["payload"]["options"]}
        # The unavailable plugin is genuinely installed but not offered.
        assert _UNAVAILABLE_SOURCE not in permitted, f"{_UNAVAILABLE_SOURCE} must be absent from the operator snapshot"

        rejected = await client.post(
            f"/api/sessions/{session_id}/guided/respond",
            json={"operation_id": str(uuid4()), "turn_token": turn["turn_token"], "chosen": [_UNAVAILABLE_SOURCE]},
        )
        assert rejected.status_code == 400, rejected.text
        assert _UNAVAILABLE_SOURCE not in rejected.text, "the denied plugin name must not leak into the rejection"

        # The session stays on the source single_select turn — the unavailable
        # plugin never entered the guided state.
        after = await client.get(f"/api/sessions/{session_id}/guided")
        assert after.status_code == 200, after.text
        assert after.json()["next_turn"]["type"] == "single_select"


# --------------------------------------------------------------------------- #
# Guided-staged: tutorial identity                                            #
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_tutorial_reaches_same_commit_as_staged_with_its_fixed_lesson(parity_env: ParityEnv) -> None:
    """A tutorial-profile staged run commits the same graph as staged, via the tutorial surface."""
    reference = parity_env.reference_state(_LINEAR)

    manifests: list[Any] = []
    real_builder = planner_module.build_planner_capability_manifest

    def capture_manifest(**kwargs: Any) -> Any:
        manifest = real_builder(**kwargs)
        manifests.append(manifest)
        return manifest

    parity_env.monkeypatch.setattr(planner_module, "build_planner_capability_manifest", capture_manifest)

    committed = await parity_env.drive_guided_staged(_LINEAR, start_profile="tutorial")

    # Same commit as staged: the positive matrix proves live-staged ≅ reference for
    # this fixture, so tutorial ≅ reference proves tutorial ≅ staged transitively.
    assert_isomorphic(committed, reference, left="tutorial:linear_transform", right="reference")
    # Its fixed lesson: the sole planner call ran on the tutorial surface/profile.
    assert len(manifests) == 1, "guided-staged makes exactly one planner call (finish outputs)"
    assert manifests[0].surface.value == "tutorial_profile"
    assert manifests[0].profile == "tutorial"


# --------------------------------------------------------------------------- #
# Guided deferred intents: retention across restart / completed-stage edit    #
# (guided integration fixture — needs restart + scripted chat provider)       #
# --------------------------------------------------------------------------- #


def test_guided_future_stage_intent_is_retained_across_a_full_restart(
    composer_test_client: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A wrong-stage instruction becomes a private deferred intent that survives a restart."""
    client = composer_test_client
    session_id = _create_session(client)
    turn = client.get(f"/api/sessions/{session_id}/guided").json()["next_turn"]
    monkeypatch.setattr(guided_route, "_run_guided_chat_provider_attempt", _provider(_action()))

    retained = _post(
        client,
        session_id,
        operation_id=str(uuid4()),
        turn_token=turn["turn_token"],
        message="Later use passthrough during topology authoring.",
    )
    assert retained.status_code == 200, retained.json()
    intents = _guided(client, session_id).deferred_intents
    assert len(intents) == 1
    assert intents[0].target_stage == "topology"

    # Rebuild the entire HTTP/service stack over the persisted store, then reload.
    restarted = client.app.state.restart_test_client()
    refreshed = restarted.get(f"/api/sessions/{session_id}/guided")
    assert refreshed.status_code == 200, refreshed.json()
    assert _guided(restarted, session_id).deferred_intents == intents, "deferred intent must survive restart, not be discarded"


def _output_intent(*, count: int, summary: str) -> DeferredIntentAction:
    """A sink-stage intent tied to the reviewed json output (a passed-stage subject)."""
    return DeferredIntentAction(
        target_stage="output",
        catalog_kind="sink",
        catalog_name="json",
        redacted_summary=summary,
        constraints=(
            ComponentCountConstraint(
                kind="component_count",
                component_kind="output",
                plugin_kind="sink",
                plugin_name="json",
                operator="at_least",
                count=count,
            ),
        ),
    )


def test_guided_completed_stage_edit_preserves_stable_id_and_rewinds_proposal(
    composer_test_client: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Editing an already-reviewed (passed-through-wire) output rewinds the proposal by its stable id.

    Distinct from the future-stage retention case above: here the output stage is
    completed (reviewed *and* wire-reviewed at ``step_4_wire``) before the operator
    goes back to edit it. The edit preserves the reviewed intent's stable id and
    rewinds the pending proposal to the output-review stage.
    """
    client = composer_test_client
    session_id = _create_session(client)
    initial = client.get(f"/api/sessions/{session_id}/guided").json()
    monkeypatch.setattr(
        guided_route,
        "_run_guided_chat_provider_attempt",
        _provider(_output_intent(count=1, summary="Retain one JSON output requirement.")),
    )
    retained_response = _post(
        client,
        session_id,
        operation_id=str(uuid4()),
        turn_token=initial["next_turn"]["turn_token"],
        message="Keep this output instruction pending through proposal review.",
    )
    assert retained_response.status_code == 200, retained_response.json()
    (retained,) = _guided(client, session_id).deferred_intents

    # Keep the intent unclaimed so it survives to be edited after the wire review.
    planner = client.app.state.composer_service
    real_plan = planner.plan_guided_pipeline

    async def plan_without_claiming_intent(*, guided: Any, **kwargs: Any) -> Any:
        plan, catalog_plugin_ids = await real_plan(guided=guided, **kwargs)
        proposal = plan.proposal
        unclaimed = PipelineProposal.create(
            pipeline=proposal.pipeline,
            base=proposal.base,
            reviewed_facts=guided_private_reviewed_facts(guided),
            surface=proposal.surface,
            repair_count=proposal.repair_count,
            skill_hash=proposal.skill_hash,
            covered_deferred_intent_ids=(),
            supersedes_draft_hash=proposal.supersedes_draft_hash,
        )
        return replace(plan, proposal=unclaimed), catalog_plugin_ids

    monkeypatch.setattr(planner, "plan_guided_pipeline", plan_without_claiming_intent)
    staged = _Step2Stager()._stage_proposal(client, session_id, filename="parity-passed-output.jsonl")
    proposal = staged["next_turn"]["payload"]
    proposal_id = proposal["proposal_id"]

    # Complete the wire-review stage (step_4_wire) — the output is now a passed stage.
    reviewed = client.post(
        f"/api/sessions/{session_id}/guided/respond",
        json={
            "operation_id": str(uuid4()),
            "turn_token": staged["next_turn"]["turn_token"],
            "proposal_id": proposal_id,
            "draft_hash": proposal["draft_hash"],
            "chosen": ["review_wiring"],
        },
    )
    assert reviewed.status_code == 200, reviewed.json()
    assert reviewed.json()["guided_session"]["step"] == "step_4_wire"
    assert [intent.intent_id for intent in _guided(client, session_id).deferred_intents] == [retained.intent_id]

    edit = DeferredIntentEditAction(
        intent_id=retained.intent_id,
        selection_token=deferred_intent_management_option(retained).selection_token,
        replacement=_output_intent(count=2, summary="Retain the revised JSON output requirement."),
    )
    monkeypatch.setattr(guided_route, "_run_guided_chat_provider_attempt", _management_provider(edit))
    edited = _post(
        client,
        session_id,
        operation_id=str(uuid4()),
        turn_token=reviewed.json()["next_turn"]["turn_token"],
        message="Revise the passed output instruction.",
    )

    assert edited.status_code == 200, edited.json()
    body = edited.json()
    assert body["guided_session"]["step"] == "step_2_sink", "the edit rewinds back to the output-review stage"
    assert body["next_turn"]["payload"]["component_kind"] == "output"
    (revised,) = _guided(client, session_id).deferred_intents
    assert revised.intent_id == retained.intent_id, "the edit addresses and preserves the reviewed intent's stable id"
    assert revised.constraints[0].to_dict()["count"] == 2, "the replacement content took effect"

    with client.app.state.session_engine.connect() as connection:
        proposal_row = (
            connection.execute(select(composition_proposals_table).where(composition_proposals_table.c.id == proposal_id)).mappings().one()
        )
    assert proposal_row["status"] == "rejected", "the reviewed pending proposal is rewound by the edit"
