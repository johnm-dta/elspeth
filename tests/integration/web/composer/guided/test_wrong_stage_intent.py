"""Atomic retention of guided instructions given at the wrong stage."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import replace
from types import SimpleNamespace
from uuid import UUID, uuid4

import pytest
from sqlalchemy import event, func, select

from elspeth.contracts.composer_llm_audit import ComposerChatTurnStatus
from elspeth.contracts.freeze import deep_thaw
from elspeth.core.canonical import stable_hash
from elspeth.web.catalog.policy_view import PolicyCatalogView
from elspeth.web.catalog.schemas import PluginKind, PluginSchemaInfo, PluginSummary
from elspeth.web.composer.guided import chat_solver
from elspeth.web.composer.guided.deferred_intents import (
    DeferredIntentAction,
    DeferredIntentCancelAction,
    DeferredIntentEditAction,
    DeferredIntentManagementAction,
)
from elspeth.web.composer.guided.intent_management import deferred_intent_management_option
from elspeth.web.composer.guided.planning import guided_private_reviewed_facts
from elspeth.web.composer.guided.stage_subjects import (
    ComponentCountConstraint,
    EdgeRouteConstraint,
    OptionValueConstraint,
    PluginSubject,
    SubjectPresenceConstraint,
)
from elspeth.web.composer.guided.state_machine import GuidedSession
from elspeth.web.composer.pipeline_proposal import PipelineProposal
from elspeth.web.plugin_policy.models import PluginAvailability, PluginAvailabilitySnapshot, PluginId, PluginUnavailableReason
from elspeth.web.sessions._guided_step_chat import (
    GuidedStepDeferredIntentResult,
    GuidedStepDeferredManagementResult,
    StepChatResult,
)
from elspeth.web.sessions.converters import state_from_record
from elspeth.web.sessions.models import (
    chat_messages_table,
    composition_proposals_table,
    composition_states_table,
    guided_operation_events_table,
    guided_operations_table,
    proposal_events_table,
)
from elspeth.web.sessions.protocol import GuidedOriginatingUserMessageDraft
from elspeth.web.sessions.routes.composer import guided as guided_route
from elspeth.web.sessions.routes.composer import guided_chat_atomic
from elspeth.web.sessions.routes.composer.guided_chat_atomic import GuidedChatProviderOutcome
from tests.integration.web.composer.guided.test_respond import TestStep2IntraStep
from tests.integration.web.composer.guided.test_step_chat import TestStepChatCrossStep, _create_session
from tests.unit.web._sync_asgi_client import SyncASGITestClient as TestClient


def _action(
    *,
    target_stage: str = "topology",
    catalog_kind: str = "transform",
    catalog_name: str = "passthrough",
    count: int = 1,
) -> DeferredIntentAction:
    component_kind = {"source": "source", "transform": "node", "sink": "output"}[catalog_kind]
    return DeferredIntentAction(
        target_stage=target_stage,  # type: ignore[arg-type]
        catalog_kind=catalog_kind,  # type: ignore[arg-type]
        catalog_name=catalog_name,
        redacted_summary=f"Include the named {catalog_kind} during {target_stage} authoring.",
        constraints=(
            ComponentCountConstraint(
                kind="component_count",
                component_kind=component_kind,  # type: ignore[arg-type]
                plugin_kind=catalog_kind,  # type: ignore[arg-type]
                plugin_name=catalog_name,
                operator="at_least",
                count=count,
            ),
        ),
    )


def _provider(action: DeferredIntentAction) -> Callable[..., Awaitable[GuidedChatProviderOutcome]]:
    async def run(**_kwargs: object) -> GuidedChatProviderOutcome:
        return GuidedStepDeferredIntentResult(
            chat=StepChatResult(
                assistant_message="provider provisional text must not become authority",
                status=ComposerChatTurnStatus.SUCCESS,
                latency_ms=1,
                error_class=None,
            ),
            action=action,
        )

    return run


def _management_provider(action: DeferredIntentManagementAction) -> Callable[..., Awaitable[GuidedChatProviderOutcome]]:
    async def run(**_kwargs: object) -> GuidedChatProviderOutcome:
        return GuidedStepDeferredManagementResult(
            chat=StepChatResult(
                assistant_message="provider provisional management text must not become authority",
                status=ComposerChatTurnStatus.SUCCESS,
                latency_ms=1,
                error_class=None,
            ),
            action=action,
        )

    return run


class _Catalog:
    def __init__(
        self,
        plugins: tuple[tuple[PluginKind, str], ...],
        schemas: dict[tuple[PluginKind, str], dict[str, object]] | None = None,
        schema_overrides: dict[tuple[PluginKind, str], PluginSchemaInfo] | None = None,
    ) -> None:
        self._plugins = plugins
        self._schemas = schemas or {}
        self._schema_overrides = schema_overrides or {}

    def _list(self, kind: PluginKind) -> list[PluginSummary]:
        return [
            PluginSummary(name=name, description=name, plugin_type=kind, config_fields=[])
            for plugin_kind, name in self._plugins
            if plugin_kind == kind
        ]

    def list_sources(self) -> list[PluginSummary]:
        return self._list("source")

    def list_transforms(self) -> list[PluginSummary]:
        return self._list("transform")

    def list_sinks(self) -> list[PluginSummary]:
        return self._list("sink")

    def get_schema(self, plugin_type: PluginKind, name: str) -> PluginSchemaInfo:
        overridden = self._schema_overrides.get((plugin_type, name))
        if overridden is not None:
            return overridden
        json_schema = self._schemas.get((plugin_type, name))
        if json_schema is None:
            raise AssertionError("wrong-stage integration must inspect schemas only for option-value constraints")
        return PluginSchemaInfo(
            name=name,
            plugin_type=plugin_type,
            description=name,
            json_schema=json_schema,
            knob_schema={"fields": []},
        )

    def post_call_hints(
        self,
        *,
        plugin_type: PluginKind,
        plugin_name: str,
        tool_name: str,
        config_snapshot: dict[str, object],
    ) -> tuple[str, ...]:
        raise AssertionError("wrong-stage integration must not dispatch plugins")


def _policy_context(
    installed: tuple[tuple[PluginKind, str], ...],
    *,
    available: frozenset[PluginId],
    schemas: dict[tuple[PluginKind, str], dict[str, object]] | None = None,
    schema_overrides: dict[tuple[PluginKind, str], PluginSchemaInfo] | None = None,
) -> tuple[PolicyCatalogView, PluginAvailabilitySnapshot]:
    snapshot = PluginAvailabilitySnapshot.create(
        policy_hash="a" * 64,
        principal_scope="local:alice",
        available=available,
        unavailable=tuple(
            PluginAvailability(plugin_id=PluginId(kind, name), reason=PluginUnavailableReason.NOT_AUTHORIZED)
            for kind, name in installed
            if PluginId(kind, name) not in available
        ),
        selected=(),
        usable_profile_aliases=(),
        selected_profile_aliases=(),
        binding_generation_fingerprint="b" * 64,
    )
    return PolicyCatalogView(_Catalog(installed, schemas, schema_overrides), snapshot, profiles=None), snapshot  # type: ignore[arg-type]


def _post(
    client: TestClient,
    session_id: str,
    *,
    operation_id: str,
    turn_token: str,
    message: str,
) -> object:
    return client.post(
        f"/api/sessions/{session_id}/guided/chat",
        json={"operation_id": operation_id, "turn_token": turn_token, "message": message},
    )


def _guided(client: TestClient, session_id: str):
    record = asyncio.run(client.app.state.session_service.get_current_state(UUID(session_id)))
    assert record is not None
    guided = state_from_record(record).guided_session
    assert guided is not None
    return guided


def _topology_presence_action() -> DeferredIntentAction:
    return DeferredIntentAction(
        target_stage="topology",
        catalog_kind="transform",
        catalog_name="passthrough",
        redacted_summary="Preserve the configured source during topology authoring.",
        constraints=(
            SubjectPresenceConstraint(
                kind="subject_presence",
                subject=PluginSubject(
                    kind="plugin",
                    subject_id="11111111-1111-4111-8111-111111111111",
                    plugin_kind="source",
                    plugin_name="csv",
                ),
                present=True,
            ),
        ),
    )


def _wire_review_action(*, present: bool) -> DeferredIntentAction:
    return DeferredIntentAction(
        target_stage="wire_review",
        catalog_kind=None,
        catalog_name=None,
        redacted_summary="Preserve the structural source-to-output route.",
        constraints=(
            EdgeRouteConstraint(
                kind="edge_route",
                from_subject=PluginSubject(
                    kind="plugin",
                    subject_id="11111111-1111-4111-8111-111111111111",
                    plugin_kind="source",
                    plugin_name="csv",
                ),
                edge_type="on_success",
                to_subject=PluginSubject(
                    kind="plugin",
                    subject_id="22222222-2222-4222-8222-222222222222",
                    plugin_kind="sink",
                    plugin_name="json",
                ),
                present=present,
            ),
        ),
    )


def _stage_schema8_topology_intent_proposal(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[str, object, dict]:
    session_id = _create_session(client)
    initial = client.get(f"/api/sessions/{session_id}/guided").json()
    monkeypatch.setattr(guided_route, "_run_guided_chat_provider_attempt", _provider(_topology_presence_action()))
    retained_response = _post(
        client,
        session_id,
        operation_id=str(uuid4()),
        turn_token=initial["next_turn"]["turn_token"],
        message="Later retain the topology constraint.",
    )
    assert retained_response.status_code == 200, retained_response.json()
    (retained,) = _guided(client, session_id).deferred_intents
    staged = TestStep2IntraStep()._stage_proposal(client, session_id, filename="schema8-rewind.jsonl")
    assert staged["guided_session"]["step"] == "step_3_transforms"
    return session_id, retained, staged


def test_unique_future_catalog_intent_is_private_atomic_retryable_and_restart_durable(
    composer_test_client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = composer_test_client
    session_id = _create_session(client)
    turn = client.get(f"/api/sessions/{session_id}/guided").json()["next_turn"]
    operation_id = str(uuid4())
    private_message = "Later use passthrough with customer-secret-needle in its private context."
    monkeypatch.setattr(guided_route, "_run_guided_chat_provider_attempt", _provider(_action()))

    first = _post(
        client,
        session_id,
        operation_id=operation_id,
        turn_token=turn["turn_token"],
        message=private_message,
    )

    assert first.status_code == 200, first.json()
    first_json = first.json()
    assert first_json["assistant_message"] == "I saved that instruction for the topology stage."
    assert private_message not in first.text
    assert "provider provisional text" not in first.text
    guided = _guided(client, session_id)
    assert len(guided.deferred_intents) == 1
    (intent,) = guided.deferred_intents
    assert intent.receiving_stage == "source"
    assert intent.target_stage == "topology"
    assert intent.catalog_kind == "transform"
    assert intent.catalog_name == "passthrough"
    assert intent.message_content_hash == stable_hash(private_message)
    assert guided.active_proposal is None
    assert private_message not in repr(intent.to_dict())
    assert guided.chat_history[-2].content == "[Future-stage instruction submitted privately.]"

    messages = asyncio.run(client.app.state.session_service.get_messages(UUID(session_id), limit=None))
    user_rows = [message for message in messages if message.role == "user"]
    assert [(str(message.id), message.content) for message in user_rows] == [(intent.originating_message_id, private_message)]
    assert all(private_message not in message.content for message in messages if message.role != "user")
    assert all(private_message not in repr(message.tool_calls) for message in messages if message.role != "user")
    audit_envelopes = [envelope for message in messages if message.role == "audit" for envelope in (message.tool_calls or ())]
    assert [envelope["_kind"] for envelope in audit_envelopes] == ["audit", "chat_turn_audit"]
    assert audit_envelopes[0]["invocation"]["tool_name"] == "guided_turn_emitted"
    assert set(audit_envelopes[1]) == {"_kind", "turn"}
    assert private_message not in repr(audit_envelopes)

    retry = _post(
        client,
        session_id,
        operation_id=operation_id,
        turn_token=turn["turn_token"],
        message=private_message,
    )
    assert retry.status_code == 200, retry.json()
    assert retry.json() == first_json
    with client.app.state.session_engine.connect() as connection:
        assert (
            connection.execute(
                select(func.count()).select_from(composition_states_table).where(composition_states_table.c.session_id == session_id)
            ).scalar_one()
            == 1
        )
        assert (
            connection.execute(
                select(func.count())
                .select_from(chat_messages_table)
                .where(
                    chat_messages_table.c.session_id == session_id,
                    chat_messages_table.c.role == "user",
                )
            ).scalar_one()
            == 1
        )
        operation = (
            connection.execute(
                select(guided_operations_table).where(
                    guided_operations_table.c.session_id == session_id,
                    guided_operations_table.c.operation_id == operation_id,
                )
            )
            .mappings()
            .one()
        )
    assert operation["originating_message_id"] == intent.originating_message_id

    restart = client.app.state.restart_test_client
    restarted = restart()
    refreshed = restarted.get(f"/api/sessions/{session_id}/guided")
    assert refreshed.status_code == 200, refreshed.json()
    restarted_guided = _guided(restarted, session_id)
    assert restarted_guided.deferred_intents == guided.deferred_intents


def test_explicit_cancel_removes_only_the_named_pending_intent_and_replays_exactly(
    composer_test_client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = composer_test_client
    session_id = _create_session(client)
    turn = client.get(f"/api/sessions/{session_id}/guided").json()["next_turn"]
    monkeypatch.setattr(guided_route, "_run_guided_chat_provider_attempt", _provider(_action()))
    retained_response = _post(
        client,
        session_id,
        operation_id=str(uuid4()),
        turn_token=turn["turn_token"],
        message="Later add passthrough.",
    )
    assert retained_response.status_code == 200, retained_response.json()
    assert retained_response.json()["assistant_message"] == "I saved that instruction for the topology stage."
    (retained,) = _guided(client, session_id).deferred_intents

    operation_id = str(uuid4())
    private_cancel_request = "Cancel the saved requirement, private-cancel-canary."
    monkeypatch.setattr(
        guided_route,
        "_run_guided_chat_provider_attempt",
        _management_provider(
            DeferredIntentCancelAction(
                intent_id=retained.intent_id,
                selection_token=deferred_intent_management_option(retained).selection_token,
            )
        ),
    )
    first = _post(
        client,
        session_id,
        operation_id=operation_id,
        turn_token=retained_response.json()["next_turn"]["turn_token"],
        message=private_cancel_request,
    )

    assert first.status_code == 200, first.json()
    assert first.json()["assistant_message"] == "I cancelled that saved topology instruction."
    assert _guided(client, session_id).deferred_intents == ()
    assert private_cancel_request not in first.text
    retry = _post(
        client,
        session_id,
        operation_id=operation_id,
        turn_token=retained_response.json()["next_turn"]["turn_token"],
        message=private_cancel_request,
    )
    assert retry.status_code == 200
    assert retry.json() == first.json()

    messages = asyncio.run(client.app.state.session_service.get_messages(UUID(session_id), limit=None))
    cancellation_events = [
        envelope
        for message in messages
        if message.role == "audit"
        for envelope in (message.tool_calls or ())
        if envelope.get("invocation", {}).get("tool_name") == "guided_intent_cancelled"
    ]
    assert len(cancellation_events) == 1
    assert private_cancel_request not in repr(cancellation_events)


def test_destructive_selection_binding_rejects_mixups_and_requires_uuid_when_selection_is_plural(
    composer_test_client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = composer_test_client
    session_id = _create_session(client)
    turn = client.get(f"/api/sessions/{session_id}/guided").json()["next_turn"]

    monkeypatch.setattr(guided_route, "_run_guided_chat_provider_attempt", _provider(_action(count=1)))
    first_response = _post(
        client,
        session_id,
        operation_id=str(uuid4()),
        turn_token=turn["turn_token"],
        message="Save the count-one transform constraint.",
    )
    monkeypatch.setattr(guided_route, "_run_guided_chat_provider_attempt", _provider(_action(count=2)))
    second_response = _post(
        client,
        session_id,
        operation_id=str(uuid4()),
        turn_token=first_response.json()["next_turn"]["turn_token"],
        message="Save the count-two transform constraint.",
    )
    first, second = _guided(client, session_id).deferred_intents

    mixed = DeferredIntentCancelAction(
        intent_id=first.intent_id,
        selection_token=deferred_intent_management_option(second).selection_token,
    )
    monkeypatch.setattr(guided_route, "_run_guided_chat_provider_attempt", _management_provider(mixed))
    mismatch_response = _post(
        client,
        session_id,
        operation_id=str(uuid4()),
        turn_token=second_response.json()["next_turn"]["turn_token"],
        message="Cancel the count-one transform constraint.",
    )
    assert mismatch_response.status_code == 200, mismatch_response.json()
    assert mismatch_response.json()["assistant_message_kind"] == "synthetic_failure"
    assert mismatch_response.json()["guided_session"]["chat_history"][-1]["synthetic_failure_reason"] == "not_applied"
    assert _guided(client, session_id).deferred_intents == (first, second)

    correct = DeferredIntentCancelAction(
        intent_id=first.intent_id,
        selection_token=deferred_intent_management_option(first).selection_token,
    )
    monkeypatch.setattr(guided_route, "_run_guided_chat_provider_attempt", _management_provider(correct))
    selected_response = _post(
        client,
        session_id,
        operation_id=str(uuid4()),
        turn_token=mismatch_response.json()["next_turn"]["turn_token"],
        message=f"Cancel exact intent {first.intent_id}.",
    )
    assert selected_response.status_code == 200, selected_response.json()
    assert _guided(client, session_id).deferred_intents == (second,)

    monkeypatch.setattr(guided_route, "_run_guided_chat_provider_attempt", _provider(_action(count=2)))
    duplicate_response = _post(
        client,
        session_id,
        operation_id=str(uuid4()),
        turn_token=selected_response.json()["next_turn"]["turn_token"],
        message="Save another count-two transform constraint.",
    )
    second, duplicate = _guided(client, session_id).deferred_intents
    duplicate_cancel = DeferredIntentCancelAction(
        intent_id=second.intent_id,
        selection_token=deferred_intent_management_option(second).selection_token,
    )
    monkeypatch.setattr(guided_route, "_run_guided_chat_provider_attempt", _management_provider(duplicate_cancel))
    ambiguous_response = _post(
        client,
        session_id,
        operation_id=str(uuid4()),
        turn_token=duplicate_response.json()["next_turn"]["turn_token"],
        message="Cancel one count-two transform constraint.",
    )
    assert ambiguous_response.status_code == 200, ambiguous_response.json()
    assert ambiguous_response.json()["assistant_message_kind"] == "synthetic_failure"
    assert ambiguous_response.json()["guided_session"]["chat_history"][-1]["synthetic_failure_reason"] == "not_applied"
    assert _guided(client, session_id).deferred_intents == (second, duplicate)

    explicit_response = _post(
        client,
        session_id,
        operation_id=str(uuid4()),
        turn_token=ambiguous_response.json()["next_turn"]["turn_token"],
        message=f"Cancel exact intent {second.intent_id}.",
    )
    assert explicit_response.status_code == 200, explicit_response.json()
    assert _guided(client, session_id).deferred_intents == (duplicate,)


def test_schema8_topology_management_rewinds_through_output_review_and_atomically_supersedes_pending_proposal(
    composer_test_client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = composer_test_client
    session_id, retained, staged = _stage_schema8_topology_intent_proposal(client, monkeypatch)
    proposal_id = staged["next_turn"]["payload"]["proposal_id"]

    operation_id = str(uuid4())
    monkeypatch.setattr(
        guided_route,
        "_run_guided_chat_provider_attempt",
        _management_provider(
            DeferredIntentCancelAction(
                intent_id=retained.intent_id,
                selection_token=deferred_intent_management_option(retained).selection_token,
            )
        ),
    )
    cancelled = _post(
        client,
        session_id,
        operation_id=operation_id,
        turn_token=staged["next_turn"]["turn_token"],
        message="Cancel that topology instruction without replanning yet.",
    )

    assert cancelled.status_code == 200, cancelled.json()
    body = cancelled.json()
    assert body["guided_session"]["step"] == "step_2_sink"
    assert body["next_turn"]["type"] == "review_components"
    assert body["next_turn"]["payload"]["component_kind"] == "output"
    assert _guided(client, session_id).deferred_intents == ()
    with client.app.state.session_engine.connect() as connection:
        proposal = (
            connection.execute(select(composition_proposals_table).where(composition_proposals_table.c.id == proposal_id)).mappings().one()
        )
        events = (
            connection.execute(select(proposal_events_table.c.event_type).where(proposal_events_table.c.proposal_id == proposal_id))
            .scalars()
            .all()
        )
    assert proposal["status"] == "rejected"
    assert events == ["proposal.created", "proposal.rejected"]

    replay = _post(
        client,
        session_id,
        operation_id=operation_id,
        turn_token=staged["next_turn"]["turn_token"],
        message="Cancel that topology instruction without replanning yet.",
    )
    assert replay.status_code == 200
    assert replay.json() == body


def test_management_non_string_provider_content_is_bounded_without_private_egress(
    composer_test_client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from structlog.testing import capture_logs

    client = composer_test_client
    session_id, _retained, staged = _stage_schema8_topology_intent_proposal(client, monkeypatch)
    private_canary = "PRIVATE-NESTED-MANAGEMENT-ROUTE-CANARY"
    malformed_content = {"summary": ["ordinary", {"private": private_canary}]}

    async def malformed_reply(**_kwargs: object) -> SimpleNamespace:
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=malformed_content, tool_calls=None))],
        )

    monkeypatch.setattr(guided_route, "_run_guided_chat_provider_attempt", guided_chat_atomic.run_guided_chat_provider_attempt)
    monkeypatch.setattr(chat_solver, "_litellm_acompletion", malformed_reply)
    operation_id = str(uuid4())
    with capture_logs() as logs:
        response = _post(
            client,
            session_id,
            operation_id=operation_id,
            turn_token=staged["next_turn"]["turn_token"],
            message="Explain the saved topology instruction.",
        )

    assert response.status_code == 200, response.json()
    body = response.json()
    assert body["assistant_message_kind"] == "synthetic_failure"
    assert body["guided_session"]["chat_history"][-1]["synthetic_failure_reason"] == "unavailable"
    assert private_canary not in response.text
    assert private_canary not in repr(logs)

    messages = asyncio.run(client.app.state.session_service.get_messages(UUID(session_id), limit=None))
    assert private_canary not in repr(messages)
    with client.app.state.session_engine.connect() as connection:
        persisted_rows = {
            "messages": connection.execute(select(chat_messages_table).where(chat_messages_table.c.session_id == session_id)).all(),
            "states": connection.execute(select(composition_states_table).where(composition_states_table.c.session_id == session_id)).all(),
            "proposals": connection.execute(
                select(composition_proposals_table).where(composition_proposals_table.c.session_id == session_id)
            ).all(),
            "operations": connection.execute(
                select(guided_operations_table).where(guided_operations_table.c.session_id == session_id)
            ).all(),
            "operation_events": connection.execute(
                select(guided_operation_events_table).where(guided_operation_events_table.c.session_id == session_id)
            ).all(),
            "proposal_events": connection.execute(
                select(proposal_events_table).where(proposal_events_table.c.session_id == session_id)
            ).all(),
        }
    assert private_canary not in repr(persisted_rows)


@pytest.mark.parametrize("covered", [False, True], ids=["uncovered", "covered"])
@pytest.mark.parametrize("management_kind", ["cancel", "edit"])
def test_active_proposal_future_wire_management_always_invalidates_and_rewinds_without_planner(
    composer_test_client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
    covered: bool,
    management_kind: str,
) -> None:
    client = composer_test_client
    session_id = _create_session(client)
    initial = client.get(f"/api/sessions/{session_id}/guided").json()
    monkeypatch.setattr(guided_route, "_run_guided_chat_provider_attempt", _provider(_wire_review_action(present=True)))
    retained_response = _post(
        client,
        session_id,
        operation_id=str(uuid4()),
        turn_token=initial["next_turn"]["turn_token"],
        message="Later preserve the source-to-output route.",
    )
    assert retained_response.status_code == 200, retained_response.json()
    (retained,) = _guided(client, session_id).deferred_intents

    planner = client.app.state.composer_service
    real_plan = planner.plan_guided_pipeline

    async def plan_with_exact_coverage(*, guided, **kwargs):
        plan, catalog_plugin_ids = await real_plan(guided=guided, **kwargs)
        proposal = plan.proposal
        rebound = PipelineProposal.create(
            pipeline=proposal.pipeline,
            base=proposal.base,
            reviewed_facts=guided_private_reviewed_facts(guided),
            surface=proposal.surface,
            repair_count=proposal.repair_count,
            skill_hash=proposal.skill_hash,
            covered_deferred_intent_ids=(retained.intent_id,) if covered else (),
            supersedes_draft_hash=proposal.supersedes_draft_hash,
        )
        return replace(plan, proposal=rebound), catalog_plugin_ids

    monkeypatch.setattr(planner, "plan_guided_pipeline", plan_with_exact_coverage)
    staged = TestStep2IntraStep()._stage_proposal(client, session_id, filename=f"wire-{covered}-{management_kind}.jsonl")
    proposal_id = staged["next_turn"]["payload"]["proposal_id"]
    assert staged["guided_session"]["step"] == "step_3_transforms"

    async def planner_must_not_run(**_kwargs: object) -> object:
        raise AssertionError("management rewind must not invoke the planner")

    monkeypatch.setattr(planner, "plan_guided_pipeline", planner_must_not_run)
    if management_kind == "cancel":
        management_action: DeferredIntentManagementAction = DeferredIntentCancelAction(
            intent_id=retained.intent_id,
            selection_token=deferred_intent_management_option(retained).selection_token,
        )
    else:
        management_action = DeferredIntentEditAction(
            intent_id=retained.intent_id,
            selection_token=deferred_intent_management_option(retained).selection_token,
            replacement=_wire_review_action(present=False),
        )
    monkeypatch.setattr(guided_route, "_run_guided_chat_provider_attempt", _management_provider(management_action))
    operation_id = str(uuid4())
    response = _post(
        client,
        session_id,
        operation_id=operation_id,
        turn_token=staged["next_turn"]["turn_token"],
        message=f"{management_kind} exact intent {retained.intent_id}",
    )

    assert response.status_code == 200, response.json()
    body = response.json()
    assert body["guided_session"]["step"] == "step_2_sink"
    assert body["next_turn"]["payload"]["component_kind"] == "output"
    remaining = _guided(client, session_id).deferred_intents
    if management_kind == "cancel":
        assert remaining == ()
    else:
        assert [intent.intent_id for intent in remaining] == [retained.intent_id]
        assert remaining[0].constraints[0].to_dict()["present"] is False
    with client.app.state.session_engine.connect() as connection:
        proposal = (
            connection.execute(select(composition_proposals_table).where(composition_proposals_table.c.id == proposal_id)).mappings().one()
        )
    assert proposal["status"] == "rejected"
    replay = _post(
        client,
        session_id,
        operation_id=operation_id,
        turn_token=staged["next_turn"]["turn_token"],
        message=f"{management_kind} exact intent {retained.intent_id}",
    )
    assert replay.status_code == 200
    assert replay.json() == body


def test_management_provider_api_error_completes_unavailable_turn_without_mutating_intent_or_active_proposal(
    composer_test_client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from litellm.exceptions import APIError as LiteLLMAPIError

    real_provider_runner = guided_route._run_guided_chat_provider_attempt
    client = composer_test_client
    session_id, retained, staged = _stage_schema8_topology_intent_proposal(client, monkeypatch)
    proposal_id = staged["next_turn"]["payload"]["proposal_id"]
    before = _guided(client, session_id)
    monkeypatch.setattr(guided_route, "_run_guided_chat_provider_attempt", real_provider_runner)

    async def provider_failure(**_kwargs: object) -> object:
        raise LiteLLMAPIError(
            status_code=500,
            message="private upstream failure detail",
            llm_provider="test",
            model="test/model",
        )

    monkeypatch.setattr(chat_solver, "_litellm_acompletion", provider_failure)
    operation_id = str(uuid4())
    response = _post(
        client,
        session_id,
        operation_id=operation_id,
        turn_token=staged["next_turn"]["turn_token"],
        message="Cancel the pending topology instruction.",
    )

    assert response.status_code == 200, response.json()
    body = response.json()
    assert body["assistant_message"] == "I'm unavailable right now; you can still use the wizard controls."
    after = _guided(client, session_id)
    assert after.deferred_intents == (retained,)
    assert after.active_proposal == before.active_proposal
    assert after.step == before.step
    with client.app.state.session_engine.connect() as connection:
        proposal = (
            connection.execute(select(composition_proposals_table).where(composition_proposals_table.c.id == proposal_id)).mappings().one()
        )
        events = (
            connection.execute(select(proposal_events_table.c.event_type).where(proposal_events_table.c.proposal_id == proposal_id))
            .scalars()
            .all()
        )
        operation = (
            connection.execute(select(guided_operations_table).where(guided_operations_table.c.operation_id == operation_id))
            .mappings()
            .one()
        )
    assert proposal["status"] == "pending"
    assert events == ["proposal.created"]
    assert operation["status"] == "completed"
    assert operation["result_state_id"] is not None


def test_schema8_passed_output_edit_preserves_stable_id_and_rewinds_without_rewriting_committed_proposal_lineage(
    composer_test_client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = composer_test_client
    session_id = _create_session(client)
    initial = client.get(f"/api/sessions/{session_id}/guided").json()
    output_action = DeferredIntentAction(
        target_stage="output",
        catalog_kind="sink",
        catalog_name="json",
        redacted_summary="Retain one JSON output requirement.",
        constraints=(
            ComponentCountConstraint(
                kind="component_count",
                component_kind="output",
                plugin_kind="sink",
                plugin_name="json",
                operator="at_least",
                count=1,
            ),
        ),
    )
    monkeypatch.setattr(guided_route, "_run_guided_chat_provider_attempt", _provider(output_action))
    retained_response = _post(
        client,
        session_id,
        operation_id=str(uuid4()),
        turn_token=initial["next_turn"]["turn_token"],
        message="Keep this output instruction pending through proposal review.",
    )
    assert retained_response.status_code == 200, retained_response.json()
    (retained,) = _guided(client, session_id).deferred_intents

    planner = client.app.state.composer_service
    real_plan = planner.plan_guided_pipeline

    async def plan_without_claiming_intent(*, guided, **kwargs):
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
    staged = TestStep2IntraStep()._stage_proposal(client, session_id, filename="schema8-passed-output.jsonl")
    proposal = staged["next_turn"]["payload"]
    accepted = client.post(
        f"/api/sessions/{session_id}/guided/respond",
        json={
            "operation_id": str(uuid4()),
            "turn_token": staged["next_turn"]["turn_token"],
            "proposal_id": proposal["proposal_id"],
            "draft_hash": proposal["draft_hash"],
            "chosen": ["accept"],
        },
    )
    assert accepted.status_code == 200, accepted.json()
    assert accepted.json()["guided_session"]["step"] == "step_4_wire"
    assert [intent.intent_id for intent in _guided(client, session_id).deferred_intents] == [retained.intent_id]

    monkeypatch.setattr(
        guided_route,
        "_run_guided_chat_provider_attempt",
        _management_provider(
            DeferredIntentEditAction(
                intent_id=retained.intent_id,
                selection_token=deferred_intent_management_option(retained).selection_token,
                replacement=DeferredIntentAction(
                    target_stage="output",
                    catalog_kind="sink",
                    catalog_name="json",
                    redacted_summary="Retain the revised JSON output requirement.",
                    constraints=(
                        ComponentCountConstraint(
                            kind="component_count",
                            component_kind="output",
                            plugin_kind="sink",
                            plugin_name="json",
                            operator="at_least",
                            count=2,
                        ),
                    ),
                ),
            )
        ),
    )
    cancelled = _post(
        client,
        session_id,
        operation_id=str(uuid4()),
        turn_token=accepted.json()["next_turn"]["turn_token"],
        message="Revise the passed output instruction.",
    )

    assert cancelled.status_code == 200, cancelled.json()
    assert cancelled.json()["guided_session"]["step"] == "step_2_sink"
    assert cancelled.json()["next_turn"]["type"] == "review_components"
    assert cancelled.json()["next_turn"]["payload"]["component_kind"] == "output"
    (revised,) = _guided(client, session_id).deferred_intents
    assert revised.intent_id == retained.intent_id
    assert revised.constraints[0].to_dict()["count"] == 2
    assert revised.originating_message_id != retained.originating_message_id
    with client.app.state.session_engine.connect() as connection:
        proposal_row = (
            connection.execute(select(composition_proposals_table).where(composition_proposals_table.c.id == proposal["proposal_id"]))
            .mappings()
            .one()
        )
        events = (
            connection.execute(
                select(proposal_events_table.c.event_type).where(proposal_events_table.c.proposal_id == proposal["proposal_id"])
            )
            .scalars()
            .all()
        )
    assert proposal_row["status"] == "committed"
    assert events == ["proposal.created", "proposal.accepted"]


@pytest.mark.parametrize("fault_point", ("proposal_event", "proposal_update"))
def test_schema8_management_proposal_invalidation_fault_rolls_back_intent_proposal_checkpoint_and_custody(
    composer_test_client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
    fault_point: str,
) -> None:
    client = composer_test_client
    session_id, retained, staged = _stage_schema8_topology_intent_proposal(client, monkeypatch)
    proposal_id = staged["next_turn"]["payload"]["proposal_id"]
    operation_id = str(uuid4())
    engine = client.app.state.session_engine
    with engine.connect() as connection:
        state_count_before = connection.execute(
            select(func.count()).select_from(composition_states_table).where(composition_states_table.c.session_id == session_id)
        ).scalar_one()
        message_count_before = connection.execute(
            select(func.count()).select_from(chat_messages_table).where(chat_messages_table.c.session_id == session_id)
        ).scalar_one()

    monkeypatch.setattr(
        guided_route,
        "_run_guided_chat_provider_attempt",
        _management_provider(
            DeferredIntentCancelAction(
                intent_id=retained.intent_id,
                selection_token=deferred_intent_management_option(retained).selection_token,
            )
        ),
    )
    reached = False

    def inject_fault(_conn, _cursor, statement, _parameters, context, _executemany):
        nonlocal reached
        target_table = getattr(getattr(getattr(context, "compiled", None), "statement", None), "table", None)
        normalized = " ".join(statement.lower().split())
        matched = (fault_point == "proposal_event" and target_table is proposal_events_table) or (
            fault_point == "proposal_update" and normalized.startswith("update composition_proposals")
        )
        if matched and not reached:
            reached = True
            raise RuntimeError(f"injected {fault_point}")

    event.listen(engine, "before_cursor_execute", inject_fault)
    try:
        response = _post(
            client,
            session_id,
            operation_id=operation_id,
            turn_token=staged["next_turn"]["turn_token"],
            message="private cancellation must roll back",
        )
    finally:
        event.remove(engine, "before_cursor_execute", inject_fault)

    assert response.status_code == 500, response.json()
    assert reached
    assert [intent.intent_id for intent in _guided(client, session_id).deferred_intents] == [retained.intent_id]
    with engine.connect() as connection:
        assert (
            connection.execute(
                select(func.count()).select_from(composition_states_table).where(composition_states_table.c.session_id == session_id)
            ).scalar_one()
            == state_count_before
        )
        assert (
            connection.execute(
                select(func.count()).select_from(chat_messages_table).where(chat_messages_table.c.session_id == session_id)
            ).scalar_one()
            == message_count_before
        )
        proposal = (
            connection.execute(select(composition_proposals_table).where(composition_proposals_table.c.id == proposal_id)).mappings().one()
        )
        proposal_event_types = (
            connection.execute(select(proposal_events_table.c.event_type).where(proposal_events_table.c.proposal_id == proposal_id))
            .scalars()
            .all()
        )
        operation = (
            connection.execute(
                select(guided_operations_table).where(
                    guided_operations_table.c.session_id == session_id,
                    guided_operations_table.c.operation_id == operation_id,
                )
            )
            .mappings()
            .one()
        )
    assert proposal["status"] == "pending"
    assert proposal_event_types == ["proposal.created"]
    assert operation["status"] == "failed"
    assert operation["originating_message_id"] is None
    assert operation["result_state_id"] is None


@pytest.mark.parametrize(
    ("stage", "arguments"),
    [
        pytest.param("source", "{", id="source-invalid-json"),
        pytest.param("sink", 17, id="sink-non-string"),
        pytest.param("source", "9" * 5_000, id="source-integer-conversion-limit"),
        pytest.param("sink", "9" * 5_000, id="sink-integer-conversion-limit"),
        pytest.param("source", "[" * 10_000 + "]" * 10_000, id="source-json-recursion-limit"),
        pytest.param("sink", "[" * 10_000 + "]" * 10_000, id="sink-json-recursion-limit"),
    ],
)
def test_real_route_malformed_future_action_keeps_raw_instruction_only_in_private_message(
    composer_test_client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
    stage: str,
    arguments: object,
) -> None:
    client = composer_test_client
    session_id = _create_session(client)
    if stage == "sink":
        TestStepChatCrossStep._seed_csv_blob(client, session_id)
        TestStepChatCrossStep._configure_csv_source(client, session_id)
    before = client.get(f"/api/sessions/{session_id}/guided").json()
    assert before["guided_session"]["step"] == ("step_1_source" if stage == "source" else "step_2_sink")
    private_message = f"Later route the private customer-secret-needle through a transform from {stage}."
    repair_message = (
        "I couldn't verify that future-stage instruction, so I didn't retain it. "
        "Please restate the target stage and the structural requirement."
    )
    provider_calls = 0

    async def malformed_completion(**_kwargs: object) -> SimpleNamespace:
        nonlocal provider_calls
        provider_calls += 1
        tool_call = SimpleNamespace(
            function=SimpleNamespace(name="retain_deferred_intent", arguments=arguments),
        )
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=None, tool_calls=[tool_call]))])

    with client.app.state.session_engine.connect() as connection:
        state_count_before = connection.execute(
            select(func.count()).select_from(composition_states_table).where(composition_states_table.c.session_id == session_id)
        ).scalar_one()
    monkeypatch.setattr(chat_solver, "_litellm_acompletion", malformed_completion)
    response = _post(
        client,
        session_id,
        operation_id=str(uuid4()),
        turn_token=before["next_turn"]["turn_token"],
        message=private_message,
    )

    assert response.status_code == 200, response.json()
    response_json = response.json()
    assert provider_calls == 1
    assert response_json["assistant_message"] == repair_message
    assert response_json["assistant_message_kind"] == "synthetic_failure"
    if before["composition_state"] is None:
        assert response_json["composition_state"]["sources"] == {}
        assert response_json["composition_state"]["nodes"] == []
        assert response_json["composition_state"]["edges"] == []
        assert response_json["composition_state"]["outputs"] == []
    else:
        for field_name in ("sources", "nodes", "edges", "outputs"):
            assert response_json["composition_state"][field_name] == before["composition_state"][field_name]
    assert private_message not in response.text
    guided = _guided(client, session_id)
    assert guided.deferred_intents == ()
    assert guided.chat_history[-2].content == "[Future-stage instruction submitted privately.]"
    assert guided.chat_history[-1].content == repair_message
    assert len(guided.chat_history) == len(before["guided_session"]["chat_history"]) + 2
    messages = asyncio.run(client.app.state.session_service.get_messages(UUID(session_id), limit=None))
    assert [(message.role, message.content) for message in messages if message.role == "user"] == [("user", private_message)]
    assert all(private_message not in message.content for message in messages if message.role != "user")
    assert all(private_message not in repr(message.tool_calls) for message in messages if message.role != "user")
    with client.app.state.session_engine.connect() as connection:
        state_count_after = connection.execute(
            select(func.count()).select_from(composition_states_table).where(composition_states_table.c.session_id == session_id)
        ).scalar_one()
    assert state_count_after == state_count_before + 1


@pytest.mark.parametrize("stage", ["source", "sink"])
def test_real_route_rejects_free_form_option_literal_without_leaking_private_prose(
    composer_test_client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
    stage: str,
) -> None:
    client = composer_test_client
    session_id = _create_session(client)
    if stage == "sink":
        TestStepChatCrossStep._seed_csv_blob(client, session_id)
        TestStepChatCrossStep._configure_csv_source(client, session_id)
    private_message = "Send the full private customer-secret sentence to an arbitrary prompt option."
    action = DeferredIntentAction(
        target_stage="topology",
        catalog_kind="transform",
        catalog_name="passthrough",
        redacted_summary=private_message,
        constraints=(
            OptionValueConstraint(
                kind="option_value",
                subject=PluginSubject(
                    kind="plugin",
                    subject_id="55555555-5555-4555-8555-555555555555",
                    plugin_kind="transform",
                    plugin_name="passthrough",
                ),
                option_path=("prompt",),
                operator="equals",
                value=private_message,
            ),
        ),
    )
    catalog, snapshot = _policy_context(
        (("transform", "passthrough"),),
        available=frozenset({PluginId("transform", "passthrough")}),
        schemas={
            ("transform", "passthrough"): {
                "type": "object",
                "properties": {"prompt": {"type": "string"}},
            }
        },
    )
    monkeypatch.setattr(guided_chat_atomic, "_request_plugin_policy_context", lambda _request, _user: (catalog, snapshot))
    monkeypatch.setattr(guided_route, "_request_plugin_policy_context", lambda _request, _user: (catalog, snapshot))
    monkeypatch.setattr(guided_route, "_run_guided_chat_provider_attempt", _provider(action))
    before = client.get(f"/api/sessions/{session_id}/guided").json()
    assert before["guided_session"]["step"] == ("step_1_source" if stage == "source" else "step_2_sink")
    with client.app.state.session_engine.connect() as connection:
        state_count_before = connection.execute(
            select(func.count()).select_from(composition_states_table).where(composition_states_table.c.session_id == session_id)
        ).scalar_one()

    response = _post(
        client,
        session_id,
        operation_id=str(uuid4()),
        turn_token=before["next_turn"]["turn_token"],
        message=private_message,
    )

    assert response.status_code == 200, response.json()
    response_json = response.json()
    assert response_json["assistant_message"] == (
        "I couldn't safely retain that as a future-stage instruction. Please clarify the target stage and structural requirement."
    )
    assert private_message not in response.text
    guided = _guided(client, session_id)
    assert guided.deferred_intents == ()
    assert guided.chat_history[-2].content == "[Future-stage instruction submitted privately.]"
    messages = asyncio.run(client.app.state.session_service.get_messages(UUID(session_id), limit=None))
    assert [(message.role, message.content) for message in messages if message.role == "user"] == [("user", private_message)]
    assert all(private_message not in message.content for message in messages if message.role != "user")
    assert all(private_message not in repr(message.tool_calls) for message in messages if message.role != "user")
    with client.app.state.session_engine.connect() as connection:
        state_count_after = connection.execute(
            select(func.count()).select_from(composition_states_table).where(composition_states_table.c.session_id == session_id)
        ).scalar_one()
    assert state_count_after == state_count_before + 1


@pytest.mark.parametrize("stage", ["source", "sink"])
def test_exact_policy_denial_wins_over_same_name_visible_in_another_kind_at_each_guided_stage(
    composer_test_client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
    stage: str,
) -> None:
    client = composer_test_client
    session_id = _create_session(client)
    if stage == "sink":
        TestStepChatCrossStep._seed_csv_blob(client, session_id)
        TestStepChatCrossStep._configure_csv_source(client, session_id)
    private_message = "Privately remember the unavailable transform for topology."
    catalog, snapshot = _policy_context(
        (("source", "llm"), ("transform", "llm")),
        available=frozenset({PluginId("source", "llm")}),
    )
    monkeypatch.setattr(guided_chat_atomic, "_request_plugin_policy_context", lambda _request, _user: (catalog, snapshot))
    monkeypatch.setattr(guided_route, "_request_plugin_policy_context", lambda _request, _user: (catalog, snapshot))
    monkeypatch.setattr(guided_route, "_run_guided_chat_provider_attempt", _provider(_action(catalog_name="llm")))
    turn = client.get(f"/api/sessions/{session_id}/guided").json()["next_turn"]

    response = _post(
        client,
        session_id,
        operation_id=str(uuid4()),
        turn_token=turn["turn_token"],
        message=private_message,
    )

    assert response.status_code == 200, response.json()
    assert response.json()["assistant_message"] == "The transform plugin 'llm' is not enabled by the current policy."
    assert private_message not in response.text
    guided = _guided(client, session_id)
    assert guided.deferred_intents == ()
    assert guided.chat_history[-2].content == "[Future-stage instruction submitted privately.]"
    messages = asyncio.run(client.app.state.session_service.get_messages(UUID(session_id), limit=None))
    assert [(message.role, message.content) for message in messages if message.role == "user"] == [("user", private_message)]
    assert all(private_message not in message.content for message in messages if message.role != "user")


@pytest.mark.parametrize("option_schema", [True, False])
def test_boolean_property_schema_is_repairable_and_does_not_write_a_deferred_intent(
    composer_test_client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
    option_schema: bool,
) -> None:
    client = composer_test_client
    session_id = _create_session(client)
    private_message = "Privately retain a literal only if catalog authority proves it closed."
    action = DeferredIntentAction(
        target_stage="topology",
        catalog_kind="transform",
        catalog_name="passthrough",
        redacted_summary="Retain one closed prompt value.",
        constraints=(
            OptionValueConstraint(
                kind="option_value",
                subject=PluginSubject(
                    kind="plugin",
                    subject_id="55555555-5555-4555-8555-555555555555",
                    plugin_kind="transform",
                    plugin_name="passthrough",
                ),
                option_path=("prompt",),
                operator="equals",
                value="safe",
            ),
        ),
    )
    catalog, snapshot = _policy_context(
        (("transform", "passthrough"),),
        available=frozenset({PluginId("transform", "passthrough")}),
        schemas={("transform", "passthrough"): {"type": "object", "properties": {"prompt": option_schema}}},
    )
    monkeypatch.setattr(guided_chat_atomic, "_request_plugin_policy_context", lambda _request, _user: (catalog, snapshot))
    monkeypatch.setattr(guided_route, "_request_plugin_policy_context", lambda _request, _user: (catalog, snapshot))
    monkeypatch.setattr(guided_route, "_run_guided_chat_provider_attempt", _provider(action))
    turn = client.get(f"/api/sessions/{session_id}/guided").json()["next_turn"]

    response = _post(
        client,
        session_id,
        operation_id=str(uuid4()),
        turn_token=turn["turn_token"],
        message=private_message,
    )

    assert response.status_code == 200, response.json()
    assert response.json()["assistant_message"] == (
        "I couldn't safely retain that as a future-stage instruction. Please clarify the target stage and structural requirement."
    )
    assert private_message not in response.text
    guided = _guided(client, session_id)
    assert guided.deferred_intents == ()
    assert guided.chat_history[-2].content == "[Future-stage instruction submitted privately.]"


@pytest.mark.parametrize(
    ("schema_info", "option_value"),
    [
        (
            PluginSchemaInfo(
                name="wrong-plugin",
                plugin_type="transform",
                description="corrupt identity",
                json_schema={"type": "object", "properties": {"prompt": {"enum": ["safe"]}}},
                knob_schema={"fields": []},
            ),
            "safe",
        ),
        (
            PluginSchemaInfo(
                name="passthrough",
                plugin_type="transform",
                description="dangling ref",
                json_schema={"type": "object", "properties": {"prompt": {"$ref": "#/$defs/Missing"}}},
                knob_schema={"fields": []},
            ),
            "safe",
        ),
        (
            PluginSchemaInfo(
                name="passthrough",
                plugin_type="transform",
                description="present null property schema",
                json_schema={"type": "object", "properties": {"prompt": None}},
                knob_schema={"fields": []},
            ),
            "safe",
        ),
        (
            PluginSchemaInfo(
                name="passthrough",
                plugin_type="transform",
                description="malformed restrictive schema",
                json_schema={"type": "object", "properties": {"prompt": {"enum": ["safe"], "not": None}}},
                knob_schema={"fields": []},
            ),
            "safe",
        ),
        (
            PluginSchemaInfo(
                name="passthrough",
                plugin_type="transform",
                description="dangling restrictive ref",
                json_schema={
                    "type": "object",
                    "properties": {"prompt": {"enum": ["safe"], "not": {"$ref": "#/$defs/Missing"}}},
                },
                knob_schema={"fields": []},
            ),
            "other",
        ),
        (
            PluginSchemaInfo(
                name="passthrough",
                plugin_type="transform",
                description="dynamic ref with in-domain value",
                json_schema={
                    "type": "object",
                    "properties": {"prompt": {"enum": ["safe"], "not": {"$dynamicRef": "#missing"}}},
                },
                knob_schema={"fields": []},
            ),
            "safe",
        ),
        (
            PluginSchemaInfo(
                name="passthrough",
                plugin_type="transform",
                description="dynamic ref with out-of-domain value",
                json_schema={
                    "type": "object",
                    "properties": {"prompt": {"enum": ["safe"], "not": {"$dynamicRef": "#missing"}}},
                },
                knob_schema={"fields": []},
            ),
            "other",
        ),
        (
            PluginSchemaInfo(
                name="passthrough",
                plugin_type="transform",
                description="nested resource dangling ref",
                json_schema={
                    "$defs": {"Mode": {"enum": ["root-safe"]}},
                    "type": "object",
                    "properties": {"prompt": {"$id": "outer", "$ref": "#/$defs/Mode"}},
                },
                knob_schema={"fields": []},
            ),
            "root-safe",
        ),
    ],
    ids=(
        "schema-identity-mismatch",
        "dangling-local-ref",
        "present-null-property",
        "malformed-restriction",
        "dangling-restriction-before-membership",
        "dynamic-ref-in-domain",
        "dynamic-ref-out-of-domain",
        "nested-resource-dangling-ref",
    ),
)
def test_catalog_schema_authority_corruption_fails_operation_without_publishing_repair_or_cohort(
    composer_test_client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
    schema_info: PluginSchemaInfo,
    option_value: str,
) -> None:
    client = composer_test_client
    session_id = _create_session(client)
    private_message = "Private future prompt text must never enter a repair response."
    action = DeferredIntentAction(
        target_stage="topology",
        catalog_kind="transform",
        catalog_name="passthrough",
        redacted_summary="Retain one closed prompt value.",
        constraints=(
            OptionValueConstraint(
                kind="option_value",
                subject=PluginSubject(
                    kind="plugin",
                    subject_id="55555555-5555-4555-8555-555555555555",
                    plugin_kind="transform",
                    plugin_name="passthrough",
                ),
                option_path=("prompt",),
                operator="equals",
                value=option_value,
            ),
        ),
    )
    catalog, snapshot = _policy_context(
        (("transform", "passthrough"),),
        available=frozenset({PluginId("transform", "passthrough")}),
        schema_overrides={("transform", "passthrough"): schema_info},
    )
    monkeypatch.setattr(guided_chat_atomic, "_request_plugin_policy_context", lambda _request, _user: (catalog, snapshot))
    monkeypatch.setattr(guided_route, "_request_plugin_policy_context", lambda _request, _user: (catalog, snapshot))
    monkeypatch.setattr(guided_route, "_run_guided_chat_provider_attempt", _provider(action))
    turn = client.get(f"/api/sessions/{session_id}/guided").json()["next_turn"]
    operation_id = str(uuid4())

    response = _post(
        client,
        session_id,
        operation_id=operation_id,
        turn_token=turn["turn_token"],
        message=private_message,
    )

    assert response.status_code == 500, response.json()
    assert response.json()["detail"]["failure_code"] == "integrity_error"
    assert "couldn't safely retain" not in response.text
    assert private_message not in response.text
    with client.app.state.session_engine.connect() as connection:
        states = connection.execute(select(composition_states_table).where(composition_states_table.c.session_id == session_id)).all()
        messages = connection.execute(select(chat_messages_table).where(chat_messages_table.c.session_id == session_id)).all()
        operation = (
            connection.execute(
                select(guided_operations_table).where(
                    guided_operations_table.c.session_id == session_id,
                    guided_operations_table.c.operation_id == operation_id,
                )
            )
            .mappings()
            .one()
        )
    assert states == []
    assert messages == []
    assert operation["status"] == "failed"
    assert operation["failure_code"] == "integrity_error"
    assert operation["originating_message_id"] is None
    assert operation["result_state_id"] is None


@pytest.mark.parametrize(
    ("action", "expected_fragment", "installed", "available"),
    [
        (_action(catalog_name="not_installed_plugin"), "is not installed", (), frozenset()),
        (
            _action(catalog_name="llm"),
            "is not enabled by the current policy",
            (("transform", "llm"),),
            frozenset(),
        ),
        (
            _action(target_stage="source", catalog_kind="source", catalog_name="csv"),
            "couldn't safely retain",
            (("source", "csv"),),
            frozenset({PluginId("source", "csv")}),
        ),
    ],
)
def test_absence_policy_denial_and_current_target_clarify_without_mutation(
    composer_test_client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
    action: DeferredIntentAction,
    expected_fragment: str,
    installed: tuple[tuple[PluginKind, str], ...],
    available: frozenset[PluginId],
) -> None:
    client = composer_test_client
    session_id = _create_session(client)
    catalog, snapshot = _policy_context(installed, available=available)
    monkeypatch.setattr(guided_chat_atomic, "_request_plugin_policy_context", lambda _request, _user: (catalog, snapshot))
    monkeypatch.setattr(guided_route, "_request_plugin_policy_context", lambda _request, _user: (catalog, snapshot))
    turn = client.get(f"/api/sessions/{session_id}/guided").json()["next_turn"]
    monkeypatch.setattr(guided_route, "_run_guided_chat_provider_attempt", _provider(action))

    response = _post(
        client,
        session_id,
        operation_id=str(uuid4()),
        turn_token=turn["turn_token"],
        message="private wrong-stage detail",
    )

    assert response.status_code == 200, response.json()
    assert expected_fragment in response.json()["assistant_message"]
    assert _guided(client, session_id).deferred_intents == ()


@pytest.mark.parametrize(
    "fault_point",
    ("state_insert", "message_insert", "audit_insert", "operation_bind", "operation_complete", "operation_event"),
)
def test_fault_at_each_settlement_boundary_rolls_back_intent_message_audit_state_and_completion(
    composer_test_client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
    fault_point: str,
) -> None:
    client = composer_test_client
    session_id = _create_session(client)
    turn = client.get(f"/api/sessions/{session_id}/guided").json()["next_turn"]
    operation_id = str(uuid4())
    monkeypatch.setattr(guided_route, "_run_guided_chat_provider_attempt", _provider(_action()))
    engine = client.app.state.session_engine
    armed = True
    writes: list[str] = []
    chat_insert_count = 0

    def inject_fault(_conn, _cursor, statement, _parameters, _context, _executemany):
        nonlocal armed, chat_insert_count
        if not armed:
            return
        normalized = " ".join(statement.lower().split())
        compiled = getattr(_context, "compiled", None)
        target_table = getattr(getattr(compiled, "statement", None), "table", None)
        label: str | None = None
        if target_table is composition_states_table:
            label = "state_insert"
        elif target_table is chat_messages_table:
            chat_insert_count += 1
            label = "message_insert" if chat_insert_count == 1 else "audit_insert"
        elif normalized.startswith("update guided_operations") and " set originating_message_id" in normalized:
            label = "operation_bind"
        elif normalized.startswith("update guided_operations") and " set status" in normalized:
            label = "operation_complete"
        elif normalized.startswith("insert into guided_operation_events") and "operation_complete" in writes:
            label = "operation_event"
        if label is None:
            return
        writes.append(label)
        if label == fault_point:
            armed = False
            raise RuntimeError(f"injected {fault_point}")

    event.listen(engine, "before_cursor_execute", inject_fault)
    try:
        response = _post(
            client,
            session_id,
            operation_id=operation_id,
            turn_token=turn["turn_token"],
            message="private message must roll back",
        )
    finally:
        event.remove(engine, "before_cursor_execute", inject_fault)

    assert response.status_code == 500, response.json()
    with engine.connect() as connection:
        assert (
            connection.execute(
                select(func.count()).select_from(composition_states_table).where(composition_states_table.c.session_id == session_id)
            ).scalar_one()
            == 0
        )
        assert (
            connection.execute(
                select(func.count()).select_from(chat_messages_table).where(chat_messages_table.c.session_id == session_id)
            ).scalar_one()
            == 0
        )
        operation = (
            connection.execute(
                select(guided_operations_table).where(
                    guided_operations_table.c.session_id == session_id,
                    guided_operations_table.c.operation_id == operation_id,
                )
            )
            .mappings()
            .one()
        )
        operation_events = (
            connection.execute(
                select(guided_operation_events_table.c.event_kind)
                .where(guided_operation_events_table.c.operation_id == operation_id)
                .order_by(guided_operation_events_table.c.sequence)
            )
            .scalars()
            .all()
        )
    assert operation["status"] == "failed"
    assert operation["originating_message_id"] is None
    assert operation["result_state_id"] is None
    assert operation["response_hash"] is None
    assert operation_events == ["claimed", "renewed", "failed"]
    assert fault_point in writes


@pytest.mark.parametrize("corruption", ("cross_session_message", "message_hash_drift"))
def test_corrupt_origin_custody_is_an_integrity_failure_and_rolls_back_atomically(
    composer_test_client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
    corruption: str,
) -> None:
    client = composer_test_client
    target_session_id = _create_session(client)
    other_session_id = _create_session(client)
    private_message = "private future transform instruction"
    other_message = asyncio.run(
        client.app.state.session_service.add_message(
            UUID(other_session_id),
            "user",
            private_message,
            writer_principal="route_user_message",
        )
    )
    turn = client.get(f"/api/sessions/{target_session_id}/guided").json()["next_turn"]
    operation_id = str(uuid4())
    monkeypatch.setattr(guided_route, "_run_guided_chat_provider_attempt", _provider(_action()))
    service = client.app.state.session_service
    real_settle = service.settle_guided_state_operation

    async def corrupt_command(command, *, payload_store=None):
        metadata = deep_thaw(command.state.composer_meta)
        guided = GuidedSession.from_dict(metadata["guided_session"])
        (intent,) = guided.deferred_intents
        if corruption == "cross_session_message":
            corrupted_intent = replace(
                intent,
                originating_message_id=str(other_message.id),
                message_content_hash=stable_hash(private_message),
            )
            originating = GuidedOriginatingUserMessageDraft(message_id=other_message.id, content=private_message)
        else:
            corrupted_intent = replace(intent, message_content_hash=stable_hash("drifted content"))
            originating = command.originating_message
        metadata["guided_session"] = replace(guided, deferred_intents=(corrupted_intent,)).to_dict()
        corrupted_state = replace(command.state, composer_meta=metadata)
        return await real_settle(
            replace(command, state=corrupted_state, originating_message=originating),
            payload_store=payload_store,
        )

    monkeypatch.setattr(service, "settle_guided_state_operation", corrupt_command)
    response = _post(
        client,
        target_session_id,
        operation_id=operation_id,
        turn_token=turn["turn_token"],
        message=private_message,
    )

    assert response.status_code == 500, response.json()
    assert response.json()["detail"]["failure_code"] == "integrity_error"
    with client.app.state.session_engine.connect() as connection:
        assert (
            connection.execute(
                select(func.count()).select_from(composition_states_table).where(composition_states_table.c.session_id == target_session_id)
            ).scalar_one()
            == 0
        )
        assert (
            connection.execute(
                select(func.count()).select_from(chat_messages_table).where(chat_messages_table.c.session_id == target_session_id)
            ).scalar_one()
            == 0
        )
        assert (
            connection.execute(
                select(func.count()).select_from(chat_messages_table).where(chat_messages_table.c.session_id == other_session_id)
            ).scalar_one()
            == 1
        )
