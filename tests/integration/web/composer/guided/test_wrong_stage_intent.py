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
from elspeth.web.composer.guided.deferred_intents import DeferredIntentAction
from elspeth.web.composer.guided.stage_subjects import ComponentCountConstraint, OptionValueConstraint, PluginSubject
from elspeth.web.composer.guided.state_machine import GuidedSession
from elspeth.web.plugin_policy.models import PluginAvailability, PluginAvailabilitySnapshot, PluginId, PluginUnavailableReason
from elspeth.web.sessions._guided_step_chat import StepChatResult
from elspeth.web.sessions.converters import state_from_record
from elspeth.web.sessions.models import (
    chat_messages_table,
    composition_states_table,
    guided_operation_events_table,
    guided_operations_table,
)
from elspeth.web.sessions.protocol import GuidedOriginatingUserMessageDraft
from elspeth.web.sessions.routes.composer import guided as guided_route
from elspeth.web.sessions.routes.composer import guided_chat_atomic
from elspeth.web.sessions.routes.composer.guided_chat_atomic import GuidedChatProviderOutcome
from tests.integration.web.composer.guided.test_step_chat import TestStepChatCrossStep, _create_session
from tests.unit.web._sync_asgi_client import SyncASGITestClient as TestClient


def _action(
    *,
    target_stage: str = "topology",
    catalog_kind: str = "transform",
    catalog_name: str = "passthrough",
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
                count=1,
            ),
        ),
    )


def _provider(action: DeferredIntentAction) -> Callable[..., Awaitable[GuidedChatProviderOutcome]]:
    async def run(**_kwargs: object) -> GuidedChatProviderOutcome:
        return GuidedChatProviderOutcome(
            chat=StepChatResult(
                assistant_message="provider provisional text must not become authority",
                status=ComposerChatTurnStatus.SUCCESS,
                latency_ms=1,
                error_class=None,
            ),
            source_resolution=None,
            sink_resolution=None,
            deferred_action=action,
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
