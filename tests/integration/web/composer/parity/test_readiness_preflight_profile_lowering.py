"""Task-6 blocker probe: the readiness preflight must not false-positive on a
committed operator-profiled multi-query LLM node.

Background
----------
``set_pipeline`` persists a profiled ``llm`` node in AUTHORED-MINIMAL form: the
``profile`` alias and ``queries`` survive, but the sequential-multi-query retry
budget (``pool_size`` / ``max_capacity_retry_seconds``) is injected only at
operator-profile LOWERING (``profiles.OperatorProfileRegistry.lower_options`` →
``max_capacity_retry_seconds = WEB_LLM_SEQUENTIAL_MULTI_QUERY_MAX_RETRY_SECONDS``).

The execution-service readiness preflight (``_execute_locked``) used to run
``web_llm_retry_budget_policy_error`` on the UN-LOWERED ``composition_state``,
so a committed profiled multi-query node — whose raw options legitimately omit
the retry budget — false-positived and the run was rejected before creation,
even though the profile-lowered executable config is web-safe. The authoritative
``validate_pipeline`` check runs on ``policy_result.executable_state`` (the
lowered state) and PASSES; the fix makes the preflight gate iterate that same
lowered state.

This probe commits the ``structured_llm`` fixture through the REAL parity
``set_pipeline`` path, then drives the committed state through the REAL
``ExecutionServiceImpl`` preflight and asserts the retry-budget gate does not
reject it. ``validate_pipeline`` is stubbed VALID to isolate the execution
service's own gate (and to avoid instantiating the profile's bedrock provider),
exactly as the sibling unit harness does — the retry-budget gate under test is
driven by the REAL operator-profile lowering (``validate_plugin_policy``).
"""

from __future__ import annotations

import asyncio
from concurrent.futures import Future
from datetime import UTC, datetime
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock, create_autospec, patch
from uuid import UUID, uuid4

import pytest

from elspeth.contracts.hashing import stable_hash
from elspeth.web.auth.models import UserIdentity
from elspeth.web.composer import yaml_generator as real_yaml_generator
from elspeth.web.composer.state import CompositionState
from elspeth.web.execution.fanout_guard import ExecutionFanoutGuardRequired
from elspeth.web.execution.progress import ProgressBroadcaster
from elspeth.web.execution.schemas import ValidationReadiness, ValidationResult
from elspeth.web.execution.service import ExecutionServiceImpl
from elspeth.web.interpretation_state import INTERPRETATION_REQUIREMENTS_KEY
from elspeth.web.sessions.protocol import CompositionStateRecord, SessionServiceProtocol
from elspeth.web.sessions.telemetry import build_sessions_telemetry
from tests.integration.web.composer.parity.conftest import PARITY_FIXTURES


def _structured_llm_fixture() -> dict[str, Any]:
    for fixture in PARITY_FIXTURES:
        if fixture["class"] == "structured_llm":
            return fixture
    raise AssertionError("structured_llm parity fixture missing")


def _resolve_prompt_template_review(node: dict[str, Any]) -> dict[str, Any]:
    """Mark the auto-staged prompt-template review resolved, as a live run would.

    A real execution resolves every interpretation review before running; the
    committed fixture leaves the ``llm_prompt_template`` review pending. Without
    resolving it, ``materialize_state_for_execution`` short-circuits with an
    interpretation-review error before the retry-budget gate is ever reached,
    which would mask the blocker under test.
    """
    options = dict(node["options"])
    prompt_template = options["prompt_template"]
    resolved_reqs = []
    for requirement in options.get(INTERPRETATION_REQUIREMENTS_KEY, ()):
        requirement = dict(requirement)
        if requirement.get("kind") == "llm_prompt_template":
            requirement.update(
                status="resolved",
                event_id=f"prompt-template-accepted:{node['id']}",
                accepted_value=prompt_template,
                resolved_prompt_template_hash=stable_hash(prompt_template),
            )
        resolved_reqs.append(requirement)
    options[INTERPRETATION_REQUIREMENTS_KEY] = resolved_reqs
    return {**node, "options": options}


def _record_from_committed(state: CompositionState, session_id: UUID) -> CompositionStateRecord:
    committed = state.to_dict()
    nodes = [_resolve_prompt_template_review(node) if node.get("plugin") == "llm" else node for node in committed["nodes"]]
    return CompositionStateRecord(
        id=uuid4(),
        session_id=session_id,
        version=1,
        source=None,
        sources=committed["sources"],
        nodes=nodes,
        edges=committed["edges"],
        outputs=committed["outputs"],
        metadata_=committed["metadata"],
        is_valid=True,
        validation_errors=None,
        created_at=datetime.now(UTC),
        derived_from_state_id=None,
        composer_meta=None,
    )


def _valid_validation_result() -> ValidationResult:
    return ValidationResult(
        is_valid=True,
        checks=[],
        errors=[],
        readiness=ValidationReadiness(
            authoring_valid=True,
            execution_ready=True,
            completion_ready=True,
            blockers=[],
        ),
    )


@pytest.mark.asyncio
async def test_committed_profiled_multi_query_llm_passes_readiness_preflight(parity_env: Any) -> None:
    fixture = _structured_llm_fixture()
    committed = parity_env.reference_state(fixture)

    # The real set_pipeline path persists the profiled multi-query node in
    # AUTHORED-MINIMAL form: profile alias + queries survive, retry budget does
    # not (it is an operator-profile lowering concern). This is the precondition
    # that made the un-lowered preflight gate false-positive.
    assess = next(node for node in committed.to_dict()["nodes"] if node["id"] == "assess_blue")
    assert assess["plugin"] == "llm"
    assert assess["options"].get("profile") == "task-role"
    assert assess["options"].get("queries") is not None
    assert "pool_size" not in assess["options"]
    assert "max_capacity_retry_seconds" not in assess["options"]

    session_id = uuid4()
    app_state = parity_env.app.state
    record = _record_from_committed(committed, session_id)

    session_service = create_autospec(SessionServiceProtocol, instance=True)
    session_service.get_active_run.return_value = None
    session_service.get_current_state.return_value = record
    session_service.create_run.return_value = SimpleNamespace(id=uuid4())

    loop = asyncio.get_running_loop()
    service = ExecutionServiceImpl(
        loop=loop,
        broadcaster=MagicMock(spec=ProgressBroadcaster),
        settings=app_state.settings,
        session_service=session_service,
        yaml_generator=real_yaml_generator,
        telemetry=build_sessions_telemetry(),
        blob_service=None,
        secret_service=None,
        plugin_snapshot_factory=lambda user_id: app_state.plugin_snapshot_factory(UserIdentity(user_id=user_id, username=user_id)),
        operator_profile_registry=app_state.operator_profile_registry,
        web_plugin_policy=app_state.web_plugin_policy,
        catalog=app_state.catalog_service,
    )

    completed: Future[None] = Future()
    completed.set_result(None)

    # Stub validate_pipeline VALID to isolate the execution-service preflight
    # gate (and avoid instantiating the profile's bedrock provider). The
    # retry-budget gate under test is driven by the REAL operator-profile
    # lowering in validate_plugin_policy.
    with (
        patch("elspeth.web.execution.validation.validate_pipeline", return_value=_valid_validation_result()),
        patch.object(service._executor, "submit", return_value=completed),
    ):
        # The profiled multi-query node must clear the retry-budget readiness
        # gate. Were the gate still evaluated on the un-lowered options it would
        # raise ``PipelineValidationError`` (``llm_retry_budget_policy``) here.
        # Instead the only remaining pre-submission control is the separate,
        # expected LLM fanout-ack confirmation — capture its token.
        with pytest.raises(ExecutionFanoutGuardRequired) as fanout_excinfo:
            await service.execute(session_id=session_id, user_id="alice")

        # Re-drive with the fanout ack: the run now clears every readiness gate
        # and reaches submission.
        run_id = await service.execute(
            session_id=session_id,
            user_id="alice",
            fanout_ack_token=fanout_excinfo.value.guard.ack_token,
        )

    assert isinstance(run_id, UUID)
    session_service.create_run.assert_awaited_once()
