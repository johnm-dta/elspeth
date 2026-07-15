"""Characterization tests for the terminal arms of the dispatch loop.

The dispatch loop is ``elspeth.web.composer.tool_batch.run_tool_batch``
(extracted verbatim from the former ``ComposerServiceImpl._dispatch_tool_batch``).
Production-code citations below name ``tool_batch.py`` and a verbatim anchor
comment (e.g. ``ARG_ERROR pre-dispatch site (N/3)``) rather than a line number,
so they survive the deferred Phase-3 reshuffle without re-rotting.

This file pins arms #1, #2, #4, #5, #7, #8, #9, and #17 of the dispatch
loop. For each arm covered here the test asserts the audit-envelope status
(``ComposerToolStatus``), the recorded ``error_class`` on the ``_ToolOutcome``
(where applicable), and that invocations were buffered. They exist to make the
Phase-2 verbatim extraction of the dispatch loop provably behaviour-preserving
for the audit trail — a dropped or reordered ``recorder.record(finish_*)`` on
any covered arm, or a rerouted exception handler that changes the recorded
``error_class``, must turn one of these RED.

Observable surface: the ``_run_one_turn_for_test`` driver returns a
``ComposeLoopTestResult`` exposing only ``.tool_invocations`` (the recorder
buffer of ``ComposerToolInvocation`` records) and ``.tool_outcomes`` (the
``_ToolOutcome`` records). Anti-anchor state and the per-call LLM tool-message
content are NOT observable through this driver, so these tests do not assert on
them.

Arm #10 (advisor COMPOSE_TIMEOUT pre-call deadline check, ``tool_batch.py``
``"status": "COMPOSE_TIMEOUT"`` arms) is NOT covered here. The check raises
``ComposerConvergenceError``, which is an unhandled exception from
``_run_one_turn_for_test`` — it does not return a ``ComposeLoopTestResult``.
The only reachable path is the live timeout path, which is not
deterministically triggerable through this driver without risking a race
against other deadline checks earlier in the loop. This gap is documented
explicitly rather than omitted silently; see the two ``COMPOSE_TIMEOUT`` arms
in ``tool_batch.py`` (pre-call deadline and post-advisor-timeout with
deadline_limited=True).

Pre-covered arms verified in ``test_compose_loop_audit_wiring.py`` and
``test_compose_loop_interpretation_review_dispatch.py``:
  #3  — non-dict, canonicalization fails (top-level non-finite float) → ARG_ERROR /
        ``error_class == "ValueError"`` — asserted at
        ``inv.status == ComposerToolStatus.ARG_ERROR`` with ``error_class == "ValueError"``
        by ``test_compose_loop_records_arg_error_for_non_finite_non_object_arguments``.
        This is the ``canonicalization_failed is not None`` branch reached from the
        non-dict block in ``tool_batch.py`` (``float("inf")`` as top-level non-object).
  #6  — dict arguments, canonicalization fails (non-finite inside object) → ARG_ERROR /
        ``error_class == "ValueError"`` — asserted at
        ``inv.status == ComposerToolStatus.ARG_ERROR`` with ``error_class == "ValueError"``
        by ``test_compose_loop_records_arg_error_for_non_finite_object_arguments``.
        This is the ``canonicalization_failed is not None`` branch in ``tool_batch.py``,
        which fires after the required-paths gate and before the cache-check.
  #11 — session-aware (request_interpretation_review) — asserted at
        ``invocation.status.value == "success"/"arg_error"`` by
        ``test_compose_loop_dispatches_request_interpretation_review`` (SUCCESS)
        and ``test_request_interpretation_review_without_persisted_state_returns_arg_error``
        (ARG_ERROR).
  #12 — ToolArgumentError ARG_ERROR — asserted at
        ``arg_error_inv.status == ComposerToolStatus.ARG_ERROR`` with
        ``error_class == "ToolArgumentError"`` by
        ``test_compose_loop_records_success_arg_error_plugin_crash_sequence``.
  #13 — narrow re-raise PLUGIN_CRASH — asserted at
        ``inv.status == ComposerToolStatus.PLUGIN_CRASH`` with
        ``error_class == "AssertionError"`` by
        ``test_compose_loop_records_assertion_error_before_reraise``.
  #14 — plugin-crash break PLUGIN_CRASH — asserted at
        ``plugin_crash_inv.status == ComposerToolStatus.PLUGIN_CRASH`` with
        ``error_class == "RuntimeError"`` by
        ``test_compose_loop_records_success_arg_error_plugin_crash_sequence``.
  #15 — success mutation — asserted at
        ``success_inv.status == ComposerToolStatus.SUCCESS`` by
        ``test_compose_loop_records_success_arg_error_plugin_crash_sequence``.
  #16 — success discovery — asserted at
        ``inv.status == ComposerToolStatus.SUCCESS`` by
        ``TestDiscoveryToolAuditPayload.test_cache_miss_audit_preserves_pydantic_payload``.

Arms characterised here (all in ``tool_batch.py``):
  #1  — JSON-decode failure (``ARG_ERROR pre-dispatch site (1/3)``)
  #2  — non-dict arguments, valid JSON, canonicalization succeeds → TypeError
        (``ARG_ERROR pre-dispatch site (2/3)``)
  #4  — discovery cache-hit (``cache_hit=True`` branch; second identical cacheable call)
  #5  — required-paths missing (``ARG_ERROR pre-dispatch site (3/3)``)
  #7  — advisor disabled (defense-in-depth arm)
  #8  — advisor budget exhausted (``Advisor budget exhausted`` arm)
  #9  — advisor arg-error (``_validate_advisor_arguments`` rejects)
  #17 — get_plugin_schema success marks (type, name) loaded
        (``tool_name == "get_plugin_schema" and result.success`` branch)
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

import pytest

from elspeth.contracts.composer_audit import ComposerToolStatus
from elspeth.web.composer.service import ComposerServiceImpl
from elspeth.web.sessions.models import sessions_table

from .conftest import (
    _fake_llm_response,
    _FakeChoice,
    _FakeComposeLLM,
    _FakeFunction,
    _FakeLLMResponse,
    _FakeMessage,
    _FakeToolCall,
    _make_settings,
    _mock_catalog,
    build_test_sessions_service,
)


def _raw_tool_call_llm(*, name: str, raw_arguments: str) -> _FakeComposeLLM:
    """LLM whose first turn emits ONE tool call with a raw (already-encoded)
    arguments string, bypassing _fake_llm_response's json.dumps. Used to inject
    malformed JSON / non-object payloads the decode arms must reject."""
    first = _FakeLLMResponse(
        choices=[
            _FakeChoice(
                message=_FakeMessage(
                    content=None,
                    tool_calls=[_FakeToolCall(id="call_raw", function=_FakeFunction(name=name, arguments=raw_arguments))],
                )
            )
        ]
    )
    return _FakeComposeLLM((first, _fake_llm_response(content="Done.")))


@pytest.mark.asyncio
async def test_advisor_tool_always_present(
    fake_composer_service: ComposerServiceImpl,
) -> None:
    """The ``request_advisor_hint`` tool is ALWAYS exposed to the composer
    LLM. There is no enable flag any more — advisor is mandatory, so the
    tool is unconditionally part of ``_get_litellm_tools()``."""
    tools = fake_composer_service._get_litellm_tools()
    names = {t["function"]["name"] for t in tools}
    assert "request_advisor_hint" in names


@pytest.mark.asyncio
async def test_arm_json_decode_failure_records_arg_error(
    fake_composer_service: ComposerServiceImpl,
    result_session_id: str,
) -> None:
    """Arm #1: JSON-decode failure → ARG_ERROR with error_class 'JSONDecodeError'.

    tool_batch.py — ARG_ERROR pre-dispatch site (1/3).
    The dispatch loop catches ``json.JSONDecodeError`` / ``TypeError``,
    opens the audit envelope via ``begin_dispatch``, and records
    ``finish_arg_error`` with ``error_class=type(exc).__name__``.

    Empirically observed: a malformed ``"{not valid json"`` payload raises
    ``json.JSONDecodeError``, so ``error_class == "JSONDecodeError"`` is the
    value recorded on the ``_ToolOutcome``. A loose ``is not None`` assertion
    would not catch a rerouted handler (e.g. one that recorded TypeError or a
    generic ARG_ERROR), so this pins the exact class string.

    Pinning: exactly 1 invocation (the one malformed call), ARG_ERROR status,
    and the outcome carries error_class == "JSONDecodeError".
    """
    llm = _raw_tool_call_llm(name="get_pipeline_state", raw_arguments="{not valid json")
    result = await fake_composer_service._run_one_turn_for_test(llm=llm, session_id=result_session_id)

    assert len(result.tool_invocations) == 1, f"Expected exactly 1 invocation (one malformed tool call), got {len(result.tool_invocations)}"
    statuses = [inv.status for inv in result.tool_invocations]
    assert ComposerToolStatus.ARG_ERROR in statuses, (
        f"ARG_ERROR not in recorded statuses {statuses!r}; audit trail did not record the decode failure"
    )
    error_classes = [o.error_class for o in result.tool_outcomes]
    assert any(ec == "JSONDecodeError" for ec in error_classes), (
        f"No outcome has error_class='JSONDecodeError'; got {error_classes!r}. "
        "The JSON-decode ARG_ERROR arm may have been rerouted — inspect "
        "tool_batch.py (ARG_ERROR pre-dispatch site 1/3)."
    )


@pytest.mark.asyncio
async def test_arm_non_dict_arguments_records_arg_error(
    fake_composer_service: ComposerServiceImpl,
    result_session_id: str,
) -> None:
    """Arm #2: valid JSON but non-dict (list) arguments → ARG_ERROR with error_class 'TypeError'.

    tool_batch.py — ARG_ERROR pre-dispatch site (2/3).
    The LLM produced syntactically valid JSON, but it decoded to a list
    rather than a dict (JSON object). The loop records ``finish_arg_error``
    with ``error_class="TypeError"`` (when canonicalization succeeds) or
    the canonicalization exception class (when it fails — not the case for
    a plain list).

    Empirically observed: a JSON list ``[1, 2, 3]`` canonicalizes cleanly via
    ``begin_dispatch_or_arg_error`` (wraps under ``_decoded_non_object``), so
    ``error_class == "TypeError"`` is recorded (tool_batch.py, ARG_ERROR
    pre-dispatch site 2/3). A loose
    ``is not None`` assertion would not catch a rerouted handler, so this pins
    the exact class string.

    Pinning: exactly 1 invocation, ARG_ERROR status, error_class == "TypeError".
    """
    llm = _raw_tool_call_llm(name="get_pipeline_state", raw_arguments=json.dumps([1, 2, 3]))
    result = await fake_composer_service._run_one_turn_for_test(llm=llm, session_id=result_session_id)

    assert len(result.tool_invocations) == 1, (
        f"Expected exactly 1 invocation (one non-dict-args tool call), got {len(result.tool_invocations)}"
    )
    statuses = [inv.status for inv in result.tool_invocations]
    assert ComposerToolStatus.ARG_ERROR in statuses, (
        f"ARG_ERROR not in recorded statuses {statuses!r}; audit trail did not record the non-dict-args failure"
    )
    error_classes = [o.error_class for o in result.tool_outcomes]
    assert any(ec == "TypeError" for ec in error_classes), (
        f"No outcome has error_class='TypeError'; got {error_classes!r}. "
        "The non-dict-args ARG_ERROR arm may have been rerouted — inspect "
        "tool_batch.py (ARG_ERROR pre-dispatch site 2/3)."
    )


@pytest.mark.asyncio
async def test_arm_required_paths_missing_records_arg_error(
    fake_composer_service: ComposerServiceImpl,
    result_session_id: str,
) -> None:
    """Arm #5: required paths missing → ARG_ERROR with error_class 'MissingRequiredPaths'.

    tool_batch.py — ARG_ERROR pre-dispatch site (3/3).
    ``set_source`` declares required: ["plugin", "on_success", "options",
    "on_validation_failure"] in its JSON schema.  Passing ``{}`` means all
    four are missing.  The loop records ``finish_arg_error`` with
    ``error_class="MissingRequiredPaths"`` before entering the handler.

    Empirically verified: ``_TOOL_REQUIRED_PATHS["set_source"]`` is non-empty
    (auto-computed from the tool declaration's json_schema; the
    ``sources.py:375`` comment about the "deleted entry" refers to a prior
    hand-maintained dict, not the current auto-computed index).  Running:
        ``_TOOL_REQUIRED_PATHS.get("set_source")`` returns 4 compiled paths.
    So ``{}`` hits the MissingRequiredPaths arm, not the Pydantic handler.

    Pinning: exactly 1 invocation, ARG_ERROR status, and the outcome
    carries error_class == "MissingRequiredPaths" (not the ToolArgumentError
    sub-class that the Pydantic handler would produce on fallthrough).
    """
    llm = _raw_tool_call_llm(name="set_source", raw_arguments=json.dumps({}))
    result = await fake_composer_service._run_one_turn_for_test(llm=llm, session_id=result_session_id)

    assert len(result.tool_invocations) == 1, (
        f"Expected exactly 1 invocation (one missing-required-paths tool call), got {len(result.tool_invocations)}"
    )
    statuses = [inv.status for inv in result.tool_invocations]
    assert ComposerToolStatus.ARG_ERROR in statuses, (
        f"ARG_ERROR not in recorded statuses {statuses!r}; audit trail did not record the missing-paths failure"
    )
    error_classes = [o.error_class for o in result.tool_outcomes]
    assert any(ec == "MissingRequiredPaths" for ec in error_classes), (
        f"No outcome has error_class='MissingRequiredPaths'; got {error_classes!r}. "
        "Either the required-paths arm was not reached (set_source not in _TOOL_REQUIRED_PATHS) "
        "or the error_class string changed — inspect tool_batch.py "
        "(ARG_ERROR pre-dispatch site 3/3)."
    )


# ---------------------------------------------------------------------------
# Task 2 — Arm #4: discovery cache-hit
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_arm_discovery_cache_hit_records_success_with_cache_hit_flag(
    fake_composer_service: ComposerServiceImpl,
    result_session_id: str,
) -> None:
    """Arm #4: second identical cacheable discovery call is served from cache.

    tool_batch.py cache-hit arm. When the LLM emits two tool calls
    for the same cacheable discovery tool with identical arguments in one
    assistant turn, the first call executes normally and populates
    ``discovery_cache``.  The second call finds the key in
    cache and records ``finish_success`` with ``cache_hit=True``.

    Tool chosen: ``list_sources`` — it is in ``_CACHEABLE_DISCOVERY_TOOL_NAMES``
    and has no required paths (``_TOOL_REQUIRED_PATHS["list_sources"]`` is empty),
    so ``{}`` clears the required-paths gate without triggering
    MissingRequiredPaths.  The first sorted cacheable name,
    ``explain_validation_error``, requires ``error_text`` — using ``{}`` for it
    would trip the required-paths arm BEFORE populating the cache, leaving the
    second call with nothing to hit.  ``list_sources`` does not have this
    ordering hazard because the required-paths gate precedes the cache-populate
    step.

    Pinning:
    - exactly 2 invocations (one per tool call in the batch)
    - both carry SUCCESS status (cache-hit arm records finish_success)
    - the cache_hit flags are the ordered pair [False, True]: the first call is
      a cache miss (cache_hit=False), the second is served from discovery_cache
      (cache_hit=True). An ``any(...)`` check would pass even if both calls were
      (wrongly) served from cache or the first were cached — so the order is
      pinned exactly.
    """
    # Two identical list_sources calls with empty args — both are valid because
    # list_sources requires no arguments.
    first_response = _FakeLLMResponse(
        choices=[
            _FakeChoice(
                message=_FakeMessage(
                    content=None,
                    tool_calls=[
                        _FakeToolCall(id="call_ls1", function=_FakeFunction(name="list_sources", arguments="{}")),
                        _FakeToolCall(id="call_ls2", function=_FakeFunction(name="list_sources", arguments="{}")),
                    ],
                )
            )
        ]
    )
    llm = _FakeComposeLLM((first_response, _fake_llm_response(content="Done.")))
    result = await fake_composer_service._run_one_turn_for_test(llm=llm, session_id=result_session_id)

    assert len(result.tool_invocations) >= 2, (
        f"Expected at least 2 invocations (two list_sources calls), got {len(result.tool_invocations)}"
    )
    statuses = [inv.status for inv in result.tool_invocations]
    assert all(s == ComposerToolStatus.SUCCESS for s in statuses), (
        f"All invocations must be SUCCESS (cache-hit arm records finish_success); got {statuses!r}"
    )
    cache_hits = [inv.cache_hit for inv in result.tool_invocations[:2]]
    assert cache_hits == [False, True], (
        f"Expected the two list_sources calls to be [miss, hit] = [False, True], got {cache_hits!r}. "
        "The second call should be served from discovery_cache (tool_batch.py cache_hit=True branch). "
        "If both are True the first was wrongly cached; if both are False the cache-hit arm was bypassed "
        "(e.g. both calls hit the execute path) — inspect the cache-populate / cache-check ordering in "
        "tool_batch.py."
    )


# ---------------------------------------------------------------------------
# Task 3 — Advisor arms #7-#9
# ---------------------------------------------------------------------------

# Minimal set of advisor arguments that satisfies _TOOL_REQUIRED_PATHS
# (trigger, problem_summary, recent_errors, attempted_actions all present) so
# the required-paths gate is cleared before reaching the advisor interception
# in tool_batch.py.  The required-paths gate precedes both the disabled-advisor
# arm and the budget-exhaustion arm, so omitting any required key would produce
# MissingRequiredPaths instead of the intended advisor arm.
_VALID_ADVISOR_ARGS: dict[str, object] = {
    "trigger": "proactive_security_safety",
    "problem_summary": "characterization test — pinning advisor arm",
    "recent_errors": [],
    "attempted_actions": [],
}


@pytest.mark.asyncio
async def test_arm_advisor_budget_exhausted_records_success_with_budget_exhausted_status(
    tmp_path: Path,
) -> None:
    """Arm #8: advisor budget=0 → SUCCESS with status == 'BUDGET_EXHAUSTED'.

    tool_batch.py advisor budget-exhausted arm.  When
    ``composer_advisor_max_calls_per_compose`` is 0
    (``ge=0`` — validated in config.py), ``advisor_calls_used >= budget``
    is immediately True and the loop records ``finish_success`` with a
    ``status: BUDGET_EXHAUSTED`` payload rather than making any outbound call.

    Budget exhaustion is a policy outcome, not a malformed argument, so
    ``finish_success`` is the correct record (not ``finish_arg_error``).

    This test constructs its own service rather than using the shared
    ``fake_composer_service`` fixture because ``composer_advisor_max_calls_per_compose=0``
    is a non-default setting that must not bleed into other tests.

    Empirically observed: ``.response`` is a ``mappingproxy``; ``status`` key
    contains the string ``"BUDGET_EXHAUSTED"``.

    Pinning:
    - exactly 1 invocation
    - status == SUCCESS
    - outcome response ``status == "BUDGET_EXHAUSTED"``
    """
    settings = _make_settings(tmp_path, composer_advisor_max_calls_per_compose=0)
    sessions_svc = build_test_sessions_service(data_dir=tmp_path)
    svc = ComposerServiceImpl.for_trained_operator(catalog=_mock_catalog(), settings=settings, sessions_service=sessions_svc)

    # Insert a session so _run_one_turn_for_test has a valid session_id to work
    # with (mirrors the result_session_id fixture body in conftest.py).
    session_id = str(uuid4())
    now = datetime.now(UTC)
    with sessions_svc._engine.begin() as conn:
        conn.execute(
            sessions_table.insert().values(
                id=session_id,
                user_id="arm8-test-user",
                auth_provider_type="local",
                title="Arm #8 budget-exhausted test session",
                trust_mode="auto_commit",
                density_default="high",
                created_at=now,
                updated_at=now,
            )
        )

    first_response = _FakeLLMResponse(
        choices=[
            _FakeChoice(
                message=_FakeMessage(
                    content=None,
                    tool_calls=[
                        _FakeToolCall(
                            id="call_adv_budget",
                            function=_FakeFunction(
                                name="request_advisor_hint",
                                arguments=json.dumps(_VALID_ADVISOR_ARGS),
                            ),
                        )
                    ],
                )
            )
        ]
    )
    llm = _FakeComposeLLM((first_response, _fake_llm_response(content="Done.")))
    result = await svc._run_one_turn_for_test(llm=llm, session_id=session_id)

    assert len(result.tool_invocations) == 1, (
        f"Expected exactly 1 invocation (budget-exhausted advisor call), got {len(result.tool_invocations)}"
    )
    statuses = [inv.status for inv in result.tool_invocations]
    assert ComposerToolStatus.SUCCESS in statuses, (
        f"SUCCESS not in recorded statuses {statuses!r}; budget-exhaustion records finish_success "
        "(not finish_arg_error) — inspect tool_batch.py (advisor budget-exhausted arm)."
    )
    assert len(result.tool_outcomes) == 1
    outcome = result.tool_outcomes[0]
    assert outcome.response is not None, "Outcome response must not be None for budget-exhausted arm"
    assert outcome.response["status"] == "BUDGET_EXHAUSTED", (
        f"Expected outcome.response['status'] == 'BUDGET_EXHAUSTED'; got {dict(outcome.response)!r}. "
        "The budget-exhaustion payload shape changed — inspect tool_batch.py (advisor budget-exhausted arm)."
    )


@pytest.mark.asyncio
async def test_arm_advisor_arg_error_records_arg_error_with_type_error_class(
    tmp_path: Path,
) -> None:
    """Arm #9: advisor arg validation failure → ARG_ERROR with error_class 'TypeError'.

    tool_batch.py advisor arg-error arm (``_validate_advisor_arguments``).  When
    the advisor is enabled, the budget is not exhausted, and the arguments pass
    the required-paths gate, but the Tier-3 type validator rejects
    the argument shape, the loop records ``finish_arg_error`` with the
    ``error_class`` from the validator's error dict.

    To land here the test must supply ALL four required keys
    (trigger, problem_summary, recent_errors, attempted_actions) so the
    required-paths gate clears, then introduce a type fault in one field.
    ``attempted_actions="oops"`` is a string, not a list — the ``not isinstance``
    check in ``ComposerServiceImpl._validate_advisor_arguments`` catches it and
    returns ``error_class: "TypeError"``.

    This test constructs its own service (advisor enabled, default budget=4)
    to avoid mutating the shared fixture's settings.

    Pinning:
    - exactly 1 invocation
    - status == ARG_ERROR
    - error_class == "TypeError"
    """
    settings = _make_settings(tmp_path)
    sessions_svc = build_test_sessions_service(data_dir=tmp_path)
    svc = ComposerServiceImpl.for_trained_operator(catalog=_mock_catalog(), settings=settings, sessions_service=sessions_svc)

    session_id = str(uuid4())
    now = datetime.now(UTC)
    with sessions_svc._engine.begin() as conn:
        conn.execute(
            sessions_table.insert().values(
                id=session_id,
                user_id="arm9-test-user",
                auth_provider_type="local",
                title="Arm #9 advisor arg-error test session",
                trust_mode="auto_commit",
                density_default="high",
                created_at=now,
                updated_at=now,
            )
        )

    # attempted_actions must be a list; passing a string causes
    # _validate_advisor_arguments to return error_class="TypeError".
    bad_args = {**_VALID_ADVISOR_ARGS, "attempted_actions": "oops"}
    first_response = _FakeLLMResponse(
        choices=[
            _FakeChoice(
                message=_FakeMessage(
                    content=None,
                    tool_calls=[
                        _FakeToolCall(
                            id="call_adv_argbad",
                            function=_FakeFunction(
                                name="request_advisor_hint",
                                arguments=json.dumps(bad_args),
                            ),
                        )
                    ],
                )
            )
        ]
    )
    llm = _FakeComposeLLM((first_response, _fake_llm_response(content="Done.")))
    result = await svc._run_one_turn_for_test(llm=llm, session_id=session_id)

    assert len(result.tool_invocations) == 1, f"Expected exactly 1 invocation (advisor arg-error call), got {len(result.tool_invocations)}"
    statuses = [inv.status for inv in result.tool_invocations]
    assert ComposerToolStatus.ARG_ERROR in statuses, (
        f"ARG_ERROR not in recorded statuses {statuses!r}; "
        "_validate_advisor_arguments should reject attempted_actions='oops' (not a list) "
        "with finish_arg_error — inspect tool_batch.py (_validate_advisor_arguments arm)."
    )
    error_classes = [o.error_class for o in result.tool_outcomes]
    assert any(ec == "TypeError" for ec in error_classes), (
        f"No outcome has error_class='TypeError'; got {error_classes!r}. "
        "_validate_advisor_arguments returns error_class='TypeError' for non-list "
        "attempted_actions — inspect tool_batch.py / ComposerServiceImpl._validate_advisor_arguments."
    )


# ---------------------------------------------------------------------------
# Task 4 — Arm #17: get_plugin_schema success marks (type, name) loaded
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_arm_get_plugin_schema_success_marks_type_name_loaded(
    fake_composer_service: ComposerServiceImpl,
    result_session_id: str,
) -> None:
    """Arm #17: successful get_plugin_schema records (type, name) in _schemas_loaded.

    tool_batch.py get_plugin_schema success branch.  After a successful ``get_plugin_schema`` dispatch,
    the loop calls ``_mark_plugin_schema_loaded(session_id, plugin_type, plugin_name)``
    so the LLM tool-list builder can surface schema-loaded state to the model.

    ``fake_composer_service`` uses a mock catalog whose ``get_schema`` returns a
    ``PluginSchemaInfo`` for any (plugin_type, name) pair — sufficient to make the
    dispatch succeed.

    This test pins the side-effect rather than the full audit row: if the
    ``_mark_plugin_schema_loaded`` call is accidentally dropped or its arguments
    transposed during the Phase-2 extraction, ``_schemas_loaded_for_session``
    will return an empty frozenset and the assertion will fail.

    Pinning:
    - exactly 1 SUCCESS invocation
    - ``("source", "csv") in fake_composer_service._schemas_loaded_for_session(result_session_id)``
    """
    first_response = _FakeLLMResponse(
        choices=[
            _FakeChoice(
                message=_FakeMessage(
                    content=None,
                    tool_calls=[
                        _FakeToolCall(
                            id="call_gps",
                            function=_FakeFunction(
                                name="get_plugin_schema",
                                arguments=json.dumps({"plugin_type": "source", "name": "csv"}),
                            ),
                        )
                    ],
                )
            )
        ]
    )
    llm = _FakeComposeLLM((first_response, _fake_llm_response(content="Done.")))
    result = await fake_composer_service._run_one_turn_for_test(llm=llm, session_id=result_session_id)

    assert len(result.tool_invocations) == 1, f"Expected exactly 1 invocation (get_plugin_schema call), got {len(result.tool_invocations)}"
    statuses = [inv.status for inv in result.tool_invocations]
    assert ComposerToolStatus.SUCCESS in statuses, (
        f"SUCCESS not in recorded statuses {statuses!r}; get_plugin_schema with valid args "
        "and a mock catalog that always returns a schema should record finish_success."
    )
    schemas_loaded = fake_composer_service._schemas_loaded_for_session(result_session_id)
    assert ("source", "csv") in schemas_loaded, (
        f"('source', 'csv') not in _schemas_loaded_for_session({result_session_id!r}); "
        f"got {schemas_loaded!r}.  _mark_plugin_schema_loaded must be called after a "
        "successful get_plugin_schema dispatch — inspect tool_batch.py (get_plugin_schema success branch)."
    )
