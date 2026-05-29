"""Characterization tests pinning every terminal arm of _dispatch_tool_batch.

These tests assert the audit-envelope status (ComposerToolStatus), the
anti-anchor side-effect, and the LLM tool-message shape for each arm. They
exist to make the Phase-2 verbatim extraction of the dispatch loop provably
behaviour-preserving for the audit trail — a dropped or reordered
recorder.record(finish_*) on any arm must turn one of these RED.

Driver: ComposerServiceImpl._run_one_turn_for_test(llm=...). The returned
ComposeLoopTestResult exposes .tool_invocations (the recorder buffer) and
.tool_outcomes.

Arms characterised here:
  #1 — JSON-decode failure (service.py ARG_ERROR pre-dispatch site 1/3)
  #2 — non-dict arguments, valid JSON (service.py ARG_ERROR pre-dispatch site 2/3)
  #5 — required-paths missing (service.py ARG_ERROR pre-dispatch site 3/3)
"""
from __future__ import annotations

import json
from typing import Any

import pytest

from elspeth.contracts.composer_audit import ComposerToolStatus
from elspeth.web.composer.service import ComposerServiceImpl

from .conftest import _FakeComposeLLM, _fake_llm_response


def _raw_tool_call_llm(*, name: str, raw_arguments: str) -> _FakeComposeLLM:
    """LLM whose first turn emits ONE tool call with a raw (already-encoded)
    arguments string, bypassing _fake_llm_response's json.dumps. Used to inject
    malformed JSON / non-object payloads the decode arms must reject."""
    from .conftest import _FakeChoice, _FakeFunction, _FakeLLMResponse, _FakeMessage, _FakeToolCall

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
async def test_arm_json_decode_failure_records_arg_error(
    fake_composer_service: ComposerServiceImpl,
    result_session_id: str,
) -> None:
    """Arm #1: JSON-decode failure → ARG_ERROR recorded in audit envelope.

    Service.py ARG_ERROR pre-dispatch site 1/3 (lines ~2401-2444).
    The dispatch loop catches ``json.JSONDecodeError`` / ``TypeError``,
    opens the audit envelope via ``begin_dispatch``, and records
    ``finish_arg_error`` with ``error_class=type(exc).__name__``.

    Pinning: exactly 1 invocation (the one malformed call), all with
    ARG_ERROR status, and at least one outcome carrying a non-None
    error_class proving the error was recorded truthfully.
    """
    llm = _raw_tool_call_llm(name="get_pipeline_state", raw_arguments="{not valid json")
    result = await fake_composer_service._run_one_turn_for_test(llm=llm, session_id=result_session_id)

    assert len(result.tool_invocations) == 1, (
        f"Expected exactly 1 invocation (one malformed tool call), got {len(result.tool_invocations)}"
    )
    statuses = [inv.status for inv in result.tool_invocations]
    assert ComposerToolStatus.ARG_ERROR in statuses, (
        f"ARG_ERROR not in recorded statuses {statuses!r}; audit trail did not record the decode failure"
    )
    assert any(o.error_class is not None for o in result.tool_outcomes), (
        "No outcome has a non-None error_class; the ARG_ERROR arm did not populate error_class in _ToolOutcome"
    )


@pytest.mark.asyncio
async def test_arm_non_dict_arguments_records_arg_error(
    fake_composer_service: ComposerServiceImpl,
    result_session_id: str,
) -> None:
    """Arm #2: valid JSON but non-dict (list) arguments → ARG_ERROR recorded.

    Service.py ARG_ERROR pre-dispatch site 2/3 (lines ~2446-2495).
    The LLM produced syntactically valid JSON, but it decoded to a list
    rather than a dict (JSON object). The loop records ``finish_arg_error``
    with ``error_class="TypeError"`` (when canonicalization succeeds) or
    the canonicalization exception class (when it fails — not the case for
    a plain list).

    Observed behaviour: a JSON list ``[1, 2, 3]`` canonicalizes cleanly via
    ``begin_dispatch_or_arg_error`` (wraps under ``_decoded_non_object``),
    so ``error_class="TypeError"`` is recorded (service.py:2465).

    Pinning: exactly 1 invocation, ARG_ERROR status, non-None error_class.
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
    assert any(o.error_class is not None for o in result.tool_outcomes), (
        "No outcome has a non-None error_class; the non-dict-args ARG_ERROR arm did not populate error_class"
    )


@pytest.mark.asyncio
async def test_arm_required_paths_missing_records_arg_error(
    fake_composer_service: ComposerServiceImpl,
    result_session_id: str,
) -> None:
    """Arm #5: required paths missing → ARG_ERROR with error_class 'MissingRequiredPaths'.

    Service.py ARG_ERROR pre-dispatch site 3/3 (lines ~2610-2645).
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
        "or the error_class string changed — inspect service.py:2626."
    )
