"""Characterization tests for the ARG_ERROR pre-dispatch arms of _dispatch_tool_batch.

This file pins arms #1, #2, and #5 of the dispatch loop (the inventory of
terminal arms is completed cumulatively across the Phase-1 characterization
tasks; this file is one task in that set). For each arm covered here the test
asserts the audit-envelope status (``ComposerToolStatus``), the recorded
``error_class`` on the ``_ToolOutcome``, and that exactly one invocation was
buffered. They exist to make the Phase-2 verbatim extraction of the dispatch
loop provably behaviour-preserving for the audit trail — a dropped or reordered
``recorder.record(finish_*)`` on any covered arm, or a rerouted exception
handler that changes the recorded ``error_class``, must turn one of these RED.

Observable surface: the ``_run_one_turn_for_test`` driver returns a
``ComposeLoopTestResult`` exposing only ``.tool_invocations`` (the recorder
buffer of ``ComposerToolInvocation`` records) and ``.tool_outcomes`` (the
``_ToolOutcome`` records). Anti-anchor state and the per-call LLM tool-message
content are NOT observable through this driver, so these tests do not assert on
them.

Arms characterised here:
  #1 — JSON-decode failure (service.py ARG_ERROR pre-dispatch site 1/3)
  #2 — non-dict arguments, valid JSON (service.py ARG_ERROR pre-dispatch site 2/3)
  #5 — required-paths missing (service.py ARG_ERROR pre-dispatch site 3/3)
"""
from __future__ import annotations

import json

import pytest

from elspeth.contracts.composer_audit import ComposerToolStatus
from elspeth.web.composer.service import ComposerServiceImpl

from .conftest import (
    _FakeChoice,
    _FakeComposeLLM,
    _FakeFunction,
    _FakeLLMResponse,
    _FakeMessage,
    _FakeToolCall,
    _fake_llm_response,
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
async def test_arm_json_decode_failure_records_arg_error(
    fake_composer_service: ComposerServiceImpl,
    result_session_id: str,
) -> None:
    """Arm #1: JSON-decode failure → ARG_ERROR with error_class 'JSONDecodeError'.

    Service.py ARG_ERROR pre-dispatch site 1/3 (lines ~2401-2444).
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

    assert len(result.tool_invocations) == 1, (
        f"Expected exactly 1 invocation (one malformed tool call), got {len(result.tool_invocations)}"
    )
    statuses = [inv.status for inv in result.tool_invocations]
    assert ComposerToolStatus.ARG_ERROR in statuses, (
        f"ARG_ERROR not in recorded statuses {statuses!r}; audit trail did not record the decode failure"
    )
    error_classes = [o.error_class for o in result.tool_outcomes]
    assert any(ec == "JSONDecodeError" for ec in error_classes), (
        f"No outcome has error_class='JSONDecodeError'; got {error_classes!r}. "
        "The JSON-decode ARG_ERROR arm may have been rerouted — inspect service.py:2424."
    )


@pytest.mark.asyncio
async def test_arm_non_dict_arguments_records_arg_error(
    fake_composer_service: ComposerServiceImpl,
    result_session_id: str,
) -> None:
    """Arm #2: valid JSON but non-dict (list) arguments → ARG_ERROR with error_class 'TypeError'.

    Service.py ARG_ERROR pre-dispatch site 2/3 (lines ~2446-2495).
    The LLM produced syntactically valid JSON, but it decoded to a list
    rather than a dict (JSON object). The loop records ``finish_arg_error``
    with ``error_class="TypeError"`` (when canonicalization succeeds) or
    the canonicalization exception class (when it fails — not the case for
    a plain list).

    Empirically observed: a JSON list ``[1, 2, 3]`` canonicalizes cleanly via
    ``begin_dispatch_or_arg_error`` (wraps under ``_decoded_non_object``), so
    ``error_class == "TypeError"`` is recorded (service.py:2465). A loose
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
        "The non-dict-args ARG_ERROR arm may have been rerouted — inspect service.py:2465."
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
