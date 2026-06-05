"""Tests for guided-mode chain solver (stubbed LLM only).

The real-LLM gated test lives in Task 4.6 closure once the real_llm marker
is registered.
"""

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest


def _make_propose_chain_response(plugin: str = "type_coerce") -> SimpleNamespace:
    """Build a LiteLLM-shaped propose_chain response for the given plugin."""
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(
                    tool_calls=[
                        SimpleNamespace(
                            function=SimpleNamespace(
                                name="emit_turn",
                                arguments=json.dumps(
                                    {
                                        "turn_type": "propose_chain",
                                        "payload": {
                                            "steps": [
                                                {
                                                    "plugin": plugin,
                                                    "options": {"fields": [{"name": "price", "type": "float"}]},
                                                    "rationale": "test rationale",
                                                }
                                            ],
                                            "why": "bridge str→float for arithmetic",
                                        },
                                    }
                                ),
                            )
                        )
                    ],
                )
            )
        ]
    )


@pytest.mark.asyncio
async def test_returns_chain_proposal() -> None:
    from elspeth.web.composer.guided.chain_solver import solve_chain
    from elspeth.web.composer.guided.state_machine import (
        SinkOutputResolved,
        SinkResolved,
        SourceResolved,
    )

    fake_response = SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(
                    tool_calls=[
                        SimpleNamespace(
                            function=SimpleNamespace(
                                name="emit_turn",
                                arguments=json.dumps(
                                    {
                                        "turn_type": "propose_chain",
                                        "payload": {
                                            "steps": [
                                                {
                                                    "plugin": "type_coerce",
                                                    "options": {"fields": [{"name": "price", "type": "float"}]},
                                                    "rationale": "price is str; downstream needs float",
                                                }
                                            ],
                                            "why": "bridge str→float for arithmetic",
                                        },
                                    }
                                ),
                            )
                        )
                    ],
                )
            )
        ]
    )

    with patch(
        "elspeth.web.composer.guided.chain_solver._litellm_acompletion",
        new_callable=AsyncMock,
        return_value=fake_response,
    ):
        proposal = await solve_chain(
            model="anthropic/claude-3-opus",
            source=SourceResolved(
                plugin="csv",
                options={},
                observed_columns=("price",),
                sample_rows=({"price": "1.99"},),
            ),
            sink=SinkResolved(
                outputs=(
                    SinkOutputResolved(
                        plugin="json",
                        options={},
                        required_fields=("price",),
                        schema_mode="fixed",
                    ),
                )
            ),
            temperature=None,
            seed=None,
        )

    assert len(proposal.steps) == 1
    assert proposal.steps[0]["plugin"] == "type_coerce"
    assert proposal.why == "bridge str→float for arithmetic"


@pytest.mark.asyncio
async def test_repair_context_appears_in_system_prompt() -> None:
    """solve_chain with repair_context= appends the repair addendum to the system prompt.

    Verifies that the repair context is visible in the messages passed to
    _litellm_acompletion — proving the addendum reaches the LLM.
    """
    from elspeth.web.composer.guided.chain_solver import solve_chain
    from elspeth.web.composer.guided.state_machine import (
        SinkOutputResolved,
        SinkResolved,
        SourceResolved,
    )

    repair_error = "plugin 'bad_plugin' not found in catalogue"

    fake_response = _make_propose_chain_response()

    captured_calls: list = []

    async def _capture(**kwargs):  # type: ignore[no-untyped-def]
        captured_calls.append(kwargs)
        return fake_response

    with patch(
        "elspeth.web.composer.guided.chain_solver._litellm_acompletion",
        side_effect=_capture,
    ):
        await solve_chain(
            model="anthropic/claude-3-opus",
            source=SourceResolved(
                plugin="csv",
                options={},
                observed_columns=("price",),
                sample_rows=({"price": "1.99"},),
            ),
            sink=SinkResolved(
                outputs=(
                    SinkOutputResolved(
                        plugin="json",
                        options={},
                        required_fields=("price",),
                        schema_mode="fixed",
                    ),
                )
            ),
            repair_context=repair_error,
            temperature=None,
            seed=None,
        )

    assert len(captured_calls) == 1
    messages = captured_calls[0]["messages"]
    system_content = messages[0]["content"]
    # The repair addendum must be present verbatim.
    assert "REPAIR ATTEMPT" in system_content
    assert repair_error in system_content


@pytest.mark.asyncio
async def test_solve_chain_without_repair_context_has_no_repair_section() -> None:
    """solve_chain without repair_context= does not add a REPAIR ATTEMPT section."""
    from elspeth.web.composer.guided.chain_solver import solve_chain
    from elspeth.web.composer.guided.state_machine import (
        SinkOutputResolved,
        SinkResolved,
        SourceResolved,
    )

    fake_response = _make_propose_chain_response()
    captured_calls: list = []

    async def _capture(**kwargs):  # type: ignore[no-untyped-def]
        captured_calls.append(kwargs)
        return fake_response

    with patch(
        "elspeth.web.composer.guided.chain_solver._litellm_acompletion",
        side_effect=_capture,
    ):
        await solve_chain(
            model="anthropic/claude-3-opus",
            source=SourceResolved(
                plugin="csv",
                options={},
                observed_columns=("price",),
                sample_rows=({"price": "1.99"},),
            ),
            sink=SinkResolved(
                outputs=(
                    SinkOutputResolved(
                        plugin="json",
                        options={},
                        required_fields=("price",),
                        schema_mode="fixed",
                    ),
                )
            ),
            temperature=None,
            seed=None,
        )

    assert len(captured_calls) == 1
    system_content = captured_calls[0]["messages"][0]["content"]
    assert "REPAIR ATTEMPT" not in system_content


# ---------------------------------------------------------------------------
# Schema and shape-failure tests (P2 — chain-solver response-shape constraint).
#
# The LLM tool schema (``_GUIDED_LLM_TOOLS``) is the primary defense; the
# consumer-side parsing block is the backstop.  These tests pin both layers:
#   * Schema shape — ``turn_type`` enum is restricted to ``["propose_chain"]``;
#     ``payload`` requires ``steps`` and ``why``.  If anyone widens the schema
#     by accident, the first test fires.
#   * Consumer-side: any tool-call name / turn_type / payload shape mismatch
#     raises :class:`ChainSolverResponseShapeError` (NOT ``InvariantError`` or
#     ``KeyError``).  The auto-drop wrapper catches this class and routes
#     through SOLVER_EXHAUSTED -- integration coverage lives in
#     ``test_auto_drop.py::TestI2ChainSolverTransientFailure``.
# ---------------------------------------------------------------------------


def test_tool_schema_constrains_turn_type_to_propose_chain_only() -> None:
    """The LLM tool schema must restrict ``turn_type`` to a single value.

    Widening the enum here without updating the consumer would re-introduce
    the P2 bug (LLM returns an allowed-but-unhandled turn_type, request
    escapes as a shape error instead of taking the auto-drop path).
    """
    from elspeth.web.composer.guided.chain_solver import _GUIDED_LLM_TOOLS

    params = _GUIDED_LLM_TOOLS[0]["function"]["parameters"]
    assert params["properties"]["turn_type"]["enum"] == ["propose_chain"]


def test_tool_schema_constrains_payload_required_keys() -> None:
    """The LLM tool schema must declare ``steps`` and ``why`` as required.

    Strict-mode-capable providers (OpenAI) enforce this at the wire; for
    others, the consumer-side backstop in ``solve_chain`` is the safety net.
    Either way, this test pins the contract.
    """
    from elspeth.web.composer.guided.chain_solver import _GUIDED_LLM_TOOLS

    params = _GUIDED_LLM_TOOLS[0]["function"]["parameters"]
    payload_schema = params["properties"]["payload"]
    assert sorted(payload_schema["required"]) == ["steps", "why"]
    assert payload_schema["additionalProperties"] is False


def _make_solve_chain_args() -> dict:
    """Common solve_chain kwargs for the shape-failure tests below."""
    from elspeth.web.composer.guided.state_machine import (
        SinkOutputResolved,
        SinkResolved,
        SourceResolved,
    )

    return {
        "model": "anthropic/claude-3-opus",
        "source": SourceResolved(
            plugin="csv",
            options={},
            observed_columns=("price",),
            sample_rows=({"price": "1.99"},),
        ),
        "sink": SinkResolved(
            outputs=(
                SinkOutputResolved(
                    plugin="json",
                    options={},
                    required_fields=("price",),
                    schema_mode="fixed",
                ),
            )
        ),
        "temperature": None,
        "seed": None,
    }


@pytest.mark.asyncio
async def test_solve_chain_wrong_tool_name_raises_shape_error() -> None:
    from elspeth.web.composer.guided.chain_solver import (
        ChainSolverResponseShapeError,
        solve_chain,
    )

    wrong_name_response = SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(
                    tool_calls=[
                        SimpleNamespace(
                            function=SimpleNamespace(name="not_emit_turn", arguments="{}"),
                        )
                    ],
                )
            )
        ]
    )
    with (
        patch(
            "elspeth.web.composer.guided.chain_solver._litellm_acompletion",
            new_callable=AsyncMock,
            return_value=wrong_name_response,
        ),
        pytest.raises(ChainSolverResponseShapeError, match="emit_turn"),
    ):
        await solve_chain(**_make_solve_chain_args())


@pytest.mark.asyncio
async def test_solve_chain_wrong_turn_type_raises_shape_error() -> None:
    from elspeth.web.composer.guided.chain_solver import (
        ChainSolverResponseShapeError,
        solve_chain,
    )

    wrong_type = SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(
                    tool_calls=[
                        SimpleNamespace(
                            function=SimpleNamespace(
                                name="emit_turn",
                                arguments=json.dumps({"turn_type": "single_select", "payload": {}}),
                            ),
                        )
                    ],
                )
            )
        ]
    )
    with (
        patch(
            "elspeth.web.composer.guided.chain_solver._litellm_acompletion",
            new_callable=AsyncMock,
            return_value=wrong_type,
        ),
        pytest.raises(ChainSolverResponseShapeError, match="propose_chain"),
    ):
        await solve_chain(**_make_solve_chain_args())


@pytest.mark.asyncio
async def test_solve_chain_missing_payload_steps_raises_shape_error() -> None:
    from elspeth.web.composer.guided.chain_solver import (
        ChainSolverResponseShapeError,
        solve_chain,
    )

    missing_steps = SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(
                    tool_calls=[
                        SimpleNamespace(
                            function=SimpleNamespace(
                                name="emit_turn",
                                arguments=json.dumps({"turn_type": "propose_chain", "payload": {"why": "no steps"}}),
                            ),
                        )
                    ],
                )
            )
        ]
    )
    with (
        patch(
            "elspeth.web.composer.guided.chain_solver._litellm_acompletion",
            new_callable=AsyncMock,
            return_value=missing_steps,
        ),
        pytest.raises(ChainSolverResponseShapeError, match="steps/why"),
    ):
        await solve_chain(**_make_solve_chain_args())


@pytest.mark.asyncio
async def test_solve_chain_missing_payload_why_raises_shape_error() -> None:
    from elspeth.web.composer.guided.chain_solver import (
        ChainSolverResponseShapeError,
        solve_chain,
    )

    missing_why = SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(
                    tool_calls=[
                        SimpleNamespace(
                            function=SimpleNamespace(
                                name="emit_turn",
                                arguments=json.dumps(
                                    {
                                        "turn_type": "propose_chain",
                                        "payload": {
                                            "steps": [
                                                {
                                                    "plugin": "noop",
                                                    "options": {},
                                                    "rationale": "stub",
                                                }
                                            ]
                                        },
                                    }
                                ),
                            ),
                        )
                    ],
                )
            )
        ]
    )
    with (
        patch(
            "elspeth.web.composer.guided.chain_solver._litellm_acompletion",
            new_callable=AsyncMock,
            return_value=missing_why,
        ),
        pytest.raises(ChainSolverResponseShapeError, match="steps/why"),
    ):
        await solve_chain(**_make_solve_chain_args())


@pytest.mark.asyncio
async def test_solve_chain_non_dict_step_element_raises_shape_error() -> None:
    """``payload.steps`` element that isn't dict-coercible (e.g., a bare int)
    must fail in the ``tuple(dict(s) for s in ...)`` coercion and surface as
    :class:`ChainSolverResponseShapeError`, not :class:`TypeError`."""
    from elspeth.web.composer.guided.chain_solver import (
        ChainSolverResponseShapeError,
        solve_chain,
    )

    bad_step = SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(
                    tool_calls=[
                        SimpleNamespace(
                            function=SimpleNamespace(
                                name="emit_turn",
                                arguments=json.dumps(
                                    {
                                        "turn_type": "propose_chain",
                                        "payload": {"steps": [42], "why": "garbage step"},
                                    }
                                ),
                            ),
                        )
                    ],
                )
            )
        ]
    )
    with (
        patch(
            "elspeth.web.composer.guided.chain_solver._litellm_acompletion",
            new_callable=AsyncMock,
            return_value=bad_step,
        ),
        pytest.raises(ChainSolverResponseShapeError, match="list of dicts"),
    ):
        await solve_chain(**_make_solve_chain_args())


@pytest.mark.asyncio
async def test_solve_chain_empty_tool_calls_raises_shape_error() -> None:
    """An empty ``tool_calls`` list -- the LLM responded without invoking
    the tool -- is an external-system shape failure, not a server
    invariant violation.  ``_extract_tool_call`` now raises
    :class:`ChainSolverResponseShapeError` for consistency with the other
    shape-failure paths."""
    from elspeth.web.composer.guided.chain_solver import (
        ChainSolverResponseShapeError,
        solve_chain,
    )

    no_tool_calls = SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(tool_calls=[]))])
    with (
        patch(
            "elspeth.web.composer.guided.chain_solver._litellm_acompletion",
            new_callable=AsyncMock,
            return_value=no_tool_calls,
        ),
        pytest.raises(ChainSolverResponseShapeError, match="no tool_calls"),
    ):
        await solve_chain(**_make_solve_chain_args())


@pytest.mark.asyncio
async def test_model_and_operator_sampling_passed_to_litellm() -> None:
    """solve_chain passes the supplied model and caller-supplied sampling.

    Asymmetry probe: if model=model is reverted to a hard-coded string, the
    ``captured_calls[0]["model"] == TEST_MODEL`` assertion fails.
    """
    from elspeth.web.composer.guided.chain_solver import solve_chain
    from elspeth.web.composer.guided.state_machine import (
        SinkOutputResolved,
        SinkResolved,
        SourceResolved,
    )

    TEST_MODEL = "openai/gpt-4o-mini"

    fake_response = _make_propose_chain_response()
    captured_calls: list = []

    async def _capture(**kwargs):  # type: ignore[no-untyped-def]
        captured_calls.append(kwargs)
        return fake_response

    with patch(
        "elspeth.web.composer.guided.chain_solver._litellm_acompletion",
        side_effect=_capture,
    ):
        await solve_chain(
            model=TEST_MODEL,
            source=SourceResolved(
                plugin="csv",
                options={},
                observed_columns=("price",),
                sample_rows=({"price": "1.99"},),
            ),
            sink=SinkResolved(
                outputs=(
                    SinkOutputResolved(
                        plugin="json",
                        options={},
                        required_fields=("price",),
                        schema_mode="fixed",
                    ),
                )
            ),
            temperature=0.0,
            seed=42,
        )

    assert len(captured_calls) == 1
    call = captured_calls[0]
    # Model must be the caller-supplied value, not any hard-coded string.
    assert call["model"] == TEST_MODEL
    # Sampling must be caller-supplied, not an internal constant/probe.
    assert call["temperature"] == 0.0
    assert call["seed"] == 42


# ---------------------------------------------------------------------------
# LLM-call audit recording — every solve_chain invocation must append exactly
# one ComposerLLMCall to the recorder, regardless of outcome.  Mirrors the
# freeform composer pattern at ``composer/service.py:3173-3309`` (advisor) and
# ``:3311-3406`` (compose-loop call).  Without this audit:
#
# - Guided Step 3 LLM calls are invisible to the audit trail
# - Token usage / cost / latency for the chain solver are unmeasurable
# - Failure mode classification (timeout vs auth vs malformed) is lost
#
# The drain into Landscape happens at the route-handler ``finally`` block
# (see ``routes.py`` ``post_guided_respond`` and ``get_guided`` —
# ``_persist_llm_calls(... recorder.llm_calls ...)``).  These tests cover
# the in-memory side of that contract; the persistence side is exercised by
# the routes integration tests.
# ---------------------------------------------------------------------------


def _make_resolved_source_and_sink():  # type: ignore[no-untyped-def]
    """Standard (source, sink) pair reused across audit tests."""
    from elspeth.web.composer.guided.state_machine import (
        SinkOutputResolved,
        SinkResolved,
        SourceResolved,
    )

    source = SourceResolved(
        plugin="csv",
        options={},
        observed_columns=("price",),
        sample_rows=({"price": "1.99"},),
    )
    sink = SinkResolved(
        outputs=(
            SinkOutputResolved(
                plugin="json",
                options={},
                required_fields=("price",),
                schema_mode="fixed",
            ),
        )
    )
    return source, sink


class TestSolveChainLLMCallAudit:
    """ComposerLLMCall recording for every solve_chain invocation."""

    @pytest.mark.asyncio
    async def test_success_appends_one_call_with_status_success(self) -> None:
        from elspeth.contracts.composer_llm_audit import ComposerLLMCallStatus
        from elspeth.web.composer.audit import BufferingRecorder
        from elspeth.web.composer.guided.chain_solver import solve_chain

        recorder = BufferingRecorder()
        source, sink = _make_resolved_source_and_sink()
        with patch(
            "elspeth.web.composer.guided.chain_solver._litellm_acompletion",
            new_callable=AsyncMock,
            return_value=_make_propose_chain_response(),
        ):
            proposal = await solve_chain(
                model="openai/gpt-4o-mini",
                source=source,
                sink=sink,
                recorder=recorder,
                temperature=None,
                seed=None,
            )

        assert proposal.steps[0]["plugin"] == "type_coerce"
        assert len(recorder.llm_calls) == 1
        call = recorder.llm_calls[0]
        assert call.status is ComposerLLMCallStatus.SUCCESS
        # Caller-supplied model lands verbatim — asymmetric probe against any
        # hard-coding regression in the audit record.
        assert call.model_requested == "openai/gpt-4o-mini"
        assert call.error_class is None
        assert call.error_message is None
        # Wire-integrity hashes must be present — these prove the audit row
        # carries the bytes actually sent to LiteLLM, not just a status label.
        assert call.messages_hash, "messages_hash must be a non-empty canonical hash"
        assert call.tools_spec_hash, "tools_spec_hash must be a non-empty canonical hash"
        # Latency is a monotonic-ns diff floor-divided by 1_000_000 — may be 0
        # under fast stubs but must never be negative.
        assert call.latency_ms >= 0
        # Tool-invocation channel is unaffected by LLM-call recording.
        assert recorder.invocations == ()

    @pytest.mark.asyncio
    async def test_no_recorder_records_nothing_and_returns_proposal(self) -> None:
        """``recorder=None`` short-circuits the audit write without raising.

        Preserves the contract that pre-existing tests (which don't pass a
        recorder) still work after the audit instrumentation lands.
        """
        from elspeth.web.composer.guided.chain_solver import solve_chain

        source, sink = _make_resolved_source_and_sink()
        with patch(
            "elspeth.web.composer.guided.chain_solver._litellm_acompletion",
            new_callable=AsyncMock,
            return_value=_make_propose_chain_response(),
        ):
            proposal = await solve_chain(
                model="openai/gpt-4o-mini",
                source=source,
                sink=sink,
                temperature=None,
                seed=None,
                # recorder defaults to None
            )

        assert proposal.steps[0]["plugin"] == "type_coerce"

    @pytest.mark.asyncio
    async def test_timeout_records_status_timeout_and_reraises(self) -> None:
        from elspeth.contracts.composer_llm_audit import ComposerLLMCallStatus
        from elspeth.web.composer.audit import BufferingRecorder
        from elspeth.web.composer.guided.chain_solver import solve_chain

        recorder = BufferingRecorder()
        source, sink = _make_resolved_source_and_sink()
        with (
            patch(
                "elspeth.web.composer.guided.chain_solver._litellm_acompletion",
                new_callable=AsyncMock,
                side_effect=TimeoutError("upstream timeout"),
            ),
            pytest.raises(TimeoutError),
        ):
            await solve_chain(
                model="openai/gpt-4o-mini",
                source=source,
                sink=sink,
                recorder=recorder,
                temperature=None,
                seed=None,
            )

        assert len(recorder.llm_calls) == 1
        call = recorder.llm_calls[0]
        assert call.status is ComposerLLMCallStatus.TIMEOUT
        assert call.error_class == "TimeoutError"
        # Per CLAUDE.md, never capture str(exc) — the audit message must be
        # the exception class name only.  Probe by asserting the upstream
        # exception's leak-shaped detail is NOT in error_message.
        assert call.error_message == "TimeoutError"
        assert "upstream timeout" not in (call.error_message or "")

    @pytest.mark.asyncio
    async def test_cancelled_records_status_cancelled_and_reraises(self) -> None:
        from elspeth.contracts.composer_llm_audit import ComposerLLMCallStatus
        from elspeth.web.composer.audit import BufferingRecorder
        from elspeth.web.composer.guided.chain_solver import solve_chain

        recorder = BufferingRecorder()
        source, sink = _make_resolved_source_and_sink()
        with (
            patch(
                "elspeth.web.composer.guided.chain_solver._litellm_acompletion",
                new_callable=AsyncMock,
                side_effect=asyncio.CancelledError(),
            ),
            pytest.raises(asyncio.CancelledError),
        ):
            await solve_chain(
                model="openai/gpt-4o-mini",
                source=source,
                sink=sink,
                recorder=recorder,
                temperature=None,
                seed=None,
            )

        assert len(recorder.llm_calls) == 1
        call = recorder.llm_calls[0]
        assert call.status is ComposerLLMCallStatus.CANCELLED

    @pytest.mark.asyncio
    async def test_litellm_auth_error_records_status_auth_error(self) -> None:
        from litellm.exceptions import AuthenticationError as LiteLLMAuthError

        from elspeth.contracts.composer_llm_audit import ComposerLLMCallStatus
        from elspeth.web.composer.audit import BufferingRecorder
        from elspeth.web.composer.guided.chain_solver import solve_chain

        recorder = BufferingRecorder()
        source, sink = _make_resolved_source_and_sink()
        auth_err = LiteLLMAuthError(
            message="bad key",
            llm_provider="openai",
            model="gpt-4o-mini",
        )
        with (
            patch(
                "elspeth.web.composer.guided.chain_solver._litellm_acompletion",
                new_callable=AsyncMock,
                side_effect=auth_err,
            ),
            pytest.raises(LiteLLMAuthError),
        ):
            await solve_chain(
                model="openai/gpt-4o-mini",
                source=source,
                sink=sink,
                recorder=recorder,
                temperature=None,
                seed=None,
            )

        assert len(recorder.llm_calls) == 1
        call = recorder.llm_calls[0]
        assert call.status is ComposerLLMCallStatus.AUTH_ERROR
        assert call.error_class == "AuthenticationError"

    @pytest.mark.asyncio
    async def test_litellm_bad_request_records_status_bad_request(self) -> None:
        from litellm.exceptions import BadRequestError as LiteLLMBadRequestError

        from elspeth.contracts.composer_llm_audit import ComposerLLMCallStatus
        from elspeth.web.composer.audit import BufferingRecorder
        from elspeth.web.composer.guided.chain_solver import solve_chain

        recorder = BufferingRecorder()
        source, sink = _make_resolved_source_and_sink()
        bad_req = LiteLLMBadRequestError(
            message="malformed request",
            llm_provider="openai",
            model="gpt-4o-mini",
        )
        with (
            patch(
                "elspeth.web.composer.guided.chain_solver._litellm_acompletion",
                new_callable=AsyncMock,
                side_effect=bad_req,
            ),
            pytest.raises(LiteLLMBadRequestError),
        ):
            await solve_chain(
                model="openai/gpt-4o-mini",
                source=source,
                sink=sink,
                recorder=recorder,
                temperature=None,
                seed=None,
            )

        assert len(recorder.llm_calls) == 1
        call = recorder.llm_calls[0]
        assert call.status is ComposerLLMCallStatus.BAD_REQUEST_ERROR
        assert call.error_class == "BadRequestError"

    @pytest.mark.asyncio
    async def test_litellm_api_error_records_status_api_error(self) -> None:
        from litellm.exceptions import APIError as LiteLLMAPIError

        from elspeth.contracts.composer_llm_audit import ComposerLLMCallStatus
        from elspeth.web.composer.audit import BufferingRecorder
        from elspeth.web.composer.guided.chain_solver import solve_chain

        recorder = BufferingRecorder()
        source, sink = _make_resolved_source_and_sink()
        api_err = LiteLLMAPIError(
            status_code=500,
            message="upstream provider failure",
            llm_provider="openai",
            model="gpt-4o-mini",
        )
        with (
            patch(
                "elspeth.web.composer.guided.chain_solver._litellm_acompletion",
                new_callable=AsyncMock,
                side_effect=api_err,
            ),
            pytest.raises(LiteLLMAPIError),
        ):
            await solve_chain(
                model="openai/gpt-4o-mini",
                source=source,
                sink=sink,
                recorder=recorder,
                temperature=None,
                seed=None,
            )

        assert len(recorder.llm_calls) == 1
        call = recorder.llm_calls[0]
        assert call.status is ComposerLLMCallStatus.API_ERROR
        assert call.error_class == "APIError"

    @pytest.mark.asyncio
    async def test_empty_choices_records_status_malformed_response(self) -> None:
        """Empty ``response.choices`` raises ``IndexError`` from ``_extract_tool_call``."""
        from elspeth.contracts.composer_llm_audit import ComposerLLMCallStatus
        from elspeth.web.composer.audit import BufferingRecorder
        from elspeth.web.composer.guided.chain_solver import solve_chain

        recorder = BufferingRecorder()
        source, sink = _make_resolved_source_and_sink()
        empty_response = SimpleNamespace(choices=[])

        with (
            patch(
                "elspeth.web.composer.guided.chain_solver._litellm_acompletion",
                new_callable=AsyncMock,
                return_value=empty_response,
            ),
            pytest.raises(IndexError),
        ):
            await solve_chain(
                model="openai/gpt-4o-mini",
                source=source,
                sink=sink,
                recorder=recorder,
                temperature=None,
                seed=None,
            )

        assert len(recorder.llm_calls) == 1
        call = recorder.llm_calls[0]
        assert call.status is ComposerLLMCallStatus.MALFORMED_RESPONSE
        assert call.error_class == "IndexError"

    @pytest.mark.asyncio
    async def test_bad_arguments_json_records_status_malformed_response(self) -> None:
        """A tool call with invalid JSON ``arguments`` raises ``JSONDecodeError``."""
        from elspeth.contracts.composer_llm_audit import ComposerLLMCallStatus
        from elspeth.web.composer.audit import BufferingRecorder
        from elspeth.web.composer.guided.chain_solver import solve_chain

        recorder = BufferingRecorder()
        source, sink = _make_resolved_source_and_sink()
        bad_json_response = SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        tool_calls=[
                            SimpleNamespace(
                                function=SimpleNamespace(
                                    name="emit_turn",
                                    arguments="{not valid json",
                                )
                            )
                        ]
                    )
                )
            ]
        )

        with (
            patch(
                "elspeth.web.composer.guided.chain_solver._litellm_acompletion",
                new_callable=AsyncMock,
                return_value=bad_json_response,
            ),
            pytest.raises(json.JSONDecodeError),
        ):
            await solve_chain(
                model="openai/gpt-4o-mini",
                source=source,
                sink=sink,
                recorder=recorder,
                temperature=None,
                seed=None,
            )

        assert len(recorder.llm_calls) == 1
        call = recorder.llm_calls[0]
        assert call.status is ComposerLLMCallStatus.MALFORMED_RESPONSE
        assert call.error_class == "JSONDecodeError"

    @pytest.mark.asyncio
    async def test_missing_payload_steps_key_records_malformed_response(self) -> None:
        """Payload missing the ``steps`` key raises ``ChainSolverResponseShapeError`` → MALFORMED_RESPONSE.

        Post-P2, the consumer-side parser wraps the underlying ``KeyError``
        from ``payload["steps"]`` into ``ChainSolverResponseShapeError`` (with
        the original KeyError preserved on ``__cause__``).  The audit row
        classifies as MALFORMED_RESPONSE via the typed-except clause and the
        error_class records the wrapper class name.
        """
        from elspeth.contracts.composer_llm_audit import ComposerLLMCallStatus
        from elspeth.web.composer.audit import BufferingRecorder
        from elspeth.web.composer.guided.chain_solver import (
            ChainSolverResponseShapeError,
            solve_chain,
        )

        recorder = BufferingRecorder()
        source, sink = _make_resolved_source_and_sink()
        no_steps = SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        tool_calls=[
                            SimpleNamespace(
                                function=SimpleNamespace(
                                    name="emit_turn",
                                    arguments=json.dumps(
                                        {
                                            "turn_type": "propose_chain",
                                            "payload": {"why": "missing steps"},
                                        }
                                    ),
                                )
                            )
                        ]
                    )
                )
            ]
        )

        with (
            patch(
                "elspeth.web.composer.guided.chain_solver._litellm_acompletion",
                new_callable=AsyncMock,
                return_value=no_steps,
            ),
            pytest.raises(ChainSolverResponseShapeError) as exc_info,
        ):
            await solve_chain(
                model="openai/gpt-4o-mini",
                source=source,
                sink=sink,
                recorder=recorder,
                temperature=None,
                seed=None,
            )

        # The underlying KeyError is preserved on __cause__ for forensics.
        assert isinstance(exc_info.value.__cause__, KeyError)

        assert len(recorder.llm_calls) == 1
        call = recorder.llm_calls[0]
        assert call.status is ComposerLLMCallStatus.MALFORMED_RESPONSE
        assert call.error_class == "ChainSolverResponseShapeError"

    @pytest.mark.asyncio
    async def test_no_tool_calls_records_status_malformed_response(self) -> None:
        """Empty ``tool_calls`` raises ``ChainSolverResponseShapeError`` → MALFORMED_RESPONSE.

        Post-P2 (chain-solver tool-shape constraint), the "no tool_calls" /
        "wrong tool name" / "wrong turn_type" gates raise
        :class:`ChainSolverResponseShapeError` — an explicit external-LLM-shape
        failure class.  The audit wrap's typed-except clause catches it
        alongside :class:`IndexError` / :class:`AttributeError` /
        :class:`json.JSONDecodeError` and maps to :attr:`MALFORMED_RESPONSE`,
        which is the semantically correct classification (the LLM did
        respond, the response failed contract).
        """
        from elspeth.contracts.composer_llm_audit import ComposerLLMCallStatus
        from elspeth.web.composer.audit import BufferingRecorder
        from elspeth.web.composer.guided.chain_solver import (
            ChainSolverResponseShapeError,
            solve_chain,
        )

        recorder = BufferingRecorder()
        source, sink = _make_resolved_source_and_sink()
        no_tool_calls = SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(tool_calls=[]),
                )
            ]
        )

        with (
            patch(
                "elspeth.web.composer.guided.chain_solver._litellm_acompletion",
                new_callable=AsyncMock,
                return_value=no_tool_calls,
            ),
            pytest.raises(ChainSolverResponseShapeError),
        ):
            await solve_chain(
                model="openai/gpt-4o-mini",
                source=source,
                sink=sink,
                recorder=recorder,
                temperature=None,
                seed=None,
            )

        assert len(recorder.llm_calls) == 1
        call = recorder.llm_calls[0]
        assert call.status is ComposerLLMCallStatus.MALFORMED_RESPONSE
        assert call.error_class == "ChainSolverResponseShapeError"

    @pytest.mark.asyncio
    async def test_escaping_exception_carries_llm_calls_attribute(self) -> None:
        """``attach_llm_calls`` attaches the recorder snapshot to the escaping exception.

        Mirrors the freeform pattern at ``composer/service.py:3307-3309``.
        The escaping exception's ``llm_calls`` attribute lets a downstream
        handler (or the auto-drop wrapper) inspect what the recorder captured
        even after control has left the recorder's drain scope.
        """
        from elspeth.web.composer.audit import BufferingRecorder
        from elspeth.web.composer.guided.chain_solver import solve_chain

        recorder = BufferingRecorder()
        source, sink = _make_resolved_source_and_sink()
        with patch(
            "elspeth.web.composer.guided.chain_solver._litellm_acompletion",
            new_callable=AsyncMock,
            side_effect=TimeoutError("upstream timeout"),
        ):
            try:
                await solve_chain(
                    model="openai/gpt-4o-mini",
                    source=source,
                    sink=sink,
                    recorder=recorder,
                    temperature=None,
                    seed=None,
                )
            except TimeoutError as exc:
                captured = exc

        # The escaping exception carries the in-recorder snapshot — same as
        # what a route handler would see from ``recorder.llm_calls``.
        assert getattr(captured, "llm_calls", None) is not None
        assert tuple(captured.llm_calls) == recorder.llm_calls  # type: ignore[attr-defined]
        assert len(captured.llm_calls) == 1  # type: ignore[attr-defined]
