"""Tests for guided-mode chain solver (stubbed LLM only).

The real-LLM gated test lives in Task 4.6 closure once the real_llm marker
is registered.
"""

from __future__ import annotations

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
async def test_model_and_temperature_passed_to_litellm() -> None:
    """solve_chain passes the supplied model and temperature=0.0 to _litellm_acompletion.

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
        )

    assert len(captured_calls) == 1
    call = captured_calls[0]
    # Model must be the caller-supplied value, not any hard-coded string.
    assert call["model"] == TEST_MODEL
    # Temperature must match the composer constant (0.0) for deterministic output.
    assert call["temperature"] == 0.0
