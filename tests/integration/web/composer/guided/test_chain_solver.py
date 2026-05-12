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
