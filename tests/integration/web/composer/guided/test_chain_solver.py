"""Tests for guided-mode chain solver (stubbed LLM only).

The real-LLM gated test lives in Task 4.6 closure once the real_llm marker
is registered.
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest


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
