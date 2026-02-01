# tests/integration/test_multi_query_integration.py
"""Integration tests for Azure Multi-Query LLM transform.

Tests verify the multi-query transform processes a 2x5 assessment matrix
(2 case studies x 5 criteria = 10 LLM calls) correctly through the executor.
"""

from __future__ import annotations

import json
from collections.abc import Generator
from contextlib import contextmanager
from typing import Any
from unittest.mock import Mock, patch

import pytest

from elspeth.contracts import NodeType, TransformResult
from elspeth.contracts.identity import TokenInfo
from elspeth.contracts.schema import SchemaConfig
from elspeth.core.landscape.database import LandscapeDB
from elspeth.core.landscape.recorder import LandscapeRecorder
from elspeth.engine.executors import TransformExecutor
from elspeth.engine.spans import SpanFactory
from elspeth.plugins.context import PluginContext
from elspeth.plugins.llm.azure_multi_query import AzureMultiQueryLLMTransform

DYNAMIC_SCHEMA = {"fields": "dynamic"}


def make_full_config() -> dict[str, Any]:
    """Create realistic config with 2 case studies x 5 criteria."""
    return {
        "deployment_name": "gpt-4o",
        "endpoint": "https://test.openai.azure.com",
        "api_key": "test-key",
        "system_prompt": 'You are an assessment AI. Respond in JSON: {"score": <0-100>, "rationale": "<text>"}',
        "template": """
Case Study:
Background: {{ row.input_1 }}
Symptoms: {{ row.input_2 }}
History: {{ row.input_3 }}

Criterion: {{ row.criterion.name }}
Description: {{ row.criterion.description }}

Assess this case against the criterion.
""",
        "case_studies": [
            {"name": "cs1", "input_fields": ["cs1_background", "cs1_symptoms", "cs1_history"]},
            {"name": "cs2", "input_fields": ["cs2_background", "cs2_symptoms", "cs2_history"]},
        ],
        "criteria": [
            {"name": "diagnosis", "code": "DIAG", "description": "Assess diagnostic accuracy"},
            {"name": "treatment", "code": "TREAT", "description": "Assess treatment plan"},
            {"name": "prognosis", "code": "PROG", "description": "Assess prognosis accuracy"},
            {"name": "risk", "code": "RISK", "description": "Assess risk identification"},
            {"name": "followup", "code": "FOLLOW", "description": "Assess follow-up planning"},
        ],
        "response_format": "standard",
        "output_mapping": {
            "score": {"suffix": "score", "type": "integer"},
            "rationale": {"suffix": "rationale", "type": "string"},
        },
        "schema": {"fields": "dynamic"},
        "required_input_fields": [],  # Explicit opt-out for this test
        "pool_size": 10,  # All 10 queries in parallel
        "temperature": 0.0,
    }


@contextmanager
def mock_azure_openai_multi_query(
    responses: list[dict[str, Any]],
) -> Generator[Mock, None, None]:
    """Context manager to mock Azure OpenAI for multi-query processing.

    Args:
        responses: List of response dicts to return in order (cycles if exhausted)

    Yields:
        Mock client instance for assertions
    """
    call_count = [0]

    def make_response(**kwargs: Any) -> Mock:
        content = json.dumps(responses[call_count[0] % len(responses)])
        call_count[0] += 1

        mock_usage = Mock()
        mock_usage.prompt_tokens = 50
        mock_usage.completion_tokens = 20

        mock_message = Mock()
        mock_message.content = content

        mock_choice = Mock()
        mock_choice.message = mock_message

        mock_response = Mock()
        mock_response.choices = [mock_choice]
        mock_response.model = "gpt-4o"
        mock_response.usage = mock_usage
        mock_response.model_dump = Mock(return_value={"model": "gpt-4o"})

        return mock_response

    with patch("openai.AzureOpenAI") as mock_azure_class:
        mock_client = Mock()
        mock_client.chat.completions.create.side_effect = make_response
        mock_azure_class.return_value = mock_client
        yield mock_client


@pytest.fixture
def recorder() -> LandscapeRecorder:
    """Create recorder with in-memory DB."""
    db = LandscapeDB.in_memory()
    rec = LandscapeRecorder(db)
    rec.record_call = Mock()  # type: ignore[method-assign]
    return rec


@pytest.fixture
def executor(recorder: LandscapeRecorder) -> TransformExecutor:
    """Create TransformExecutor for testing."""
    spans = SpanFactory()
    return TransformExecutor(recorder, spans)


@pytest.fixture
def run_id(recorder: LandscapeRecorder) -> str:
    """Create a run for testing."""
    run = recorder.begin_run(config={}, canonical_version="v1")
    return run.run_id


@pytest.fixture
def node_id(recorder: LandscapeRecorder, run_id: str) -> str:
    """Create a node for testing."""
    schema = SchemaConfig.from_dict(DYNAMIC_SCHEMA)
    node = recorder.register_node(
        run_id=run_id,
        plugin_name="azure_multi_query_llm",
        node_type=NodeType.TRANSFORM,
        plugin_version="1.0",
        config={},
        schema_config=schema,
    )
    return node.node_id


def create_token_in_recorder(
    recorder: LandscapeRecorder,
    run_id: str,
    node_id: str,
    row_id: str,
    token_id: str,
    row_data: dict[str, Any],
    row_index: int = 0,
) -> TokenInfo:
    """Create row and token in recorder, return TokenInfo."""
    row = recorder.create_row(
        run_id=run_id,
        source_node_id=node_id,
        row_index=row_index,
        data=row_data,
        row_id=row_id,
    )
    recorder.create_token(row_id=row.row_id, token_id=token_id)
    return TokenInfo(row_id=row_id, token_id=token_id, row_data=row_data)


class TestMultiQueryIntegration:
    """Full integration tests for multi-query transform via TransformExecutor."""

    def test_full_assessment_matrix(
        self,
        recorder: LandscapeRecorder,
        executor: TransformExecutor,
        run_id: str,
        node_id: str,
    ) -> None:
        """Test complete 2x5 assessment matrix (10 LLM calls)."""
        # Generate responses for all 10 queries
        # Order: cs1 x all criteria, then cs2 x all criteria
        responses: list[dict[str, Any]] = []
        for cs in ["cs1", "cs2"]:
            for crit in ["diagnosis", "treatment", "prognosis", "risk", "followup"]:
                responses.append(
                    {
                        "score": len(responses) * 10,
                        "rationale": f"Assessment for {cs}_{crit}",
                    }
                )

        with mock_azure_openai_multi_query(responses) as mock_client:
            transform = AzureMultiQueryLLMTransform(make_full_config())
            transform.node_id = node_id

            ctx = PluginContext(
                run_id=run_id,
                config={},
                landscape=recorder,
            )
            transform.on_start(ctx)

            row_data = {
                "user_id": "user-001",
                "cs1_background": "45yo male, office worker",
                "cs1_symptoms": "Chest pain, shortness of breath",
                "cs1_history": "Family history of heart disease",
                "cs2_background": "32yo female, athlete",
                "cs2_symptoms": "Knee pain after running",
                "cs2_history": "Previous ACL injury",
            }

            token = create_token_in_recorder(
                recorder,
                run_id,
                node_id,
                row_id="row-1",
                token_id="token-1",
                row_data=row_data,
            )

            result, _, error_sink = executor.execute_transform(
                transform=transform,
                token=token,
                ctx=ctx,
                step_in_pipeline=0,
            )

            # Should succeed
            assert result.status == "success", f"Expected success, got {result.status}: {result.reason}"
            assert error_sink is None

            # Should have made 10 LLM calls (2 case studies x 5 criteria)
            assert mock_client.chat.completions.create.call_count == 10

            # Output should have all 20 assessment fields (10 scores + 10 rationales)
            output = result.row
            assert output is not None, "Result row should not be None"
            assert output["user_id"] == "user-001"  # Original preserved

            # Check all score and rationale fields exist
            for cs in ["cs1", "cs2"]:
                for crit in ["diagnosis", "treatment", "prognosis", "risk", "followup"]:
                    assert f"{cs}_{crit}_score" in output, f"Missing {cs}_{crit}_score"
                    assert f"{cs}_{crit}_rationale" in output, f"Missing {cs}_{crit}_rationale"

            transform.close()

    def test_multiple_rows_through_multi_query(
        self,
        recorder: LandscapeRecorder,
        executor: TransformExecutor,
        run_id: str,
        node_id: str,
    ) -> None:
        """Verify multiple rows process without deadlock through multi-query transform."""
        # Simplified config: 1 case study x 2 criteria = 2 LLM calls per row
        config = {
            "deployment_name": "gpt-4o",
            "endpoint": "https://test.openai.azure.com",
            "api_key": "test-key",
            "system_prompt": "Respond in JSON",
            "template": "Assess: {{ row.input_1 }} against {{ row.criterion.name }}",
            "case_studies": [
                {"name": "case", "input_fields": ["background"]},
            ],
            "criteria": [
                {"name": "quality", "code": "Q", "description": "Quality check"},
                {"name": "safety", "code": "S", "description": "Safety check"},
            ],
            "response_format": "standard",
            "output_mapping": {"score": {"suffix": "score", "type": "integer"}},
            "schema": {"fields": "dynamic"},
            "required_input_fields": [],  # Explicit opt-out for this test
            "pool_size": 5,
        }

        # 2 criteria per row, cycle for multiple rows
        responses = [
            {"score": 80},
            {"score": 90},
        ]

        with mock_azure_openai_multi_query(responses) as mock_client:
            transform = AzureMultiQueryLLMTransform(config)
            transform.node_id = node_id

            ctx = PluginContext(
                run_id=run_id,
                config={},
                landscape=recorder,
            )
            transform.on_start(ctx)

            # Process 3 rows
            rows = [
                {"background": "Row 1 data"},
                {"background": "Row 2 data"},
                {"background": "Row 3 data"},
            ]
            results: list[TransformResult] = []

            for i, row_data in enumerate(rows):
                token = create_token_in_recorder(
                    recorder,
                    run_id,
                    node_id,
                    row_id=f"row-{i}",
                    token_id=f"token-{i}",
                    row_data=row_data,
                    row_index=i,
                )

                result, _, _error_sink = executor.execute_transform(
                    transform=transform,
                    token=token,
                    ctx=ctx,
                    step_in_pipeline=0,
                )
                results.append(result)

            # All 3 should succeed (this would hang with the old bug)
            assert all(r.status == "success" for r in results)

            # Each row makes 2 LLM calls (1 case x 2 criteria), so 6 total
            assert mock_client.chat.completions.create.call_count == 6

            # Verify each row has the expected output fields
            for result in results:
                assert result.row is not None
                assert "case_quality_score" in result.row
                assert "case_safety_score" in result.row

            transform.close()
