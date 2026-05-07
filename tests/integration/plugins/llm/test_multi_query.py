# tests/integration/plugins/llm/test_multi_query.py
"""Integration tests for multi-query LLM transform.

Tests verify the multi-query transform processes multiple named queries
correctly through the executor, using the unified LLMTransform with
provider="azure" and queries dict config.
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
from elspeth.core.landscape.factory import RecorderFactory
from elspeth.engine.executors import TransformExecutor
from elspeth.engine.spans import SpanFactory
from elspeth.plugins.transforms.llm.transform import LLMTransform
from elspeth.testing import make_pipeline_row
from tests.fixtures.factories import make_context
from tests.fixtures.landscape import make_factory

DYNAMIC_SCHEMA = {"mode": "observed"}


def make_full_config() -> dict[str, Any]:
    """Create realistic config with 10 named queries (2 case studies x 5 criteria).

    Uses the unified LLMTransform queries dict format where each query has
    named input_fields mapping template variables to row columns, and
    output_fields defining typed output columns.
    """
    return {
        "provider": "azure",
        "deployment_name": "gpt-4o",
        "endpoint": "https://test.openai.azure.com",
        "api_key": "test-key",
        "system_prompt": 'You are an assessment AI. Respond in JSON: {"score": <0-100>, "rationale": "<text>"}',
        "template": """
Case Study:
Background: {{ row.background }}
Symptoms: {{ row.symptoms }}
History: {{ row.history }}

Criterion: {{ row.criterion_name }}
Description: {{ row.criterion_description }}

Assess this case against the criterion.
""",
        "queries": {
            "cs1_diagnosis": {
                "input_fields": {
                    "background": "cs1_background",
                    "symptoms": "cs1_symptoms",
                    "history": "cs1_history",
                    "criterion_name": "cs1_diagnosis_criterion_name",
                    "criterion_description": "cs1_diagnosis_criterion_desc",
                },
                "output_fields": [
                    {"suffix": "score", "type": "integer"},
                    {"suffix": "rationale", "type": "string"},
                ],
            },
            "cs1_treatment": {
                "input_fields": {
                    "background": "cs1_background",
                    "symptoms": "cs1_symptoms",
                    "history": "cs1_history",
                    "criterion_name": "cs1_treatment_criterion_name",
                    "criterion_description": "cs1_treatment_criterion_desc",
                },
                "output_fields": [
                    {"suffix": "score", "type": "integer"},
                    {"suffix": "rationale", "type": "string"},
                ],
            },
            "cs1_prognosis": {
                "input_fields": {
                    "background": "cs1_background",
                    "symptoms": "cs1_symptoms",
                    "history": "cs1_history",
                    "criterion_name": "cs1_prognosis_criterion_name",
                    "criterion_description": "cs1_prognosis_criterion_desc",
                },
                "output_fields": [
                    {"suffix": "score", "type": "integer"},
                    {"suffix": "rationale", "type": "string"},
                ],
            },
            "cs1_risk": {
                "input_fields": {
                    "background": "cs1_background",
                    "symptoms": "cs1_symptoms",
                    "history": "cs1_history",
                    "criterion_name": "cs1_risk_criterion_name",
                    "criterion_description": "cs1_risk_criterion_desc",
                },
                "output_fields": [
                    {"suffix": "score", "type": "integer"},
                    {"suffix": "rationale", "type": "string"},
                ],
            },
            "cs1_followup": {
                "input_fields": {
                    "background": "cs1_background",
                    "symptoms": "cs1_symptoms",
                    "history": "cs1_history",
                    "criterion_name": "cs1_followup_criterion_name",
                    "criterion_description": "cs1_followup_criterion_desc",
                },
                "output_fields": [
                    {"suffix": "score", "type": "integer"},
                    {"suffix": "rationale", "type": "string"},
                ],
            },
            "cs2_diagnosis": {
                "input_fields": {
                    "background": "cs2_background",
                    "symptoms": "cs2_symptoms",
                    "history": "cs2_history",
                    "criterion_name": "cs2_diagnosis_criterion_name",
                    "criterion_description": "cs2_diagnosis_criterion_desc",
                },
                "output_fields": [
                    {"suffix": "score", "type": "integer"},
                    {"suffix": "rationale", "type": "string"},
                ],
            },
            "cs2_treatment": {
                "input_fields": {
                    "background": "cs2_background",
                    "symptoms": "cs2_symptoms",
                    "history": "cs2_history",
                    "criterion_name": "cs2_treatment_criterion_name",
                    "criterion_description": "cs2_treatment_criterion_desc",
                },
                "output_fields": [
                    {"suffix": "score", "type": "integer"},
                    {"suffix": "rationale", "type": "string"},
                ],
            },
            "cs2_prognosis": {
                "input_fields": {
                    "background": "cs2_background",
                    "symptoms": "cs2_symptoms",
                    "history": "cs2_history",
                    "criterion_name": "cs2_prognosis_criterion_name",
                    "criterion_description": "cs2_prognosis_criterion_desc",
                },
                "output_fields": [
                    {"suffix": "score", "type": "integer"},
                    {"suffix": "rationale", "type": "string"},
                ],
            },
            "cs2_risk": {
                "input_fields": {
                    "background": "cs2_background",
                    "symptoms": "cs2_symptoms",
                    "history": "cs2_history",
                    "criterion_name": "cs2_risk_criterion_name",
                    "criterion_description": "cs2_risk_criterion_desc",
                },
                "output_fields": [
                    {"suffix": "score", "type": "integer"},
                    {"suffix": "rationale", "type": "string"},
                ],
            },
            "cs2_followup": {
                "input_fields": {
                    "background": "cs2_background",
                    "symptoms": "cs2_symptoms",
                    "history": "cs2_history",
                    "criterion_name": "cs2_followup_criterion_name",
                    "criterion_description": "cs2_followup_criterion_desc",
                },
                "output_fields": [
                    {"suffix": "score", "type": "integer"},
                    {"suffix": "rationale", "type": "string"},
                ],
            },
        },
        "schema": {"mode": "observed"},
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
        mock_response.model_dump = Mock(
            return_value={
                "model": "gpt-4o",
                "choices": [{"finish_reason": "stop", "message": {"content": content}}],
            }
        )

        return mock_response

    with patch("openai.AzureOpenAI") as mock_azure_class:
        mock_client = Mock()
        mock_client.chat.completions.create.side_effect = make_response
        mock_azure_class.return_value = mock_client
        yield mock_client


@pytest.fixture
def factory(tmp_path) -> RecorderFactory:
    """Create factory with file-based DB for cross-thread access.

    LLMTransform uses BatchTransformMixin which processes
    rows in a background thread. SQLite in-memory databases are
    per-connection, so the background thread would get an empty DB.
    A file-based temp DB is shared across threads correctly.
    """
    db = LandscapeDB.from_url(f"sqlite:///{tmp_path / 'audit.db'}")
    return make_factory(db)


@pytest.fixture
def executor(factory: RecorderFactory) -> TransformExecutor:
    """Create TransformExecutor for testing."""
    spans = SpanFactory()
    step_resolver = lambda node_id: 0  # noqa: E731
    return TransformExecutor(factory.execution, spans, step_resolver, data_flow=factory.data_flow)


@pytest.fixture
def run_id(factory: RecorderFactory) -> str:
    """Create a run for testing."""
    run = factory.run_lifecycle.begin_run(config={}, canonical_version="v1")
    return run.run_id


@pytest.fixture
def node_id(factory: RecorderFactory, run_id: str) -> str:
    """Create a node for testing."""
    schema = SchemaConfig.from_dict(DYNAMIC_SCHEMA)
    node = factory.data_flow.register_node(
        run_id=run_id,
        plugin_name="llm",
        node_type=NodeType.TRANSFORM,
        plugin_version="1.0",
        config={},
        schema_config=schema,
    )
    return node.node_id


def create_token_in_factory(
    factory: RecorderFactory,
    run_id: str,
    node_id: str,
    row_id: str,
    token_id: str,
    row_data: dict[str, Any],
    row_index: int = 0,
) -> TokenInfo:
    """Create row and token in factory, return TokenInfo."""
    row = factory.data_flow.create_row(
        run_id=run_id,
        source_node_id=node_id,
        row_index=row_index,
        data=row_data,
        row_id=row_id,
    )
    factory.data_flow.create_token(row_id=row.row_id, token_id=token_id)
    # Wrap row_data in PipelineRow with contract
    pipeline_row = make_pipeline_row(row_data)
    return TokenInfo(row_id=row_id, token_id=token_id, row_data=pipeline_row)


class TestMultiQueryIntegration:
    """Full integration tests for multi-query transform via TransformExecutor."""

    def test_full_assessment_matrix(
        self,
        factory: RecorderFactory,
        executor: TransformExecutor,
        run_id: str,
        node_id: str,
    ) -> None:
        """Test complete 2x5 assessment matrix (10 LLM calls)."""
        # Generate responses for all 10 queries
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
            transform = LLMTransform(make_full_config())
            transform.node_id = node_id
            transform.on_error = "discard"

            ctx = make_context(
                run_id=run_id,
                landscape=factory.plugin_audit_writer(),
            )
            transform.on_start(ctx)

            # Build row data with all named input fields that queries reference
            row_data: dict[str, Any] = {
                "user_id": "user-001",
                "cs1_background": "45yo male, office worker",
                "cs1_symptoms": "Chest pain, shortness of breath",
                "cs1_history": "Family history of heart disease",
                "cs2_background": "32yo female, athlete",
                "cs2_symptoms": "Knee pain after running",
                "cs2_history": "Previous ACL injury",
            }
            # Add criterion name/description fields for each query
            criteria_info = {
                "diagnosis": ("Diagnostic accuracy", "Assess diagnostic accuracy"),
                "treatment": ("Treatment plan", "Assess treatment plan"),
                "prognosis": ("Prognosis accuracy", "Assess prognosis accuracy"),
                "risk": ("Risk identification", "Assess risk identification"),
                "followup": ("Follow-up planning", "Assess follow-up planning"),
            }
            for cs in ["cs1", "cs2"]:
                for crit, (name, desc) in criteria_info.items():
                    row_data[f"{cs}_{crit}_criterion_name"] = name
                    row_data[f"{cs}_{crit}_criterion_desc"] = desc

            token = create_token_in_factory(
                factory,
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
            )

            # Should succeed
            assert result.status == "success", f"Expected success, got {result.status}: {result.reason}"
            assert error_sink is None

            # Should have made 10 LLM calls (10 named queries)
            assert mock_client.chat.completions.create.call_count == 10

            # Output should have all assessment fields
            output = result.row
            assert output is not None, "Result row should not be None"
            assert output["user_id"] == "user-001"  # Original preserved

            # Check all score and rationale fields exist (prefixed by query name)
            for cs in ["cs1", "cs2"]:
                for crit in ["diagnosis", "treatment", "prognosis", "risk", "followup"]:
                    query_name = f"{cs}_{crit}"
                    assert f"{query_name}_score" in output, f"Missing {query_name}_score"
                    assert f"{query_name}_rationale" in output, f"Missing {query_name}_rationale"

            # Verify audit trail - LLM calls were recorded via AuditedLLMClient
            from elspeth.contracts import CallStatus, CallType

            assert ctx.state_id is not None
            calls = factory.query.get_calls(ctx.state_id)
            llm_calls = [c for c in calls if c.call_type == CallType.LLM]
            assert len(llm_calls) == 10, f"Expected 10 LLM calls recorded, got {len(llm_calls)}"
            assert all(c.status == CallStatus.SUCCESS for c in llm_calls)

            transform.close()

    def test_multiple_rows_through_multi_query(
        self,
        factory: RecorderFactory,
        executor: TransformExecutor,
        run_id: str,
        node_id: str,
    ) -> None:
        """Verify multiple rows process without deadlock through multi-query transform."""
        # Simplified config: 2 named queries = 2 LLM calls per row
        config: dict[str, Any] = {
            "provider": "azure",
            "deployment_name": "gpt-4o",
            "endpoint": "https://test.openai.azure.com",
            "api_key": "test-key",
            "system_prompt": "Respond in JSON",
            "template": "Assess: {{ row.background }} against {{ row.criterion_name }}",
            "queries": {
                "case_quality": {
                    "input_fields": {
                        "background": "background",
                        "criterion_name": "quality_criterion_name",
                    },
                    "output_fields": [{"suffix": "score", "type": "integer"}],
                },
                "case_safety": {
                    "input_fields": {
                        "background": "background",
                        "criterion_name": "safety_criterion_name",
                    },
                    "output_fields": [{"suffix": "score", "type": "integer"}],
                },
            },
            "schema": {"mode": "observed"},
            "required_input_fields": [],  # Explicit opt-out for this test
            "pool_size": 5,
        }

        # 2 queries per row, cycle for multiple rows
        responses = [
            {"score": 80},
            {"score": 90},
        ]

        with mock_azure_openai_multi_query(responses) as mock_client:
            transform = LLMTransform(config)
            transform.node_id = node_id
            transform.on_error = "discard"

            ctx = make_context(
                run_id=run_id,
                landscape=factory.plugin_audit_writer(),
            )
            transform.on_start(ctx)

            # Process 3 rows
            rows = [
                {"background": "Row 1 data", "quality_criterion_name": "Quality check", "safety_criterion_name": "Safety check"},
                {"background": "Row 2 data", "quality_criterion_name": "Quality check", "safety_criterion_name": "Safety check"},
                {"background": "Row 3 data", "quality_criterion_name": "Quality check", "safety_criterion_name": "Safety check"},
            ]
            results: list[TransformResult] = []

            for i, row_data in enumerate(rows):
                token = create_token_in_factory(
                    factory,
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
                )
                results.append(result)

            # All 3 should succeed (this would hang with the old bug)
            assert all(r.status == "success" for r in results)

            # Each row makes 2 LLM calls (2 queries), so 6 total
            assert mock_client.chat.completions.create.call_count == 6

            # Verify each row has the expected output fields
            for result in results:
                assert result.row is not None
                assert "case_quality_score" in result.row
                assert "case_safety_score" in result.row

            # Verify audit trail - LLM calls recorded for at least the last state
            from elspeth.contracts import CallStatus, CallType

            assert ctx.state_id is not None
            calls = factory.query.get_calls(ctx.state_id)
            llm_calls = [c for c in calls if c.call_type == CallType.LLM]
            assert len(llm_calls) == 2, f"Expected 2 LLM calls for last row, got {len(llm_calls)}"
            assert all(c.status == CallStatus.SUCCESS for c in llm_calls)

            transform.close()
