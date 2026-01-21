# tests/integration/test_multi_query_integration.py
"""Integration tests for Azure Multi-Query LLM transform."""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import Mock, patch

from elspeth.plugins.context import PluginContext
from elspeth.plugins.llm.azure_multi_query import AzureMultiQueryLLMTransform


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
        "response_format": "json",
        "output_mapping": {"score": "score", "rationale": "rationale"},
        "schema": {"fields": "dynamic"},
        "pool_size": 10,  # All 10 queries in parallel
        "temperature": 0.0,
    }


def make_plugin_context(state_id: str = "state-123") -> PluginContext:
    """Create a PluginContext with mocked landscape."""
    mock_landscape = Mock()
    mock_landscape.record_external_call = Mock()
    mock_landscape.record_call = Mock()
    return PluginContext(
        run_id="run-123",
        config={},
        landscape=mock_landscape,
        state_id=state_id,
    )


class TestMultiQueryIntegration:
    """Full integration tests for multi-query transform."""

    def test_full_assessment_matrix(self) -> None:
        """Test complete 2x5 assessment matrix."""
        # Generate responses for all 10 queries
        # Build responses in the expected call order (cs1 x all criteria, then cs2 x all criteria)
        responses: list[dict[str, Any]] = []
        for cs in ["cs1", "cs2"]:
            for crit in ["diagnosis", "treatment", "prognosis", "risk", "followup"]:
                responses.append(
                    {
                        "score": len(responses) * 10,
                        "rationale": f"Assessment for {cs}_{crit}",
                    }
                )

        call_idx = [0]

        def make_response(**kwargs: Any) -> Mock:
            content = json.dumps(responses[call_idx[0] % len(responses)])
            call_idx[0] += 1

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
            mock_response.model_dump = Mock(return_value={})

            return mock_response

        with patch("openai.AzureOpenAI") as mock_azure_class:
            mock_client = Mock()
            mock_client.chat.completions.create.side_effect = lambda **kwargs: make_response(**kwargs)
            mock_azure_class.return_value = mock_client

            transform = AzureMultiQueryLLMTransform(make_full_config())

            # Create context and call on_start to set up recorder
            start_ctx = make_plugin_context(state_id="init-state")
            transform.on_start(start_ctx)

            # Process row
            ctx = make_plugin_context(state_id="state-456")

            row = {
                "user_id": "user-001",
                "cs1_background": "45yo male, office worker",
                "cs1_symptoms": "Chest pain, shortness of breath",
                "cs1_history": "Family history of heart disease",
                "cs2_background": "32yo female, athlete",
                "cs2_symptoms": "Knee pain after running",
                "cs2_history": "Previous ACL injury",
            }

            result = transform.process(row, ctx)

            # Should succeed
            assert result.status == "success", f"Expected success, got {result.status}: {result.reason}"

            # Should have made 10 LLM calls
            assert mock_client.chat.completions.create.call_count == 10

            # Output should have all 20 assessment fields (10 scores + 10 rationales)
            output = result.row
            assert output is not None, "Result row should not be None"
            assert output["user_id"] == "user-001"  # Original preserved

            # Check all score fields exist
            for cs in ["cs1", "cs2"]:
                for crit in ["diagnosis", "treatment", "prognosis", "risk", "followup"]:
                    assert f"{cs}_{crit}_score" in output, f"Missing {cs}_{crit}_score"
                    assert f"{cs}_{crit}_rationale" in output, f"Missing {cs}_{crit}_rationale"

            transform.close()
