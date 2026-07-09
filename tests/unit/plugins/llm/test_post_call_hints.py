# tests/unit/plugins/llm/test_post_call_hints.py
"""Focused coverage for LLM transform post-call hints."""

from __future__ import annotations

from elspeth.plugins.transforms.llm.transform import LLMTransform


class TestLLMPostCallHints:
    def test_manual_usage_model_fields_warns_for_normal_schema_alias(self) -> None:
        """Composer passes LLM node options with ``schema.fields``, not ``output_schema.fields``."""
        hints = LLMTransform.get_post_call_hints(
            tool_name="upsert_node",
            config_snapshot={
                "response_field": "analysis",
                "schema": {
                    "mode": "fixed",
                    "fields": [
                        "analysis: str",
                        "analysis_usage: dict",
                        "analysis_model: str",
                    ],
                },
            },
        )

        assert len(hints) == 1
        assert "analysis_usage" in hints[0]
        assert "analysis_model" in hints[0]
        assert "schema.fields" in hints[0]

    def test_manual_usage_model_fields_absent_for_schema_alias_has_no_warning(self) -> None:
        hints = LLMTransform.get_post_call_hints(
            tool_name="upsert_node",
            config_snapshot={
                "response_field": "analysis",
                "schema": {
                    "mode": "fixed",
                    "fields": [
                        "analysis: str",
                        "confidence: float",
                    ],
                },
            },
        )

        assert hints == ()
