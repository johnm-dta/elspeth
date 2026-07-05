# tests/unit/plugins/llm/test_p1_bug_fixes.py
"""Regression tests for P1 LLM plugin bug fixes (2026-02-14 batch).

Each test class corresponds to one bug fix:
1. Azure process_row mutable ctx.state_id in cleanup
2. Base LLM transform output schema diverges from output_schema_config
3. enable_content_recording accepted but never applied

(The original bug 2 — OpenRouter batch HTTP client eviction — and bug 4 —
Azure batch terminal failure call recording — covered the retired batch-LLM
transforms and were removed alongside ADR-020.)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from types import ModuleType
from typing import Any
from unittest.mock import patch

from elspeth.contracts.schema import SchemaConfig
from elspeth.plugins.transforms.llm import (
    _build_augmented_output_schema,
)
from elspeth.plugins.transforms.llm.transform import LLMTransform
from elspeth.testing import make_pipeline_row
from tests.fixtures.factories import make_context

from .conftest import DYNAMIC_SCHEMA


@dataclass
class RecordingAzureMonitorConfigurator:
    calls: list[dict[str, Any]] = field(default_factory=list)

    def __call__(self, **kwargs: Any) -> None:
        self.calls.append(kwargs)


@dataclass
class RecordingAIInferenceInstrumentor:
    instrument_calls: list[dict[str, Any]] = field(default_factory=list)

    def instrument(self, **kwargs: Any) -> None:
        self.instrument_calls.append(kwargs)


@dataclass
class RecordingAIInferenceInstrumentorFactory:
    instances: list[RecordingAIInferenceInstrumentor] = field(default_factory=list)

    def __call__(self) -> RecordingAIInferenceInstrumentor:
        instance = RecordingAIInferenceInstrumentor()
        self.instances.append(instance)
        return instance


def _tracing_module_with(
    instrumentor_factory: RecordingAIInferenceInstrumentorFactory,
) -> ModuleType:
    tracing_module = ModuleType("azure.ai.inference.tracing")
    tracing_module.AIInferenceInstrumentor = instrumentor_factory
    return tracing_module


def _only_instrumentor(
    instrumentor_factory: RecordingAIInferenceInstrumentorFactory,
) -> RecordingAIInferenceInstrumentor:
    assert len(instrumentor_factory.instances) == 1
    return instrumentor_factory.instances[0]


# ---------------------------------------------------------------------------
# Bug 1: Azure process_row uses mutable ctx.state_id in cleanup
# ---------------------------------------------------------------------------


class TestAzureStateIdSnapshot:
    """Regression: ctx.state_id must be snapshotted at _process_row entry.

    The engine can rewrite ctx.state_id between attempts on the same context.
    If the finally block uses ctx.state_id instead of the snapshot, it can
    evict the wrong cached client during retry/timeout races.

    Migrated to LLMTransform: the unified transform delegates to _provider,
    which no longer uses a per-state_id client cache. This test verifies that
    _process_row correctly uses the state_id from ctx at call time (the strategy
    captures it), and that processing succeeds with different state_ids.
    """

    def test_process_row_uses_snapshot_for_cleanup(self, chaosllm_server: Any) -> None:
        """Verify that _process_row works correctly even when ctx.state_id
        is mutated between calls."""
        from .conftest import chaosllm_azure_openai_client

        config = {
            "provider": "azure",
            "deployment_name": "test-deploy",
            "endpoint": "https://test.openai.azure.com",
            "api_key": "test-key",
            "prompt_template": "hello",
            "schema": DYNAMIC_SCHEMA,
            "required_input_fields": [],
        }

        with chaosllm_azure_openai_client(chaosllm_server, mode="echo"):
            transform = LLMTransform(config)
            ctx = make_context(state_id="state-A")
            transform.on_start(ctx)

            row = make_pipeline_row({"text": "test"})

            # Process row with state_id="state-A"
            result = transform._process_row(row, ctx)
            assert result.status == "success"

            # Mutate ctx.state_id and process again — the strategy should
            # snapshot state_id at entry and use it consistently within
            # the call, so this must also succeed cleanly.
            ctx.state_id = "state-X"
            result2 = transform._process_row(row, ctx)
            assert result2.status == "success"

            transform.close()


class TestLLMOutputSchemaDivergence:
    """Regression: output_schema must include LLM-added fields.

    output_schema_config.guaranteed_fields includes llm_response, _usage, _model
    but output_schema was a copy of input_schema and lacked these fields.
    This caused DAG validation failures for explicit-schema pipelines.
    """

    def test_build_augmented_output_schema_observed_passthrough(self) -> None:
        """_build_augmented_output_schema returns dynamic schema for observed mode."""
        schema_config = SchemaConfig(mode="observed", fields=None)
        result = _build_augmented_output_schema(
            base_schema_config=schema_config,
            response_field="llm_response",
            schema_name="TestObserved",
        )
        # Dynamic schema has no fields and allows extras
        assert len(result.model_fields) == 0
        assert result.model_config["extra"] == "allow"

    def test_llm_transform_has_augmented_output_schema(self) -> None:
        """LLMTransform output_schema differs from input_schema when explicit."""
        with patch("openai.AzureOpenAI"):
            transform = LLMTransform(
                {
                    "provider": "azure",
                    "deployment_name": "test",
                    "endpoint": "https://test.azure.com",
                    "api_key": "key",
                    "prompt_template": "hello",
                    "schema": {"mode": "flexible", "fields": ["text: str"]},
                    "required_input_fields": [],
                }
            )

        # output_schema should have LLM fields
        assert "llm_response" in transform.output_schema.model_fields
        # input_schema should not
        assert "llm_response" not in transform.input_schema.model_fields


# ---------------------------------------------------------------------------
# Bug 5: enable_content_recording accepted but never applied
# ---------------------------------------------------------------------------


class TestEnableContentRecording:
    """Regression: enable_content_recording must be wired to Azure Monitor setup.

    The config field was accepted and logged but never passed to the Azure
    Monitor SDK or environment variable, leaving it as a dead config field.

    Note on mocking strategy: _configure_azure_monitor() uses module-level imports:
    - `configure_azure_monitor` is imported at module level in providers/azure.py
    - `from azure.ai.inference.tracing import AIInferenceInstrumentor` is a local import

    We must patch the already-imported reference at
    ``elspeth.plugins.transforms.llm.providers.azure.configure_azure_monitor``, NOT the
    source module ``azure.monitor.opentelemetry.configure_azure_monitor``.
    For AIInferenceInstrumentor, we inject a mock module into sys.modules
    since the real package is not installed.

    Each test resets the module-level idempotency guard via
    ``_reset_azure_monitor_state()`` to ensure isolation.
    """

    def test_content_recording_wired_via_instrumentor(self) -> None:
        """enable_content_recording is passed to AIInferenceInstrumentor when available."""
        import sys

        from elspeth.plugins.transforms.llm.providers.azure import (
            _configure_azure_monitor,
            _reset_azure_monitor_state,
        )
        from elspeth.plugins.transforms.llm.tracing import AzureAITracingConfig

        _reset_azure_monitor_state()

        config = AzureAITracingConfig(
            connection_string="InstrumentationKey=test-key",
            enable_content_recording=True,
            enable_live_metrics=False,
        )

        azure_monitor = RecordingAzureMonitorConfigurator()
        instrumentor_factory = RecordingAIInferenceInstrumentorFactory()

        # Inject a fake azure.ai.inference.tracing module so the local import succeeds
        tracing_module = _tracing_module_with(instrumentor_factory)

        try:
            with (
                patch("elspeth.plugins.transforms.llm.providers.azure.configure_azure_monitor", new=azure_monitor),
                patch.dict(sys.modules, {"azure.ai.inference.tracing": tracing_module}),
            ):
                result = _configure_azure_monitor(config)

            assert result is True
            assert azure_monitor.calls == [
                {
                    "connection_string": "InstrumentationKey=test-key",
                    "enable_live_metrics": False,
                }
            ]
            instrumentor = _only_instrumentor(instrumentor_factory)
            assert instrumentor.instrument_calls == [{"enable_content_recording": True}]
        finally:
            _reset_azure_monitor_state()

    def test_content_recording_false_wired_via_instrumentor(self) -> None:
        """enable_content_recording=False is correctly passed through."""
        import sys

        from elspeth.plugins.transforms.llm.providers.azure import (
            _configure_azure_monitor,
            _reset_azure_monitor_state,
        )
        from elspeth.plugins.transforms.llm.tracing import AzureAITracingConfig

        _reset_azure_monitor_state()

        config = AzureAITracingConfig(
            connection_string="InstrumentationKey=test-key",
            enable_content_recording=False,
            enable_live_metrics=False,
        )

        azure_monitor = RecordingAzureMonitorConfigurator()
        instrumentor_factory = RecordingAIInferenceInstrumentorFactory()

        tracing_module = _tracing_module_with(instrumentor_factory)

        try:
            with (
                patch("elspeth.plugins.transforms.llm.providers.azure.configure_azure_monitor", new=azure_monitor),
                patch.dict(sys.modules, {"azure.ai.inference.tracing": tracing_module}),
            ):
                _configure_azure_monitor(config)

            assert azure_monitor.calls == [
                {
                    "connection_string": "InstrumentationKey=test-key",
                    "enable_live_metrics": False,
                }
            ]
            instrumentor = _only_instrumentor(instrumentor_factory)
            assert instrumentor.instrument_calls == [{"enable_content_recording": False}]
        finally:
            _reset_azure_monitor_state()

    def test_content_recording_falls_back_to_env_var(self) -> None:
        """When AIInferenceInstrumentor is not available, falls back to env var.

        Since azure.ai.inference is not installed in the test environment,
        the ImportError path is the natural path. We just need to mock
        configure_azure_monitor and verify the env var is set.
        """
        import os

        from elspeth.plugins.transforms.llm.providers.azure import (
            _configure_azure_monitor,
            _reset_azure_monitor_state,
        )
        from elspeth.plugins.transforms.llm.tracing import AzureAITracingConfig

        _reset_azure_monitor_state()

        config = AzureAITracingConfig(
            connection_string="InstrumentationKey=test-key",
            enable_content_recording=True,
            enable_live_metrics=False,
        )

        # Remove any injected mock from previous tests to ensure ImportError path
        env_key = "AZURE_TRACING_GEN_AI_CONTENT_RECORDING_ENABLED"
        old_value = os.environ.pop(env_key, None)
        azure_monitor = RecordingAzureMonitorConfigurator()
        try:
            with patch("elspeth.plugins.transforms.llm.providers.azure.configure_azure_monitor", new=azure_monitor):
                _configure_azure_monitor(config)

            assert azure_monitor.calls == [
                {
                    "connection_string": "InstrumentationKey=test-key",
                    "enable_live_metrics": False,
                }
            ]
            assert os.environ.get(env_key) == "true"
        finally:
            # Clean up
            _reset_azure_monitor_state()
            os.environ.pop(env_key, None)
            if old_value is not None:
                os.environ[env_key] = old_value

    def test_enable_live_metrics_forwarded_when_true(self) -> None:
        """enable_live_metrics=True is forwarded to configure_azure_monitor SDK call.

        All existing tests use enable_live_metrics=False. This test catches
        a hardcoded False that would pass all other tests but silently ignore
        the user's configuration.
        """
        import sys

        from elspeth.plugins.transforms.llm.providers.azure import (
            _configure_azure_monitor,
            _reset_azure_monitor_state,
        )
        from elspeth.plugins.transforms.llm.tracing import AzureAITracingConfig

        _reset_azure_monitor_state()

        config = AzureAITracingConfig(
            connection_string="InstrumentationKey=test-key",
            enable_content_recording=False,
            enable_live_metrics=True,
        )

        azure_monitor = RecordingAzureMonitorConfigurator()
        instrumentor_factory = RecordingAIInferenceInstrumentorFactory()

        tracing_module = _tracing_module_with(instrumentor_factory)

        try:
            with (
                patch("elspeth.plugins.transforms.llm.providers.azure.configure_azure_monitor", new=azure_monitor),
                patch.dict(sys.modules, {"azure.ai.inference.tracing": tracing_module}),
            ):
                result = _configure_azure_monitor(config)

            assert result is True
            assert azure_monitor.calls == [
                {
                    "connection_string": "InstrumentationKey=test-key",
                    "enable_live_metrics": True,
                }
            ]
            instrumentor = _only_instrumentor(instrumentor_factory)
            assert instrumentor.instrument_calls == [{"enable_content_recording": False}]
        finally:
            _reset_azure_monitor_state()

    def test_content_recording_false_env_var(self) -> None:
        """enable_content_recording=False sets env var to 'false'."""
        import os

        from elspeth.plugins.transforms.llm.providers.azure import (
            _configure_azure_monitor,
            _reset_azure_monitor_state,
        )
        from elspeth.plugins.transforms.llm.tracing import AzureAITracingConfig

        _reset_azure_monitor_state()

        config = AzureAITracingConfig(
            connection_string="InstrumentationKey=test-key",
            enable_content_recording=False,
            enable_live_metrics=False,
        )

        env_key = "AZURE_TRACING_GEN_AI_CONTENT_RECORDING_ENABLED"
        old_value = os.environ.pop(env_key, None)
        azure_monitor = RecordingAzureMonitorConfigurator()
        try:
            with patch("elspeth.plugins.transforms.llm.providers.azure.configure_azure_monitor", new=azure_monitor):
                _configure_azure_monitor(config)

            assert azure_monitor.calls == [
                {
                    "connection_string": "InstrumentationKey=test-key",
                    "enable_live_metrics": False,
                }
            ]
            assert os.environ.get(env_key) == "false"
        finally:
            # Clean up
            _reset_azure_monitor_state()
            os.environ.pop(env_key, None)
            if old_value is not None:
                os.environ[env_key] = old_value
