# src/elspeth/testing/chaosllm/__init__.py
"""ChaosLLM: Fake LLM server for load testing and fault injection.

ChaosLLM provides:
- OpenAI and Azure OpenAI compatible endpoints
- Configurable error injection (rate limits, capacity errors, timeouts, malformed responses)
- Response generation (random, template, preset bank)
- Burst pattern simulation for AIMD throttle testing
- SQLite metrics storage for analysis

Usage:
    # CLI - Start server
    chaosllm serve --preset=stress_aimd --port=8000

    # CLI - List presets
    chaosllm presets

    # Pytest fixture
    def test_pipeline(chaosllm_server):
        response = chaosllm_server.post_completion()
        assert response.status_code == 200

    # With marker for configuration
    @pytest.mark.chaosllm(preset="stress_aimd")
    def test_under_stress(chaosllm_server):
        ...
"""

from elspeth.testing.chaosllm.config import (
    DEFAULT_MEMORY_DB,
    ChaosLLMConfig,
    ErrorInjectionConfig,
    LatencyConfig,
    ResponseConfig,
    ServerConfig,
    list_presets,
    load_config,
    load_preset,
)
from elspeth.testing.chaosllm.error_injector import ErrorDecision, ErrorInjector
from elspeth.testing.chaosllm.latency_simulator import LatencySimulator
from elspeth.testing.chaosllm.metrics import MetricsRecorder
from elspeth.testing.chaosllm.response_generator import OpenAIResponse, ResponseGenerator
from elspeth.testing.chaosllm.server import ChaosLLMServer, create_app

__all__ = [
    "DEFAULT_MEMORY_DB",
    "ChaosLLMConfig",
    "ChaosLLMServer",
    "ErrorDecision",
    "ErrorInjectionConfig",
    "ErrorInjector",
    "LatencyConfig",
    "LatencySimulator",
    "MetricsRecorder",
    "OpenAIResponse",
    "ResponseConfig",
    "ResponseGenerator",
    "ServerConfig",
    "create_app",
    "list_presets",
    "load_config",
    "load_preset",
]
