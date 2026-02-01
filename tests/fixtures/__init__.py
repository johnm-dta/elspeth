# tests/fixtures/__init__.py
"""Shared pytest fixtures for ELSPETH tests.

Available fixtures:
- chaosllm_server: ChaosLLM fake LLM server for testing
"""

from tests.fixtures.chaosllm import ChaosLLMFixture, chaosllm_server

__all__ = [
    "ChaosLLMFixture",
    "chaosllm_server",
]
