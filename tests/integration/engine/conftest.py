# tests/integration/engine/conftest.py
"""Engine-chaos integration test fixtures.

Re-exports the in-process ChaosLLM TestClient fixture (the documented usage
pattern in tests/fixtures/chaosllm.py) so test modules can take
``chaosllm_server`` as a parameter without shadowing an import.
"""

from __future__ import annotations

from tests.fixtures.chaosllm import chaosllm_server  # noqa: F401
