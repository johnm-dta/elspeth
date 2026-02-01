# tests/stress/llm/conftest.py
"""Fixtures for LLM stress tests.

Provides configuration factories and test infrastructure for stress testing
LLM transforms against ChaosLLM HTTP server.
"""

from __future__ import annotations

import uuid
from collections.abc import Generator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from elspeth.contracts.identity import TokenInfo
from elspeth.core.landscape.database import LandscapeDB
from elspeth.core.landscape.recorder import LandscapeRecorder
from elspeth.plugins.context import PluginContext

# Re-export the HTTP server fixture for convenience
from tests.stress.conftest import ChaosLLMHTTPFixture

# Dynamic schema for LLM transforms
DYNAMIC_SCHEMA = {"fields": "dynamic"}


@dataclass
class StressTestContext:
    """Context container for stress test execution.

    Holds all the resources needed for running LLM transforms
    against ChaosLLM with proper audit recording.
    """

    landscape: LandscapeRecorder
    run_id: str
    chaosllm_url: str


def make_token(row_id: str = "row-1", token_id: str | None = None) -> TokenInfo:
    """Create a TokenInfo for testing."""
    return TokenInfo(
        row_id=row_id,
        token_id=token_id or f"token-{row_id}",
        row_data={},  # Not used in these tests
    )


def make_plugin_context(
    landscape: LandscapeRecorder,
    run_id: str,
    state_id: str | None = None,
    token: TokenInfo | None = None,
) -> PluginContext:
    """Create a PluginContext with real landscape.

    Args:
        landscape: LandscapeRecorder for audit trail
        run_id: Run ID for context
        state_id: Optional state ID (generated if not provided)
        token: Optional token info (generated if not provided)

    Returns:
        PluginContext ready for use
    """
    if state_id is None:
        state_id = f"state-{uuid.uuid4().hex[:12]}"
    if token is None:
        token = make_token()

    return PluginContext(
        run_id=run_id,
        landscape=landscape,
        state_id=state_id,
        config={},
        token=token,
    )


def make_azure_llm_config(
    chaosllm_url: str,
    **overrides: Any,
) -> dict[str, Any]:
    """Create valid Azure LLM config pointed at ChaosLLM.

    Args:
        chaosllm_url: Base URL of ChaosLLM server
        **overrides: Override any config values

    Returns:
        Config dict ready for AzureLLMTransform
    """
    config = {
        "deployment_name": "gpt-4o",
        "endpoint": chaosllm_url,
        "api_key": "test-key",
        "template": "Analyze: {{ row.text }}",
        "system_prompt": "You are a helpful assistant.",
        "schema": DYNAMIC_SCHEMA,
        "pool_size": 4,
        "max_capacity_retry_seconds": 30,
        "temperature": 0.7,
        "max_tokens": 500,
        "required_input_fields": [],  # Explicit opt-out for tests
    }
    config.update(overrides)
    return config


def make_openrouter_llm_config(
    chaosllm_url: str,
    **overrides: Any,
) -> dict[str, Any]:
    """Create valid OpenRouter LLM config pointed at ChaosLLM.

    Args:
        chaosllm_url: Base URL of ChaosLLM server
        **overrides: Override any config values

    Returns:
        Config dict ready for OpenRouterLLMTransform

    Note:
        OpenRouter uses /chat/completions, but ChaosLLM serves at /v1/chat/completions.
        We append /v1 to the base_url so the paths match.
    """
    config = {
        "model": "anthropic/claude-3-opus",
        "base_url": f"{chaosllm_url}/v1",  # Append /v1 for ChaosLLM compatibility
        "api_key": "test-key",
        "template": "Analyze: {{ row.text }}",
        "system_prompt": "You are a helpful assistant.",
        "schema": DYNAMIC_SCHEMA,
        "pool_size": 4,
        "max_capacity_retry_seconds": 30,
        "temperature": 0.7,
        "max_tokens": 500,
        "required_input_fields": [],  # Explicit opt-out for tests
    }
    config.update(overrides)
    return config


def make_azure_multi_query_config(
    chaosllm_url: str,
    **overrides: Any,
) -> dict[str, Any]:
    """Create valid Azure multi-query config pointed at ChaosLLM.

    Args:
        chaosllm_url: Base URL of ChaosLLM server
        **overrides: Override any config values

    Returns:
        Config dict ready for AzureMultiQueryLLMTransform
    """
    config = {
        "deployment_name": "gpt-4o",
        "endpoint": chaosllm_url,
        "api_key": "test-key",
        "template": "Input: {{ row.input_1 }}\nCriterion: {{ row.criterion.name }}",
        "system_prompt": "You are an assessment AI. Respond in JSON.",
        "case_studies": [
            {"name": "cs1", "input_fields": ["cs1_bg", "cs1_sym", "cs1_hist"]},
            {"name": "cs2", "input_fields": ["cs2_bg", "cs2_sym", "cs2_hist"]},
        ],
        "criteria": [
            {"name": "diagnosis", "code": "DIAG"},
            {"name": "treatment", "code": "TREAT"},
        ],
        "response_format": "standard",
        "output_mapping": {
            "score": {"suffix": "score", "type": "integer"},
            "rationale": {"suffix": "rationale", "type": "string"},
        },
        "schema": DYNAMIC_SCHEMA,
        "required_input_fields": [],  # Explicit opt-out for tests
        "pool_size": 4,
        "max_capacity_retry_seconds": 30,
    }
    config.update(overrides)
    return config


def make_openrouter_multi_query_config(
    chaosllm_url: str,
    **overrides: Any,
) -> dict[str, Any]:
    """Create valid OpenRouter multi-query config pointed at ChaosLLM.

    Args:
        chaosllm_url: Base URL of ChaosLLM server
        **overrides: Override any config values

    Returns:
        Config dict ready for OpenRouterMultiQueryLLMTransform

    Note:
        OpenRouter uses /chat/completions, but ChaosLLM serves at /v1/chat/completions.
        We append /v1 to the base_url so the paths match.
    """
    config = {
        "model": "anthropic/claude-3-opus",
        "base_url": f"{chaosllm_url}/v1",  # Append /v1 for ChaosLLM compatibility
        "api_key": "test-key",
        "template": "Input: {{ row.input_1 }}\nCriterion: {{ row.criterion.name }}",
        "system_prompt": "You are an assessment AI. Respond in JSON.",
        "case_studies": [
            {"name": "cs1", "input_fields": ["cs1_bg", "cs1_sym", "cs1_hist"]},
            {"name": "cs2", "input_fields": ["cs2_bg", "cs2_sym", "cs2_hist"]},
        ],
        "criteria": [
            {"name": "diagnosis", "code": "DIAG"},
            {"name": "treatment", "code": "TREAT"},
        ],
        "response_format": "standard",
        "output_mapping": {
            "score": {"suffix": "score", "type": "integer"},
            "rationale": {"suffix": "rationale", "type": "string"},
        },
        "schema": DYNAMIC_SCHEMA,
        "required_input_fields": [],  # Explicit opt-out for tests
        "pool_size": 4,
        "max_capacity_retry_seconds": 30,
    }
    config.update(overrides)
    return config


@pytest.fixture
def stress_landscape_db(tmp_path: Path) -> Generator[LandscapeRecorder, None, None]:
    """Create a fresh LandscapeRecorder for stress testing.

    Uses SQLite file database for persistence during test.

    Yields:
        LandscapeRecorder ready for use
    """
    db_path = tmp_path / "stress-audit.db"
    db = LandscapeDB(f"sqlite:///{db_path}")
    recorder = LandscapeRecorder(db)

    yield recorder

    # Cleanup handled by tmp_path fixture


@pytest.fixture
def stress_test_context(
    chaosllm_http_server: ChaosLLMHTTPFixture,
    stress_landscape_db: LandscapeRecorder,
) -> Generator[StressTestContext, None, None]:
    """Create a complete stress test context.

    Combines ChaosLLM server with landscape recorder and begins a run.

    Yields:
        StressTestContext with all resources ready
    """
    # Begin a run
    run = stress_landscape_db.begin_run(
        config={"test": "stress"},
        run_id=f"stress-run-{uuid.uuid4().hex[:8]}",
        canonical_version="v1",
    )

    yield StressTestContext(
        landscape=stress_landscape_db,
        run_id=run.run_id,
        chaosllm_url=chaosllm_http_server.url,
    )

    # Complete the run
    from elspeth.contracts import RunStatus

    stress_landscape_db.complete_run(run.run_id, status=RunStatus.COMPLETED)


def generate_test_rows(count: int, prefix: str = "row") -> list[dict[str, Any]]:
    """Generate test rows for stress testing.

    Args:
        count: Number of rows to generate
        prefix: Prefix for row IDs

    Returns:
        List of row dicts with text and id fields
    """
    return [{"id": f"{prefix}-{i}", "text": f"Test input for row {i}"} for i in range(count)]


def generate_multi_query_rows(count: int, prefix: str = "row") -> list[dict[str, Any]]:
    """Generate test rows for multi-query stress testing.

    Creates rows with all case study input fields.

    Args:
        count: Number of rows to generate
        prefix: Prefix for row IDs

    Returns:
        List of row dicts with case study fields
    """
    return [
        {
            "id": f"{prefix}-{i}",
            # Case study 1 fields
            "cs1_bg": f"Background info for case study 1, row {i}",
            "cs1_sym": f"Symptoms for case study 1, row {i}",
            "cs1_hist": f"History for case study 1, row {i}",
            # Case study 2 fields
            "cs2_bg": f"Background info for case study 2, row {i}",
            "cs2_sym": f"Symptoms for case study 2, row {i}",
            "cs2_hist": f"History for case study 2, row {i}",
        }
        for i in range(count)
    ]


@dataclass
class StressTestResult:
    """Results from a stress test run.

    Attributes:
        total_rows: Total number of rows processed
        successful_rows: Rows that completed successfully
        failed_rows: Rows that failed
        error_rate_observed: Actual error rate from ChaosLLM
        total_requests: Total requests made to ChaosLLM
        fifo_preserved: Whether output order matched input order
    """

    total_rows: int
    successful_rows: int
    failed_rows: int
    error_rate_observed: float
    total_requests: int
    fifo_preserved: bool

    @property
    def success_rate(self) -> float:
        """Calculate row success rate."""
        if self.total_rows == 0:
            return 0.0
        return self.successful_rows / self.total_rows


def verify_audit_integrity(
    landscape: LandscapeRecorder,
    run_id: str,
    expected_rows: int,
) -> bool:
    """Verify that audit trail is complete for all rows.

    Checks that every row has:
    - A token record
    - Node state(s)
    - Either completed or failed outcome

    Args:
        landscape: LandscapeRecorder with test data
        run_id: Run ID to verify
        expected_rows: Expected number of source rows

    Returns:
        True if audit trail is complete, False otherwise
    """

    # Get all tokens for this run
    with landscape._db.session() as session:
        from sqlalchemy import select

        from elspeth.core.landscape.schema import tokens_table

        result = session.execute(select(tokens_table).where(tokens_table.c.run_id == run_id))
        tokens = list(result.fetchall())

    # Every row should have at least one token
    unique_row_ids = {t.row_id for t in tokens}

    if len(unique_row_ids) < expected_rows:
        return False

    # Get outcomes
    with landscape._db.session() as session:
        from elspeth.core.landscape.schema import token_outcomes_table

        result = session.execute(select(token_outcomes_table).where(token_outcomes_table.c.run_id == run_id))
        outcomes = list(result.fetchall())

    # Every token should have an outcome
    token_ids_with_outcome = {o.token_id for o in outcomes}
    all_token_ids = {t.token_id for t in tokens}

    return all_token_ids.issubset(token_ids_with_outcome)
