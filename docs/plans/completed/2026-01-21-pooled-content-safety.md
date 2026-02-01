# Pooled Execution for Azure Content Safety Transform

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add pooled concurrent execution to Azure Content Safety transform, enabling parallel API calls with AIMD throttling to eliminate sequential bottlenecks in content filtering pipelines.

**Architecture:** Use the shared `PooledExecutor` and `PoolConfig` from `plugins/pooling/` (extracted in Prompt Shield plan). Follow same pattern as Prompt Shield pooled execution.

**Tech Stack:** Existing PooledExecutor from plugins/pooling/ (must complete Prompt Shield plan Task 1 first)

**Prerequisite:** Execute `2026-01-21-pooled-prompt-shield.md` Task 1 first to extract pooling infrastructure.

---

## Overview

This plan adds pooling to Content Safety, completing the pipeline optimization:

| Transform | Before | After (pool_size=5) |
|-----------|--------|---------------------|
| keyword_filter | instant | instant |
| content_safety | 20s (sequential) | ~4s (pooled) |
| prompt_shield | 20s (sequential) | ~4s (pooled) |
| azure_llm | ~2s (pooled) | ~2s (pooled) |
| **Total** | **~42s** | **~10s** |

---

## Task 1: Add Pool Config Fields to Content Safety Config

**Files:**
- Modify: `src/elspeth/plugins/transforms/azure/content_safety.py`
- Modify: `tests/plugins/transforms/azure/test_content_safety.py`

**Step 1: Write failing tests for pool_size config**

Add to `tests/plugins/transforms/azure/test_content_safety.py`:

```python
class TestContentSafetyPoolConfig:
    """Tests for Content Safety pool configuration."""

    def test_pool_size_default_is_one(self) -> None:
        """Default pool_size is 1 (sequential)."""
        from elspeth.plugins.transforms.azure.content_safety import (
            AzureContentSafetyConfig,
        )

        cfg = AzureContentSafetyConfig.from_dict(
            {
                "endpoint": "https://test.cognitiveservices.azure.com",
                "api_key": "test-key",
                "fields": ["content"],
                "thresholds": {"hate": 2, "violence": 2, "sexual": 2, "self_harm": 0},
                "schema": {"fields": "dynamic"},
            }
        )

        assert cfg.pool_size == 1

    def test_pool_size_configurable(self) -> None:
        """pool_size can be configured."""
        from elspeth.plugins.transforms.azure.content_safety import (
            AzureContentSafetyConfig,
        )

        cfg = AzureContentSafetyConfig.from_dict(
            {
                "endpoint": "https://test.cognitiveservices.azure.com",
                "api_key": "test-key",
                "fields": ["content"],
                "thresholds": {"hate": 2, "violence": 2, "sexual": 2, "self_harm": 0},
                "schema": {"fields": "dynamic"},
                "pool_size": 5,
            }
        )

        assert cfg.pool_size == 5

    def test_pool_config_property_returns_none_when_sequential(self) -> None:
        """pool_config returns None when pool_size=1."""
        from elspeth.plugins.transforms.azure.content_safety import (
            AzureContentSafetyConfig,
        )

        cfg = AzureContentSafetyConfig.from_dict(
            {
                "endpoint": "https://test.cognitiveservices.azure.com",
                "api_key": "test-key",
                "fields": ["content"],
                "thresholds": {"hate": 2, "violence": 2, "sexual": 2, "self_harm": 0},
                "schema": {"fields": "dynamic"},
                "pool_size": 1,
            }
        )

        assert cfg.pool_config is None

    def test_pool_config_property_returns_config_when_pooled(self) -> None:
        """pool_config returns PoolConfig when pool_size>1."""
        from elspeth.plugins.transforms.azure.content_safety import (
            AzureContentSafetyConfig,
        )

        cfg = AzureContentSafetyConfig.from_dict(
            {
                "endpoint": "https://test.cognitiveservices.azure.com",
                "api_key": "test-key",
                "fields": ["content"],
                "thresholds": {"hate": 2, "violence": 2, "sexual": 2, "self_harm": 0},
                "schema": {"fields": "dynamic"},
                "pool_size": 3,
            }
        )

        assert cfg.pool_config is not None
        assert cfg.pool_config.pool_size == 3
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/plugins/transforms/azure/test_content_safety.py::TestContentSafetyPoolConfig -v`
Expected: FAIL - pool_size field doesn't exist

**Step 3: Add pool config fields to AzureContentSafetyConfig**

In `src/elspeth/plugins/transforms/azure/content_safety.py`, update imports and config class:

```python
from elspeth.plugins.pooling import PoolConfig  # Add this import


class AzureContentSafetyConfig(TransformDataConfig):
    """Configuration for Azure Content Safety transform.

    Requires:
        endpoint: Azure Content Safety endpoint URL
        api_key: Azure Content Safety API key
        fields: Field name(s) to analyze, or 'all' for all string fields
        thresholds: Per-category severity thresholds (0-6)
        schema: Schema configuration

    Optional:
        pool_size: Number of concurrent API calls (1=sequential, >1=pooled)
        max_dispatch_delay_ms: Maximum AIMD backoff delay (default 5000)
        max_capacity_retry_seconds: Timeout for capacity error retries (default 3600)
    """

    endpoint: str = Field(..., description="Azure Content Safety endpoint URL")
    api_key: str = Field(..., description="Azure Content Safety API key")
    fields: str | list[str] = Field(
        ...,
        description="Field name(s) to analyze, or 'all' for all string fields",
    )
    thresholds: ContentSafetyThresholds = Field(
        ...,
        description="Per-category severity thresholds (0-6)",
    )

    # Pool configuration fields
    pool_size: int = Field(1, ge=1, description="Number of concurrent API calls (1=sequential)")
    min_dispatch_delay_ms: int = Field(0, ge=0, description="Minimum dispatch delay in milliseconds")
    max_dispatch_delay_ms: int = Field(5000, ge=0, description="Maximum dispatch delay in milliseconds")
    backoff_multiplier: float = Field(2.0, gt=1.0, description="Backoff multiplier on capacity error")
    recovery_step_ms: int = Field(50, ge=0, description="Recovery step in milliseconds")
    max_capacity_retry_seconds: int = Field(3600, gt=0, description="Max seconds to retry capacity errors")

    @property
    def pool_config(self) -> PoolConfig | None:
        """Get pool configuration if pooling is enabled.

        Returns None if pool_size <= 1 (sequential mode).
        """
        if self.pool_size <= 1:
            return None
        return PoolConfig(
            pool_size=self.pool_size,
            min_dispatch_delay_ms=self.min_dispatch_delay_ms,
            max_dispatch_delay_ms=self.max_dispatch_delay_ms,
            backoff_multiplier=self.backoff_multiplier,
            recovery_step_ms=self.recovery_step_ms,
            max_capacity_retry_seconds=self.max_capacity_retry_seconds,
        )
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/plugins/transforms/azure/test_content_safety.py::TestContentSafetyPoolConfig -v`
Expected: All tests pass

**Step 5: Commit**

```bash
git add src/elspeth/plugins/transforms/azure/content_safety.py tests/plugins/transforms/azure/test_content_safety.py
git commit -m "$(cat <<'EOF'
feat(content-safety): add pool configuration fields

Add pool_size and AIMD throttle config options to Content Safety.
pool_size=1 (default) maintains sequential behavior.

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>
EOF
)"
```

---

## Task 2: Implement Pooled Execution in Content Safety

**Files:**
- Modify: `src/elspeth/plugins/transforms/azure/content_safety.py`
- Modify: `tests/plugins/transforms/azure/test_content_safety.py`

> **Pre-existing Test Failures:** The existing tests in `test_content_safety.py` have 12 failures because they mock `ctx.http_client.post`, but the implementation uses `self._http_client` (its own httpx.Client instance). These tests need to be fixed as part of this task by updating the mock pattern.

**Step 0: Fix existing test helper to use correct mock pattern**

The existing tests mock the wrong interface. Update the test pattern to properly mock httpx.Client:

```python
from typing import Any
from unittest.mock import MagicMock

from elspeth.plugins.transforms.azure.content_safety import AzureContentSafety


def make_content_safety_with_mock_response(
    config: dict[str, Any],
    response_data: dict[str, Any],
) -> tuple[AzureContentSafety, MagicMock]:
    """Create Content Safety transform with mocked HTTP client.

    Returns the transform and the mock client for assertions.
    """
    transform = AzureContentSafety(config)

    # Create mock response
    response_mock = MagicMock()
    response_mock.status_code = 200
    response_mock.json.return_value = response_data
    response_mock.raise_for_status = MagicMock()

    # Create mock client
    mock_client = MagicMock()
    mock_client.post.return_value = response_mock

    # Inject mock client directly (bypassing _get_http_client)
    transform._http_client = mock_client

    return transform, mock_client
```

Update existing tests to use this new helper pattern. Run existing tests after fixing: `pytest tests/plugins/transforms/azure/test_content_safety.py -v`
Expected: All 12 previously failing tests now pass.

**Step 1: Write failing tests for pooled batch execution**

Add to test file:

```python
class TestContentSafetyPooledExecution:
    """Tests for Content Safety pooled execution."""

    def test_batch_aware_is_true_when_pooled(self) -> None:
        """Transform is batch_aware when pool_size > 1."""
        from elspeth.plugins.transforms.azure.content_safety import AzureContentSafety

        transform = AzureContentSafety(
            {
                "endpoint": "https://test.cognitiveservices.azure.com",
                "api_key": "test-key",
                "fields": ["content"],
                "thresholds": {"hate": 2, "violence": 2, "sexual": 2, "self_harm": 0},
                "schema": {"fields": "dynamic"},
                "pool_size": 3,
            }
        )

        assert transform.is_batch_aware is True

    def test_batch_aware_is_false_when_sequential(self) -> None:
        """Transform is not batch_aware when pool_size=1."""
        from elspeth.plugins.transforms.azure.content_safety import AzureContentSafety

        transform = AzureContentSafety(
            {
                "endpoint": "https://test.cognitiveservices.azure.com",
                "api_key": "test-key",
                "fields": ["content"],
                "thresholds": {"hate": 2, "violence": 2, "sexual": 2, "self_harm": 0},
                "schema": {"fields": "dynamic"},
                "pool_size": 1,
            }
        )

        assert transform.is_batch_aware is False

    def test_pooled_execution_processes_batch_concurrently(self) -> None:
        """Pooled transform processes batch rows concurrently."""
        from unittest.mock import MagicMock, patch

        from elspeth.plugins.transforms.azure.content_safety import AzureContentSafety

        transform = AzureContentSafety(
            {
                "endpoint": "https://test.cognitiveservices.azure.com",
                "api_key": "test-key",
                "fields": ["content"],
                "thresholds": {"hate": 2, "violence": 2, "sexual": 2, "self_harm": 0},
                "schema": {"fields": "dynamic"},
                "pool_size": 3,
            }
        )

        ctx = MagicMock()
        ctx.landscape = MagicMock()
        ctx.state_id = "test-state"
        ctx.run_id = "test-run"

        # Mock API response - all content safe
        response_mock = MagicMock()
        response_mock.status_code = 200
        response_mock.json.return_value = {
            "categoriesAnalysis": [
                {"category": "Hate", "severity": 0},
                {"category": "Violence", "severity": 0},
                {"category": "Sexual", "severity": 0},
                {"category": "SelfHarm", "severity": 0},
            ]
        }
        response_mock.raise_for_status = MagicMock()

        with patch("httpx.Client") as mock_client_class:
            mock_client = MagicMock()
            mock_client.post.return_value = response_mock
            mock_client_class.return_value = mock_client

            rows = [
                {"content": "row 1", "id": 1},
                {"content": "row 2", "id": 2},
                {"content": "row 3", "id": 3},
            ]
            result = transform.process(rows, ctx)

        assert result.status == "success"
        assert result.rows is not None
        assert len(result.rows) == 3

    def test_pooled_execution_handles_threshold_violations(self) -> None:
        """Pooled execution correctly handles content violations."""
        from unittest.mock import MagicMock, patch

        from elspeth.plugins.transforms.azure.content_safety import AzureContentSafety

        transform = AzureContentSafety(
            {
                "endpoint": "https://test.cognitiveservices.azure.com",
                "api_key": "test-key",
                "fields": ["content"],
                "thresholds": {"hate": 2, "violence": 2, "sexual": 2, "self_harm": 0},
                "schema": {"fields": "dynamic"},
                "pool_size": 3,
            }
        )

        ctx = MagicMock()
        ctx.landscape = MagicMock()
        ctx.state_id = "test-state"
        ctx.run_id = "test-run"

        call_count = [0]

        def mock_post(*args, **kwargs):
            call_count[0] += 1
            response = MagicMock()
            response.status_code = 200
            response.raise_for_status = MagicMock()

            # Second row has violence violation
            if call_count[0] == 2:
                response.json.return_value = {
                    "categoriesAnalysis": [
                        {"category": "Hate", "severity": 0},
                        {"category": "Violence", "severity": 4},  # Exceeds threshold of 2
                        {"category": "Sexual", "severity": 0},
                        {"category": "SelfHarm", "severity": 0},
                    ]
                }
            else:
                response.json.return_value = {
                    "categoriesAnalysis": [
                        {"category": "Hate", "severity": 0},
                        {"category": "Violence", "severity": 0},
                        {"category": "Sexual", "severity": 0},
                        {"category": "SelfHarm", "severity": 0},
                    ]
                }
            return response

        with patch("httpx.Client") as mock_client_class:
            mock_client = MagicMock()
            mock_client.post.side_effect = mock_post
            mock_client_class.return_value = mock_client

            rows = [
                {"content": "safe 1", "id": 1},
                {"content": "violent content", "id": 2},
                {"content": "safe 2", "id": 3},
            ]
            result = transform.process(rows, ctx)

        # Should return ALL rows with per-row error tracking
        assert result.status == "success"
        assert result.rows is not None
        assert len(result.rows) == 3  # ALL rows included, not just successes

        # Row 1: success (no error marker)
        assert result.rows[0].get("_content_safety_error") is None
        assert result.rows[0]["id"] == 1

        # Row 2: error (violence violation) - has error marker
        assert result.rows[1].get("_content_safety_error") is not None
        assert result.rows[1]["_content_safety_error"]["reason"] == "content_safety_violation"
        assert result.rows[1]["id"] == 2

        # Row 3: success (no error marker)
        assert result.rows[2].get("_content_safety_error") is None
        assert result.rows[2]["id"] == 3

    def test_pooled_rate_limit_triggers_capacity_error(self) -> None:
        """Rate limit (429) triggers CapacityError for AIMD retry."""
        from unittest.mock import MagicMock

        import httpx
        import pytest

        from elspeth.plugins.pooling import CapacityError
        from elspeth.plugins.transforms.azure.content_safety import AzureContentSafety

        transform = AzureContentSafety(
            {
                "endpoint": "https://test.cognitiveservices.azure.com",
                "api_key": "test-key",
                "fields": ["content"],
                "thresholds": {"hate": 2, "violence": 2, "sexual": 2, "self_harm": 0},
                "schema": {"fields": "dynamic"},
                "pool_size": 3,
            }
        )

        # Inject mock client that returns 429
        mock_response = MagicMock()
        mock_response.status_code = 429
        mock_client = MagicMock()
        mock_client.post.side_effect = httpx.HTTPStatusError(
            "Rate limited",
            request=MagicMock(),
            response=mock_response,
        )

        # Inject mock client directly
        transform._http_client = mock_client

        with pytest.raises(CapacityError) as exc_info:
            transform._process_single_with_state({"content": "test", "id": 1}, "test-state")

        assert exc_info.value.status_code == 429
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/plugins/transforms/azure/test_content_safety.py::TestContentSafetyPooledExecution -v`
Expected: FAIL - batch processing not implemented

**Step 3: Implement pooled execution**

Update `src/elspeth/plugins/transforms/azure/content_safety.py`:

```python
"""Azure Content Safety transform with optional pooled execution."""

from __future__ import annotations

import time
from threading import Lock
from typing import TYPE_CHECKING, Any

import httpx
from pydantic import BaseModel, Field

from elspeth.contracts import CallStatus, CallType, Determinism
from elspeth.plugins.base import BaseTransform
from elspeth.plugins.config_base import TransformDataConfig
from elspeth.plugins.context import PluginContext
from elspeth.plugins.pooling import CapacityError, PoolConfig, PooledExecutor, RowContext
from elspeth.plugins.results import TransformResult
from elspeth.plugins.schema_factory import create_schema_from_config

if TYPE_CHECKING:
    from elspeth.core.landscape.recorder import LandscapeRecorder


class ContentSafetyThresholds(BaseModel):
    """Per-category severity thresholds for Azure Content Safety."""
    # ... (keep existing)


class AzureContentSafetyConfig(TransformDataConfig):
    """Configuration for Azure Content Safety transform."""
    # ... (as defined in Task 1)


class AzureContentSafety(BaseTransform):
    """Analyze content using Azure Content Safety API with optional pooling.

    Design Notes:
    - is_batch_aware is DYNAMIC based on pool_size (True when pool_size > 1)
    - Single shared HTTP client (httpx.Client is stateless)
    - All API calls recorded to audit trail via CallType.HTTP
    """

    name = "azure_content_safety"
    determinism = Determinism.EXTERNAL_CALL
    plugin_version = "1.0.0"
    creates_tokens = False

    API_VERSION = "2024-09-01"

    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__(config)

        cfg = AzureContentSafetyConfig.from_dict(config)
        self._endpoint = cfg.endpoint.rstrip("/")
        self._api_key = cfg.api_key
        self._fields = cfg.fields
        self._thresholds = cfg.thresholds
        self._on_error = cfg.on_error

        assert cfg.schema_config is not None
        schema = create_schema_from_config(
            cfg.schema_config,
            "AzureContentSafetySchema",
            allow_coercion=False,
        )
        self.input_schema = schema
        self.output_schema = schema

        # Pooling setup
        self._pool_config = cfg.pool_config
        if self._pool_config is not None:
            self._executor: PooledExecutor | None = PooledExecutor(self._pool_config)
            self._is_batch_aware = True
        else:
            self._executor = None
            self._is_batch_aware = False

        # Single shared HTTP client (thread-safe, stateless)
        self._http_client: httpx.Client | None = None

        # Recorder for audit trail (captured in on_start)
        self._recorder: LandscapeRecorder | None = None

        # Call index counter for audit trail (thread-safe)
        self._call_index = 0
        self._call_index_lock = Lock()

    def _next_call_index(self) -> int:
        """Get next call index for audit trail (thread-safe)."""
        with self._call_index_lock:
            idx = self._call_index
            self._call_index += 1
            return idx

    @property
    def is_batch_aware(self) -> bool:
        """Dynamic batch_aware based on pool_size."""
        return self._is_batch_aware

    def on_start(self, ctx: PluginContext) -> None:
        """Capture recorder for audit trail recording."""
        self._recorder = ctx.landscape

    def _get_http_client(self) -> httpx.Client:
        """Get or create shared HTTP client (thread-safe, stateless)."""
        if self._http_client is None:
            self._http_client = httpx.Client(timeout=30.0)
        return self._http_client

    def process(
        self,
        row: dict[str, Any] | list[dict[str, Any]],
        ctx: PluginContext,
    ) -> TransformResult:
        """Process row(s) through Content Safety.

        Routes to batch processing if given a list (batch_aware mode).
        """
        # Dispatch to batch processing if given a list
        if isinstance(row, list):
            return self._process_batch(row, ctx)

        # Pooled single-row execution
        if self._executor is not None:
            if ctx.state_id is None:
                raise RuntimeError("Pooled execution requires state_id")
            row_ctx = RowContext(row=row, state_id=ctx.state_id, row_index=0)
            results = self._executor.execute_batch(
                contexts=[row_ctx],
                process_fn=self._process_single_with_state,
            )
            return results[0]

        # Sequential execution (original path)
        return self._process_single_with_state(row, ctx.state_id or "unknown")

    def _process_batch(
        self,
        rows: list[dict[str, Any]],
        ctx: PluginContext,
    ) -> TransformResult:
        """Process batch with pooled execution."""
        if not rows:
            return TransformResult.success({"batch_empty": True, "row_count": 0})

        if ctx.state_id is None:
            raise RuntimeError("Batch processing requires state_id")

        if self._executor is None:
            # Sequential fallback - process one at a time
            results = [
                self._process_single_with_state(row, ctx.state_id)
                for row in rows
            ]
            return self._assemble_batch_results(rows, results)

        # Pooled execution
        contexts = [
            RowContext(row=row, state_id=ctx.state_id, row_index=i)
            for i, row in enumerate(rows)
        ]

        results = self._executor.execute_batch(
            contexts=contexts,
            process_fn=self._process_single_with_state,
        )

        return self._assemble_batch_results(rows, results)

    def _assemble_batch_results(
        self,
        rows: list[dict[str, Any]],
        results: list[TransformResult],
    ) -> TransformResult:
        """Assemble batch results with per-row error tracking.

        Follows AzureLLMTransform pattern:
        - Include ALL rows in output (success and failures)
        - Embed errors per-row via _content_safety_error field
        - Only return TransformResult.error if ALL rows failed
        """
        output_rows: list[dict[str, Any]] = []
        all_failed = True

        for i, (row, result) in enumerate(zip(rows, results, strict=True)):
            output_row = dict(row)

            if result.status == "success" and result.row is not None:
                all_failed = False
                # Copy through the successfully processed row
                output_row = result.row
            else:
                # Mark as failed but include in output for audit trail
                output_row["_content_safety_error"] = result.reason or {
                    "reason": "unknown_error",
                    "row_index": i,
                }

            output_rows.append(output_row)

        # Only return error if ALL rows failed
        if all_failed and output_rows:
            return TransformResult.error(
                {"reason": "all_rows_failed", "row_count": len(rows)}
            )

        return TransformResult.success_multi(output_rows)

    def _process_single_with_state(
        self,
        row: dict[str, Any],
        state_id: str,
    ) -> TransformResult:
        """Process single row through Content Safety API.

        This is the unified processing method used by both sequential and pooled
        execution paths. All external API calls are recorded to the audit trail.

        Raises:
            CapacityError: On rate limits (429) for AIMD retry in pooled mode
        """
        fields_to_scan = self._get_fields_to_scan(row)

        for field_name in fields_to_scan:
            if field_name not in row:
                continue

            value = row[field_name]
            if not isinstance(value, str):
                continue

            try:
                analysis = self._analyze_content(value, state_id)
            except httpx.HTTPStatusError as e:
                if e.response.status_code == 429:
                    # Raise CapacityError for AIMD retry in pooled execution
                    raise CapacityError(429, str(e)) from e
                return TransformResult.error(
                    {
                        "reason": "api_error",
                        "error_type": "http_error",
                        "status_code": e.response.status_code,
                        "message": str(e),
                        "retryable": False,
                    }
                )
            except httpx.RequestError as e:
                return TransformResult.error(
                    {
                        "reason": "api_error",
                        "error_type": "network_error",
                        "message": str(e),
                        "retryable": True,
                    },
                    retryable=True,
                )

            violation = self._check_thresholds(analysis)
            if violation:
                return TransformResult.error(
                    {
                        "reason": "content_safety_violation",
                        "field": field_name,
                        "categories": violation,
                        "retryable": False,
                    }
                )

        return TransformResult.success(row)

    def _analyze_content(
        self,
        text: str,
        state_id: str,
    ) -> dict[str, int]:
        """Call Azure Content Safety API with audit trail recording.

        All API calls are recorded to the Landscape for full auditability.
        This ensures explain() queries can trace content safety decisions.

        Args:
            text: Text content to analyze
            state_id: State ID for audit trail association

        Returns:
            Dict mapping category names to severity scores

        Raises:
            httpx.HTTPStatusError: On HTTP errors (including rate limits)
            httpx.RequestError: On network errors or malformed responses
        """
        client = self._get_http_client()
        url = f"{self._endpoint}/contentsafety/text:analyze?api-version={self.API_VERSION}"

        # Build request data for audit trail
        request_data = {
            "text": text,
            "endpoint": self._endpoint,
            "api_version": self.API_VERSION,
        }

        call_index = self._next_call_index()
        start = time.perf_counter()

        try:
            response = client.post(
                url,
                json={"text": text},
                headers={
                    "Ocp-Apim-Subscription-Key": self._api_key,
                    "Content-Type": "application/json",
                },
            )
            response.raise_for_status()
            latency_ms = (time.perf_counter() - start) * 1000

            # Parse response - fail closed on malformed response (security transform)
            try:
                data = response.json()
                result: dict[str, int] = {}
                for item in data["categoriesAnalysis"]:
                    category = item["category"].lower().replace("selfharm", "self_harm")
                    result[category] = item["severity"]
            except (KeyError, TypeError, ValueError) as e:
                raise httpx.RequestError(f"Malformed Content Safety response: {e}") from e

            # Record successful call to audit trail
            if self._recorder is not None:
                self._recorder.record_call(
                    state_id=state_id,
                    call_index=call_index,
                    call_type=CallType.HTTP,
                    status=CallStatus.SUCCESS,
                    request_data=request_data,
                    response_data=result,
                    latency_ms=latency_ms,
                )

            return result

        except Exception as e:
            latency_ms = (time.perf_counter() - start) * 1000

            # Record error to audit trail
            if self._recorder is not None:
                self._recorder.record_call(
                    state_id=state_id,
                    call_index=call_index,
                    call_type=CallType.HTTP,
                    status=CallStatus.ERROR,
                    request_data=request_data,
                    error={
                        "type": type(e).__name__,
                        "message": str(e),
                    },
                    latency_ms=latency_ms,
                )

            raise

    # Keep existing: _get_fields_to_scan, _check_thresholds

    def close(self) -> None:
        """Release resources."""
        if self._http_client is not None:
            self._http_client.close()
            self._http_client = None
        if self._executor is not None:
            self._executor.shutdown(wait=True)
```

**Step 4: Run tests to verify they pass**

Run: `pytest tests/plugins/transforms/azure/test_content_safety.py -v`
Expected: All tests pass

**Step 5: Run type checker**

Run: `mypy src/elspeth/plugins/transforms/azure/content_safety.py`
Expected: Success

**Step 6: Commit implementation**

```bash
git add src/elspeth/plugins/transforms/azure/content_safety.py tests/plugins/transforms/azure/test_content_safety.py
git commit -m "$(cat <<'EOF'
feat(content-safety): add pooled execution support

Content Safety now supports concurrent API calls with pool_size>1.
Uses same PooledExecutor infrastructure as Prompt Shield and LLM transforms.

Features:
- pool_size config option (default 1 = sequential)
- AIMD throttle for adaptive rate limiting
- Batch processing for aggregation compatibility
- Per-row error tracking in batch results

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>
EOF
)"
```

---

## Task 3: Update Example Configuration

**Files:**
- Modify: `examples/azure_blob_sentiment/settings.yaml`
- Modify: `examples/azure_blob_sentiment/README.md`

**Step 1: Add pool_size to Content Safety in settings.yaml**

```yaml
  # Step 2: Check content safety before sending to LLM
  - plugin: azure_content_safety
    options:
      endpoint: "${AZURE_CONTENT_SAFETY_ENDPOINT}"
      api_key: "${AZURE_CONTENT_SAFETY_KEY}"
      fields: text
      thresholds:
        hate: 2
        violence: 2
        sexual: 2
        self_harm: 0
      on_error: flagged
      schema:
        fields: dynamic
      pool_size: 5  # Process 5 rows concurrently
```

**Step 2: Update README pooled safety section**

Update the pooled safety transforms documentation:

```markdown
### Pooled Safety Transforms

Both Content Safety and Prompt Shield support pooled execution:

```yaml
# Content Safety with pooling
- plugin: azure_content_safety
  options:
    endpoint: "${AZURE_CONTENT_SAFETY_ENDPOINT}"
    api_key: "${AZURE_CONTENT_SAFETY_KEY}"
    fields: text
    thresholds:
      hate: 2
      violence: 2
      sexual: 2
      self_harm: 0
    pool_size: 5  # Process 5 rows concurrently

# Prompt Shield with pooling
- plugin: azure_prompt_shield
  options:
    endpoint: "${AZURE_CONTENT_SAFETY_ENDPOINT}"
    api_key: "${AZURE_CONTENT_SAFETY_KEY}"
    fields: text
    pool_size: 5  # Process 5 rows concurrently
```

**Performance impact** (100 rows at 200ms/call):

| Transform | Sequential | Pooled (pool_size=5) |
|-----------|------------|----------------------|
| content_safety | 20s | ~4s |
| prompt_shield | 20s | ~4s |
| **Total safety checks** | **40s** | **~8s** |

Pooled transforms use AIMD (Additive Increase, Multiplicative Decrease) throttling
to automatically back off on rate limits (HTTP 429) and gradually increase
throughput as capacity allows.
```

**Step 3: Commit**

```bash
git add examples/azure_blob_sentiment/
git commit -m "$(cat <<'EOF'
docs: add pool_size to Content Safety example config

Demonstrates concurrent Content Safety and Prompt Shield execution.
Includes performance comparison table.

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>
EOF
)"
```

---

## Task 4: Run Full Test Suite

**Files:** None (verification only)

**Step 1: Run linter**

Run: `ruff check src/elspeth/plugins/transforms/azure/`
Expected: No errors

**Step 2: Run type checker**

Run: `mypy src/elspeth/plugins/transforms/azure/`
Expected: Success

**Step 3: Run all Azure transform tests**

Run: `pytest tests/plugins/transforms/azure/ -v`
Expected: All tests pass

**Step 4: Run full test suite**

Run: `pytest tests/ -x --ignore=tests/integration`
Expected: All tests pass

---

## Summary

| Task | Description | Files |
|------|-------------|-------|
| 1 | Add pool config to Content Safety | `content_safety.py`, tests |
| 2 | Implement pooled execution | `content_safety.py`, tests |
| 3 | Update example config | `settings.yaml`, `README.md` |
| 4 | Full verification | (none) |

**Prerequisite:** Complete Prompt Shield plan Task 1 (extract pooling infrastructure) first.

**New Configuration Options:**
```yaml
- plugin: azure_content_safety
  options:
    pool_size: 5                      # Concurrent workers (default: 1)
    max_dispatch_delay_ms: 5000       # Max AIMD backoff (default: 5000)
    max_capacity_retry_seconds: 3600  # Retry timeout (default: 3600)
```

**Execution Order:**

1. `2026-01-21-pooled-prompt-shield.md` Task 1 (extract pooling)
2. `2026-01-21-pooled-prompt-shield.md` Tasks 2-5 (Prompt Shield pooling)
3. `2026-01-21-pooled-content-safety.md` Tasks 1-4 (this plan)

Or execute both safety transform plans in parallel after Task 1 completes.
