# Pooled Execution for Azure Prompt Shield Transform

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add pooled concurrent execution to Azure Prompt Shield transform, enabling parallel API calls with AIMD throttling to eliminate sequential bottlenecks in content filtering pipelines.

**Architecture:** Reuse the existing `PooledExecutor` and `PoolConfig` from `plugins/llm/` by extracting them to a shared location. Modify Prompt Shield to support `pool_size` config and batch-aware processing following the same pattern as `AzureLLMTransform`.

**Tech Stack:** Existing PooledExecutor, CapacityError, AIMD throttle infrastructure from plugins/llm/

---

## Overview

The pipeline `keyword_filter → content_safety → prompt_shield → azure_llm` currently has sequential bottlenecks. With 100 rows at 200ms per API call:
- Content Safety: 20 seconds (sequential)
- Prompt Shield: 20 seconds (sequential)
- LLM with pool_size=10: ~2 seconds

This plan adds pooling to Prompt Shield. Content Safety will follow the same pattern in a future plan.

---

## Task 1: Move Pooling Infrastructure to Shared Location

**Files:**
- Create: `src/elspeth/plugins/pooling/__init__.py`
- Create: `src/elspeth/plugins/pooling/executor.py`
- Create: `src/elspeth/plugins/pooling/config.py`
- Create: `src/elspeth/plugins/pooling/errors.py`
- Modify: `src/elspeth/plugins/llm/base.py` (import from new location)
- Modify: `src/elspeth/plugins/llm/azure.py` (import from new location)
- Modify: `src/elspeth/plugins/llm/openrouter.py` (import from new location)
- Modify: `src/elspeth/plugins/llm/__init__.py` (re-export from new location)

**Step 1: Create pooling package init**

```python
# src/elspeth/plugins/pooling/__init__.py
"""Shared pooling infrastructure for parallel API transforms."""

from elspeth.plugins.pooling.config import PoolConfig
from elspeth.plugins.pooling.errors import CapacityError, is_capacity_error
from elspeth.plugins.pooling.executor import PooledExecutor, RowContext
from elspeth.plugins.pooling.throttle import AIMDThrottle, ThrottleConfig

__all__ = [
    "AIMDThrottle",
    "CapacityError",
    "PoolConfig",
    "PooledExecutor",
    "RowContext",
    "ThrottleConfig",
    "is_capacity_error",
]
```

**Step 2: Move PoolConfig to pooling/config.py**

```python
# src/elspeth/plugins/pooling/config.py
"""Pool configuration for concurrent API transforms."""

from __future__ import annotations

from pydantic import BaseModel, Field

from elspeth.plugins.pooling.throttle import ThrottleConfig


class PoolConfig(BaseModel):
    """Pool configuration for concurrent API requests.

    Attributes:
        pool_size: Number of concurrent requests (must be >= 1)
        min_dispatch_delay_ms: Floor for delay between dispatches
        max_dispatch_delay_ms: Ceiling for delay
        backoff_multiplier: Multiply delay on capacity error (must be > 1)
        recovery_step_ms: Subtract from delay on success
        max_capacity_retry_seconds: Max time to retry capacity errors per row
    """

    model_config = {"extra": "forbid"}

    pool_size: int = Field(1, ge=1, description="Number of concurrent requests")
    min_dispatch_delay_ms: int = Field(0, ge=0, description="Minimum dispatch delay in milliseconds")
    max_dispatch_delay_ms: int = Field(5000, ge=0, description="Maximum dispatch delay in milliseconds")
    backoff_multiplier: float = Field(2.0, gt=1.0, description="Backoff multiplier on capacity error")
    recovery_step_ms: int = Field(50, ge=0, description="Recovery step in milliseconds")
    max_capacity_retry_seconds: int = Field(3600, gt=0, description="Max seconds to retry capacity errors")

    def to_throttle_config(self) -> ThrottleConfig:
        """Convert to ThrottleConfig for runtime use."""
        return ThrottleConfig(
            min_dispatch_delay_ms=self.min_dispatch_delay_ms,
            max_dispatch_delay_ms=self.max_dispatch_delay_ms,
            backoff_multiplier=self.backoff_multiplier,
            recovery_step_ms=self.recovery_step_ms,
        )
```

**Step 3: Move CapacityError to pooling/errors.py**

```python
# src/elspeth/plugins/pooling/errors.py
"""Capacity error classification for pooled API transforms."""

from __future__ import annotations

# HTTP status codes that indicate capacity/rate limiting
CAPACITY_ERROR_CODES: frozenset[int] = frozenset({429, 503, 529})


def is_capacity_error(status_code: int) -> bool:
    """Check if HTTP status code indicates a capacity error."""
    return status_code in CAPACITY_ERROR_CODES


class CapacityError(Exception):
    """Exception for capacity/rate limit errors.

    Raised when an API call fails due to capacity limits.
    These errors trigger AIMD throttle and are retried.

    Attributes:
        status_code: HTTP status code that triggered this error
        retryable: Always True for capacity errors
    """

    def __init__(self, status_code: int, message: str) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.retryable = True
```

**Step 4: Move throttle to pooling/throttle.py**

Copy `src/elspeth/plugins/llm/aimd_throttle.py` to `src/elspeth/plugins/pooling/throttle.py` (no changes needed to content).

**Step 5: Move reorder buffer to pooling/reorder_buffer.py**

Copy `src/elspeth/plugins/llm/reorder_buffer.py` to `src/elspeth/plugins/pooling/reorder_buffer.py` (no changes needed to content).

**Step 6: Move executor to pooling/executor.py**

Copy `src/elspeth/plugins/llm/pooled_executor.py` to `src/elspeth/plugins/pooling/executor.py` and update imports:

```python
# At top of file, change:
from elspeth.plugins.llm.aimd_throttle import AIMDThrottle
from elspeth.plugins.llm.base import PoolConfig
from elspeth.plugins.llm.capacity_errors import CapacityError
from elspeth.plugins.llm.reorder_buffer import ReorderBuffer

# To:
from elspeth.plugins.pooling.config import PoolConfig
from elspeth.plugins.pooling.errors import CapacityError
from elspeth.plugins.pooling.reorder_buffer import ReorderBuffer
from elspeth.plugins.pooling.throttle import AIMDThrottle
```

**Step 7: Update LLM imports**

In `src/elspeth/plugins/llm/base.py`, change:
```python
# Old imports (remove these):
from elspeth.plugins.llm.aimd_throttle import ThrottleConfig

# New imports:
from elspeth.plugins.pooling.config import PoolConfig
from elspeth.plugins.pooling.throttle import ThrottleConfig
```

Then remove the local `PoolConfig` class definition.

In `src/elspeth/plugins/llm/azure.py`, change:
```python
# Old:
from elspeth.plugins.llm.capacity_errors import CapacityError
from elspeth.plugins.llm.pooled_executor import PooledExecutor, RowContext

# New:
from elspeth.plugins.pooling import CapacityError, PooledExecutor, RowContext
```

Similar changes in `openrouter.py`.

**Step 8: Update LLM __init__.py re-exports**

```python
# In src/elspeth/plugins/llm/__init__.py, change:
# from elspeth.plugins.llm.base import BaseLLMTransform, LLMConfig, PoolConfig
# from elspeth.plugins.llm.capacity_errors import CapacityError

# To:
from elspeth.plugins.llm.base import BaseLLMTransform, LLMConfig
from elspeth.plugins.pooling import CapacityError, PoolConfig
```

**Step 9: Run tests to verify refactor is correct**

Run: `pytest tests/plugins/llm/ -v`
Expected: All tests pass (no functional changes)

**Step 10: Commit refactor**

```bash
git add src/elspeth/plugins/pooling/ src/elspeth/plugins/llm/
git commit -m "$(cat <<'EOF'
refactor: extract pooling infrastructure to shared location

Move PoolConfig, PooledExecutor, CapacityError, AIMD throttle to
plugins/pooling/ so they can be reused by non-LLM transforms like
Prompt Shield and Content Safety.

No functional changes - just reorganization.

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>
EOF
)"
```

---

## Task 2: Add Pool Config Fields to Prompt Shield Config

**Files:**
- Modify: `src/elspeth/plugins/transforms/azure/prompt_shield.py`

**Step 1: Write failing test for pool_size config**

Add to `tests/plugins/transforms/azure/test_prompt_shield.py`:

```python
class TestPromptShieldPoolConfig:
    """Tests for Prompt Shield pool configuration."""

    def test_pool_size_default_is_one(self) -> None:
        """Default pool_size is 1 (sequential)."""
        from elspeth.plugins.transforms.azure.prompt_shield import (
            AzurePromptShieldConfig,
        )

        cfg = AzurePromptShieldConfig.from_dict(
            {
                "endpoint": "https://test.cognitiveservices.azure.com",
                "api_key": "test-key",
                "fields": ["prompt"],
                "schema": {"fields": "dynamic"},
            }
        )

        assert cfg.pool_size == 1

    def test_pool_size_configurable(self) -> None:
        """pool_size can be configured."""
        from elspeth.plugins.transforms.azure.prompt_shield import (
            AzurePromptShieldConfig,
        )

        cfg = AzurePromptShieldConfig.from_dict(
            {
                "endpoint": "https://test.cognitiveservices.azure.com",
                "api_key": "test-key",
                "fields": ["prompt"],
                "schema": {"fields": "dynamic"},
                "pool_size": 5,
            }
        )

        assert cfg.pool_size == 5

    def test_pool_config_property_returns_none_when_sequential(self) -> None:
        """pool_config returns None when pool_size=1."""
        from elspeth.plugins.transforms.azure.prompt_shield import (
            AzurePromptShieldConfig,
        )

        cfg = AzurePromptShieldConfig.from_dict(
            {
                "endpoint": "https://test.cognitiveservices.azure.com",
                "api_key": "test-key",
                "fields": ["prompt"],
                "schema": {"fields": "dynamic"},
                "pool_size": 1,
            }
        )

        assert cfg.pool_config is None

    def test_pool_config_property_returns_config_when_pooled(self) -> None:
        """pool_config returns PoolConfig when pool_size>1."""
        from elspeth.plugins.transforms.azure.prompt_shield import (
            AzurePromptShieldConfig,
        )

        cfg = AzurePromptShieldConfig.from_dict(
            {
                "endpoint": "https://test.cognitiveservices.azure.com",
                "api_key": "test-key",
                "fields": ["prompt"],
                "schema": {"fields": "dynamic"},
                "pool_size": 3,
            }
        )

        assert cfg.pool_config is not None
        assert cfg.pool_config.pool_size == 3
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/plugins/transforms/azure/test_prompt_shield.py::TestPromptShieldPoolConfig -v`
Expected: FAIL - pool_size field doesn't exist

**Step 3: Add pool config fields to AzurePromptShieldConfig**

In `src/elspeth/plugins/transforms/azure/prompt_shield.py`, update the config class:

```python
from pydantic import Field

from elspeth.plugins.pooling import PoolConfig  # Add this import


class AzurePromptShieldConfig(TransformDataConfig):
    """Configuration for Azure Prompt Shield transform.

    Requires:
        endpoint: Azure Content Safety endpoint URL
        api_key: Azure Content Safety API key
        fields: Field name(s) to analyze, or 'all' for all string fields
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

Run: `pytest tests/plugins/transforms/azure/test_prompt_shield.py::TestPromptShieldPoolConfig -v`
Expected: All tests pass

**Step 5: Commit**

```bash
git add src/elspeth/plugins/transforms/azure/prompt_shield.py tests/plugins/transforms/azure/test_prompt_shield.py
git commit -m "$(cat <<'EOF'
feat(prompt-shield): add pool configuration fields

Add pool_size and AIMD throttle config options to Prompt Shield.
pool_size=1 (default) maintains sequential behavior.

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>
EOF
)"
```

---

## Task 3: Implement Pooled Execution in Prompt Shield

**Files:**
- Modify: `src/elspeth/plugins/transforms/azure/prompt_shield.py`
- Modify: `tests/plugins/transforms/azure/test_prompt_shield.py`

> **Pre-existing Test Failures:** The existing tests in `test_prompt_shield.py` have 26 failures because they mock `ctx.http_client.post`, but the implementation uses `self._http_client` (its own httpx.Client instance). These tests need to be fixed as part of this task by updating the mock pattern to use `patch("httpx.Client")` instead.

**Step 0: Fix existing test helper to use correct mock pattern**

The existing `make_mock_context()` helper mocks the wrong interface. Update it to properly mock httpx.Client:

```python
# Replace the existing make_mock_context function with this:
from unittest.mock import MagicMock, patch

def make_prompt_shield_with_mock_response(
    config: dict[str, Any],
    response_data: dict[str, Any],
) -> tuple[AzurePromptShield, MagicMock]:
    """Create Prompt Shield transform with mocked HTTP client.

    Returns the transform and the mock client for assertions.
    The httpx.Client is patched at module level.
    """
    transform = AzurePromptShield(config)

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


# Update existing tests to use the new helper pattern, for example:
class TestAzurePromptShieldTransform:
    """Tests for AzurePromptShield transform."""

    def test_clean_content_passes(self) -> None:
        """Content without attacks passes through."""
        transform, mock_client = make_prompt_shield_with_mock_response(
            {
                "endpoint": "https://test.cognitiveservices.azure.com",
                "api_key": "test-key",
                "fields": ["prompt"],
                "schema": {"fields": "dynamic"},
            },
            {
                "userPromptAnalysis": {"attackDetected": False},
                "documentsAnalysis": [{"attackDetected": False}],
            },
        )

        # Create minimal mock context (no http_client needed)
        ctx = MagicMock()
        ctx.state_id = "test-state"
        ctx.run_id = "test-run"
        ctx.landscape = None  # No audit recording in unit tests

        row = {"prompt": "What is the weather?", "id": 1}
        result = transform.process(row, ctx)

        assert result.status == "success"
        assert result.row == row
        mock_client.post.assert_called_once()
```

Run existing tests after fixing: `pytest tests/plugins/transforms/azure/test_prompt_shield.py -v`
Expected: All 26 previously failing tests now pass.

**Step 1: Write failing test for pooled batch execution**

Add to test file:

```python
class TestPromptShieldPooledExecution:
    """Tests for Prompt Shield pooled execution."""

    def test_batch_aware_is_true_when_pooled(self) -> None:
        """Transform is batch_aware when pool_size > 1."""
        from elspeth.plugins.transforms.azure.prompt_shield import AzurePromptShield

        transform = AzurePromptShield(
            {
                "endpoint": "https://test.cognitiveservices.azure.com",
                "api_key": "test-key",
                "fields": ["prompt"],
                "schema": {"fields": "dynamic"},
                "pool_size": 3,
            }
        )

        assert transform.is_batch_aware is True

    def test_batch_aware_is_false_when_sequential(self) -> None:
        """Transform is not batch_aware when pool_size=1."""
        from elspeth.plugins.transforms.azure.prompt_shield import AzurePromptShield

        transform = AzurePromptShield(
            {
                "endpoint": "https://test.cognitiveservices.azure.com",
                "api_key": "test-key",
                "fields": ["prompt"],
                "schema": {"fields": "dynamic"},
                "pool_size": 1,
            }
        )

        assert transform.is_batch_aware is False

    def test_pooled_execution_processes_batch_concurrently(self) -> None:
        """Pooled transform processes batch rows concurrently."""
        from unittest.mock import MagicMock, patch

        from elspeth.plugins.transforms.azure.prompt_shield import AzurePromptShield

        transform = AzurePromptShield(
            {
                "endpoint": "https://test.cognitiveservices.azure.com",
                "api_key": "test-key",
                "fields": ["prompt"],
                "schema": {"fields": "dynamic"},
                "pool_size": 3,
            }
        )

        # Create mock context with landscape and state_id
        ctx = MagicMock()
        ctx.landscape = MagicMock()
        ctx.state_id = "test-state"
        ctx.run_id = "test-run"

        # Mock HTTP responses (all clean)
        response_mock = MagicMock()
        response_mock.status_code = 200
        response_mock.json.return_value = {
            "userPromptAnalysis": {"attackDetected": False},
            "documentsAnalysis": [{"attackDetected": False}],
        }
        response_mock.raise_for_status = MagicMock()

        # Patch httpx.Client to return mock
        with patch("httpx.Client") as mock_client_class:
            mock_client = MagicMock()
            mock_client.post.return_value = response_mock
            mock_client_class.return_value = mock_client

            # Process batch of 3 rows
            rows = [
                {"prompt": "row 1", "id": 1},
                {"prompt": "row 2", "id": 2},
                {"prompt": "row 3", "id": 3},
            ]
            result = transform.process(rows, ctx)

        # Should return success_multi with all rows
        assert result.status == "success"
        assert result.rows is not None
        assert len(result.rows) == 3

    def test_pooled_execution_handles_mixed_results(self) -> None:
        """Pooled execution correctly tracks errors per row."""
        from unittest.mock import MagicMock, patch

        from elspeth.plugins.transforms.azure.prompt_shield import AzurePromptShield

        transform = AzurePromptShield(
            {
                "endpoint": "https://test.cognitiveservices.azure.com",
                "api_key": "test-key",
                "fields": ["prompt"],
                "schema": {"fields": "dynamic"},
                "pool_size": 3,
            }
        )

        ctx = MagicMock()
        ctx.landscape = MagicMock()
        ctx.state_id = "test-state"
        ctx.run_id = "test-run"

        # Track call count to return different responses
        call_count = [0]

        def mock_post(*args, **kwargs):
            call_count[0] += 1
            response = MagicMock()
            response.status_code = 200
            response.raise_for_status = MagicMock()

            # Second row has attack detected
            if call_count[0] == 2:
                response.json.return_value = {
                    "userPromptAnalysis": {"attackDetected": True},
                    "documentsAnalysis": [{"attackDetected": False}],
                }
            else:
                response.json.return_value = {
                    "userPromptAnalysis": {"attackDetected": False},
                    "documentsAnalysis": [{"attackDetected": False}],
                }
            return response

        with patch("httpx.Client") as mock_client_class:
            mock_client = MagicMock()
            mock_client.post.side_effect = mock_post
            mock_client_class.return_value = mock_client

            rows = [
                {"prompt": "clean 1", "id": 1},
                {"prompt": "attack", "id": 2},
                {"prompt": "clean 2", "id": 3},
            ]
            result = transform.process(rows, ctx)

        # Should return ALL rows with per-row error tracking
        assert result.status == "success"
        assert result.rows is not None
        assert len(result.rows) == 3  # ALL rows included, not just successes

        # Row 1: success (no error marker)
        assert result.rows[0].get("_prompt_shield_error") is None
        assert result.rows[0]["id"] == 1

        # Row 2: error (attack detected) - has error marker
        assert result.rows[1].get("_prompt_shield_error") is not None
        assert result.rows[1]["_prompt_shield_error"]["reason"] == "prompt_injection_detected"
        assert result.rows[1]["id"] == 2

        # Row 3: success (no error marker)
        assert result.rows[2].get("_prompt_shield_error") is None
        assert result.rows[2]["id"] == 3
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/plugins/transforms/azure/test_prompt_shield.py::TestPromptShieldPooledExecution -v`
Expected: FAIL - batch processing not implemented

**Step 3: Implement pooled execution**

Update `src/elspeth/plugins/transforms/azure/prompt_shield.py`:

```python
"""Azure Prompt Shield transform with optional pooled execution."""

from __future__ import annotations

import time
from threading import Lock
from typing import TYPE_CHECKING, Any

import httpx
from pydantic import Field

from elspeth.contracts import CallStatus, CallType, Determinism
from elspeth.plugins.base import BaseTransform
from elspeth.plugins.config_base import TransformDataConfig
from elspeth.plugins.context import PluginContext
from elspeth.plugins.pooling import CapacityError, PoolConfig, PooledExecutor, RowContext
from elspeth.plugins.results import TransformResult
from elspeth.plugins.schema_factory import create_schema_from_config

if TYPE_CHECKING:
    from elspeth.core.landscape.recorder import LandscapeRecorder


class AzurePromptShieldConfig(TransformDataConfig):
    """Configuration for Azure Prompt Shield transform."""
    # ... (keep existing fields, add pool fields as shown in Task 2)


class AzurePromptShield(BaseTransform):
    """Detect jailbreak and prompt injection with optional pooled execution.

    Design Notes:
    - is_batch_aware is DYNAMIC based on pool_size (True when pool_size > 1)
    - This differs from AzureLLMTransform which is always batch_aware=True
    - Rationale: Prompt Shield without pooling should preserve original sequential
      behavior for backwards compatibility. With pooling, batch processing enables
      concurrent execution.
    """

    name = "azure_prompt_shield"
    determinism = Determinism.EXTERNAL_CALL
    plugin_version = "1.0.0"
    creates_tokens = False

    API_VERSION = "2024-09-01"

    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__(config)

        cfg = AzurePromptShieldConfig.from_dict(config)
        self._endpoint = cfg.endpoint.rstrip("/")
        self._api_key = cfg.api_key
        self._fields = cfg.fields
        self._on_error = cfg.on_error

        assert cfg.schema_config is not None
        schema = create_schema_from_config(
            cfg.schema_config,
            "AzurePromptShieldSchema",
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
        """Process row(s) through Prompt Shield.

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
        - Embed errors per-row via _prompt_shield_error field
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
                output_row["_prompt_shield_error"] = result.reason or {
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
        """Process single row through Prompt Shield API.

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
                analysis = self._analyze_prompt(value, state_id)
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

            if analysis["user_prompt_attack"] or analysis["document_attack"]:
                return TransformResult.error(
                    {
                        "reason": "prompt_injection_detected",
                        "field": field_name,
                        "attacks": analysis,
                        "retryable": False,
                    }
                )

        return TransformResult.success(row)

    def _analyze_prompt(
        self,
        text: str,
        state_id: str,
    ) -> dict[str, bool]:
        """Call Azure Prompt Shield API with audit trail recording.

        All API calls are recorded to the Landscape for full auditability.
        This ensures explain() queries can trace why rows were flagged.

        Args:
            text: Text content to analyze
            state_id: State ID for audit trail association

        Returns:
            Dict with user_prompt_attack and document_attack booleans

        Raises:
            httpx.HTTPStatusError: On HTTP errors (including rate limits)
            httpx.RequestError: On network errors or malformed responses
        """
        client = self._get_http_client()
        url = f"{self._endpoint}/contentsafety/text:shieldPrompt?api-version={self.API_VERSION}"

        # Build request data for audit trail
        request_data = {
            "userPrompt": text,
            "documents": [text],
            "endpoint": self._endpoint,
            "api_version": self.API_VERSION,
        }

        call_index = self._next_call_index()
        start = time.perf_counter()

        try:
            response = client.post(
                url,
                json={"userPrompt": text, "documents": [text]},
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
                user_attack = data["userPromptAnalysis"]["attackDetected"]
                documents_analysis = data["documentsAnalysis"]
                doc_attack = any(doc["attackDetected"] for doc in documents_analysis)
            except (KeyError, TypeError) as e:
                raise httpx.RequestError(f"Malformed Prompt Shield response: {e}") from e

            # Record successful call to audit trail
            if self._recorder is not None:
                self._recorder.record_call(
                    state_id=state_id,
                    call_index=call_index,
                    call_type=CallType.HTTP,
                    status=CallStatus.SUCCESS,
                    request_data=request_data,
                    response_data={
                        "user_prompt_attack": user_attack,
                        "document_attack": doc_attack,
                    },
                    latency_ms=latency_ms,
                )

            return {"user_prompt_attack": user_attack, "document_attack": doc_attack}

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

    def _get_fields_to_scan(self, row: dict[str, Any]) -> list[str]:
        """Determine which fields to scan based on config."""
        if self._fields == "all":
            return [k for k, v in row.items() if isinstance(v, str)]
        elif isinstance(self._fields, str):
            return [self._fields]
        else:
            return self._fields

    def close(self) -> None:
        """Release resources."""
        if self._http_client is not None:
            self._http_client.close()
            self._http_client = None
        if self._executor is not None:
            self._executor.shutdown(wait=True)
```

**Step 4: Run tests to verify they pass**

Run: `pytest tests/plugins/transforms/azure/test_prompt_shield.py -v`
Expected: All tests pass

**Step 5: Run type checker**

Run: `mypy src/elspeth/plugins/transforms/azure/prompt_shield.py`
Expected: Success

**Step 6: Commit implementation**

```bash
git add src/elspeth/plugins/transforms/azure/prompt_shield.py tests/plugins/transforms/azure/test_prompt_shield.py
git commit -m "$(cat <<'EOF'
feat(prompt-shield): add pooled execution with audit trail

Prompt Shield now supports concurrent API calls with pool_size>1.
Uses shared PooledExecutor infrastructure from plugins/pooling/.

Features:
- pool_size config option (default 1 = sequential)
- AIMD throttle for adaptive rate limiting
- Batch processing for aggregation compatibility
- Per-row error tracking in batch results
- Full audit trail recording for all API calls (fixes auditability gap)

Technical changes:
- Single shared HTTP client (not per-state_id) for efficiency
- Consolidated _analyze_prompt method with audit recording
- Dynamic is_batch_aware based on pool_size configuration

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>
EOF
)"
```

---

## Task 4: Update Example Configuration

**Files:**
- Modify: `examples/azure_blob_sentiment/settings.yaml`
- Modify: `examples/azure_blob_sentiment/README.md`

**Step 1: Add pool_size to Prompt Shield in settings.yaml**

```yaml
  # Step 3: Check for prompt injection attacks
  - plugin: azure_prompt_shield
    options:
      endpoint: "${AZURE_CONTENT_SAFETY_ENDPOINT}"
      api_key: "${AZURE_CONTENT_SAFETY_KEY}"
      fields: text
      on_error: attacks
      schema:
        fields: dynamic
      pool_size: 5  # Process 5 rows concurrently
```

**Step 2: Add documentation to README**

Add after the LLM pooling section:

```markdown
### Pooled Safety Transforms

Content Safety and Prompt Shield also support pooled execution:

```yaml
- plugin: azure_prompt_shield
  options:
    endpoint: "${AZURE_CONTENT_SAFETY_ENDPOINT}"
    api_key: "${AZURE_CONTENT_SAFETY_KEY}"
    fields: text
    pool_size: 5  # Process 5 rows concurrently
```

This significantly reduces pipeline latency when processing large batches.
```

**Step 3: Commit**

```bash
git add examples/azure_blob_sentiment/
git commit -m "$(cat <<'EOF'
docs: add pool_size to Prompt Shield example config

Demonstrates concurrent Prompt Shield execution with pool_size: 5.

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>
EOF
)"
```

---

## Task 5: Run Full Test Suite

**Files:** None (verification only)

**Step 1: Run linter**

Run: `ruff check src/elspeth/plugins/pooling/ src/elspeth/plugins/transforms/azure/`
Expected: No errors

**Step 2: Run type checker**

Run: `mypy src/elspeth/plugins/pooling/ src/elspeth/plugins/transforms/azure/`
Expected: Success

**Step 3: Run all related tests**

Run: `pytest tests/plugins/transforms/azure/ tests/plugins/llm/ -v`
Expected: All tests pass

**Step 4: Run full test suite**

Run: `pytest tests/ -x --ignore=tests/integration`
Expected: All tests pass

---

## Summary

| Task | Description | Files |
|------|-------------|-------|
| 1 | Extract pooling infrastructure | `plugins/pooling/*`, `plugins/llm/*` |
| 2 | Add pool config to Prompt Shield | `prompt_shield.py`, tests |
| 3 | Implement pooled execution | `prompt_shield.py`, tests |
| 4 | Update example config | `settings.yaml`, `README.md` |
| 5 | Full verification | (none) |

**New Configuration Options:**
```yaml
- plugin: azure_prompt_shield
  options:
    pool_size: 5                      # Concurrent workers (default: 1)
    max_dispatch_delay_ms: 5000       # Max AIMD backoff (default: 5000)
    max_capacity_retry_seconds: 3600  # Retry timeout (default: 3600)
```

**Future Work:**
- Apply same pattern to `azure_content_safety` transform
- Consider unified `SafetyTransformConfig` base class for shared config
