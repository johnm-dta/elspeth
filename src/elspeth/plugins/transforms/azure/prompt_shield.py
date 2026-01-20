"""Azure Prompt Shield transform for jailbreak and prompt injection detection.

This module provides the AzurePromptShield transform which uses Azure's
Prompt Shield API to detect:
- User prompt attacks (jailbreak attempts in the user's message)
- Document attacks (prompt injection in documents/context)

Unlike Content Safety, Prompt Shield is binary detection - no thresholds.
Either an attack is detected or it isn't.

Supports both sequential (pool_size=1) and pooled (pool_size>1) execution modes.
"""

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
    """Configuration for Azure Prompt Shield transform.

    Requires:
        endpoint: Azure Content Safety endpoint URL
        api_key: Azure Content Safety API key
        fields: Field name(s) to analyze, or 'all' for all string fields
        schema: Schema configuration

    Optional:
        pool_size: Number of concurrent API calls (1=sequential, >1=pooled)
        min_dispatch_delay_ms: Minimum AIMD backoff delay (default 0)
        max_dispatch_delay_ms: Maximum AIMD backoff delay (default 5000)
        backoff_multiplier: Multiply delay on capacity error (default 2.0)
        recovery_step_ms: Subtract from delay on success (default 50)
        max_capacity_retry_seconds: Timeout for capacity error retries (default 3600)

    Example YAML:
        transforms:
          - plugin: azure_prompt_shield
            options:
              endpoint: https://my-resource.cognitiveservices.azure.com
              api_key: ${AZURE_CONTENT_SAFETY_KEY}
              fields: [prompt, user_message]
              on_error: quarantine_sink
              schema:
                fields: dynamic
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


class AzurePromptShield(BaseTransform):
    """Detect jailbreak attempts and prompt injection using Azure Prompt Shield.

    Analyzes text against Azure's Prompt Shield API which detects:
    - User prompt attacks: Direct jailbreak attempts in the user's message
    - Document attacks: Prompt injection hidden in documents or context

    Returns error result if any attack is detected (binary, no thresholds).

    Supports both sequential (pool_size=1) and pooled (pool_size>1) execution.
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
        self._pool_size = cfg.pool_size

        assert cfg.schema_config is not None
        schema = create_schema_from_config(
            cfg.schema_config,
            "AzurePromptShieldSchema",
            allow_coercion=False,
        )
        self.input_schema = schema
        self.output_schema = schema

        # Create own HTTP client (following OpenRouter pattern)
        # Single shared client since httpx.Client is stateless
        self._http_client: httpx.Client | None = None

        # Recorder reference for pooled execution (set in on_start)
        self._recorder: LandscapeRecorder | None = None

        # Create pooled executor if pool_size > 1
        if cfg.pool_config is not None:
            self._executor: PooledExecutor | None = PooledExecutor(cfg.pool_config)
        else:
            self._executor = None

        # Thread-safe call index counter for audit trail
        self._call_index = 0
        self._call_index_lock = Lock()

        # Dynamic is_batch_aware based on pool_size
        # Set as instance attribute to override class attribute
        self.is_batch_aware = self._pool_size > 1

    def on_start(self, ctx: PluginContext) -> None:
        """Capture recorder reference for pooled execution.

        In pooled mode, _process_single_with_state() is called from worker
        threads that don't have access to PluginContext. This captures the
        recorder reference at pipeline start so it can be used later.
        """
        self._recorder = ctx.landscape

    def process(
        self,
        row: dict[str, Any] | list[dict[str, Any]],
        ctx: PluginContext,
    ) -> TransformResult:
        """Analyze row content for prompt injection attacks.

        When is_batch_aware=True and used in aggregation, receives list[dict].
        Otherwise receives single dict.

        Routes to pooled or sequential execution based on pool_size config.
        """
        # Dispatch to batch processing if given a list
        # NOTE: This isinstance check is legitimate polymorphic dispatch for
        # batch-aware transforms, not defensive programming to hide bugs.
        if isinstance(row, list):
            return self._process_batch(row, ctx)

        # Route to pooled execution if configured (single row)
        if self._executor is not None:
            if ctx.landscape is None or ctx.state_id is None:
                raise RuntimeError(
                    "Pooled execution requires landscape recorder and state_id. Ensure transform is executed through the engine."
                )
            row_ctx = RowContext(row=row, state_id=ctx.state_id, row_index=0)
            results = self._executor.execute_batch(
                contexts=[row_ctx],
                process_fn=self._process_single_with_state,
            )
            return results[0]

        # Sequential execution path (pool_size=1)
        return self._process_single(row, ctx)

    def _process_single(
        self,
        row: dict[str, Any],
        ctx: PluginContext,
    ) -> TransformResult:
        """Process a single row sequentially.

        This is the original sequential processing logic.
        """
        fields_to_scan = self._get_fields_to_scan(row)

        for field_name in fields_to_scan:
            if field_name not in row:
                continue  # Skip fields not present in this row

            value = row[field_name]
            if not isinstance(value, str):
                continue

            # Call Azure API
            try:
                analysis = self._analyze_prompt(value, ctx.state_id)
            except httpx.HTTPStatusError as e:
                is_rate_limit = e.response.status_code == 429
                return TransformResult.error(
                    {
                        "reason": "api_error",
                        "error_type": "rate_limited" if is_rate_limit else "http_error",
                        "status_code": e.response.status_code,
                        "message": str(e),
                        "retryable": is_rate_limit,
                    },
                    retryable=is_rate_limit,
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

            # Check if any attack was detected
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

    def _process_batch(
        self,
        rows: list[dict[str, Any]],
        ctx: PluginContext,
    ) -> TransformResult:
        """Process batch of rows with parallel execution via PooledExecutor.

        Called when transform is used as aggregation node and trigger fires.
        All rows share the same state_id; call_index provides audit uniqueness.
        """
        if not rows:
            return TransformResult.success({"batch_empty": True, "row_count": 0})

        if ctx.landscape is None or ctx.state_id is None:
            raise RuntimeError(
                "Batch processing requires landscape recorder and state_id. Ensure transform is executed through the engine."
            )

        # Ensure we have an executor for parallel processing
        if self._executor is None:
            # Fallback: process sequentially if no pool configured
            return self._process_batch_sequential(rows, ctx)

        # Create contexts - all rows share same state_id (call_index provides uniqueness)
        contexts = [RowContext(row=row, state_id=ctx.state_id, row_index=i) for i, row in enumerate(rows)]

        # Execute all rows in parallel
        results = self._executor.execute_batch(
            contexts=contexts,
            process_fn=self._process_single_with_state,
        )

        # Assemble output with per-row error tracking
        return self._assemble_batch_results(rows, results)

    def _process_batch_sequential(
        self,
        rows: list[dict[str, Any]],
        ctx: PluginContext,
    ) -> TransformResult:
        """Fallback for batch processing without executor (pool_size=1).

        Processes rows one at a time using existing sequential logic.
        """
        results: list[TransformResult] = []
        for row in rows:
            result = self._process_single(row, ctx)
            results.append(result)
        return self._assemble_batch_results(rows, results)

    def _process_single_with_state(
        self,
        row: dict[str, Any],
        state_id: str,
    ) -> TransformResult:
        """Process a single row with explicit state_id.

        This is used by the pooled executor where each row has its own state.

        Raises:
            CapacityError: On rate limit errors (for pooled retry)
        """
        fields_to_scan = self._get_fields_to_scan(row)

        for field_name in fields_to_scan:
            if field_name not in row:
                continue  # Skip fields not present in this row

            value = row[field_name]
            if not isinstance(value, str):
                continue

            # Call Azure API with state_id for audit trail
            try:
                analysis = self._analyze_prompt(value, state_id)
            except httpx.HTTPStatusError as e:
                is_rate_limit = e.response.status_code == 429
                if is_rate_limit:
                    # Convert to CapacityError for pooled executor retry
                    raise CapacityError(429, str(e)) from e
                return TransformResult.error(
                    {
                        "reason": "api_error",
                        "error_type": "http_error",
                        "status_code": e.response.status_code,
                        "message": str(e),
                        "retryable": False,
                    },
                    retryable=False,
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

            # Check if any attack was detected
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

    def _assemble_batch_results(
        self,
        rows: list[dict[str, Any]],
        results: list[TransformResult],
    ) -> TransformResult:
        """Assemble batch results with per-row error tracking.

        Follows AzureLLMTransform pattern: include all rows in output,
        mark failures with _prompt_shield_error instead of failing entire batch.
        """
        output_rows: list[dict[str, Any]] = []
        all_failed = True

        for row, result in zip(rows, results, strict=True):
            output_row = dict(row)

            if result.status == "success" and result.row is not None:
                all_failed = False
                # Success - row passes through unchanged
                # (Prompt Shield doesn't add fields on success)
            else:
                # Per-row error tracking - embed error in row
                output_row["_prompt_shield_error"] = result.reason or {
                    "reason": "unknown_error",
                }

            output_rows.append(output_row)

        # Only return error if ALL rows failed
        if all_failed and output_rows:
            return TransformResult.error(
                {
                    "reason": "all_rows_failed",
                    "row_count": len(rows),
                }
            )

        return TransformResult.success_multi(output_rows)

    def _get_fields_to_scan(self, row: dict[str, Any]) -> list[str]:
        """Determine which fields to scan based on config."""
        if self._fields == "all":
            return [k for k, v in row.items() if isinstance(v, str)]
        elif isinstance(self._fields, str):
            return [self._fields]
        else:
            return self._fields

    def _get_http_client(self) -> httpx.Client:
        """Get or create HTTP client for API calls."""
        if self._http_client is None:
            self._http_client = httpx.Client(timeout=30.0)
        return self._http_client

    def _next_call_index(self) -> int:
        """Get next call index in thread-safe manner."""
        with self._call_index_lock:
            index = self._call_index
            self._call_index += 1
            return index

    def _analyze_prompt(
        self,
        text: str,
        state_id: str | None,
    ) -> dict[str, bool]:
        """Call Azure Prompt Shield API.

        Args:
            text: Text to analyze for prompt injection
            state_id: State ID for audit trail recording (None if no audit)

        Returns dict with:
            user_prompt_attack: True if jailbreak detected in user prompt
            document_attack: True if prompt injection detected in any document

        Records all API calls to audit trail for full traceability.
        """
        client = self._get_http_client()

        url = f"{self._endpoint}/contentsafety/text:shieldPrompt?api-version={self.API_VERSION}"

        request_data = {
            "userPrompt": text,
            "documents": [text],
        }

        # Track timing for audit
        start_time = time.monotonic()
        call_index = self._next_call_index()

        try:
            response = client.post(
                url,
                json=request_data,
                headers={
                    "Ocp-Apim-Subscription-Key": self._api_key,
                    "Content-Type": "application/json",
                },
            )
            response.raise_for_status()

            # Parse response - Azure API responses are external data (Tier 3: Zero Trust)
            # Security transform: fail CLOSED on malformed response
            data = response.json()
            user_attack = data["userPromptAnalysis"]["attackDetected"]
            documents_analysis = data["documentsAnalysis"]
            doc_attack = any(doc["attackDetected"] for doc in documents_analysis)

            latency_ms = (time.monotonic() - start_time) * 1000

            # Record successful call to audit trail
            if self._recorder is not None and state_id is not None:
                self._recorder.record_call(
                    state_id=state_id,
                    call_index=call_index,
                    call_type=CallType.HTTP,
                    status=CallStatus.SUCCESS,
                    request_data={"url": url, "body": request_data},
                    response_data=data,
                    latency_ms=latency_ms,
                )

            return {
                "user_prompt_attack": user_attack,
                "document_attack": doc_attack,
            }

        except (KeyError, TypeError) as e:
            latency_ms = (time.monotonic() - start_time) * 1000
            # Record failed call to audit trail
            if self._recorder is not None and state_id is not None:
                self._recorder.record_call(
                    state_id=state_id,
                    call_index=call_index,
                    call_type=CallType.HTTP,
                    status=CallStatus.ERROR,
                    request_data={"url": url, "body": request_data},
                    error={"reason": "malformed_response", "message": str(e)},
                    latency_ms=latency_ms,
                )
            raise httpx.RequestError(f"Malformed Prompt Shield response: {e}") from e

        except httpx.HTTPStatusError as e:
            latency_ms = (time.monotonic() - start_time) * 1000
            # Record failed call to audit trail
            if self._recorder is not None and state_id is not None:
                self._recorder.record_call(
                    state_id=state_id,
                    call_index=call_index,
                    call_type=CallType.HTTP,
                    status=CallStatus.ERROR,
                    request_data={"url": url, "body": request_data},
                    error={
                        "reason": "http_error",
                        "status_code": e.response.status_code,
                        "message": str(e),
                    },
                    latency_ms=latency_ms,
                )
            raise

        except httpx.RequestError as e:
            latency_ms = (time.monotonic() - start_time) * 1000
            # Record failed call to audit trail
            if self._recorder is not None and state_id is not None:
                self._recorder.record_call(
                    state_id=state_id,
                    call_index=call_index,
                    call_type=CallType.HTTP,
                    status=CallStatus.ERROR,
                    request_data={"url": url, "body": request_data},
                    error={"reason": "network_error", "message": str(e)},
                    latency_ms=latency_ms,
                )
            raise

    def close(self) -> None:
        """Release resources."""
        if self._executor is not None:
            self._executor.shutdown(wait=True)
        if self._http_client is not None:
            self._http_client.close()
            self._http_client = None
        self._recorder = None
