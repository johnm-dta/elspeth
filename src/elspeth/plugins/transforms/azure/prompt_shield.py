"""Azure Prompt Shield transform for jailbreak and prompt injection detection.

This module provides the AzurePromptShield transform which uses Azure's
Prompt Shield API to detect:
- User prompt attacks (jailbreak attempts in the user's message)
- Document attacks (prompt injection in documents/context)

Unlike Content Safety, Prompt Shield is binary detection - no thresholds.
Either an attack is detected or it isn't.

Uses BatchTransformMixin for row-level pipelining (multiple rows in flight
with FIFO output ordering) and PooledExecutor for internal concurrency.
"""

from __future__ import annotations

from collections.abc import Callable
from threading import Lock
from typing import TYPE_CHECKING, Any

import httpx
from pydantic import Field

from elspeth.contracts import Determinism
from elspeth.plugins.base import BaseTransform
from elspeth.plugins.batching import BatchTransformMixin, OutputPort
from elspeth.plugins.config_base import TransformDataConfig
from elspeth.plugins.context import PluginContext
from elspeth.plugins.pooling import CapacityError, PoolConfig, PooledExecutor, is_capacity_error
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


class AzurePromptShield(BaseTransform, BatchTransformMixin):
    """Detect jailbreak attempts and prompt injection using Azure Prompt Shield.

    Analyzes text against Azure's Prompt Shield API which detects:
    - User prompt attacks: Direct jailbreak attempts in the user's message
    - Document attacks: Prompt injection hidden in documents or context

    Returns error result if any attack is detected (binary, no thresholds).

    Uses BatchTransformMixin for row-level pipelining: multiple rows can be
    in flight concurrently with FIFO output ordering.

    Architecture:
        Orchestrator → accept() → [RowReorderBuffer] → [Worker Pool]
            → _process_row() → Azure API
            → emit() → OutputPort (sink or next transform)

    Usage:
        # 1. Instantiate
        transform = AzurePromptShield(config)

        # 2. Connect output port (required before accept())
        transform.connect_output(output_port, max_pending=30)

        # 3. Feed rows (blocks on backpressure)
        for row in source:
            transform.accept(row, ctx)

        # 4. Flush and close
        transform.flush_batch_processing()
        transform.close()
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
        self._max_capacity_retry_seconds = cfg.max_capacity_retry_seconds

        schema = create_schema_from_config(
            cfg.schema_config,
            "AzurePromptShieldSchema",
            allow_coercion=False,
        )
        self.input_schema = schema
        # BatchTransformMixin processes rows individually, output schema matches input
        self.output_schema = schema

        # Per-state_id HTTP client cache for audit trail
        # Each AuditedHTTPClient has its own call_index counter, ensuring
        # (state_id, call_index) uniqueness even across retries.
        self._http_clients: dict[str, Any] = {}  # state_id -> AuditedHTTPClient
        self._http_clients_lock = Lock()

        # Recorder reference for pooled execution (set in on_start or first accept)
        self._recorder: LandscapeRecorder | None = None
        self._run_id: str = ""
        self._telemetry_emit: Callable[[Any], None] = lambda event: None
        self._limiter: Any = None  # RateLimiter | NoOpLimiter | None

        # Create pooled executor if pool_size > 1 (for internal concurrency)
        if cfg.pool_config is not None:
            self._executor: PooledExecutor | None = PooledExecutor(cfg.pool_config)
        else:
            self._executor = None

        # Batch processing state (initialized by connect_output)
        self._batch_initialized = False

    def on_start(self, ctx: PluginContext) -> None:
        """Capture recorder, telemetry, and rate limit context for pooled execution."""
        self._recorder = ctx.landscape
        self._run_id = ctx.run_id
        self._telemetry_emit = ctx.telemetry_emit
        # Get rate limiter for Azure Prompt Shield service (None if rate limiting disabled)
        self._limiter = ctx.rate_limit_registry.get_limiter("azure_prompt_shield") if ctx.rate_limit_registry is not None else None

    def connect_output(
        self,
        output: OutputPort,
        max_pending: int = 30,
    ) -> None:
        """Connect output port and initialize batch processing.

        Call this after __init__ but before accept(). The output port
        receives results in FIFO order (submission order, not completion order).

        Args:
            output: Output port to emit results to (sink adapter or next transform)
            max_pending: Maximum rows in flight before accept() blocks (backpressure)

        Raises:
            RuntimeError: If called more than once
        """
        if self._batch_initialized:
            raise RuntimeError("connect_output() already called")

        self.init_batch_processing(
            max_pending=max_pending,
            output=output,
            name=self.name,
            max_workers=max_pending,  # Match workers to max_pending
            batch_wait_timeout=float(self._max_capacity_retry_seconds),
        )
        self._batch_initialized = True

    def accept(self, row: dict[str, Any], ctx: PluginContext) -> None:
        """Accept a row for processing.

        Submits the row to the batch processing pipeline. Returns quickly
        unless backpressure is applied (buffer full). Results flow through
        the output port in FIFO order.

        Args:
            row: Input row with fields to analyze
            ctx: Plugin context with token and landscape

        Raises:
            RuntimeError: If connect_output() was not called
            ValueError: If ctx.token is None
        """
        if not self._batch_initialized:
            raise RuntimeError("connect_output() must be called before accept(). This wires up the output port for result emission.")

        # Capture recorder on first row (same as on_start)
        if self._recorder is None and ctx.landscape is not None:
            self._recorder = ctx.landscape

        self.accept_row(row, ctx, self._process_row)

    def process(
        self,
        row: dict[str, Any],
        ctx: PluginContext,
    ) -> TransformResult:
        """Not supported - use accept() for row-level pipelining.

        This transform uses BatchTransformMixin for concurrent row processing
        with FIFO output ordering. Call accept() instead of process().

        Raises:
            NotImplementedError: Always, directing callers to use accept()
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} uses row-level pipelining. Use accept() instead of process(). See class docstring for usage."
        )

    def _process_row(
        self,
        row: dict[str, Any],
        ctx: PluginContext,
    ) -> TransformResult:
        """Process a single row. Called by worker threads.

        This is the processor function passed to accept_row(). It runs in
        the BatchTransformMixin's worker pool.

        Args:
            row: Input row with fields to analyze
            ctx: Plugin context with state_id for audit trail

        Returns:
            TransformResult indicating success or attack detection
        """
        if ctx.state_id is None:
            raise RuntimeError("state_id is required for batch processing. Ensure transform is executed through the engine.")

        try:
            return self._process_single_with_state(row, ctx.state_id)
        finally:
            # Clean up cached HTTP client for this state_id
            with self._http_clients_lock:
                self._http_clients.pop(ctx.state_id, None)

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
                status_code = e.response.status_code
                if is_capacity_error(status_code):
                    # Convert to CapacityError for pooled executor retry (429/503/529)
                    raise CapacityError(status_code, str(e)) from e
                return TransformResult.error(
                    {
                        "reason": "api_error",
                        "error_type": "http_error",
                        "status_code": status_code,
                        "message": str(e),
                    },
                    retryable=False,
                )
            except httpx.RequestError as e:
                return TransformResult.error(
                    {
                        "reason": "api_error",
                        "error_type": "network_error",
                        "message": str(e),
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
                    },
                    retryable=False,
                )

        return TransformResult.success(
            row,
            success_reason={"action": "validated"},
        )

    def _get_fields_to_scan(self, row: dict[str, Any]) -> list[str]:
        """Determine which fields to scan based on config."""
        if self._fields == "all":
            return [k for k, v in row.items() if isinstance(v, str)]
        elif isinstance(self._fields, str):
            return [self._fields]
        else:
            return self._fields

    def _get_http_client(self, state_id: str) -> Any:
        """Get or create audited HTTP client for a state_id.

        Clients are cached to preserve call_index across retries.
        This ensures uniqueness of (state_id, call_index) even when
        the pooled executor retries after CapacityError.

        Args:
            state_id: State ID for audit trail recording

        Returns:
            AuditedHTTPClient instance with per-instance call_index counter
        """
        from elspeth.plugins.clients.http import AuditedHTTPClient

        with self._http_clients_lock:
            if state_id not in self._http_clients:
                if self._recorder is None:
                    raise RuntimeError("PromptShield requires recorder for audited calls.")
                self._http_clients[state_id] = AuditedHTTPClient(
                    recorder=self._recorder,
                    state_id=state_id,
                    run_id=self._run_id,
                    telemetry_emit=self._telemetry_emit,
                    timeout=30.0,
                    headers={
                        "Ocp-Apim-Subscription-Key": self._api_key,
                        "Content-Type": "application/json",
                    },
                    limiter=self._limiter,
                )
            return self._http_clients[state_id]

    def _analyze_prompt(
        self,
        text: str,
        state_id: str,
    ) -> dict[str, bool]:
        """Call Azure Prompt Shield API.

        Args:
            text: Text to analyze for prompt injection
            state_id: State ID for audit trail recording

        Returns dict with:
            user_prompt_attack: True if jailbreak detected in user prompt
            document_attack: True if prompt injection detected in any document

        Uses AuditedHTTPClient for automatic audit recording and telemetry emission.
        """
        # Use AuditedHTTPClient - handles recording and telemetry automatically
        http_client = self._get_http_client(state_id)

        url = f"{self._endpoint}/contentsafety/text:shieldPrompt?api-version={self.API_VERSION}"

        # Make HTTP call - AuditedHTTPClient records to Landscape and emits telemetry
        response = http_client.post(
            url,
            json={"userPrompt": text, "documents": [text]},
        )
        response.raise_for_status()

        # Parse response - Azure API responses are external data (Tier 3: Zero Trust)
        # Security transform: fail CLOSED on malformed response
        try:
            data = response.json()
            user_attack = data["userPromptAnalysis"]["attackDetected"]
            documents_analysis = data["documentsAnalysis"]
            doc_attack = any(doc["attackDetected"] for doc in documents_analysis)

            return {
                "user_prompt_attack": user_attack,
                "document_attack": doc_attack,
            }

        except (KeyError, TypeError) as e:
            # Malformed response - the HTTP call was recorded as SUCCESS by AuditedHTTPClient
            # (because we got a 200), but the response is unusable at the application level
            raise httpx.RequestError(f"Malformed Prompt Shield response: {e}") from e

    def close(self) -> None:
        """Release resources."""
        # Shutdown batch processing infrastructure first
        if self._batch_initialized:
            self.shutdown_batch_processing()

        # Then shutdown query-level executor (if used for internal concurrency)
        if self._executor is not None:
            self._executor.shutdown(wait=True)

        # Close all cached HTTP clients
        with self._http_clients_lock:
            for client in self._http_clients.values():
                client.close()
            self._http_clients.clear()

        self._recorder = None
