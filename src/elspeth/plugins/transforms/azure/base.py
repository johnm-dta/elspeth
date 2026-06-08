"""Base class for Azure safety transforms.

Provides shared infrastructure for Azure Content Safety and Azure Prompt Shield:
- Batch processing lifecycle (connect_output, accept, process, close)
- Audited HTTP client management with per-state_id caching
- Rate limiting integration
- Recorder/telemetry capture from LifecycleContext
- Field scanning loop with shared error handling

Subclasses implement _analyze_field() for their specific Azure API.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from threading import Event, Lock
from typing import Any

import httpx
import structlog
from pydantic import Field, field_validator

from elspeth.contracts import Determinism
from elspeth.contracts.audit_protocols import PluginAuditWriter
from elspeth.contracts.contexts import LifecycleContext, TransformContext
from elspeth.contracts.errors import CapacityError, FrameworkBugError, PluginRetryableError, is_capacity_error
from elspeth.contracts.schema_contract import PipelineRow
from elspeth.plugins.infrastructure.base import BaseTransform
from elspeth.plugins.infrastructure.batching import BatchTransformMixin, OutputPort
from elspeth.plugins.infrastructure.clients.json_utils import parse_json_strict
from elspeth.plugins.infrastructure.config_base import TransformDataConfig
from elspeth.plugins.infrastructure.results import TransformResult
from elspeth.plugins.infrastructure.schema_factory import create_schema_from_config
from elspeth.plugins.infrastructure.telemetry import make_warn_telemetry_before_start
from elspeth.plugins.infrastructure.url_validation import validate_credential_safe_https_url
from elspeth.plugins.transforms.azure.errors import MalformedResponseError
from elspeth.plugins.transforms.safety_utils import get_fields_to_scan
from elspeth.plugins.transforms.safety_utils import validate_fields_not_empty as _validate_fields

logger = structlog.get_logger(__name__)

_CAPACITY_RETRY_INITIAL_DELAY_SECONDS = 0.05
_CAPACITY_RETRY_MAX_DELAY_SECONDS = 1.0
_AZURE_HTTP_TIMEOUT_SECONDS = 30.0


_warn_telemetry_before_start = make_warn_telemetry_before_start(logger)


class BaseAzureSafetyConfig(TransformDataConfig):
    """Shared configuration fields for Azure safety transforms.

    Subclass to add transform-specific fields (thresholds, analysis_type, etc.).
    """

    endpoint: str = Field(..., description="Azure Content Safety endpoint URL")
    api_key: str = Field(..., description="Azure Content Safety API key")
    fields: str | list[str] = Field(
        ...,
        description="Field name(s) to analyze, or 'all' for all string fields",
    )
    max_capacity_retry_seconds: int = Field(3600, gt=0, description="Max seconds to retry capacity errors")
    batch_wait_timeout_seconds: int = Field(3600, gt=0, description="Max seconds to wait for a batch row result")

    @field_validator("endpoint")
    @classmethod
    def _validate_endpoint_url(cls, v: str) -> str:
        return validate_credential_safe_https_url(v, field_name="endpoint")

    @field_validator("api_key")
    @classmethod
    def _reject_empty_api_key(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("api_key must not be empty")
        return v

    @field_validator("fields")
    @classmethod
    def validate_fields_not_empty(cls, v: str | list[str]) -> str | list[str]:
        """Reject empty fields — security transform must scan at least one field."""
        return _validate_fields(v)


class BaseAzureSafetyTransform(BaseTransform, BatchTransformMixin):
    """Base class for Azure safety transforms with shared batch infrastructure.

    Handles:
    - Batch processing lifecycle (connect_output, accept, flush, close)
    - Per-state_id AuditedHTTPClient caching for audit trail integrity
    - Rate limiting, recorder, and telemetry capture
    - Field scanning loop with shared error handling

    Subclasses must:
    - Set class attribute ``name`` (used for rate limiter lookup and plugin registration)
    - Override ``__init__`` to parse their config and call ``super().__init__``
    - Implement ``_analyze_field()`` for API-specific analysis and result checking
    """

    determinism = Determinism.EXTERNAL_CALL
    plugin_version = "1.0.0"
    creates_tokens = False

    API_VERSION = "2024-09-01"

    def __init__(
        self,
        config: dict[str, Any],
        cfg: BaseAzureSafetyConfig,
        schema_name: str,
    ) -> None:
        super().__init__(config)
        self._initialize_declared_input_fields(cfg)

        self._endpoint = cfg.endpoint.rstrip("/")
        self._api_key = cfg.api_key
        self._fields = cfg.fields
        self._max_capacity_retry_seconds = cfg.max_capacity_retry_seconds
        self._batch_wait_timeout_seconds = cfg.batch_wait_timeout_seconds
        self._effective_batch_wait_timeout_seconds = max(
            float(self._batch_wait_timeout_seconds),
            float(self._max_capacity_retry_seconds) + _AZURE_HTTP_TIMEOUT_SECONDS,
        )

        schema = create_schema_from_config(cfg.schema_config, schema_name, allow_coercion=False)
        self.input_schema = schema
        self.output_schema = schema

        self._recorder: PluginAuditWriter | None = None
        self._run_id: str = ""
        self._telemetry_emit: Callable[[Any], None] = _warn_telemetry_before_start
        self._limiter: Any = None  # RateLimiter | NoOpLimiter | None

        self._http_clients: dict[str, Any] = {}  # state_id -> AuditedHTTPClient
        self._http_clients_lock = Lock()
        self._capacity_retry_shutdown = Event()

        self._batch_initialized = False

    def on_start(self, ctx: LifecycleContext) -> None:
        """Capture recorder, telemetry, and rate limit context for pooled execution."""
        super().on_start(ctx)
        self._recorder = ctx.landscape
        self._run_id = ctx.run_id
        self._telemetry_emit = ctx.telemetry_emit
        self._limiter = ctx.rate_limit_registry.get_limiter(self.name) if ctx.rate_limit_registry is not None else None

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
            batch_wait_timeout=self._effective_batch_wait_timeout_seconds,
        )
        self._batch_initialized = True

    def accept(self, row: PipelineRow, ctx: TransformContext) -> None:
        """Accept a row for processing.

        Submits the row to the batch processing pipeline. Returns quickly
        unless backpressure is applied (buffer full). Results flow through
        the output port in FIFO order.

        Args:
            row: Input row with fields to analyze
            ctx: Plugin context with token and landscape

        Raises:
            RuntimeError: If connect_output() was not called
        """
        if not self._batch_initialized:
            raise RuntimeError("connect_output() must be called before accept(). This wires up the output port for result emission.")
        self.accept_row(row, ctx, self._process_row)

    def process(
        self,
        row: PipelineRow,
        ctx: TransformContext,
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

    def _execute_forward_invariant_probe_with_client(
        self,
        probe_rows: list[PipelineRow],
        ctx: TransformContext,
        *,
        client: Any,
    ) -> TransformResult:
        """Drive the single-row validation path with a hermetic HTTP client."""
        if len(probe_rows) != 1:
            raise FrameworkBugError(
                f"{self.__class__.__name__}.execute_forward_invariant_probe() "
                f"received {len(probe_rows)} rows; Azure safety probes require exactly 1 row."
            )

        had_client_override = "_get_http_client" in self.__dict__
        original_get_http_client = self._get_http_client
        state_id = ctx.state_id or "invariant-probe-state"
        token_id = ctx.token.token_id if ctx.token is not None else None

        def _fake_get_http_client(
            requested_state_id: str,
            *,
            token_id: str | None = None,
        ) -> Any:
            del requested_state_id, token_id
            return client

        try:
            self.__dict__["_get_http_client"] = _fake_get_http_client
            return self._process_single_with_state(
                probe_rows[0],
                state_id,
                token_id=token_id,
            )
        finally:
            if had_client_override:
                self.__dict__["_get_http_client"] = original_get_http_client
            else:
                self.__dict__.pop("_get_http_client", None)
            client.close()

    def _process_row(
        self,
        row: PipelineRow,
        ctx: TransformContext,
    ) -> TransformResult:
        """Process a single row. Called by worker threads.

        This is the processor function passed to accept_row(). It runs in
        the BatchTransformMixin's worker pool.
        """
        # Capture state_id at entry — ctx is mutable and shared across retry
        # attempts, so ctx.state_id can change between try and finally if a
        # timeout triggers a retry with a new state_id.
        state_id = ctx.state_id
        if state_id is None:
            raise RuntimeError("state_id is required for batch processing. Ensure transform is executed through the engine.")
        token_id = ctx.token.token_id if ctx.token is not None else None

        try:
            return self._process_single_with_state(row, state_id, token_id=token_id)
        finally:
            # Clean up cached HTTP client for this state_id
            with self._http_clients_lock:
                client = self._http_clients.pop(state_id, None)
            if client is not None:
                client.close()

    def _process_single_with_state(
        self,
        row: PipelineRow,
        state_id: str,
        *,
        token_id: str | None = None,
    ) -> TransformResult:
        """Process a single row with explicit state_id.

        Validates configured fields before any external call, then delegates
        to _analyze_field() for API-specific analysis.
        Handles shared exception types (httpx errors, MalformedResponseError).
        """
        fields_to_scan = get_fields_to_scan(self._fields, row)
        all_mode = self._fields == "all"
        validated_fields: list[tuple[str, str]] = []

        for field_name in fields_to_scan:
            if field_name not in row:
                if all_mode:
                    continue  # "all" mode scans only present string fields
                # Explicitly-configured field is missing — fail CLOSED.
                # Security transform must not report "validated" when configured
                # fields were never analyzed.
                return TransformResult.error(
                    {"reason": "missing_field", "field": field_name},
                    retryable=False,
                )

            value = row[field_name]
            if not isinstance(value, str):
                # Explicitly-configured field is non-string — fail CLOSED.
                # ("all" mode pre-filters to string fields in _get_fields_to_scan,
                # so this branch only fires for explicitly-configured fields.)
                return TransformResult.error(
                    {
                        "reason": "non_string_field",
                        "field": field_name,
                        "actual_type": type(value).__name__,
                    },
                    retryable=False,
                )

            validated_fields.append((field_name, value))

        capacity_retry_started_at = time.monotonic()
        capacity_retry_deadline = capacity_retry_started_at + float(self._max_capacity_retry_seconds)

        for field_name, value in validated_fields:
            violation = self._analyze_field_with_capacity_retry(
                value,
                field_name,
                state_id,
                token_id=token_id,
                retry_started_at=capacity_retry_started_at,
                retry_deadline=capacity_retry_deadline,
            )
            if violation is not None:
                return violation

        return TransformResult.success(
            self._align_output_row_contract(row),
            success_reason={"action": "validated"},
        )

    def _analyze_field_once(
        self,
        value: str,
        field_name: str,
        state_id: str,
        *,
        token_id: str | None = None,
    ) -> TransformResult | None:
        """Run one Azure field analysis attempt and classify non-capacity failures."""
        try:
            return self._analyze_field(value, field_name, state_id, token_id=token_id)
        except httpx.HTTPStatusError as e:
            status_code = e.response.status_code
            if is_capacity_error(status_code):
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
            # Network errors are transient — re-raise for engine retry.
            # TransformResult.error(retryable=True) was dead code: the engine
            # retries by catching PluginRetryableError, not by inspecting results.
            raise PluginRetryableError(f"Azure API network error: {e}", retryable=True) from e
        except MalformedResponseError as e:
            # Malformed API response — fail CLOSED, non-retryable
            # (response structure won't improve on retry)
            return TransformResult.error(
                {
                    "reason": "api_error",
                    "error_type": "malformed_response",
                    "message": str(e),
                },
                retryable=False,
            )

    def _analyze_field_with_capacity_retry(
        self,
        value: str,
        field_name: str,
        state_id: str,
        *,
        token_id: str | None = None,
        retry_started_at: float | None = None,
        retry_deadline: float | None = None,
    ) -> TransformResult | None:
        """Retry Azure capacity responses within the row-scoped retry budget."""
        started_at = retry_started_at if retry_started_at is not None else time.monotonic()
        deadline = retry_deadline if retry_deadline is not None else started_at + float(self._max_capacity_retry_seconds)
        retry_delay = _CAPACITY_RETRY_INITIAL_DELAY_SECONDS

        while True:
            if self._capacity_retry_shutdown.is_set():
                return self._capacity_retry_shutdown_result(started_at)

            try:
                return self._analyze_field_once(value, field_name, state_id, token_id=token_id)
            except CapacityError as e:
                if self._capacity_retry_shutdown.is_set():
                    return self._capacity_retry_shutdown_result(started_at)

                now = time.monotonic()
                if now >= deadline:
                    return self._capacity_retry_timeout_result(e, started_at)

                sleep_seconds = min(retry_delay, deadline - now)
                if sleep_seconds > 0 and self._capacity_retry_shutdown.wait(timeout=sleep_seconds):
                    return self._capacity_retry_shutdown_result(started_at)
                if time.monotonic() >= deadline:
                    return self._capacity_retry_timeout_result(e, started_at)

                retry_delay = min(retry_delay * 2.0, _CAPACITY_RETRY_MAX_DELAY_SECONDS)

    def _capacity_retry_shutdown_result(self, started_at: float) -> TransformResult:
        """Build a row result for transform shutdown during capacity backoff."""
        elapsed = time.monotonic() - started_at
        return TransformResult.error(
            {
                "reason": "shutdown_requested",
                "error": "Azure safety capacity retry stopped because transform shutdown was requested",
                "elapsed_seconds": elapsed,
                "max_seconds": self._max_capacity_retry_seconds,
            },
            retryable=False,
        )

    def _capacity_retry_timeout_result(
        self,
        capacity_error: CapacityError,
        started_at: float,
    ) -> TransformResult:
        """Build the same retry_timeout row result shape as the pooled executor."""
        elapsed = time.monotonic() - started_at
        return TransformResult.error(
            {
                "reason": "retry_timeout",
                "error": str(capacity_error),
                "error_type": type(capacity_error).__name__,
                "elapsed_seconds": elapsed,
                "max_seconds": self._max_capacity_retry_seconds,
                "status_code": capacity_error.status_code,
            },
            retryable=False,
        )

    def _analyze_field(
        self,
        value: str,
        field_name: str,
        state_id: str,
        *,
        token_id: str | None = None,
    ) -> TransformResult | None:
        """Analyze a single field value. Subclasses must implement.

        Returns TransformResult.error if the field fails analysis, or None
        if it passes. May raise httpx.HTTPStatusError, httpx.RequestError,
        or MalformedResponseError — these are handled by the base class's
        _process_single_with_state().
        """
        raise NotImplementedError

    def _parse_external_json_response(self, response: Any, *, label: str) -> Any:
        """Parse an Azure JSON response body with the Tier-3 strict parser."""
        text = response.text
        if type(text) is not str:
            raise MalformedResponseError(f"{label} body must be text, got {type(text).__name__}")
        try:
            data, parse_error = parse_json_strict(text)
        except RecursionError as exc:
            raise MalformedResponseError(f"Invalid JSON in {label}: {exc}") from exc
        if parse_error is not None:
            raise MalformedResponseError(f"Invalid JSON in {label}: {parse_error}")
        return data

    def _get_http_client(self, state_id: str, *, token_id: str | None = None) -> Any:
        """Get or create audited HTTP client for a state_id.

        Clients are cached to preserve call_index across retries.
        This ensures uniqueness of (state_id, call_index) when the local
        capacity retry loop retries after CapacityError.

        Thread-safe: multiple workers can call this concurrently.
        """
        from elspeth.plugins.infrastructure.clients.http import AuditedHTTPClient

        with self._http_clients_lock:
            if state_id not in self._http_clients:
                if self._recorder is None:
                    raise RuntimeError(f"{self.name}: recorder not initialized — call on_start() before processing")
                self._http_clients[state_id] = AuditedHTTPClient(
                    execution=self._recorder,
                    state_id=state_id,
                    run_id=self._run_id,
                    telemetry_emit=self._telemetry_emit,
                    timeout=_AZURE_HTTP_TIMEOUT_SECONDS,
                    headers={
                        "Ocp-Apim-Subscription-Key": self._api_key,
                        "Content-Type": "application/json",
                    },
                    limiter=self._limiter,
                    token_id=token_id,
                )
            return self._http_clients[state_id]

    def close(self) -> None:
        """Release resources."""
        self._capacity_retry_shutdown.set()

        # Shutdown batch processing infrastructure first
        if self._batch_initialized:
            self.shutdown_batch_processing()

        # Close all cached HTTP clients
        with self._http_clients_lock:
            for client in self._http_clients.values():
                client.close()
            self._http_clients.clear()

        self._recorder = None
