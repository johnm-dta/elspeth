"""Azure Content Safety transform for content moderation.

This module provides the AzureContentSafety transform which uses Azure's
Content Safety API to analyze text for harmful content categories:
- Hate speech
- Violence
- Sexual content
- Self-harm

Content is flagged when severity scores exceed configured thresholds.

Uses BatchTransformMixin for row-level pipelining (multiple rows in flight
with FIFO output ordering).
"""

from __future__ import annotations

from collections.abc import Callable
from threading import Lock
from typing import TYPE_CHECKING, Any

import httpx
from pydantic import BaseModel, Field

from elspeth.contracts import Determinism
from elspeth.contracts.plugin_context import PluginContext
from elspeth.contracts.schema_contract import PipelineRow
from elspeth.plugins.base import BaseTransform
from elspeth.plugins.batching import BatchTransformMixin, OutputPort
from elspeth.plugins.config_base import TransformDataConfig
from elspeth.plugins.pooling import CapacityError, is_capacity_error
from elspeth.plugins.results import TransformResult
from elspeth.plugins.schema_factory import create_schema_from_config
from elspeth.plugins.transforms.azure.errors import MalformedResponseError

if TYPE_CHECKING:
    from elspeth.core.landscape.recorder import LandscapeRecorder


class ContentSafetyThresholds(BaseModel):
    """Per-category severity thresholds for Azure Content Safety.

    Azure Content Safety returns severity scores from 0-6 for each category.
    Content is flagged when its severity exceeds the configured threshold.

    A threshold of 0 means all content of that type is blocked.
    A threshold of 6 means only the most severe content is blocked.
    """

    model_config = {"extra": "forbid"}

    hate: int = Field(..., ge=0, le=6, description="Hate content threshold (0-6)")
    violence: int = Field(..., ge=0, le=6, description="Violence content threshold (0-6)")
    sexual: int = Field(..., ge=0, le=6, description="Sexual content threshold (0-6)")
    self_harm: int = Field(..., ge=0, le=6, description="Self-harm content threshold (0-6)")


class AzureContentSafetyConfig(TransformDataConfig):
    """Configuration for Azure Content Safety transform.

    Requires:
        endpoint: Azure Content Safety endpoint URL
        api_key: Azure Content Safety API key
        fields: Field name(s) to analyze, or 'all' for all string fields
        thresholds: Per-category severity thresholds (0-6)
        schema: Schema configuration

    Optional:
        max_capacity_retry_seconds: Timeout for capacity error retries (default 3600)

    Example YAML:
        transforms:
          - plugin: azure_content_safety
            options:
              endpoint: https://my-resource.cognitiveservices.azure.com
              api_key: ${AZURE_CONTENT_SAFETY_KEY}
              fields: [content, title]
              thresholds:
                hate: 2
                violence: 2
                sexual: 2
                self_harm: 0
              on_error: quarantine_sink
              schema:
                mode: observed
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

    # Batch processing timeout
    max_capacity_retry_seconds: int = Field(3600, gt=0, description="Max seconds to retry capacity errors")


# Rebuild model to resolve nested model references
AzureContentSafetyConfig.model_rebuild()


# Explicit mapping from Azure Content Safety API category names to internal names.
# This is the ONLY place where Azure category names are translated.
# Unknown categories are REJECTED (fail closed) — see _analyze_content().
_AZURE_CATEGORY_MAP: dict[str, str] = {
    "Hate": "hate",
    "Violence": "violence",
    "Sexual": "sexual",
    "SelfHarm": "self_harm",
}


class AzureContentSafety(BaseTransform, BatchTransformMixin):
    """Analyze content using Azure Content Safety API.

    Checks text against Azure's moderation categories (hate, violence,
    sexual, self-harm) and blocks content exceeding configured thresholds.

    Uses BatchTransformMixin for row-level pipelining: multiple rows can be
    in flight concurrently with FIFO output ordering.

    Architecture:
        Orchestrator → accept() → [RowReorderBuffer] → [Worker Pool]
            → _process_row() → Azure API
            → emit() → OutputPort (sink or next transform)

    Usage:
        # 1. Instantiate
        transform = AzureContentSafety(config)

        # 2. Connect output port (required before accept())
        transform.connect_output(output_port, max_pending=30)

        # 3. Feed rows (blocks on backpressure)
        for row in source:
            transform.accept(row, ctx)

        # 4. Flush and close
        transform.flush_batch_processing()
        transform.close()
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
        self._max_capacity_retry_seconds = cfg.max_capacity_retry_seconds

        schema = create_schema_from_config(
            cfg.schema_config,
            "AzureContentSafetySchema",
            allow_coercion=False,
        )
        self.input_schema = schema
        # BatchTransformMixin processes rows individually, output schema matches input
        self.output_schema = schema

        # Recorder reference for pooled execution (set in on_start or first accept)
        self._recorder: LandscapeRecorder | None = None
        self._run_id: str = ""
        self._telemetry_emit: Callable[[Any], None] = lambda event: None
        self._limiter: Any = None  # RateLimiter | NoOpLimiter | None

        # Per-state_id HTTP client cache - ensures call_index uniqueness across retries
        # Each state_id gets its own AuditedHTTPClient with monotonically increasing call indices
        self._http_clients: dict[str, Any] = {}  # AuditedHTTPClient instances
        self._http_clients_lock = Lock()

        # Batch processing state (initialized by connect_output)
        self._batch_initialized = False

    def on_start(self, ctx: PluginContext) -> None:
        """Capture recorder, telemetry, and rate limit context for pooled execution."""
        self._recorder = ctx.landscape
        self._run_id = ctx.run_id
        self._telemetry_emit = ctx.telemetry_emit
        # Get rate limiter for Azure Content Safety service (None if rate limiting disabled)
        self._limiter = ctx.rate_limit_registry.get_limiter("azure_content_safety") if ctx.rate_limit_registry is not None else None

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

    def accept(self, row: PipelineRow, ctx: PluginContext) -> None:
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
        row: PipelineRow,
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
        row: PipelineRow,
        ctx: PluginContext,
    ) -> TransformResult:
        """Process a single row. Called by worker threads.

        This is the processor function passed to accept_row(). It runs in
        the BatchTransformMixin's worker pool.

        Args:
            row: Input row with fields to analyze
            ctx: Plugin context with state_id for audit trail

        Returns:
            TransformResult indicating success or content violation
        """
        if ctx.state_id is None:
            raise RuntimeError("state_id is required for batch processing. Ensure transform is executed through the engine.")
        token_id = ctx.token.token_id if ctx.token is not None else None

        try:
            return self._process_single_with_state(row, ctx.state_id, token_id=token_id)
        finally:
            # Clean up cached HTTP client for this state_id
            with self._http_clients_lock:
                if ctx.state_id in self._http_clients:
                    client = self._http_clients.pop(ctx.state_id)
                else:
                    client = None
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

        Called by _process_row() in the BatchTransformMixin worker pool.

        Raises:
            CapacityError: On rate limit errors (for worker pool retry)
        """
        fields_to_scan = self._get_fields_to_scan(row)
        all_mode = self._fields == "all"

        for field_name in fields_to_scan:
            if field_name not in row:
                if all_mode:
                    continue  # "all" mode scans only present string fields
                # Explicitly-configured field is missing — fail CLOSED.
                # Security transform must not report "validated" when configured
                # fields were never analyzed.
                return TransformResult.error(
                    {
                        "reason": "missing_field",
                        "field": field_name,
                    },
                    retryable=False,
                )

            value = row[field_name]
            if not isinstance(value, str):
                # Explicitly-configured field is non-string — fail CLOSED.
                # Security transform cannot analyze non-string content for safety.
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

            # Call Azure API with state_id for audit trail
            try:
                analysis = self._analyze_content(value, state_id, token_id=token_id)
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
            except ValueError as e:
                # Unknown category from Azure — fail CLOSED (security transform).
                # Not retryable: the API response is structurally valid but contains
                # categories we can't assess. Requires code update to handle.
                return TransformResult.error(
                    {
                        "reason": "unknown_category",
                        "field": field_name,
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

            # Check thresholds
            violation = self._check_thresholds(analysis)
            if violation:
                return TransformResult.error(
                    {
                        "reason": "content_safety_violation",
                        "field": field_name,
                        "categories": violation,
                    },
                    retryable=False,
                )

        return TransformResult.success(
            row,
            success_reason={"action": "validated"},
        )

    def _get_fields_to_scan(self, row: PipelineRow) -> list[str]:
        """Determine which fields to scan based on config."""
        if self._fields == "all":
            return [field_name for field_name in row if isinstance(row[field_name], str)]
        elif isinstance(self._fields, str):
            return [self._fields]
        else:
            return self._fields

    def _get_http_client(self, state_id: str, *, token_id: str | None = None) -> Any:  # Returns AuditedHTTPClient
        """Get or create audited HTTP client for a state_id.

        Clients are cached to preserve call_index across retries.
        This ensures uniqueness of (state_id, call_index) even when
        the worker pool retries after CapacityError.

        Thread-safe: multiple workers can call this concurrently.

        Args:
            state_id: Node state ID for audit linkage

        Returns:
            AuditedHTTPClient instance for this state_id
        """
        from elspeth.plugins.clients.http import AuditedHTTPClient

        with self._http_clients_lock:
            if state_id not in self._http_clients:
                if self._recorder is None:
                    raise RuntimeError("ContentSafety requires recorder. Ensure on_start was called.")
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
                    token_id=token_id,
                )
            return self._http_clients[state_id]

    def _analyze_content(
        self,
        text: str,
        state_id: str,
        *,
        token_id: str | None = None,
    ) -> dict[str, int]:
        """Call Azure Content Safety API.

        Args:
            text: Text to analyze for content safety
            state_id: State ID for audit trail recording

        Returns dict with category -> severity mapping.

        Uses AuditedHTTPClient for automatic audit recording and telemetry emission.
        """
        # Use AuditedHTTPClient - handles recording and telemetry automatically
        http_client = self._get_http_client(state_id, token_id=token_id)

        url = f"{self._endpoint}/contentsafety/text:analyze?api-version={self.API_VERSION}"

        # Make HTTP call - AuditedHTTPClient records to Landscape and emits telemetry
        response = http_client.post(url, json={"text": text})
        response.raise_for_status()

        # Parse response into category -> severity mapping
        # Azure API responses are external data (Tier 3: Zero Trust) — validate immediately
        try:
            data = response.json()
        except Exception as e:
            raise MalformedResponseError(f"Invalid JSON in Content Safety response: {e}") from e

        try:
            # Initialize all expected categories to 0 (safe)
            _EXPECTED_CATEGORIES = {"hate", "violence", "sexual", "self_harm"}
            result: dict[str, int] = dict.fromkeys(_EXPECTED_CATEGORIES, 0)

            for item in data["categoriesAnalysis"]:
                azure_category = item["category"]
                internal_name = _AZURE_CATEGORY_MAP.get(azure_category)
                if internal_name is None:
                    # Fail CLOSED: unknown category means Azure updated their taxonomy.
                    # We cannot assess content safety with unknown categories — reject.
                    raise ValueError(
                        f"Unknown Azure Content Safety category: {azure_category!r}. "
                        f"Known categories: {sorted(_AZURE_CATEGORY_MAP.keys())}. "
                        f"Update _AZURE_CATEGORY_MAP to handle this category."
                    )
                if not isinstance(item["severity"], int):
                    raise MalformedResponseError(f"severity for {azure_category!r} must be int, got {type(item['severity']).__name__}")
                result[internal_name] = item["severity"]

            # Fail CLOSED: verify all expected categories were returned by Azure.
            # If Azure changes to only returning flagged categories, absent ones
            # would silently default to 0 (safe) — that's a fail-open path.
            returned_categories = {
                _AZURE_CATEGORY_MAP[item["category"]] for item in data["categoriesAnalysis"] if item["category"] in _AZURE_CATEGORY_MAP
            }
            missing = _EXPECTED_CATEGORIES - returned_categories
            if missing:
                raise MalformedResponseError(
                    f"Azure Content Safety response missing expected categories: "
                    f"{sorted(missing)}. Returned: {sorted(returned_categories)}. "
                    f"Cannot assess content safety without all categories."
                )

            return result

        except (KeyError, TypeError) as e:
            # Malformed response structure — non-retryable
            raise MalformedResponseError(f"Malformed Content Safety response: {e}") from e

    def _check_thresholds(
        self,
        analysis: dict[str, int],
    ) -> dict[str, dict[str, Any]] | None:
        """Check if any category exceeds its threshold.

        Args:
            analysis: Category -> severity mapping from _analyze_content.
                      All 4 categories are guaranteed to be present (defaults applied at boundary).
        """
        categories: dict[str, dict[str, Any]] = {
            "hate": {
                "severity": analysis["hate"],
                "threshold": self._thresholds.hate,
            },
            "violence": {
                "severity": analysis["violence"],
                "threshold": self._thresholds.violence,
            },
            "sexual": {
                "severity": analysis["sexual"],
                "threshold": self._thresholds.sexual,
            },
            "self_harm": {
                "severity": analysis["self_harm"],
                "threshold": self._thresholds.self_harm,
            },
        }

        for info in categories.values():
            info["exceeded"] = info["severity"] > info["threshold"]

        if any(info["exceeded"] for info in categories.values()):
            return categories
        return None

    def close(self) -> None:
        """Release resources."""
        # Shutdown batch processing infrastructure first
        if self._batch_initialized:
            self.shutdown_batch_processing()

        # Close all cached HTTP clients
        with self._http_clients_lock:
            for client in self._http_clients.values():
                client.close()
            self._http_clients.clear()

        self._recorder = None
