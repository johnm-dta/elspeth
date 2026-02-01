# src/elspeth/plugins/llm/azure_batch.py
"""Azure OpenAI Batch API transform - 50% cost savings for high volume.

Uses two-phase checkpoint approach:
1. SUBMIT: Render templates, build JSONL, upload, submit batch, checkpoint batch_id
2. CHECK/COMPLETE: Resume with checkpoint, check status, download results or wait

Benefits:
- 50% cost reduction vs real-time API
- Crash recovery via checkpointed batch_id
- Resource efficiency (no blocking waits)

Requires Azure OpenAI Batch API (GA as of Azure API version 2024-10-21).
"""

from __future__ import annotations

import io
import json
import time
import uuid
from datetime import UTC, datetime
from typing import Any

from pydantic import Field

from elspeth.contracts import BatchPendingError, CallStatus, CallType, Determinism, RowErrorEntry, TransformErrorReason, TransformResult
from elspeth.contracts.schema import SchemaConfig
from elspeth.plugins.base import BaseTransform
from elspeth.plugins.config_base import TransformDataConfig
from elspeth.plugins.context import PluginContext
from elspeth.plugins.llm import get_llm_audit_fields, get_llm_guaranteed_fields
from elspeth.plugins.llm.templates import PromptTemplate, TemplateError
from elspeth.plugins.schema_factory import create_schema_from_config


class AzureBatchConfig(TransformDataConfig):
    """Azure Batch API-specific configuration.

    Extends TransformDataConfig with Azure-specific settings for batch processing.

    Required fields:
        deployment_name: Azure deployment name (used as model identifier)
        endpoint: Azure OpenAI endpoint URL
        api_key: Azure OpenAI API key
        template: Jinja2 prompt template

    Optional fields:
        api_version: Azure API version (default: 2024-10-21)
        system_prompt: Optional system message
        temperature: Sampling temperature (default: 0.0 for determinism)
        max_tokens: Maximum response tokens
        response_field: Output field name for LLM response (default: llm_response)
        poll_interval_seconds: Batch status check interval (default: 300s)
        max_wait_hours: Maximum batch wait time (default: 24h)
        on_error: Sink for failed rows (optional)
    """

    deployment_name: str = Field(..., description="Azure deployment name")
    endpoint: str = Field(..., description="Azure OpenAI endpoint URL")
    api_key: str = Field(..., description="Azure OpenAI API key")
    api_version: str = Field(default="2024-10-21", description="Azure API version")

    template: str = Field(..., description="Jinja2 prompt template")
    system_prompt: str | None = Field(None, description="Optional system prompt")
    temperature: float = Field(0.0, ge=0.0, le=2.0, description="Sampling temperature")
    max_tokens: int | None = Field(None, gt=0, description="Maximum response tokens")
    response_field: str = Field("llm_response", description="Output field name for LLM response")

    poll_interval_seconds: int = Field(300, ge=1, description="Batch status check interval in seconds")
    max_wait_hours: int = Field(24, ge=1, le=24, description="Maximum batch wait time in hours")

    # Template metadata fields for audit trail (matching LLMConfig)
    lookup: dict[str, Any] | None = Field(None, description="Lookup data loaded from YAML file")
    template_source: str | None = Field(None, description="Template file path for audit (None if inline)")
    lookup_source: str | None = Field(None, description="Lookup file path for audit (None if no lookup)")
    system_prompt_source: str | None = Field(None, description="System prompt file path for audit (None if inline)")


class AzureBatchLLMTransform(BaseTransform):
    """Batch LLM transform using Azure OpenAI Batch API.

    Uses two-phase checkpoint approach for crash recovery:

    Phase 1 (Submit):
        1. Render all templates for input rows
        2. Build JSONL request file
        3. Upload to Azure Blob storage
        4. Create batch job
        5. CHECKPOINT: Save batch_id immediately
        6. Raise BatchPendingError("submitted")

    Phase 2 (Check/Complete):
        1. Check checkpoint - if batch_id exists, resume
        2. Check Azure batch status:
           - "completed" -> Download results, return success
           - "failed" -> Return TransformResult.error()
           - "in_progress" -> Raise BatchPendingError again

    Benefits:
    - 50% cost reduction vs real-time API
    - Crash recovery via checkpointed batch_id
    - Resource efficiency (no blocking waits)

    Configuration example:
        transforms:
          - plugin: azure_batch_llm
            options:
              deployment_name: "gpt-4o-batch"
              endpoint: "${AZURE_OPENAI_ENDPOINT}"
              api_key: "${AZURE_OPENAI_KEY}"
              template: |
                Analyze: {{ row.text }}
              schema:
                fields: dynamic
              poll_interval_seconds: 300
    """

    name = "azure_batch_llm"
    is_batch_aware = True  # Engine passes list[dict] for batch processing

    # LLM transforms are non-deterministic by nature
    determinism: Determinism = Determinism.NON_DETERMINISTIC

    def __init__(self, config: dict[str, Any]) -> None:
        """Initialize Azure Batch LLM transform.

        Args:
            config: Transform configuration dictionary
        """
        super().__init__(config)

        cfg = AzureBatchConfig.from_dict(config)
        self._deployment_name = cfg.deployment_name
        self._endpoint = cfg.endpoint.rstrip("/")
        self._api_key = cfg.api_key
        self._api_version = cfg.api_version
        self._template = PromptTemplate(
            cfg.template,
            template_source=cfg.template_source,
            lookup_data=cfg.lookup,
            lookup_source=cfg.lookup_source,
        )
        self._system_prompt = cfg.system_prompt
        self._system_prompt_source = cfg.system_prompt_source
        self._temperature = cfg.temperature
        self._max_tokens = cfg.max_tokens
        self._response_field = cfg.response_field
        self._poll_interval = cfg.poll_interval_seconds
        self._max_wait_hours = cfg.max_wait_hours

        # Error routing - required for TransformResult.error() to work
        self._on_error = cfg.on_error

        # Schema from config (TransformDataConfig guarantees schema_config is not None)
        schema_config = cfg.schema_config
        schema = create_schema_from_config(
            schema_config,
            "AzureBatchSchema",
            allow_coercion=False,  # Transforms do NOT coerce
        )
        self.input_schema = schema
        self.output_schema = schema

        # Build output schema config with field categorization
        guaranteed = get_llm_guaranteed_fields(self._response_field)
        audit = get_llm_audit_fields(self._response_field)

        # Merge with any existing fields from base schema
        base_guaranteed = schema_config.guaranteed_fields or ()
        base_audit = schema_config.audit_fields or ()

        self._output_schema_config = SchemaConfig(
            mode=schema_config.mode,
            fields=schema_config.fields,
            is_dynamic=schema_config.is_dynamic,
            guaranteed_fields=tuple(set(base_guaranteed) | set(guaranteed)),
            audit_fields=tuple(set(base_audit) | set(audit)),
            required_fields=schema_config.required_fields,
        )

        # Azure OpenAI client (lazy init)
        self._client: Any = None

    def _get_client(self) -> Any:
        """Lazy-initialize Azure OpenAI client.

        Returns:
            openai.AzureOpenAI client instance
        """
        if self._client is None:
            from openai import AzureOpenAI

            self._client = AzureOpenAI(
                azure_endpoint=self._endpoint,
                api_key=self._api_key,
                api_version=self._api_version,
            )
        return self._client

    def process(
        self,
        row: dict[str, Any] | list[dict[str, Any]],
        ctx: PluginContext,
    ) -> TransformResult:
        """Process batch with checkpoint-based recovery.

        When is_batch_aware=True, the engine passes list[dict].
        For single-row fallback, the engine passes dict.

        Args:
            row: Single row dict OR list of row dicts (batch)
            ctx: Plugin context with checkpoint support

        Returns:
            TransformResult with processed rows or error

        Raises:
            BatchPendingError: When batch is submitted but not complete
        """
        if isinstance(row, list):
            return self._process_batch(row, ctx)
        else:
            # Single row fallback - wrap in list and process
            return self._process_single(row, ctx)

    def _process_single(
        self,
        row: dict[str, Any],
        ctx: PluginContext,
    ) -> TransformResult:
        """Process a single row (fallback for non-batch mode).

        For single rows, we still use the batch API but with just one request.
        This ensures consistent behavior and audit trail.

        Args:
            row: Single row dict
            ctx: Plugin context

        Returns:
            TransformResult with processed row
        """
        result = self._process_batch([row], ctx)

        # Convert multi-row result back to single-row
        if result.status == "success" and result.rows:
            # Propagate success_reason from batch result
            return TransformResult.success(
                result.rows[0],
                success_reason=result.success_reason or {"action": "enriched", "fields_added": [self._response_field]},
            )
        elif result.status == "error":
            return result
        else:
            # Empty rows from empty batch - shouldn't happen for single row
            return TransformResult.success(
                row,
                success_reason={"action": "passthrough"},
            )

    def _process_batch(
        self,
        rows: list[dict[str, Any]],
        ctx: PluginContext,
    ) -> TransformResult:
        """Process batch with two-phase checkpoint approach.

        Phase 1: Submit fresh batch
        Phase 2: Resume with checkpoint, check status

        Args:
            rows: List of row dicts to process
            ctx: Plugin context with checkpoint support

        Returns:
            TransformResult with all processed rows

        Raises:
            BatchPendingError: When batch is pending completion
        """
        if not rows:
            # Engine invariant: AggregationExecutor.execute_flush() guards against empty buffers.
            # If we reach here, something bypassed that guard - this is a bug, not a valid state.
            # Per CLAUDE.md "crash on plugin bugs" principle, fail fast rather than emit garbage.
            raise RuntimeError(
                f"Empty batch passed to batch-aware transform '{self.name}'. "
                f"This should never happen - AggregationExecutor.execute_flush() guards against "
                f"empty buffers. This indicates a bug in the engine or test setup."
            )

        # Check checkpoint for resume
        checkpoint = self._get_checkpoint(ctx)

        if checkpoint is not None:
            if "batch_id" not in checkpoint:
                raise RuntimeError(
                    f"Checkpoint missing required 'batch_id' for AzureBatchLLMTransform. Checkpoint keys: {sorted(checkpoint.keys())}"
                )
            # PHASE 2: Resume - batch already submitted
            return self._check_batch_status(checkpoint, rows, ctx)

        # PHASE 1: Fresh batch - submit new
        return self._submit_batch(rows, ctx)

    def _get_checkpoint(self, ctx: PluginContext) -> dict[str, Any] | None:
        """Get checkpoint state from context.

        Args:
            ctx: Plugin context

        Returns:
            Checkpoint dict or None if no checkpoint

        Raises:
            RuntimeError: If checkpoint API is not available on context
        """
        if not hasattr(ctx, "get_checkpoint"):
            raise RuntimeError(
                "AzureBatchLLMTransform requires checkpoint API on PluginContext. "
                "Ensure engine provides get_checkpoint/update_checkpoint/clear_checkpoint methods."
            )
        return ctx.get_checkpoint()  # type: ignore[no-any-return]

    def _update_checkpoint(self, ctx: PluginContext, data: dict[str, Any]) -> None:
        """Update checkpoint state.

        Args:
            ctx: Plugin context
            data: Checkpoint data to save

        Raises:
            RuntimeError: If checkpoint API is not available on context
        """
        if not hasattr(ctx, "update_checkpoint"):
            raise RuntimeError(
                "AzureBatchLLMTransform requires checkpoint API on PluginContext. "
                "Ensure engine provides get_checkpoint/update_checkpoint/clear_checkpoint methods."
            )
        ctx.update_checkpoint(data)

    def _clear_checkpoint(self, ctx: PluginContext) -> None:
        """Clear checkpoint state.

        Args:
            ctx: Plugin context

        Raises:
            RuntimeError: If checkpoint API is not available on context
        """
        if not hasattr(ctx, "clear_checkpoint"):
            raise RuntimeError(
                "AzureBatchLLMTransform requires checkpoint API on PluginContext. "
                "Ensure engine provides get_checkpoint/update_checkpoint/clear_checkpoint methods."
            )
        ctx.clear_checkpoint()

    def _submit_batch(
        self,
        rows: list[dict[str, Any]],
        ctx: PluginContext,
    ) -> TransformResult:
        """Submit new batch to Azure Batch API.

        Args:
            rows: Rows to process
            ctx: Plugin context

        Returns:
            Never returns - always raises BatchPendingError

        Raises:
            BatchPendingError: After successful submission
        """
        # 1. Render templates for all rows, track failures
        requests: list[dict[str, Any]] = []
        row_mapping: dict[str, dict[str, Any]] = {}  # custom_id -> {index, variables_hash}
        template_errors: list[tuple[int, str]] = []  # (index, error)

        for idx, row in enumerate(rows):
            custom_id = f"row-{idx}-{uuid.uuid4().hex[:8]}"

            try:
                rendered = self._template.render_with_metadata(row)
            except TemplateError as e:
                template_errors.append((idx, str(e)))
                continue

            # Build messages
            messages: list[dict[str, str]] = []
            if self._system_prompt:
                messages.append({"role": "system", "content": self._system_prompt})
            messages.append({"role": "user", "content": rendered.prompt})

            # Build batch request in Azure Batch API format
            request: dict[str, Any] = {
                "custom_id": custom_id,
                "method": "POST",
                "url": "/chat/completions",
                "body": {
                    "model": self._deployment_name,
                    "messages": messages,
                    "temperature": self._temperature,
                },
            }
            if self._max_tokens is not None:
                request["body"]["max_tokens"] = self._max_tokens

            requests.append(request)
            row_mapping[custom_id] = {
                "index": idx,
                "variables_hash": rendered.variables_hash,
            }

        # Build request lookup for audit recording (custom_id -> request body)
        # This allows the audit trail to record exactly what was sent to the LLM
        requests_by_id = {req["custom_id"]: req["body"] for req in requests}

        # If ALL rows failed template rendering, return error immediately
        if not requests:
            return TransformResult.error(
                {
                    "reason": "all_templates_failed",
                    "template_errors": [{"row_index": idx, "error": err} for idx, err in template_errors],
                }
            )

        # 2. Build JSONL content
        jsonl_content = "\n".join(json.dumps(req) for req in requests)

        # 3. Upload and create batch
        client = self._get_client()

        # Upload JSONL file (with audit recording)
        file_bytes = io.BytesIO(jsonl_content.encode("utf-8"))
        upload_request = {
            "operation": "files.create",
            "filename": "batch_input.jsonl",
            "purpose": "batch",
            "content": jsonl_content,  # BUG-AZURE-01 FIX: Include actual JSONL content
            "content_size": len(jsonl_content),
        }
        start = time.perf_counter()
        try:
            batch_file = client.files.create(
                file=("batch_input.jsonl", file_bytes),
                purpose="batch",
            )
            ctx.record_call(
                call_type=CallType.HTTP,
                status=CallStatus.SUCCESS,
                request_data=upload_request,
                response_data={"file_id": batch_file.id, "status": batch_file.status},
                latency_ms=(time.perf_counter() - start) * 1000,
                provider="azure",
            )
        except Exception as e:
            # External API failure - record error and return structured result
            ctx.record_call(
                call_type=CallType.HTTP,
                status=CallStatus.ERROR,
                request_data=upload_request,
                response_data={"error": str(e), "error_type": type(e).__name__},
                latency_ms=(time.perf_counter() - start) * 1000,
                provider="azure",
            )
            return TransformResult.error(
                {
                    "reason": "file_upload_failed",
                    "error": str(e),
                    "error_type": type(e).__name__,
                },
                retryable=True,  # Network/auth errors are retryable
            )

        # Create batch job (with audit recording)
        batch_request = {
            "operation": "batches.create",
            "input_file_id": batch_file.id,
            "endpoint": "/chat/completions",
            "completion_window": "24h",
        }
        start = time.perf_counter()
        try:
            batch = client.batches.create(
                input_file_id=batch_file.id,
                endpoint="/chat/completions",
                completion_window="24h",
            )
            ctx.record_call(
                call_type=CallType.HTTP,
                status=CallStatus.SUCCESS,
                request_data=batch_request,
                response_data={"batch_id": batch.id, "status": batch.status},
                latency_ms=(time.perf_counter() - start) * 1000,
                provider="azure",
            )
        except Exception as e:
            # External API failure - record error and return structured result
            ctx.record_call(
                call_type=CallType.HTTP,
                status=CallStatus.ERROR,
                request_data=batch_request,
                response_data={"error": str(e), "error_type": type(e).__name__},
                latency_ms=(time.perf_counter() - start) * 1000,
                provider="azure",
            )
            return TransformResult.error(
                {
                    "reason": "batch_create_failed",
                    "error": str(e),
                    "error_type": type(e).__name__,
                },
                retryable=True,  # Network/auth errors are retryable
            )

        # 4. CHECKPOINT immediately after submit
        checkpoint_data = {
            "batch_id": batch.id,
            "input_file_id": batch_file.id,
            "row_mapping": row_mapping,
            "template_errors": template_errors,
            "submitted_at": datetime.now(UTC).isoformat(),
            "row_count": len(rows),
            "requests": requests_by_id,  # BUG-AZURE-01: Store original requests for audit recording
        }
        self._update_checkpoint(ctx, checkpoint_data)

        # 5. Raise BatchPendingError - engine handles retry scheduling
        # Include checkpoint and node_id so caller can persist and restore
        raise BatchPendingError(
            batch.id,
            "submitted",
            check_after_seconds=self._poll_interval,
            checkpoint=checkpoint_data,
            node_id=self.node_id,
        )

    def _check_batch_status(
        self,
        checkpoint: dict[str, Any],
        rows: list[dict[str, Any]],
        ctx: PluginContext,
    ) -> TransformResult:
        """Check batch status and complete if ready.

        Args:
            checkpoint: Checkpoint data with batch_id
            rows: Original rows (needed for output assembly)
            ctx: Plugin context

        Returns:
            TransformResult with results if complete

        Raises:
            BatchPendingError: If batch still in progress
        """
        batch_id = checkpoint["batch_id"]
        client = self._get_client()

        # Check batch status (with audit recording)
        retrieve_request = {
            "operation": "batches.retrieve",
            "batch_id": batch_id,
        }
        start = time.perf_counter()
        try:
            batch = client.batches.retrieve(batch_id)
            ctx.record_call(
                call_type=CallType.HTTP,
                status=CallStatus.SUCCESS,
                request_data=retrieve_request,
                response_data={
                    "batch_id": batch.id,
                    "status": batch.status,
                    "output_file_id": getattr(batch, "output_file_id", None),
                },
                latency_ms=(time.perf_counter() - start) * 1000,
                provider="azure",
            )
        except Exception as e:
            # External API failure - record error and return structured result
            ctx.record_call(
                call_type=CallType.HTTP,
                status=CallStatus.ERROR,
                request_data=retrieve_request,
                response_data={"error": str(e), "error_type": type(e).__name__},
                latency_ms=(time.perf_counter() - start) * 1000,
                provider="azure",
            )
            # DON'T clear checkpoint - batch exists on Azure, retry should resume checking
            return TransformResult.error(
                {
                    "reason": "batch_retrieve_failed",
                    "error": str(e),
                    "error_type": type(e).__name__,
                    "batch_id": batch_id,
                },
                retryable=True,  # Transient failure - retry status check
            )

        if batch.status == "completed":
            # Download results and assemble output
            return self._download_results(batch, checkpoint, rows, ctx)

        elif batch.status == "failed":
            # Batch failed - clear checkpoint and return error
            self._clear_checkpoint(ctx)

            error_info: TransformErrorReason = {
                "reason": "batch_failed",
                "batch_id": batch_id,
            }
            if hasattr(batch, "errors") and batch.errors:
                error_info["errors"] = [{"message": e.message, "error_type": e.code} for e in batch.errors.data]

            return TransformResult.error(error_info)

        elif batch.status == "cancelled":
            # Batch was cancelled - clear checkpoint and return error
            self._clear_checkpoint(ctx)
            return TransformResult.error(
                {
                    "reason": "batch_cancelled",
                    "batch_id": batch_id,
                }
            )

        elif batch.status == "expired":
            # Batch expired (exceeded 24h) - clear checkpoint and return error
            self._clear_checkpoint(ctx)
            return TransformResult.error(
                {
                    "reason": "batch_expired",
                    "batch_id": batch_id,
                }
            )

        else:
            # Still processing (validating, in_progress, finalizing)
            # Check if we've exceeded max wait time
            submitted_at_str = checkpoint["submitted_at"]
            submitted_at = datetime.fromisoformat(submitted_at_str)
            elapsed_hours = (datetime.now(UTC) - submitted_at).total_seconds() / 3600

            if elapsed_hours > self._max_wait_hours:
                self._clear_checkpoint(ctx)
                return TransformResult.error(
                    {
                        "reason": "batch_timeout",
                        "batch_id": batch_id,
                        "elapsed_hours": elapsed_hours,
                        "max_wait_hours": self._max_wait_hours,
                    }
                )

            # Still waiting - raise BatchPendingError for retry
            # Include checkpoint and node_id so caller can persist and restore
            raise BatchPendingError(
                batch_id,
                batch.status,
                check_after_seconds=self._poll_interval,
                checkpoint=checkpoint,
                node_id=self.node_id,
            )

    def _download_results(
        self,
        batch: Any,
        checkpoint: dict[str, Any],
        rows: list[dict[str, Any]],
        ctx: PluginContext,
    ) -> TransformResult:
        """Download batch results and assemble output rows.

        Args:
            batch: Completed Azure batch object
            checkpoint: Checkpoint data with row mapping
            rows: Original input rows
            ctx: Plugin context

        Returns:
            TransformResult with all processed rows
        """
        client = self._get_client()
        if "row_mapping" not in checkpoint:
            raise RuntimeError("Checkpoint missing required 'row_mapping' for AzureBatchLLMTransform.")
        if "template_errors" not in checkpoint:
            raise RuntimeError("Checkpoint missing required 'template_errors' for AzureBatchLLMTransform.")
        row_mapping: dict[str, dict[str, Any]] = checkpoint["row_mapping"]
        template_errors: list[tuple[int, str]] = checkpoint["template_errors"]

        # Download output file (with audit recording)
        output_file_id = batch.output_file_id
        download_request = {
            "operation": "files.content",
            "file_id": output_file_id,
        }
        start = time.perf_counter()
        try:
            output_content = client.files.content(output_file_id)
            ctx.record_call(
                call_type=CallType.HTTP,
                status=CallStatus.SUCCESS,
                request_data=download_request,
                response_data={
                    "file_id": output_file_id,
                    "content": output_content.text,  # BUG-AZURE-01 FIX: Include actual JSONL output
                    "content_length": len(output_content.text),
                },
                latency_ms=(time.perf_counter() - start) * 1000,
                provider="azure",
            )
        except Exception as e:
            # External API failure - record error and return structured result
            ctx.record_call(
                call_type=CallType.HTTP,
                status=CallStatus.ERROR,
                request_data=download_request,
                response_data={"error": str(e), "error_type": type(e).__name__},
                latency_ms=(time.perf_counter() - start) * 1000,
                provider="azure",
            )
            # DON'T clear checkpoint - batch completed on Azure, retry should re-attempt download
            return TransformResult.error(
                {
                    "reason": "file_download_failed",
                    "error": str(e),
                    "error_type": type(e).__name__,
                    "output_file_id": output_file_id,
                },
                retryable=True,  # Transient failure - retry download
            )

        # Parse JSONL results (EXTERNAL DATA - wrap parsing)
        results_by_id: dict[str, dict[str, Any]] = {}
        malformed_lines: list[str] = []

        for line_num, line in enumerate(output_content.text.strip().split("\n"), start=1):
            if not line:
                continue

            try:
                result = json.loads(line)
            except json.JSONDecodeError as e:
                # Log malformed line but continue processing other lines
                malformed_lines.append(f"Line {line_num}: JSON parse error - {e}")
                continue

            # Validate custom_id presence
            custom_id = result.get("custom_id")
            if custom_id is None:
                malformed_lines.append(f"Line {line_num}: Missing 'custom_id' field")
                continue

            # Validate custom_id is one we sent (membership check at Tier 3 boundary)
            # Azure returning an unknown custom_id would indicate a serious API issue
            if custom_id not in row_mapping:
                malformed_lines.append(f"Line {line_num}: Unknown 'custom_id': {custom_id}")
                continue

            # Tier 3 -> Tier 2 boundary: Validate structure immediately
            # Azure Batch API returns either "error" OR "response", never both
            has_error = "error" in result
            has_response = "response" in result

            if has_error and has_response:
                malformed_lines.append(f"Line {line_num}: Has both 'error' and 'response'")
                continue
            if not has_error and not has_response:
                malformed_lines.append(f"Line {line_num}: Missing both 'error' and 'response'")
                continue

            # Validate response structure if present
            if has_response:
                response = result["response"]
                if not isinstance(response, dict):
                    malformed_lines.append(f"Line {line_num}: 'response' is not a dict")
                    continue
                if "body" not in response:
                    malformed_lines.append(f"Line {line_num}: Missing 'response.body'")
                    continue

            # Now validated - store as Tier 2 data
            results_by_id[custom_id] = result

        # If ALL lines are malformed, fail the entire batch
        if not results_by_id and malformed_lines:
            self._clear_checkpoint(ctx)
            return TransformResult.error(
                {
                    "reason": "all_output_lines_malformed",
                    "malformed_count": len(malformed_lines),
                    "errors": list(malformed_lines[:10]),  # First 10 errors for diagnosis
                }
            )

        # Assemble output rows in original order
        output_rows: list[dict[str, Any]] = []
        row_errors: list[RowErrorEntry] = []

        # Track which rows had template errors (excluded from batch)
        template_error_indices = {idx for idx, _ in template_errors}

        # Build reverse mapping once (O(n) instead of O(n^2) lookup per row)
        idx_to_custom_id: dict[int, str] = {info["index"]: cid for cid, info in row_mapping.items()}

        for idx, row in enumerate(rows):
            if idx in template_error_indices:
                # Row had template error - include original row with error field
                error_msg = next((err for i, err in template_errors if i == idx), "Unknown error")
                output_row = dict(row)
                output_row[self._response_field] = None
                output_row[f"{self._response_field}_error"] = {
                    "reason": "template_rendering_failed",
                    "error": error_msg,
                }
                output_rows.append(output_row)
                continue

            # Find result by custom_id using pre-built reverse mapping
            if idx not in idx_to_custom_id:
                raise RuntimeError(f"Checkpoint row_mapping missing entry for row index {idx} in AzureBatchLLMTransform.")
            custom_id = idx_to_custom_id[idx]

            if custom_id not in results_by_id:
                # Result not found in Azure batch output - request was sent but no response received
                # This is rare but can happen with Azure Batch API edge cases
                output_row = dict(row)
                output_row[self._response_field] = None
                output_row[f"{self._response_field}_error"] = {
                    "reason": "result_not_found",
                    "custom_id": custom_id,
                }
                output_rows.append(output_row)
                row_errors.append({"row_index": idx, "reason": "result_not_found"})

                # Record Call for audit trail completeness - request WAS made but no response
                # Without this, explain(token_id) would show incomplete lineage
                # Access directly from checkpoint (Tier 1 data - we wrote it)
                original_request = checkpoint["requests"][custom_id]
                ctx.record_call(
                    call_type=CallType.LLM,
                    status=CallStatus.ERROR,
                    request_data={
                        "custom_id": custom_id,
                        "row_index": idx,
                        **original_request,
                    },
                    response_data=None,
                    error={"reason": "result_not_found", "custom_id": custom_id},
                    provider="azure",
                )
                continue

            result = results_by_id[custom_id]

            if "error" in result:
                # API error for this row
                output_row = dict(row)
                output_row[self._response_field] = None
                output_row[f"{self._response_field}_error"] = {
                    "reason": "api_error",
                    "error": result["error"],
                }
                output_rows.append(output_row)
                row_errors.append({"row_index": idx, "reason": "api_error", "error": result["error"]})
            else:
                # Success - extract response
                response = result["response"]
                body = response["body"]
                choices = body.get("choices", [])

                if choices:
                    content = choices[0].get("message", {}).get("content", "")
                    usage = body.get("usage", {})

                    output_row = dict(row)
                    output_row[self._response_field] = content

                    # Retrieve variables_hash from checkpoint
                    row_info = row_mapping[custom_id]
                    variables_hash = row_info["variables_hash"]

                    # Guaranteed fields (contract-stable)
                    output_row[f"{self._response_field}_usage"] = usage
                    output_row[f"{self._response_field}_model"] = body.get("model", self._deployment_name)

                    # Audit fields (provenance metadata)
                    output_row[f"{self._response_field}_template_hash"] = self._template.template_hash
                    output_row[f"{self._response_field}_variables_hash"] = variables_hash
                    output_row[f"{self._response_field}_template_source"] = self._template.template_source
                    output_row[f"{self._response_field}_lookup_hash"] = self._template.lookup_hash
                    output_row[f"{self._response_field}_lookup_source"] = self._template.lookup_source
                    output_row[f"{self._response_field}_system_prompt_source"] = self._system_prompt_source

                    output_rows.append(output_row)
                else:
                    # No choices in response
                    output_row = dict(row)
                    output_row[self._response_field] = None
                    output_row[f"{self._response_field}_error"] = {
                        "reason": "no_choices_in_response",
                    }
                    output_rows.append(output_row)
                    row_errors.append({"row_index": idx, "reason": "no_choices_in_response"})

        # Record per-row LLM calls against the batch's state
        # Uses existing state_id from context (set by AggregationExecutor)
        # Note: checkpoint["requests"] is Tier 1 data (we wrote it) - crash if missing
        requests_data = checkpoint["requests"]

        for custom_id, result in results_by_id.items():
            original_request = requests_data[custom_id]
            row_index = row_mapping[custom_id]["index"]

            # Determine call status from result (Tier 2 - validated at boundary)
            if "error" in result:
                call_status = CallStatus.ERROR
                response_data = None
                error_data = {"error": result["error"]}
            else:
                call_status = CallStatus.SUCCESS
                # response.body guaranteed by boundary validation
                response_data = result["response"]["body"]
                error_data = None

            # Record LLM call with custom_id for token mapping
            ctx.record_call(
                call_type=CallType.LLM,
                status=call_status,
                request_data={
                    "custom_id": custom_id,
                    "row_index": row_index,
                    **original_request,
                },
                response_data=response_data,
                error=error_data,
                provider="azure",
            )

        # Clear checkpoint after successful completion
        self._clear_checkpoint(ctx)

        # Return results
        if not output_rows:
            # All rows failed - return error
            return TransformResult.error(
                {
                    "reason": "all_rows_failed",
                    "row_errors": row_errors,
                }
            )

        return TransformResult.success_multi(
            output_rows,
            success_reason={"action": "enriched", "fields_added": [self._response_field]},
        )

    @property
    def azure_config(self) -> dict[str, Any]:
        """Azure configuration for executor (if needed).

        Returns:
            Dict containing endpoint, api_key, api_version, and provider
        """
        return {
            "endpoint": self._endpoint,
            "api_key": self._api_key,
            "api_version": self._api_version,
            "provider": "azure_batch",
        }

    @property
    def deployment_name(self) -> str:
        """Azure deployment name."""
        return self._deployment_name

    def close(self) -> None:
        """Release resources."""
        self._client = None
