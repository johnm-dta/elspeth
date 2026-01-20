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

from elspeth.contracts import CallStatus, CallType, Determinism, TransformResult
from elspeth.plugins.base import BaseTransform
from elspeth.plugins.config_base import TransformDataConfig
from elspeth.plugins.context import PluginContext
from elspeth.plugins.llm.batch_errors import BatchPendingError
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
    response_field: str = Field(
        "llm_response", description="Output field name for LLM response"
    )

    poll_interval_seconds: int = Field(
        300, ge=1, description="Batch status check interval in seconds"
    )
    max_wait_hours: int = Field(
        24, ge=1, le=24, description="Maximum batch wait time in hours"
    )


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
        self._template = PromptTemplate(cfg.template)
        self._system_prompt = cfg.system_prompt
        self._temperature = cfg.temperature
        self._max_tokens = cfg.max_tokens
        self._response_field = cfg.response_field
        self._poll_interval = cfg.poll_interval_seconds
        self._max_wait_hours = cfg.max_wait_hours

        # Schema from config
        assert cfg.schema_config is not None
        schema = create_schema_from_config(
            cfg.schema_config,
            "AzureBatchSchema",
            allow_coercion=False,  # Transforms do NOT coerce
        )
        self.input_schema = schema
        self.output_schema = schema

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
            return TransformResult.success(result.rows[0])
        elif result.status == "error":
            return result
        else:
            # Empty rows from empty batch - shouldn't happen for single row
            return TransformResult.success(row)

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
            # Empty batch - return success with metadata (matches BatchReplicate pattern)
            return TransformResult.success({"batch_empty": True, "row_count": 0})

        # Check checkpoint for resume
        checkpoint = self._get_checkpoint(ctx)

        if checkpoint and checkpoint.get("batch_id"):
            # PHASE 2: Resume - batch already submitted
            return self._check_batch_status(checkpoint, rows, ctx)
        else:
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
        row_mapping: dict[str, int] = {}  # custom_id -> row index
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
            row_mapping[custom_id] = idx

        # If ALL rows failed template rendering, return error immediately
        if not requests:
            return TransformResult.error(
                {
                    "reason": "all_templates_failed",
                    "template_errors": [
                        {"row_index": idx, "error": err} for idx, err in template_errors
                    ],
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
            "content_size": len(jsonl_content),
        }
        start = time.perf_counter()
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
        )

        # Create batch job (with audit recording)
        batch_request = {
            "operation": "batches.create",
            "input_file_id": batch_file.id,
            "endpoint": "/chat/completions",
            "completion_window": "24h",
        }
        start = time.perf_counter()
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
        )

        # 4. CHECKPOINT immediately after submit
        checkpoint_data = {
            "batch_id": batch.id,
            "input_file_id": batch_file.id,
            "row_mapping": row_mapping,
            "template_errors": template_errors,
            "submitted_at": datetime.now(UTC).isoformat(),
            "row_count": len(rows),
        }
        self._update_checkpoint(ctx, checkpoint_data)

        # 5. Raise BatchPendingError - engine handles retry scheduling
        raise BatchPendingError(
            batch.id,
            "submitted",
            check_after_seconds=self._poll_interval,
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
        )

        if batch.status == "completed":
            # Download results and assemble output
            return self._download_results(batch, checkpoint, rows, ctx)

        elif batch.status == "failed":
            # Batch failed - clear checkpoint and return error
            self._clear_checkpoint(ctx)

            error_info: dict[str, Any] = {
                "reason": "batch_failed",
                "batch_id": batch_id,
            }
            if hasattr(batch, "errors") and batch.errors:
                error_info["errors"] = [
                    {"code": e.code, "message": e.message} for e in batch.errors.data
                ]

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
            submitted_at_str = checkpoint.get("submitted_at")
            if submitted_at_str:
                submitted_at = datetime.fromisoformat(submitted_at_str)
                elapsed_hours = (
                    datetime.now(UTC) - submitted_at
                ).total_seconds() / 3600

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
            raise BatchPendingError(
                batch_id,
                batch.status,
                check_after_seconds=self._poll_interval,
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
        row_mapping: dict[str, int] = checkpoint.get("row_mapping", {})
        template_errors: list[tuple[int, str]] = checkpoint.get("template_errors", [])

        # Download output file (with audit recording)
        output_file_id = batch.output_file_id
        download_request = {
            "operation": "files.content",
            "file_id": output_file_id,
        }
        start = time.perf_counter()
        output_content = client.files.content(output_file_id)
        ctx.record_call(
            call_type=CallType.HTTP,
            status=CallStatus.SUCCESS,
            request_data=download_request,
            response_data={
                "file_id": output_file_id,
                "content_length": len(output_content.text),
            },
            latency_ms=(time.perf_counter() - start) * 1000,
        )

        # Parse JSONL results
        results_by_id: dict[str, dict[str, Any]] = {}
        for line in output_content.text.strip().split("\n"):
            if line:
                result = json.loads(line)
                results_by_id[result["custom_id"]] = result

        # Assemble output rows in original order
        output_rows: list[dict[str, Any]] = []
        row_errors: list[dict[str, Any]] = []

        # Track which rows had template errors (excluded from batch)
        template_error_indices = {idx for idx, _ in template_errors}

        # Build reverse mapping once (O(n) instead of O(n^2) lookup per row)
        idx_to_custom_id: dict[int, str] = {
            ridx: cid for cid, ridx in row_mapping.items()
        }

        for idx, row in enumerate(rows):
            if idx in template_error_indices:
                # Row had template error - include original row with error field
                error_msg = next(
                    (err for i, err in template_errors if i == idx), "Unknown error"
                )
                output_row = dict(row)
                output_row[self._response_field] = None
                output_row[f"{self._response_field}_error"] = {
                    "reason": "template_rendering_failed",
                    "error": error_msg,
                }
                output_rows.append(output_row)
                continue

            # Find result by custom_id using pre-built reverse mapping
            custom_id = idx_to_custom_id.get(idx)

            if custom_id is None or custom_id not in results_by_id:
                # Result not found - should not happen
                output_row = dict(row)
                output_row[self._response_field] = None
                output_row[f"{self._response_field}_error"] = {
                    "reason": "result_not_found",
                    "custom_id": custom_id,
                }
                output_rows.append(output_row)
                row_errors.append({"row_index": idx, "reason": "result_not_found"})
                continue

            result = results_by_id[custom_id]

            if result.get("error"):
                # API error for this row
                output_row = dict(row)
                output_row[self._response_field] = None
                output_row[f"{self._response_field}_error"] = {
                    "reason": "api_error",
                    "error": result["error"],
                }
                output_rows.append(output_row)
                row_errors.append(
                    {"row_index": idx, "reason": "api_error", "error": result["error"]}
                )
            else:
                # Success - extract response
                response = result.get("response", {})
                body = response.get("body", {})
                choices = body.get("choices", [])

                if choices:
                    content = choices[0].get("message", {}).get("content", "")
                    usage = body.get("usage", {})

                    output_row = dict(row)
                    output_row[self._response_field] = content
                    output_row[f"{self._response_field}_usage"] = usage
                    # Add template hash for audit
                    output_row[f"{self._response_field}_template_hash"] = (
                        self._template.template_hash
                    )
                    output_rows.append(output_row)
                else:
                    # No choices in response
                    output_row = dict(row)
                    output_row[self._response_field] = None
                    output_row[f"{self._response_field}_error"] = {
                        "reason": "no_choices_in_response",
                    }
                    output_rows.append(output_row)
                    row_errors.append(
                        {"row_index": idx, "reason": "no_choices_in_response"}
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

        return TransformResult.success_multi(output_rows)

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
