"""Azure Multi-Query LLM transform for case study x criteria evaluation.

Executes multiple LLM queries per row in parallel, merging all results
into a single output row with all-or-nothing error handling.
"""

from __future__ import annotations

import json
from threading import Lock
from typing import TYPE_CHECKING, Any

from elspeth.contracts import Determinism, TransformResult
from elspeth.plugins.base import BaseTransform
from elspeth.plugins.clients.llm import AuditedLLMClient, LLMClientError, RateLimitError
from elspeth.plugins.context import PluginContext
from elspeth.plugins.llm.multi_query import MultiQueryConfig, QuerySpec
from elspeth.plugins.llm.templates import PromptTemplate, TemplateError
from elspeth.plugins.pooling import CapacityError, PooledExecutor
from elspeth.plugins.schema_factory import create_schema_from_config

if TYPE_CHECKING:
    from openai import AzureOpenAI

    from elspeth.core.landscape.recorder import LandscapeRecorder


class AzureMultiQueryLLMTransform(BaseTransform):
    """LLM transform that executes case_studies x criteria queries per row.

    For each row, expands the cross-product of case studies and criteria
    into individual LLM queries. All queries run in parallel (up to pool_size),
    with all-or-nothing error semantics (if any query fails, the row fails).

    Configuration example:
        transforms:
          - plugin: azure_multi_query_llm
            options:
              deployment_name: "gpt-4o"
              endpoint: "${AZURE_OPENAI_ENDPOINT}"
              api_key: "${AZURE_OPENAI_KEY}"
              template: |
                Case: {{ input_1 }}, {{ input_2 }}
                Criterion: {{ criterion.name }}
              case_studies:
                - name: cs1
                  input_fields: [cs1_bg, cs1_sym]
                - name: cs2
                  input_fields: [cs2_bg, cs2_sym]
              criteria:
                - name: diagnosis
                  code: DIAG
                - name: treatment
                  code: TREAT
              response_format: json
              output_mapping:
                score: score
                rationale: rationale
              pool_size: 4
              schema:
                fields: dynamic

    Output fields per query:
        {case_study}_{criterion}_{json_field} for each output_mapping entry
        Plus metadata: _usage, _template_hash, _model
    """

    name = "azure_multi_query_llm"
    is_batch_aware = True
    creates_tokens = False  # Does not create new tokens (1 row in -> 1 row out)
    determinism: Determinism = Determinism.NON_DETERMINISTIC
    plugin_version = "1.0.0"

    def __init__(self, config: dict[str, Any]) -> None:
        """Initialize transform with multi-query configuration."""
        super().__init__(config)

        # Parse config
        cfg = MultiQueryConfig.from_dict(config)

        # Store Azure connection settings
        self._azure_endpoint = cfg.endpoint
        self._azure_api_key = cfg.api_key
        self._azure_api_version = cfg.api_version
        self._deployment_name = cfg.deployment_name
        self._model = cfg.model or cfg.deployment_name

        # Store template settings
        self._template = PromptTemplate(
            cfg.template,
            template_source=cfg.template_source,
            lookup_data=cfg.lookup,
            lookup_source=cfg.lookup_source,
        )
        self._system_prompt = cfg.system_prompt
        self._temperature = cfg.temperature
        self._max_tokens = cfg.max_tokens
        self._on_error = cfg.on_error

        # Multi-query specific settings
        self._output_mapping = cfg.output_mapping
        self._response_format = cfg.response_format

        # Pre-expand query specs (case_studies x criteria)
        self._query_specs: list[QuerySpec] = cfg.expand_queries()

        # Schema from config
        assert cfg.schema_config is not None
        schema = create_schema_from_config(
            cfg.schema_config,
            f"{self.name}Schema",
            allow_coercion=False,
        )
        self.input_schema = schema
        self.output_schema = schema

        # Pooled execution setup
        if cfg.pool_config is not None:
            self._executor: PooledExecutor | None = PooledExecutor(cfg.pool_config)
        else:
            self._executor = None

        # Client caching (same pattern as AzureLLMTransform)
        self._recorder: LandscapeRecorder | None = None
        self._llm_clients: dict[str, AuditedLLMClient] = {}
        self._llm_clients_lock = Lock()
        self._underlying_client: AzureOpenAI | None = None

    def on_start(self, ctx: PluginContext) -> None:
        """Capture recorder reference for pooled execution."""
        self._recorder = ctx.landscape

    def _get_underlying_client(self) -> AzureOpenAI:
        """Get or create the underlying Azure OpenAI client."""
        if self._underlying_client is None:
            from openai import AzureOpenAI

            self._underlying_client = AzureOpenAI(
                azure_endpoint=self._azure_endpoint,
                api_key=self._azure_api_key,
                api_version=self._azure_api_version,
            )
        return self._underlying_client

    def _get_llm_client(self, state_id: str) -> AuditedLLMClient:
        """Get or create LLM client for a state_id."""
        with self._llm_clients_lock:
            if state_id not in self._llm_clients:
                if self._recorder is None:
                    raise RuntimeError("Transform requires recorder. Ensure on_start was called.")
                self._llm_clients[state_id] = AuditedLLMClient(
                    recorder=self._recorder,
                    state_id=state_id,
                    underlying_client=self._get_underlying_client(),
                    provider="azure",
                )
            return self._llm_clients[state_id]

    def _process_single_query(
        self,
        row: dict[str, Any],
        spec: QuerySpec,
        state_id: str,
    ) -> TransformResult:
        """Process a single query (one case_study x criterion pair).

        Args:
            row: Full input row
            spec: Query specification with input field mapping
            state_id: State ID for audit trail

        Returns:
            TransformResult with mapped output fields

        Raises:
            CapacityError: On rate limit (for pooled retry)
        """
        # 1. Build synthetic row for PromptTemplate
        # Templates use {{ row.input_1 }}, {{ row.criterion }}, {{ row.original }}, {{ lookup }}
        # This preserves PromptTemplate's audit metadata (template_hash, variables_hash)
        synthetic_row = spec.build_template_context(row)
        # synthetic_row now contains: input_1, input_2, ..., criterion, row (original)

        # 2. Render template using PromptTemplate (preserves audit metadata)
        # THEIR DATA - wrap in try/catch
        try:
            rendered = self._template.render_with_metadata(synthetic_row)
        except TemplateError as e:
            return TransformResult.error(
                {
                    "reason": "template_rendering_failed",
                    "error": str(e),
                    "query": spec.output_prefix,
                    "template_hash": self._template.template_hash,
                }
            )

        # 3. Build messages
        messages: list[dict[str, str]] = []
        if self._system_prompt:
            messages.append({"role": "system", "content": self._system_prompt})
        messages.append({"role": "user", "content": rendered.prompt})

        # 4. Get LLM client
        llm_client = self._get_llm_client(state_id)

        # 5. Call LLM (EXTERNAL - wrap, raise CapacityError for retry)
        try:
            response = llm_client.chat_completion(
                model=self._model,
                messages=messages,
                temperature=self._temperature,
                max_tokens=self._max_tokens,
            )
        except RateLimitError as e:
            raise CapacityError(429, str(e)) from e
        except LLMClientError as e:
            return TransformResult.error(
                {
                    "reason": "llm_call_failed",
                    "error": str(e),
                    "query": spec.output_prefix,
                }
            )

        # 6. Parse JSON response (THEIR DATA - wrap)
        try:
            parsed = json.loads(response.content)
        except json.JSONDecodeError as e:
            return TransformResult.error(
                {
                    "reason": "json_parse_failed",
                    "error": str(e),
                    "query": spec.output_prefix,
                    "raw_response": response.content[:500],  # Truncate for audit
                }
            )

        # 7. Map output fields
        output: dict[str, Any] = {}
        for json_field, suffix in self._output_mapping.items():
            output_key = f"{spec.output_prefix}_{suffix}"
            if json_field not in parsed:
                return TransformResult.error(
                    {
                        "reason": "missing_output_field",
                        "field": json_field,
                        "query": spec.output_prefix,
                    }
                )
            output[output_key] = parsed[json_field]

        # 8. Add metadata for audit trail
        output[f"{spec.output_prefix}_usage"] = response.usage
        output[f"{spec.output_prefix}_model"] = response.model
        # Template metadata for reproducibility
        output[f"{spec.output_prefix}_template_hash"] = rendered.template_hash
        output[f"{spec.output_prefix}_variables_hash"] = rendered.variables_hash
        output[f"{spec.output_prefix}_template_source"] = rendered.template_source
        output[f"{spec.output_prefix}_lookup_hash"] = rendered.lookup_hash
        output[f"{spec.output_prefix}_lookup_source"] = rendered.lookup_source

        return TransformResult.success(output)

    def process(
        self,
        row: dict[str, Any] | list[dict[str, Any]],
        ctx: PluginContext,
    ) -> TransformResult:
        """Process row(s) with all queries in parallel.

        For single row: executes all (case_study x criterion) queries,
        merges results into one output row.

        For batch: processes each row independently (batch of multi-query rows).
        """
        # Batch dispatch
        if isinstance(row, list):
            return self._process_batch(row, ctx)

        # Single row processing
        return self._process_single_row(row, ctx)

    def _process_single_row_internal(
        self,
        row: dict[str, Any],
        state_id: str,
    ) -> TransformResult:
        """Internal row processing with explicit state_id.

        Used by both single-row and batch processing paths.

        Args:
            row: Input row with all case study fields
            state_id: State ID for audit trail

        Returns:
            TransformResult with all query results merged, or error
        """
        # Execute all queries (parallel or sequential)
        if self._executor is not None:
            results = self._execute_queries_parallel(row, state_id)
        else:
            results = self._execute_queries_sequential(row, state_id)

        # Check for failures (all-or-nothing for this row)
        failed = [(spec, r) for spec, r in zip(self._query_specs, results, strict=True) if r.status != "success"]
        if failed:
            return TransformResult.error(
                {
                    "reason": "query_failed",
                    "failed_queries": [{"query": spec.output_prefix, "error": r.reason} for spec, r in failed],
                    "succeeded_count": len(results) - len(failed),
                    "total_count": len(results),
                }
            )

        # Merge all results into output row
        output = dict(row)
        for result in results:
            # Check for row presence: successful results should always have a row,
            # but TransformResult supports multi-output scenarios where row may be
            # None even on success. This check is defensive for that edge case.
            if result.row is not None:
                output.update(result.row)

        return TransformResult.success(output)

    def _process_single_row(
        self,
        row: dict[str, Any],
        ctx: PluginContext,
    ) -> TransformResult:
        """Process a single row with all queries in parallel.

        Executes all (case_study x criterion) queries for this row.
        All-or-nothing: if any query fails, the entire row fails.

        Args:
            row: Input row with all case study fields
            ctx: Plugin context with landscape and state_id

        Returns:
            TransformResult with all query results merged, or error
        """
        if ctx.landscape is None or ctx.state_id is None:
            raise RuntimeError("Multi-query transform requires landscape recorder and state_id.")

        # Capture recorder for pooled execution
        if self._recorder is None:
            self._recorder = ctx.landscape

        try:
            return self._process_single_row_internal(row, ctx.state_id)
        finally:
            # Clean up cached clients for this state_id
            with self._llm_clients_lock:
                self._llm_clients.pop(ctx.state_id, None)

    def _execute_queries_parallel(
        self,
        row: dict[str, Any],
        state_id: str,
    ) -> list[TransformResult]:
        """Execute queries in parallel via ThreadPoolExecutor.

        This method uses ThreadPoolExecutor directly for per-row query parallelism
        rather than PooledExecutor.execute_batch(). The distinction:

        - PooledExecutor.execute_batch(): Designed for cross-row batching with AIMD
          throttling to adaptively manage rate limits across many rows.
        - ThreadPoolExecutor here: Simple parallel execution within a single row.
          All queries share the same underlying AzureOpenAI client which handles
          its own rate limiting, so AIMD overhead is unnecessary.

        Args:
            row: The input row data
            state_id: State ID for audit trail

        Returns:
            List of TransformResults in query spec order
        """
        from concurrent.futures import ThreadPoolExecutor, as_completed

        # Type narrowing - caller ensures executor is not None
        assert self._executor is not None

        results_by_index: dict[int, TransformResult] = {}

        with ThreadPoolExecutor(max_workers=self._executor.pool_size) as executor:
            futures = {
                executor.submit(
                    self._process_single_query,
                    row,
                    spec,
                    state_id,
                ): i
                for i, spec in enumerate(self._query_specs)
            }

            for future in as_completed(futures):
                idx = futures[future]
                try:
                    results_by_index[idx] = future.result()
                except CapacityError as e:
                    # If capacity error escapes, treat as error
                    results_by_index[idx] = TransformResult.error(
                        {
                            "reason": "capacity_exhausted",
                            "query": self._query_specs[idx].output_prefix,
                            "error": str(e),
                        }
                    )

        # Return in submission order
        return [results_by_index[i] for i in range(len(self._query_specs))]

    def _execute_queries_sequential(
        self,
        row: dict[str, Any],
        state_id: str,
    ) -> list[TransformResult]:
        """Execute queries sequentially (fallback when no executor).

        Args:
            row: The input row data
            state_id: State ID for audit trail

        Returns:
            List of TransformResults in query spec order
        """
        results: list[TransformResult] = []

        for spec in self._query_specs:
            try:
                result = self._process_single_query(row, spec, state_id)
            except CapacityError as e:
                result = TransformResult.error(
                    {
                        "reason": "rate_limited",
                        "error": str(e),
                        "query": spec.output_prefix,
                    }
                )
            results.append(result)

        return results

    def _process_batch(
        self,
        rows: list[dict[str, Any]],
        ctx: PluginContext,
    ) -> TransformResult:
        """Process batch of rows (aggregation mode).

        Each row is processed independently. Failed rows get _error marker
        while successful rows continue. This implements partial success semantics
        for batch processing.

        Args:
            rows: List of input rows
            ctx: Plugin context with landscape and state_id

        Returns:
            TransformResult.success_multi with all row results, or
            TransformResult.success for empty batch
        """
        if not rows:
            return TransformResult.success({"batch_empty": True, "row_count": 0})

        if ctx.landscape is None or ctx.state_id is None:
            raise RuntimeError("Batch processing requires landscape recorder and state_id.")

        if self._recorder is None:
            self._recorder = ctx.landscape

        output_rows: list[dict[str, Any]] = []

        for i, row in enumerate(rows):
            row_state_id = f"{ctx.state_id}_row{i}"

            try:
                result = self._process_single_row_internal(row, row_state_id)

                if result.status == "success" and result.row is not None:
                    output_rows.append(result.row)
                else:
                    # Row failed - include original with error marker
                    error_row = dict(row)
                    error_row["_error"] = result.reason
                    output_rows.append(error_row)
            finally:
                # Clean up per-row client cache
                with self._llm_clients_lock:
                    self._llm_clients.pop(row_state_id, None)

        return TransformResult.success_multi(output_rows)

    def close(self) -> None:
        """Release resources."""
        if self._executor is not None:
            self._executor.shutdown(wait=True)
        self._recorder = None
        with self._llm_clients_lock:
            self._llm_clients.clear()
        self._underlying_client = None
