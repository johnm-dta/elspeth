"""TransformExecutor - wraps transform.process() with audit recording."""

import time
from typing import TYPE_CHECKING, Any, cast

from pydantic import ValidationError

import elspeth.contracts.errors as contract_errors
from elspeth.contracts import (
    BatchTransformRuntimeProtocol,
    ExecutionError,
    TokenInfo,
    TransformProtocol,
    TransformResult,
)
from elspeth.contracts.audit import TokenRef
from elspeth.contracts.declaration_contracts import (
    AggregateDeclarationContractViolation,
    DeclarationContractViolation,
    PostEmissionInputs,
    PostEmissionOutputs,
    PreEmissionInputs,
    derive_effective_input_fields,
)
from elspeth.contracts.enums import (
    NodeStateStatus,
    RoutingMode,
    TerminalOutcome,
    TerminalPath,
)
from elspeth.contracts.errors import (
    AuditIntegrityError,
    OrchestrationInvariantError,
    PassThroughContractViolation,
    PluginContractViolation,
    ZeroEmissionSuccessContractViolation,
)
from elspeth.contracts.plugin_context import PluginContext, plugin_context_scope
from elspeth.contracts.secret_scrub import scrub_payload_for_audit
from elspeth.contracts.types import NodeID, StepResolver
from elspeth.core.canonical import stable_hash
from elspeth.core.landscape.data_flow_repository import DataFlowRepository
from elspeth.core.landscape.errors import LandscapeRecordError
from elspeth.core.landscape.execution_repository import ExecutionRepository
from elspeth.engine._error_hash import compute_error_hash
from elspeth.engine.executors.declaration_dispatch import (
    run_post_emission_checks,
    run_pre_emission_checks,
)
from elspeth.engine.executors.state_guard import NodeStateGuard
from elspeth.engine.spans import SpanFactory

if TYPE_CHECKING:
    from elspeth.contracts import TransformErrorReason
    from elspeth.contracts.schema_contract import PipelineRow
    from elspeth.engine.batch_adapter import SharedBatchAdapter


def _scrub_transform_error_details(error_details: "TransformErrorReason") -> "TransformErrorReason":
    """Scrub freeform transform error payload while preserving category fields."""
    scrubbed = scrub_payload_for_audit(error_details)
    if scrubbed == error_details:
        return error_details
    return cast("TransformErrorReason", scrubbed)


def record_transform_error_with_routing(
    *,
    ctx: PluginContext,
    execution: ExecutionRepository,
    error_edge_ids: dict[NodeID, str],
    state_id: str | None,
    token: TokenInfo,
    transform: TransformProtocol,
    row: "dict[str, Any] | PipelineRow",
    error_details: "TransformErrorReason",
    on_error: str,
) -> None:
    """Record a transform error + its DIVERT routing_event (audit trail).

    Shared, sequencing-agnostic error-audit routine (elspeth-aeb0a8f756):
    used by TransformExecutor's error-result branch (which records BEFORE
    terminal completion) and by RowProcessor's retryable-exception
    conversion (which records AFTER the guard already auto-failed the
    state). It therefore must NOT complete node states — each caller owns
    its own completion sequencing.

    ``row`` accepts a plain dict or a PipelineRow: canonicalization
    normalizes PipelineRow via ``.to_dict()``, so the persisted row_hash /
    row_data_json are identical either way.
    """
    node_id = transform.node_id
    if node_id is None:
        raise OrchestrationInvariantError(f"Transform '{transform.name}' executed without node_id - orchestrator bug")

    scrubbed_error_details = _scrub_transform_error_details(error_details)

    ctx.record_transform_error(
        token_id=token.token_id,
        transform_id=node_id,
        row=row,
        error_details=scrubbed_error_details,
        destination=on_error,
    )

    # Record DIVERT routing_event for audit trail (AUD-002), co-located with
    # the transform_error record. 'discard' has no destination edge.
    if on_error != "discard":
        if state_id is None:
            # The executor passes guard.state_id (never None); the processor
            # passes the id NodeStateGuard stamped on the propagating
            # exception (ctx.state_id is scope-restored during unwind) — a
            # missing id here means no attempt ever opened a node state, and
            # there is no state to attach the DIVERT to.
            raise OrchestrationInvariantError(
                f"state_id is required to record the DIVERT routing_event for transform '{node_id}' (on_error={on_error!r})"
            )
        try:
            error_edge_id = error_edge_ids[NodeID(node_id)]
        except KeyError as exc:
            raise OrchestrationInvariantError(
                f"Transform '{node_id}' has on_error={on_error!r} but no "
                f"DIVERT edge registered. DAG construction should have created an "
                f"__error_{{name}}__ edge in from_plugin_instances()."
            ) from exc
        execution.record_routing_event(
            state_id=state_id,
            edge_id=error_edge_id,
            mode=RoutingMode.DIVERT,
            reason=scrubbed_error_details,
        )


class TransformExecutor:
    """Executes transforms with audit recording.

    Wraps transform.process() to:
    1. Record node state start
    2. Time the operation
    3. Populate audit fields in result
    4. Record node state completion
    5. Emit OpenTelemetry span

    Node state terminality is guaranteed by NodeStateGuard: if any
    post-processing step (output hashing, contract evolution) raises
    before the state is explicitly completed, the guard auto-completes
    it as FAILED.  This prevents orphan OPEN states in the audit trail.

    Example:
        executor = TransformExecutor(execution, span_factory, step_resolver, data_flow=data_flow)
        result, updated_token, error_sink = executor.execute_transform(
            transform=my_transform,
            token=token,
            ctx=ctx,
        )
    """

    def __init__(
        self,
        execution: ExecutionRepository,
        span_factory: SpanFactory,
        step_resolver: StepResolver,
        data_flow: DataFlowRepository,
        max_workers: int | None = None,
        error_edge_ids: dict[NodeID, str] | None = None,
    ) -> None:
        """Initialize executor.

        Args:
            execution: Execution repository for audit trail
            span_factory: Span factory for tracing
            step_resolver: Resolves NodeID to 1-indexed audit step position
            data_flow: Data flow repository for schema contract updates.
            max_workers: Maximum concurrent workers (None = no limit)
            error_edge_ids: Map of transform node_id -> DIVERT edge_id for error routing.
                           Built by the processor from the edge_map using error_edge_label().
                           Only populated for transforms with on_error pointing to a real sink.
        """
        self._execution = execution
        self._data_flow = data_flow
        self._spans = span_factory
        self._step_resolver = step_resolver
        self._max_workers = max_workers
        self._error_edge_ids = error_edge_ids or {}
        # Adapter storage keyed by node_id — one SharedBatchAdapter per
        # row-pipelined batch transform, owned by the executor (not monkey-patched
        # onto the transform instance).
        self._batch_adapters: dict[str, "SharedBatchAdapter"] = {}  # noqa: UP037 — forward ref, no __future__ annotations
        # OpenTelemetry counter for pass-through cross-check violations now lives
        # at module scope in engine.executors.pass_through (ADR-009 §Clause 2).
        # Both this executor and the processor's batch-flush cross-check share
        # the same instrument without needing constructor plumbing.

    def _record_terminal_contract_failure(
        self,
        *,
        transform: TransformProtocol,
        token: TokenInfo,
        run_id: str,
        violation: (
            DeclarationContractViolation
            | AggregateDeclarationContractViolation
            | PassThroughContractViolation
            | ZeroEmissionSuccessContractViolation
        ),
    ) -> None:
        """Persist the matching FAILED token_outcome for declaration-path failures."""
        if self._data_flow is None:
            raise OrchestrationInvariantError(
                f"TransformExecutor.data_flow is None but declaration-path failures for "
                f"transform '{transform.name}' must record terminal token_outcomes."
            )

        if type(violation) is PassThroughContractViolation:
            summary = f"PassThroughContractViolation:{transform.name}:{sorted(violation.divergence_set)}"
        else:
            summary = f"{type(violation).__name__}:{transform.name}"
        error_hash = compute_error_hash(summary)
        audit_context = violation.to_audit_dict()

        try:
            self._data_flow.record_token_outcome(
                ref=TokenRef(token_id=token.token_id, run_id=run_id),
                outcome=TerminalOutcome.FAILURE,
                path=TerminalPath.UNROUTED,
                error_hash=error_hash,
                context=audit_context,
            )
        except LandscapeRecordError as record_failure:
            raise AuditIntegrityError(
                f"Failed to record {type(violation).__name__} FAILED outcome for token "
                f"{token.token_id!r} (transform={transform.name!r}, node={transform.node_id!r}). "
                f"Node state is already FAILED but terminal token_outcome is missing. "
                f"Recorder failure: {type(record_failure).__name__}: {record_failure}. "
                f"Original violation: {violation!s}"
            ) from record_failure

    def _get_batch_adapter(self, transform: BatchTransformRuntimeProtocol) -> "SharedBatchAdapter":
        """Get or create shared batch adapter for a row-pipelined transform.

        Creates adapter once per transform and stores it in the executor's
        own dict (keyed by node_id). On first call, connects the adapter as
        the transform's output port.

        Caller must verify ``isinstance(transform, BatchTransformRuntimeProtocol)``
        before calling.

        Args:
            transform: Row-pipelined batch transform (must have node_id set)

        Returns:
            SharedBatchAdapter for this transform
        """
        from elspeth.engine.batch_adapter import SharedBatchAdapter

        # node_id is always set by orchestrator before execution
        node_id = transform.node_id
        if node_id is None:
            raise OrchestrationInvariantError("node_id must be set before execute_transform")

        if node_id not in self._batch_adapters:
            adapter = SharedBatchAdapter()
            self._batch_adapters[node_id] = adapter

            # Connect output (one-time setup)
            # Cap pool_size to max_workers if configured (global concurrency limit)
            max_pending = transform.batch_pool_size
            if self._max_workers is not None:
                max_pending = min(max_pending, self._max_workers)
            transform.connect_output(output=adapter, max_pending=max_pending)

        return self._batch_adapters[node_id]

    def _run_preflight(
        self,
        *,
        transform: TransformProtocol,
        token: TokenInfo,
        input_dict: dict[str, Any],
        run_id: str,
        node_id: str,
    ) -> tuple[frozenset[str], frozenset[str]]:
        """Run pre-invocation checks before the transform executes.

        Owns the preflight phase: lifecycle guard, field-collision policy,
        pre-emission declaration-contract dispatch (with terminal outcome
        recording), and input-schema validation. Violations raise; the
        caller's NodeStateGuard auto-fails the node state, so this helper
        never touches the guard.

        Returns:
            Tuple of (effective_input_fields, static_contract), derived once
            here and reused by the post-emission call site (panel F1
            resolution: contracts use the caller-derived set, not their own
            re-derivation).
        """
        # --- LIFECYCLE GUARD (pre-execution) ---
        # Centralized check: ensure on_start() was called before process().
        # All transforms are system-owned and must inherit BaseTransform.
        # AttributeError here means a transform violates the interface contract.
        if not transform._on_start_called:
            raise PluginContractViolation(
                f"Transform '{transform.name}' was called before on_start(). "
                f"This is an engine lifecycle bug — on_start() must be called "
                f"before any process() invocation."
            )

        # --- FIELD COLLISION ENFORCEMENT (pre-execution) ---
        # Centralized check: if this transform declares output fields,
        # verify none collide with input fields BEFORE running the transform.
        # This prevents wasted API calls AND makes collision detection mandatory
        # (not opt-in per plugin).
        if transform.declared_output_fields:
            from elspeth.contracts.field_collision import detect_field_collisions

            collisions = detect_field_collisions(
                set(input_dict.keys()),
                transform.declared_output_fields,
            )
            if collisions is not None:
                raise PluginContractViolation(
                    f"Transform '{transform.name}' would overwrite existing input fields "
                    f"{collisions}. This is a pipeline configuration error — the transform's "
                    f"output fields collide with fields already present in the row."
                )

        # --- PRE-EMISSION DECLARATION-CONTRACT DISPATCH (ADR-010 §Decision 3 + F2) ---
        # Fires BEFORE generic input_schema validation so the current
        # pre-emission adopter (DeclaredRequiredFieldsContract) can attribute missing
        # declared input fields to ADR-013 rather than collapsing them into a
        # generic validation error when the schema requires the same field.
        # It also runs BEFORE transform.process() so a missing-field crash in
        # the plugin body cannot steal attribution from the declaration surface.
        effective_input_fields = derive_effective_input_fields(token.row_data)
        static_contract = transform.effective_static_contract()
        try:
            run_pre_emission_checks(
                inputs=PreEmissionInputs(
                    plugin=transform,
                    node_id=node_id,
                    run_id=run_id,
                    row_id=token.row_id,
                    token_id=token.token_id,
                    input_row=token.row_data,
                    static_contract=static_contract,
                    effective_input_fields=effective_input_fields,
                ),
            )
        except (DeclarationContractViolation, AggregateDeclarationContractViolation) as violation:
            self._record_terminal_contract_failure(
                transform=transform,
                token=token,
                run_id=run_id,
                violation=violation,
            )
            raise

        # --- INPUT VALIDATION (pre-execution) ---
        # Validate input against input_schema before calling process().
        # Wrong types at a transform boundary are upstream plugin bugs (Tier 2).
        # ADR-013 declaration checks run first so missing declared fields stay
        # on the declaration-contract audit surface instead of being diluted
        # into ordinary schema validation failures.
        try:
            transform.input_schema.model_validate(input_dict, strict=True)
        except ValidationError as e:
            raise PluginContractViolation(
                f"Transform '{transform.name}' input validation failed: {e}. This indicates an upstream transform/source schema bug."
            ) from e

        return effective_input_fields, static_contract

    def _invoke_transform(
        self,
        *,
        transform: TransformProtocol,
        batch_runtime: BatchTransformRuntimeProtocol | None,
        token: TokenInfo,
        ctx: PluginContext,
        state_id: str,
    ) -> TransformResult:
        """Invoke the transform and return its result (invocation phase).

        Owns invocation-mode selection: synchronous ``transform.process()``
        vs the row-pipelined batch runtime (``accept()`` + waiter). Exceptions
        propagate unhandled — the caller owns timing, span, FAILED completion,
        and timeout eviction, because those bracket ``guard.complete()`` and
        must stay with the guard.
        """
        if batch_runtime is not None:
            # Batch transform: use accept() with SharedBatchAdapter
            # One adapter per transform, multiple waiters per adapter
            adapter = self._get_batch_adapter(batch_runtime)

            # Register waiter for THIS token AND attempt (before accept!)
            # Using (token_id, state_id) ensures retry safety: if a timeout
            # occurs and retry happens, the new attempt's waiter won't receive
            # stale results from the previous attempt.
            waiter = adapter.register(token.token_id, state_id)

            # Submit work - this returns immediately
            batch_runtime.accept(token.row_data, ctx)

            # Block until THIS row's result arrives.
            #
            # DESIGN DECISION: Sequential row processing
            # The orchestrator processes rows one at a time, blocking here
            # until each row completes. This is intentional:
            # - Concurrency happens WITHIN each row (multi-query transforms
            #   make 10+ LLM calls concurrently for a single row)
            # - Across rows, processing is sequential for:
            #   1. Simpler audit ordering (deterministic state progression)
            #   2. Natural backpressure (no unbounded queue growth)
            #   3. Single-threaded orchestrator (easier to reason about)
            #
            # For true cross-row parallelism, the orchestrator would need
            # to be async/await or multi-threaded, which adds complexity.
            #
            # Timeout is derived from transform's batch_wait_timeout config
            # (default 3600s = 1 hour) to allow for sustained rate limiting
            # and AIMD backoff during capacity errors.
            result = waiter.wait(timeout=batch_runtime.batch_wait_timeout, shutdown_event=ctx.shutdown_event)
        else:
            # Regular transform: synchronous process()
            result = transform.process(token.row_data, ctx)
        return result

    def _verify_success_emissions(
        self,
        *,
        result: TransformResult,
        transform: TransformProtocol,
        token: TokenInfo,
        run_id: str,
        node_id: str,
        static_contract: frozenset[str],
        effective_input_fields: frozenset[str],
    ) -> None:
        """Verify a success result's emitted rows (success-audit phase, part 1).

        Owns the zero-emission declaration path, post-emission
        declaration-contract dispatch (ADR-010 §Decision 3 + §Semantics
        amendment 2026-04-20, audit-complete collect-then-raise dispatcher)
        with terminal-failure recording, and per-row output-schema validation.
        Violations raise; the caller's NodeStateGuard auto-fails the state.

        ``static_contract`` + ``effective_input_fields`` are the
        preflight-derived values, reused here (panel F1 resolution — single
        caller-side derivation).
        """
        if result.row is not None:
            emitted_rows: tuple[Any, ...] = (result.row,)
        elif result.rows is not None:
            emitted_rows = tuple(result.rows)
        else:
            emitted_rows = ()
        used_success_empty = result.rows is not None and len(result.rows) == 0
        try:
            run_post_emission_checks(
                inputs=PostEmissionInputs(
                    plugin=transform,
                    node_id=node_id,
                    run_id=run_id,
                    row_id=token.row_id,
                    token_id=token.token_id,
                    input_row=token.row_data,
                    static_contract=static_contract,
                    effective_input_fields=effective_input_fields,
                ),
                outputs=PostEmissionOutputs(
                    emitted_rows=emitted_rows,
                    used_success_empty=used_success_empty,
                ),
            )
        except (
            DeclarationContractViolation,
            AggregateDeclarationContractViolation,
            PassThroughContractViolation,
            ZeroEmissionSuccessContractViolation,
        ) as violation:
            self._record_terminal_contract_failure(
                transform=transform,
                token=token,
                run_id=run_id,
                violation=violation,
            )
            raise

        for idx, emitted_row in enumerate(emitted_rows):
            try:
                transform.output_schema.model_validate(emitted_row.to_dict(), strict=True)
            except ValidationError as e:
                raise PluginContractViolation(
                    f"Transform '{transform.name}' output validation failed for emitted row {idx}: {e}. "
                    "This indicates a transform schema bug."
                ) from e

    def _populate_result_audit_fields(
        self,
        *,
        result: TransformResult,
        transform: TransformProtocol,
        input_hash: str,
        duration_ms: float,
    ) -> None:
        """Populate audit fields on the result (both success and error results).

        Wraps stable_hash calls to convert canonicalization errors to
        PluginContractViolation: stable_hash calls canonical_json, which
        rejects NaN, Infinity, and non-serializable types. Per CLAUDE.md:
        plugin bugs must crash with clear error messages.
        """
        result.input_hash = input_hash
        try:
            if result.row is not None:
                result.output_hash = stable_hash(result.row)
            elif result.rows is not None:
                result.output_hash = stable_hash(result.rows)
            else:
                result.output_hash = None
        except (TypeError, ValueError) as e:
            raise PluginContractViolation(
                f"Transform '{transform.name}' emitted non-canonical data: {e}. "
                f"Ensure output contains only JSON-serializable types. "
                f"Use None instead of NaN for missing values."
            ) from e
        result.duration_ms = duration_ms

    def _prepare_success_completion(
        self,
        *,
        result: TransformResult,
        transform: TransformProtocol,
        token: TokenInfo,
        run_id: str,
        node_id: str,
    ) -> dict[str, Any] | list[dict[str, Any]]:
        """Extract output data + record contract evolution (success-audit phase, part 2).

        Runs immediately before the shell completes the state as COMPLETED.
        Owns the has-output-data invariant, output-data extraction for the
        audit trail, and output-contract evolution recording. Recording
        happens BEFORE guard.complete() in the shell so a recording failure
        auto-fails the state (no "completed-then-crash" window — B1
        terminality doctrine).

        Returns:
            Output data (single row dict, or list of row dicts for
            multi-row results) for the COMPLETED node-state record.
        """
        # TransformResult.success() or success_multi() always sets output data
        if not result.has_output_data:
            raise RuntimeError(f"Transform '{transform.name}' returned success but has no output data")

        # Extract dicts for audit trail (Tier 1: full trust - store plain dicts)
        # Transforms return PipelineRow — extract underlying dicts for storage
        output_data: dict[str, Any] | list[dict[str, Any]]
        if result.row is not None:
            output_data = result.row.to_dict()
        else:
            if result.rows is None:
                raise OrchestrationInvariantError("has_output_data guarantees rows when row is None")
            output_data = [r.to_dict() for r in result.rows]

        # Record the transform's output contract BEFORE completing the state.
        # This ensures that if contract recording fails, the state is
        # auto-completed as FAILED by the guard (no "completed-then-crash"
        # window).  Fix for B1 terminality bug.
        #
        # Use the contract from the result directly — the plugin emitted
        # it and success_multi() guarantees all rows share the same instance.
        output_contract = None
        if result.row is not None:
            output_contract = result.row.contract
        elif result.rows is not None and result.rows:
            output_contract = result.rows[0].contract

        if output_contract is not None and output_contract is not token.row_data.contract:
            if self._data_flow is None:
                raise OrchestrationInvariantError("TransformExecutor.data_flow is None but contract evolution requires DataFlowRepository")
            self._data_flow.update_node_output_contract(
                run_id=run_id,
                node_id=node_id,
                contract=output_contract,
            )

        return output_data

    def execute_transform(
        self,
        transform: TransformProtocol,
        token: TokenInfo,
        ctx: PluginContext,
        attempt: int = 0,
    ) -> tuple[TransformResult, TokenInfo, str | None]:
        """Execute a transform with full audit recording and error routing.

        This method handles a SINGLE ATTEMPT. Retry logic is the caller's
        responsibility (e.g., RetryManager wraps this for retryable transforms).
        Each attempt gets its own node_state record with attempt number tracked
        by the caller.

        Supports two execution modes:
        1. Synchronous: transform.process() returns TransformResult immediately
        2. Asynchronous row-pipelined batch runtime: transform.accept() submits work,
           results flow through output port and are awaited synchronously

        Error Routing:
        - TransformResult.error() is a LEGITIMATE processing failure
        - Routes to configured sink via transform.on_error
        - RuntimeError if transform errors without on_error config
        - Exceptions are BUGS and propagate (not routed)

        The step position in the DAG is resolved internally via StepResolver
        using transform.node_id, rather than being passed as a parameter.

        Resume state (attempt offset and checkpoint provenance) is read from the
        token itself (token.resume_attempt_offset, token.resume_checkpoint_id).
        These fields propagate automatically via TokenInfo.with_updated_data()
        (which uses dataclasses.replace) so no explicit threading is needed.

        Args:
            transform: Transform plugin to execute
            token: Current token with row data; token.resume_attempt_offset and
                token.resume_checkpoint_id carry the resume state for this token.
            ctx: Plugin context
            attempt: Attempt number for retry tracking (0-indexed, default 0)

        Returns:
            Tuple of (TransformResult with audit fields, updated TokenInfo, error_sink)
            where error_sink is:
            - None if transform succeeded
            - "discard" if transform errored and on_error == "discard"
            - The sink name if transform errored and on_error is a sink name

        Raises:
            Exception: Re-raised from transform.process() after recording failure
            RuntimeError: Transform returned error but has no on_error configured
        """
        if transform.node_id is None:
            raise OrchestrationInvariantError(f"Transform '{transform.name}' executed without node_id - orchestrator bug")
        # Narrowed once; all downstream node-id plumbing uses this local.
        node_id = transform.node_id

        # Resolve step position from node_id (injected StepResolver)
        step = self._step_resolver(NodeID(node_id))

        # Extract dict from PipelineRow for hashing and Landscape recording
        # Landscape stores raw dicts, not PipelineRow objects
        input_dict = token.row_data.to_dict()
        input_hash = stable_hash(input_dict)

        # Detect row-pipelined concurrent transforms (accept/connect_output pattern).
        # ``is_batch_aware`` is intentionally not used here: that flag belongs to
        # aggregation via BatchTransformProtocol, a separate concept.
        batch_runtime: BatchTransformRuntimeProtocol | None = (
            transform if isinstance(transform, BatchTransformRuntimeProtocol) and transform.batch_runtime_enabled else None
        )

        # NodeStateGuard guarantees the node state reaches terminal status.
        # If any unhandled exception occurs before guard.complete() is called
        # (e.g., in output hashing or contract evolution), the guard auto-
        # completes the state as FAILED.
        with NodeStateGuard(
            self._execution,
            token_id=token.token_id,
            node_id=node_id,
            run_id=ctx.run_id,
            step_index=step,
            input_data=input_dict,
            # resume_attempt_offset is the generation base (run-1 max+1 for a re-driven token;
            # 0 for run-1 tokens); `attempt` is the tenacity retry index within this generation.
            attempt=token.resume_attempt_offset + attempt,
            resume_checkpoint_id=token.resume_checkpoint_id,
        ) as guard:
            # --- PREFLIGHT (pre-invocation checks) ---
            # Lifecycle guard, field-collision enforcement, pre-emission
            # declaration-contract dispatch (ADR-010/ADR-013), and input-schema
            # validation. Violations raise and the guard auto-fails the state.
            effective_input_fields, static_contract = self._run_preflight(
                transform=transform,
                token=token,
                input_dict=input_dict,
                run_id=ctx.run_id,
                node_id=node_id,
            )

            # Set per-token execution metadata only for this operation. The
            # shared context is reused across plugin calls, so these fields
            # must restore even when transform execution or post-processing
            # raises.
            with plugin_context_scope(
                ctx,
                state_id=guard.state_id,
                node_id=node_id,
                contract=token.row_data.contract,
                token=token,
            ):
                # Execute with timing and span
                # Pass token_id for accurate child token attribution in traces
                # Pass node_id for disambiguation when multiple plugin instances exist
                with self._spans.transform_span(
                    transform.name,
                    node_id=node_id,
                    input_hash=input_hash,
                    token_id=token.token_id,
                ):
                    start = time.perf_counter()
                    try:
                        # Invocation-mode selection (sync process() vs batch-runtime
                        # accept()+wait) lives in _invoke_transform; timing, FAILED
                        # completion, and timeout eviction stay here with the guard.
                        result = self._invoke_transform(
                            transform=transform,
                            batch_runtime=batch_runtime,
                            token=token,
                            ctx=ctx,
                            state_id=guard.state_id,
                        )
                        duration_ms = (time.perf_counter() - start) * 1000
                    except contract_errors.TIER_1_ERRORS:
                        raise  # Tier 1 errors must crash — never record as row FAILED
                    except Exception as e:
                        duration_ms = (time.perf_counter() - start) * 1000
                        # Record failure
                        error = ExecutionError(
                            exception=str(e),
                            exception_type=type(e).__name__,
                        )
                        guard.complete(
                            NodeStateStatus.FAILED,
                            duration_ms=duration_ms,
                            error=error,
                        )

                        # For TimeoutError on batch transforms, evict the buffer entry
                        # to prevent FIFO blocking on retry attempts.
                        #
                        # The eviction flow:
                        # 1. First attempt times out at waiter.wait()
                        # 2. We call evict_submission() to remove buffer entry
                        # 3. Retry attempt gets new sequence number and can proceed
                        # 4. Original worker may still complete, but result is discarded
                        if isinstance(e, TimeoutError) and batch_runtime is not None:
                            try:
                                batch_runtime.evict_submission(token.token_id, guard.state_id)
                            except Exception as evict_err:
                                raise RuntimeError(f"Failed to evict timed-out submission for token {token.token_id}") from evict_err

                        raise

                # -- Post-processing (GUARDED by NodeStateGuard) --
                # If any of the following steps raise before guard.complete() is
                # called, the guard auto-completes the state as FAILED in __exit__.

                # Post-emission declaration-contract dispatch + output-schema
                # validation. Runs BEFORE output hashing below so declaration
                # violations keep attribution over canonicalization failures.
                if result.status == "success":
                    self._verify_success_emissions(
                        result=result,
                        transform=transform,
                        token=token,
                        run_id=ctx.run_id,
                        node_id=node_id,
                        static_contract=static_contract,
                        effective_input_fields=effective_input_fields,
                    )

                # Populate audit fields (success and error results alike); a
                # canonicalization failure raises PluginContractViolation.
                self._populate_result_audit_fields(
                    result=result,
                    transform=transform,
                    input_hash=input_hash,
                    duration_ms=duration_ms,
                )

                # Initialize error_sink - will be set if transform errors with on_error configured
                error_sink: str | None = None

                # Complete node state
                if result.status == "success":
                    # Output-data extraction + contract-evolution recording
                    # (record-before-complete — B1 terminality doctrine: a
                    # recording failure must auto-fail the state via the guard).
                    output_data = self._prepare_success_completion(
                        result=result,
                        transform=transform,
                        token=token,
                        run_id=ctx.run_id,
                        node_id=node_id,
                    )

                    # NOW complete as COMPLETED — all validation has succeeded
                    guard.complete(
                        NodeStateStatus.COMPLETED,
                        output_data=output_data,
                        duration_ms=duration_ms,
                        success_reason=result.success_reason,
                        context_after=result.context_after,
                    )

                    # Update token with new PipelineRow, preserving all lineage metadata
                    # For multi-row results, keep original row_data (engine will expand tokens later)
                    if result.row is not None:
                        # Single-row result: transforms return PipelineRow with correct contract
                        updated_token = token.with_updated_data(result.row)
                    else:
                        # Multi-row result: keep original row_data (engine will expand tokens later)
                        updated_token = token.with_updated_data(token.row_data)
                else:
                    # Transform returned error status (not exception)
                    # This is a LEGITIMATE processing failure, not a bug

                    # Handle error routing - on_error is part of TransformProtocol
                    on_error = transform.on_error
                    # on_error is always set (required by TransformSettings) — Tier 1 invariant
                    if on_error is None:
                        raise OrchestrationInvariantError(
                            f"Transform '{transform.name}' has on_error=None — this should be impossible since TransformSettings requires on_error"
                        )

                    # Set error_sink so caller knows where the error was routed
                    error_sink = on_error

                    # result.reason MUST be set for error results - TransformResult.error() requires it.
                    # If None, that's a bug in the transform (constructed error result without reason).
                    if result.reason is None:
                        raise OrchestrationInvariantError(
                            f"Transform '{transform.name}' returned error but reason is None. "
                            'Use TransformResult.error({{"reason": "...", ...}}) to create error results.'
                        )
                    sanitized_reason = _scrub_transform_error_details(result.reason)
                    result.reason = sanitized_reason

                    # Record transform_error + DIVERT routing_event BEFORE terminal
                    # completion (record-before-complete, elspeth-2d65e04912 —
                    # same B1 doctrine as the success path above): once
                    # guard.complete() runs the guard stands down, and a failing
                    # audit write would escape its terminality protection. If a
                    # write raises here, the guard auto-fails the state
                    # (phase='executor_post_process', carrying the audit-write
                    # error rather than result.reason) and the error propagates —
                    # fail-closed, never completed-then-crashed.
                    record_transform_error_with_routing(
                        ctx=ctx,
                        execution=self._execution,
                        error_edge_ids=self._error_edge_ids,
                        state_id=guard.state_id,
                        token=token,
                        transform=transform,
                        row=input_dict,  # Use extracted dict for Landscape recording
                        error_details=sanitized_reason,
                        on_error=on_error,
                    )

                    guard.complete(
                        NodeStateStatus.FAILED,
                        duration_ms=duration_ms,
                        error=sanitized_reason,
                        context_after=result.context_after,
                    )

                    updated_token = token

        return result, updated_token, error_sink
