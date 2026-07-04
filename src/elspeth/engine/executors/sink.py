"""SinkExecutor - wraps sink.write() with artifact recording."""

import logging
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime

from pydantic import ValidationError

import elspeth.contracts.errors as contract_errors
from elspeth.contracts import (
    Artifact,
    ExecutionError,
    NodeStateOpen,
    PendingOutcome,
    SinkProtocol,
    TokenInfo,
)
from elspeth.contracts.audit import TokenRef
from elspeth.contracts.declaration_contracts import (
    AggregateDeclarationContractViolation,
    BoundaryInputs,
    BoundaryOutputs,
    DeclarationContractViolation,
)
from elspeth.contracts.diversion import SinkWriteResult
from elspeth.contracts.enums import NodeStateStatus, RoutingMode, TerminalOutcome, TerminalPath
from elspeth.contracts.errors import (
    AuditIntegrityError,
    FrameworkBugError,
    OrchestrationInvariantError,
    PluginContractViolation,
    SinkDiversionReason,
    SinkTransactionalInvariantError,
)
from elspeth.contracts.plugin_context import PluginContext
from elspeth.contracts.results import ArtifactDescriptor
from elspeth.contracts.schema_contract import SchemaContract
from elspeth.core.landscape.data_flow_repository import DataFlowRepository
from elspeth.core.landscape.errors import LandscapeRecordError
from elspeth.core.landscape.execution_repository import ExecutionRepository
from elspeth.core.operations import track_operation
from elspeth.engine._error_hash import compute_error_hash
from elspeth.engine.executors.declaration_dispatch import run_boundary_checks
from elspeth.engine.spans import SpanFactory

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class DiversionCounts:
    """Split sink diversion counts by ADR-019 terminal flavor."""

    failsink_mode: int = 0
    discard_mode: int = 0

    @property
    def total(self) -> int:
        return self.failsink_mode + self.discard_mode


class SinkExecutor:
    """Executes sinks with artifact recording.

    Wraps sink.write() with a three-phase flow:
    1. Call sink.write() inside track_operation (discover diversions)
    2. Open/complete node_states and record outcomes for primary tokens
    3. Route diverted tokens to failsink or discard with audit trail
    4. Register artifacts and emit OpenTelemetry span

    CRITICAL: Every token reaching a sink gets a node_state at the primary
    sink. Accepted rows get COMPLETED; diverted rows get FAILED (the row
    didn't reach its destination). Failsink-mode diverted tokens also get
    a second node_state at the failsink (COMPLETED — the row was written
    there). Discard-mode diverted tokens have only the primary FAILED state.

    Note: Unlike TransformExecutor/GateExecutor/AggregationExecutor, SinkExecutor
    does NOT use StepResolver. Sinks are not DAG processing nodes — their step is
    always max(processing_steps) + 1, computed by RowProcessor.resolve_sink_step()
    and passed as step_in_pipeline by the orchestrator. This is intentional: sinks
    exist after all processing nodes and have a fixed, deterministic step position.

    Example:
        executor = SinkExecutor(execution, data_flow, span_factory, run_id)
        artifact, diversion_counts = executor.write(
            sink=my_sink,
            tokens=tokens_to_write,
            ctx=ctx,
            step_in_pipeline=5,
            sink_name="output",
            pending_outcome=pending,
        )
        print(diversion_counts.total)
    """

    def __init__(
        self,
        execution: ExecutionRepository,
        data_flow: DataFlowRepository,
        span_factory: SpanFactory,
        run_id: str,
    ) -> None:
        """Initialize executor.

        Args:
            execution: Execution repository for node states, routing, artifacts
            data_flow: Data flow repository for token outcomes
            span_factory: Span factory for tracing
            run_id: Run identifier for artifact registration
        """
        self._execution = execution
        self._data_flow = data_flow
        self._spans = span_factory
        self._run_id = run_id

    def _complete_states_failed(
        self,
        *,
        states: list[tuple[TokenInfo, NodeStateOpen]],
        duration_ms: float,
        error: ExecutionError,
    ) -> None:
        """Complete all opened sink states as FAILED."""
        if not states:
            return
        per_token_ms = duration_ms / len(states)
        for _, state in states:
            self._execution.complete_node_state(
                state_id=state.state_id,
                status=NodeStateStatus.FAILED,
                duration_ms=per_token_ms,
                error=error,
            )

    def _best_effort_cleanup(
        self,
        states: list[tuple[TokenInfo, NodeStateOpen]],
        original_error: Exception,
        phase: str,
    ) -> None:
        """Best-effort cleanup of OPEN states before re-raising a system error.

        On FrameworkBugError/AuditIntegrityError, the system is crashing. But
        leaving node_states permanently OPEN is itself a Tier 1 violation —
        they falsely claim "in progress" when no processing is happening.
        Attempt to close them as FAILED; if that also fails, log and let the
        original error propagate.
        """
        cleanup_error = ExecutionError(
            exception=str(original_error),
            exception_type=type(original_error).__name__,
            phase=phase,
        )
        try:
            self._complete_states_failed(
                states=states,
                duration_ms=0.0,
                error=cleanup_error,
            )
        except contract_errors.TIER_1_ERRORS:
            raise  # Audit corruption during cleanup is higher priority than original error
        except Exception as cleanup_exc:
            logger.warning(
                "Best-effort cleanup of %d OPEN states failed during %s crash — "
                "states may remain OPEN. Original error: %s. Cleanup error: %s: %s",
                len(states),
                type(original_error).__name__,
                original_error,
                type(cleanup_exc).__name__,
                cleanup_exc,
            )

    @staticmethod
    def _validate_sink_input(
        sink: SinkProtocol,
        rows: list[dict[str, object]],
        *,
        skip_schema: bool = False,
        contracts: list[SchemaContract] | None = None,
    ) -> None:
        """Validate rows against a sink's input schema and required fields.

        Args:
            sink: Sink to validate against.
            rows: Row dicts to validate.
            skip_schema: If True, skip input_schema.model_validate() and only
                check required fields. Used for failsink validation where the
                executor injects enrichment fields (__diversion_*) that are
                outside the failsink's declared schema.
            contracts: Optional per-row SchemaContracts for context-aware error
                messages. When provided, a missing-field error annotates any
                field that is optional in the row's contract, pointing at
                coalesce merge as the likely root cause.
        """
        if not skip_schema:
            for row in rows:
                try:
                    sink.input_schema.model_validate(row)
                except ValidationError as e:
                    raise PluginContractViolation(
                        f"Sink '{sink.name}' input validation failed: {e}. This indicates an upstream transform/source schema bug."
                    ) from e

        if sink.declared_required_fields:
            # TWO-LAYER SINK INVARIANT ARCHITECTURE (ADR-010 §H2 landing scope F3).
            #
            # This inline check is the TRANSACTIONAL BACKSTOP (Layer 2).
            # It runs INSIDE the sink's commit boundary where the dispatcher-
            # owned pre-write contract (Layer 1 — SinkRequiredFieldsContract)
            # cannot see partial state. The two layers serve different audit
            # purposes:
            #
            #   * Layer 1 (pre-write declaration contract): intent validation.
            #     Triage SQL: WHERE exception_type = 'SinkRequiredFieldsViolation'
            #     (the concrete 2C contract class).
            #   * Layer 2 (THIS check): transactional backstop.
            #     Triage SQL: WHERE exception_type = 'SinkTransactionalInvariantError'.
            #
            # Before F3 both layers raised PluginContractViolation and the
            # audit trail conflated pre-write contract failures with
            # commit-boundary state-divergence failures. The reclassification
            # to SinkTransactionalInvariantError separates the triage surfaces
            # so auditors can distinguish "intent validation failed" from
            # "state diverged mid-transaction".
            #
            # Both classes are in TIER_1_ERRORS (neither can be absorbed by
            # on_error routing).
            #
            # Caller invariant: when contracts is provided, len(contracts) == len(rows).
            # Both call sites build them from the same tokens iterable, so a desync
            # would be a refactor bug — assert it loudly rather than silently masking.
            if contracts is not None and len(contracts) != len(rows):
                raise OrchestrationInvariantError(
                    f"Sink '{sink.name}' _validate_sink_input received {len(rows)} rows "
                    f"but {len(contracts)} contracts. These must be paired 1:1."
                )
            for row_index, row in enumerate(rows):
                missing = sorted(f for f in sink.declared_required_fields if f not in row)
                if not missing:
                    continue
                # If a contract is available, annotate missing fields that are
                # marked optional in the contract (coalesce merge artifact).
                contract_context = ""
                if contracts is not None:
                    contract = contracts[row_index]
                    # Hoist the property out of the comprehension: it builds a
                    # new frozenset on every access, and we need it once per row.
                    required_in_contract = contract.required_field_names
                    # Use find_name() to handle the original-vs-normalized name
                    # namespace mismatch: declared_required_fields may carry raw
                    # field names while the contract indexes by normalized names.
                    optional_in_contract: list[str] = []
                    for missing_name in missing:
                        normalized = contract.find_name(missing_name)
                        if normalized is not None and normalized not in required_in_contract:
                            optional_in_contract.append(missing_name)
                    if optional_in_contract:
                        contract_context = (
                            f" Fields {optional_in_contract} are optional in the row's "
                            f"schema contract (likely from coalesce merge). "
                            f"Fix: ensure all branches produce these fields as required."
                        )
                raise SinkTransactionalInvariantError(
                    f"Sink '{sink.name}' row {row_index} is missing required fields "
                    f"{missing} at the transactional commit boundary. This is "
                    f"the Layer 2 transactional backstop (ADR-010 F3); "
                    f"if a Layer 1 pre-write SinkRequiredFieldsContract is "
                    f"registered it should have fired first. If Layer 1 "
                    f"passed and Layer 2 fired, row state diverged between "
                    f"contract evaluation and commit — investigate "
                    f"cross-token mutation or mid-batch field "
                    f"removal.{contract_context}"
                )

    @staticmethod
    def _build_boundary_error(
        *,
        exc: DeclarationContractViolation | AggregateDeclarationContractViolation | PluginContractViolation,
        phase: str,
    ) -> ExecutionError:
        return ExecutionError(
            exception=str(exc),
            exception_type=type(exc).__name__,
            phase=phase,
            context=exc.to_audit_dict(),
        )

    def _record_boundary_failure_outcomes(
        self,
        *,
        tokens: Sequence[TokenInfo],
        sink_name: str,
        phase: str,
        violation: (DeclarationContractViolation | AggregateDeclarationContractViolation | SinkTransactionalInvariantError),
    ) -> None:
        """Record FAILED token_outcomes for sink boundary failures before write."""
        base_context = dict(violation.to_audit_dict())
        failing_token_id = base_context["token_id"] if "token_id" in base_context else None
        failing_row_id = base_context["row_id"] if "row_id" in base_context else None
        error_hash = compute_error_hash(f"{type(violation).__name__}:{sink_name}:{phase}")

        for token in tokens:
            context = dict(base_context)
            context["token_id"] = token.token_id
            context["row_id"] = token.row_id
            if failing_token_id is not None:
                context["failing_token_id"] = failing_token_id
            if failing_row_id is not None:
                context["failing_row_id"] = failing_row_id
            try:
                self._data_flow.record_token_outcome(
                    ref=TokenRef(token_id=token.token_id, run_id=self._run_id),
                    outcome=TerminalOutcome.FAILURE,
                    path=TerminalPath.UNROUTED,
                    error_hash=error_hash,
                    context=context,
                )
            except LandscapeRecordError as record_failure:
                raise AuditIntegrityError(
                    f"Failed to record {type(violation).__name__} FAILED outcome for token "
                    f"{token.token_id!r} during {phase} at sink {sink_name!r}. "
                    f"Node states are already FAILED but terminal token_outcomes are incomplete. "
                    f"Recorder failure: {type(record_failure).__name__}: {record_failure}. "
                    f"Original violation: {violation!s}"
                ) from record_failure

    @staticmethod
    def _run_sink_boundary_checks(
        *,
        sink: SinkProtocol,
        rows: list[dict[str, object]],
        tokens: list[TokenInfo],
        run_id: str,
        node_id: str,
        row_contracts: Sequence[SchemaContract | None] | None,
    ) -> None:
        """Run Layer 1 boundary contracts once per row before sink validation."""
        for row_index, (token, row) in enumerate(zip(tokens, rows, strict=True)):
            row_contract = None if row_contracts is None else row_contracts[row_index]
            run_boundary_checks(
                inputs=BoundaryInputs(
                    plugin=sink,
                    node_id=node_id,
                    run_id=run_id,
                    row_id=token.row_id,
                    token_id=token.token_id,
                    static_contract=sink.declared_required_fields,
                    row_data=row,
                    row_contract=row_contract,
                ),
                outputs=BoundaryOutputs(),
            )

    def _merge_batch_contract(self, tokens: list[TokenInfo]) -> SchemaContract:
        """Merge the sink-bound tokens' row contracts into one batch contract.

        A merge failure is a framework bug (contracts that reached a sink should
        always be batch-mergeable); TIER_1/audit-integrity errors propagate
        untouched.
        """
        contract_merge_start = time.perf_counter()
        try:
            batch_contract = tokens[0].row_data.contract
            for token in tokens[1:]:
                batch_contract = batch_contract.merge_for_batch(token.row_data.contract)
        except contract_errors.TIER_1_ERRORS:
            raise
        except Exception as e:
            merge_duration_ms = (time.perf_counter() - contract_merge_start) * 1000
            raise FrameworkBugError(f"Contract merge failed after {merge_duration_ms:.1f}ms: {e}") from e
        return batch_contract

    def _open_primary_states(
        self,
        *,
        tokens: list[TokenInfo],
        rows: list[dict[str, object]],
        sink_node_id: str,
        step_in_pipeline: int,
        ctx: PluginContext,
    ) -> list[tuple[TokenInfo, NodeStateOpen]]:
        """Open node_states for ALL tokens at the primary sink (PRE-PHASE).

        Opened BEFORE I/O so that Phase 1 failures can record FAILED states,
        preserving the invariant that every token reaches a terminal state. We
        don't yet know which tokens will be diverted — that's discovered by
        sink.write() — so we open states for ALL tokens and partition later. On a
        begin failure, best-effort close any partially-opened states before
        re-raising (TIER_1 and generic paths both clean up then propagate).
        """
        all_states: list[tuple[TokenInfo, NodeStateOpen]] = []
        try:
            can_use_bulk_begin = type(self._execution) is ExecutionRepository and all(
                token.resume_attempt_offset == 0 and token.resume_checkpoint_id is None for token in tokens
            )
            if can_use_bulk_begin:
                opened_states = self._execution.begin_node_states_many(
                    tuple(
                        (
                            token.token_id,
                            sink_node_id,
                            ctx.run_id,
                            step_in_pipeline,
                            row,
                        )
                        for token, row in zip(tokens, rows, strict=True)
                    )
                )
                all_states.extend(zip(tokens, opened_states, strict=True))
            else:
                for token, input_dict in zip(tokens, rows, strict=True):
                    state = self._execution.begin_node_state(
                        token_id=token.token_id,
                        node_id=sink_node_id,
                        run_id=ctx.run_id,
                        step_index=step_in_pipeline,
                        input_data=input_dict,
                        attempt=token.resume_attempt_offset,
                        resume_checkpoint_id=token.resume_checkpoint_id,
                    )
                    all_states.append((token, state))
        except contract_errors.TIER_1_ERRORS as e:
            if all_states:
                self._best_effort_cleanup(all_states, e, "begin_node_state")
            raise
        except Exception as e:
            if all_states:
                self._best_effort_cleanup(all_states, e, "begin_node_state")
            raise
        return all_states

    def write(
        self,
        sink: SinkProtocol,
        tokens: list[TokenInfo],
        ctx: PluginContext,
        step_in_pipeline: int,
        *,
        sink_name: str,
        pending_outcome: PendingOutcome | None,
        failsink: SinkProtocol | None = None,
        failsink_name: str | None = None,
        failsink_edge_id: str | None = None,
        on_token_written: Callable[[TokenInfo], None] | None = None,
    ) -> tuple[Artifact | None, DiversionCounts]:
        """Write tokens to sink with artifact recording and failsink routing.

        CRITICAL: Creates a node_state for EACH token written AND records
        token outcomes. Node_states are opened BEFORE I/O so that Phase 1
        failures result in FAILED states (not silent drops). States are
        completed as COMPLETED only after sink.flush() confirms durability.

        This is the ONLY place terminal outcomes should be recorded for sink-bound
        tokens. Recording here (not in the orchestrator processing loop) ensures the
        token outcome contract is honored:
        - Invariant 3: "COMPLETED/ROUTED implies the token has a completed sink node_state"
        - Invariant 4: "Completed sink node_state implies a terminal token_outcome"

        Four-phase flow:
        - Pre-phase: Open node_states for ALL tokens at primary sink
        - Phase 1: Call sink.write() → discover diversions (FAILED on error)
        - Phase 2: Complete states for primary (non-diverted) tokens
        - Phase 3: Handle diversions (failsink write or discard)

        Args:
            sink: Sink plugin to write to
            tokens: Tokens to write (may be empty)
            ctx: Plugin context
            step_in_pipeline: Current position in DAG (Orchestrator is authority)
            sink_name: Name of the sink (for token_outcome recording)
            pending_outcome: PendingOutcome containing outcome and optional error_hash.
                    Required - all sink-bound tokens must have their outcome recorded.
            failsink: Resolved failsink instance (or None for discard mode)
            failsink_name: Config-level name of the failsink (for outcome recording)
            failsink_edge_id: Edge ID of the __failsink__ DIVERT edge in the DAG
            on_token_written: Optional callback called for each token after its
                             path completes durably. Primary tokens are checkpointed
                             after Phase 2, diverted tokens after Phase 3.

        Returns:
            Tuple of (Artifact if tokens were written else None, diversion counts)

        Raises:
            Exception: Propagated from sink.write(), sink.flush(), or failsink.write()
        """
        if not tokens:
            return None, DiversionCounts()

        # pending_outcome is required for all sink-bound tokens.
        # PendingTokenMap allows None in its type alias, but _route_to_sink()
        # always wraps in PendingOutcome. None here means a routing bug.
        if pending_outcome is None:
            raise OrchestrationInvariantError(
                f"Sink '{sink_name}' received pending_outcome=None — all sink-bound tokens must have a PendingOutcome."
            )

        # Extract dicts from PipelineRow for sink write
        rows = [t.row_data.to_dict() for t in tokens]

        # Sink must have node_id assigned by orchestrator before execution
        if sink.node_id is None:
            raise OrchestrationInvariantError(f"Sink '{sink.name}' executed without node_id - orchestrator bug")
        sink_node_id: str = sink.node_id

        # Synchronize context contract to the sink-bound tokens.
        ctx.contract = self._merge_batch_contract(tokens)

        # CRITICAL: Clear state_id before entering operation context.
        ctx.state_id = None

        # ── PRE-PHASE: Open node_states for ALL tokens at primary sink ──
        all_states = self._open_primary_states(
            tokens=tokens,
            rows=rows,
            sink_node_id=sink_node_id,
            step_in_pipeline=step_in_pipeline,
            ctx=ctx,
        )

        # Index by token_id for O(1) lookup in Phases 2 and 3.
        state_by_token_id: dict[str, NodeStateOpen] = {token.token_id: state for token, state in all_states}

        primary_states_closed_by_boundary_failure = False

        # ── PHASE 1: External I/O (inside track_operation) ──
        # If any operation here raises, complete ALL pre-opened states as FAILED
        # before re-raising — no token may exit this method without a terminal state.
        try:
            with track_operation(
                recorder=self._execution,
                run_id=self._run_id,
                node_id=sink_node_id,
                operation_type="sink_write",
                ctx=ctx,
                input_data={"sink_plugin": sink.name, "row_count": len(tokens)},
            ) as handle:
                sink_token_ids = [t.token_id for t in tokens]
                with self._spans.sink_span(
                    sink.name,
                    node_id=sink_node_id,
                    token_ids=sink_token_ids,
                ):
                    row_contracts = [t.row_data.contract for t in tokens]
                    try:
                        self._run_sink_boundary_checks(
                            sink=sink,
                            rows=rows,
                            tokens=tokens,
                            run_id=ctx.run_id,
                            node_id=sink_node_id,
                            row_contracts=row_contracts,
                        )

                        # Centralized input validation (before sink.write).
                        # Wrong types at a sink boundary are upstream plugin bugs (Tier 2).
                        # Pass per-row contracts for context-aware error messages.
                        self._validate_sink_input(sink, rows, contracts=row_contracts)
                    except (
                        DeclarationContractViolation,
                        AggregateDeclarationContractViolation,
                        SinkTransactionalInvariantError,
                    ) as boundary_violation:
                        self._complete_states_failed(
                            states=all_states,
                            duration_ms=0.0,
                            error=self._build_boundary_error(exc=boundary_violation, phase="sink_write"),
                        )
                        primary_states_closed_by_boundary_failure = True
                        self._record_boundary_failure_outcomes(
                            tokens=tokens,
                            sink_name=sink.name,
                            phase="sink_write",
                            violation=boundary_violation,
                        )
                        raise

                    # Reset diversion log and call sink.write()
                    sink._reset_diversion_log()
                    start = time.perf_counter()
                    write_result = sink.write(rows, ctx)
                    # First-party plugin-contract guard: SinkWriteResult is this
                    # system-owned sink's typed return contract (see Plugin
                    # Ownership), so a wrong type is a plugin bug we crash on. The
                    # isinstance->raise is offensive programming (a maximally
                    # informative crash), not coercion and not silent recovery —
                    # i.e. the prescribed CRASH response, not the defensive
                    # isinstance-to-suppress that R5 targets.
                    if not isinstance(write_result, SinkWriteResult):
                        raise PluginContractViolation(
                            f"Sink '{sink.name}' returned {type(write_result).__name__}, "
                            f"expected SinkWriteResult. This is a sink plugin bug."
                        )
                    artifact_info = write_result.artifact
                    # First-party plugin-contract guard: SinkWriteResult.artifact
                    # must be an ArtifactDescriptor under this system-owned sink's
                    # typed contract (see Plugin Ownership), so a wrong type is a
                    # plugin bug we crash on. The isinstance->raise is offensive
                    # programming (a maximally informative crash), not coercion and
                    # not silent recovery — i.e. the prescribed CRASH response, not
                    # the defensive isinstance-to-suppress that R5 targets.
                    if not isinstance(artifact_info, ArtifactDescriptor):
                        raise PluginContractViolation(
                            f"Sink '{sink.name}' returned SinkWriteResult with artifact of type "
                            f"{type(artifact_info).__name__}, expected ArtifactDescriptor. "
                            f"This is a sink plugin bug."
                        )
                    diversions = write_result.diversions
                    duration_ms = (time.perf_counter() - start) * 1000

                    # Validate diversion indices against the batch we passed in.
                    # SinkWriteResult.__post_init__ already rejects duplicates;
                    # here we check range (only the executor knows the batch size).
                    batch_size = len(tokens)
                    for d in diversions:
                        if d.row_index >= batch_size:
                            raise PluginContractViolation(
                                f"Sink '{sink.name}' returned diversion with row_index={d.row_index} "
                                f"but batch has only {batch_size} rows (valid range: 0..{batch_size - 1}). "
                                f"This is a sink plugin bug."
                            )

                # Flush primary sink for durability
                sink.flush()

                # Set output data on operation handle for audit trail
                handle.output_data = {
                    "artifact_path": artifact_info.path_or_uri,
                    "content_hash": artifact_info.content_hash,
                }
        except contract_errors.TIER_1_ERRORS as e:
            if not primary_states_closed_by_boundary_failure:
                self._best_effort_cleanup(all_states, e, "sink_write")
            raise
        except Exception as e:
            io_error = ExecutionError(
                exception=str(e),
                exception_type=type(e).__name__,
                phase="sink_write",
            )
            if not primary_states_closed_by_boundary_failure:
                self._complete_states_failed(
                    states=all_states,
                    duration_ms=0.0,
                    error=io_error,
                )
            raise

        # ── PHASE 2: Partition and complete primary tokens ──
        diverted_indices = {d.row_index for d in diversions}
        primary_tokens = [(token, i) for i, token in enumerate(tokens) if i not in diverted_indices]
        diverted_tokens = [(token, i) for i, token in enumerate(tokens) if i in diverted_indices]

        artifact: Artifact | None = None

        if primary_tokens:
            # Retrieve pre-opened states for primary tokens.
            primary_states: list[tuple[TokenInfo, NodeStateOpen]] = [
                (token, state_by_token_id[token.token_id]) for token, _ in primary_tokens
            ]

            # Amortize batch write time across ALL tokens (including diverted)
            # since sink.write() processed the entire batch
            per_token_ms = duration_ms / len(tokens)
            primary_sink_outputs: list[tuple[str, dict[str, object], float]] = []
            for token, state in primary_states:
                output_dict = token.row_data.to_dict()
                primary_sink_outputs.append(
                    (
                        state.state_id,
                        {
                            "row": output_dict,
                            "artifact_path": artifact_info.path_or_uri,
                            "content_hash": artifact_info.content_hash,
                        },
                        per_token_ms,
                    )
                )
            if type(self._execution) is ExecutionRepository:
                self._execution.complete_node_states_completed_many(tuple(primary_sink_outputs))
            else:
                for state_id, sink_output, duration in primary_sink_outputs:
                    self._execution.complete_node_state(
                        state_id=state_id,
                        status=NodeStateStatus.COMPLETED,
                        output_data=sink_output,
                        duration_ms=duration,
                    )

            # Register artifact (linked to first primary state)
            first_state = primary_states[0][1]
            artifact = self._execution.register_artifact(
                run_id=self._run_id,
                state_id=first_state.state_id,
                sink_node_id=sink_node_id,
                artifact_type=artifact_info.artifact_type,
                path=artifact_info.path_or_uri,
                content_hash=artifact_info.content_hash,
                size_bytes=artifact_info.size_bytes,
            )

            # Record durable outcomes for primary tokens.
            for token, _ in primary_states:
                join_group_id: str | None = None
                if pending_outcome.path == TerminalPath.COALESCED:
                    join_group_id = token.join_group_id
                    if join_group_id is None:
                        raise OrchestrationInvariantError(
                            f"(SUCCESS, COALESCED) pending outcome for token {token.token_id!r} "
                            "requires token.join_group_id before sink recording"
                        )
                self._data_flow.record_token_outcome(
                    ref=TokenRef(token_id=token.token_id, run_id=self._run_id),
                    outcome=pending_outcome.outcome,
                    path=pending_outcome.path,
                    error_hash=pending_outcome.error_hash,
                    sink_name=sink_name,
                    join_group_id=join_group_id,
                )

            # Checkpoint callback — only for primary tokens.
            # Failures crash with AuditIntegrityError: the sink write is durable
            # but the checkpoint record is missing, leaving the audit trail
            # inconsistent. Logging-and-continuing would silently cause duplicate
            # writes on resume — a worse outcome than crashing.
            if on_token_written is not None:
                for token, _ in primary_tokens:
                    try:
                        on_token_written(token)
                    except contract_errors.TIER_1_ERRORS:
                        raise
                    except Exception as exc:
                        raise AuditIntegrityError(
                            f"Checkpoint failed after durable sink write for token {token.token_id}. "
                            f"Sink artifact exists but no checkpoint record created — "
                            f"audit trail is inconsistent. "
                            f"Original error: {type(exc).__name__}: {exc}"
                        ) from exc

        # ── PHASE 3: Handle diversions ──
        # Diverted tokens already have node_states at the PRIMARY sink from
        # the pre-phase. These are the routing anchors — routing_event.state_id
        # points here. Failsink-mode tokens ALSO get a NEW state at the
        # failsink node (the destination).
        failsink_count = 0
        discard_count = 0
        if diverted_tokens:
            diversion_by_index = {d.row_index: d for d in diversions}

            # Retrieve pre-opened states for diverted tokens.
            primary_divert_states: list[tuple[TokenInfo, int, NodeStateOpen]] = [
                (token, idx, state_by_token_id[token.token_id]) for token, idx in diverted_tokens
            ]

            if failsink is not None:
                primary_divert_pairs = [(t, s) for t, _, s in primary_divert_states]
                diverted_only_tokens = [token for token, _idx, _state in primary_divert_states]
                # True once BOTH the primary divert anchors and any opened
                # failsink states have been completed by an inner handler —
                # the outer handlers below must not double-complete them.
                divert_states_closed = False
                failsink_states: list[tuple[TokenInfo, NodeStateOpen]] = []

                # Failsink mode: write enriched rows to failsink
                try:
                    if failsink.node_id is None:
                        raise OrchestrationInvariantError(f"Failsink '{failsink.name}' executed without node_id - orchestrator bug")
                    if failsink_edge_id is None:
                        raise OrchestrationInvariantError("failsink_edge_id is None but failsink is not None — orchestrator bug")
                    if failsink_name is None:
                        raise OrchestrationInvariantError("failsink_name is None but failsink is not None — orchestrator bug")
                    failsink_node_id: str = failsink.node_id

                    # Build enriched rows — keyed by token_id so failsink node states
                    # can record the enriched payload (what was actually written), not
                    # the original row data.
                    iso_ts = datetime.now(UTC).isoformat()
                    enriched_rows: list[dict[str, object]] = []
                    enriched_by_token: dict[str, dict[str, object]] = {}
                    for token, idx, _state in primary_divert_states:
                        diversion = diversion_by_index[idx]
                        enriched_row = {
                            **diversion.row_data,
                            "__diversion_reason": diversion.reason,
                            "__diverted_from": sink_name,
                            "__diversion_timestamp": iso_ts,
                        }
                        enriched_rows.append(enriched_row)
                        enriched_by_token[token.token_id] = enriched_row

                    # Open node_states at the failsink node (destination) BEFORE
                    # the external failsink I/O, mirroring the primary pre-phase
                    # (elspeth-adaca19c75): a durable failsink write must never
                    # exist without a failsink node_state — a crash between
                    # flush and audit recording would otherwise leave a durable
                    # artifact with no state/routing_event/outcome/checkpoint.
                    # Use the enriched payload (what will be written), not the
                    # original row data — the audit trail must reflect the
                    # persisted data.
                    try:
                        for token, _idx, _primary_state in primary_divert_states:
                            input_dict = enriched_by_token[token.token_id]
                            state = self._execution.begin_node_state(
                                token_id=token.token_id,
                                node_id=failsink_node_id,
                                run_id=ctx.run_id,
                                # Failsink handling is a second sink visit for the
                                # same token. Record it at the next path position
                                # so it cannot collide with the primary sink
                                # node_state's (token_id, step_index, attempt).
                                step_index=step_in_pipeline + 1,
                                input_data=input_dict,
                                attempt=token.resume_attempt_offset,
                                resume_checkpoint_id=token.resume_checkpoint_id,
                            )
                            failsink_states.append((token, state))
                    except contract_errors.TIER_1_ERRORS as e:
                        # Best-effort: close partially-opened failsink states +
                        # primary divert states before the crash propagates.
                        all_open = failsink_states + primary_divert_pairs
                        if all_open:
                            self._best_effort_cleanup(all_open, e, "begin_node_state_failsink")
                        divert_states_closed = True
                        raise
                    except Exception as e:
                        begin_error = ExecutionError(
                            exception=str(e),
                            exception_type=type(e).__name__,
                            phase="begin_node_state_failsink",
                        )
                        # Close any partially-opened failsink states
                        if failsink_states:
                            self._complete_states_failed(
                                states=failsink_states,
                                duration_ms=0.0,
                                error=begin_error,
                            )
                        # Also close the already-open primary divert states.
                        self._complete_states_failed(
                            states=primary_divert_pairs,
                            duration_ms=0.0,
                            error=begin_error,
                        )
                        divert_states_closed = True
                        raise

                    with track_operation(
                        recorder=self._execution,
                        run_id=self._run_id,
                        node_id=failsink_node_id,
                        operation_type="sink_write",
                        ctx=ctx,
                        input_data={
                            "sink_plugin": failsink.name,
                            "row_count": len(primary_divert_states),
                            "diverted_from": sink_name,
                        },
                    ) as failsink_handle:
                        with self._spans.sink_span(
                            failsink.name,
                            node_id=failsink_node_id,
                            token_ids=[token.token_id for token in diverted_only_tokens],
                        ):
                            failsink._reset_diversion_log()
                            try:
                                self._run_sink_boundary_checks(
                                    sink=failsink,
                                    rows=enriched_rows,
                                    tokens=diverted_only_tokens,
                                    run_id=ctx.run_id,
                                    node_id=failsink_node_id,
                                    row_contracts=None,
                                )
                                # Validate enriched rows against failsink required fields.
                                # skip_schema=True because the executor injects __diversion_*
                                # fields that are outside the failsink's declared schema —
                                # a fixed-schema failsink (extra="forbid") would reject them.
                                # Required-field checking still catches missing upstream fields.
                                # Don't pass primary-path contracts here: the failsink's
                                # declared_required_fields are diagnostic-shaped (e.g.,
                                # __diversion_reason) and have no relationship to the
                                # primary contract's field optionality. Annotating with
                                # "optional in coalesce merge" would be misdirection —
                                # the operator is debugging a failsink write, not a
                                # missing primary field.
                                self._validate_sink_input(failsink, enriched_rows, skip_schema=True)
                            except (
                                DeclarationContractViolation,
                                AggregateDeclarationContractViolation,
                                SinkTransactionalInvariantError,
                            ) as boundary_violation:
                                self._complete_states_failed(
                                    states=primary_divert_pairs + failsink_states,
                                    duration_ms=0.0,
                                    error=self._build_boundary_error(exc=boundary_violation, phase="failsink_write"),
                                )
                                self._record_boundary_failure_outcomes(
                                    tokens=diverted_only_tokens,
                                    sink_name=failsink.name,
                                    phase="failsink_write",
                                    violation=boundary_violation,
                                )
                                divert_states_closed = True
                                raise

                            failsink_write_result = failsink.write(enriched_rows, ctx)
                            # First-party plugin-contract guard: SinkWriteResult is
                            # this system-owned failsink's typed return contract (see
                            # Plugin Ownership), so a wrong type is a plugin bug we
                            # crash on. The isinstance->raise is offensive programming
                            # (a maximally informative crash), not coercion and not
                            # silent recovery — i.e. the prescribed CRASH response,
                            # not the defensive isinstance-to-suppress that R5 targets.
                            if not isinstance(failsink_write_result, SinkWriteResult):
                                raise PluginContractViolation(
                                    f"Failsink '{failsink_name}' returned {type(failsink_write_result).__name__}, "
                                    f"expected SinkWriteResult. This is a failsink plugin bug."
                                )
                            failsink_artifact_info = failsink_write_result.artifact
                            # First-party plugin-contract guard:
                            # SinkWriteResult.artifact must be an ArtifactDescriptor
                            # under this system-owned failsink's typed contract (see
                            # Plugin Ownership), so a wrong type is a plugin bug we
                            # crash on. The isinstance->raise is offensive programming
                            # (a maximally informative crash), not coercion and not
                            # silent recovery — i.e. the prescribed CRASH response,
                            # not the defensive isinstance-to-suppress that R5 targets.
                            if not isinstance(failsink_artifact_info, ArtifactDescriptor):
                                raise PluginContractViolation(
                                    f"Failsink '{failsink_name}' returned SinkWriteResult with artifact of type "
                                    f"{type(failsink_artifact_info).__name__}, expected ArtifactDescriptor. "
                                    f"This is a failsink plugin bug."
                                )
                            if failsink_write_result.diversions:
                                raise FrameworkBugError(
                                    f"Failsink '{failsink_name}' produced {len(failsink_write_result.diversions)} "
                                    f"diversions during failsink write — failsinks must not divert rows."
                                )

                        failsink.flush()
                        failsink_handle.output_data = {
                            "artifact_path": failsink_artifact_info.path_or_uri,
                            "content_hash": failsink_artifact_info.content_hash,
                        }
                except contract_errors.TIER_1_ERRORS as e:
                    if not divert_states_closed:
                        self._best_effort_cleanup(primary_divert_pairs + failsink_states, e, "failsink_write")
                    raise
                except Exception as e:
                    if not divert_states_closed:
                        fs_write_error = ExecutionError(
                            exception=str(e),
                            exception_type=type(e).__name__,
                            phase="failsink_write",
                        )
                        self._complete_states_failed(
                            states=primary_divert_pairs + failsink_states,
                            duration_ms=0.0,
                            error=fs_write_error,
                        )
                    raise

                # Record routing_event anchored to PRIMARY sink state (the routing node),
                # complete primary state as FAILED, then complete failsink state.
                # This matches the quarantine pattern: routing_event lives at the
                # node that made the routing decision.
                #
                # Wrapped in try/except to clean up any remaining OPEN states
                # if a recorder call fails mid-loop (F3 fix from review).
                completed_primary_indices: set[int] = set()
                completed_failsink_indices: set[int] = set()
                try:
                    for loop_idx, ((token, idx, primary_state), (_, fs_state)) in enumerate(
                        zip(primary_divert_states, failsink_states, strict=True)
                    ):
                        diversion = diversion_by_index[idx]
                        reason: SinkDiversionReason = {"diversion_reason": diversion.reason}

                        # Routing event anchored to primary sink state
                        self._execution.record_routing_event(
                            state_id=primary_state.state_id,
                            edge_id=failsink_edge_id,
                            mode=RoutingMode.DIVERT,
                            reason=reason,
                        )

                        # Complete primary state as FAILED — the row didn't get where
                        # it was going. FAILED is a row state, not a system state:
                        # the pipeline is healthy, the row failed at this stop.
                        # Matches the quarantine pattern (core.py:1835).
                        divert_error = ExecutionError(
                            exception=diversion.reason,
                            exception_type="SinkDiversion",
                            phase="write",
                        )
                        self._execution.complete_node_state(
                            state_id=primary_state.state_id,
                            status=NodeStateStatus.FAILED,
                            output_data={"diverted_to": failsink_name, "reason": diversion.reason},
                            duration_ms=0.0,
                            error=divert_error,
                        )
                        completed_primary_indices.add(loop_idx)

                        # Complete failsink state (token written to failsink).
                        # Use enriched row — that's what was actually persisted.
                        failsink_output = {
                            "row": enriched_by_token[token.token_id],
                            "artifact_path": failsink_artifact_info.path_or_uri,
                            "content_hash": failsink_artifact_info.content_hash,
                        }
                        self._execution.complete_node_state(
                            state_id=fs_state.state_id,
                            status=NodeStateStatus.COMPLETED,
                            output_data=failsink_output,
                            duration_ms=0.0,
                        )
                        completed_failsink_indices.add(loop_idx)
                except contract_errors.TIER_1_ERRORS as e:
                    # Best-effort: close remaining OPEN states before crash.
                    remaining = [(t, s) for i, (t, _, s) in enumerate(primary_divert_states) if i not in completed_primary_indices] + [
                        (t, s) for i, (t, s) in enumerate(failsink_states) if i not in completed_failsink_indices
                    ]
                    if remaining:
                        self._best_effort_cleanup(remaining, e, "failsink_audit_recording")
                    raise
                except Exception as e:
                    # Close any remaining OPEN states from tokens not yet processed.
                    loop_error = ExecutionError(
                        exception=str(e),
                        exception_type=type(e).__name__,
                        phase="failsink_audit_recording",
                    )
                    remaining_primary = [(t, s) for i, (t, _, s) in enumerate(primary_divert_states) if i not in completed_primary_indices]
                    remaining_failsink = [(t, s) for i, (t, s) in enumerate(failsink_states) if i not in completed_failsink_indices]
                    if remaining_primary:
                        self._complete_states_failed(
                            states=remaining_primary,
                            duration_ms=0.0,
                            error=loop_error,
                        )
                    if remaining_failsink:
                        self._complete_states_failed(
                            states=remaining_failsink,
                            duration_ms=0.0,
                            error=loop_error,
                        )
                    raise

                # Register failsink artifact
                first_fs_state = failsink_states[0][1]
                failsink_artifact = self._execution.register_artifact(
                    run_id=self._run_id,
                    state_id=first_fs_state.state_id,
                    sink_node_id=failsink_node_id,
                    artifact_type=failsink_artifact_info.artifact_type,
                    path=failsink_artifact_info.path_or_uri,
                    content_hash=failsink_artifact_info.content_hash,
                    size_bytes=failsink_artifact_info.size_bytes,
                )

                # Record DIVERTED outcomes
                for token, idx, _primary_state in primary_divert_states:
                    diversion = diversion_by_index[idx]
                    error_hash = compute_error_hash(diversion.reason)
                    self._data_flow.record_token_outcome(
                        ref=TokenRef(token_id=token.token_id, run_id=self._run_id),
                        outcome=TerminalOutcome.TRANSIENT,
                        path=TerminalPath.SINK_FALLBACK_TO_FAILSINK,
                        error_hash=error_hash,
                        sink_name=failsink_name,
                        sink_node_id=failsink_node_id,
                        artifact_id=failsink_artifact.artifact_id,
                    )
                    failsink_count += 1

                # Checkpoint diverted tokens — failsink write is now durable.
                # Without this, a crash after failsink write but before the next
                # primary checkpoint leaves diverted tokens uncheckpointed,
                # causing duplicate failsink writes on resume.
                if on_token_written is not None:
                    for token, _idx, _state in primary_divert_states:
                        try:
                            on_token_written(token)
                        except contract_errors.TIER_1_ERRORS:
                            raise
                        except Exception as exc:
                            raise AuditIntegrityError(
                                f"Checkpoint failed after durable failsink write for diverted token {token.token_id}. "
                                f"Failsink artifact exists but no checkpoint record created — "
                                f"audit trail is inconsistent. "
                                f"Original error: {type(exc).__name__}: {exc}"
                            ) from exc

            else:
                # Discard mode: complete primary states and record DIVERTED outcomes.
                # No routing_event (no DAG edge for discard), no failsink write.
                for token, idx, primary_state in primary_divert_states:
                    diversion = diversion_by_index[idx]

                    # FAILED — the row didn't reach its destination (discarded).
                    discard_error = ExecutionError(
                        exception=diversion.reason,
                        exception_type="SinkDiscard",
                        phase="write",
                    )
                    self._execution.complete_node_state(
                        state_id=primary_state.state_id,
                        status=NodeStateStatus.FAILED,
                        output_data={"discarded": True, "reason": diversion.reason},
                        duration_ms=0.0,
                        error=discard_error,
                    )

                    error_hash = compute_error_hash(diversion.reason)
                    # ADR-019: discard-mode diversions are predicate-input
                    # failures, not transient failsink bookkeeping.
                    self._data_flow.record_token_outcome(
                        ref=TokenRef(token_id=token.token_id, run_id=self._run_id),
                        outcome=TerminalOutcome.FAILURE,
                        path=TerminalPath.SINK_DISCARDED,
                        error_hash=error_hash,
                        sink_name="__discard__",
                    )
                    discard_count += 1

                # Checkpoint diverted tokens — discard recording is now durable.
                # Discard is idempotent, but checkpointing keeps resume state consistent.
                if on_token_written is not None:
                    for token, _idx, _state in primary_divert_states:
                        try:
                            on_token_written(token)
                        except contract_errors.TIER_1_ERRORS:
                            raise
                        except Exception as exc:
                            raise AuditIntegrityError(
                                f"Checkpoint failed after discard recording for diverted token {token.token_id}. "
                                f"Original error: {type(exc).__name__}: {exc}"
                            ) from exc

        return artifact, DiversionCounts(failsink_mode=failsink_count, discard_mode=discard_count)
