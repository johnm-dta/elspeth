"""Effect-only sink execution with durable artifact and diversion recording."""

from __future__ import annotations

import logging
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC
from typing import TYPE_CHECKING, Protocol, runtime_checkable

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
from elspeth.contracts.audit import NodeState, NodeStateFailed, TokenRef
from elspeth.contracts.declaration_contracts import (
    AggregateDeclarationContractViolation,
    BoundaryInputs,
    BoundaryOutputs,
    DeclarationContractViolation,
)
from elspeth.contracts.diversion import RowDiversion
from elspeth.contracts.enums import NodeStateStatus, RoutingMode, TerminalOutcome, TerminalPath
from elspeth.contracts.errors import (
    AuditIntegrityError,
    FrameworkBugError,
    OrchestrationInvariantError,
    PluginContractViolation,
    SinkDiversionReason,
    SinkTransactionalInvariantError,
)
from elspeth.contracts.hashing import stable_hash
from elspeth.contracts.plugin_context import PluginContext
from elspeth.contracts.schema_contract import SchemaContract
from elspeth.contracts.sink_effects import (
    SinkEffectFinalizationMember,
    SinkEffectInputKind,
    SinkEffectMemberCandidate,
    SinkEffectPipelineMembersInput,
    SinkEffectReservationRequest,
    SinkEffectRole,
)
from elspeth.core.landscape.data_flow_repository import DataFlowRepository
from elspeth.core.landscape.errors import LandscapeRecordError
from elspeth.core.landscape.execution.sink_effect_identity import compute_pipeline_effect_identity, resolve_sink_effect_members
from elspeth.core.landscape.execution_repository import ExecutionRepository
from elspeth.engine._error_hash import compute_error_hash
from elspeth.engine.executors.declaration_dispatch import run_boundary_checks
from elspeth.engine.executors.sink_effects import (
    SinkEffectCoordinator,
    SinkEffectExecutionRequest,
    SinkEffectExecutionSeam,
)
from elspeth.engine.executors.sink_required_fields import _format_optional_missing_fields_context
from elspeth.engine.spans import SpanFactory

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from elspeth.core.landscape.factory import RecorderFactory


@runtime_checkable
class _BulkBeginNodeStateRepository(Protocol):
    def begin_node_states_many(
        self,
        entries: Sequence[tuple[str, str, str, int, Mapping[str, object]]],
    ) -> list[NodeStateOpen]: ...


@dataclass(frozen=True, slots=True)
class DiversionCounts:
    """Split sink diversion counts by ADR-019 terminal flavor."""

    failsink_mode: int = 0
    discard_mode: int = 0

    @property
    def total(self) -> int:
        return self.failsink_mode + self.discard_mode


@dataclass(frozen=True, slots=True)
class _EffectPrimaryWrite:
    artifact: Artifact
    effect_id: str
    diversions: tuple[RowDiversion, ...]
    diversion_error_hashes: Mapping[int, str]
    diversion_reason_hashes: Mapping[int, str]
    accepted_token_ids: frozenset[str]


class SinkExecutor:
    """Executes sinks with artifact recording.

    Every non-empty batch executes through the sink-effect protocol:
    1. Open or recover the batch's node states.
    2. Reconcile or publish the primary effect and finalize accepted members.
    3. Route diverted members through a linked failsink effect or durable discard.
    4. Checkpoint only after the selected terminal path is exact.

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
            effect_mode="write",
        )
        print(diversion_counts.total)
    """

    def __init__(
        self,
        execution: ExecutionRepository,
        data_flow: DataFlowRepository,
        span_factory: SpanFactory,
        run_id: str,
        *,
        factory: RecorderFactory | None = None,
        worker_id: str | None = None,
        sink_effect_fault_hook: Callable[[SinkEffectExecutionSeam], None] | None = None,
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
        self._factory = factory
        self._worker_id = worker_id or f"sink-effects:{run_id}"
        self._sink_effect_fault_hook = sink_effect_fault_hook

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
                row_contract = None if contracts is None else contracts[row_index]
                contract_context = _format_optional_missing_fields_context(missing=missing, row_contract=row_contract)
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
            can_use_bulk_begin = isinstance(self._execution, _BulkBeginNodeStateRepository) and all(
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

    def _open_or_reuse_effect_states(
        self,
        *,
        tokens: list[TokenInfo],
        rows: list[dict[str, object]],
        sink_node_id: str,
        step_in_pipeline: int,
        ctx: PluginContext,
        role: SinkEffectRole = SinkEffectRole.PRIMARY,
    ) -> list[tuple[TokenInfo, NodeState]]:
        """Reuse the exact current state set left by an interrupted effect call."""
        if self._factory is None:
            raise OrchestrationInvariantError("effect state recovery requires the owning RecorderFactory")
        token_ids = tuple(token.token_id for token in tokens)
        open_ids = self._execution.get_open_node_state_ids(
            self._run_id,
            node_ids=(sink_node_id,),
            token_ids=token_ids,
        )
        durable_members = self._execution.sink_effects.get_members_for_tokens(
            run_id=self._run_id,
            sink_node_id=sink_node_id,
            role=role,
            token_ids=token_ids,
        )
        if not open_ids and not durable_members:
            return [
                (token, state)
                for token, state in self._open_primary_states(
                    tokens=tokens,
                    rows=rows,
                    sink_node_id=sink_node_id,
                    step_in_pipeline=step_in_pipeline,
                    ctx=ctx,
                )
            ]
        if {member.token_id for member in durable_members} != set(token_ids):
            raise AuditIntegrityError("interrupted sink effect has a partial durable member witness set")
        states: list[tuple[TokenInfo, NodeState]] = []
        for token in tokens:
            open_state_id = open_ids.get(token.token_id)
            if open_state_id is not None:
                state = self._execution.get_node_state(open_state_id)
            else:
                candidates = [
                    item for item in self._factory.query.get_node_states_for_token(token.token_id) if item.node_id == sink_node_id
                ]
                state = max(candidates, key=lambda item: (item.attempt, item.started_at, item.state_id)) if candidates else None
            if state is None or state.token_id != token.token_id or state.node_id != sink_node_id:
                raise AuditIntegrityError("interrupted sink effect current node-state witness is divergent")
            states.append((token, state))
        return states

    def _write_primary_effect(
        self,
        *,
        sink: SinkProtocol,
        effect_mode: str,
        rows: list[dict[str, object]],
        tokens: list[TokenInfo],
        pending_outcome: PendingOutcome,
        all_states: list[tuple[TokenInfo, NodeState]],
        sink_name: str,
        sink_node_id: str,
    ) -> _EffectPrimaryWrite:
        """Publish one primary batch through the durable effect coordinator."""
        if self._factory is None:
            raise OrchestrationInvariantError("effect-capable sink execution requires the owning RecorderFactory")
        if pending_outcome.outcome is None:
            raise OrchestrationInvariantError("buffered outcomes cannot cross the sink-effect publication boundary")

        row_contracts = [token.row_data.contract for token in tokens]
        try:
            self._run_sink_boundary_checks(
                sink=sink,
                rows=rows,
                tokens=tokens,
                run_id=self._run_id,
                node_id=sink_node_id,
                row_contracts=row_contracts,
            )
            self._validate_sink_input(sink, rows, contracts=row_contracts)
        except (DeclarationContractViolation, AggregateDeclarationContractViolation, SinkTransactionalInvariantError) as violation:
            self._complete_states_failed(
                states=[(token, state) for token, state in all_states if isinstance(state, NodeStateOpen)],
                duration_ms=0.0,
                error=self._build_boundary_error(exc=violation, phase="sink_write"),
            )
            self._record_boundary_failure_outcomes(
                tokens=tokens,
                sink_name=sink_name,
                phase="sink_write",
                violation=violation,
            )
            raise

        pending_identity = {
            "error_hash": pending_outcome.error_hash,
            "outcome": pending_outcome.outcome.value,
            "path": pending_outcome.path.value,
            "scheduler_pending_sink": pending_outcome.scheduler_pending_sink,
        }
        candidates = tuple(
            SinkEffectMemberCandidate(token_id=token.token_id, row=row, pending_identity=pending_identity)
            for token, row in zip(tokens, rows, strict=True)
        )
        members = resolve_sink_effect_members(self._factory, candidates)  # type: ignore[arg-type]
        target_config = dict(sink.config)
        identity = compute_pipeline_effect_identity(
            run_id=self._run_id,
            sink_node_id=sink_node_id,
            role=SinkEffectRole.PRIMARY,
            sink_config={
                "effect_mode": effect_mode,
                "sink_name": sink_name,
                "sink_type": f"{type(sink).__module__}.{type(sink).__qualname__}",
            },
            target_config=target_config,
            members=members,
        )
        reservation = SinkEffectReservationRequest(
            run_id=self._run_id,
            sink_node_id=sink_node_id,
            role=SinkEffectRole.PRIMARY,
            input_kind=SinkEffectInputKind.PIPELINE_MEMBERS,
            requested_target_hash=identity.requested_target_hash,
            members=identity.members,
            audit_export_snapshot_id=None,
            config_hash=identity.config_hash,
            replacing_target=True,
            primary_effect_id=None,
        )
        token_by_id = {token.token_id: token for token in tokens}
        finalization_members = tuple(
            SinkEffectFinalizationMember(
                ordinal=member.ordinal,
                output_data={"row": dict(member.row)},
                duration_ms=0.0,
                outcome=pending_outcome.outcome,
                path=pending_outcome.path,
                sink_name=sink_name,
                join_group_id=(token_by_id[member.token_id].join_group_id if pending_outcome.path is TerminalPath.COALESCED else None),
                error_hash=pending_outcome.error_hash,
            )
            for member in identity.members
        )
        sink._reset_diversion_log()
        result = SinkEffectCoordinator(
            factory=self._factory,
            worker_id=self._worker_id,
            fault_hook=self._sink_effect_fault_hook,
        ).execute(
            SinkEffectExecutionRequest(
                reservation=reservation,
                effect_input=SinkEffectPipelineMembersInput(
                    members=identity.members,
                    target_snapshot_members=identity.members,
                ),
                finalization_members=finalization_members,
            ),
            sink,  # type: ignore[arg-type]  # capability was statically admitted before execution
        )
        requested_token_ids = tuple(member.token_id for member in identity.members)
        durable_members = self._execution.sink_effects.get_members_for_tokens(
            run_id=self._run_id,
            sink_node_id=sink_node_id,
            role=SinkEffectRole.PRIMARY,
            token_ids=requested_token_ids,
        )
        if {member.token_id for member in durable_members} != set(requested_token_ids):
            raise AuditIntegrityError("durable effect partition does not cover every requested primary token")
        caller_index_by_token = {token.token_id: index for index, token in enumerate(tokens)}
        durable_by_ordinal = {member.ordinal: member for member in durable_members}
        diverted_ordinals = {member.ordinal for member in durable_members if member.prepared_disposition == "diverted"}
        get_diversions = getattr(sink, "_get_diversions", None)
        returned_diversions = tuple(get_diversions()) if callable(get_diversions) else ()
        returned_by_ordinal = {item.row_index: item for item in returned_diversions}
        plan = SinkEffectCoordinator._load_plan(result.effect)
        raw_attribution = plan.safe_evidence.get("diversion_attribution", ())
        attribution_by_ordinal: dict[int, tuple[str, str]] = {}
        if isinstance(raw_attribution, Sequence) and not isinstance(raw_attribution, (str, bytes, bytearray)):
            for item in raw_attribution:
                if not isinstance(item, Mapping):
                    raise AuditIntegrityError("effect diversion attribution is not a mapping")
                ordinal = item.get("ordinal")
                reason_hash = item.get("reason_hash")
                error_hash = item.get("error_hash")
                if type(ordinal) is not int or not isinstance(reason_hash, str) or not isinstance(error_hash, str):
                    raise AuditIntegrityError("effect diversion attribution is incomplete")
                attribution_by_ordinal[ordinal] = (reason_hash, error_hash)
        if returned_by_ordinal and set(returned_by_ordinal) != diverted_ordinals:
            raise AuditIntegrityError("effect result diversion evidence does not match the durable member partition")
        if not returned_by_ordinal and set(attribution_by_ordinal) != diverted_ordinals:
            raise AuditIntegrityError("recovered effect is missing durable diversion attribution")
        diversions: list[RowDiversion] = []
        diversion_error_hashes: dict[int, str] = {}
        diversion_reason_hashes: dict[int, str] = {}
        token_by_id = {token.token_id: token for token in tokens}
        for durable_ordinal in sorted(diverted_ordinals):
            durable = durable_by_ordinal[durable_ordinal]
            caller_index = caller_index_by_token[durable.token_id]
            returned = returned_by_ordinal.get(durable_ordinal)
            attribution = attribution_by_ordinal.get(durable_ordinal)
            reason = returned.reason if returned is not None else f"effect-diversion:{attribution[0]}"  # type: ignore[index]
            error_hash = attribution[1] if attribution is not None else compute_error_hash(reason)
            reason_hash = attribution[0] if attribution is not None else stable_hash({"diversion_reason": reason})
            diversions.append(
                RowDiversion(
                    row_index=caller_index,
                    reason=reason,
                    row_data=token_by_id[durable.token_id].row_data.to_dict(),
                )
            )
            diversion_error_hashes[caller_index] = error_hash
            diversion_reason_hashes[caller_index] = reason_hash
        accepted_token_ids = frozenset(member.token_id for member in durable_members if member.prepared_disposition == "accepted")
        return _EffectPrimaryWrite(
            artifact=result.artifact,
            effect_id=result.effect.effect_id,
            diversions=tuple(diversions),
            diversion_error_hashes=diversion_error_hashes,
            diversion_reason_hashes=diversion_reason_hashes,
            accepted_token_ids=accepted_token_ids,
        )

    def _handle_failsink_effect_diversions(
        self,
        *,
        failsink: SinkProtocol,
        failsink_name: str,
        failsink_effect_mode: str,
        failsink_edge_id: str,
        primary_effect_id: str,
        primary_divert_states: list[tuple[TokenInfo, int, NodeState]],
        diversion_by_index: dict[int, RowDiversion],
        diversion_error_hashes: Mapping[int, str],
        diversion_reason_hashes: Mapping[int, str],
        sink_name: str,
        step_in_pipeline: int,
        ctx: PluginContext,
        on_token_written: Callable[[TokenInfo], None] | None,
    ) -> int:
        """Publish one linked failsink effect and then close its primary anchors."""
        if self._factory is None or failsink.node_id is None:
            raise OrchestrationInvariantError("linked failsink effects require an owning factory and sink node")
        failsink_node_id = failsink.node_id
        run = self._factory.run_lifecycle.get_run(self._run_id)
        if run is None:
            raise OrchestrationInvariantError("linked failsink effect run is missing")
        stable_timestamp = run.started_at.astimezone(UTC).isoformat()
        enriched_rows: list[dict[str, object]] = []
        enriched_by_token: dict[str, dict[str, object]] = {}
        diverted_tokens: list[TokenInfo] = []
        for token, index, _state in primary_divert_states:
            diversion = diversion_by_index[index]
            reason_hash = diversion_reason_hashes[index]
            row = {
                **diversion.row_data,
                "__diversion_reason": f"effect-diversion:{reason_hash}",
                "__diverted_from": sink_name,
                "__diversion_timestamp": stable_timestamp,
            }
            enriched_rows.append(row)
            enriched_by_token[token.token_id] = row
            diverted_tokens.append(token)

        self._open_or_reuse_effect_states(
            tokens=diverted_tokens,
            rows=enriched_rows,
            sink_node_id=failsink_node_id,
            step_in_pipeline=step_in_pipeline + 1,
            ctx=ctx,
            role=SinkEffectRole.FAILSINK,
        )
        self._run_sink_boundary_checks(
            sink=failsink,
            rows=enriched_rows,
            tokens=diverted_tokens,
            run_id=self._run_id,
            node_id=failsink_node_id,
            row_contracts=None,
        )
        self._validate_sink_input(failsink, enriched_rows, skip_schema=True)
        candidates = tuple(
            SinkEffectMemberCandidate(
                token_id=token.token_id,
                row=enriched_by_token[token.token_id],
                pending_identity={
                    "error_hash": diversion_error_hashes[index],
                    "outcome": TerminalOutcome.TRANSIENT.value,
                    "path": TerminalPath.SINK_FALLBACK_TO_FAILSINK.value,
                    "primary_effect_id": primary_effect_id,
                },
            )
            for token, index, _state in primary_divert_states
        )
        members = resolve_sink_effect_members(self._factory, candidates)  # type: ignore[arg-type]
        identity = compute_pipeline_effect_identity(
            run_id=self._run_id,
            sink_node_id=failsink_node_id,
            role=SinkEffectRole.FAILSINK,
            sink_config={
                "effect_mode": failsink_effect_mode,
                "sink_name": failsink_name,
                "sink_type": f"{type(failsink).__module__}.{type(failsink).__qualname__}",
            },
            target_config=dict(failsink.config),
            members=members,
        )
        reservation = SinkEffectReservationRequest(
            run_id=self._run_id,
            sink_node_id=failsink_node_id,
            role=SinkEffectRole.FAILSINK,
            input_kind=SinkEffectInputKind.PIPELINE_MEMBERS,
            requested_target_hash=identity.requested_target_hash,
            members=identity.members,
            audit_export_snapshot_id=None,
            config_hash=identity.config_hash,
            replacing_target=True,
            primary_effect_id=primary_effect_id,
        )
        caller_index_by_token = {token.token_id: index for token, index, _state in primary_divert_states}
        finalization_members = tuple(
            SinkEffectFinalizationMember(
                ordinal=member.ordinal,
                output_data={"row": dict(member.row)},
                duration_ms=0.0,
                outcome=TerminalOutcome.TRANSIENT,
                path=TerminalPath.SINK_FALLBACK_TO_FAILSINK,
                sink_name=failsink_name,
                error_hash=diversion_error_hashes[caller_index_by_token[member.token_id]],
            )
            for member in identity.members
        )
        failsink._reset_diversion_log()
        result = SinkEffectCoordinator(
            factory=self._factory,
            worker_id=self._worker_id,
            fault_hook=self._sink_effect_fault_hook,
        ).execute(
            SinkEffectExecutionRequest(
                reservation=reservation,
                effect_input=SinkEffectPipelineMembersInput(members=identity.members, target_snapshot_members=identity.members),
                finalization_members=finalization_members,
            ),
            failsink,  # type: ignore[arg-type]
        )
        durable = self._execution.sink_effects.get_members(result.effect.effect_id)
        if any(member.prepared_disposition != "accepted" for member in durable):
            raise FrameworkBugError(f"Failsink '{failsink_name}' diverted a linked failsink member")

        for token, index, primary_state in primary_divert_states:
            current = self._execution.get_node_state(primary_state.state_id)
            reason_hash = diversion_reason_hashes[index]
            reason: SinkDiversionReason = {"diversion_reason": f"effect-diversion:{reason_hash}"}
            if isinstance(current, NodeStateOpen):
                self._execution.record_routing_event(
                    state_id=current.state_id,
                    edge_id=failsink_edge_id,
                    mode=RoutingMode.DIVERT,
                    reason=reason,
                )
                self._execution.complete_node_state(
                    state_id=current.state_id,
                    status=NodeStateStatus.FAILED,
                    output_data={"diverted_to": failsink_name, "reason_hash": reason_hash},
                    duration_ms=0.0,
                    error=ExecutionError(exception=reason["diversion_reason"], exception_type="SinkDiversion", phase="write"),
                )
            elif isinstance(current, NodeStateFailed):
                events = self._factory.query.get_routing_events(current.state_id)
                if len(events) != 1 or events[0].edge_id != failsink_edge_id or events[0].mode is not RoutingMode.DIVERT:
                    raise AuditIntegrityError("finalized linked failsink has divergent primary routing evidence")
            else:
                raise AuditIntegrityError("finalized linked failsink primary anchor is not OPEN or FAILED")
            if on_token_written is not None:
                on_token_written(token)
        return len(primary_divert_states)

    def _handle_discard_diversions(
        self,
        *,
        primary_divert_states: list[tuple[TokenInfo, int, NodeState]],
        diversion_error_hashes: Mapping[int, str],
        diversion_reason_hashes: Mapping[int, str],
        on_token_written: Callable[[TokenInfo], None] | None,
    ) -> int:
        """PHASE 3 (discard mode): drop diverted tokens with a FAILED primary state.

        No routing_event (no DAG edge for discard) and no failsink write — the
        row does not reach its destination. Records SINK_DISCARDED outcomes and
        checkpoints. Returns the number of discarded tokens.
        """
        discard_count = 0
        # Discard mode: complete primary states and record DIVERTED outcomes.
        # No routing_event (no DAG edge for discard), no failsink write.
        for token, idx, primary_state in primary_divert_states:
            durable_reason = f"effect-diversion:{diversion_reason_hashes[idx]}"

            # FAILED — the row didn't reach its destination (discarded).
            discard_error = ExecutionError(
                exception=durable_reason,
                exception_type="SinkDiscard",
                phase="write",
            )
            current = self._execution.get_node_state(primary_state.state_id)
            if isinstance(current, NodeStateOpen):
                self._execution.complete_node_state(
                    state_id=current.state_id,
                    status=NodeStateStatus.FAILED,
                    output_data={"discarded": True, "reason": durable_reason},
                    duration_ms=0.0,
                    error=discard_error,
                )
            elif not isinstance(current, NodeStateFailed):
                raise AuditIntegrityError("recovered sink discard primary anchor is not OPEN or FAILED")

            error_hash = diversion_error_hashes[idx]
            # ADR-019: discard-mode diversions are predicate-input
            # failures, not transient failsink bookkeeping.
            existing = self._data_flow.get_token_outcome(token.token_id)
            if existing is None:
                self._data_flow.record_token_outcome(
                    ref=TokenRef(token_id=token.token_id, run_id=self._run_id),
                    outcome=TerminalOutcome.FAILURE,
                    path=TerminalPath.SINK_DISCARDED,
                    error_hash=error_hash,
                    sink_name="__discard__",
                )
            elif (
                existing.outcome is not TerminalOutcome.FAILURE
                or existing.path is not TerminalPath.SINK_DISCARDED
                or existing.error_hash != error_hash
                or existing.sink_name != "__discard__"
            ):
                raise AuditIntegrityError("recovered sink discard outcome is divergent")
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

        return discard_count

    def write(
        self,
        sink: SinkProtocol,
        tokens: list[TokenInfo],
        ctx: PluginContext,
        step_in_pipeline: int,
        *,
        sink_name: str,
        pending_outcome: PendingOutcome | None,
        effect_mode: str | None = None,
        failsink: SinkProtocol | None = None,
        failsink_name: str | None = None,
        failsink_effect_mode: str | None = None,
        failsink_edge_id: str | None = None,
        on_token_written: Callable[[TokenInfo], None] | None = None,
    ) -> tuple[Artifact | None, DiversionCounts]:
        """Write tokens to sink with artifact recording and failsink routing.

        CRITICAL: creates or reuses one node_state per token before effect
        publication, then finalizes states and outcomes from durable effect
        evidence. Direct ``sink.write``/``sink.flush`` publication is forbidden.

        This is the ONLY place terminal outcomes should be recorded for sink-bound
        tokens. Recording here (not in the orchestrator processing loop) ensures the
        token outcome contract is honored:
        - Invariant 3: "COMPLETED/ROUTED implies the token has a completed sink node_state"
        - Invariant 4: "Completed sink node_state implies a terminal token_outcome"

        Flow:
        - Open or recover node states for all primary members.
        - Execute/reconcile the primary effect and persist its partition.
        - Finalize accepted members; route diverted members to a linked effect
          or record discard evidence.

        Args:
            sink: Sink plugin to write to
            tokens: Tokens to write (may be empty)
            ctx: Plugin context
            step_in_pipeline: Current position in DAG (Orchestrator is authority)
            sink_name: Name of the sink (for token_outcome recording)
            pending_outcome: PendingOutcome containing outcome and optional error_hash.
                    Required - all sink-bound tokens must have their outcome recorded.
            effect_mode: Validated sink-effect mode. Required for non-empty batches.
            failsink: Resolved failsink instance (or None for discard mode)
            failsink_name: Config-level name of the failsink (for outcome recording)
            failsink_edge_id: Edge ID of the __failsink__ DIVERT edge in the DAG
            on_token_written: Optional callback called for each token after its
                             path completes durably. Primary tokens are checkpointed
                             after Phase 2, diverted tokens after Phase 3.

        Returns:
            Tuple of (Artifact if tokens were written else None, diversion counts)

        Raises:
            OrchestrationInvariantError: If a non-empty batch has no validated effect mode.
            Exception: Propagated from effect inspection, preparation, commit, reconciliation, or audit recording.
        """
        if not tokens:
            return None, DiversionCounts()
        if effect_mode is None:
            raise OrchestrationInvariantError(
                f"Sink '{sink_name}' reached execution without a validated effect mode; legacy publication is forbidden"
            )

        # pending_outcome is required for all sink-bound tokens.
        # PendingTokenMap allows None in its type alias, but _route_to_sink()
        # always wraps in PendingOutcome. None here means a routing bug.
        if pending_outcome is None:
            raise OrchestrationInvariantError(
                f"Sink '{sink_name}' received pending_outcome=None — all sink-bound tokens must have a PendingOutcome."
            )

        # Extract exact member payloads for effect identity and preparation.
        rows = [t.row_data.to_dict() for t in tokens]

        # Sink must have node_id assigned by orchestrator before execution
        if sink.node_id is None:
            raise OrchestrationInvariantError(f"Sink '{sink.name}' executed without node_id - orchestrator bug")
        sink_node_id: str = sink.node_id

        # Scope the sink row contract to the primary operation. The caller's
        # context may be reused by later operations with different row shapes.
        primary_ctx = ctx.for_contract(self._merge_batch_contract(tokens))

        # CRITICAL: Clear state_id before entering operation context.
        primary_ctx.state_id = None

        # ── PRE-PHASE: Open or recover node_states for the durable effect ──
        all_states = self._open_or_reuse_effect_states(
            tokens=tokens,
            rows=rows,
            sink_node_id=sink_node_id,
            step_in_pipeline=step_in_pipeline,
            ctx=primary_ctx,
        )

        # Index by token_id for O(1) lookup in Phases 2 and 3.
        state_by_token_id: dict[str, NodeState] = {token.token_id: state for token, state in all_states}

        effect_write = self._write_primary_effect(
            sink=sink,
            effect_mode=effect_mode,
            rows=rows,
            tokens=tokens,
            pending_outcome=pending_outcome,
            all_states=all_states,
            sink_name=sink_name,
            sink_node_id=sink_node_id,
        )
        diversions = effect_write.diversions

        # ── PHASE 2: Partition and complete primary tokens ──
        diverted_indices = {d.row_index for d in diversions}
        primary_tokens = [(token, i) for i, token in enumerate(tokens) if i not in diverted_indices]
        diverted_tokens = [(token, i) for i, token in enumerate(tokens) if i in diverted_indices]

        artifact = effect_write.artifact

        # Retrieve pre-opened states for diverted tokens. Built BEFORE Phase 2
        # so its cleanup envelope can close these anchors if Phase 2 fails —
        # Phase 3 would never run to terminalize them (elspeth-5a5e83d3e5).
        primary_divert_states: list[tuple[TokenInfo, int, NodeState]] = [
            (token, idx, state_by_token_id[token.token_id]) for token, idx in diverted_tokens
        ]

        if primary_tokens and on_token_written is not None:
            for token, _index in primary_tokens:
                if token.token_id not in effect_write.accepted_token_ids:
                    raise AuditIntegrityError("durable effect partition disagrees with accepted primary tokens")
                try:
                    on_token_written(token)
                except contract_errors.TIER_1_ERRORS:
                    raise
                except Exception as exc:
                    raise AuditIntegrityError(
                        f"Checkpoint failed after durable sink effect for token {token.token_id}. "
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

            if failsink is not None:
                if failsink_name is None or failsink_effect_mode is None or failsink_edge_id is None:
                    raise OrchestrationInvariantError("effect-capable failsink execution requires its name, mode, and routing edge")
                failsink_ctx = ctx.for_contract(None)
                failsink_ctx.state_id = None
                failsink_count = self._handle_failsink_effect_diversions(
                    failsink=failsink,
                    failsink_name=failsink_name,
                    failsink_effect_mode=failsink_effect_mode,
                    failsink_edge_id=failsink_edge_id,
                    primary_effect_id=effect_write.effect_id,
                    primary_divert_states=primary_divert_states,
                    diversion_by_index=diversion_by_index,
                    diversion_error_hashes=effect_write.diversion_error_hashes,
                    diversion_reason_hashes=effect_write.diversion_reason_hashes,
                    sink_name=sink_name,
                    step_in_pipeline=step_in_pipeline,
                    ctx=failsink_ctx,
                    on_token_written=on_token_written,
                )
            else:
                discard_count = self._handle_discard_diversions(
                    primary_divert_states=primary_divert_states,
                    diversion_error_hashes=effect_write.diversion_error_hashes,
                    diversion_reason_hashes=effect_write.diversion_reason_hashes,
                    on_token_written=on_token_written,
                )

        return artifact, DiversionCounts(failsink_mode=failsink_count, discard_mode=discard_count)
