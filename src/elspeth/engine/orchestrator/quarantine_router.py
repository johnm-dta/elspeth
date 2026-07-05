"""QuarantineRouter: route a validation-failed source row to its sink.

Extracted from ``SourceIterationDriver.handle_quarantine_row`` (filigree
elspeth-27d7bfc14b). Source quarantine is a self-contained workflow — validate
the destination, sanitize the row at the Tier-3 boundary, create a quarantine
token, record the FAILED source node_state, record the DIVERT routing_event,
emit ``RowCreated`` telemetry, compute the error_hash, and append the
``PendingOutcome`` to the sink's pending-token bucket. It takes explicit args
and holds no cross-method state beyond the ``RunCeremony`` used for telemetry,
so it lives as a focused collaborator the driver delegates to and unit tests
can drive in isolation.
"""

from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from elspeth.contracts import PendingOutcome, SourceRow
from elspeth.contracts.enums import NodeStateStatus, RoutingMode, TerminalOutcome, TerminalPath
from elspeth.contracts.errors import (
    ExecutionError,
    OrchestrationInvariantError,
    SourceQuarantineReason,
)
from elspeth.contracts.events import RowCreated
from elspeth.contracts.types import NodeID
from elspeth.core.canonical import sanitize_for_canonical, stable_hash
from elspeth.core.landscape.factory import RecorderFactory
from elspeth.engine._error_hash import compute_error_hash
from elspeth.engine.orchestrator.ceremony import RunCeremony
from elspeth.engine.orchestrator.run_state import LoopContext
from elspeth.engine.orchestrator.types import RouteValidationError

if TYPE_CHECKING:
    from collections.abc import Mapping

    from elspeth.contracts import SourceProtocol

# Backstop cap for plugin-authored quarantine error text (elspeth-a300402c58).
# Source plugins own producing input-free error strings
# (plugins/sources/_safe_validation_errors.py); this bound stops an
# unbounded or input-echoing plugin string from flooding every audit
# surface the text lands on (node_states.error_json, the DIVERT routing
# reason, exports). Genuine validation messages are far shorter.
QUARANTINE_ERROR_MAX_CHARS = 2000


def _bound_quarantine_error(error_text: str) -> str:
    """Truncate over-long quarantine error text with an explicit marker."""
    if len(error_text) <= QUARANTINE_ERROR_MAX_CHARS:
        return error_text
    elided = len(error_text) - QUARANTINE_ERROR_MAX_CHARS
    return f"{error_text[:QUARANTINE_ERROR_MAX_CHARS]} [truncated {elided} chars]"


class QuarantineRouter:
    """Route a quarantined source row directly to its configured sink.

    Holds only the ``RunCeremony`` used to emit ``RowCreated`` telemetry; all
    per-row state arrives through ``route`` arguments.
    """

    def __init__(self, *, ceremony: RunCeremony) -> None:
        self._ceremony = ceremony

    def route(
        self,
        factory: RecorderFactory,
        run_id: str,
        source_id: NodeID,
        source_item: SourceRow,
        row_index: int,
        source_row_index: int,
        ingest_sequence: int,
        edge_map: Mapping[tuple[NodeID, str], str],
        loop_ctx: LoopContext,
        *,
        active_source: SourceProtocol,
    ) -> None:
        """Handle a quarantined source row: route directly to configured sink.

        Accesses loop_ctx.processor for token creation and loop_ctx.counters
        for incrementing quarantine count. Appends to loop_ctx.pending_tokens.

        This method performs the complete quarantine workflow:
        1. Validate quarantine destination exists
        2. Sanitize data for canonical JSON
        3. Create quarantine token
        4. Record source node_state (FAILED)
        5. Record DIVERT routing_event
        6. Emit telemetry
        7. Compute error_hash
        8. Append to pending_tokens with PendingOutcome
        """

        config = loop_ctx.config
        counters = loop_ctx.counters
        processor = loop_ctx.processor
        pending_tokens = loop_ctx.pending_tokens

        # Route quarantined row to configured sink
        # Per CLAUDE.md: plugin bugs must crash, no silent drops
        quarantine_sink = source_item.quarantine_destination

        # Validate destination exists - crash on plugin bug
        if not quarantine_sink:
            raise RouteValidationError(
                f"Source '{active_source.name}' yielded quarantined row "
                f"(source_row_index={source_row_index}, ingest_sequence={ingest_sequence}) "
                f"with missing quarantine_destination. "
                f"This is a plugin bug: quarantined rows MUST specify a destination. "
                f"Use SourceRow.quarantined(row, error, destination, source_row_index=...) factory method."
            )
        if quarantine_sink not in config.sinks:
            raise RouteValidationError(
                f"Source '{active_source.name}' yielded quarantined row "
                f"(source_row_index={source_row_index}, ingest_sequence={ingest_sequence}) "
                f"with invalid quarantine_destination='{quarantine_sink}'. "
                f"No sink named '{quarantine_sink}' exists. "
                f"Available sinks: {sorted(config.sinks.keys())}. "
                f"This is a plugin bug: quarantine_destination must match "
                f"source._on_validation_failure='{active_source._on_validation_failure}'."
            )

        # Destination validated. Source quarantine is a FAILURE lifecycle with
        # a quarantine reporting subset, so bump both counters.
        counters.rows_quarantined += 1
        counters.rows_failed += 1
        validation_error_id = loop_ctx.ctx.pop_pending_quarantine_validation_error_id(source_item.row)
        # Sanitize quarantine data at Tier-3 boundary: replace non-finite
        # floats (NaN, Infinity) with None so downstream canonical JSON
        # and stable_hash operations succeed. The quarantine_error records
        # what was originally wrong with the data.
        # SourceRow is frozen — create a new instance with sanitized row data.
        source_item = replace(source_item, row=sanitize_for_canonical(source_item.row))

        # Create a token for the quarantined row using specialized method
        # (quarantine rows don't have contracts - they failed validation)
        quarantine_token = processor.token_manager.create_quarantine_token(
            run_id=run_id,
            source_node_id=source_id,
            row_index=row_index,
            source_row_index=source_row_index,
            ingest_sequence=ingest_sequence,
            source_row=source_item,
            validation_error_id=validation_error_id,
            # ADR-030 §C.4 row 9: the quarantine arm is an ingest-adjacent
            # durable rows write — it rides the leader epoch fence (rows +
            # token in ONE fenced transaction).
            coordination_token=processor.coordination_token,
        )

        # Record source node_state (step_index=0) for quarantine audit lineage.
        # Status is FAILED because the source validation rejected this row.
        quarantine_data = source_item.row if isinstance(source_item.row, dict) else {"_raw": source_item.row}
        quarantine_error_msg = source_item.quarantine_error
        if quarantine_error_msg is None or not quarantine_error_msg.strip():
            raise RouteValidationError(
                f"Source '{active_source.name}' yielded quarantined row "
                f"(source_row_index={source_row_index}, ingest_sequence={ingest_sequence}) "
                f"with missing quarantine_error. "
                f"This is a plugin bug: quarantined rows MUST specify a non-empty validation error. "
                f"Use SourceRow.quarantined(row, error, destination, source_row_index=...) factory method."
            )
        # Backstop length-bound (elspeth-a300402c58): applied BEFORE every use
        # below — node_state error, routing reason, and error_hash all see the
        # same bounded text, so the hash stays stable for the persisted evidence.
        quarantine_error_msg = _bound_quarantine_error(quarantine_error_msg)
        source_state = factory.execution.begin_node_state(
            token_id=quarantine_token.token_id,
            node_id=source_id,
            run_id=run_id,
            step_index=0,
            input_data=quarantine_data,
            quarantined=True,
        )
        factory.execution.complete_node_state(
            state_id=source_state.state_id,
            status=NodeStateStatus.FAILED,
            duration_ms=0,
            error=ExecutionError(
                exception=quarantine_error_msg,
                exception_type="ValidationError",
            ),
        )

        # Record DIVERT routing_event for the quarantine edge.
        # The __quarantine__ edge MUST exist — DAG creates it in
        # the source quarantine edge block of from_plugin_instances().
        quarantine_edge_key = (source_id, "__quarantine__")
        try:
            quarantine_edge_id = edge_map[quarantine_edge_key]
        except KeyError as exc:
            raise OrchestrationInvariantError(
                f"Quarantine row reached orchestrator but no __quarantine__ "
                f"DIVERT edge exists in DAG for source '{source_id}'. "
                f"This is a DAG construction bug — "
                f"on_validation_failure should have created a DIVERT edge "
                f"in from_plugin_instances()."
            ) from exc
        factory.execution.record_routing_event(
            state_id=source_state.state_id,
            edge_id=quarantine_edge_id,
            mode=RoutingMode.DIVERT,
            reason=SourceQuarantineReason(
                quarantine_error=quarantine_error_msg,
            ),
        )

        # Emit RowCreated telemetry AFTER Landscape recording succeeds.
        # source_item.row was already sanitized for Tier-3 non-canonical values
        # (NaN/Infinity -> None) above, so stable_hash gives a single deterministic
        # semantics for content_hash. No repr_hash fallback: after sanitization the
        # only residual stable_hash failure is a structurally non-serializable type,
        # which is a plugin-contract violation that must surface, not be masked by a
        # second, divergent hash function recorded under the same field name.
        quarantine_content_hash = stable_hash(source_item.row)
        self._ceremony.emit_telemetry(
            RowCreated(
                timestamp=datetime.now(UTC),
                run_id=run_id,
                row_id=quarantine_token.row_id,
                token_id=quarantine_token.token_id,
                content_hash=quarantine_content_hash,
            )
        )

        # Compute error_hash for QUARANTINED outcome audit trail
        # Per CLAUDE.md: every row must reach exactly one terminal state
        # Do NOT record outcome here — record after sink durability in SinkExecutor.write()
        quarantine_error_hash = compute_error_hash(quarantine_error_msg)

        # Pass PendingOutcome with error_hash - outcome recorded after sink durability
        pending_tokens[quarantine_sink].append(
            (
                quarantine_token,
                PendingOutcome(
                    outcome=TerminalOutcome.FAILURE,
                    path=TerminalPath.QUARANTINED_AT_SOURCE,
                    error_hash=quarantine_error_hash,
                ),
            )
        )
