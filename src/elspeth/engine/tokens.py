# src/elspeth/engine/tokens.py
"""TokenManager: High-level token operations for the SDA engine.

Provides a simplified interface over LandscapeRecorder for managing
tokens (row instances flowing through the DAG).
"""

from __future__ import annotations

__all__ = ["TokenInfo", "TokenManager"]

import copy
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from elspeth.contracts.payload_store import PayloadStore

from elspeth.contracts import SourceRow, TokenInfo
from elspeth.contracts.schema_contract import PipelineRow, SchemaContract
from elspeth.contracts.types import NodeID, StepResolver
from elspeth.core.landscape import LandscapeRecorder


class TokenManager:
    """Manages token lifecycle for the SDA engine.

    Provides high-level operations:
    - Create initial token from source row
    - Fork token to multiple branches
    - Coalesce tokens from branches
    - Update token row data after transforms

    Example:
        manager = TokenManager(recorder, step_resolver=graph.resolve_step)

        # Create token for source row
        token = manager.create_initial_token(
            run_id=run.run_id,
            source_node_id=source.node_id,
            row_index=0,
            row_data={"value": 42},
        )

        # After transform
        token = manager.update_row_data(token, {"value": 42, "processed": True})

        # Fork to branches (node_id resolved to step internally)
        children = manager.fork_token(
            parent_token=token,
            branches=["stats", "classifier"],
            node_id=NodeID("gate_classifier_abc123"),
        )
    """

    def __init__(
        self,
        recorder: LandscapeRecorder,
        *,
        step_resolver: StepResolver,
        payload_store: PayloadStore | None = None,
    ) -> None:
        """Initialize with recorder, step resolver, and optional payload store.

        Args:
            recorder: LandscapeRecorder for audit trail
            step_resolver: Callable that resolves NodeID to 1-indexed audit step position.
                The canonical implementation is RowProcessor._resolve_audit_step_for_node.
            payload_store: Optional PayloadStore for persisting source row payloads
        """
        self._recorder = recorder
        self._step_resolver = step_resolver
        self._payload_store = payload_store

    def create_initial_token(
        self,
        run_id: str,
        source_node_id: str,
        row_index: int,
        source_row: SourceRow,
    ) -> TokenInfo:
        """Create a token for a source row.

        Args:
            run_id: Run identifier
            source_node_id: Source node that loaded the row
            row_index: Position in source (0-indexed)
            source_row: SourceRow from source (must have contract)

        Returns:
            TokenInfo with row and token IDs, row_data as PipelineRow

        Raises:
            ValueError: If source_row has no contract

        Note:
            Payload persistence is now handled by LandscapeRecorder.create_row(),
            not by TokenManager. This ensures Landscape owns its audit format.
        """
        # Guard: source must provide contract
        if source_row.contract is None:
            raise ValueError("SourceRow must have contract to create token. Source plugins must set contract on all valid rows.")

        # Convert to PipelineRow
        pipeline_row = source_row.to_pipeline_row()

        # Create row record - recorder stores dict representation
        row = self._recorder.create_row(
            run_id=run_id,
            source_node_id=source_node_id,
            row_index=row_index,
            data=pipeline_row.to_dict(),
        )

        # Create initial token
        token = self._recorder.create_token(row_id=row.row_id)

        return TokenInfo(
            row_id=row.row_id,
            token_id=token.token_id,
            row_data=pipeline_row,
        )

    def create_quarantine_token(
        self,
        run_id: str,
        source_node_id: str,
        row_index: int,
        source_row: SourceRow,
    ) -> TokenInfo:
        """Create a token for a quarantined row.

        Quarantined rows are invalid data that failed source validation.
        They don't have contracts (SourceRow.quarantined sets contract=None).
        They are routed directly to a quarantine sink for investigation.

        Creates a minimal PipelineRow with an empty OBSERVED contract for audit
        trail consistency, but the data is not validated or transformed.

        Args:
            run_id: Run identifier
            source_node_id: Source node that loaded the row
            row_index: Position in source (0-indexed)
            source_row: Quarantined SourceRow (contract=None is expected)

        Returns:
            TokenInfo with row and token IDs

        Raises:
            ValueError: If source_row is not quarantined
        """
        if not source_row.is_quarantined:
            raise ValueError("create_quarantine_token requires a quarantined SourceRow")

        # For quarantine rows, row may not be a dict (could be malformed external data)
        # Ensure we have a dict for the audit trail
        row_data: dict[str, Any] = source_row.row if isinstance(source_row.row, dict) else {"_raw": source_row.row}

        # Create minimal OBSERVED contract for audit consistency
        # Quarantine rows don't go through transforms, but audit trail needs a contract
        from elspeth.contracts.schema_contract import SchemaContract

        quarantine_contract = SchemaContract(
            mode="OBSERVED",
            fields=(),  # Empty - no declared fields
            locked=False,  # Not locked - quarantine doesn't validate types
        )

        # Create PipelineRow with minimal contract
        pipeline_row = PipelineRow(row_data, quarantine_contract)

        # Create row record — quarantined=True enables safe hashing for
        # Tier-3 external data that may contain non-canonical values (NaN, Infinity)
        row = self._recorder.create_row(
            run_id=run_id,
            source_node_id=source_node_id,
            row_index=row_index,
            data=pipeline_row.to_dict(),
            quarantined=True,
        )

        # Create initial token
        token = self._recorder.create_token(row_id=row.row_id)

        return TokenInfo(
            row_id=row.row_id,
            token_id=token.token_id,
            row_data=pipeline_row,
        )

    def create_token_for_existing_row(
        self,
        row_id: str,
        row_data: PipelineRow,
    ) -> TokenInfo:
        """Create a token for a row that already exists in the database.

        Used during resume when rows were created in the original run
        but tokens need to be created for reprocessing.

        Args:
            row_id: Existing row ID in the database
            row_data: Row data as PipelineRow (reconstructed from checkpoint)

        Returns:
            TokenInfo with row and token IDs
        """
        # Create token for existing row
        token = self._recorder.create_token(row_id=row_id)

        return TokenInfo(
            row_id=row_id,
            token_id=token.token_id,
            row_data=row_data,
        )

    def fork_token(
        self,
        parent_token: TokenInfo,
        branches: list[str],
        node_id: NodeID,
        run_id: str,
        row_data: PipelineRow | None = None,
    ) -> tuple[list[TokenInfo], str]:
        """Fork a token to multiple branches.

        ATOMIC: Creates children AND records parent FORKED outcome in single transaction.

        Args:
            parent_token: Parent token to fork
            branches: List of branch names
            node_id: NodeID of the gate/transform performing the fork (resolved to
                audit step position internally via step_resolver)
            run_id: Run ID (required for atomic outcome recording)
            row_data: Optional PipelineRow (defaults to parent's data)

        Returns:
            Tuple of (child TokenInfo list, fork_group_id)

        Note:
            Contract is propagated from row_data to all children via deepcopy.
            PipelineRow.__deepcopy__ preserves contract reference (immutable).
        """
        data = row_data if row_data is not None else parent_token.row_data
        step = self._step_resolver(node_id)

        children, fork_group_id = self._recorder.fork_token(
            parent_token_id=parent_token.token_id,
            row_id=parent_token.row_id,
            branches=branches,
            run_id=run_id,
            step_in_pipeline=step,
        )

        # CRITICAL: Use deepcopy to prevent nested mutable objects from being
        # shared across forked children. Shallow copy would cause mutations in
        # one branch to leak to siblings, breaking audit trail integrity.
        child_infos = [
            TokenInfo(
                row_id=parent_token.row_id,
                token_id=child.token_id,
                row_data=copy.deepcopy(data),
                branch_name=child.branch_name,
                fork_group_id=child.fork_group_id,
            )
            for child in children
        ]
        return child_infos, fork_group_id

    def coalesce_tokens(
        self,
        parents: list[TokenInfo],
        merged_data: PipelineRow,
        node_id: NodeID,
    ) -> TokenInfo:
        """Coalesce multiple tokens into one.

        Args:
            parents: Parent tokens to merge
            merged_data: Merged row data as PipelineRow (with merged contract)
            node_id: NodeID of the coalesce node performing the merge (resolved to
                audit step position internally via step_resolver)

        Returns:
            Merged TokenInfo with PipelineRow row_data
        """
        # Use first parent's row_id (they should all be the same)
        row_id = parents[0].row_id
        step = self._step_resolver(node_id)

        merged = self._recorder.coalesce_tokens(
            parent_token_ids=[p.token_id for p in parents],
            row_id=row_id,
            step_in_pipeline=step,
        )

        return TokenInfo(
            row_id=row_id,
            token_id=merged.token_id,
            row_data=merged_data,
            join_group_id=merged.join_group_id,
        )

    def update_row_data(
        self,
        token: TokenInfo,
        new_data: PipelineRow,
    ) -> TokenInfo:
        """Update token's row data after a transform.

        Args:
            token: Token to update
            new_data: New PipelineRow with updated data

        Returns:
            Updated TokenInfo (same token_id, new row_data, all lineage preserved)
        """
        return token.with_updated_data(new_data)

    def expand_token(
        self,
        parent_token: TokenInfo,
        expanded_rows: list[dict[str, Any]],
        output_contract: SchemaContract,
        node_id: NodeID,
        run_id: str,
        record_parent_outcome: bool = True,
    ) -> tuple[list[TokenInfo], str]:
        """Create child tokens for deaggregation (1 input -> N outputs).

        ATOMIC: Creates children AND optionally records parent EXPANDED outcome
        in single transaction.

        Unlike fork_token (which creates parallel paths through the same DAG),
        expand_token creates sequential children that all continue down the
        same path. Used when a transform outputs multiple rows from single input.

        Args:
            parent_token: The token being expanded
            expanded_rows: List of output row dicts (transforms output dicts, not PipelineRow)
            output_contract: Contract for output rows (from TransformResult.contract)
            node_id: NodeID of the transform performing the expansion (resolved to
                audit step position internally via step_resolver)
            run_id: Run ID (required for atomic outcome recording)
            record_parent_outcome: If True (default), record EXPANDED outcome for parent.
                Set to False for batch aggregation where parent gets CONSUMED_IN_BATCH.

        Returns:
            Tuple of (child TokenInfo list, expand_group_id)

        Note:
            Expanded rows are dicts from transform output; we wrap them in PipelineRow
            with the output_contract (post-transform schema), not parent's contract.
        """
        # Guard - contract must be locked before any expansion side effects.
        # Expansion writes child tokens and may record parent EXPANDED outcome
        # atomically in the recorder; validate preconditions first.
        if not output_contract.locked:
            raise ValueError(
                f"Output contract must be locked before token expansion. "
                f"Contract mode={output_contract.mode}, locked={output_contract.locked}"
            )

        # Delegate to recorder which handles DB operations and parent linking
        step = self._step_resolver(node_id)
        db_children, expand_group_id = self._recorder.expand_token(
            parent_token_id=parent_token.token_id,
            row_id=parent_token.row_id,
            count=len(expanded_rows),
            run_id=run_id,
            step_in_pipeline=step,
            record_parent_outcome=record_parent_outcome,
        )

        # Use output_contract (post-transform schema) for all expanded children
        # This ensures downstream transforms can access newly added/renamed fields
        #
        # CRITICAL: Use deepcopy to prevent nested mutable objects from being
        # shared across expanded children. Same reasoning as fork_token - without
        # this, mutations in one sibling leak to others, corrupting audit trail.
        # Bug: P2-2026-01-21-expand-token-shared-row-data
        child_infos = [
            TokenInfo(
                row_id=parent_token.row_id,
                token_id=db_child.token_id,
                # Create PipelineRow with output contract
                row_data=PipelineRow(copy.deepcopy(row_data), output_contract),
                branch_name=parent_token.branch_name,  # Inherit branch
                expand_group_id=db_child.expand_group_id,
            )
            for db_child, row_data in zip(db_children, expanded_rows, strict=True)
        ]
        return child_infos, expand_group_id

    # NOTE: Step resolution is handled by the injected StepResolver, which
    # maps NodeID → 1-indexed audit step position. The canonical implementation
    # is RowProcessor._resolve_audit_step_for_node. TokenManager resolves steps
    # internally — callers pass node_id, not step_in_pipeline.
