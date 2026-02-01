# src/elspeth/engine/tokens.py
"""TokenManager: High-level token operations for the SDA engine.

Provides a simplified interface over LandscapeRecorder for managing
tokens (row instances flowing through the DAG).
"""

from __future__ import annotations

import copy
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from elspeth.contracts.payload_store import PayloadStore

from elspeth.contracts import TokenInfo
from elspeth.core.landscape import LandscapeRecorder


class TokenManager:
    """Manages token lifecycle for the SDA engine.

    Provides high-level operations:
    - Create initial token from source row
    - Fork token to multiple branches
    - Coalesce tokens from branches
    - Update token row data after transforms

    Example:
        manager = TokenManager(recorder)

        # Create token for source row
        token = manager.create_initial_token(
            run_id=run.run_id,
            source_node_id=source.node_id,
            row_index=0,
            row_data={"value": 42},
        )

        # After transform
        token = manager.update_row_data(token, {"value": 42, "processed": True})

        # Fork to branches (step_in_pipeline from Orchestrator)
        children = manager.fork_token(
            parent_token=token,
            branches=["stats", "classifier"],
            step_in_pipeline=2,  # Orchestrator provides step position
        )
    """

    def __init__(self, recorder: LandscapeRecorder, *, payload_store: PayloadStore | None = None) -> None:
        """Initialize with recorder and optional payload store.

        Args:
            recorder: LandscapeRecorder for audit trail
            payload_store: Optional PayloadStore for persisting source row payloads
        """
        self._recorder = recorder
        self._payload_store = payload_store

    def create_initial_token(
        self,
        run_id: str,
        source_node_id: str,
        row_index: int,
        row_data: dict[str, Any],
    ) -> TokenInfo:
        """Create a token for a source row.

        Args:
            run_id: Run identifier
            source_node_id: Source node that loaded the row
            row_index: Position in source (0-indexed)
            row_data: Row data from source

        Returns:
            TokenInfo with row and token IDs

        Note:
            Payload persistence is now handled by LandscapeRecorder.create_row(),
            not by TokenManager. This ensures Landscape owns its audit format.
        """
        # Create row record - recorder handles payload persistence internally
        row = self._recorder.create_row(
            run_id=run_id,
            source_node_id=source_node_id,
            row_index=row_index,
            data=row_data,
        )

        # Create initial token
        token = self._recorder.create_token(row_id=row.row_id)

        return TokenInfo(
            row_id=row.row_id,
            token_id=token.token_id,
            row_data=row_data,
        )

    def create_token_for_existing_row(
        self,
        row_id: str,
        row_data: dict[str, Any],
    ) -> TokenInfo:
        """Create a token for a row that already exists in the database.

        Used during resume when rows were created in the original run
        but tokens need to be created for reprocessing.

        Args:
            row_id: Existing row ID in the database
            row_data: Row data (retrieved from payload store)

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
        step_in_pipeline: int,
        run_id: str,
        row_data: dict[str, Any] | None = None,
    ) -> tuple[list[TokenInfo], str]:
        """Fork a token to multiple branches.

        ATOMIC: Creates children AND records parent FORKED outcome in single transaction.

        The step_in_pipeline is required because the Orchestrator/RowProcessor
        owns step position - TokenManager doesn't track it.

        Args:
            parent_token: Parent token to fork
            branches: List of branch names
            step_in_pipeline: Current step position in the DAG (stored in audit trail)
            run_id: Run ID (required for atomic outcome recording)
            row_data: Optional row data (defaults to parent's data)

        Returns:
            Tuple of (child TokenInfo list, fork_group_id)
        """
        data = row_data if row_data is not None else parent_token.row_data

        children, fork_group_id = self._recorder.fork_token(
            parent_token_id=parent_token.token_id,
            row_id=parent_token.row_id,
            branches=branches,
            run_id=run_id,
            step_in_pipeline=step_in_pipeline,
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
        merged_data: dict[str, Any],
        step_in_pipeline: int,
    ) -> TokenInfo:
        """Coalesce multiple tokens into one.

        The step_in_pipeline is required because the Orchestrator/RowProcessor
        owns step position - TokenManager doesn't track it.

        Args:
            parents: Parent tokens to merge
            merged_data: Merged row data
            step_in_pipeline: Current step position in the DAG (stored in audit trail)

        Returns:
            Merged TokenInfo
        """
        # Use first parent's row_id (they should all be the same)
        row_id = parents[0].row_id

        merged = self._recorder.coalesce_tokens(
            parent_token_ids=[p.token_id for p in parents],
            row_id=row_id,
            step_in_pipeline=step_in_pipeline,
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
        new_data: dict[str, Any],
    ) -> TokenInfo:
        """Update token's row data after a transform.

        Args:
            token: Token to update
            new_data: New row data

        Returns:
            Updated TokenInfo (same token_id, new row_data, all lineage preserved)
        """
        return token.with_updated_data(new_data)

    def expand_token(
        self,
        parent_token: TokenInfo,
        expanded_rows: list[dict[str, Any]],
        step_in_pipeline: int,
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
            expanded_rows: List of output row dicts
            step_in_pipeline: Current step (for audit)
            run_id: Run ID (required for atomic outcome recording)
            record_parent_outcome: If True (default), record EXPANDED outcome for parent.
                Set to False for batch aggregation where parent gets CONSUMED_IN_BATCH.

        Returns:
            Tuple of (child TokenInfo list, expand_group_id)
        """
        # Delegate to recorder which handles DB operations and parent linking
        db_children, expand_group_id = self._recorder.expand_token(
            parent_token_id=parent_token.token_id,
            row_id=parent_token.row_id,
            count=len(expanded_rows),
            run_id=run_id,
            step_in_pipeline=step_in_pipeline,
            record_parent_outcome=record_parent_outcome,
        )

        # CRITICAL: Use deepcopy to prevent nested mutable objects from being
        # shared across expanded children. Same reasoning as fork_token - without
        # this, mutations in one sibling leak to others, corrupting audit trail.
        # Bug: P2-2026-01-21-expand-token-shared-row-data
        child_infos = [
            TokenInfo(
                row_id=parent_token.row_id,
                token_id=db_child.token_id,
                row_data=copy.deepcopy(row_data),
                branch_name=parent_token.branch_name,  # Inherit branch
                expand_group_id=db_child.expand_group_id,
            )
            for db_child, row_data in zip(db_children, expanded_rows, strict=True)
        ]
        return child_infos, expand_group_id

    # NOTE: No advance_step() method - step position is the authority of
    # Orchestrator/RowProcessor, not TokenManager. They track where tokens
    # are in the DAG and pass step_in_pipeline when needed.
