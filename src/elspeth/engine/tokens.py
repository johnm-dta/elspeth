# src/elspeth/engine/tokens.py
"""TokenManager: High-level token operations for the SDA engine.

Provides a simplified interface over LandscapeRecorder for managing
tokens (row instances flowing through the DAG).
"""

from dataclasses import dataclass
from typing import Any

from elspeth.core.landscape import LandscapeRecorder


@dataclass
class TokenInfo:
    """Information about a token in flight.

    Carries both identity (IDs) and current state (row_data).

    Note: Step position is NOT tracked here - the Orchestrator/RowProcessor
    is the authority for where a token is in the DAG. TokenInfo is just
    identity + payload, not position.
    """

    row_id: str
    token_id: str
    row_data: dict[str, Any]
    branch_name: str | None = None


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

    def __init__(self, recorder: LandscapeRecorder) -> None:
        """Initialize with recorder.

        Args:
            recorder: LandscapeRecorder for audit trail
        """
        self._recorder = recorder

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
        """
        # Create row record
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

    def fork_token(
        self,
        parent_token: TokenInfo,
        branches: list[str],
        step_in_pipeline: int,
        row_data: dict[str, Any] | None = None,
    ) -> list[TokenInfo]:
        """Fork a token to multiple branches.

        The step_in_pipeline is required because the Orchestrator/RowProcessor
        owns step position - TokenManager doesn't track it. Note: The current
        LandscapeRecorder.fork_token() does not use step_in_pipeline, but we
        accept it here for interface consistency with the engine.

        Args:
            parent_token: Parent token to fork
            branches: List of branch names
            step_in_pipeline: Current step position in the DAG (for engine use)
            row_data: Optional row data (defaults to parent's data)

        Returns:
            List of child TokenInfo, one per branch
        """
        data = row_data if row_data is not None else parent_token.row_data

        children = self._recorder.fork_token(
            parent_token_id=parent_token.token_id,
            row_id=parent_token.row_id,
            branches=branches,
        )

        return [
            TokenInfo(
                row_id=parent_token.row_id,
                token_id=child.token_id,
                row_data=data.copy(),
                branch_name=child.branch_name,
            )
            for child in children
        ]

    def coalesce_tokens(
        self,
        parents: list[TokenInfo],
        merged_data: dict[str, Any],
        step_in_pipeline: int,
    ) -> TokenInfo:
        """Coalesce multiple tokens into one.

        The step_in_pipeline is required because the Orchestrator/RowProcessor
        owns step position - TokenManager doesn't track it. Note: The current
        LandscapeRecorder.coalesce_tokens() does not use step_in_pipeline, but
        we accept it here for interface consistency with the engine.

        Args:
            parents: Parent tokens to merge
            merged_data: Merged row data
            step_in_pipeline: Current step position in the DAG (for engine use)

        Returns:
            Merged TokenInfo
        """
        # Use first parent's row_id (they should all be the same)
        row_id = parents[0].row_id

        merged = self._recorder.coalesce_tokens(
            parent_token_ids=[p.token_id for p in parents],
            row_id=row_id,
        )

        return TokenInfo(
            row_id=row_id,
            token_id=merged.token_id,
            row_data=merged_data,
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
            Updated TokenInfo (same token_id, new row_data)
        """
        return TokenInfo(
            row_id=token.row_id,
            token_id=token.token_id,
            row_data=new_data,
            branch_name=token.branch_name,
        )

    # NOTE: No advance_step() method - step position is the authority of
    # Orchestrator/RowProcessor, not TokenManager. They track where tokens
    # are in the DAG and pass step_in_pipeline when needed.
