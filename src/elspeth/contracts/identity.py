"""Entity identifiers and token structures.

These types answer: "How do we refer to things?"
"""

from dataclasses import dataclass, replace
from typing import Any


@dataclass
class TokenInfo:
    """Identity and data for a token flowing through the DAG.

    Tokens track row instances through forks/joins:
    - row_id: Stable source row identity
    - token_id: Instance of row in a specific DAG path
    - branch_name: Which fork path this token is on (if forked)
    - fork_group_id: Groups all children from a fork operation
    - join_group_id: Groups all tokens merged in a coalesce operation
    - expand_group_id: Groups all children from an expand operation

    Note: NOT frozen because row_data is mutable dict and executors
    update tokens as they flow through the pipeline.
    """

    row_id: str
    token_id: str
    row_data: dict[str, Any]
    branch_name: str | None = None
    fork_group_id: str | None = None
    join_group_id: str | None = None
    expand_group_id: str | None = None

    def with_updated_data(self, new_data: dict[str, Any]) -> "TokenInfo":
        """Return a new TokenInfo with updated row_data, preserving all lineage fields.

        This method ensures that when row_data is updated after a transform,
        all identity and lineage metadata (branch_name, fork_group_id,
        join_group_id, expand_group_id) are preserved.

        Use this instead of constructing TokenInfo manually when updating
        a token's data after processing.

        Args:
            new_data: The new row_data to use

        Returns:
            A new TokenInfo with the same identity/lineage but new row_data
        """
        return replace(self, row_data=new_data)
