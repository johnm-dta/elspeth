"""Entity identifiers and token structures.

These types answer: "How do we refer to things?"
"""

from dataclasses import dataclass
from typing import Any


@dataclass
class TokenInfo:
    """Identity and data for a token flowing through the DAG.

    Tokens track row instances through forks/joins:
    - row_id: Stable source row identity
    - token_id: Instance of row in a specific DAG path
    - branch_name: Which fork path this token is on (if forked)

    Note: NOT frozen because row_data is mutable dict and executors
    update tokens as they flow through the pipeline.
    """

    row_id: str
    token_id: str
    row_data: dict[str, Any]
    branch_name: str | None = None
