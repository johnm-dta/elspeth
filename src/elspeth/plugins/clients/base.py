# src/elspeth/plugins/clients/base.py
"""Base class for audited clients."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from elspeth.core.landscape.recorder import LandscapeRecorder


class AuditedClientBase:
    """Base class for clients that automatically record to audit trail.

    Provides common infrastructure for tracking external calls:
    - Reference to LandscapeRecorder for audit storage
    - State ID linking calls to the current processing state
    - Call index counter for ordering multiple calls within a state

    Subclasses implement specific client protocols (LLM, HTTP, etc.)
    while inheriting automatic audit recording.
    """

    def __init__(
        self,
        recorder: LandscapeRecorder,
        state_id: str,
    ) -> None:
        """Initialize audited client.

        Args:
            recorder: LandscapeRecorder for audit trail storage
            state_id: Node state ID to associate calls with
        """
        self._recorder = recorder
        self._state_id = state_id
        self._call_index = 0

    def _next_call_index(self) -> int:
        """Get and increment call index.

        Each call within a node state gets a unique index for ordering.

        Returns:
            The next call index (0-based)
        """
        idx = self._call_index
        self._call_index += 1
        return idx
