# src/elspeth/core/landscape/recorder.py
"""LandscapeRecorder: High-level API for audit recording.

This is the main interface for recording audit trail entries during
pipeline execution. It wraps the low-level database operations.

Implementation is split across mixins for maintainability:
- _run_recording.py: Run lifecycle (begin, complete, finalize, secrets, contracts)
- _graph_recording.py: Node and edge registration/queries
- _node_state_recording.py: Node state recording and routing events
- _token_recording.py: Row/token creation, fork/coalesce/expand, outcomes
- _call_recording.py: External call recording, operations, replay lookup
- _batch_recording.py: Batch management and artifact registration
- _error_recording.py: Validation and transform error recording
- _query_methods.py: Read-only entity queries, bulk retrieval, explain
"""

from __future__ import annotations

from threading import Lock
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from elspeth.contracts.payload_store import PayloadStore

from elspeth.core.landscape._batch_recording import BatchRecordingMixin
from elspeth.core.landscape._call_recording import CallRecordingMixin
from elspeth.core.landscape._database_ops import DatabaseOps
from elspeth.core.landscape._error_recording import ErrorRecordingMixin
from elspeth.core.landscape._graph_recording import GraphRecordingMixin
from elspeth.core.landscape._node_state_recording import NodeStateRecordingMixin
from elspeth.core.landscape._query_methods import QueryMethodsMixin
from elspeth.core.landscape._run_recording import RunRecordingMixin
from elspeth.core.landscape._token_recording import TokenRecordingMixin
from elspeth.core.landscape.database import LandscapeDB
from elspeth.core.landscape.repositories import (
    ArtifactRepository,
    BatchMemberRepository,
    BatchRepository,
    CallRepository,
    EdgeRepository,
    NodeRepository,
    NodeStateRepository,
    RoutingEventRepository,
    RowRepository,
    RunRepository,
    TokenOutcomeRepository,
    TokenParentRepository,
    TokenRepository,
    TransformErrorRepository,
    ValidationErrorRepository,
)


class LandscapeRecorder(
    RunRecordingMixin,
    GraphRecordingMixin,
    NodeStateRecordingMixin,
    TokenRecordingMixin,
    CallRecordingMixin,
    BatchRecordingMixin,
    ErrorRecordingMixin,
    QueryMethodsMixin,
):
    """High-level API for recording audit trail entries.

    This class provides methods to record:
    - Runs and their configuration
    - Nodes (plugin instances) and edges
    - Rows and tokens (data flow)
    - Node states (processing records)
    - Routing events, batches, artifacts

    Example:
        db = LandscapeDB.in_memory()
        recorder = LandscapeRecorder(db)

        run = recorder.begin_run(config={"source": "data.csv"})
        # ... execute pipeline ...
        recorder.complete_run(run.run_id, status=RunStatus.COMPLETED)
    """

    def __init__(self, db: LandscapeDB, *, payload_store: PayloadStore | None = None) -> None:
        """Initialize recorder with database connection.

        Args:
            db: LandscapeDB instance for audit storage
            payload_store: Optional payload store for retrieving row data
        """
        self._db = db
        self._payload_store = payload_store

        # Per-state_id call index allocation
        # Ensures UNIQUE(state_id, call_index) across all client types and retries
        self._call_indices: dict[str, int] = {}  # state_id → next_index
        self._call_index_lock: Lock = Lock()

        # Per-operation_id call index allocation (parallel to state call indices)
        # Operations (source/sink I/O) need their own call numbering
        self._operation_call_indices: dict[str, int] = {}  # operation_id → next_index

        # Database operations helper for reduced boilerplate
        self._ops = DatabaseOps(db)

        # Repository instances for row-to-object conversions
        self._run_repo = RunRepository()
        self._node_repo = NodeRepository()
        self._edge_repo = EdgeRepository()
        self._row_repo = RowRepository()
        self._token_repo = TokenRepository()
        self._token_parent_repo = TokenParentRepository()
        self._call_repo = CallRepository()
        self._routing_event_repo = RoutingEventRepository()
        self._batch_repo = BatchRepository()
        self._node_state_repo = NodeStateRepository()
        self._validation_error_repo = ValidationErrorRepository()
        self._transform_error_repo = TransformErrorRepository()
        self._token_outcome_repo = TokenOutcomeRepository()
        self._artifact_repo = ArtifactRepository()
        self._batch_member_repo = BatchMemberRepository()
