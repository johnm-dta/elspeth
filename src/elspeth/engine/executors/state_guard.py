"""NodeStateGuard — structural guarantee that node states reach terminal status.

The invariant "every token reaches exactly one terminal state" is central to
ELSPETH's audit integrity.  Before this guard, the invariant was enforced by
manually-scoped try/except blocks in each executor.  The problem: post-processing
code (output hashing, contract evolution) lived OUTSIDE those blocks, so failures
there left node_states permanently OPEN — violating the audit trail.

NodeStateGuard encodes the invariant structurally: any unhandled exception within
the ``with`` block automatically completes the state as FAILED before propagating.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Mapping
from types import TracebackType
from typing import TYPE_CHECKING, Any

from elspeth.contracts import ExecutionError, NodeStateOpen
from elspeth.contracts.audit_evidence import AuditEvidenceBase
from elspeth.contracts.enums import NodeStateStatus
from elspeth.contracts.errors import (
    AuditIntegrityError,
    OrchestrationInvariantError,
)
from elspeth.core.canonical import canonical_json
from elspeth.core.landscape.errors import LandscapePostCommitError, LandscapeRecordError
from elspeth.core.landscape.execution_repository import ExecutionRepository

if TYPE_CHECKING:
    from elspeth.contracts.errors import CoalesceFailureReason, TransformErrorReason, TransformSuccessReason
    from elspeth.contracts.node_state_context import NodeStateContext

logger = logging.getLogger(__name__)


class NodeStateGuard:
    """Context manager that guarantees a node state reaches terminal status.

    Opens a node state in ``__enter__`` and, if the caller has not explicitly
    completed it before an exception triggers ``__exit__``, auto-completes the
    state as FAILED with the exception details.

    Usage::

        with NodeStateGuard(recorder, token_id=..., node_id=..., ...) as guard:
            ctx.state_id = guard.state_id
            # ... processing ...
            guard.complete(NodeStateStatus.COMPLETED, output_data=..., ...)

    If an exception is raised before ``guard.complete()`` is called, the state
    is automatically completed as FAILED.  If ``guard.complete()`` was already
    called, ``__exit__`` is a no-op (the state is already terminal).
    """

    __slots__ = (
        "_attempt",
        "_completed",
        "_enter_time",
        "_execution",
        "_input_data",
        "_node_id",
        "_run_id",
        "_state",
        "_step_index",
        "_terminal_persisted",
        "_token_id",
    )

    def __init__(
        self,
        execution: ExecutionRepository,
        *,
        token_id: str,
        node_id: str,
        run_id: str,
        step_index: int,
        input_data: dict[str, Any],  # Row data (Tier 2 pipeline data)
        attempt: int = 0,
    ) -> None:
        self._execution = execution
        self._token_id = token_id
        self._node_id = node_id
        self._run_id = run_id
        self._step_index = step_index
        self._input_data = input_data
        self._attempt = attempt
        self._enter_time: float = 0.0
        self._state: NodeStateOpen | None = None
        self._completed = False
        self._terminal_persisted = False

    # -- context manager protocol ------------------------------------------

    def __enter__(self) -> NodeStateGuard:
        self._enter_time = time.perf_counter()
        self._state = self._execution.begin_node_state(
            token_id=self._token_id,
            node_id=self._node_id,
            run_id=self._run_id,
            step_index=self._step_index,
            input_data=self._input_data,
            attempt=self._attempt,
        )
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        if self._completed:
            return

        if self._terminal_persisted:
            # complete() or auto-fail already wrote a terminal state (or the
            # repository explicitly told us the terminal write happened before
            # raising). Writing FAILED on top would corrupt the audit trail.
            return

        duration_ms = (time.perf_counter() - self._enter_time) * 1000

        if exc_type is None:
            # Clean exit without calling complete() — programming error.
            # Record FAILED first (preserve audit invariant), then crash.
            error = ExecutionError(
                exception="NodeStateGuard exited normally without complete()",
                exception_type="OrchestrationInvariantError",
                phase="executor_guard_missing_complete",
            )
            try:
                self._execution.complete_node_state(
                    state_id=self.state_id,
                    status=NodeStateStatus.FAILED,
                    duration_ms=duration_ms,
                    error=error,
                )
                self._terminal_persisted = True
            except LandscapePostCommitError as db_err:
                self._terminal_persisted = True
                raise AuditIntegrityError(
                    f"FAILED state for {self.state_id} was persisted after missing complete(), "
                    f"but it became unreadable immediately after completion (Tier 1 violation). "
                    f"Recorder error: {type(db_err).__name__}: {db_err}"
                ) from db_err
            except LandscapeRecordError as db_err:
                raise AuditIntegrityError(
                    f"Cannot record FAILED for state {self.state_id} after missing complete() — "
                    f"audit trail has permanent OPEN state (Tier 1 violation). "
                    f"DB error: {type(db_err).__name__}: {db_err}"
                ) from db_err
            raise OrchestrationInvariantError(
                f"NodeStateGuard for state {self.state_id} exited without complete(). "
                f"This is a bug in the calling executor — every code path must call "
                f"guard.complete() before the with-block exits normally."
            )

        # An exception occurred and the state was never completed.
        # Auto-complete as FAILED so the audit trail has a terminal record.
        #
        # ADR-010 §Decision 1: nominal AuditEvidenceBase check. Classes must
        # explicitly inherit to contribute structured context; see ADR-010
        # §Alternative 3 for the security rationale (accidental-match spoofing
        # was the reason we did not use a structural Protocol here).
        context, evidence_failure = self._extract_audit_evidence_context(exc_val)

        exc_error = ExecutionError(
            exception=str(exc_val),
            exception_type=exc_type.__name__,
            phase="executor_post_process",
            context=context,
        )
        try:
            self._execution.complete_node_state(
                state_id=self.state_id,
                status=NodeStateStatus.FAILED,
                duration_ms=duration_ms,
                error=exc_error,
            )
            self._terminal_persisted = True
        except LandscapePostCommitError as db_err:
            self._terminal_persisted = True
            raise AuditIntegrityError(
                f"FAILED state for {self.state_id} was persisted while handling "
                f"{exc_type.__name__}: {exc_val}, but it became unreadable immediately "
                f"after completion (Tier 1 violation). Recorder error: "
                f"{type(db_err).__name__}: {db_err}"
            ) from db_err
        except LandscapeRecordError as db_err:
            # Audit trail corruption (permanent OPEN state) is MORE critical than
            # the original exception. Raise AuditIntegrityError with both contexts.
            raise AuditIntegrityError(
                f"Cannot record FAILED for state {self.state_id} while handling "
                f"{exc_type.__name__}: {exc_val} — audit trail has permanent OPEN state "
                f"(Tier 1 violation). DB error: {type(db_err).__name__}: {db_err}"
            ) from db_err

        if evidence_failure is not None:
            raise AuditIntegrityError(
                f"Recorded FAILED for state {self.state_id} while handling "
                f"{exc_type.__name__}: {exc_val}, but audit evidence serialization failed "
                f"and structured context was dropped to preserve terminality."
            ) from evidence_failure

    # -- public API --------------------------------------------------------

    @property
    def state(self) -> NodeStateOpen:
        """The opened node state.  Only valid after ``__enter__``."""
        if self._state is None:
            raise OrchestrationInvariantError("NodeStateGuard.state accessed before __enter__")
        return self._state

    @property
    def state_id(self) -> str:
        """Shorthand for ``guard.state.state_id``."""
        return self.state.state_id

    @property
    def completed(self) -> bool:
        """Whether the caller has explicitly completed this state."""
        return self._completed

    def complete(
        self,
        status: NodeStateStatus,
        *,
        output_data: dict[str, Any] | list[dict[str, Any]] | None = None,
        duration_ms: float | None = None,
        error: ExecutionError | TransformErrorReason | CoalesceFailureReason | None = None,
        success_reason: TransformSuccessReason | None = None,
        context_after: NodeStateContext | None = None,
    ) -> None:
        """Complete the node state.  Guard will not intervene after this.

        ``__exit__`` only stands down once terminal persistence is known.
        Pre-persistence failures must still flow through the auto-fail path,
        but post-commit materialization failures must not trigger a second
        terminal write on top of an already-persisted state.
        """
        try:
            self._execution.complete_node_state(  # type: ignore[call-overload,misc]  # generic NodeStateStatus vs Literal overloads
                state_id=self.state_id,
                status=status,
                output_data=output_data,
                duration_ms=duration_ms,
                error=error,
                success_reason=success_reason,
                context_after=context_after,
            )
        except LandscapePostCommitError:
            self._terminal_persisted = True
            raise

        self._terminal_persisted = True
        self._completed = True

    def _extract_audit_evidence_context(
        self,
        exc_val: object | None,
    ) -> tuple[Mapping[str, Any] | None, BaseException | None]:
        """Extract structured exception context without letting it strand the state OPEN."""
        if not isinstance(exc_val, AuditEvidenceBase):
            return None, None

        try:
            context = exc_val.to_audit_dict()
            if not isinstance(context, Mapping):
                raise TypeError(f"Audit evidence must be a mapping, got {type(context).__name__}")
            canonical_json(context)
            return context, None
        except Exception as exc:
            return None, exc
